//! `qwen4_exp` end to end: embeddings in, logits out.
//!
//! ```text
//! input_ids ─ embed_tokens ─► x [B,T,2560]
//!                              │  repeated hc_count times
//!                              ▼
//!                            h [B,T,10240]   four identical streams
//!                              │
//!                            48 decoder layers   (PLE joins at layer 1)
//!                              │
//!                            hyper_connection_mixer   collapses 4 → 1
//!                              ▼
//!                            lm_head
//! ```
//!
//! Three things about this shape are unlike every other model we serve,
//! and all three are visible in the checkpoint rather than inferred:
//!
//! 1. **The residual between layers is `hidden_size * hc_count` wide**,
//!    initialised as four copies of the token embedding.
//! 2. **There is no `model.norm`.** The index has
//!    `hyper_connection_mixer.{hc_norm, input_mix_weight_down,
//!    input_mix_weight_up}` and nothing else before `lm_head`, so the
//!    mixer's own `hc_norm` is the only normalisation the logits see.
//!    It also has **no `block_inject_weight`** — it collapses the
//!    streams rather than scattering back into them.
//! 3. **The n-gram embedding is computed once per step, not per
//!    layer.** It depends only on the input ids, and exactly one layer
//!    consumes it.
//!
//! The weights hang off two different roots: everything structural is
//! under `model.language_model`, but `lm_head` is at the top level.
//!
//! See `doc/qwen4_exp-port-spec.md`.

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::var_builder::ShardedVarBuilder;
use candle_nn::{Embedding, Linear};
use std::sync::Arc;

use crate::harness::arch::qwen3_5::rope::RotaryEmbedding;

use super::config::{Config, TextConfig};
use super::decoder::DecoderLayer;
use super::hyper::HyperConnection;
use super::ple::{NGramHasher, NGramTable, ShardedNGramTable};
use crate::harness::arch::snapshot::{KvCacheSnapshot, PleSnapshot};

/// The hashed n-gram lookup, hoisted to the model because it is a
/// function of the input ids alone.
struct NGramEmbedding {
    hasher: NGramHasher,
    table: Box<dyn NGramTable>,
    heads: usize,
    head_dim: usize,
    /// The previous `ngram_size - 1` ids per batch row, so a decode step
    /// hashes the same n-grams a prefill would. Upstream keeps this as a
    /// `(B, 2)` "conv state" on the PLE layer; here it is what it is —
    /// token ids, not activations.
    carried: Vec<Vec<i64>>,
}

impl NGramEmbedding {
    /// `[B, T, heads * head_dim]` for this step's ids.
    fn forward(&mut self, ids: &[Vec<i64>], device: &Device, dtype: DType) -> Result<Tensor> {
        let seq_len = ids.first().map_or(0, Vec::len);
        if self.carried.len() != ids.len() {
            self.carried = vec![Vec::new(); ids.len()];
        }
        let mut rows = Vec::with_capacity(ids.len());
        for (row, step) in ids.iter().enumerate() {
            let mut history = self.carried[row].clone();
            history.extend_from_slice(step);
            let per_position = self.hasher.rows(&history, step.len())?;
            let flat: Vec<i64> = per_position.into_iter().flatten().collect();
            rows.push(
                self.table
                    .gather(&flat)?
                    .reshape((step.len(), self.heads * self.head_dim))?,
            );
            // Keep only what the next step's shifts can still reach.
            let keep = self.hasher.context_len().min(history.len());
            self.carried[row] = history[history.len() - keep..].to_vec();
        }
        let stacked = Tensor::stack(&rows, 0)?
            .to_dtype(dtype)?
            .to_device(device)?;
        debug_assert_eq!(stacked.dims()[1], seq_len);
        Ok(stacked)
    }

    /// The carried ids, for a prefix snapshot. Two integers per batch
    /// row — small, but they decide which n-grams the next token
    /// hashes, so a restore that forgets them silently addresses the
    /// wrong rows of a 320-million-row table.
    fn snapshot(&self) -> Vec<Vec<i64>> {
        self.carried.clone()
    }

    fn restore(&mut self, carried: &[Vec<i64>]) {
        self.carried = carried.to_vec();
    }

    fn reset(&mut self) {
        for row in &mut self.carried {
            row.clear();
        }
    }
}

pub struct Qwen4ExpForCausalLM {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    /// `hyper_connection_mixer` — `use_combine = false`, so it collapses
    /// rather than scatters, and there is no final norm beside it.
    mixer: HyperConnection,
    lm_head: Linear,
    ngram: Option<NGramEmbedding>,
    rotary: Arc<RotaryEmbedding>,
    hc_count: usize,
    device: Device,
    dtype: DType,
}

impl Qwen4ExpForCausalLM {
    pub fn load(
        cfg: &Config,
        dtype: DType,
        device: &Device,
        vb: &ShardedVarBuilder,
    ) -> Result<Self> {
        let text = &cfg.text_config;
        // `RotaryEmbedding` reads a qwen3_5 config; the rope block is the
        // same type and the same values (spec §7), so the reuse is real
        // rather than a coincidence — see `config::tests`.
        let rotary = Arc::new(rotary_for(text, dtype, device)?);

        let root = vb.pp("model").pp("language_model");
        let embed_weight = root
            .pp("embed_tokens")
            .get((text.vocab_size, text.hidden_size), "weight")
            .context("load 'model.language_model.embed_tokens.weight'")?;
        let embed_tokens = Embedding::new(embed_weight, text.hidden_size);

        let layers_vb = root.pp("layers");
        let mut layers = Vec::with_capacity(text.num_hidden_layers);
        for i in 0..text.num_hidden_layers {
            layers.push(
                DecoderLayer::load(text, rotary.clone(), i, &layers_vb.pp(i))
                    .with_context(|| format!("load decoder layer {i}"))?,
            );
        }

        let mixer = HyperConnection::load(
            &root.pp("hyper_connection_mixer"),
            text.hidden_size,
            text.hc_count,
            text.hc_lowrank,
            text.rms_norm_eps,
            // No block_inject_weight in the checkpoint: this one ends
            // the streams rather than feeding a sublayer.
            false,
        )
        .context("load hyper_connection_mixer")?;

        let lm_head_weight = vb
            .pp("lm_head")
            .get((text.vocab_size, text.hidden_size), "weight")
            .context("load 'lm_head.weight'")?;

        let ngram = match text.ple_layers().first() {
            Some(layer) => Some(load_ngram(text, &layers_vb.pp(*layer).pp("ple"))?),
            None => None,
        };

        Ok(Self {
            embed_tokens,
            layers,
            mixer,
            lm_head: Linear::new(lm_head_weight, None),
            ngram,
            rotary,
            hc_count: text.hc_count,
            device: device.clone(),
            dtype,
        })
    }

    /// `input_ids` is `(B, T)`; `offset` is the sequence position the
    /// first of them sits at. Returns logits `(B, T, vocab)`.
    pub fn forward(&mut self, input_ids: &Tensor, offset: usize) -> Result<Tensor> {
        let (batch, seq_len) = input_ids.dims2()?;

        // Four identical streams — the residual this architecture
        // carries between layers, not one.
        let x = self.embed_tokens.forward(input_ids)?;
        let mut h = x.repeat((1, 1, self.hc_count))?;

        let (cos, sin) = self.rotary.plain_cos_sin(offset, seq_len)?;
        let mask = if seq_len > 1 {
            Some(self.causal_mask(batch, seq_len, offset)?)
        } else {
            None
        };

        // Computed once: it is a function of the ids, and one layer
        // consumes it.
        let ngram = match &mut self.ngram {
            Some(ngram) => {
                let ids = ids_to_i64(input_ids)?;
                Some(ngram.forward(&ids, &self.device, self.dtype)?)
            }
            None => None,
        };

        for (i, layer) in self.layers.iter_mut().enumerate() {
            h = layer
                .forward(&h, ngram.as_ref(), mask.as_ref(), &cos, &sin, offset)
                .with_context(|| format!("layer {i}"))?;
        }

        // The mixer's hc_norm is the only normalisation before the head.
        let x = self.mixer.collapse(&h)?;
        Ok(self.lm_head.forward(&x)?)
    }

    /// Capture every piece of per-request state at one token boundary:
    /// the attention K/V and the QSA indexer cache per full-attention
    /// layer, the GatedDeltaNet conv and recurrent states per linear
    /// layer, and PLE's conv context and carried ids.
    ///
    /// There is no `rope_delta` here — this architecture's text path
    /// takes positions from the caller's offset rather than tracking a
    /// vision-induced skew, so the field is recorded as zero. When the
    /// vision tower lands (#314) that stops being true.
    pub fn snapshot_kv_cache(&self) -> candle_core::Result<KvCacheSnapshot> {
        let layers = self
            .layers
            .iter()
            .map(|l| l.snapshot_kv())
            .collect::<candle_core::Result<Vec<_>>>()?;
        let conv_state = self
            .layers
            .iter()
            .map(|l| l.snapshot_ple())
            .collect::<candle_core::Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .next();
        Ok(KvCacheSnapshot {
            layers,
            rope_delta: 0,
            ple: self.ngram.as_ref().map(|n| PleSnapshot {
                conv_state,
                carried_ids: n.snapshot(),
            }),
        })
    }

    /// Replace the live state from a snapshot. The snapshot stays valid
    /// for further restores.
    pub fn restore_kv_cache(&mut self, snap: &KvCacheSnapshot) -> candle_core::Result<()> {
        if snap.layers.len() != self.layers.len() {
            candle_core::bail!(
                "restore_kv_cache: snapshot has {} layers, model has {}",
                snap.layers.len(),
                self.layers.len()
            );
        }
        // A snapshot from a model whose PLE layer sits elsewhere would
        // otherwise restore the conv context onto the wrong layer and
        // leave this one holding the previous request's.
        if snap.ple.is_some() != self.ngram.is_some() {
            candle_core::bail!(
                "restore_kv_cache: snapshot {} PLE state, model {} a PLE layer",
                if snap.ple.is_some() {
                    "carries"
                } else {
                    "has no"
                },
                if self.ngram.is_some() {
                    "has"
                } else {
                    "has none"
                }
            );
        }
        for (layer, layer_snap) in self.layers.iter_mut().zip(snap.layers.iter()) {
            layer.restore_kv(layer_snap)?;
        }
        let conv = snap.ple.as_ref().and_then(|p| p.conv_state.as_ref());
        for layer in self.layers.iter_mut() {
            if layer.has_ple() {
                layer.restore_ple(conv)?;
            }
        }
        if let (Some(ngram), Some(ple)) = (&mut self.ngram, snap.ple.as_ref()) {
            ngram.restore(&ple.carried_ids);
        }
        Ok(())
    }

    pub fn clear_kv_cache(&mut self) -> Result<()> {
        for layer in &mut self.layers {
            layer.clear_kv_cache()?;
        }
        if let Some(ngram) = &mut self.ngram {
            ngram.reset();
        }
        Ok(())
    }

    fn causal_mask(&self, b: usize, tgt: usize, offset: usize) -> Result<Tensor> {
        let minf = f32::NEG_INFINITY;
        let mask: Vec<f32> = (0..tgt)
            .flat_map(|i| (0..(tgt + offset)).map(move |j| if j <= i + offset { 0. } else { minf }))
            .collect();
        Ok(
            Tensor::from_vec(mask, (b, 1, tgt, tgt + offset), &self.device)?
                .to_dtype(self.dtype)?,
        )
    }
}

fn load_ngram(cfg: &TextConfig, ple_vb: &ShardedVarBuilder) -> Result<NGramEmbedding> {
    let vb = ple_vb.pp("ple_embedding");
    let heads = cfg.ngram_heads();
    let head_dim = cfg.ngram_head_dim();

    // The three derived buffers ship as I64. Reading them through a
    // bf16 VarBuilder would silently round multipliers of order 1e13
    // and prime vocab sizes of 2e7 into nonsense, so the dtype is
    // forced rather than inherited.
    let ints = vb.to_dtype(DType::I64);
    let multipliers = int_vec(&ints, "layer_multipliers", cfg.ngram_size)?;
    let vocab_sizes = int_vec(&ints, "ngram_heads_vocab_sizes", heads)?;
    let offsets = int_vec(&ints, "ngram_heads_offsets", heads)?;

    // Total rows: the last head's offset plus its own vocab, rounded up
    // the way the checkpoint pads it.
    let declared: i64 = offsets[heads - 1] + vocab_sizes[heads - 1];
    let pad = cfg.make_ngram_vocab_size_divisible_by.max(1) as i64;
    let rows = (declared + pad - 1) / pad * pad;
    let rows = rows as usize;
    ensure!(
        rows.is_multiple_of(cfg.split_ngram_parts),
        "{rows} padded rows do not divide into {} shards",
        cfg.split_ngram_parts
    );

    let table = ShardedNGramTable::load(
        &vb.pp("ngram_embedding"),
        cfg.split_ngram_parts,
        rows,
        head_dim,
    )?;

    Ok(NGramEmbedding {
        hasher: NGramHasher::new(
            cfg.ngram_size,
            cfg.heads_per_ngram,
            multipliers,
            vocab_sizes,
            offsets,
            cfg.eos_token_id,
        )?,
        table: Box::new(table),
        heads,
        head_dim,
        carried: Vec::new(),
    })
}

fn int_vec(vb: &ShardedVarBuilder, name: &str, len: usize) -> Result<Vec<i64>> {
    let t = vb
        .get(len, name)
        .with_context(|| format!("load '{}/{name}'", vb.prefix()))?;
    Ok(t.to_dtype(DType::I64)?.to_vec1::<i64>()?)
}

fn ids_to_i64(input_ids: &Tensor) -> Result<Vec<Vec<i64>>> {
    let ids = input_ids.to_dtype(DType::I64)?;
    let (batch, _) = ids.dims2()?;
    let mut out = Vec::with_capacity(batch);
    for row in 0..batch {
        out.push(ids.i(row)?.to_vec1::<i64>()?);
    }
    Ok(out)
}

/// Build the rotary tables from this architecture's config.
///
/// The rotary reads only `head_dim`, `max_position_embeddings` and the
/// `rope_parameters` block, and this checkpoint declares all three
/// identically to Qwen3.6 (spec §7) — so `qwen3_5`'s module is reused
/// as-is, through the parts it actually needs rather than through a
/// fabricated config.
fn rotary_for(cfg: &TextConfig, dtype: DType, device: &Device) -> Result<RotaryEmbedding> {
    RotaryEmbedding::from_parts(
        dtype,
        cfg.head_dim,
        cfg.max_position_embeddings,
        &cfg.rope_parameters,
        device,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::arch::snapshot::LayerKvSnapshot;
    use candle_nn::var_builder::ShardedSafeTensors;
    use std::collections::HashMap;

    /// A model small enough to build in a test and complete enough to
    /// exercise every loader: two layers so both attention flavours
    /// appear, PLE on layer 1 because `ple_layer_ids` is one-indexed,
    /// and a sharded n-gram table with real prime vocab sizes.
    const TINY: &str = r#"{
        "model_type": "qwen4_exp",
        "tie_word_embeddings": false,
        "text_config": {
            "vocab_size": 32, "hidden_size": 8, "num_hidden_layers": 2,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4,
            "max_position_embeddings": 64, "rms_norm_eps": 1e-6,
            "hidden_act": "silu", "eos_token_id": 3,
            "full_attention_interval": 2,
            "rope_parameters": {"rope_theta": 10000.0, "partial_rotary_factor": 0.5},
            "hc_count": 2, "hc_lowrank": 4,
            "ple_layer_ids": [2], "ple_embed_dim": 8, "ple_conv_kernel_size": 2,
            "ngram_size": 3, "heads_per_ngram": 2,
            "ngram_vocab_size_base": 10,
            "make_ngram_vocab_size_divisible_by": 4,
            "split_ngram_parts": 2,
            "indexer_n_heads": 2, "indexer_kv_heads": 1, "indexer_head_dim": 4,
            "indexer_budget": 8, "indexer_compress_ratio": 2,
            "linear_num_key_heads": 1, "linear_key_head_dim": 4,
            "linear_num_value_heads": 2, "linear_value_head_dim": 4,
            "linear_conv_kernel_dim": 2,
            "mamba_ssm_dtype": "float32", "output_gate_type": "sigmoid",
            "num_experts": 2, "num_experts_per_tok": 1,
            "moe_intermediate_size": 4, "shared_expert_intermediate_size": 4
        }
    }"#;

    /// Per-head vocab sizes and their running offsets, as the checkpoint
    /// would ship them: four distinct primes, cumulative offsets, padded
    /// to a multiple of `make_ngram_vocab_size_divisible_by`.
    const VOCABS: [i64; 4] = [11, 13, 17, 19];
    const OFFSETS: [i64; 4] = [0, 11, 24, 41];
    const TABLE_ROWS: usize = 60; // 41 + 19, already a multiple of 4

    fn tensors(cfg: &TextConfig) -> HashMap<String, Tensor> {
        let dev = Device::Cpu;
        let mut t: HashMap<String, Tensor> = HashMap::new();
        // Small non-zero values: zeros would hide a transposed load, and
        // large ones make the MoE softmax degenerate.
        let mut seed = 0u32;
        let mut rand = |shape: Vec<usize>| {
            let n: usize = shape.iter().product();
            let v: Vec<f32> = (0..n)
                .map(|i| {
                    seed = seed
                        .wrapping_mul(1664525)
                        .wrapping_add(1013904223 + i as u32);
                    ((seed >> 16) as f32 / 32768.0 - 1.0) * 0.1
                })
                .collect();
            Tensor::from_vec(v, shape, &dev).unwrap()
        };

        let (h, hc) = (cfg.hidden_size, cfg.hc_count);
        let wide = h * hc;
        t.insert("lm_head.weight".to_string(), rand(vec![cfg.vocab_size, h]));
        let root = "model.language_model";
        t.insert(
            format!("{root}.embed_tokens.weight"),
            rand(vec![cfg.vocab_size, h]),
        );
        t.insert(
            format!("{root}.hyper_connection_mixer.hc_norm.weight"),
            rand(vec![wide]),
        );
        t.insert(
            format!("{root}.hyper_connection_mixer.input_mix_weight_down.weight"),
            rand(vec![cfg.hc_lowrank, wide]),
        );
        t.insert(
            format!("{root}.hyper_connection_mixer.input_mix_weight_up.weight"),
            rand(vec![wide, cfg.hc_lowrank]),
        );

        for i in 0..cfg.num_hidden_layers {
            let l = format!("{root}.layers.{i}");
            for which in ["attn_hyper_connection", "mlp_hyper_connection"] {
                t.insert(format!("{l}.{which}.hc_norm.weight"), rand(vec![wide]));
                t.insert(
                    format!("{l}.{which}.input_mix_weight_down.weight"),
                    rand(vec![cfg.hc_lowrank, wide]),
                );
                t.insert(
                    format!("{l}.{which}.input_mix_weight_up.weight"),
                    rand(vec![wide, cfg.hc_lowrank]),
                );
                t.insert(
                    format!("{l}.{which}.block_inject_weight.weight"),
                    rand(vec![hc, wide]),
                );
            }

            if cfg.is_full_attention(i) {
                let a = format!("{l}.self_attn");
                let heads = cfg.num_attention_heads * cfg.head_dim;
                t.insert(format!("{a}.q_proj.weight"), rand(vec![heads * 2, h]));
                t.insert(
                    format!("{a}.k_proj.weight"),
                    rand(vec![cfg.num_key_value_heads * cfg.head_dim, h]),
                );
                t.insert(
                    format!("{a}.v_proj.weight"),
                    rand(vec![cfg.num_key_value_heads * cfg.head_dim, h]),
                );
                t.insert(format!("{a}.o_proj.weight"), rand(vec![h, heads]));
                t.insert(format!("{a}.q_norm.weight"), rand(vec![cfg.head_dim]));
                t.insert(format!("{a}.k_norm.weight"), rand(vec![cfg.head_dim]));
                let fused = (cfg.indexer_n_heads + cfg.indexer_kv_heads) * cfg.indexer_head_dim;
                t.insert(
                    format!("{a}.indexer.index_qk_proj.weight"),
                    rand(vec![fused, h]),
                );
                t.insert(
                    format!("{a}.indexer.q_layernorm.weight"),
                    rand(vec![cfg.indexer_head_dim]),
                );
                t.insert(
                    format!("{a}.indexer.k_layernorm.weight"),
                    rand(vec![cfg.indexer_head_dim]),
                );
            } else {
                let n = format!("{l}.linear_attn");
                let key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
                let value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
                let conv_dim = key_dim * 2 + value_dim;
                t.insert(format!("{n}.in_proj_qkv.weight"), rand(vec![conv_dim, h]));
                t.insert(format!("{n}.in_proj_z.weight"), rand(vec![value_dim, h]));
                t.insert(
                    format!("{n}.in_proj_a.weight"),
                    rand(vec![cfg.linear_num_value_heads, h]),
                );
                t.insert(
                    format!("{n}.in_proj_b.weight"),
                    rand(vec![cfg.linear_num_value_heads, h]),
                );
                t.insert(
                    format!("{n}.conv1d.weight"),
                    rand(vec![conv_dim, 1, cfg.linear_conv_kernel_dim]),
                );
                t.insert(
                    format!("{n}.dt_bias"),
                    rand(vec![cfg.linear_num_value_heads]),
                );
                t.insert(format!("{n}.A_log"), rand(vec![cfg.linear_num_value_heads]));
                t.insert(
                    format!("{n}.norm.weight"),
                    rand(vec![cfg.linear_value_head_dim]),
                );
                t.insert(format!("{n}.out_proj.weight"), rand(vec![h, value_dim]));
            }

            let m = format!("{l}.mlp");
            t.insert(format!("{m}.gate.weight"), rand(vec![cfg.num_experts, h]));
            t.insert(
                format!("{m}.experts.gate_up_proj"),
                rand(vec![cfg.num_experts, cfg.moe_intermediate_size * 2, h]),
            );
            t.insert(
                format!("{m}.experts.down_proj"),
                rand(vec![cfg.num_experts, h, cfg.moe_intermediate_size]),
            );
            for p in ["gate_proj", "up_proj"] {
                t.insert(
                    format!("{m}.shared_expert.{p}.weight"),
                    rand(vec![cfg.shared_expert_intermediate_size, h]),
                );
            }
            t.insert(
                format!("{m}.shared_expert.down_proj.weight"),
                rand(vec![h, cfg.shared_expert_intermediate_size]),
            );
            t.insert(format!("{m}.shared_expert_gate.weight"), rand(vec![1, h]));

            if cfg.ple_layers().contains(&i) {
                let p = format!("{l}.ple");
                t.insert(format!("{p}.key_proj.weight"), rand(vec![wide, h]));
                t.insert(format!("{p}.value_proj.weight"), rand(vec![h, h]));
                for norm in ["norm_key", "norm_query", "norm_conv"] {
                    t.insert(format!("{p}.{norm}.weight"), rand(vec![wide]));
                }
                t.insert(
                    format!("{p}.conv1d.weight"),
                    rand(vec![wide, 1, cfg.ple_conv_kernel_size]),
                );
                let e = format!("{p}.ple_embedding");
                // The three derived buffers, as I64 — the dtype is the
                // point of this fixture.
                t.insert(
                    format!("{e}.layer_multipliers"),
                    Tensor::from_vec(vec![7i64, 13, 29], 3, &dev).unwrap(),
                );
                t.insert(
                    format!("{e}.ngram_heads_vocab_sizes"),
                    Tensor::from_vec(VOCABS.to_vec(), VOCABS.len(), &dev).unwrap(),
                );
                t.insert(
                    format!("{e}.ngram_heads_offsets"),
                    Tensor::from_vec(OFFSETS.to_vec(), OFFSETS.len(), &dev).unwrap(),
                );
                let per_shard = TABLE_ROWS / cfg.split_ngram_parts;
                for s in 0..cfg.split_ngram_parts {
                    t.insert(
                        format!("{e}.ngram_embedding.shard_{s}.weight"),
                        rand(vec![per_shard, cfg.ngram_head_dim()]),
                    );
                }
            }
        }
        t
    }

    fn build() -> (tempfile::TempDir, Qwen4ExpForCausalLM) {
        let cfg = Config::from_config_json(TINY).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        candle_core::safetensors::save(&tensors(&cfg.text_config), &path).unwrap();
        let vb =
            unsafe { ShardedSafeTensors::var_builder(&[&path], DType::F32, &Device::Cpu).unwrap() };
        let model = Qwen4ExpForCausalLM::load(&cfg, DType::F32, &Device::Cpu, &vb).unwrap();
        (dir, model)
    }

    /// Every loader in the architecture, against a checkpoint laid out
    /// the way the real one is. Tensor names and shapes are the failure
    /// mode a maths test cannot reach, and finding them on a 360 GB
    /// download is the expensive way.
    #[test]
    fn the_whole_model_loads_and_produces_logits() {
        let (_dir, mut model) = build();
        let ids = Tensor::from_vec(vec![5u32, 9, 3, 12, 7], (1, 5), &Device::Cpu).unwrap();

        let logits = model.forward(&ids, 0).unwrap();
        assert_eq!(logits.dims(), &[1, 5, 32]);
        let values: Vec<f32> = logits.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            values.iter().all(|v| v.is_finite()),
            "a wrong stream width or a dead mask row shows up as NaN"
        );
    }

    /// A decode step continues the prefill: the caches, the rotary
    /// offset and PLE's carried ids all have to agree, and any one of
    /// them being wrong still returns a tensor of the right shape.
    #[test]
    fn a_decode_step_follows_a_prefill() {
        let (_dir, mut model) = build();
        let prompt = Tensor::from_vec(vec![5u32, 9, 3, 12], (1, 4), &Device::Cpu).unwrap();
        model.forward(&prompt, 0).unwrap();

        let next = Tensor::from_vec(vec![7u32], (1, 1), &Device::Cpu).unwrap();
        let logits = model.forward(&next, 4).unwrap();
        assert_eq!(logits.dims(), &[1, 1, 32]);
        let values: Vec<f32> = logits.flatten_all().unwrap().to_vec1().unwrap();
        assert!(values.iter().all(|v| v.is_finite()));

        // And clearing lets the same model serve a fresh request.
        model.clear_kv_cache().unwrap();
        assert!(model.forward(&prompt, 0).is_ok());
    }

    /// The n-gram buffers are I64 in a checkpoint whose activations are
    /// not. Loading them through the model's dtype would round a prime
    /// vocab size and a multiplier into nonsense, and the addressing
    /// would still return rows.
    #[test]
    fn the_ngram_buffers_survive_a_non_integer_model_dtype() {
        let (_dir, model) = build();
        let ngram = model.ngram.as_ref().expect("layer 1 carries PLE");
        assert_eq!(ngram.heads, 4);
        assert_eq!(ngram.head_dim, 2);
        // The table was sized from offsets + vocab sizes read as I64;
        // a bf16 round-trip of 41 + 19 would not land on 60.
        assert_eq!(ngram.table.head_dim(), 2);
        assert!(ngram.table.gather(&[(TABLE_ROWS - 1) as i64]).is_ok());
        assert!(ngram.table.gather(&[TABLE_ROWS as i64]).is_err());
    }

    fn tensor_vec(t: &Tensor) -> Vec<f32> {
        t.flatten_all()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1()
            .unwrap()
    }

    fn logits_vec(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1().unwrap()
    }

    /// The property prefix caching rests on: a snapshot must *rewind*
    /// the live state, not merely coexist with it.
    ///
    /// So the state is deliberately moved past the snapshot before the
    /// restore — two further decode steps — and the restored step is
    /// then required to reproduce the original exactly. An
    /// implementation that captured nothing would not reproduce it; one
    /// that restored only the attention K/V and left the indexer cache
    /// at seven positions would trip the divergence guard rather than
    /// return wrong numbers.
    ///
    /// The control is that a *cleared* model cannot take this step at
    /// all: with an empty cache and a query at position 4 the guard
    /// fires. That is what distinguishes restore from clear here, and
    /// it is asserted rather than assumed.
    #[test]
    fn a_restored_snapshot_rewinds_rather_than_merely_continuing() {
        let (_dir, mut model) = build();
        let prompt = Tensor::from_vec(vec![5u32, 9, 3, 12], (1, 4), &Device::Cpu).unwrap();
        let next = Tensor::from_vec(vec![7u32], (1, 1), &Device::Cpu).unwrap();

        model.forward(&prompt, 0).unwrap();
        let snap = model.snapshot_kv_cache().unwrap();
        let want = logits_vec(&model.forward(&next, 4).unwrap());

        // Move the live state well past the snapshot.
        model.forward(&next, 5).unwrap();
        model.forward(&next, 6).unwrap();

        model.restore_kv_cache(&snap).unwrap();
        let got = logits_vec(&model.forward(&next, 4).unwrap());
        assert_eq!(got, want, "restore must rewind to the snapshot's boundary");

        // Control: a cleared model cannot even take the step, so the
        // restore above did something a clear does not.
        model.clear_kv_cache().unwrap();
        // `{:#}` to walk the context chain: the forward wraps the
        // failure in "layer 1", and the guard is what it wraps.
        let err = format!("{:#}", model.forward(&next, 4).unwrap_err());
        assert!(
            err.contains("diverged"),
            "expected the cache-divergence guard, got: {err}"
        );
    }

    /// The snapshot carries PLE's state, and PLE is the piece with two
    /// distinct kinds of it — a rolling conv context and the carried
    /// n-gram ids. Restoring one without the other reads the wrong rows
    /// of the table, so the snapshot has to hold both.
    #[test]
    fn the_snapshot_carries_ple_state() {
        let (_dir, mut model) = build();
        let prompt = Tensor::from_vec(vec![5u32, 9, 3, 12], (1, 4), &Device::Cpu).unwrap();
        model.forward(&prompt, 0).unwrap();

        let snap = model.snapshot_kv_cache().unwrap();
        let ple = snap.ple.as_ref().expect("layer 1 carries PLE");
        assert!(ple.conv_state.is_some(), "the dilated conv context");
        assert_eq!(ple.carried_ids.len(), 1, "one batch row");
        assert_eq!(
            ple.carried_ids[0].len(),
            2,
            "ngram_size - 1 ids carried across the step"
        );
        assert!(snap.size_bytes() > 0);
    }

    /// Round-trip the snapshot itself, not the logits it leads to.
    ///
    /// Taken from llama.cpp's `test-save-load-state` for this
    /// architecture (their #27941, which fixed the indexer cache
    /// dropping `ext.x`/`ext.y` on restore): comparing *generated
    /// output* cannot see a field dropped on the way back in, because a
    /// dropped field need not change the next token. Comparing the
    /// captured state can. Their fix moved 198 bytes of 335,692 — a
    /// logits comparison would never have noticed.
    ///
    /// So: snapshot, restore, snapshot again, and require the two to
    /// agree field for field. Anything `restore_kv_cache` silently
    /// fails to put back shows up here as a shape or value difference.
    #[test]
    fn a_snapshot_survives_its_own_round_trip() {
        let (_dir, mut model) = build();
        let prompt = Tensor::from_vec(vec![5u32, 9, 3, 12, 7], (1, 5), &Device::Cpu).unwrap();
        model.forward(&prompt, 0).unwrap();

        let first = model.snapshot_kv_cache().unwrap();
        model.restore_kv_cache(&first).unwrap();
        let second = model.snapshot_kv_cache().unwrap();

        assert_eq!(first.layer_count(), second.layer_count());
        assert_eq!(
            first.size_bytes(),
            second.size_bytes(),
            "total captured bytes"
        );

        for (i, (a, b)) in first.layers.iter().zip(second.layers.iter()).enumerate() {
            match (a, b) {
                (
                    LayerKvSnapshot::FullSparse {
                        kv: ka,
                        indexer_keys: ia,
                    },
                    LayerKvSnapshot::FullSparse {
                        kv: kb,
                        indexer_keys: ib,
                    },
                ) => {
                    assert_eq!(ka.is_some(), kb.is_some(), "layer {i} kv presence");
                    if let (Some((k1, v1)), Some((k2, v2))) = (ka, kb) {
                        assert_eq!(k1.dims(), k2.dims(), "layer {i} k");
                        assert_eq!(v1.dims(), v2.dims(), "layer {i} v");
                        assert_eq!(tensor_vec(k1), tensor_vec(k2), "layer {i} k values");
                    }
                    // The field their fix was about: dropped silently,
                    // and invisible to anything that looks at outputs.
                    assert_eq!(
                        ia.is_some(),
                        ib.is_some(),
                        "layer {i} indexer cache presence"
                    );
                    if let (Some(x), Some(y)) = (ia, ib) {
                        assert_eq!(x.dims(), y.dims(), "layer {i} indexer keys");
                        assert_eq!(tensor_vec(x), tensor_vec(y), "layer {i} indexer values");
                    }
                }
                (
                    LayerKvSnapshot::Linear {
                        conv_state: ca,
                        recurrent_state: ra,
                    },
                    LayerKvSnapshot::Linear {
                        conv_state: cb,
                        recurrent_state: rb,
                    },
                ) => {
                    assert_eq!(ca.is_some(), cb.is_some(), "layer {i} conv presence");
                    assert_eq!(ra.is_some(), rb.is_some(), "layer {i} recurrent presence");
                    if let (Some(x), Some(y)) = (ca, cb) {
                        assert_eq!(tensor_vec(x), tensor_vec(y), "layer {i} conv values");
                    }
                    if let (Some(x), Some(y)) = (ra, rb) {
                        assert_eq!(tensor_vec(x), tensor_vec(y), "layer {i} recurrent values");
                    }
                }
                _ => panic!("layer {i}: snapshot variant changed across a round trip"),
            }
        }

        let (pa, pb) = (first.ple.as_ref().unwrap(), second.ple.as_ref().unwrap());
        assert_eq!(pa.carried_ids, pb.carried_ids, "PLE carried ids");
        assert_eq!(pa.conv_state.is_some(), pb.conv_state.is_some());
        if let (Some(x), Some(y)) = (&pa.conv_state, &pb.conv_state) {
            assert_eq!(tensor_vec(x), tensor_vec(y), "PLE conv context");
        }
    }

    /// A snapshot from a differently-shaped model must be refused
    /// rather than restored onto whatever lines up.
    #[test]
    fn a_mismatched_snapshot_is_refused() {
        let (_dir, mut model) = build();
        let prompt = Tensor::from_vec(vec![5u32, 9], (1, 2), &Device::Cpu).unwrap();
        model.forward(&prompt, 0).unwrap();
        let mut snap = model.snapshot_kv_cache().unwrap();

        snap.layers.pop();
        let err = model.restore_kv_cache(&snap).unwrap_err().to_string();
        assert!(err.contains("layers"), "got: {err}");

        let mut snap = model.snapshot_kv_cache().unwrap();
        snap.ple = None;
        let err = model.restore_kv_cache(&snap).unwrap_err().to_string();
        assert!(err.contains("PLE"), "got: {err}");
    }
}
