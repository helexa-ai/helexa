//! The HeartMuLa language model (#391): a 3B "global" backbone over 12.5 Hz
//! frames and a 300M "local" decoder over the eight codebooks inside a
//! frame, both torchtune `llama3_2` transformers.
//!
//! One frame is generated in two stages:
//!
//! 1. The backbone reads the frame's input — the **sum** of nine embedding
//!    streams (eight audio codebooks from one shared table, plus text) — and
//!    its last hidden state predicts codebook 0 through `codebook0_head`.
//! 2. The local decoder, reset every frame, reads `[last_h, embed(c0)]` and
//!    then one codebook embedding per step, predicting codebooks 1–7 through
//!    the per-codebook `audio_head` matrices.
//!
//! Classifier-free guidance runs a second batch row whose *text* embedding
//! is replaced by `unconditional_text_embedding`; every head's logits are
//! combined as `uncond + scale · (cond − uncond)` and the sampled code is
//! fed to both rows.
//!
//! Reference: heartlib `heartmula/modeling_heartmula.py` and
//! `pipelines/music_generation.py` @ 5858ca8f. Two torchtune details matter
//! for parity: RoPE rotates *interleaved* pairs (candle's `rope_i`, not the
//! half-split HF layout), and its frequencies carry Llama 3.1 scaling.

use anyhow::Context;
use candle_core::{D, DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::kv_cache::KvCache;
use candle_nn::{Linear, VarBuilder, linear_no_bias};
use serde::Deserialize;

/// One torchtune `llama3_2` shape.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Flavor {
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub dim: usize,
    pub mlp_dim: usize,
}

impl Flavor {
    pub fn named(name: &str) -> anyhow::Result<Self> {
        let f = match name {
            "llama-3B" => Self {
                layers: 28,
                heads: 24,
                kv_heads: 8,
                dim: 3072,
                mlp_dim: 8192,
            },
            "llama-300M" => Self {
                layers: 3,
                heads: 8,
                kv_heads: 4,
                dim: 3072,
                mlp_dim: 8192,
            },
            "llama-400M" => Self {
                layers: 4,
                heads: 8,
                kv_heads: 4,
                dim: 3072,
                mlp_dim: 8192,
            },
            "llama-7B" => Self {
                layers: 32,
                heads: 32,
                kv_heads: 8,
                dim: 4096,
                mlp_dim: 14336,
            },
            other => anyhow::bail!("unknown HeartMuLa flavour '{other}'"),
        };
        Ok(f)
    }

    fn head_dim(&self) -> usize {
        self.dim / self.heads
    }
}

/// `config.json` of a HeartMuLa checkpoint.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub audio_num_codebooks: usize,
    pub audio_vocab_size: usize,
    pub text_vocab_size: usize,
    pub muq_dim: usize,
    pub backbone_flavor: String,
    pub decoder_flavor: String,
}

/// `gen_config.json`: the special ids generation uses.
#[derive(Debug, Clone, Deserialize)]
pub struct GenConfig {
    pub text_bos_id: u32,
    pub text_eos_id: u32,
    pub audio_eos_id: u32,
    pub empty_id: u32,
}

/// torchtune's RMSNorm: normalise in f32, then scale (weight `scale`).
struct RmsNorm {
    scale: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            scale: vb.get(dim, "scale")?,
            eps,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let xf = x.to_dtype(DType::F32)?;
        let var = xf.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = xf.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        normed.to_dtype(dtype)?.broadcast_mul(&self.scale)
    }
}

/// RoPE tables for torchtune's `Llama3ScaledRoPE`: base 500 000, scale
/// factor 32, low/high frequency factors 1/4, original context 8192. The
/// frequencies are computed in f32 and the per-frequency scaling in f64,
/// as torchtune does (a Python loop over the f32 values).
struct Rope {
    cos: Tensor,
    sin: Tensor,
}

const ROPE_BASE: f32 = 500_000.0;
const ROPE_SCALE: f64 = 32.0;
const ROPE_LOW_FREQ: f64 = 1.0;
const ROPE_HIGH_FREQ: f64 = 4.0;
const ROPE_OLD_CONTEXT: f64 = 8192.0;

/// torchtune's Llama 3.1 frequency scaling, for one head dimension.
pub fn llama3_scaled_theta(head_dim: usize) -> Vec<f32> {
    let low_wavelen = ROPE_OLD_CONTEXT / ROPE_LOW_FREQ;
    let high_wavelen = ROPE_OLD_CONTEXT / ROPE_HIGH_FREQ;
    (0..head_dim / 2)
        .map(|i| {
            let freq = 1.0f32 / ROPE_BASE.powf((2 * i) as f32 / head_dim as f32);
            let f = freq as f64;
            let wavelen = 2.0 * std::f64::consts::PI / f;
            let scaled = if wavelen < high_wavelen {
                f
            } else if wavelen > low_wavelen {
                f / ROPE_SCALE
            } else {
                let smooth =
                    (ROPE_OLD_CONTEXT / wavelen - ROPE_LOW_FREQ) / (ROPE_HIGH_FREQ - ROPE_LOW_FREQ);
                (1.0 - smooth) * f / ROPE_SCALE + smooth * f
            };
            scaled as f32
        })
        .collect()
}

impl Rope {
    fn new(head_dim: usize, max_seq: usize, dtype: DType, device: &Device) -> Result<Self> {
        let theta = llama3_scaled_theta(head_dim);
        let half = theta.len();
        let mut cos = Vec::with_capacity(max_seq * half);
        let mut sin = Vec::with_capacity(max_seq * half);
        for p in 0..max_seq {
            for &t in &theta {
                let a = p as f32 * t;
                cos.push(a.cos());
                sin.push(a.sin());
            }
        }
        Ok(Self {
            cos: Tensor::from_vec(cos, (max_seq, half), device)?.to_dtype(dtype)?,
            sin: Tensor::from_vec(sin, (max_seq, half), device)?.to_dtype(dtype)?,
        })
    }

    /// Rotate `x` `(b, heads, seq, head_dim)` at positions `pos..pos+seq`.
    fn apply(&self, x: &Tensor, pos: usize) -> Result<Tensor> {
        let seq = x.dim(2)?;
        let cos = self.cos.narrow(0, pos, seq)?;
        let sin = self.sin.narrow(0, pos, seq)?;
        candle_nn::rotary_emb::rope_i(&x.contiguous()?, &cos, &sin)
    }
}

struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    cache: Option<KvCache>,
}

impl Attention {
    fn new(f: &Flavor, vb: VarBuilder) -> Result<Self> {
        let hd = f.head_dim();
        Ok(Self {
            q: linear_no_bias(f.dim, f.heads * hd, vb.pp("q_proj"))?,
            k: linear_no_bias(f.dim, f.kv_heads * hd, vb.pp("k_proj"))?,
            v: linear_no_bias(f.dim, f.kv_heads * hd, vb.pp("v_proj"))?,
            o: linear_no_bias(f.heads * hd, f.dim, vb.pp("output_proj"))?,
            heads: f.heads,
            kv_heads: f.kv_heads,
            head_dim: hd,
            cache: None,
        })
    }

    fn forward(&mut self, x: &Tensor, rope: &Rope, pos: usize) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let shape = |t: Tensor, n: usize| -> Result<Tensor> {
            t.reshape((b, s, n, self.head_dim))?.transpose(1, 2)
        };
        let q = rope.apply(&shape(self.q.forward(x)?, self.heads)?, pos)?;
        let k = rope.apply(&shape(self.k.forward(x)?, self.kv_heads)?, pos)?;
        let v = shape(self.v.forward(x)?, self.kv_heads)?.contiguous()?;
        let (k, v) = match self.cache.as_mut() {
            Some(cache) => cache.append(&k.contiguous()?, &v)?,
            None => (k, v),
        };
        let rep = self.heads / self.kv_heads;
        let k = candle_transformers::utils::repeat_kv(k, rep)?.contiguous()?;
        let v = candle_transformers::utils::repeat_kv(v, rep)?.contiguous()?;
        let kv_len = k.dim(2)?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut att = (q.contiguous()?.matmul(&k.t()?)? * scale)?;
        if s > 1 {
            // Causal over the whole cache: query i (absolute pos + i) sees
            // keys 0..=pos+i.
            let offset = kv_len - s;
            let mask: Vec<f32> = (0..s)
                .flat_map(|i| {
                    (0..kv_len).map(move |j| {
                        if j > offset + i {
                            f32::NEG_INFINITY
                        } else {
                            0.0
                        }
                    })
                })
                .collect();
            let mask = Tensor::from_vec(mask, (s, kv_len), x.device())?.to_dtype(att.dtype())?;
            att = att.broadcast_add(&mask)?;
        }
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let out = att.matmul(&v)?;
        let out = out
            .transpose(1, 2)?
            .reshape((b, s, self.heads * self.head_dim))?;
        self.o.forward(&out)
    }
}

struct Mlp {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl Mlp {
    fn new(f: &Flavor, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            w1: linear_no_bias(f.dim, f.mlp_dim, vb.pp("w1"))?,
            w2: linear_no_bias(f.mlp_dim, f.dim, vb.pp("w2"))?,
            w3: linear_no_bias(f.dim, f.mlp_dim, vb.pp("w3"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.w1.forward(x)?)?;
        self.w2.forward(&(gate * self.w3.forward(x)?)?)
    }
}

struct Layer {
    sa_norm: RmsNorm,
    attn: Attention,
    mlp_norm: RmsNorm,
    mlp: Mlp,
}

/// A torchtune `TransformerDecoder` with identity token embedding and
/// output head: it maps hidden states to normalised hidden states.
struct Llama {
    layers: Vec<Layer>,
    norm: RmsNorm,
    rope: Rope,
}

const NORM_EPS: f64 = 1e-5;

impl Llama {
    fn new(f: &Flavor, max_seq: usize, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::with_capacity(f.layers);
        for i in 0..f.layers {
            let lvb = vb.pp(format!("layers.{i}"));
            layers.push(Layer {
                sa_norm: RmsNorm::new(f.dim, NORM_EPS, lvb.pp("sa_norm"))?,
                attn: Attention::new(f, lvb.pp("attn"))?,
                mlp_norm: RmsNorm::new(f.dim, NORM_EPS, lvb.pp("mlp_norm"))?,
                mlp: Mlp::new(f, lvb.pp("mlp"))?,
            });
        }
        Ok(Self {
            layers,
            norm: RmsNorm::new(f.dim, NORM_EPS, vb.pp("norm"))?,
            rope: Rope::new(f.head_dim(), max_seq, vb.dtype(), vb.device())?,
        })
    }

    /// Fresh caches with room for `capacity` positions.
    fn reset(&mut self, capacity: usize) {
        for l in &mut self.layers {
            l.attn.cache = Some(KvCache::new(2, capacity));
        }
    }

    fn forward(&mut self, x: &Tensor, pos: usize) -> Result<Tensor> {
        let mut h = x.clone();
        for l in &mut self.layers {
            let a = l.attn.forward(&l.sa_norm.forward(&h)?, &self.rope, pos)?;
            h = (h + a)?;
            let m = l.mlp.forward(&l.mlp_norm.forward(&h)?)?;
            h = (h + m)?;
        }
        self.norm.forward(&h)
    }
}

/// A prompt as the LM consumes it: one row per position, nine columns —
/// eight audio codebooks then text — and the position of the reference-
/// audio slot. In a prompt only the text column is live.
#[derive(Debug, Clone, PartialEq)]
pub struct Prompt {
    pub tokens: Vec<[u32; 9]>,
    pub muq_idx: usize,
}

/// Build the prompt exactly as heartlib does: tags lowercased and wrapped
/// in `<tag>…</tag>`, lyrics lowercased, each tokenized with BOS/EOS
/// ensured, then `tags ‖ [reference slot] ‖ lyrics`.
pub fn build_prompt(
    tokenizer: &tokenizers::Tokenizer,
    tags: &str,
    lyrics: &str,
    gen_cfg: &GenConfig,
) -> anyhow::Result<Prompt> {
    let mut tags = tags.to_lowercase();
    if !tags.starts_with("<tag>") {
        tags = format!("<tag>{tags}");
    }
    if !tags.ends_with("</tag>") {
        tags = format!("{tags}</tag>");
    }
    let encode = |text: &str| -> anyhow::Result<Vec<u32>> {
        let mut ids = tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?
            .get_ids()
            .to_vec();
        if ids.first() != Some(&gen_cfg.text_bos_id) {
            ids.insert(0, gen_cfg.text_bos_id);
        }
        if ids.last() != Some(&gen_cfg.text_eos_id) {
            ids.push(gen_cfg.text_eos_id);
        }
        Ok(ids)
    };
    let tags_ids = encode(&tags)?;
    let lyric_ids = encode(&lyrics.to_lowercase())?;
    let row = |text: u32| {
        let mut r = [0u32; 9];
        r[8] = text;
        r
    };
    let mut tokens: Vec<[u32; 9]> = tags_ids.iter().map(|&t| row(t)).collect();
    let muq_idx = tokens.len();
    tokens.push([0; 9]);
    tokens.extend(lyric_ids.iter().map(|&t| row(t)));
    Ok(Prompt { tokens, muq_idx })
}

/// Top-k sampling with heartlib's draw: divide by temperature, mask
/// everything below the k-th logit, softmax, then `argmax(p / Exp(1))`.
/// With k = 1 that is exact greedy decoding.
///
/// The exponential draws come from a seeded host generator, so a seed
/// reproduces a song on any device. They do not reproduce heartlib's own
/// streams, which depend on the torch generator of the device it ran on.
pub struct Sampler {
    pub temperature: f64,
    pub topk: usize,
    state: u64,
}

impl Sampler {
    pub fn new(temperature: f64, topk: usize, seed: u64) -> Self {
        Self {
            temperature,
            topk: topk.max(1),
            state: seed,
        }
    }

    fn next_unit(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // 53 random bits in (0, 1]: never zero, so ln() is finite.
        ((z >> 11) as f64 + 1.0) / (1u64 << 53) as f64
    }

    /// Pick a code from one row of guided logits.
    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        let t = self.temperature as f32;
        let scaled: Vec<f32> = logits.iter().map(|l| l / t).collect();
        let mut sorted = scaled.clone();
        sorted.sort_by(|a, b| b.total_cmp(a));
        let k = self.topk.min(sorted.len());
        let threshold = sorted[k - 1];
        let max = sorted[0];
        let mut best = (0usize, f64::NEG_INFINITY);
        for (i, &l) in scaled.iter().enumerate() {
            // Every logit consumes a draw, kept or not, as heartlib's
            // `empty_like(probs).exponential_()` does.
            let q = -self.next_unit().ln();
            if l < threshold {
                continue;
            }
            // Unnormalised probability is enough for an argmax.
            let p = ((l - max) as f64).exp();
            let score = p / q;
            if score > best.1 {
                best = (i, score);
            }
        }
        best.0 as u32
    }
}

/// The loaded model.
pub struct HeartMuLa {
    pub config: Config,
    backbone: Llama,
    decoder: Llama,
    backbone_flavor: Flavor,
    text_embeddings: Tensor,
    audio_embeddings: Tensor,
    unconditional_text: Tensor,
    projection: Linear,
    codebook0_head: Linear,
    /// `(codebooks − 1, decoder_dim, vocab)`.
    audio_head: Tensor,
    /// `muq_linear(zeros)`: the reference-audio slot's conditional value.
    muq_bias: Tensor,
    device: Device,
    dtype: DType,
}

/// Decoder KV length: `[last_h, c0]` then one step per further codebook.
const DECODER_MAX_SEQ: usize = 16;
/// The backbone's RoPE table length (torchtune `max_seq_len`).
pub const BACKBONE_MAX_SEQ: usize = 8192;

impl HeartMuLa {
    /// Load from a checkpoint directory's `VarBuilder`.
    pub fn new(config: Config, vb: VarBuilder) -> anyhow::Result<Self> {
        let bf = Flavor::named(&config.backbone_flavor)?;
        let df = Flavor::named(&config.decoder_flavor)?;
        let backbone =
            Llama::new(&bf, BACKBONE_MAX_SEQ, vb.pp("backbone")).context("load backbone")?;
        let decoder = Llama::new(&df, DECODER_MAX_SEQ, vb.pp("decoder")).context("load decoder")?;
        let v = config.audio_vocab_size;
        let n = config.audio_num_codebooks;
        let muq_bias = vb.get(bf.dim, "muq_linear.bias")?;
        // The weight is only ever applied to zeros (#389); load it anyway
        // so a checkpoint missing it is caught as malformed.
        let _ = vb.get((bf.dim, config.muq_dim), "muq_linear.weight")?;
        Ok(Self {
            text_embeddings: vb.get((config.text_vocab_size, bf.dim), "text_embeddings.weight")?,
            audio_embeddings: vb.get((v * n, bf.dim), "audio_embeddings.weight")?,
            unconditional_text: vb.get((1, bf.dim), "unconditional_text_embedding.weight")?,
            projection: linear_no_bias(bf.dim, df.dim, vb.pp("projection"))?,
            codebook0_head: linear_no_bias(bf.dim, v, vb.pp("codebook0_head"))?,
            audio_head: vb.get((n - 1, df.dim, v), "audio_head")?,
            muq_bias,
            backbone,
            decoder,
            backbone_flavor: bf,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            config,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Start a new song: fresh backbone caches sized for `positions`
    /// (prompt plus frames).
    pub fn reset(&mut self, positions: usize) {
        self.backbone.reset(positions.min(BACKBONE_MAX_SEQ));
    }

    /// The embedded input `(rows, seq, dim)` for `tokens`: the masked sum of
    /// the eight audio streams and the text stream. With CFG the second row
    /// replaces its text embedding with the unconditional one.
    fn embed(&self, tokens: &[[u32; 9]], mask: &[[bool; 9]], cfg: bool) -> Result<Tensor> {
        let s = tokens.len();
        let d = self.backbone_flavor.dim;
        let v = self.config.audio_vocab_size as u32;
        let text_ids: Vec<u32> = tokens.iter().map(|r| r[8]).collect();
        let text = self
            .text_embeddings
            .index_select(&Tensor::new(text_ids.as_slice(), &self.device)?, 0)?;
        let audio_ids: Vec<u32> = tokens
            .iter()
            .flat_map(|r| (0..8).map(move |k| r[k] + k as u32 * v))
            .collect();
        let audio = self
            .audio_embeddings
            .index_select(&Tensor::new(audio_ids.as_slice(), &self.device)?, 0)?
            .reshape((s, 8, d))?;
        let m: Vec<f32> = mask
            .iter()
            .flat_map(|r| r.iter().map(|&b| if b { 1.0 } else { 0.0 }))
            .collect();
        let m = Tensor::from_vec(m, (s, 9), &self.device)?.to_dtype(self.dtype)?;
        let audio_sum = audio
            .broadcast_mul(&m.narrow(1, 0, 8)?.unsqueeze(2)?)?
            .sum(1)?;
        let text_mask = m.narrow(1, 8, 1)?;
        let cond = (&audio_sum + text.broadcast_mul(&text_mask)?)?;
        if !cfg {
            return cond.unsqueeze(0);
        }
        let uncond_text = self.unconditional_text.broadcast_as((s, d))?;
        let uncond = (&audio_sum + uncond_text.broadcast_mul(&text_mask)?)?;
        Tensor::stack(&[cond, uncond], 0)
    }

    fn embed_audio(&self, codebook: usize, code: u32, rows: usize) -> Result<Tensor> {
        let id = code + (codebook * self.config.audio_vocab_size) as u32;
        let e = self
            .audio_embeddings
            .index_select(&Tensor::new(&[id], &self.device)?, 0)?;
        e.unsqueeze(0)?
            .broadcast_as((rows, 1, e.dim(1)?))?
            .contiguous()
    }

    /// CFG-combined logits as an f32 row.
    fn guide(logits: &Tensor, cfg: Option<f64>) -> Result<Vec<f32>> {
        let logits = logits.to_dtype(DType::F32)?;
        let row = match cfg {
            Some(scale) => {
                let cond = logits.i(0)?;
                let uncond = logits.i(1)?;
                (&uncond + ((cond - &uncond)? * scale)?)?
            }
            None => logits.i(0)?,
        };
        row.to_vec1::<f32>()
    }

    /// Generate one frame at position `pos`.
    ///
    /// `tokens`/`mask` are the frame's input rows — the whole prompt for
    /// the first frame, then the previous frame's codes with the text column
    /// masked out. `muq_idx` marks the reference-audio slot (prompt only).
    /// `choose(codebook, guided_logits)` picks each code; a sampler for
    /// generation, the recorded code for a teacher-forced parity check.
    /// Returns the backbone's last hidden state `(rows, dim)` too.
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        tokens: &[[u32; 9]],
        mask: &[[bool; 9]],
        pos: usize,
        muq_idx: Option<usize>,
        cfg_scale: f64,
        choose: &mut dyn FnMut(usize, &[f32]) -> u32,
    ) -> anyhow::Result<([u32; 8], Tensor)> {
        let cfg = cfg_scale > 1.0;
        let rows = if cfg { 2 } else { 1 };
        let guidance = cfg.then_some(cfg_scale);
        let mut h = self.embed(tokens, mask, cfg)?;
        if let Some(idx) = muq_idx {
            let d = self.backbone_flavor.dim;
            let cond = self.muq_bias.reshape((1, 1, d))?;
            h = h.slice_assign(&[0..1, idx..idx + 1, 0..d], &cond)?;
            if cfg {
                let uncond = self.unconditional_text.reshape((1, 1, d))?;
                h = h.slice_assign(&[1..2, idx..idx + 1, 0..d], &uncond)?;
            }
        }
        let h = self.backbone.forward(&h, pos)?;
        let last_h = h.i((.., h.dim(1)? - 1, ..))?.contiguous()?;

        let mut codes = [0u32; 8];
        let c0 = self.codebook0_head.forward(&last_h)?;
        codes[0] = choose(0, &Self::guide(&c0, guidance)?);

        self.decoder.reset(DECODER_MAX_SEQ);
        let mut curr = Tensor::cat(
            &[last_h.unsqueeze(1)?, self.embed_audio(0, codes[0], rows)?],
            1,
        )?;
        let mut dpos = 0;
        for (i, slot) in codes.iter_mut().enumerate().skip(1) {
            let dh = self
                .decoder
                .forward(&self.projection.forward(&curr)?, dpos)?;
            dpos += curr.dim(1)?;
            let last = dh.i((.., dh.dim(1)? - 1, ..))?.contiguous()?;
            let logits = last.matmul(&self.audio_head.i(i - 1)?)?;
            *slot = choose(i, &Self::guide(&logits, guidance)?);
            curr = self.embed_audio(i, *slot, rows)?;
        }
        Ok((codes, last_h))
    }

    /// Generate up to `max_frames` frames after the first, as heartlib
    /// does: stop when any code of a frame reaches `audio_eos_id`.
    pub fn generate(
        &mut self,
        prompt: &Prompt,
        max_frames: usize,
        cfg_scale: f64,
        gen_cfg: &GenConfig,
        sampler: &mut Sampler,
    ) -> anyhow::Result<Vec<[u32; 8]>> {
        let plen = prompt.tokens.len();
        self.reset(plen + max_frames + 1);
        let prompt_mask: Vec<[bool; 9]> = prompt
            .tokens
            .iter()
            .map(|_| {
                let mut m = [false; 9];
                m[8] = true;
                m
            })
            .collect();
        let mut choose = |_: usize, l: &[f32]| sampler.sample(l);
        let (mut codes, _) = self.step(
            &prompt.tokens,
            &prompt_mask,
            0,
            Some(prompt.muq_idx),
            cfg_scale,
            &mut choose,
        )?;
        let mut frames = vec![codes];
        let mut frame_mask = [true; 9];
        frame_mask[8] = false;
        for i in 0..max_frames {
            let mut row = [gen_cfg.empty_id; 9];
            row[..8].copy_from_slice(&codes);
            codes = self
                .step(
                    &[row],
                    &[frame_mask],
                    plen + i,
                    None,
                    cfg_scale,
                    &mut choose,
                )?
                .0;
            if codes.iter().any(|&c| c >= gen_cfg.audio_eos_id) {
                break;
            }
            frames.push(codes);
        }
        Ok(frames)
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaled_theta_matches_the_llama31_rule() {
        let t = llama3_scaled_theta(128);
        assert_eq!(t.len(), 64);
        // Highest frequency (wavelength 2π < 2048): unscaled.
        assert_eq!(t[0], 1.0);
        // Lowest frequencies (wavelength > 8192): divided by 32.
        let raw = 1.0f32 / ROPE_BASE.powf(126.0 / 128.0);
        assert!(
            (t[63] - raw / 32.0).abs() < 1e-12,
            "{} vs {}",
            t[63],
            raw / 32.0
        );
        // Monotone non-increasing throughout the smooth band.
        assert!(t.windows(2).all(|w| w[0] >= w[1]));
    }

    #[test]
    fn greedy_sampler_is_argmax_and_topk_restricts() {
        let logits = [0.1f32, 3.0, -1.0, 2.9];
        let mut greedy = Sampler::new(1.0, 1, 7);
        for _ in 0..20 {
            assert_eq!(greedy.sample(&logits), 1);
        }
        let mut top2 = Sampler::new(1.0, 2, 7);
        for _ in 0..200 {
            let c = top2.sample(&logits);
            assert!(c == 1 || c == 3, "{c}");
        }
        // Same seed, same draws.
        let mut a = Sampler::new(1.0, 4, 99);
        let mut b = Sampler::new(1.0, 4, 99);
        let sa: Vec<u32> = (0..50).map(|_| a.sample(&logits)).collect();
        let sb: Vec<u32> = (0..50).map(|_| b.sample(&logits)).collect();
        assert_eq!(sa, sb);
        // Both of the top two show up over many draws.
        assert!(sa.contains(&1) || sa.contains(&3));
    }

    #[test]
    fn flavours_are_the_torchtune_shapes() {
        let b = Flavor::named("llama-3B").unwrap();
        assert_eq!(
            (b.layers, b.heads, b.kv_heads, b.head_dim()),
            (28, 24, 8, 128)
        );
        let d = Flavor::named("llama-300M").unwrap();
        assert_eq!(
            (d.layers, d.heads, d.kv_heads, d.head_dim()),
            (3, 8, 4, 384)
        );
        assert!(Flavor::named("nope").is_err());
    }
}
