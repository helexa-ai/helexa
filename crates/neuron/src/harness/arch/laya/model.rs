//! A whole Laya checkpoint: encoder + decision head, batch in, raw
//! scores out.

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::path::Path;

use super::backbone::{Masks, ModernBert};
use super::config::CheckpointConfig;
use super::head::DecisionHead;
use super::{DecisionBatch, DecisionRow};

pub struct LayaModel {
    encoder: ModernBert,
    head: DecisionHead,
    config: CheckpointConfig,
    device: Device,
}

impl LayaModel {
    /// Load one checkpoint directory (the repo root, or one of its
    /// subfolders): `rl_agent_config.json`, `encoder/config.json` and
    /// `model.safetensors`.
    pub fn load(dir: &Path, device: &Device, dtype: DType) -> Result<Self> {
        let config = CheckpointConfig::from_dir(dir)?;
        let weights = dir.join("model.safetensors");
        // SAFETY: the file is memory-mapped read-only for the lifetime
        // of the VarBuilder; nothing writes to a checkpoint in place.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&weights], dtype, device) }
            .with_context(|| format!("mmap {}", weights.display()))?;
        Self::from_var_builder(vb, config, device)
    }

    pub fn from_var_builder(
        vb: VarBuilder,
        config: CheckpointConfig,
        device: &Device,
    ) -> Result<Self> {
        let encoder = ModernBert::load(vb.pp("encoder"), &config.encoder)?;
        let head = DecisionHead::load(
            vb.clone(),
            config.encoder.hidden_size,
            config.laya.head_layers,
            config.laya.n_act(),
        )?;
        Ok(Self {
            encoder,
            head,
            config,
            device: device.clone(),
        })
    }

    pub fn config(&self) -> &CheckpointConfig {
        &self.config
    }

    /// One forward over every sequence in `batch`. Rows are right-padded
    /// to the longest, as the reference collates them; padding changes
    /// nothing about an unpadded row's result.
    pub fn forward(&self, batch: &DecisionBatch) -> Result<Vec<DecisionRow>> {
        let seqs = &batch.seqs;
        ensure!(!seqs.is_empty(), "empty decision batch");
        let max_pos = self.config.encoder.max_position_embeddings;
        let seq_len = seqs.iter().map(|s| s.input_ids.len()).max().unwrap_or(0);
        ensure!(seq_len > 0, "decision batch has an empty sequence");
        ensure!(
            seq_len <= max_pos,
            "sequence of {seq_len} tokens exceeds max_position_embeddings {max_pos}"
        );
        let vocab = self.config.encoder.vocab_size as u32;
        let pad = self.config.encoder.pad_token_id;
        let b = seqs.len();

        let mut ids = Vec::with_capacity(b * seq_len);
        let mut lens = Vec::with_capacity(b);
        let mut qtypes = Vec::with_capacity(b);
        let k_max = seqs.iter().map(|s| s.markers.len()).max().unwrap_or(0);
        ensure!(
            k_max > 0,
            "decision batch has a sequence with no option markers"
        );
        let mut marker_idx = Vec::with_capacity(b * k_max);
        for s in seqs {
            let n = s.input_ids.len();
            ensure!(n > 0, "decision batch has an empty sequence");
            ensure!(!s.markers.is_empty(), "sequence has no option markers");
            ensure!(s.qtype < 3, "question type {} out of range", s.qtype);
            if let Some(&bad) = s.input_ids.iter().find(|&&t| t >= vocab) {
                anyhow::bail!("token id {bad} outside vocabulary of {vocab}");
            }
            if let Some(&bad) = s.markers.iter().find(|&&m| m >= n) {
                anyhow::bail!("marker position {bad} outside a {n}-token sequence");
            }
            ids.extend_from_slice(&s.input_ids);
            ids.extend(std::iter::repeat_n(pad, seq_len - n));
            lens.push(n);
            qtypes.push(s.qtype as u32);
            marker_idx.extend(s.markers.iter().map(|&m| m as u32));
            // Absent slots gather position 0; their scores are discarded.
            marker_idx.extend(std::iter::repeat_n(0u32, k_max - s.markers.len()));
        }

        let ids = Tensor::from_vec(ids, (b, seq_len), &self.device)?;
        let masks = Masks::new(&lens, seq_len, self.encoder.half_window(), &self.device)?;
        let qtype = Tensor::from_vec(qtypes, b, &self.device)?;

        let h = self.encoder.forward(&ids, &masks)?;
        let h = self.head.contextualise(&h, &qtype, &masks.global)?;
        let d = self.config.encoder.hidden_size;
        let idx = Tensor::from_vec(marker_idx, (b, k_max, 1), &self.device)?
            .broadcast_as((b, k_max, d))?
            .contiguous()?;
        let marker_h = h.gather(&idx, 1)?;
        let logits: Vec<Vec<f32>> = self.head.score(&marker_h)?.to_vec2()?;

        let mut feats = Vec::with_capacity(b * 4);
        let mut row_logits = Vec::with_capacity(b);
        for (s, full) in seqs.iter().zip(logits) {
            let row: Vec<f32> = full[..s.markers.len()].to_vec();
            feats.extend_from_slice(&act_features(&row));
            row_logits.push(row);
        }
        let feats = Tensor::from_vec(feats, (b, 4), &self.device)?;
        let pooled = h.narrow(1, 0, 1)?.squeeze(1)?;
        let act_logits = self.head.act(&pooled, &feats)?;
        let act = candle_nn::ops::softmax_last_dim(&act_logits)?.to_vec2::<f32>()?;

        Ok(row_logits
            .into_iter()
            .zip(act)
            .map(|(logits, act)| DecisionRow { logits, act })
            .collect())
    }
}

/// The act head's four inputs, from one row's uncalibrated logits:
/// top-1 probability, top-1 minus top-2, entropy normalised by
/// `ln(max(k, 2))`, and `k / 255`.
///
/// The reference computes these over the batch's padded width, with
/// absent option slots filled with a logit of -1e4. Those slots carry
/// exactly zero probability in f32, so computing per row over only the
/// real options gives the same result.
pub fn act_features(logits: &[f32]) -> [f32; 4] {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&z| (z - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let mut p: Vec<f32> = exps.iter().map(|e| e / sum).collect();
    let k = (logits.len().max(2)) as f32;
    let ent = -p.iter().map(|&x| x * x.max(1e-9).ln()).sum::<f32>() / k.ln();
    p.sort_by(|a, b| b.total_cmp(a));
    let top1 = p[0];
    let top2 = p.get(1).copied().unwrap_or(0.0);
    [top1, top1 - top2, ent, k / 255.0]
}
