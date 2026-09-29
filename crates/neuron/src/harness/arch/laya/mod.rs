//! Laya — a non-autoregressive decision model (`convaiinnovations/laya`).
//!
//! Not a language model: a ModernBERT encoder plus a small decision
//! head that scores the options of a typed question in one forward
//! pass. The caller assembles each question into one token sequence
//! (`[CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK]
//! opt1 … [SEP] state [SEP]`) and says where each option's `[MASK]`
//! marker sits; this module returns one raw score per marker plus the
//! act head's distribution. Calibration (temperature scaling) and
//! answer shaping happen after, CPU-side, from the checkpoint's
//! `rl_agent_config.json`.
//!
//! The reference implementation is the `laya` SDK's `DecisionModel`
//! (`laya/common.py`), not the `rl_common.py` shipped in the model
//! repository, which has diverged from it.
//!
//! - [`config`] — the two config files a checkpoint ships.
//! - [`backbone`] — ModernBERT, vendored so it runs outside f32.
//! - [`head`] — type embedding, the head's transformer layers, the
//!   marker scorer and the act head.
//! - [`model`] — the composition, batch in, [`DecisionRow`]s out.

pub mod backbone;
pub mod config;
pub mod head;
pub mod model;

pub use config::CheckpointConfig;
pub use model::LayaModel;

/// Question type, as the checkpoint's type embedding indexes it.
pub const QTYPE_CHOICE: u8 = 0;
pub const QTYPE_SCORE: u8 = 1;
pub const QTYPE_NOUL: u8 = 2;

/// One assembled question: its token ids, where each option's
/// `[MASK]` marker sits, and its type.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionSeq {
    pub input_ids: Vec<u32>,
    pub markers: Vec<usize>,
    /// [`QTYPE_CHOICE`], [`QTYPE_SCORE`] or [`QTYPE_NOUL`].
    pub qtype: u8,
}

/// Sequences scored in one forward. One request's questions normally
/// travel together, so the state they share is encoded in one pass.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DecisionBatch {
    pub seqs: Vec<DecisionSeq>,
}

/// One sequence's result, CPU-side.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionRow {
    /// One uncalibrated score per marker, in marker order.
    pub logits: Vec<f32>,
    /// Softmax over the act head's outputs. Index 0 is what the
    /// reference reports as `action.act_probability`.
    pub act: Vec<f32>,
}

#[cfg(test)]
mod tests;
