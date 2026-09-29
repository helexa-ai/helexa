//! Laya decision models: everything around the forward pass (#335).
//!
//! A Laya checkpoint answers typed questions over a *state* (text, a JSON
//! document, a conversation) in one encoder forward pass: each question
//! becomes one sequence,
//!
//! ```text
//! [CLS] "<type> question: <instructions>" [SEP] [MASK] opt0 [MASK] opt1 … [SEP] state [SEP]
//! ```
//!
//! and the decision head scores the hidden state at each option's `[MASK]`
//! marker. This module is the part that touches no tensor:
//!
//! - [`question`] — validate and normalise the question definitions, and
//!   render each option's text;
//! - [`sequence`] — assemble the token sequences under the model's token
//!   budgets ([`prepare`]);
//! - [`shape`] — turn per-option logits into calibrated, typed answers
//!   ([`shape()`]);
//! - [`config`] / [`tokenizer`] — the per-checkpoint pieces those need.
//!
//! The reference is the `laya` Python SDK (`laya/common.py`,
//! `laya/agent.py`, `laya/serve.py`), whose behaviour this reproduces token
//! for token: the parity tests replay fixtures recorded from it
//! (`script/laya-reference.py`). The calibration arithmetic follows numpy's
//! float32 evaluation order, so rounded probabilities match to the last
//! published digit.
//!
//! The wire types (`/v1/systemone`) live in [`cortex_core::decisions`].

pub mod config;
pub mod question;
pub mod sequence;
pub mod shape;
pub mod tokenizer;

pub use config::DecisionConfig;
pub use cortex_core::decisions::DecisionRequestError;
pub use question::{QType, Question};
pub use sequence::{Budget, OptionStats, Prepared, Row, prepare};
pub use shape::{RowOutput, shape};
pub use tokenizer::{DecisionTokenizer, SpecialTokens, TextEncoder};

/// The constant `model` field of every response: the decision head's name.
/// Which checkpoint answered is reported in `routing`.
pub const DECISION_MODEL_NAME: &str = "laya-rl-agent";

/// A request error in the reference's `ValueError` class: the server
/// answers it with 422 and the message as `detail`.
pub(crate) fn unprocessable(detail: impl Into<String>) -> DecisionRequestError {
    DecisionRequestError::new(422, detail)
}

#[cfg(test)]
mod parity_tests;
