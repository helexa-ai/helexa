//! `qwen4_exp` — the architecture behind `Qwen/Qwen3.8-Flash-Next`.
//!
//! A Qwen 4 experimental generation rather than a Qwen3 variant. The
//! hybrid skeleton (36 linear-attention layers to 12 full-attention,
//! `full_attention_interval: 4`) and the linear-attention tensor layout
//! are shared with [`super::qwen3_5`], so much of that implementation
//! carries over. What is new is the residual structure — four streams
//! joined by hyper-connections, with no layernorms of the usual kind —
//! plus QSA sparse attention and PLE hashed n-gram embeddings.
//!
//! The written specification lives at `doc/qwen4_exp-port-spec.md`;
//! it is the reference for every dimension and dataflow here, and it
//! records what was measured from the checkpoint rather than inferred.
//!
//! Built in dependency order, so each piece can be parity-tested before
//! anything stacks on it:
//! - [`hyper`] — hyper-connections, the residual structure itself.

pub mod hyper;
