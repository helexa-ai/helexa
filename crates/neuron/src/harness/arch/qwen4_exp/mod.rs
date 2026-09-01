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
//! - [`config`] — the checkpoint's own `config.json`, and what its
//!   fields mean where they are not what they look like.
//! - [`hyper`] — hyper-connections, the residual structure itself.
//! - [`ple`] — the hashed n-gram table on layer 1: how its rows
//!   are addressed, and how the gathered rows are consumed.
//! - [`qsa`] — which blocks of the past a full-attention layer is
//!   allowed to look at.

pub mod config;
pub mod hyper;
pub mod ple;
pub mod qsa;
