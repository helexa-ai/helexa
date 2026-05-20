//! Custom architecture implementations.
//!
//! When candle-transformers ships a model family unchanged
//! (`models::llama`, `models::qwen3`, `models::qwen3_moe`, etc.), the
//! handler in `harness/candle.rs` just wraps the upstream type in a
//! `ModelArch` variant.
//!
//! When candle has nothing for the architecture and we have to write
//! it from scratch — Qwen3-Next / Qwen3.6 (`qwen3_5`) being the
//! motivating example — the implementation lands here, one file per
//! architecture.
//!
//! Each architecture module is expected to expose:
//! - A `Config` type deserialised from the model's `config.json`
//!   (some architectures nest the real hyperparams under `text_config`,
//!   in which case the module owns the unwrapping).
//! - A `ForCausalLM` struct with `new`, `forward(&mut self, x, offset)
//!   -> Result<Tensor>`, and `clear_kv_cache(&mut self)`.
//!
//! TP-aware analogues live in `harness/tp/tp_<family>.rs` and follow
//! the pattern set by `tp_qwen3.rs`.

pub mod qwen3_5;
