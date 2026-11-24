//! The native qwen4_exp decode engine (#427).
//!
//! A decode step as fixed work over buffers allocated once: Strata's
//! kernels (and neuron's own) driven over device pointers, so the step can
//! be captured as one CUDA graph. Candle remains the loader for
//! everything else, the prompt path, and the parity oracle every layer
//! here is checked against.

pub mod weights;

#[cfg(feature = "strata-kernels")]
pub mod attn;
#[cfg(feature = "strata-kernels")]
pub mod decoder;
#[cfg(feature = "strata-kernels")]
pub mod gdn;
#[cfg(feature = "strata-kernels")]
pub mod moe;
#[cfg(feature = "strata-kernels")]
pub mod ple;
