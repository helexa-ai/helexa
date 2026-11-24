//! Z-Image text-to-image models.
//!
//! Vendored from candle-transformers' `z_image` (huggingface/candle at
//! 0.11.0, MIT OR Apache-2.0; original module by SpenserCai) together
//! with neuron's four changes to it, which used to live in the
//! `grenade/candle` fork (#426): an all-ones attention-mask elision so
//! flash-attn engages (offered upstream as huggingface/candle#3798), the
//! quantized transformer `QuantizedZImageTransformer2DModel`, reuse of
//! the transformer's `Config`, and an f32 -> bf16 cast around flash-attn
//! on the quantized path. Changes here beyond those: crate paths.
//! Keeping the module here lets neuron depend on stock candle.
//!
//! Z-Image is a text-to-image generation model from Alibaba using Flow Matching.
//!
//! - 🤗 [Hugging Face Model](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
//! - [Official Website](https://z-image-turbo.org/)
//!
//! # Example
//!
//! ```bash
//! cargo run --features metal --example z_image --release -- \
//!     --prompt "A beautiful landscape" --height 1024 --width 1024
//! ```
//!
//! # Architecture
//!
//! - Transformer: ~24B parameters, 30 main layers + 2 noise_refiner + 2 context_refiner
//! - Text Encoder: Qwen3 (hidden_size=2560, 36 layers)
//! - VAE: AutoencoderKL (diffusers format)
//! - Scheduler: FlowMatchEulerDiscreteScheduler (shift=3.0)

pub mod preprocess;
pub mod quantized_transformer;
pub mod sampling;
pub mod scheduler;
pub mod text_encoder;
pub mod transformer;
pub mod vae;

// Re-export main types
pub use preprocess::{PreparedInputs, prepare_inputs};
pub use quantized_transformer::QuantizedZImageTransformer2DModel;
pub use sampling::{get_noise, get_schedule, postprocess_image};
pub use scheduler::{FlowMatchEulerDiscreteScheduler, SchedulerConfig, calculate_shift};
pub use text_encoder::{TextEncoderConfig, ZImageTextEncoder};
pub use transformer::{Config, ZImageTransformer2DModel};
pub use vae::{AutoEncoderKL, VaeConfig};
