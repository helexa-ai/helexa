//! Z-Image text-to-image pipeline (#198).
//!
//! Wraps candle-transformers' `z_image` model family (6B single-stream
//! DiT + Qwen3-4B text encoder + 16-ch VAE, flow matching) behind the
//! same device-worker discipline as the text archs: every CUDA op runs
//! on the context-owning worker thread, and tensors never escape it
//! alive. The async side hands in prompt tokens and generation
//! parameters; the worker replies with CPU-side RGB pixels.
//!
//! Component lifetime is the VRAM story (bench, 2026-07-29 on beast):
//! all-resident BF16 is ~25 GB peak and OOMs a 32 GB card at 1024²,
//! while dropping the ~8 GB text encoder after prompt encoding brings
//! steady state to ~13 GB (DiT + VAE). The pipeline therefore keeps
//! the DiT and VAE resident and rebuilds the text encoder from the
//! mmap'd safetensors per generation (page-cache warm: ~1.2 s),
//! unless `te_resident` asks to pin it.
//!
//! Everything in this module except [`ZImagePipeline::load`] and the
//! forward passes is pure CPU logic, unit-testable without weights.

use anyhow::{Context, Result};
use candle_core::quantized::GgmlDType;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::z_image::{
    AutoEncoderKL, Config as DitConfig, FlowMatchEulerDiscreteScheduler,
    QuantizedZImageTransformer2DModel, SchedulerConfig, TextEncoderConfig, VaeConfig,
    ZImageTextEncoder, ZImageTransformer2DModel, calculate_shift, get_noise, postprocess_image,
};
use std::path::PathBuf;

/// Scheduler shift constants from the reference pipeline. Identical to
/// the upstream candle example; the shift interpolates between
/// BASE_SHIFT and MAX_SHIFT by image sequence length.
const BASE_IMAGE_SEQ_LEN: usize = 256;
const MAX_IMAGE_SEQ_LEN: usize = 4096;
const BASE_SHIFT: f64 = 0.5;
const MAX_SHIFT: f64 = 1.15;

/// Image dimensions must divide the VAE alignment: vae_scale_factor
/// (8) × patch_size (2).
pub const DIM_ALIGN: usize = 16;

/// Default and ceiling generation parameters. The ceiling is enforced
/// before admission so an oversized request never reaches the device
/// (an OOM would poison the worker context).
pub const DEFAULT_STEPS: usize = 9;
pub const MAX_STEPS: usize = 50;
pub const DEFAULT_MAX_DIM: usize = 2048;
pub const DEFAULT_GUIDANCE_SCALE: f64 = 5.0;

/// Resolved on-disk layout of a diffusers-format Z-Image repository.
///
/// Assembled by `resolve_image_files` in `candle.rs` (async, hf-hub)
/// and consumed by [`ZImagePipeline::load`] on the worker thread.
#[derive(Debug, Clone)]
pub struct ZImageFiles {
    pub tokenizer: PathBuf,
    pub text_encoder_config: PathBuf,
    pub text_encoder_shards: Vec<PathBuf>,
    pub transformer_config: PathBuf,
    pub transformer_shards: Vec<PathBuf>,
    pub vae_config: PathBuf,
    pub vae_weights: PathBuf,
}

/// One generation request, as the worker sees it. Prompt text has
/// already been chat-templated and tokenized on the async side —
/// the worker receives ids only.
#[derive(Debug, Clone)]
pub struct ImageGenParams {
    pub tokens: Vec<u32>,
    /// Negative-prompt tokens; `Some` enables classifier-free guidance,
    /// which doubles per-step cost (two DiT forwards).
    pub negative_tokens: Option<Vec<u32>>,
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    pub guidance_scale: f64,
    pub seed: Option<u64>,
}

/// Wall-clock phase breakdown for one generation, surfaced to the
/// caller the same way text inference surfaces `helexa_timing`.
#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct ImageGenTiming {
    pub encode_ms: u64,
    pub denoise_ms: u64,
    pub decode_ms: u64,
    pub steps: usize,
    /// True when CFG ran (negative prompt present): two forwards/step.
    pub cfg: bool,
}

/// CPU-side result of one generation: tightly-packed row-major RGB.
pub struct ImageGenResult {
    pub rgb: Vec<u8>,
    pub width: usize,
    pub height: usize,
    pub timing: ImageGenTiming,
}

/// Validate requested dimensions against alignment and the configured
/// ceiling. Returns the latent dims `(latent_h, latent_w)` on success.
///
/// The latent formula is `2 * (px / 16)` per the reference pipeline:
/// divisible by patch_size=2, and VAE decode (8×) restores the pixel
/// dimension exactly.
pub fn validate_dims(width: usize, height: usize, max_dim: usize) -> Result<(usize, usize)> {
    if width == 0 || height == 0 {
        anyhow::bail!("image dimensions must be non-zero, got {width}x{height}");
    }
    if !width.is_multiple_of(DIM_ALIGN) || !height.is_multiple_of(DIM_ALIGN) {
        anyhow::bail!("image dimensions must be divisible by {DIM_ALIGN}, got {width}x{height}");
    }
    if width > max_dim || height > max_dim {
        anyhow::bail!(
            "image dimensions {width}x{height} exceed the configured ceiling {max_dim}x{max_dim}"
        );
    }
    Ok((2 * (height / DIM_ALIGN), 2 * (width / DIM_ALIGN)))
}

/// Bound the step count. Zero steps is meaningless; past MAX_STEPS the
/// turbo checkpoint adds nothing but latency.
pub fn validate_steps(steps: usize) -> Result<usize> {
    if steps == 0 || steps > MAX_STEPS {
        anyhow::bail!("num_steps must be in 1..={MAX_STEPS}, got {steps}");
    }
    Ok(steps)
}

/// Qwen3 chat-template wrapper the Z-Image text encoder expects
/// (`add_generation_prompt=True`, matching the reference pipeline).
pub fn format_prompt(prompt: &str) -> String {
    format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
}

/// Metered work for one generation in megapixel-steps (#202): pixel
/// count × denoise steps, doubled under CFG. The per-image floor is
/// applied by the caller (a 512²/9-step image is well under one unit).
pub fn megapixel_steps(width: usize, height: usize, steps: usize, cfg: bool) -> f64 {
    let mp = (width as f64 * height as f64) / 1_000_000.0;
    let factor = if cfg { 2.0 } else { 1.0 };
    mp * steps as f64 * factor
}

/// Tiled VAE decode (#199): decode the latent in overlapping tiles and
/// blend the seams, bounding the decoder's conv/attention transients to
/// one tile regardless of output resolution. Full-frame decode at
/// 1024² needs more transient VRAM than a 24 GB card has left beside
/// the resident DiT (benjy E2E, 2026-07-29); at ≥1536² it exceeds even
/// a 32 GB card. Port of diffusers' `AutoencoderKL.tiled_decode`
/// linear seam blending.
///
/// `tile` and `overlap` are in latent units (×8 = pixels). Tiles are
/// placed at `stride = tile - overlap`; the final tile is pulled back
/// flush with the edge so every tile is full-size.
fn tiled_decode(
    vae: &AutoEncoderKL,
    latents: &Tensor,
    tile: usize,
    overlap: usize,
) -> Result<Tensor> {
    let (_b, _c, lh, lw) = latents.dims4()?;
    if lh <= tile && lw <= tile {
        return vae.decode(latents).map_err(anyhow::Error::from);
    }
    let stride = tile - overlap;
    let positions = |extent: usize| -> Vec<usize> {
        if extent <= tile {
            return vec![0];
        }
        let mut out = Vec::new();
        let mut pos = 0;
        loop {
            if pos + tile >= extent {
                out.push(extent - tile);
                break;
            }
            out.push(pos);
            pos += stride;
        }
        out
    };
    let rows = positions(lh);
    let cols = positions(lw);

    // Decode every tile first (each is small: 3 × (tile·8)² bf16).
    let mut decoded: Vec<Vec<Tensor>> = Vec::with_capacity(rows.len());
    for &r in &rows {
        let mut row_tiles = Vec::with_capacity(cols.len());
        for &c in &cols {
            let tile_latent = latents
                .narrow(2, r, tile.min(lh))?
                .narrow(3, c, tile.min(lw))?;
            row_tiles.push(vae.decode(&tile_latent)?);
        }
        decoded.push(row_tiles);
    }

    // Blend seams. Overlap in pixels is the *actual* overlap between
    // adjacent tile placements (the edge-flushed last tile may overlap
    // its neighbour by more; blend over the full shared region).
    let px = 8; // VAE upsampling factor
    let (_, ch, th, tw) = decoded[0][0].dims4()?;
    let device = latents.device();
    let dtype = decoded[0][0].dtype();
    let out_h = lh * px;
    let out_w = lw * px;

    // Accumulate into weighted sums so arbitrary overlaps compose.
    let mut acc = Tensor::zeros((1, ch, out_h, out_w), candle_core::DType::F32, device)?;
    let mut weight = Tensor::zeros((1, 1, out_h, out_w), candle_core::DType::F32, device)?;
    for (ri, &r) in rows.iter().enumerate() {
        for (ci, &c) in cols.iter().enumerate() {
            let t = decoded[ri][ci].to_dtype(candle_core::DType::F32)?;
            // Per-tile blend weight: linear ramp up over the overlap
            // on edges that have a neighbour, flat 1.0 elsewhere.
            let ramp = |n: usize, lead: bool, trail: bool| -> Vec<f32> {
                let ov = overlap * px;
                (0..n)
                    .map(|i| {
                        let mut w = 1.0f32;
                        if lead && i < ov {
                            w = w.min((i + 1) as f32 / (ov + 1) as f32);
                        }
                        if trail && i >= n - ov {
                            w = w.min((n - i) as f32 / (ov + 1) as f32);
                        }
                        w
                    })
                    .collect()
            };
            let wy = ramp(th, ri > 0, ri + 1 < rows.len());
            let wx = ramp(tw, ci > 0, ci + 1 < cols.len());
            let wy = Tensor::from_vec(wy, (1, 1, th, 1), device)?;
            let wx = Tensor::from_vec(wx, (1, 1, 1, tw), device)?;
            let w2d = wy.broadcast_mul(&wx)?; // (1,1,th,tw)

            let weighted = t.broadcast_mul(&w2d)?;
            // acc[.., r*8.., c*8..] += weighted — via slice_assign on
            // narrowed views (candle has no in-place scatter-add on
            // views; rebuild with slice_assign of the summed region).
            let (y0, x0) = (r * px, c * px);
            let acc_region = acc.narrow(2, y0, th)?.narrow(3, x0, tw)?;
            let acc_new = (acc_region + weighted)?;
            acc = acc.slice_assign(&[0..1, 0..ch, y0..y0 + th, x0..x0 + tw], &acc_new)?;
            let w_region = weight.narrow(2, y0, th)?.narrow(3, x0, tw)?;
            let w_new = w_region.broadcast_add(&w2d)?;
            weight = weight.slice_assign(&[0..1, 0..1, y0..y0 + th, x0..x0 + tw], &w_new)?;
        }
    }
    let out = acc.broadcast_div(&weight)?;
    out.to_dtype(dtype).map_err(anyhow::Error::from)
}

/// The worker-thread-resident pipeline state for one loaded Z-Image
/// model. Owned by `DeviceWorkerState::image_models`; construction,
/// every forward, and Drop all happen on the worker thread.
/// The DiT in either precision. Quantized (#204) loads the same dense
/// safetensors and quantizes in situ; activations run in f32 (the
/// quantized kernels accumulate there anyway).
enum Dit {
    Dense(ZImageTransformer2DModel),
    Quantized(QuantizedZImageTransformer2DModel),
}

impl Dit {
    fn forward(
        &self,
        latents: &Tensor,
        t: &Tensor,
        cap_feats: &Tensor,
        cap_mask: &Tensor,
    ) -> Result<Tensor> {
        match self {
            Dit::Dense(m) => Ok(m.forward(latents, t, cap_feats, cap_mask)?),
            Dit::Quantized(m) => Ok(m.forward(latents, t, cap_feats, cap_mask)?),
        }
    }
}

pub struct ZImagePipeline {
    device: Device,
    dtype: DType,
    /// Where the text encoder runs. `Device::Cpu` (the default) keeps
    /// the ~8 GB Qwen3-4B entirely off the GPU: prompt encoding is one
    /// short-sequence forward, the resulting `(1, seq, 2560)` features
    /// are trivially copied to the DiT's device, and the 24 GB tier
    /// (RTX 4090) gains the headroom that the 2026-07-29 benjy E2E
    /// showed it needs (DiT 14.6 GB resident + 8 GB GPU TE = OOM).
    te_device: Device,
    te_dtype: DType,
    files: ZImageFiles,
    dit: Dit,
    vae: AutoEncoderKL,
    /// Text encoder residency: `Some` when pinned (`te_resident` at
    /// load) or between rebuild and drop within a generation; `None`
    /// otherwise.
    text_encoder: Option<ZImageTextEncoder>,
    te_config: TextEncoderConfig,
}

impl ZImagePipeline {
    /// Build the pipeline on the current thread. The DiT and VAE are
    /// loaded resident; the text encoder is loaded now only when
    /// `te_resident` (otherwise it is rebuilt per generation).
    pub fn load(
        files: ZImageFiles,
        device: &Device,
        te_on_cpu: bool,
        te_resident: bool,
        quant: Option<GgmlDType>,
    ) -> Result<Self> {
        // Quantized runs the whole pipeline in f32: the quantized
        // matmuls accumulate there, and mixing dtypes only adds casts.
        let dtype = match quant {
            Some(_) => DType::F32,
            None => device.bf16_default_to_f32(),
        };
        // CPU runs the TE in f32 — candle's CPU bf16 path is a
        // convert-per-op crawl, and precision only helps here.
        let (te_device, te_dtype) = if te_on_cpu {
            (Device::Cpu, DType::F32)
        } else {
            (device.clone(), dtype)
        };

        let dit_cfg: DitConfig = serde_json::from_reader(
            std::fs::File::open(&files.transformer_config)
                .context("open transformer/config.json")?,
        )
        .context("parse transformer/config.json")?;
        let dit_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&files.transformer_shards, dtype, device)?
        };
        let dit = match quant {
            Some(dt) => Dit::Quantized(
                QuantizedZImageTransformer2DModel::new(&dit_cfg, dit_vb, dt)
                    .context("build quantized Z-Image transformer")?,
            ),
            None => Dit::Dense(
                ZImageTransformer2DModel::new(&dit_cfg, dit_vb)
                    .context("build Z-Image transformer")?,
            ),
        };

        let vae_cfg: VaeConfig = serde_json::from_reader(
            std::fs::File::open(&files.vae_config).context("open vae/config.json")?,
        )
        .context("parse vae/config.json")?;
        let vae_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                std::slice::from_ref(&files.vae_weights),
                dtype,
                device,
            )?
        };
        let vae = AutoEncoderKL::new(&vae_cfg, vae_vb).context("build Z-Image VAE")?;

        let te_config: TextEncoderConfig = serde_json::from_reader(
            std::fs::File::open(&files.text_encoder_config)
                .context("open text_encoder/config.json")?,
        )
        .context("parse text_encoder/config.json")?;

        let mut pipeline = Self {
            device: device.clone(),
            dtype,
            te_device,
            te_dtype,
            files,
            dit,
            vae,
            text_encoder: None,
            te_config,
        };
        if te_resident {
            pipeline.text_encoder = Some(pipeline.build_text_encoder()?);
        }
        Ok(pipeline)
    }

    fn build_text_encoder(&self) -> Result<ZImageTextEncoder> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &self.files.text_encoder_shards,
                self.te_dtype,
                &self.te_device,
            )?
        };
        ZImageTextEncoder::new(&self.te_config, vb).context("build Z-Image text encoder")
    }

    /// Encode one token sequence to caption features + all-ones mask.
    fn encode(&self, te: &ZImageTextEncoder, tokens: &[u32]) -> Result<(Tensor, Tensor)> {
        let ids = Tensor::from_vec(tokens.to_vec(), (1, tokens.len()), &self.te_device)?;
        let feats = te
            .forward(&ids)?
            .to_device(&self.device)?
            .to_dtype(self.dtype)?;
        let mask = Tensor::ones((1, tokens.len()), DType::U8, &self.device)?;
        Ok((feats, mask))
    }

    /// Run one full generation on the calling (worker) thread.
    ///
    /// Staged lifetime: the text encoder exists only for the encode
    /// phase unless pinned; the DiT and VAE stay resident across
    /// generations. Latents and embeddings drop at return — nothing
    /// device-side survives except the resident weights.
    pub fn generate(&mut self, params: &ImageGenParams, max_dim: usize) -> Result<ImageGenResult> {
        let (latent_h, latent_w) = validate_dims(params.width, params.height, max_dim)?;
        let steps = validate_steps(params.steps)?;
        if params.tokens.is_empty() {
            anyhow::bail!("prompt tokens must be non-empty");
        }
        if let Some(seed) = params.seed {
            self.device.set_seed(seed)?;
        }

        // ---- Encode (text encoder alive only inside this block unless pinned)
        let t_encode = std::time::Instant::now();
        let cfg_active = params.negative_tokens.is_some() && params.guidance_scale > 1.0;
        let (cap_feats, cap_mask, neg) = {
            let te_owned;
            let te = match &self.text_encoder {
                Some(te) => te,
                None => {
                    te_owned = self.build_text_encoder()?;
                    &te_owned
                }
            };
            let (cap_feats, cap_mask) = self.encode(te, &params.tokens)?;
            let neg = match (&params.negative_tokens, cfg_active) {
                (Some(neg_tokens), true) => Some(self.encode(te, neg_tokens)?),
                _ => None,
            };
            (cap_feats, cap_mask, neg)
        };
        let encode_ms = t_encode.elapsed().as_millis() as u64;

        // ---- Denoise
        let t_denoise = std::time::Instant::now();
        let mut scheduler = FlowMatchEulerDiscreteScheduler::new(SchedulerConfig::z_image_turbo());
        let image_seq_len = (latent_h / 2) * (latent_w / 2);
        let mu = calculate_shift(
            image_seq_len,
            BASE_IMAGE_SEQ_LEN,
            MAX_IMAGE_SEQ_LEN,
            BASE_SHIFT,
            MAX_SHIFT,
        );
        scheduler.set_timesteps(steps, Some(mu));

        let mut latents = get_noise(1, 16, latent_h, latent_w, &self.device)?
            .to_dtype(self.dtype)?
            .unsqueeze(2)?; // (B, C, H, W) -> (B, C, 1, H, W)

        for _step in 0..steps {
            let t = scheduler.current_timestep_normalized();
            let t_tensor =
                Tensor::from_vec(vec![t as f32], (1,), &self.device)?.to_dtype(self.dtype)?;

            let noise_pred = self
                .dit
                .forward(&latents, &t_tensor, &cap_feats, &cap_mask)?;
            let noise_pred = match &neg {
                Some((neg_feats, neg_mask)) => {
                    let neg_pred = self.dit.forward(&latents, &t_tensor, neg_feats, neg_mask)?;
                    let diff = (&noise_pred - &neg_pred)?;
                    (&neg_pred + (diff * params.guidance_scale)?)?
                }
                None => noise_pred,
            };
            // Z-Image predicts the negated flow.
            let noise_pred = noise_pred.neg()?.squeeze(2)?;
            let latents_4d = latents.squeeze(2)?;
            latents = scheduler.step(&noise_pred, &latents_4d)?.unsqueeze(2)?;
        }
        let denoise_ms = t_denoise.elapsed().as_millis() as u64;

        // ---- Decode
        let t_decode = std::time::Instant::now();
        let latents = latents.squeeze(2)?;
        // Tile latent = 64 (512 px) with 16-latent (128 px) seam
        // overlap: transients stay bounded to one 512² decode.
        let image = tiled_decode(&self.vae, &latents, 64, 16)?;
        let image = postprocess_image(&image)?; // [-1,1] -> u8 [0,255]
        let image = image.i(0)?; // drop batch dim -> (3, H, W)
        let (_c, h, w) = image.dims3()?;
        // (3, H, W) -> (H, W, 3) row-major RGB on the CPU.
        let rgb = image
            .permute((1, 2, 0))?
            .flatten_all()?
            .to_device(&Device::Cpu)?
            .to_vec1::<u8>()?;
        let decode_ms = t_decode.elapsed().as_millis() as u64;

        if h != params.height || w != params.width {
            anyhow::bail!(
                "decoded image is {w}x{h}, expected {}x{}",
                params.width,
                params.height
            );
        }

        Ok(ImageGenResult {
            rgb,
            width: w,
            height: h,
            timing: ImageGenTiming {
                encode_ms,
                denoise_ms,
                decode_ms,
                steps,
                cfg: cfg_active,
            },
        })
    }
}

/// Parse the operator's `quant` string for the image path (#204).
/// Only `q8_0` is accepted — the one dtype with a fixed-seed A/B
/// against BF16; broaden deliberately, with comparisons, not by
/// default.
pub fn parse_image_quant(quant: &str) -> Result<GgmlDType> {
    match quant.to_ascii_lowercase().as_str() {
        "q8_0" | "q8" => Ok(GgmlDType::Q8_0),
        other => anyhow::bail!(
            "unsupported image quant '{other}'; the image path serves q8_0 \
             (bf16 when quant is omitted)"
        ),
    }
}

/// True when a repo file listing is a diffusers-style pipeline rather
/// than a transformers-style LLM: `model_index.json` at the root is
/// the marker diffusers itself uses.
pub fn is_diffusers_layout(filenames: &[&str]) -> bool {
    filenames.contains(&"model_index.json")
}

/// True when a diffusers `model_index.json` declares a pipeline class
/// this module can serve.
pub fn is_supported_pipeline_class(model_index_json: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(model_index_json)
        .ok()
        .and_then(|v| {
            v.get("_class_name")
                .and_then(|c| c.as_str())
                .map(|c| c == "ZImagePipeline")
        })
        .unwrap_or(false)
}

/// Encode packed RGB rows to PNG bytes. CPU-side; called from the
/// async handler, never the worker thread.
pub fn encode_png(rgb: &[u8], width: usize, height: usize) -> Result<Vec<u8>> {
    anyhow::ensure!(
        rgb.len() == width * height * 3,
        "rgb buffer is {} bytes, expected {}",
        rgb.len(),
        width * height * 3
    );
    let img: image::RgbImage =
        image::ImageBuffer::from_raw(width as u32, height as u32, rgb.to_vec())
            .context("rgb buffer does not match dimensions")?;
    let mut out = std::io::Cursor::new(Vec::new());
    img.write_to(&mut out, image::ImageFormat::Png)
        .context("encode png")?;
    Ok(out.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_dims_accepts_aligned() {
        assert_eq!(validate_dims(1024, 1024, 2048).unwrap(), (128, 128));
        assert_eq!(validate_dims(512, 768, 2048).unwrap(), (96, 64));
    }

    #[test]
    fn validate_dims_rejects_misaligned() {
        assert!(validate_dims(1000, 1024, 2048).is_err());
        assert!(validate_dims(1024, 1018, 2048).is_err());
        assert!(validate_dims(0, 1024, 2048).is_err());
    }

    #[test]
    fn validate_dims_rejects_over_ceiling() {
        assert!(validate_dims(2064, 1024, 2048).is_err());
        assert!(validate_dims(1024, 4096, 2048).is_err());
        // At the ceiling is fine.
        assert!(validate_dims(2048, 2048, 2048).is_ok());
    }

    #[test]
    fn validate_steps_bounds() {
        assert!(validate_steps(0).is_err());
        assert!(validate_steps(MAX_STEPS + 1).is_err());
        assert_eq!(validate_steps(9).unwrap(), 9);
    }

    #[test]
    fn format_prompt_wraps_chat_template() {
        let p = format_prompt("a cat");
        assert!(p.starts_with("<|im_start|>user\na cat<|im_end|>"));
        assert!(p.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn megapixel_steps_tracks_cost() {
        // 1024² × 9 steps ≈ 9.44 Mp-steps; CFG doubles it.
        let base = megapixel_steps(1024, 1024, 9, false);
        assert!((base - 9.437184).abs() < 1e-6);
        assert_eq!(megapixel_steps(1024, 1024, 9, true), base * 2.0);
        // 512² is a quarter of the pixels.
        assert!((megapixel_steps(512, 512, 9, false) - base / 4.0).abs() < 1e-6);
    }

    #[test]
    fn diffusers_layout_detection() {
        assert!(is_diffusers_layout(&[
            "model_index.json",
            "transformer/config.json"
        ]));
        assert!(!is_diffusers_layout(&["config.json", "model.safetensors"]));
    }

    #[test]
    fn pipeline_class_gate() {
        assert!(is_supported_pipeline_class(
            r#"{"_class_name": "ZImagePipeline", "_diffusers_version": "0.36.0"}"#
        ));
        assert!(!is_supported_pipeline_class(
            r#"{"_class_name": "FluxPipeline"}"#
        ));
        assert!(!is_supported_pipeline_class("not json"));
    }

    #[test]
    fn encode_png_roundtrip() {
        let rgb = vec![128u8; 16 * 16 * 3];
        let png = encode_png(&rgb, 16, 16).unwrap();
        assert_eq!(&png[1..4], b"PNG");
        assert!(encode_png(&rgb, 17, 16).is_err());
    }
}
