//! Image preprocessing for vision-capable models.
//!
//! Decodes `data:image/...;base64,...` URIs from OpenAI-style
//! `image_url` content parts into the patch tensors a candle vision
//! tower expects. Stage A ships **fixed resolution** — every image
//! is resized to the same target dimensions (default 448×448 for
//! Qwen3.6, configurable per-call) so the patch count is constant
//! per image. Variable resolution per [Qwen2VL convention] is tracked
//! as issue #14.
//!
//! Spec reference: `doc/vision-qwen3_6-spec.md` — preprocessor
//! section.
//!
//! Normalisation: pixel value `p ∈ [0, 255]` becomes
//! `(p/255 - mean) / std`. Qwen3.6's preprocessor_config.json
//! specifies `image_mean = image_std = [0.5, 0.5, 0.5]`, which
//! simplifies to `2p/255 - 1` mapping `[0,255]` → `[-1, 1]`. We
//! still parameterise mean/std so the same code generalises to other
//! VL families (Qwen2-VL uses imagenet stats, for instance).
//!
//! Pipeline (per image):
//!   1. data: URI → base64 decode → bytes
//!   2. bytes → image::DynamicImage (PNG/JPEG/WebP/etc)
//!   3. resize_exact to target H×W (pixel space)
//!   4. RGB→f32, normalise per mean/std
//!   5. layout to (C, H, W) tensor
//!
//! Patchification (cutting the HxW tensor into `patch_size` blocks)
//! happens inside the vision tower's `patch_embed` conv, so this
//! module stops at "preprocessed RGB f32 tensor."

use anyhow::{Context, Result, anyhow};
use base64::Engine;
use image::DynamicImage;
use image::imageops::FilterType;

/// Preprocessing target. Captures the resize dimensions and the
/// channel-wise normalisation constants from the model's
/// `preprocessor_config.json`. Stage A ships a single `qwen3_6()`
/// constructor for fixed-resolution Qwen3.6 preprocessing; other
/// models can ship their own profile when added.
#[derive(Debug, Clone)]
pub struct PreprocessProfile {
    pub target_height: u32,
    pub target_width: u32,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

impl PreprocessProfile {
    /// Stage A profile for Qwen3.6. Resize to 448×448, normalise to
    /// `[-1, 1]` via mean=std=0.5. Fits within the model's
    /// `num_position_embeddings=2304` budget at 28×28 = 784 patches
    /// before merging.
    pub fn qwen3_6() -> Self {
        Self {
            target_height: 448,
            target_width: 448,
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5],
        }
    }

    /// Per-channel CHW tensor length: 3 * H * W.
    pub fn pixels_chw(&self) -> usize {
        3 * (self.target_height as usize) * (self.target_width as usize)
    }
}

/// Decode a `data:image/...;base64,...` URI into an in-memory image.
///
/// Accepts the OpenAI Chat Completions `image_url` shape — a string
/// URL with `data:` scheme and base64 payload. The MIME type is read
/// from the URI for diagnostics but `image::load_from_memory` sniffs
/// the format from the bytes themselves, so the MIME is advisory.
///
/// Bare `http(s)://` URLs are explicitly rejected here — fetching
/// them from a vision-model server is a fingerprintable behaviour
/// (server-side request forgery, infinite recursion if the URL
/// points at the gateway itself, etc.). Clients that want remote
/// images can fetch them and pass base64 themselves.
pub fn decode_data_uri(uri: &str) -> Result<DynamicImage> {
    let after_scheme = uri
        .strip_prefix("data:")
        .ok_or_else(|| anyhow!("image_url must use data: scheme; got {uri:.40}…"))?;
    let (meta, payload) = after_scheme
        .split_once(',')
        .ok_or_else(|| anyhow!("malformed data URI: missing ',' separator"))?;
    if !meta.contains(";base64") {
        anyhow::bail!(
            "data URI must use base64 encoding (got '{meta}'); raw URL-encoded payloads not supported"
        );
    }
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(payload.trim())
        .context("base64-decode image data URI payload")?;
    image::load_from_memory(&bytes).context("decode image bytes (PNG/JPEG/WebP/etc)")
}

/// Resize and normalise an image into a `(3, H, W)` row-major
/// `Vec<f32>` ready to hand to the vision tower's `patch_embed`
/// conv.
///
/// Uses bilinear resampling — Qwen2-VL's reference uses bicubic, but
/// bilinear is what the candle ecosystem standardises on and is
/// faster on CPU. Quality difference is marginal for downstream
/// vision-encoder consumption. The numerical-validation issue (#15)
/// will quantify any discrepancy.
pub fn preprocess(img: &DynamicImage, profile: &PreprocessProfile) -> Vec<f32> {
    let rgb = img
        .resize_exact(
            profile.target_width,
            profile.target_height,
            FilterType::Triangle,
        )
        .to_rgb8();
    let h = profile.target_height as usize;
    let w = profile.target_width as usize;
    let mut out = vec![0.0_f32; 3 * h * w];
    // Row-major (C, H, W). Candle's Conv2d expects NCHW, so this is
    // the natural layout — the caller stacks `n` of these along the
    // batch axis as needed.
    for c in 0..3 {
        let mean = profile.image_mean[c];
        let std = profile.image_std[c];
        for y in 0..h {
            for x in 0..w {
                let pixel = rgb.get_pixel(x as u32, y as u32);
                let raw = pixel[c] as f32 / 255.0;
                out[c * h * w + y * w + x] = (raw - mean) / std;
            }
        }
    }
    out
}

/// Combined helper: decode + preprocess in one call. Most call
/// sites just want the final tensor; the two-step path exists for
/// callers (tests, future video preprocessing) that need the
/// intermediate `DynamicImage`.
pub fn preprocess_data_uri(uri: &str, profile: &PreprocessProfile) -> Result<Vec<f32>> {
    let img = decode_data_uri(uri)?;
    Ok(preprocess(&img, profile))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    /// A 1×1 red PNG, hand-built. Matches the well-known smallest
    /// valid PNG we use in tests/curl examples elsewhere.
    const ONE_BY_ONE_RED_PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==";

    fn red_png_uri() -> String {
        format!("data:image/png;base64,{ONE_BY_ONE_RED_PNG_B64}")
    }

    #[test]
    fn decodes_well_formed_png_data_uri() {
        let img = decode_data_uri(&red_png_uri()).expect("decode 1x1 png");
        assert_eq!(img.width(), 1);
        assert_eq!(img.height(), 1);
    }

    #[test]
    fn rejects_non_data_scheme() {
        let err = decode_data_uri("https://example.com/cat.jpg")
            .expect_err("http(s) URLs must be rejected");
        assert!(format!("{err:#}").contains("data:"));
    }

    #[test]
    fn rejects_malformed_uri_without_comma() {
        let err = decode_data_uri("data:image/png;base64").unwrap_err();
        assert!(format!("{err:#}").contains("','"));
    }

    #[test]
    fn rejects_non_base64_payload() {
        let err = decode_data_uri("data:image/png,raw-bytes-here").unwrap_err();
        assert!(format!("{err:#}").contains("base64"));
    }

    #[test]
    fn rejects_bad_base64_payload() {
        let err = decode_data_uri("data:image/png;base64,not!valid!base64!").unwrap_err();
        assert!(format!("{err:#}").contains("base64"));
    }

    #[test]
    fn rejects_garbage_image_bytes() {
        // Valid base64 ("Hello World!"), invalid image bytes.
        let err = decode_data_uri("data:image/png;base64,SGVsbG8gV29ybGQh").unwrap_err();
        assert!(
            format!("{err:#}").contains("decode image"),
            "should fail at image-decode step"
        );
    }

    #[test]
    fn preprocess_red_image_produces_correct_shape_and_values() {
        let profile = PreprocessProfile::qwen3_6();
        // Build a tiny pure-red image directly, skipping data: URI
        // decoding so this test isolates the resize+normalise path.
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_pixel(2, 2, Rgb([255, 0, 0]));
        let dyn_img = DynamicImage::ImageRgb8(img);
        let out = preprocess(&dyn_img, &profile);

        assert_eq!(out.len(), profile.pixels_chw());
        // After mean=0.5, std=0.5: red channel (255/255=1.0) → (1.0 - 0.5)/0.5 = 1.0
        // green/blue (0.0) → (0.0 - 0.5)/0.5 = -1.0
        let h = profile.target_height as usize;
        let w = profile.target_width as usize;
        assert!(
            (out[0] - 1.0).abs() < 1e-5,
            "R[0] should be 1.0, got {}",
            out[0]
        );
        assert!((out[h * w] - (-1.0)).abs() < 1e-5, "G[0] should be -1.0");
        assert!(
            (out[2 * h * w] - (-1.0)).abs() < 1e-5,
            "B[0] should be -1.0"
        );
        // All values are finite
        assert!(out.iter().all(|v| v.is_finite()), "no NaN/Inf in output");
    }

    #[test]
    fn preprocess_data_uri_end_to_end() {
        let profile = PreprocessProfile::qwen3_6();
        let out = preprocess_data_uri(&red_png_uri(), &profile).expect("e2e preprocess");
        assert_eq!(out.len(), profile.pixels_chw());
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn preprocess_grayscale_image_promotes_to_rgb() {
        let profile = PreprocessProfile::qwen3_6();
        // 1x1 grayscale = 200 → after conversion to RGB, all three
        // channels equal 200, normalised → (200/255 - 0.5)/0.5 ≈ 0.569
        let gray = DynamicImage::ImageLuma8(ImageBuffer::from_pixel(1, 1, image::Luma([200])));
        let out = preprocess(&gray, &profile);
        let expected = ((200.0 / 255.0) - 0.5) / 0.5;
        let h = profile.target_height as usize;
        let w = profile.target_width as usize;
        for c in 0..3 {
            let v = out[c * h * w];
            assert!(
                (v - expected).abs() < 1e-3,
                "channel {c}: expected {expected}, got {v}"
            );
        }
    }
}
