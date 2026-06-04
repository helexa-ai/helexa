//! Image preprocessing for vision-capable models.
//!
//! Decodes `data:image/...;base64,...` URIs from OpenAI-style
//! `image_url` content parts into the patch tensors a candle vision
//! tower expects. Resolution is **dynamic** (#14): each image is
//! resized to its native aspect via Qwen `smart_resize` — a
//! factor-aligned `(h, w)` whose pixel count lands in the profile's
//! `[min_pixels, max_pixels]` budget — so the LM token count varies per
//! image (`(h/factor) × (w/factor)`).
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
//!   3. smart_resize to a native-aspect, factor-aligned H×W (pixel space)
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

/// Preprocessing target. Captures the resize policy (Qwen `smart_resize`
/// factor + pixel budget) and the channel-wise normalisation constants
/// from the model's `preprocessor_config.json`. Images are resized to
/// their **native aspect** — a factor-aligned `(h, w)` whose pixel count
/// lands in `[min_pixels, max_pixels]` — not a fixed square (#14).
#[derive(Debug, Clone)]
pub struct PreprocessProfile {
    /// Both output dims are multiples of this. For Qwen3.6 it is
    /// `patch_size(16) × spatial_merge_size(2) = 32`, so the post-merge
    /// LM grid is exactly `(h/factor, w/factor)`.
    pub factor: u32,
    /// Lower pixel bound — tiny images are upscaled to at least this.
    pub min_pixels: u32,
    /// Upper pixel bound — large images are downscaled to at most this.
    /// Caps per-image LM tokens (`max_pixels / factor²`) and the
    /// O(patches²) ViT attention cost.
    pub max_pixels: u32,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

impl PreprocessProfile {
    /// Profile for Qwen3.6. Native-aspect `smart_resize` (factor 32),
    /// normalise to `[-1, 1]` via mean=std=0.5. Pixel budget defaults:
    /// `min = 256² = 65536` (→ 8×8 = 64 LM tokens) and
    /// `max = 1024² = 1048576` (→ 32×32 = 1024 LM tokens) — generous for
    /// documents/OCR, bounded for serving on 2×RTX5090. (Operator
    /// override lands with the `[harness.candle.vision]` config in #14 C5.)
    pub fn qwen3_6() -> Self {
        Self {
            factor: 32,
            min_pixels: 65_536,
            max_pixels: 1_048_576,
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5],
        }
    }

    /// The factor-aligned `(h, w)` this profile would resize a source
    /// `src_h × src_w` image to. Pure integer policy — no pixel work.
    pub fn resized_dims(&self, src_h: u32, src_w: u32) -> Result<(u32, u32)> {
        smart_resize(src_h, src_w, self.factor, self.min_pixels, self.max_pixels)
    }
}

/// Qwen `smart_resize`: the smallest `factor`-aligned `(h_bar, w_bar)`
/// that preserves aspect ratio as closely as possible while keeping the
/// pixel count within `[min_pixels, max_pixels]`. Direct port of the
/// canonical Qwen2-VL / Qwen3-VL image-processor function (so neuron's
/// grid matches what the model was trained on).
///
/// Returns `(height, width)`. Errors if the aspect ratio exceeds 200:1
/// (degenerate input — a 1-pixel-tall strip), matching upstream.
pub fn smart_resize(
    height: u32,
    width: u32,
    factor: u32,
    min_pixels: u32,
    max_pixels: u32,
) -> Result<(u32, u32)> {
    let h = height.max(1) as f64;
    let w = width.max(1) as f64;
    let ratio = h.max(w) / h.min(w);
    if ratio > 200.0 {
        anyhow::bail!(
            "image aspect ratio {ratio:.1}:1 exceeds the 200:1 limit ({height}×{width}); \
             refusing to resize"
        );
    }
    let f = factor as f64;
    let (minp, maxp) = (min_pixels as f64, max_pixels as f64);
    // round-to-nearest-factor (may be 0 for sub-factor inputs; the
    // min-pixels branch below grows it back up).
    let mut h_bar = (h / f).round() * f;
    let mut w_bar = (w / f).round() * f;
    if h_bar * w_bar > maxp {
        let beta = (h * w / maxp).sqrt();
        h_bar = f.max((h / beta / f).floor() * f);
        w_bar = f.max((w / beta / f).floor() * f);
    } else if h_bar * w_bar < minp {
        let beta = (minp / (h * w)).sqrt();
        h_bar = (h * beta / f).ceil() * f;
        w_bar = (w * beta / f).ceil() * f;
    }
    Ok((h_bar as u32, w_bar as u32))
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
pub fn preprocess(img: &DynamicImage, profile: &PreprocessProfile) -> Result<(Vec<f32>, u32, u32)> {
    let (h_bar, w_bar) = profile.resized_dims(img.height(), img.width())?;
    let rgb = img
        .resize_exact(w_bar, h_bar, FilterType::Triangle)
        .to_rgb8();
    let h = h_bar as usize;
    let w = w_bar as usize;
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
    Ok((out, h_bar, w_bar))
}

/// Combined helper: decode + preprocess in one call. Returns the
/// `(3, h, w)` row-major pixels plus the resized `(h, w)` — the caller
/// needs the dims to build the tensor and to derive the LM token grid
/// `(h/factor, w/factor)`. Most call sites use this; the two-step path
/// exists for callers (tests, future video preprocessing) that need the
/// intermediate `DynamicImage`.
pub fn preprocess_data_uri(uri: &str, profile: &PreprocessProfile) -> Result<(Vec<f32>, u32, u32)> {
    let img = decode_data_uri(uri)?;
    preprocess(&img, profile)
}

/// Resized `(h, w)` for a data-URI image **without** running the pixel
/// normalisation — decode header + `smart_resize` only. Lets a caller
/// that just needs the LM token count (e.g. the TP leader expanding the
/// prompt) avoid materialising the full pixel tensor twice.
pub fn resized_dims_for_uri(uri: &str, profile: &PreprocessProfile) -> Result<(u32, u32)> {
    let img = decode_data_uri(uri)?;
    profile.resized_dims(img.height(), img.width())
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
        let (out, h_bar, w_bar) = preprocess(&dyn_img, &profile).expect("preprocess");

        let h = h_bar as usize;
        let w = w_bar as usize;
        assert_eq!(out.len(), 3 * h * w);
        // Dims are factor-aligned and at least the min-pixel floor.
        assert_eq!(h_bar % profile.factor, 0);
        assert_eq!(w_bar % profile.factor, 0);
        assert!(h * w >= profile.min_pixels as usize);
        // After mean=0.5, std=0.5: red channel (255/255=1.0) → (1.0 - 0.5)/0.5 = 1.0
        // green/blue (0.0) → (0.0 - 0.5)/0.5 = -1.0
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
        let (out, h, w) = preprocess_data_uri(&red_png_uri(), &profile).expect("e2e preprocess");
        assert_eq!(out.len(), 3 * h as usize * w as usize);
        assert!(out.iter().all(|v| v.is_finite()));
        // resized_dims_for_uri agrees with the full preprocess.
        let (h2, w2) = resized_dims_for_uri(&red_png_uri(), &profile).expect("dims");
        assert_eq!((h, w), (h2, w2));
    }

    #[test]
    fn preprocess_grayscale_image_promotes_to_rgb() {
        let profile = PreprocessProfile::qwen3_6();
        // 1x1 grayscale = 200 → after conversion to RGB, all three
        // channels equal 200, normalised → (200/255 - 0.5)/0.5 ≈ 0.569
        let gray = DynamicImage::ImageLuma8(ImageBuffer::from_pixel(1, 1, image::Luma([200])));
        let (out, h_bar, w_bar) = preprocess(&gray, &profile).expect("preprocess");
        let expected = ((200.0 / 255.0) - 0.5) / 0.5;
        let h = h_bar as usize;
        let w = w_bar as usize;
        for c in 0..3 {
            let v = out[c * h * w];
            assert!(
                (v - expected).abs() < 1e-3,
                "channel {c}: expected {expected}, got {v}"
            );
        }
    }

    #[test]
    fn smart_resize_keeps_factor_aligned_square_in_budget() {
        // 448×448 sits inside [65536, 1048576] and is factor-aligned →
        // unchanged. (Regression guard for the old fixed-res sweet spot.)
        let (h, w) = smart_resize(448, 448, 32, 65_536, 1_048_576).unwrap();
        assert_eq!((h, w), (448, 448));
    }

    #[test]
    fn smart_resize_preserves_aspect_and_caps_at_max() {
        // 3000×4000 (landscape) → downscaled under max_pixels, aspect kept.
        let (h, w) = smart_resize(3000, 4000, 32, 65_536, 1_048_576).unwrap();
        assert_eq!(h % 32, 0);
        assert_eq!(w % 32, 0);
        assert!(
            (h as u64) * (w as u64) <= 1_048_576,
            "must respect max_pixels"
        );
        assert!(w > h, "landscape orientation preserved");
        // aspect ≈ 4000/3000 = 1.333; allow a factor-rounding tolerance.
        let ar = w as f64 / h as f64;
        assert!((ar - 4.0 / 3.0).abs() < 0.15, "aspect ~4:3, got {ar:.3}");
    }

    #[test]
    fn smart_resize_floors_tiny_image_at_min() {
        // 16×16 → upscaled to at least min_pixels, factor-aligned.
        let (h, w) = smart_resize(16, 16, 32, 65_536, 1_048_576).unwrap();
        assert_eq!(h % 32, 0);
        assert_eq!(w % 32, 0);
        assert!((h as u64) * (w as u64) >= 65_536, "must respect min_pixels");
    }

    #[test]
    fn smart_resize_tall_nonsquare_stays_nonsquare() {
        // A tall screenshot keeps portrait orientation.
        let (h, w) = smart_resize(2000, 500, 32, 65_536, 1_048_576).unwrap();
        assert!(h > w, "portrait orientation preserved");
        assert_eq!(h % 32, 0);
        assert_eq!(w % 32, 0);
    }

    #[test]
    fn smart_resize_rejects_extreme_aspect() {
        let err = smart_resize(1, 500, 32, 65_536, 1_048_576).unwrap_err();
        assert!(format!("{err:#}").contains("200:1"));
    }
}
