//! OpenAI-compatible images API envelope (#200).
//!
//! Shared between neuron (which serves `/v1/images/generations`) and
//! cortex (which proxies it). The shape follows OpenAI's images API —
//! `model`/`prompt`/`n`/`size`/`response_format` — with helexa
//! extensions (`seed`, `negative_prompt`, `guidance_scale`,
//! `num_steps`) as sibling fields, mirroring how the chat surface
//! carries `helexa_timing` inside `usage`.
//!
//! v1 constraints, enforced at the neuron: `n` must be 1,
//! `response_format` must be `b64_json` (neuron has no object store to
//! host `url` responses), output is always PNG.

use serde::{Deserialize, Serialize};

/// `POST /v1/images/generations` request body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImagesGenerationRequest {
    pub model: String,
    pub prompt: String,
    /// Number of images. v1 serves exactly 1; larger values are
    /// rejected with a clear error rather than silently truncated.
    #[serde(default = "default_n")]
    pub n: u32,
    /// `"WIDTHxHEIGHT"`, e.g. `"1024x1024"`. Defaults to 1024².
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,
    /// Only `"b64_json"` is served (the default). `"url"` requires an
    /// object store the fleet doesn't run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,
    /// Only `"png"` is served (the default).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_format: Option<String>,

    // ── helexa extensions ──────────────────────────────────────
    /// Fixed RNG seed for reproducible generations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Negative prompt; enables classifier-free guidance, which doubles
    /// per-step cost (and therefore the metered units, #202).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub negative_prompt: Option<String>,
    /// CFG scale, meaningful only with `negative_prompt`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub guidance_scale: Option<f64>,
    /// Denoise steps. Defaults to the model profile's step count
    /// (9 for Z-Image-Turbo).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_steps: Option<usize>,
}

fn default_n() -> u32 {
    1
}

impl ImagesGenerationRequest {
    /// Parse `size` into `(width, height)`; defaults to 1024².
    /// Dimension *validation* (alignment, ceiling) stays server-side —
    /// this only parses the syntax.
    pub fn parse_size(&self) -> Result<(usize, usize), String> {
        let raw = self.size.as_deref().unwrap_or("1024x1024");
        let (w, h) = raw
            .split_once(['x', 'X'])
            .ok_or_else(|| format!("size '{raw}' is not of the form WIDTHxHEIGHT"))?;
        let width: usize = w
            .trim()
            .parse()
            .map_err(|_| format!("size '{raw}': width is not a number"))?;
        let height: usize = h
            .trim()
            .parse()
            .map_err(|_| format!("size '{raw}': height is not a number"))?;
        Ok((width, height))
    }
}

/// `POST /v1/images/generations` response body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImagesGenerationResponse {
    /// Unix seconds.
    pub created: u64,
    pub data: Vec<ImageData>,
    /// Metering + timing, in the same spirit as chat's
    /// `usage.helexa_timing`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<ImagesUsage>,
}

/// One generated image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageData {
    /// Base64-encoded PNG bytes.
    pub b64_json: String,
}

/// Usage block for the images surface. `helexa_image_units` is the
/// metered work in megapixel-steps (#202): `w × h × steps / 1e6`,
/// doubled under CFG. cortex reads it for budget settlement; clients
/// can read it to anticipate spend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImagesUsage {
    pub helexa_image_units: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub helexa_timing: Option<ImageTiming>,
}

/// Phase timing for one generation, mirrored from the neuron's
/// worker-side measurement.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ImageTiming {
    pub encode_ms: u64,
    pub denoise_ms: u64,
    pub decode_ms: u64,
    pub steps: usize,
    /// True when classifier-free guidance ran (two forwards per step).
    pub cfg: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(size: Option<&str>) -> ImagesGenerationRequest {
        ImagesGenerationRequest {
            model: "m".into(),
            prompt: "p".into(),
            n: 1,
            size: size.map(String::from),
            response_format: None,
            output_format: None,
            seed: None,
            negative_prompt: None,
            guidance_scale: None,
            num_steps: None,
        }
    }

    #[test]
    fn parse_size_default_and_explicit() {
        assert_eq!(req(None).parse_size().unwrap(), (1024, 1024));
        assert_eq!(req(Some("512x768")).parse_size().unwrap(), (512, 768));
        assert_eq!(req(Some("2048X1024")).parse_size().unwrap(), (2048, 1024));
    }

    #[test]
    fn parse_size_rejects_garbage() {
        assert!(req(Some("1024")).parse_size().is_err());
        assert!(req(Some("axb")).parse_size().is_err());
        assert!(req(Some("1024x")).parse_size().is_err());
    }

    #[test]
    fn request_deserializes_with_defaults() {
        let r: ImagesGenerationRequest =
            serde_json::from_str(r#"{"model": "m", "prompt": "a cat"}"#).unwrap();
        assert_eq!(r.n, 1);
        assert!(r.size.is_none());
        assert!(r.seed.is_none());
    }
}
