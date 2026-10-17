//! Music / audio generation surface (#382).
//!
//! OpenAI has no music endpoint, so this is a helexa extension. It follows
//! the images surface where that makes sense (#197) but is **asynchronous**:
//! a song takes seconds to minutes to render, so the public API is a job API.
//!
//! - `POST   /v1/audio/generations`               submit → `202` + [`AudioJob`]
//! - `GET    /v1/audio/generations`               list the caller's jobs
//! - `GET    /v1/audio/generations/{id}`          poll → [`AudioJob`]
//! - `GET    /v1/audio/generations/{id}/content`  the rendered audio bytes
//! - `DELETE /v1/audio/generations/{id}`          cancel
//!
//! cortex owns the job record. A neuron renders exactly one generation per
//! dispatch, synchronously, on `POST /v1/audio/generations`: the response
//! body is the audio (`audio/wav`) and the result metadata travels in the
//! [`RESULT_HEADER`] header as JSON ([`NeuronAudioResult`]). Keeping the
//! bytes out of a JSON envelope matters here — ten minutes of 48 kHz stereo
//! is over 100 MB of PCM.

use serde::{Deserialize, Serialize};

/// Header carrying [`NeuronAudioResult`] JSON on a neuron's audio response.
pub const RESULT_HEADER: &str = "x-helexa-audio-result";

/// Capability string a model advertises when it serves this surface.
pub const CAPABILITY: &str = "audio";

/// `POST /v1/audio/generations` request body. The same body is accepted by
/// cortex (which turns it into a job) and by neuron (which renders it).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AudioGenerationRequest {
    pub model: String,
    /// Style / caption: what the music should sound like.
    pub prompt: String,
    /// Lyrics for models that sing. Absent, empty, or `[Instrumental]`
    /// means an instrumental.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lyrics: Option<String>,
    /// Requested length in seconds. Defaults to the model's advertised
    /// default; validated against its advertised range.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration: Option<f64>,
    /// Fixed RNG seed for reproducible renders. The seed actually used is
    /// always reported back, so a random render can be reproduced too.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Container for the delivered audio. Only `"wav"` today.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,

    // ── model-specific extensions ───────────────────────────────
    /// Diffusion steps (diffusion models only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steps: Option<u32>,
    /// Guidance / CFG scale.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub guidance_scale: Option<f64>,
    /// Song planner: `"off"`, `"auto"` (model default) or `"on"`, for
    /// models that have one (ACE-Step's LM).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub planner: Option<String>,
    /// Sampling temperature (autoregressive models only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Top-k sampling (autoregressive models only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
}

/// Response formats the surface delivers.
pub const SUPPORTED_FORMATS: &[&str] = &["wav"];

impl AudioGenerationRequest {
    /// True when the request asks for an instrumental (no singing).
    pub fn is_instrumental(&self) -> bool {
        match self.lyrics.as_deref().map(str::trim) {
            None | Some("") => true,
            Some(l) => l.eq_ignore_ascii_case("[instrumental]"),
        }
    }

    /// Transport-level validation every hop can do without knowing the
    /// model: required fields, sane numbers, a known format. Model-specific
    /// limits ([`AudioLimits`]) are checked separately.
    pub fn validate_shape(&self) -> Result<(), String> {
        if self.model.trim().is_empty() {
            return Err("'model' must be non-empty".into());
        }
        if self.prompt.trim().is_empty() {
            return Err("'prompt' must be non-empty".into());
        }
        if let Some(d) = self.duration
            && !(d.is_finite() && d > 0.0)
        {
            return Err(format!("'duration' must be a positive number, got {d}"));
        }
        if let Some(f) = self.response_format.as_deref()
            && !SUPPORTED_FORMATS.contains(&f)
        {
            return Err(format!(
                "response_format '{f}' is not supported; use one of {SUPPORTED_FORMATS:?}"
            ));
        }
        if let Some(p) = self.planner.as_deref()
            && !matches!(p, "off" | "auto" | "on")
        {
            return Err(format!("planner must be 'off', 'auto' or 'on', got '{p}'"));
        }
        Ok(())
    }

    /// Validate against a model's advertised limits.
    pub fn validate_limits(&self, limits: &AudioLimits) -> Result<(), String> {
        let d = self.duration.unwrap_or(limits.default_duration);
        if d < limits.min_duration || d > limits.max_duration {
            return Err(format!(
                "duration {d}s is outside this model's range {}..={}s",
                limits.min_duration, limits.max_duration
            ));
        }
        if !limits.vocals && !self.is_instrumental() {
            return Err("this model does not sing; omit 'lyrics' or use [Instrumental]".into());
        }
        if let (Some(s), Some(max)) = (self.steps, limits.max_steps)
            && (s == 0 || s > max)
        {
            return Err(format!("steps must be in 1..={max}, got {s}"));
        }
        Ok(())
    }
}

/// What a music model can do, advertised in `/v1/models` metadata so
/// clients (and the `/music` page) can build their forms from it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AudioLimits {
    pub min_duration: f64,
    pub max_duration: f64,
    pub default_duration: f64,
    /// Whether the model sings (accepts lyrics).
    pub vocals: bool,
    pub sample_rate: u32,
    pub channels: u16,
    /// Upper bound for `steps`, when the model takes one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_steps: Option<u32>,
}

/// Lifecycle of a job.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Queued,
    Running,
    Succeeded,
    Failed,
    Cancelled,
}

impl JobStatus {
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Succeeded | Self::Failed | Self::Cancelled)
    }
}

/// The public job object returned by submit, poll, list and cancel.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AudioJob {
    pub id: String,
    /// Always `"audio.generation"`.
    pub object: String,
    pub status: JobStatus,
    pub model: String,
    /// Unix seconds.
    pub created_at: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub started_at: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<u64>,
    /// Requested duration, echoed so a client can show progress context.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration: Option<f64>,
    /// Present when `status == succeeded`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<AudioResult>,
    /// Present when `status == succeeded`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<AudioUsage>,
    /// Present when `status == failed`: the #63 envelope's inner `error`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<serde_json::Value>,
}

pub const JOB_OBJECT: &str = "audio.generation";

/// The rendered audio's properties.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AudioResult {
    /// Actual length in seconds.
    pub duration: f64,
    pub sample_rate: u32,
    pub channels: u16,
    /// The seed the render used (random when the request gave none).
    pub seed: u64,
    /// Container of the stored audio (`"wav"`).
    pub format: String,
    /// Size of the stored audio in bytes.
    pub bytes: u64,
}

/// Metering block. `helexa_audio_units` is audio-seconds generated (#386).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AudioUsage {
    pub helexa_audio_units: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub helexa_timing: Option<AudioTiming>,
}

/// Phase timing for one render, measured on the neuron.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub struct AudioTiming {
    /// Text / lyric encoding.
    pub encode_ms: u64,
    /// Song planner (0 when it did not run).
    pub plan_ms: u64,
    /// Denoising, or autoregressive token generation.
    pub generate_ms: u64,
    /// Latent / codec decode to waveform.
    pub decode_ms: u64,
}

/// What a neuron reports, in [`RESULT_HEADER`], beside the audio bytes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NeuronAudioResult {
    pub duration: f64,
    pub sample_rate: u32,
    pub channels: u16,
    pub seed: u64,
    pub usage: AudioUsage,
}

/// `GET /v1/audio/generations` response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioJobList {
    /// Always `"list"`.
    pub object: String,
    pub data: Vec<AudioJob>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req() -> AudioGenerationRequest {
        serde_json::from_str(r#"{"model": "m", "prompt": "lofi"}"#).unwrap()
    }

    fn limits() -> AudioLimits {
        AudioLimits {
            min_duration: 10.0,
            max_duration: 600.0,
            default_duration: 30.0,
            vocals: true,
            sample_rate: 48_000,
            channels: 2,
            max_steps: Some(20),
        }
    }

    #[test]
    fn minimal_request_deserializes_and_validates() {
        let r = req();
        assert!(r.validate_shape().is_ok());
        assert!(r.validate_limits(&limits()).is_ok());
        assert!(r.is_instrumental());
    }

    #[test]
    fn instrumental_detection() {
        let mut r = req();
        r.lyrics = Some("  [Instrumental] ".into());
        assert!(r.is_instrumental());
        r.lyrics = Some("[verse]\nla la".into());
        assert!(!r.is_instrumental());
    }

    #[test]
    fn shape_rejects_bad_values() {
        let mut r = req();
        r.prompt = " ".into();
        assert!(r.validate_shape().is_err());
        let mut r = req();
        r.duration = Some(f64::NAN);
        assert!(r.validate_shape().is_err());
        let mut r = req();
        r.duration = Some(-1.0);
        assert!(r.validate_shape().is_err());
        let mut r = req();
        r.response_format = Some("ogg".into());
        assert!(r.validate_shape().is_err());
        let mut r = req();
        r.planner = Some("maybe".into());
        assert!(r.validate_shape().is_err());
    }

    #[test]
    fn limits_reject_out_of_range() {
        let l = limits();
        let mut r = req();
        r.duration = Some(5.0);
        assert!(r.validate_limits(&l).is_err());
        r.duration = Some(601.0);
        assert!(r.validate_limits(&l).is_err());
        r.duration = Some(600.0);
        assert!(r.validate_limits(&l).is_ok());
        r.steps = Some(21);
        assert!(r.validate_limits(&l).is_err());
        r.steps = Some(0);
        assert!(r.validate_limits(&l).is_err());
    }

    #[test]
    fn lyrics_refused_by_non_singing_model() {
        let mut l = limits();
        l.vocals = false;
        let mut r = req();
        r.lyrics = Some("sing this".into());
        assert!(r.validate_limits(&l).is_err());
        r.lyrics = Some("[Instrumental]".into());
        assert!(r.validate_limits(&l).is_ok());
    }

    #[test]
    fn job_status_serializes_snake_case() {
        assert_eq!(
            serde_json::to_string(&JobStatus::Succeeded).unwrap(),
            "\"succeeded\""
        );
        assert!(JobStatus::Cancelled.is_terminal());
        assert!(!JobStatus::Running.is_terminal());
    }
}
