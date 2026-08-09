//! Capability discovery for models that are not loaded (#241).
//!
//! A loaded model reports its own modalities — the handle knows whether
//! it built a vision tower or a diffusion pipeline. A model sitting cold
//! in the cache reports nothing, which is how image generation came to be
//! invisible: the image model is *normally* cold, so `/v1/models` showed
//! an entry with no capabilities and no hint that it produces pictures.
//!
//! The alternative to discovery is an operator writing `capabilities =
//! ["image"]` into a catalogue file. That is a fact the runtime already
//! holds, restated by hand in a second place, where it can rot silently —
//! nothing fails when the catalogue is wrong, the model just lies about
//! itself. So derive it instead.
//!
//! Everything here reads the local Hugging Face cache and nothing else:
//! no network, no weight load, no device. The two signals are the same
//! ones the loader itself dispatches on, which is what keeps this honest
//! rather than a parallel guess:
//!
//! - **image** — `model_index.json` at the repo root, the diffusers
//!   layout marker. [`super::preflight::classify`] already returns
//!   [`SourceFormat::Diffusers`] for it, and `load_model` routes on
//!   exactly that to reach the image path.
//! - **vision** — a `vision_config` block in `config.json`, the same
//!   thing `VisionMeta::from_config_path` keys on to decide whether a
//!   loaded model advertises `"vision"`.
//!
//! A repo that was never cached yields `None` rather than a guess. That
//! is a real limitation — a model no node has downloaded stays
//! undiscoverable until one does — but reporting "unknown" is the honest
//! answer, and it is strictly better than the empty list it replaces.

use super::preflight::{self, SourceFormat};
use std::path::Path;

/// Modalities a model would serve, derived from its cached repo.
///
/// `None` when the repo has no usable snapshot in this cache — the model
/// was never downloaded here, so this node cannot say anything about it.
pub fn from_cache(cache_path: &Path, repo_path: &str) -> Option<Vec<String>> {
    let files = preflight::cached_snapshot_files(cache_path, repo_path)?;
    let refs: Vec<&str> = files.iter().map(String::as_str).collect();
    Some(classify_files(&refs, || {
        let dir = preflight::cached_snapshot_dir(cache_path, repo_path)?;
        std::fs::read_to_string(dir.join("config.json")).ok()
    }))
}

/// Classify a repo's file listing into modalities.
///
/// `read_config` is called lazily and only for non-diffusers layouts, so
/// a diffusers repo costs no file read at all. Split from [`from_cache`]
/// so the tests can drive it from fixtures without a cache on disk.
pub fn classify_files(
    filenames: &[&str],
    read_config: impl FnOnce() -> Option<String>,
) -> Vec<String> {
    match preflight::classify(filenames) {
        SourceFormat::Diffusers => vec!["image".to_string()],
        // A repo with no recognised weights (tokenizer-only, or an empty
        // entry) is not servable, so claiming "text" would be a lie of
        // the same kind this module exists to remove.
        SourceFormat::Empty => Vec::new(),
        _ => {
            let mut caps = vec!["text".to_string()];
            if read_config().is_some_and(|text| declares_vision(&text)) {
                caps.push("vision".to_string());
            }
            caps
        }
    }
}

/// Whether a `config.json` describes a model with a vision tower.
fn declares_vision(config_json: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(config_json)
        .ok()
        .is_some_and(|v| v.get("vision_config").is_some())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Z-Image layout: component subdirectories under a
    /// `model_index.json`, with safetensors nested inside them. The
    /// nested shards are why this must be checked before the
    /// safetensors sniff.
    #[test]
    fn diffusers_repo_is_an_image_model() {
        let files = [
            "model_index.json",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors",
            "vae/config.json",
            "text_encoder/model.safetensors",
        ];
        assert_eq!(
            classify_files(&files, || panic!("config must not be read for diffusers")),
            vec!["image".to_string()]
        );
    }

    #[test]
    fn dense_repo_without_vision_config_is_text_only() {
        let files = ["config.json", "tokenizer.json", "model.safetensors"];
        let caps = classify_files(&files, || Some(r#"{"model_type":"qwen3"}"#.to_string()));
        assert_eq!(caps, vec!["text".to_string()]);
    }

    #[test]
    fn dense_repo_with_vision_config_adds_vision() {
        let files = [
            "config.json",
            "tokenizer.json",
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
        ];
        let caps = classify_files(&files, || {
            Some(r#"{"model_type":"qwen3_5","vision_config":{"patch_size":16}}"#.to_string())
        });
        assert_eq!(caps, vec!["text".to_string(), "vision".to_string()]);
    }

    #[test]
    fn gguf_repo_is_text() {
        let files = ["Qwen3-0.6B-Q4_K_M.gguf", "tokenizer.json"];
        assert_eq!(classify_files(&files, || None), vec!["text".to_string()]);
    }

    /// A tokenizer-only repo has nothing to serve. Reporting "text"
    /// would advertise a model that cannot answer.
    #[test]
    fn repo_without_weights_claims_nothing() {
        let files = ["tokenizer.json", "README.md"];
        assert!(classify_files(&files, || None).is_empty());
    }

    /// An unparseable or absent config must not promote to vision.
    #[test]
    fn unreadable_config_does_not_claim_vision() {
        let files = ["config.json", "model.safetensors"];
        assert_eq!(
            classify_files(&files, || Some("{ not json".to_string())),
            vec!["text".to_string()]
        );
        assert_eq!(classify_files(&files, || None), vec!["text".to_string()]);
    }

    #[test]
    fn uncached_repo_is_unknown_not_empty() {
        let dir = std::env::temp_dir().join("helexa-capability-probe-missing");
        assert!(from_cache(&dir, "does-not/exist").is_none());
    }
}
