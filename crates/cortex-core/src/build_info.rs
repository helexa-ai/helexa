//! Build/version metadata shared between cortex and neuron.
//!
//! neuron captures these facts at compile time in its `build.rs`
//! (git SHA, enabled cargo features, rustc/candle versions, …) and
//! serves them from `GET /version`. cortex and `helexa-bench`
//! deserialize the same struct so a benchmark run can be attributed to
//! the exact daemon build that produced it — not just the host's CUDA
//! and driver versions that `/discovery` already reports.
//!
//! Every field beyond the always-present package version is
//! `#[serde(default)]` so a newer reader stays compatible with an
//! older neuron that omits a field (and vice versa) — the same
//! forward/backward-compat discipline as
//! [`crate::discovery::ActivationStatus`].

use serde::{Deserialize, Serialize};

/// Build-time identity of a neuron daemon.
///
/// Returned by `GET /version`. The `git_sha` is the canonical "which
/// build is live" key — benchmark records are bucketed by it, so a
/// regression can be pinned to a daemon change rather than a host
/// change. When neuron is built from a source tarball with no git
/// metadata available (and no `HELEXA_BUILD_SHA` injected by CI/RPM),
/// `git_sha` is the string `"unknown"`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BuildInfo {
    /// Crate version from `CARGO_PKG_VERSION` (e.g. `"0.1.16"`).
    pub package_version: String,
    /// Short git SHA, or `"unknown"` when unavailable at build time.
    #[serde(default = "unknown")]
    pub git_sha: String,
    /// Full 40-char git SHA when available.
    #[serde(default)]
    pub git_sha_long: Option<String>,
    /// Whether the working tree had uncommitted changes at build time.
    /// `false` when the SHA is unknown (tarball build).
    #[serde(default)]
    pub git_dirty: bool,
    /// RFC3339 build timestamp.
    #[serde(default)]
    pub build_timestamp: Option<String>,
    /// `rustc --version` output of the compiler used.
    #[serde(default)]
    pub rustc_version: Option<String>,
    /// Cargo build profile: `"release"` or `"debug"`.
    #[serde(default)]
    pub profile: Option<String>,
    /// Target triple the binary was compiled for.
    #[serde(default)]
    pub target: Option<String>,
    /// Enabled cargo features (e.g. `["cuda", "cudnn"]`). These define
    /// the performance envelope, so they are recorded against every
    /// benchmark run.
    #[serde(default)]
    pub features: Vec<String>,
    /// Locked `candle-core` version, best-effort from `Cargo.lock`.
    #[serde(default)]
    pub candle_version: Option<String>,
}

fn unknown() -> String {
    "unknown".to_string()
}

impl BuildInfo {
    /// A placeholder used by non-neuron benchmark targets (and tests)
    /// that have no build metadata to report.
    pub fn unknown() -> Self {
        BuildInfo {
            package_version: env!("CARGO_PKG_VERSION").to_string(),
            git_sha: unknown(),
            git_sha_long: None,
            git_dirty: false,
            build_timestamp: None,
            rustc_version: None,
            profile: None,
            target: None,
            features: Vec::new(),
            candle_version: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_full() {
        let info = BuildInfo {
            package_version: "0.1.16".into(),
            git_sha: "30d50d6".into(),
            git_sha_long: Some("30d50d6abc123".into()),
            git_dirty: true,
            build_timestamp: Some("2026-06-13T10:00:00+00:00".into()),
            rustc_version: Some("rustc 1.85.0".into()),
            profile: Some("release".into()),
            target: Some("x86_64-unknown-linux-gnu".into()),
            features: vec!["cuda".into(), "cudnn".into()],
            candle_version: Some("0.10.2".into()),
        };
        let json = serde_json::to_string(&info).unwrap();
        let back: BuildInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(info, back);
    }

    #[test]
    fn deserializes_minimal_payload() {
        // An older neuron might send only the package version; every
        // other field must default rather than fail.
        let back: BuildInfo = serde_json::from_str(r#"{"package_version":"0.1.0"}"#).unwrap();
        assert_eq!(back.package_version, "0.1.0");
        assert_eq!(back.git_sha, "unknown");
        assert!(!back.git_dirty);
        assert!(back.features.is_empty());
        assert!(back.candle_version.is_none());
    }
}
