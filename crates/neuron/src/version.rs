//! The daemon's own build identity, captured at compile time by
//! `build.rs` and served from `GET /version`.
//!
//! The `env!()` reads below resolve to the `cargo:rustc-env=` values
//! emitted by `build.rs::emit_build_metadata`. When neuron is built
//! from a source tarball with no git metadata and no injected
//! `HELEXA_BUILD_SHA`, `HELEXA_GIT_SHA` is the literal `"unknown"`.

use cortex_core::build_info::BuildInfo;

/// Assemble the compiled-in build metadata into a [`BuildInfo`].
pub fn build_info() -> BuildInfo {
    BuildInfo {
        package_version: env!("CARGO_PKG_VERSION").to_string(),
        git_sha: env!("HELEXA_GIT_SHA").to_string(),
        git_sha_long: non_empty(env!("HELEXA_GIT_SHA_LONG")),
        git_dirty: env!("HELEXA_GIT_DIRTY") == "true",
        build_timestamp: non_empty(env!("HELEXA_BUILD_TIMESTAMP")),
        rustc_version: non_empty(env!("HELEXA_RUSTC_VERSION")),
        profile: non_empty(env!("HELEXA_BUILD_PROFILE")),
        target: non_empty(env!("HELEXA_TARGET")),
        features: split_features(env!("HELEXA_FEATURES")),
        candle_version: non_empty(env!("HELEXA_CANDLE_VERSION")),
    }
}

/// A one-line version string for clap's `--version` long form, as a
/// `&'static str` (clap requires `'static`). Computed once.
pub fn long_version_static() -> &'static str {
    static V: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    V.get_or_init(long_version).as_str()
}

/// A one-line version string for clap's `--version` long form.
pub fn long_version() -> String {
    let info = build_info();
    let dirty = if info.git_dirty { "-dirty" } else { "" };
    let features = if info.features.is_empty() {
        String::new()
    } else {
        format!(" [{}]", info.features.join(","))
    };
    format!(
        "{} ({}{}){}",
        info.package_version, info.git_sha, dirty, features
    )
}

fn non_empty(s: &str) -> Option<String> {
    if s.is_empty() {
        None
    } else {
        Some(s.to_string())
    }
}

fn split_features(s: &str) -> Vec<String> {
    s.split(',')
        .map(str::trim)
        .filter(|f| !f.is_empty())
        .map(str::to_string)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_info_is_populated() {
        let info = build_info();
        // Always present regardless of git availability.
        assert_eq!(info.package_version, env!("CARGO_PKG_VERSION"));
        assert!(!info.git_sha.is_empty());
    }

    #[test]
    fn long_version_includes_sha() {
        let v = long_version();
        assert!(v.contains(env!("CARGO_PKG_VERSION")));
        assert!(v.contains(env!("HELEXA_GIT_SHA")));
    }
}
