//! Build script: capture build/version metadata for `GET /version`,
//! and (under the `cuda` feature) compile the CUDA kernels in
//! `src/cuda/*.cu` into a static library and link it.
//!
//! The CUDA portion is patterned on
//! `EricLBuehler/mistral.rs::mistralrs-core/build.rs` — same
//! `cudaforge::KernelBuilder` invocation, same NVCC flag set.

use std::process::Command;

fn main() {
    emit_build_metadata();

    #[cfg(feature = "cuda")]
    {
        use std::path::PathBuf;
        println!("cargo:rerun-if-changed=build.rs");
        println!("cargo:rerun-if-changed=src/cuda/");

        let build_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

        let mut builder = cudaforge::KernelBuilder::new()
            .source_glob("src/cuda/*.cu")
            .out_dir(&build_dir)
            .arg("-std=c++17")
            .arg("-O3")
            .arg("-U__CUDA_NO_HALF_OPERATORS__")
            .arg("-U__CUDA_NO_HALF_CONVERSIONS__")
            .arg("-U__CUDA_NO_HALF2_OPERATORS__")
            .arg("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
            .arg("--expt-relaxed-constexpr")
            .arg("--expt-extended-lambda")
            .arg("--use_fast_math")
            .arg("--compiler-options")
            .arg("-fPIC");

        // sm_<80 doesn't have bf16 intrinsics for WMMA — gate the
        // bf16-only kernels off in that case. (Mirrors upstream.)
        if let Some(compute_cap) = builder.get_compute_cap()
            && compute_cap < 80
        {
            builder = builder.arg("-DNO_BF16_KERNEL");
        }

        let target = std::env::var("TARGET").unwrap();
        let out_file = if target.contains("msvc") {
            build_dir.join("neuroncuda.lib")
        } else {
            build_dir.join("libneuroncuda.a")
        };

        builder
            .build_lib(out_file)
            .expect("neuron cuda build failed");
        println!("cargo:rustc-link-search={}", build_dir.display());
        println!("cargo:rustc-link-lib=neuroncuda");
        println!("cargo:rustc-link-lib=dylib=cudart");

        if target.contains("msvc") {
            // No extra runtime library needed.
        } else if target.contains("apple")
            || target.contains("freebsd")
            || target.contains("openbsd")
        {
            println!("cargo:rustc-link-lib=dylib=c++");
        } else if target.contains("android") {
            println!("cargo:rustc-link-lib=dylib=c++_shared");
        } else {
            println!("cargo:rustc-link-lib=dylib=stdc++");
        }
    }
}

/// Emit `cargo:rustc-env=` vars consumed by `env!()` in `src/version.rs`
/// so the daemon can report its own build identity from `GET /version`.
///
/// We re-run only when HEAD moves or the SHA override changes — not on
/// every compile — so the captured timestamp is stable for a given
/// build input rather than churning on each `cargo build`.
fn emit_build_metadata() {
    println!("cargo:rerun-if-env-changed=HELEXA_BUILD_SHA");
    println!("cargo:rerun-if-changed=.git/HEAD");
    // A detached/normal HEAD points at a ref whose file is what actually
    // changes on commit; watch the packed-refs fallback too.
    println!("cargo:rerun-if-changed=.git/packed-refs");

    // SHA: prefer the CI/RPM-injected override (tarball builds have no
    // .git), then fall back to git, then to "unknown".
    let (sha_short, sha_long, dirty) = match std::env::var("HELEXA_BUILD_SHA") {
        Ok(s) if !s.trim().is_empty() => {
            let s = s.trim().to_string();
            let short = s.chars().take(7).collect::<String>();
            (short, Some(s), false)
        }
        _ => {
            let long = git(&["rev-parse", "HEAD"]);
            let short = git(&["rev-parse", "--short", "HEAD"]);
            let dirty = git(&["status", "--porcelain"])
                .map(|s| !s.trim().is_empty())
                .unwrap_or(false);
            match short {
                Some(short) => (short, long, dirty),
                None => ("unknown".to_string(), None, false),
            }
        }
    };
    println!("cargo:rustc-env=HELEXA_GIT_SHA={sha_short}");
    println!(
        "cargo:rustc-env=HELEXA_GIT_SHA_LONG={}",
        sha_long.unwrap_or_default()
    );
    println!("cargo:rustc-env=HELEXA_GIT_DIRTY={dirty}");

    // RFC3339 build timestamp. `date` is universally present on the
    // Linux hosts neuron targets; empty if it ever isn't.
    let ts = Command::new("date")
        .args(["-u", "+%Y-%m-%dT%H:%M:%SZ"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    println!("cargo:rustc-env=HELEXA_BUILD_TIMESTAMP={ts}");

    // Compiler version: cargo sets $RUSTC to the rustc it invokes.
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let rustc_version = Command::new(rustc)
        .arg("--version")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    println!("cargo:rustc-env=HELEXA_RUSTC_VERSION={rustc_version}");

    println!(
        "cargo:rustc-env=HELEXA_BUILD_PROFILE={}",
        std::env::var("PROFILE").unwrap_or_default()
    );
    println!(
        "cargo:rustc-env=HELEXA_TARGET={}",
        std::env::var("TARGET").unwrap_or_default()
    );

    // Enabled features: cargo exports CARGO_FEATURE_<NAME> for each.
    // Reverse the mangling (uppercase, '-'→'_') best-effort for display.
    let mut features: Vec<String> = std::env::vars()
        .filter_map(|(k, _)| k.strip_prefix("CARGO_FEATURE_").map(|f| f.to_string()))
        .map(|f| f.to_lowercase().replace('_', "-"))
        // `default` is the meta-feature, not a perf-relevant flag.
        .filter(|f| f != "default")
        .collect();
    features.sort();
    println!("cargo:rustc-env=HELEXA_FEATURES={}", features.join(","));

    println!(
        "cargo:rustc-env=HELEXA_CANDLE_VERSION={}",
        candle_version().unwrap_or_default()
    );
}

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() { None } else { Some(s) }
}

/// Best-effort: read the locked `candle-core` version from the workspace
/// `Cargo.lock` (two levels up from this crate). Returns `None` if the
/// lockfile is absent (e.g. some packaging flows) or the entry isn't
/// found.
fn candle_version() -> Option<String> {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").ok()?;
    let lock = std::path::Path::new(&manifest)
        .join("..")
        .join("..")
        .join("Cargo.lock");
    println!("cargo:rerun-if-changed={}", lock.display());
    let text = std::fs::read_to_string(lock).ok()?;
    // Cargo.lock entries are `[[package]]\nname = "x"\nversion = "y"`.
    let mut in_candle = false;
    for line in text.lines() {
        let line = line.trim();
        if line == "[[package]]" {
            in_candle = false;
        } else if line == "name = \"candle-core\"" {
            in_candle = true;
        } else if in_candle && let Some(rest) = line.strip_prefix("version = \"") {
            return Some(rest.trim_end_matches('"').to_string());
        }
    }
    None
}
