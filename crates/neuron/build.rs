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
        #[cfg(feature = "strata-kernels")]
        build_strata_kernels(&build_dir);
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

/// Compile the vendored Strata expert kernels and neuron's C shim.
///
/// They need C++20 and must not get the `--use_fast_math` the kernels
/// above are built with: upstream builds this file without it, and
/// contraction would change the rounding its parity relies on.
#[cfg(feature = "strata-kernels")]
fn build_strata_kernels(build_dir: &std::path::Path) {
    println!("cargo:rerun-if-changed=vendor/strata");
    println!("cargo:rerun-if-changed=src/cuda/strata_adapter/strata_shim.cu");
    println!("cargo:rerun-if-changed=src/cuda/strata_adapter/strata_probe.cu");
    println!("cargo:rerun-if-changed=src/cuda/strata_adapter/engine_shim.cu");
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");
    let cap = std::env::var("CUDA_COMPUTE_CAP").expect("strata-kernels needs CUDA_COMPUTE_CAP");
    let root = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let vendor = root.join("vendor/strata/engine");
    let mut sources = vec![
        // UD-IQ4_XS has no NVFP4 experts, so upstream's stub satisfies
        // the reference from the grouped dispatch.
        (
            "strata_nvfp4_absent.o".to_string(),
            vendor.join("src/kernels/cuda/nvfp4_decode_sm120_absent.cpp"),
        ),
        (
            "strata_shim.o".to_string(),
            root.join("src/cuda/strata_adapter/strata_shim.cu"),
        ),
        (
            "strata_probe.o".to_string(),
            root.join("src/cuda/strata_adapter/strata_probe.cu"),
        ),
        (
            "engine_shim.o".to_string(),
            root.join("src/cuda/strata_adapter/engine_shim.cu"),
        ),
    ];
    // The experts, then the decode kernels the native engine (#427)
    // drives: linear attention, hyper-connections, router, MoE combine,
    // activation quantisation, the BF16 projections, PLE and QSA.
    for kernel in [
        "iq_kernels",
        "native_gdn",
        "native_gdn_preprocess",
        "fused_gdn",
        "native_gr_norm",
        "native_gr_postops",
        "gr",
        "native_router",
        "native_rope",
        "native_moe",
        "quantize_act",
        "bf16_gemv",
        "native_bf16",
        "native_ple_postops",
        "native_qsa_indexer",
        "qsa",
        "qsa_select",
        "qsa_decode_attn",
    ] {
        sources.push((
            format!("strata_{kernel}.o"),
            vendor.join(format!("src/kernels/cuda/{kernel}.cu")),
        ));
    }
    // Independent translation units: compile them side by side.
    let objects: Vec<std::path::PathBuf> = std::thread::scope(|scope| {
        let jobs: Vec<_> = sources
            .iter()
            .map(|(name, source)| {
                let object = build_dir.join(name);
                let (cap, vendor) = (&cap, &vendor);
                scope.spawn(move || {
                    let status = Command::new("nvcc")
                        .args(["-O3", "-DNDEBUG", "-std=c++20"])
                        .arg(format!(
                            "--generate-code=arch=compute_{cap},code=[compute_{cap},sm_{cap}]"
                        ))
                        .args(["-DSTRATA_Q6K_EXPERTS=1", "-DSTRATA_VERIFY_MAX_T=8"])
                        .arg(format!("-I{}", vendor.join("third_party/ggml").display()))
                        .arg(format!("-I{}", vendor.join("include").display()))
                        .args(["--compiler-options", "-fPIC", "-c"])
                        .arg(source)
                        .arg("-o")
                        .arg(&object)
                        .status()
                        .expect("run nvcc");
                    assert!(status.success(), "nvcc failed on {}", source.display());
                    object
                })
            })
            .collect();
        jobs.into_iter()
            .map(|j| j.join().expect("nvcc thread"))
            .collect()
    });
    let library = build_dir.join("libneuronstrata.a");
    let _ = std::fs::remove_file(&library);
    let status = Command::new("ar")
        .arg("rcs")
        .arg(&library)
        .args(&objects)
        .status()
        .expect("run ar");
    assert!(status.success(), "ar failed");
    println!("cargo:rustc-link-lib=static=neuronstrata");
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
