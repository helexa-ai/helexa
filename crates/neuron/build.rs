//! Build script: compile the CUDA kernels in `src/cuda/*.cu` into a
//! static library and link it under the `cuda` feature.
//!
//! Patterned on `EricLBuehler/mistral.rs::mistralrs-core/build.rs` —
//! same `cudaforge::KernelBuilder` invocation, same NVCC flag set.

fn main() {
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
