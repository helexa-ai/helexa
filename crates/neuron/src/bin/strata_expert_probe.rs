//! Checks the vendored Strata expert kernels against real GGUF weights.
//!
//! ```sh
//! neuron-strata-expert-probe SHARD GU_TYPE DOWN_TYPE GATE_OFF UP_OFF DOWN_OFF [EXPERT] [TOKENS]
//! ```
//!
//! Offsets are the absolute file offsets of the layer's fused tensors.

use std::ffi::CString;

use neuron::cuda::strata::neuron_strata_expert_probe;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 7 {
        eprintln!(
            "usage: {} SHARD GU DOWN GATE_OFF UP_OFF DOWN_OFF [EXPERT] [TOKENS]",
            args[0]
        );
        std::process::exit(64);
    }
    let int = |i: usize, default: u64| -> u64 {
        args.get(i)
            .map(|v| v.parse().expect("integer argument"))
            .unwrap_or(default)
    };
    let path = CString::new(args[1].as_str()).expect("path");
    let (mut error, mut ms) = (f64::NAN, f64::NAN);
    let (expert, tokens) = (int(7, 0) as i32, int(8, 8) as i32);
    // SAFETY: the path is NUL-terminated and both out-pointers are live
    // for the call; the probe owns every device allocation it makes.
    let rc = unsafe {
        neuron_strata_expert_probe(
            path.as_ptr(),
            int(2, 0) as i32,
            int(3, 0) as i32,
            int(4, 0),
            int(5, 0),
            int(6, 0),
            expert,
            tokens,
            &mut error,
            &mut ms,
        )
    };
    if rc != 0 {
        eprintln!("probe failed: code {rc}");
        std::process::exit(rc);
    }
    println!(
        "gu={} down={} expert={expert} tokens={tokens} relative_error={error:.5} kernel_ms={ms:.3}",
        args[2], args[3]
    );
}
