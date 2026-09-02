//! Catalogue quant names to ggml dtypes.
//!
//! Shared rather than owned by the TP path: `quant = "q4k"` in
//! `models.toml` means the same thing whichever loader reads it. It sat
//! behind `cfg(feature = "cuda")` while only the TP worker used it,
//! which put it out of reach of the single-GPU path — where, for
//! `qwen4_exp`, in-situ quantisation is the difference between a model
//! that loads and 241.6 GB that does not (#315).

/// Parse a `ModelSpec.quant` string into a `GgmlDType`. Accepts the
/// common ggml format names (case-insensitive). `None` and `Some("")`
/// both map to "no quantization".
///
/// Supported: `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`, `q8_1`,
/// `q2k`/`q2_k`, `q3k`/`q3_k`, `q4k`/`q4_k`, `q5k`/`q5_k`,
/// `q6k`/`q6_k`, `q8k`/`q8_k`, `f16`, `bf16`, `f32`. The underscore
/// is optional and the prefix is case-insensitive.
pub(crate) fn parse_quant_string(
    s: Option<&str>,
) -> anyhow::Result<Option<candle_core::quantized::GgmlDType>> {
    use candle_core::quantized::GgmlDType;
    let s = match s {
        Some(s) if !s.is_empty() => s,
        _ => return Ok(None),
    };
    let normalised = s.to_ascii_lowercase().replace('_', "");
    let dtype = match normalised.as_str() {
        "q40" => GgmlDType::Q4_0,
        "q41" => GgmlDType::Q4_1,
        "q50" => GgmlDType::Q5_0,
        "q51" => GgmlDType::Q5_1,
        "q80" => GgmlDType::Q8_0,
        "q81" => GgmlDType::Q8_1,
        "q2k" => GgmlDType::Q2K,
        "q3k" => GgmlDType::Q3K,
        "q4k" | "q4km" => GgmlDType::Q4K,
        "q5k" | "q5km" => GgmlDType::Q5K,
        "q6k" => GgmlDType::Q6K,
        "q8k" => GgmlDType::Q8K,
        "f16" => GgmlDType::F16,
        "bf16" => GgmlDType::BF16,
        "f32" => GgmlDType::F32,
        other => anyhow::bail!(
            "unknown quant '{other}' (expected one of: q4_0, q4_1, q5_0, q5_1, q8_0, \
             q8_1, q2k, q3k, q4k, q5k, q6k, q8k, f16, bf16, f32)"
        ),
    };
    Ok(Some(dtype))
}
