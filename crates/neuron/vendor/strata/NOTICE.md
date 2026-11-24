# Vendored from Strata

These files are an unmodified subset of [Strata](https://github.com/Niko1221/Strata)
at commit `6a6db7ff58c283cf28d2b913404b5fa31a767eaa`, kept at their upstream paths so
their relative includes work and so they can be diffed against upstream.

- Strata is MIT licensed. See `LICENSE`.
- Its quantised-block layouts and dot products derive from llama.cpp / ggml, also MIT
  licensed. See `engine/third_party/ggml/LICENSE` and `VERSION.txt`.

To update, copy the same paths from a newer upstream commit and change the commit above.
Do not edit these files in place; neuron's adapter is `crates/neuron/src/cuda/strata_adapter/strata_shim.cu`.

## Decode kernels for the native engine (#427)

Added from the same Strata revision for neuron's native qwen4_exp decode
engine, unmodified: `native_gdn`, `native_gdn_preprocess`, `fused_gdn`,
`native_gr_norm`, `native_gr_postops`, `gr`, `native_router`, `native_moe`,
`quantize_act`, `bf16_gemv` and `native_bf16` (`engine/src/kernels/cuda/`),
with the headers they include. neuron calls them through
`src/cuda/strata_adapter/engine_shim.cu`.
