# Vendored from Strata

These files are an unmodified subset of [Strata](https://github.com/Niko1221/Strata)
at commit `6a6db7ff58c283cf28d2b913404b5fa31a767eaa`, kept at their upstream paths so
their relative includes work and so they can be diffed against upstream.

- Strata is MIT licensed. See `LICENSE`.
- Its quantised-block layouts and dot products derive from llama.cpp / ggml, also MIT
  licensed. See `engine/third_party/ggml/LICENSE` and `VERSION.txt`.

To update, copy the same paths from a newer upstream commit and change the commit above.
Do not edit these files in place; neuron's adapter is `crates/neuron/src/cuda/strata_adapter/strata_shim.cu`.
