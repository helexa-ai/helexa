#!/usr/bin/env bash
# Type-check neuron's `cuda` feature locally, in CI's own image.
#
# Why this exists: `#[cfg(feature = "cuda")]` code is invisible to a
# default-feature build, so `cargo build`, `cargo clippy --all-targets`
# and the whole test suite can pass on code that does not compile for
# the target the fleet actually serves on. The only gate that catches it
# is CI's `cuda-check` job — a ~20 minute round trip, and the failure
# arrives after the push rather than before it.
#
# This runs the same command, with the same env, in the same image.
#
# Do NOT try to install the CUDA toolkit on a Fedora 44 host instead:
# CUDA 13.0's nvcc rejects host compilers newer than gcc 15 and F44
# ships gcc 16 with no compat package. The image is Fedora 43 for
# exactly that reason, and additionally patches a glibc 2.41+ /
# CUDA-header `noexcept` mismatch that nvcc hard-errors on. See
# gongfoo's images/runner-cuda-13.0/Containerfile.
#
# No GPU is required — this is a borrow-check, not a run.
#
# First run pulls the image (~GBs) and compiles the candle/flash-attn
# kernels, so it costs about what CI costs. Afterwards the cargo target
# volume makes an incremental check a fraction of that, which is the
# whole point.
set -euo pipefail

IMAGE="${CUDA_CHECK_IMAGE:-git.lair.cafe/gongfoo/runner-cuda-13.0:latest}"
ROOT="$(git rev-parse --show-toplevel)"

# A dedicated target volume. Sharing the host's `target/` would make the
# default-feature and cuda-feature builds evict each other's artifacts on
# every alternation, which would cost more than it saves.
TARGET_VOL="${CUDA_CHECK_TARGET_VOL:-helexa-cuda-check-target}"
# Mounted at the registry rather than over CARGO_HOME: the toolchain
# binaries live in /usr/local/cargo/bin and must stay visible.
REGISTRY_VOL="${CUDA_CHECK_REGISTRY_VOL:-helexa-cuda-check-registry}"

if [ "${1:-}" = "--pull" ]; then
    shift
    podman pull "${IMAGE}"
fi

# Default to the exact command ci.yml's cuda-check job runs. Extra
# arguments replace it, for narrowing to one crate while iterating.
if [ "$#" -gt 0 ]; then
    CARGO_CMD=("$@")
else
    CARGO_CMD=(cargo check -p neuron --features cuda,flash-attn --all-targets)
fi

# Allocate a TTY only when there is one to allocate. `-it` against a
# pipe or a CI log wedges the container before cargo ever starts, which
# looks exactly like a very slow compile.
TTY_ARGS=()
if [ -t 0 ] && [ -t 1 ]; then
    TTY_ARGS=(-it)
fi

exec podman run --rm "${TTY_ARGS[@]}" \
    -v "${ROOT}:/work:Z" \
    -v "${TARGET_VOL}:/target" \
    -v "${REGISTRY_VOL}:/usr/local/cargo/registry" \
    -w /work \
    -e CARGO_TARGET_DIR=/target \
    -e CARGO_TERM_COLOR=always \
    `# candle-kernels' build script falls back to nvidia-smi for` \
    `# compute-cap detection when this is unset, and there is no GPU` \
    `# here. Any valid cap borrow-checks; the real per-flavour caps` \
    `# live in build-prerelease.yml's matrix.` \
    -e CUDA_COMPUTE_CAP=86 \
    `# The image ships sccache, but the local run has no shared cache` \
    `# to talk to; an unset wrapper avoids a hard cargo failure.` \
    -e RUSTC_WRAPPER= \
    "${IMAGE}" \
    bash -lc '
        set -euo pipefail
        # CUDA_HOME/PATH/LD_LIBRARY_PATH come from the image. LIBRARY_PATH
        # does not, and cudarc needs it at link time — ci.yml exports all
        # three explicitly, so keep this in step with that job.
        export LIBRARY_PATH="${CUDA_HOME}/targets/x86_64-linux/lib:${CUDA_HOME}/lib64:${LIBRARY_PATH:-}"
        exec "$@"
    ' _ "${CARGO_CMD[@]}"
