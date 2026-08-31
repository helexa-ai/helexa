#!/bin/env bash
#
# Pre-seed the CUTLASS checkout that cudaforge would otherwise clone from
# GitHub during every CUDA build.
#
# candle-flash-attn's build script calls `cudaforge`, which fetches CUTLASS
# into `${CUDAFORGE_HOME:-$HOME/.cudaforge}/git/checkouts/cutlass-<commit[..16]>`.
# Our runners are one VM per job, so that cache starts empty every time and
# each CUDA job clones CUTLASS afresh. A burst of concurrent jobs is then
# enough to start getting refused:
#
#   fatal: could not read Username for 'https://github.com': No such device
#   Error: GitOperationFailed("git clone failed with status: exit status: 128")
#
# which fails a 20-minute job on someone else's rate limit, with a message
# that looks like a credentials problem rather than a network one.
#
# cudaforge short-circuits when the cache directory already holds `include/`
# at the right commit (dependency.rs, `fetch_with_lock`), so seeding it here
# — with retries and backoff, which the build script has none of — removes
# the fetch from the build's critical path.
#
# Best effort by design: if seeding fails, the build behaves exactly as it
# does today and cargo emits the authoritative error. This script never
# fails a build that would otherwise have passed.
set -uo pipefail

CUTLASS_REPO="${CUTLASS_REPO:-https://github.com/NVIDIA/cutlass.git}"
CACHE_ROOT="${CUDAFORGE_HOME:-$HOME/.cudaforge}/git/checkouts"

# The commit is pinned in candle-flash-attn, not here — read it from the
# source cargo actually resolved, so this cannot drift from the candle pin.
BUILD_RS=$(find "${CARGO_HOME:-$HOME/.cargo}/git/checkouts" \
  -path '*/candle-flash-attn/build.rs' -print -quit 2>/dev/null)
if [[ -z "${BUILD_RS}" ]]; then
  echo "seed-cutlass: candle-flash-attn not in the cargo git checkouts yet;" >&2
  echo "              run 'cargo fetch' before this step. Skipping." >&2
  exit 0
fi

COMMIT=$(sed -nE 's/^const CUTLASS_COMMIT: &str = "([0-9a-f]+)";/\1/p' "${BUILD_RS}")
if [[ ! "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "seed-cutlass: could not read CUTLASS_COMMIT from ${BUILD_RS}. Skipping." >&2
  exit 0
fi

DEST="${CACHE_ROOT}/cutlass-${COMMIT:0:16}"
echo "seed-cutlass: commit ${COMMIT}"
echo "seed-cutlass: target ${DEST}"

# Same short-circuit cudaforge applies, so a warm cache costs nothing.
if [[ -d "${DEST}/include" ]] && [[ "$(git -C "${DEST}" rev-parse HEAD 2>/dev/null)" == "${COMMIT}" ]]; then
  echo "seed-cutlass: already seeded"
  exit 0
fi

mkdir -p "${CACHE_ROOT}" || exit 0
rm -rf "${DEST}.partial"

# Never prompt: without this a 401 blocks on a username read instead of
# failing, which is what turns a refused clone into a confusing hang.
export GIT_TERMINAL_PROMPT=0
export GIT_ASKPASS=/bin/true

for attempt in 1 2 3 4 5; do
  if git init -q "${DEST}.partial" \
    && git -C "${DEST}.partial" remote add origin "${CUTLASS_REPO}" \
    && git -C "${DEST}.partial" config core.sparseCheckout true \
    && printf 'include/\ntools/util/include/\n' \
         > "${DEST}.partial/.git/info/sparse-checkout" \
    && git -C "${DEST}.partial" fetch -q --depth 1 origin "${COMMIT}" \
    && git -C "${DEST}.partial" checkout -q FETCH_HEAD; then
    rm -rf "${DEST}"
    mv "${DEST}.partial" "${DEST}"
    echo "seed-cutlass: seeded at attempt ${attempt}"
    exit 0
  fi
  rm -rf "${DEST}.partial"
  backoff=$((attempt * 15))
  echo "seed-cutlass: attempt ${attempt} failed; retrying in ${backoff}s" >&2
  sleep "${backoff}"
done

echo "seed-cutlass: could not seed after 5 attempts; leaving it to the build" >&2
exit 0
