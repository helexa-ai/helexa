#!/bin/env bash
#
# Pre-seed the CUTLASS checkout that cudaforge would otherwise clone from
# GitHub during every CUDA build.
#
# candle-flash-attn's build script calls `cudaforge`, which fetches CUTLASS
# into `${CUDAFORGE_HOME:-$HOME/.cudaforge}/git/checkouts/cutlass-<commit[..16]>`.
# If the cache starts empty — as it does on a fresh or single-use build
# machine — every CUDA job clones CUTLASS afresh, and a clone GitHub
# refuses fails the whole job late, after the 20 minutes of kernel
# compilation that preceded it:
#
#   fatal: could not read Username for 'https://github.com': No such device
#   Error: GitOperationFailed("git clone failed with status: exit status: 128")
#
# That message reads like a missing credential. It is the opposite: see
# the note on anonymity below.
#
# cudaforge short-circuits when the cache directory already holds `include/`
# at the right commit (dependency.rs, `fetch_with_lock`), so seeding it here
# — with retries and backoff, which the build script has none of — removes
# the fetch from the build's critical path.
#
# The fetch is forced ANONYMOUS, and reports diagnostics when it fails.
# CUTLASS is public, so no credential is needed or wanted; sending none
# takes our side of the connection out of the picture, so a failure is
# unambiguously not something this build machine attached to the request.
# What has actually been observed is a 401 that nobody has yet traced to
# a responder:
#
#   error: RPC failed; HTTP 401 curl 22 The requested URL returned error: 401
#
# Do not assume that is GitHub. GitHub rate-limits its API, not clones,
# and a filtering proxy or a captive resolver on the path answers 401
# just as readily. The diagnostics below print who replied and with what
# headers, which is the thing that tells them apart.
#
# Best effort by design: if seeding fails, the build behaves exactly as it
# does today and cargo emits the authoritative error. This script never
# fails a build that would otherwise have passed.
set -uo pipefail

# Read no git configuration at all for the duration of this script.
# Clearing the individual keys instead (an empty credential.helper and an
# empty http.<url>.extraheader) does neutralise an injected auth header,
# but it leaves url.<...>.insteadOf rewrites and any config injected
# through GIT_CONFIG_COUNT untouched — both of which can also redirect or
# authenticate the fetch. Discarding the config files closes every vector
# at once and needs no guess about which one a given machine carries;
# this script has no configuration of its own to lose.
# GIT_CONFIG_GLOBAL / GIT_CONFIG_SYSTEM require git >= 2.32.
export GIT_CONFIG_GLOBAL=/dev/null
export GIT_CONFIG_SYSTEM=/dev/null
export GIT_CONFIG_COUNT=0

# Never prompt: without this a 401 blocks on a username read instead of
# failing, which turns a refused fetch into a confusing hang.
export GIT_TERMINAL_PROMPT=0
export GIT_ASKPASS=/bin/true

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

# Dumped once, on the first failed attempt. Everything here is best
# effort and must never fail the script: this runs on a path that is
# already going wrong. `env -u` restores the real git configuration for
# the inspection only — the fetch itself still runs without it.
diagnose() {
  echo "seed-cutlass: --- diagnostics ---" >&2
  # Deliberately no public-IP lookup: that would mean calling a third
  # party from every failing build. The reply headers below already say
  # who answered, which is the question that matters.
  echo "seed-cutlass: github.com resolves to $(getent ahostsv4 github.com 2>/dev/null | awk '{print $1}' | sort -u | tr '\n' ' ')" >&2
  proxies=$(env | grep -Ei '^(https?_proxy|all_proxy|no_proxy)=' | tr '\n' ' ')
  echo "seed-cutlass: proxy env ${proxies:-none}" >&2
  relevant=$(env -u GIT_CONFIG_GLOBAL -u GIT_CONFIG_SYSTEM -u GIT_CONFIG_COUNT \
    git config --list --show-origin 2>/dev/null \
    | grep -Ei 'extraheader|insteadof|credential|proxy|askpass' | tr '\n' ' ')
  echo "seed-cutlass: git config of interest ${relevant:-none}" >&2
  # The decisive one: a real GitHub reply carries x-github-request-id and
  # a GitHub server header. A middlebox answering 401 carries neither.
  echo "seed-cutlass: response headers from the git endpoint:" >&2
  curl -sS -o /dev/null -D - --max-time 20 \
    "${CUTLASS_REPO%.git}.git/info/refs?service=git-upload-pack" 2>&1 \
    | sed 's/^/seed-cutlass:   /' >&2
  echo "seed-cutlass: --- end diagnostics ---" >&2
}

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
  if [[ "${attempt}" == 1 ]]; then
    diagnose
  fi
  backoff=$((attempt * 15))
  echo "seed-cutlass: attempt ${attempt} failed; retrying in ${backoff}s" >&2
  sleep "${backoff}"
done

echo "seed-cutlass: could not seed after 5 attempts; leaving it to the build" >&2
exit 0
