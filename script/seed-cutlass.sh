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
# The fetch is forced ANONYMOUS, and probes the remote on every run.
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
# headers on every build, so a refusal arrives with healthy samples
# beside it rather than alone.
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

# ---------------------------------------------------------------------
# Instrumentation. Two levels, both best effort: none of it may fail a
# build, so every probe is bounded by a timeout and every failure is
# swallowed.
#
#   probe_remote  runs on EVERY seeding run, successful ones included,
#                 so consecutive builds form a comparable series. A
#                 failure observed alone is nearly unreadable; the same
#                 failure beside twenty healthy samples is evidence.
#   diagnose      runs once, on the first failed attempt, adding the
#                 slower checks only worth their time once something has
#                 already gone wrong.
#
# What each field is for:
#
#   server, x-github-request-id  whether GitHub answered at all. A proxy
#                                or captive portal returning 401 on its
#                                behalf carries neither.
#   x-github-edge-region, peer   which edge served us, so a fault
#                                specific to one edge becomes visible.
#   x-ratelimit-*                whether this egress address is being
#                                throttled, measured rather than assumed.
#   response body                distinguishes "Repository not found"
#                                from any other refusal.
#   tls issuer                   a re-signed certificate means something
#                                is terminating TLS in the path.
#   egress                       the identity any per-source limit is
#                                applied to, and the only field that
#                                makes failures on different runners
#                                comparable. Costs one call to a third
#                                party, which is why it is in the
#                                failure path and not the common one.
# ---------------------------------------------------------------------
REFS_URL="${CUTLASS_REPO%.git}.git/info/refs?service=git-upload-pack"

probe_remote() {
  fmt='http=%{http_code} peer=%{remote_ip}:%{remote_port} tls_verify=%{ssl_verify_result}'
  fmt="${fmt} connects=%{num_connects} redirects=%{num_redirects}"
  fmt="${fmt} bytes=%{size_download} time=%{time_total}s"
  # One request, not two: this runs on every build, and doubling the
  # traffic to the endpoint under suspicion would corrupt what it
  # measures. Headers land in a file so the summary and the identity
  # come from the same response.
  hdr=$(mktemp 2>/dev/null) || return 0
  echo "seed-cutlass: probe $(curl -sS -o /dev/null -D "${hdr}" --max-time 20 \
    -w "${fmt}" "${REFS_URL}" 2>&1 | tr '\n' ' ')" >&2
  tr -d '\r' < "${hdr}" \
    | grep -iE '^(HTTP/|server:|www-authenticate:|retry-after:|x-github-request-id:|x-github-edge-region:|x-ratelimit-)' \
    | sed 's/^/seed-cutlass:   /' >&2 || true
  rm -f "${hdr}"
}

diagnose() {
  echo "seed-cutlass: --- diagnostics ---" >&2
  echo "seed-cutlass: $(git --version 2>/dev/null || echo 'git --version failed')" \
       "| $(uname -sr 2>/dev/null) | host $(hostname 2>/dev/null)" >&2

  echo "seed-cutlass: dns $(getent ahosts github.com 2>/dev/null \
    | awk '{print $1}' | sort -u | tr '\n' ' ')" >&2
  echo "seed-cutlass: egress $(curl -sS --max-time 10 https://api.ipify.org 2>/dev/null \
    || echo unknown)" >&2

  proxies=$(env | grep -Ei '^(https?_proxy|all_proxy|no_proxy)=' | tr '\n' ' ')
  echo "seed-cutlass: proxy env ${proxies:-none}" >&2

  # `env -u` restores the real git configuration for this inspection
  # only; the fetch itself still runs with config disabled.
  relevant=$(env -u GIT_CONFIG_GLOBAL -u GIT_CONFIG_SYSTEM -u GIT_CONFIG_COUNT \
    git config --list --show-origin 2>/dev/null \
    | grep -Ei 'extraheader|insteadof|credential|proxy|askpass|http\.' | tr '\n' ' ')
  echo "seed-cutlass: git config of interest ${relevant:-none}" >&2

  echo "seed-cutlass: what this egress has left on the github api:" >&2
  curl -sS -o /dev/null -D - --max-time 15 https://api.github.com/rate_limit 2>&1 \
    | tr -d '\r' | grep -iE '^(HTTP/|x-ratelimit-)' \
    | sed 's/^/seed-cutlass:   /' >&2 || true

  echo "seed-cutlass: tls path:" >&2
  curl -sSv -o /dev/null --max-time 20 "${REFS_URL}" 2>&1 \
    | grep -iE 'subject:|issuer:|SSL connection using|ALPN, server accepted|subjectAltName' \
    | sed 's/^[*] *//; s/^/seed-cutlass:   /' >&2 || true

  echo "seed-cutlass: refusal body: $(curl -sS --max-time 20 "${REFS_URL}" 2>&1 \
    | head -c 200 | tr '\n' ' ')" >&2

  # Replay the fetch under git's own HTTP tracing. Everything above
  # inspects a request of our choosing — a GET of the ref advertisement.
  # The exchange that actually fails is the POST to git-upload-pack that
  # follows it, and anything in the path may treat the two differently,
  # so a clean GET proves very little. This shows which request was
  # refused and what came back.
  #
  # It also settles, without inference, whether anything attached
  # credentials: git logs an Authorization header here as <redacted>, so
  # the line appearing at all means one was sent.
  echo "seed-cutlass: traced replay of the real fetch:" >&2
  rm -rf "${DEST}.trace"
  if git init -q "${DEST}.trace" 2>/dev/null \
    && git -C "${DEST}.trace" remote add origin "${CUTLASS_REPO}" 2>/dev/null; then
    GIT_TRACE_CURL=1 GIT_TRACE_CURL_NO_DATA=1 GIT_TRACE_REDACT=1 \
      git -C "${DEST}.trace" fetch --depth 1 origin "${COMMIT}" 2>&1 \
      | grep -iE 'Send header: (GET|POST)|Recv header: (HTTP/|server:|www-authenticate:|x-github|content-type:|retry-after:)|Authorization|fatal:|error:' \
      | sed 's/.*\(Send header\|Recv header\)/\1/; s/^/seed-cutlass:   /' \
      | head -40 >&2 || true
  fi
  rm -rf "${DEST}.trace"

  echo "seed-cutlass: --- end diagnostics ---" >&2
}

# Unconditional: one sample per run, so the failures have a baseline.
probe_remote

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
