#!/usr/bin/env bash
# CI cargo runner with failure-aware sccache escalation.
#
# Why this exists: the previous inline retry loop treated every failure
# identically and, on its final attempt, rebuilt with sccache disabled.
# For a rustc/linker crash (SIGSEGV) or an OOM-kill that is exactly
# wrong — an uncached rebuild does *more* work under the same memory
# pressure, which is how a single OOM snowballed into a 90-minute hung
# job. Classify the failure instead:
#
#   * signal death  (exit >=128, or cargo reporting `signal: N` /
#                    SIGKILL / SIGSEGV) — a compiler crash, NOT an
#                    sccache fault. Keep the cache: one warm retry, then
#                    fail fast. Never escalate to uncached.
#   * sccache fault  (a recognisable sccache error in the output) —
#                    restart the server and retry; if it still faults,
#                    one final uncached attempt.
#   * anything else  (a deterministic compile/test error) — fail fast;
#                    a retry, cached or not, just burns minutes.
#
# Usage: ci-cargo-escalate.sh <cargo> <args...>
# The caller sets up the environment (PATH, CUDA_*, SCCACHE_*,
# HELEXA_BUILD_SHA, CARGO_BUILD_JOBS, …) before invoking.
set -uo pipefail

[ "$#" -ge 1 ] || {
  echo "usage: $0 <cargo command…>" >&2
  exit 2
}
cmd=("$@")

log="$(mktemp)"
trap 'rm -f "$log"' EXIT

have_sccache() { command -v sccache >/dev/null 2>&1; }

if have_sccache; then
  export RUSTC_WRAPPER=sccache
  sccache --start-server 2>/dev/null || true
  echo "sccache enabled"
else
  export RUSTC_WRAPPER=""
  echo "sccache not on PATH — building uncached"
fi

rc=0
run() {
  # Tee combined output so the failure can be classified, preserving the
  # command's own exit status (not tee's) via PIPESTATUS.
  "${cmd[@]}" 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
}

is_signal_death() {
  # Direct kill of the build process (OOM-killer, etc.).
  [ "$rc" -ge 128 ] && return 0
  # cargo catches a rustc/linker signal and itself exits 101, surfacing
  # it as `process didn't exit successfully: … (signal: 11, SIGSEGV…)`.
  grep -qiE "process didn't exit successfully.*\(signal: [0-9]+|\(signal: [0-9]+,|SIGSEGV|SIGKILL|SIGABRT|SIGBUS|SIGILL" "$log"
}

is_sccache_error() {
  have_sccache && [ -n "${RUSTC_WRAPPER:-}" ] || return 1
  grep -qiE "sccache: (error|fatal)|encountered fatal error|failed to (start|connect to)[^.]*sccache|sccache.*server (startup|unavailable|failed)|error: failed to start sccache server|Timed out waiting for sccache" "$log"
}

show_stats() { have_sccache && [ -n "${RUSTC_WRAPPER:-}" ] && sccache --show-stats || true; }

echo "::group::cargo: ${cmd[*]} (attempt 1)"
run
echo "::endgroup::"
if [ "$rc" -eq 0 ]; then
  show_stats
  exit 0
fi

# 1) Compiler crash / kill — keep the cache, one warm retry, then stop.
if is_signal_death; then
  echo "compiler crash or kill (exit ${rc}) — not an sccache fault; keeping cache for one warm retry"
  sleep 5
  echo "::group::cargo: ${cmd[*]} (signal retry, cached)"
  run
  echo "::endgroup::"
  if [ "$rc" -eq 0 ]; then
    show_stats
    exit 0
  fi
  echo "crash persisted (exit ${rc}) — failing fast, NOT dropping the cache"
  exit "$rc"
fi

# 2) sccache server fault — restart, retry, then one uncached attempt.
if is_sccache_error; then
  echo "sccache server fault detected — restarting and retrying"
  sccache --stop-server || true
  sccache --start-server || true
  echo "::group::cargo: ${cmd[*]} (after sccache restart)"
  run
  echo "::endgroup::"
  if [ "$rc" -eq 0 ]; then
    show_stats
    exit 0
  fi
  if is_signal_death; then
    echo "crash after sccache restart (exit ${rc}) — failing fast"
    exit "$rc"
  fi
  if is_sccache_error; then
    echo "sccache still faulting — final attempt without the cache"
    export RUSTC_WRAPPER=""
    echo "::group::cargo: ${cmd[*]} (uncached)"
    run
    echo "::endgroup::"
    exit "$rc"
  fi
  echo "non-sccache failure after restart (exit ${rc}) — failing fast"
  exit "$rc"
fi

# 3) Deterministic compile/test error — retrying changes nothing.
echo "deterministic failure (exit ${rc}) — failing fast (no cache drop, no retry)"
exit "$rc"
