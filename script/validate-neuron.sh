#!/bin/env bash
#
# End-to-end smoke test for a deployed neuron.
#
# Confirms the daemon is reachable, loads a small public Qwen3 GGUF,
# fires a reasoning probe at /v1/chat/completions, and prints the
# answer. Use this to validate the candle harness on a real GPU host
# before trusting it for production traffic, and as a regression test
# after pushing new neuron builds.
#
# Usage:
#   script/validate-neuron.sh [host] [model_id] [quant]
#
# Defaults:
#   host     = beast.hanzalova.internal
#   model_id = Qwen/Qwen3-1.7B-GGUF
#   quant    = Q4_K_M

set -euo pipefail

HOST="${1:-beast.hanzalova.internal}"
MODEL_ID="${2:-Qwen/Qwen3-1.7B-GGUF}"
QUANT="${3:-Q4_K_M}"
PORT="${NEURON_PORT:-13131}"
BASE="http://${HOST}:${PORT}"

# Reasoning probe — concrete, low-temperature answer that small models
# can still get right. "Paris" is a strong signal of basic competence
# beyond gibberish.
PROBE_PROMPT='What is the capital of France? Respond with the city name only, no punctuation.'
EXPECT_SUBSTR='Paris'
MAX_TOKENS=32

# Polling cadence while the model loads.
LOAD_POLL_INTERVAL=5
LOAD_POLL_MAX=120   # 10 min worst-case for a fresh HF download

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

say() { printf '[%s] %s\n' "${HOST}" "$*"; }
die() { say "FAIL: $*"; exit 1; }

probe_health() {
    curl --silent --fail --max-time 5 "${BASE}/health" >/dev/null \
        || die "neuron not reachable at ${BASE}/health"
}

list_loaded_ids() {
    curl --silent --fail "${BASE}/models" \
        | yq -r '.[].id'
}

is_loaded() {
    list_loaded_ids | grep -Fxq "${MODEL_ID}"
}

trigger_load() {
    say "POST /models/load ${MODEL_ID} (quant=${QUANT}, device=[0])"
    curl --silent --fail --max-time 30 \
        -X POST "${BASE}/models/load" \
        -H 'content-type: application/json' \
        --data-binary @- <<EOF >/dev/null
{
    "model_id": "${MODEL_ID}",
    "harness": "candle",
    "quant": "${QUANT}",
    "devices": [0]
}
EOF
}

wait_for_load() {
    local elapsed=0
    while ! is_loaded; do
        if (( elapsed >= LOAD_POLL_MAX )); then
            die "model did not appear in /models after ${LOAD_POLL_MAX} polls"
        fi
        sleep "${LOAD_POLL_INTERVAL}"
        elapsed=$(( elapsed + 1 ))
        say "still loading... (${elapsed}/${LOAD_POLL_MAX})"
    done
    say "model loaded"
}

run_probe() {
    say "POST /v1/chat/completions (probe: ${PROBE_PROMPT})"
    local resp
    resp=$(
        curl --silent --fail --max-time 120 \
            -X POST "${BASE}/v1/chat/completions" \
            -H 'content-type: application/json' \
            --data-binary @- <<EOF
{
    "model": "${MODEL_ID}",
    "messages": [{"role": "user", "content": ${PROBE_PROMPT@Q}}],
    "temperature": 0.1,
    "max_tokens": ${MAX_TOKENS}
}
EOF
    )
    echo "${resp}"
}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

say "validating neuron at ${BASE}"
probe_health
say "/health OK"

if is_loaded; then
    say "${MODEL_ID} already loaded"
else
    # Note: /models/load returns once the load is initiated. For large
    # models the actual materialisation continues asynchronously; the
    # registry only reflects success once it's complete, hence the
    # subsequent poll loop.
    trigger_load
    wait_for_load
fi

raw=$(run_probe)
echo "---"
echo "${raw}" | yq -r '.'
echo "---"

content=$(echo "${raw}" | yq -r '.choices[0].message.content // empty')
if [[ -z "${content}" ]]; then
    die "no content in chat completion response"
fi
say "assistant said: ${content}"

if echo "${content}" | grep -qiF "${EXPECT_SUBSTR}"; then
    say "PASS — response contains expected substring '${EXPECT_SUBSTR}'"
    exit 0
else
    die "response did not contain '${EXPECT_SUBSTR}'"
fi
