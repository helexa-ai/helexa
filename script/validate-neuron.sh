#!/bin/env bash
#
# End-to-end smoke test for a deployed neuron.
#
# Confirms the daemon is reachable, loads a small public Qwen3 GGUF,
# fires a reasoning probe at /v1/chat/completions, and prints the
# answer. Used to validate the candle harness on a real GPU host
# before trusting it for production traffic, and as a regression test
# after pushing new neuron builds.
#
# Usage:
#   script/validate-neuron.sh [host] [model_id] [quant] [tp_size]
#
# Defaults:
#   host     = beast.hanzalova.internal
#   model_id = unsloth/Qwen3-0.6B-GGUF  (official Qwen3-*-GGUF repos
#              ship Q8_0 only; unsloth's mirror ships the full Q-spectrum
#              including Q4_K_M)
#   quant    = Q4_K_M  (empty = dense safetensors path)
#   tp_size  = unset   (= 1 = single-GPU; pass 2 to drive the TP path)

set -euo pipefail

HOST="${1:-beast.hanzalova.internal}"
MODEL_ID="${2:-unsloth/Qwen3-0.6B-GGUF}"
# `${3-Q4_K_M}` (no colon) only uses the default when the arg is
# UNSET — passing an explicit empty string drives the dense path.
QUANT="${3-Q4_K_M}"
# tp_size > 1 forces the dense path (TP requires safetensors) and adds
# `tensor_parallel: N` to the load payload. The harness picks device
# indices 0..N-1 by default; override by passing NEURON_DEVICES="0,1,..."
# in the environment.
TP_SIZE="${4-1}"
PORT="${NEURON_PORT:-13131}"
BASE="http://${HOST}:${PORT}"

# Reasoning probe — concrete, low-temperature answer that small models
# can still get right. "Paris" is a strong signal of basic competence
# beyond gibberish.
PROBE_PROMPT='What is the capital of Georgia (Caucasus)? Respond with the city name only, no punctuation.'
EXPECT_SUBSTR='Tbilisi'
# Qwen3 prepends <think>...</think> reasoning before the answer when the
# chat template enables thinking mode, which eats most of a small token
# budget. 256 leaves enough room for thinking + final answer.
MAX_TOKENS=256

# /models/load is synchronous — neuron blocks the response until the
# hf-hub download + (GGUF parse or safetensors mmap) + tensor
# materialisation is done. Small GGUF (0.6B-Q4_K_M, ~400 MB) is
# typically a minute on a warm cache, several on a cold one. A
# Qwen3.6-class dense model is ~54 GB and can easily take an hour to
# download cold over a residential link, so default high. Override
# with NEURON_LOAD_TIMEOUT=N (seconds) for smaller targets if you'd
# rather fail fast.
LOAD_TIMEOUT="${NEURON_LOAD_TIMEOUT:-3600}"
INFER_TIMEOUT="${NEURON_INFER_TIMEOUT:-120}"

# Status messages go to stderr so command substitutions like
# `raw=$(run_probe)` capture only the function's intended return value
# (an HTTP body), not the progress chatter.
say() { printf '[%s] %s\n' "${HOST}" "$*" >&2; }
die() { say "FAIL: $*"; exit 1; }

probe_health() {
    curl --silent --fail --max-time 5 "${BASE}/health" >/dev/null \
        || die "neuron not reachable at ${BASE}/health"
}

# Block until the neuron reports `activation.state == "ready"` on
# `/health`. Without this, validate-neuron.sh used to race the
# background pre-warm (the listener binds immediately but big TP
# loads run for minutes after) and either fail with ECONNREFUSED
# (pre-2026-05-26 build, where load was synchronous before bind) or
# get a 404 from /models/load against a partially-loaded model.
#
# The poll cap is `NEURON_LOAD_TIMEOUT` since pre-warm and an
# on-demand load are the same operation under different triggers.
# Short interval at the start (catches a quick-loading host without
# extra latency) backs off after the first few iterations to keep
# log spam down on a slow load.
wait_for_ready() {
    local deadline=$(( $(date +%s) + LOAD_TIMEOUT ))
    local state= attempt=0
    while (( $(date +%s) < deadline )); do
        attempt=$(( attempt + 1 ))
        state=$(
            curl --silent --max-time 5 "${BASE}/health" \
                | jq -r '.activation.state // "unknown"'
        ) || state=unreachable
        case "${state}" in
            ready)
                say "/health activation.state=ready (after ${attempt} probe(s))"
                return 0
                ;;
            pre_warming)
                local in_progress
                in_progress=$(
                    curl --silent --max-time 5 "${BASE}/health" \
                        | jq -r '.activation.in_progress // "<none>"'
                ) || in_progress='<unreadable>'
                say "/health pre_warming (in_progress=${in_progress}); waiting"
                ;;
            unreachable)
                say "/health unreachable; waiting"
                ;;
            *)
                say "/health unexpected activation.state=${state}; waiting"
                ;;
        esac
        # 2s for the first few iterations to catch quick loads, then
        # 10s to avoid log spam on a multi-minute TP load.
        if (( attempt < 5 )); then
            sleep 2
        else
            sleep 10
        fi
    done
    die "neuron not ready within ${LOAD_TIMEOUT}s (last state: ${state})"
}

list_loaded_ids() {
    # The manifest is YAML and uses yq; HTTP responses are JSON and use
    # jq directly. pip-yq parses input as YAML by default, which trips
    # on JSON content that happens to look like YAML aliases (chatcmpl
    # ids, escaped quotes inside `<think>...</think>` blocks, etc.).
    curl --silent --fail "${BASE}/models" | jq -r '.[].id'
}

is_loaded() {
    list_loaded_ids 2>/dev/null | grep -Fxq "${MODEL_ID}"
}

trigger_load() {
    # Build the per-rank CUDA device list as a JSON array. Either
    # honour NEURON_DEVICES (`0,1,2`) verbatim or default to
    # `[0, 1, ..., tp_size - 1]`.
    local devices_json
    if [[ -n "${NEURON_DEVICES:-}" ]]; then
        devices_json=$(jq -n -c --arg s "${NEURON_DEVICES}" \
            '$s | split(",") | map(tonumber)')
    else
        devices_json=$(jq -n -c --argjson n "${TP_SIZE}" '[range(0; $n)]')
    fi
    say "POST /models/load ${MODEL_ID} (quant=${QUANT:-<dense>}, tp=${TP_SIZE}, devices=${devices_json})"
    say "  (synchronous; may take a minute on first run while HF downloads)"
    # Build the payload via jq so optional fields are omitted entirely
    # when not in use. `tensor_parallel` is dropped when tp_size == 1;
    # `quant` is dropped when empty. Both can coexist: tp_size > 1 +
    # ISQ quant (q5k/q8_0/etc.) loads safetensors and quantizes the
    # per-rank shard at load time. GGUF quants (Q4_K_M) are incompatible
    # with TP — but the harness rejects that combination at load time
    # rather than here.
    local payload
    local base
    base=$(jq -n -c \
        --arg id "${MODEL_ID}" \
        --argjson devices "${devices_json}" \
        '{model_id: $id, harness: "candle", devices: $devices}')
    if [[ -n "${QUANT}" ]]; then
        base=$(echo "${base}" | jq -c --arg q "${QUANT}" '. + {quant: $q}')
    fi
    if (( TP_SIZE > 1 )); then
        base=$(echo "${base}" | jq -c --argjson tp "${TP_SIZE}" '. + {tensor_parallel: $tp}')
    fi
    payload="${base}"
    # --write-out captures the response code on a separate line so we
    # can surface a real diagnostic instead of relying on --fail.
    local resp http_code body
    resp=$(curl --silent --show-error --max-time "${LOAD_TIMEOUT}" \
        --write-out '\n__HTTP__%{http_code}' \
        -X POST "${BASE}/models/load" \
        -H 'content-type: application/json' \
        --data "${payload}") || die "curl /models/load failed: $?"
    http_code=$(echo "${resp}" | grep -oP '(?<=__HTTP__)\d+$' | tail -1)
    body=$(echo "${resp}" | sed '$ s/__HTTP__.*$//')
    if [[ "${http_code}" != "200" ]]; then
        die "load returned HTTP ${http_code}: ${body}"
    fi
    say "load returned ${http_code}: ${body}"
}

run_probe() {
    say "POST /v1/chat/completions (probe: ${PROBE_PROMPT})"
    local payload
    payload=$(jq -n -c \
        --arg model "${MODEL_ID}" \
        --arg content "${PROBE_PROMPT}" \
        --argjson tokens "${MAX_TOKENS}" \
        '{
            model: $model,
            messages: [{role: "user", content: $content}],
            temperature: 0.1,
            max_tokens: $tokens
        }')
    local resp http_code body
    resp=$(curl --silent --show-error --max-time "${INFER_TIMEOUT}" \
        --write-out '\n__HTTP__%{http_code}' \
        -X POST "${BASE}/v1/chat/completions" \
        -H 'content-type: application/json' \
        --data "${payload}") || die "curl /v1/chat/completions failed: $?"
    http_code=$(echo "${resp}" | grep -oP '(?<=__HTTP__)\d+$' | tail -1)
    body=$(echo "${resp}" | sed '$ s/__HTTP__.*$//')
    if [[ "${http_code}" != "200" ]]; then
        die "inference returned HTTP ${http_code}: ${body}"
    fi
    echo "${body}"
}

say "validating neuron at ${BASE}"
probe_health
say "/health OK"
# Background pre-warm from default_models means /health is reachable
# but `activation.state` can still be `pre_warming` for minutes after
# service start. Block here so the subsequent is_loaded / trigger_load
# steps don't race a partially-materialised model.
wait_for_ready

if is_loaded; then
    say "${MODEL_ID} already loaded"
else
    trigger_load
fi

raw=$(run_probe)
echo "---"
# Dump the raw JSON. Don't pipe through `yq -r '.'` — yq's default
# YAML output mode chokes on JSON strings that contain `<` (and the
# `<think>` markers Qwen3 emits during reasoning are a perfect
# example). The targeted `yq -r '.path'` calls below work fine
# because jq's path filter mode bypasses the YAML re-emit.
echo "${raw}"
echo "---"

content=$(echo "${raw}" | jq -r '.choices[0].message.content // empty')
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
