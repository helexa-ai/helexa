#!/bin/env bash
#
# End-to-end smoke test for music generation (#387).
#
# Two stages:
#
# 1. Neuron, directly: load the model, render a fixed-seed 10 s
#    instrumental twice through /v1/audio/generations, and check the WAV
#    (header, 48 kHz stereo, duration, not silent, not clipped flat),
#    the metered units, and that the same seed gives the same bytes.
# 2. The chain, through cortex's async job API: submit, poll to
#    completion, fetch the content, check it the same way.
#
# A co-resident model evicted for the probe window is restored on every
# exit path, failures included. The probe model is unloaded afterwards
# only if this script loaded it.
#
# Usage:
#   script/validate-audio.sh [host] [cortex_url] [model_id]
#
# Defaults:
#   host       = benjy.hanzalova.internal   (the v1 placement)
#   cortex_url = http://hanzalova.internal:31313
#   model_id   = ACE-Step/Ace-Step1.5
#
# Set SKIP_CHAIN=1 to run only the neuron stage.

set -euo pipefail

HOST="${1:-benjy.hanzalova.internal}"
CORTEX="${2:-http://hanzalova.internal:31313}"
MODEL_ID="${3:-ACE-Step/Ace-Step1.5}"
BASE="http://${HOST}:13131"
WORK="$(mktemp -d)"

echo "==> /version"
curl -sf -m 10 "${BASE}/version" | python3 -c 'import json,sys; b=json.load(sys.stdin); print("  ", b.get("git_sha","?")[:12], b.get("profile"), b.get("features"))'

# The checks every rendered file must pass. Argument: path to the WAV,
# the expected duration in seconds.
check_wav() {
  python3 - "$1" "$2" <<'PY'
import struct, sys
path, want = sys.argv[1], float(sys.argv[2])
b = open(path, "rb").read()
assert b[:4] == b"RIFF" and b[8:12] == b"WAVE", "not a WAV file"
fmt, channels, rate = struct.unpack("<HHI", b[20:28])
bits = struct.unpack("<H", b[34:36])[0]
assert (fmt, channels, rate, bits) == (1, 2, 48000, 16), (fmt, channels, rate, bits)
n = struct.unpack("<I", b[40:44])[0] // 4
secs = n / rate
assert abs(secs - want) < 0.5, f"duration {secs:.2f}s, wanted {want}s"
samples = struct.unpack(f"<{n * 2}h", b[44:44 + n * 4])
rms = (sum(s * s for s in samples) / len(samples)) ** 0.5 / 32767
peak = max(abs(s) for s in samples) / 32767
assert rms > 0.01, f"near-silent: rms {rms:.4f}"
assert 0.5 < peak <= 1.0, f"peak {peak:.3f}"
print(f"   {secs:.2f}s 48kHz stereo | rms {rms:.3f} peak {peak:.3f}")
PY
}

EVICTED=()
LOADED_HERE=0
restore() {
  local rc=$?
  if [[ "${LOADED_HERE}" == 1 ]]; then
    curl -s -m 60 -X POST "${BASE}/models/unload" \
      -H 'Content-Type: application/json' \
      -d "{\"model_id\": \"${MODEL_ID}\"}" > /dev/null || true
  fi
  for id in "${EVICTED[@]}"; do
    echo "==> restoring ${id}"
    curl -s -m 900 -X POST "${BASE}/models/load" \
      -H 'Content-Type: application/json' \
      -d "{\"model_id\": \"${id}\", \"harness\": \"candle\"}" \
      || echo "   RESTORE FAILED: reload ${id} by hand"
    echo
  done
  rm -rf "${WORK}"
  exit "${rc}"
}
trap restore EXIT

listed=$(curl -sf -m 10 "${BASE}/models")
if ! echo "${listed}" | python3 -c '
import json, sys
m = next((m for m in json.load(sys.stdin) if m["id"] == sys.argv[1]), None)
sys.exit(0 if m and m["status"] == "loaded" else 1)
' "${MODEL_ID}"; then
  # Direct neuron loads have no evictor, so make room for the probe the
  # way cortex would: unload the other resident models, and put them
  # back afterwards.
  mapfile -t RESIDENT < <(echo "${listed}" | python3 -c '
import json, sys
for m in json.load(sys.stdin):
    if m["status"] == "loaded":
        print(m["id"])
')
  for id in "${RESIDENT[@]}"; do
    [[ -z "${id}" ]] && continue
    echo "==> evicting co-resident ${id} for the probe window"
    curl -sf -m 120 -X POST "${BASE}/models/unload" \
      -H 'Content-Type: application/json' \
      -d "{\"model_id\": \"${id}\"}"
    echo
    EVICTED+=("${id}")
  done
  echo "==> load ${MODEL_ID}"
  LOADED_HERE=1
  curl -sf -m 1800 -X POST "${BASE}/models/load" \
    -H 'Content-Type: application/json' \
    -d "{\"model_id\": \"${MODEL_ID}\", \"harness\": \"candle\"}"
  echo
fi

echo "==> capabilities"
curl -sf -m 10 "${BASE}/models" | python3 -c '
import json, sys
m = next((m for m in json.load(sys.stdin) if m["id"] == sys.argv[1]), None)
assert m, f"{sys.argv[1]} not listed"
assert m["status"] == "loaded", m["status"]
assert "audio" in m.get("capabilities", []), m.get("capabilities")
print("  ", m["id"], m["status"], m["capabilities"])
' "${MODEL_ID}"

REQ="{\"model\": \"${MODEL_ID}\", \"prompt\": \"Warm lo-fi hip hop beat with mellow Rhodes chords and dusty vinyl drums, 82 BPM\", \"duration\": 10, \"seed\": 42}"
for run in 1 2; do
  echo "==> neuron render ${run}: 10 s instrumental, seed 42"
  curl -sf -m 600 -D "${WORK}/h${run}" -o "${WORK}/n${run}.wav" -X POST "${BASE}/v1/audio/generations" \
    -H 'Content-Type: application/json' -d "${REQ}"
  check_wav "${WORK}/n${run}.wav" 10
  grep -i '^x-helexa-audio-result:' "${WORK}/h${run}" | cut -d' ' -f2- | python3 -c '
import json, sys
r = json.loads(sys.stdin.read())
assert r["seed"] == 42 and r["sample_rate"] == 48000 and r["channels"] == 2, r
assert abs(r["usage"]["helexa_audio_units"] - r["duration"]) < 1e-6, r
t = r["usage"]["helexa_timing"]
units = r["usage"]["helexa_audio_units"]
print("   units %.2f | encode %sms plan %sms generate %sms decode %sms" % (units, t["encode_ms"], t["plan_ms"], t["generate_ms"], t["decode_ms"]))
'
done
cmp -s "${WORK}/n1.wav" "${WORK}/n2.wav" && echo "   same seed, same bytes" \
  || { echo "   FAIL: same seed rendered different audio"; exit 1; }

if [[ "${SKIP_CHAIN:-0}" != 1 ]]; then
  echo "==> chain: submit through ${CORTEX}"
  job=$(curl -sf -m 30 -X POST "${CORTEX}/v1/audio/generations" \
    -H 'Content-Type: application/json' -d "${REQ}")
  id=$(echo "${job}" | python3 -c 'import json,sys; j=json.load(sys.stdin); assert j["status"]=="queued", j; print(j["id"])')
  echo "   ${id} queued"
  for _ in $(seq 1 300); do
    status=$(curl -sf -m 10 "${CORTEX}/v1/audio/generations/${id}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])')
    [[ "${status}" == succeeded || "${status}" == failed || "${status}" == cancelled ]] && break
    sleep 2
  done
  [[ "${status}" == succeeded ]] || { curl -s "${CORTEX}/v1/audio/generations/${id}"; echo; echo "   FAIL: job ${status}"; exit 1; }
  curl -sf -m 60 -o "${WORK}/c.wav" "${CORTEX}/v1/audio/generations/${id}/content"
  check_wav "${WORK}/c.wav" 10
  cmp -s "${WORK}/n1.wav" "${WORK}/c.wav" && echo "   chain render matches the direct render" \
    || { echo "   FAIL: chain render differs from the direct render at the same seed"; exit 1; }
fi

echo "==> health"
curl -sf -m 10 "${BASE}/health" > /dev/null && echo "   healthy"
echo "PASS"
