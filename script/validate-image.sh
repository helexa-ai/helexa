#!/bin/env bash
#
# End-to-end smoke test for text-to-image serving on a neuron (#203).
#
# Loads Z-Image-Turbo (or a same-architecture variant), generates a
# fixed-seed 512² probe image via /v1/images/generations, verifies the
# PNG magic + dimensions + metered units, unloads, and confirms the
# host still lists a healthy state. Mirrors validate-neuron.sh for the
# image modality.
#
# Usage:
#   script/validate-image.sh [host] [model_id]
#
# Defaults:
#   host     = benjy.hanzalova.internal   (the v1 image placement)
#   model_id = Tongyi-MAI/Z-Image-Turbo

set -euo pipefail

HOST="${1:-benjy.hanzalova.internal}"
MODEL_ID="${2:-Tongyi-MAI/Z-Image-Turbo}"
BASE="http://${HOST}:13131"

echo "==> /version"
curl -sf -m 10 "${BASE}/version" | python3 -c 'import json,sys; b=json.load(sys.stdin); print("  ", b.get("git_sha","?")[:12], b.get("profile"), b.get("features"))'

# Co-residency (#203): direct neuron loads have no evictor — that is
# cortex's job. On a host whose resident text model leaves too little
# VRAM for the DiT (benjy: 8B @ 16 GB + 14.6 GB DiT > 24 GB), evict
# it for the probe window and restore it afterwards.
RESTORE_JSON=""
EVICT=$(curl -sf -m 10 "${BASE}/models" | python3 -c '
import json, sys
models = json.load(sys.stdin)
loaded = [m for m in models if m["status"] == "loaded" and "image" not in m.get("capabilities", [])]
print(loaded[0]["id"] if loaded else "")
')
if [[ -n "${EVICT}" ]]; then
  echo "==> evicting co-resident ${EVICT} for the probe window"
  curl -sf -m 120 -X POST "${BASE}/models/unload" \
    -H 'Content-Type: application/json' \
    -d "{\"model_id\": \"${EVICT}\"}"
  echo
  RESTORE_JSON="{\"model_id\": \"${EVICT}\", \"harness\": \"candle\"}"
fi

echo "==> load ${MODEL_ID}"
curl -sf -m 900 -X POST "${BASE}/models/load" \
  -H 'Content-Type: application/json' \
  -d "{\"model_id\": \"${MODEL_ID}\", \"harness\": \"candle\"}"
echo

echo "==> capabilities"
curl -sf -m 10 "${BASE}/models" | python3 - "$MODEL_ID" << 'PYEOF'
import json, sys
model_id = sys.argv[1]
models = json.load(sys.stdin)
m = next((m for m in models if m["id"] == model_id), None)
assert m, f"{model_id} not listed"
assert m["status"] == "loaded", m["status"]
assert "image" in m.get("capabilities", []), m.get("capabilities")
print("  ", m["id"], m["status"], m["capabilities"])
PYEOF

echo "==> generate 512x512 seed 42"
curl -sf -m 300 -X POST "${BASE}/v1/images/generations" \
  -H 'Content-Type: application/json' \
  -d "{\"model\": \"${MODEL_ID}\", \"prompt\": \"a lighthouse at dusk, photorealistic\", \"size\": \"512x512\", \"seed\": 42}" \
  | python3 << 'PYEOF'
import base64, json, struct, sys
r = json.load(sys.stdin)
png = base64.b64decode(r["data"][0]["b64_json"])
assert png[1:4] == b"PNG", "not a PNG"
# IHDR width/height live at fixed offsets in the first chunk.
w, h = struct.unpack(">II", png[16:24])
assert (w, h) == (512, 512), (w, h)
usage = r["usage"]
units = usage["helexa_image_units"]
assert abs(units - 512 * 512 * 9 / 1e6) < 1e-6, units
t = usage["helexa_timing"]
print(f"   512x512 ok | units={units:.3f} | encode={t['encode_ms']}ms "
      f"denoise={t['denoise_ms']}ms decode={t['decode_ms']}ms")
PYEOF

echo "==> unload"
curl -sf -m 60 -X POST "${BASE}/models/unload" \
  -H 'Content-Type: application/json' \
  -d "{\"model_id\": \"${MODEL_ID}\"}"
echo

if [[ -n "${RESTORE_JSON}" ]]; then
  echo "==> restoring ${EVICT}"
  curl -sf -m 900 -X POST "${BASE}/models/load" \
    -H 'Content-Type: application/json' \
    -d "${RESTORE_JSON}"
  echo
fi

echo "==> health"
curl -sf -m 10 "${BASE}/health" > /dev/null && echo "   healthy"
echo "PASS"
