#!/usr/bin/env bash
# Open cortex's Prometheus metrics port (:31314) on the cortex gateway host
# (hanzalova.internal, 10.6.0.46) to the monitoring host ONLY
# (golgafrinchans.kosherinata.internal, which reaches hanzalova as 10.3.101.4).
#
# cortex binds 0.0.0.0:31314 but firewalld has no rule for it, so a
# cross-host scrape times out. This adds a scoped rich rule — the metrics
# stay closed to everything except the Prometheus host.
#
# Run as root on hanzalova.internal. Idempotent.
set -euo pipefail

MONITOR_IP="10.3.101.4"   # golgafrinchans.kosherinata.internal (mesh source)
PORT="31314"

rule="rule family=\"ipv4\" source address=\"${MONITOR_IP}/32\" port port=\"${PORT}\" protocol=\"tcp\" accept"

if firewall-cmd --permanent --query-rich-rule="${rule}" >/dev/null 2>&1; then
  echo "rich rule already present; nothing to do"
else
  firewall-cmd --permanent --add-rich-rule="${rule}"
  firewall-cmd --reload
  echo "opened tcp/${PORT} to ${MONITOR_IP} and reloaded firewalld"
fi

# Verify from this host; the real check is a scrape from the monitoring host:
#   curl -s -o /dev/null -w '%{http_code}\n' http://hanzalova.internal:31314/metrics
firewall-cmd --list-rich-rules | grep "${PORT}" || true
