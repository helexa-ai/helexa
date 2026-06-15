#!/usr/bin/env bash
#
# Set RUST_LOG across the helexa fleet via systemd drop-ins.
#
# By default this touches only cortex (the gateway) — that's where the
# wire-debug instrumentation lives, and a neuron restart would needlessly
# drop loaded models and force a cold reload. Pass --with-neuron to also
# reconfigure + restart the neuron daemons.
#
# Usage:
#   infra-log-verbosity.sh [--with-neuron] [CORTEX_RUST_LOG] [NEURON_RUST_LOG]
#
# Examples:
#   # cortex only, default filter (trace the new /v1/messages wire logging):
#   infra-log-verbosity.sh
#
#   # cortex back down to plain debug, cortex only:
#   infra-log-verbosity.sh debug
#
#   # also flip neuron to trace its harness while debugging tool-call parsing:
#   infra-log-verbosity.sh --with-neuron \
#     'debug,cortex_gateway::handlers=trace,cortex_gateway::anthropic_sse=trace' \
#     'debug,neuron::wire=trace,neuron::harness=trace'
#
set -euo pipefail

with_neuron=0
positional=()
for arg in "$@"; do
    case "$arg" in
        --with-neuron) with_neuron=1 ;;
        *) positional+=("$arg") ;;
    esac
done

# Independent defaults: $1 drives cortex, $2 drives neuron. Passing one
# no longer clobbers the other.
cortex_verbosity=${positional[0]:-"debug,cortex_gateway::handlers=trace,cortex_gateway::anthropic_sse=trace"}
neuron_verbosity=${positional[1]:-debug}

cortex_host=hanzalova.internal
neuron_hosts=(beast benjy quadbrat)

# Write an authoritative `log.conf` drop-in for SERVICE on HOST, strip
# RUST_LOG from any sibling drop-in so ours is the single source of
# truth, then daemon-reload + restart so the new env is picked up.
#
# The remote half runs as `sudo bash -s`; SERVICE and the RUST_LOG value
# arrive as positional args (single-quoted by the local shell), so the
# value's `::`, `,` and `=` need no further escaping. The heredoc is
# quoted (<<'REMOTE') so it is sent verbatim and expanded remotely.
configure_host() {
    local host=$1 service=$2 verbosity=$3
    echo "[${host}] RUST_LOG=${verbosity} -> ${service}.service"
    ssh "$host" "sudo bash -s -- '$service' '$verbosity'" <<'REMOTE'
set -euo pipefail
shopt -s nullglob
service=$1
verbosity=$2
dropin="/etc/systemd/system/${service}.service.d"
mkdir -p "$dropin"
# Keep log.conf authoritative: drop RUST_LOG from any other drop-in
# (e.g. local.conf) so there's no competing assignment.
for f in "$dropin"/*.conf; do
    [ "$f" = "$dropin/log.conf" ] && continue
    sed -i '/RUST_LOG/d' "$f"
done
printf '[Service]\nEnvironment=RUST_LOG=%s\n' "$verbosity" > "$dropin/log.conf"
systemctl daemon-reload
systemctl restart "${service}.service"
REMOTE
}

configure_host "$cortex_host" cortex "$cortex_verbosity"

if [ "$with_neuron" -eq 1 ]; then
    for host in "${neuron_hosts[@]}"; do
        configure_host "${host}.${cortex_host}" neuron "$neuron_verbosity"
    done
else
    echo "Skipping neuron hosts (pass --with-neuron to include them)."
fi
