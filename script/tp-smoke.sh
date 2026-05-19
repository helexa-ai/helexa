#!/bin/env bash
#
# TP smoke test against a deployed neuron host.
#
# SSHes into the target host and runs `neuron --tp-smoke --tp-size N
# --cuda-devices ...` directly — no HTTP API involved. The smoke
# subcommand spawns N-1 worker subprocesses, joins them in an NCCL
# communicator, runs one AllReduce(Sum) of `1u32` across every rank, and
# verifies the observed sum equals world_size on every rank.
#
# This validates the lower-half of the TP stack (NCCL + IPC topology +
# subprocess lifecycle) without touching model load, inference, or HTTP.
# A failure here means the host cannot run any TP model and there is no
# point debugging the higher layers.
#
# Usage:
#   script/tp-smoke.sh [host] [tp_size] [cuda_devices]
#
# Defaults:
#   host         = beast.hanzalova.internal  (only fleet host with 2 GPUs)
#   tp_size      = 2
#   cuda_devices = 0,1

set -euo pipefail

HOST="${1:-beast.hanzalova.internal}"
TP_SIZE="${2:-2}"
CUDA_DEVICES="${3:-0,1}"

say() { printf '[%s] %s\n' "${HOST}" "$*" >&2; }
die() { say "FAIL: $*"; exit 1; }

say "running neuron --tp-smoke --tp-size ${TP_SIZE} --cuda-devices ${CUDA_DEVICES}"

# Run as root via sudo because:
#   - cuda contexts under a user account require either the nvidia
#     uvm/peer devices to be world-readable or the user to be in a
#     priviliged group (neither is true on stock fc43);
#   - the installed binary lives at /usr/bin/neuron with no setuid;
# Running through root is the simplest path that matches how
# systemd-managed neuron sees the GPUs in production.
#
# The smoke command is read-only — it allocates a transient NCCL comm
# and a 1u32 buffer per rank, then tears it all down.
if ! ssh -o BatchMode=yes "${HOST}" \
    sudo /usr/bin/neuron \
        --tp-smoke \
        --tp-size "${TP_SIZE}" \
        --cuda-devices "${CUDA_DEVICES}" 2>&1 | tee /tmp/tp-smoke-"${HOST}".log
then
    die "tp-smoke exited non-zero (see /tmp/tp-smoke-${HOST}.log)"
fi

# Final stdout line is `status=ok` on success.
if grep -q '^status=ok$' /tmp/tp-smoke-"${HOST}".log; then
    say "PASS — NCCL handshake + AllReduce sanity check OK across ${TP_SIZE} ranks"
    exit 0
else
    die "no status=ok line in output"
fi
