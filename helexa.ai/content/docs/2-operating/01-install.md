---
title: Installing helexa
sidebar_label: Install
description: The two components, what runs where, and how to get them onto a fleet.
---

# Installing helexa

helexa is two programs. **cortex** is the control plane: one instance,
in front of the fleet, presenting the API and deciding which host serves
what. **neuron** is the node plane: one instance on every GPU host,
running inference in-process and managing local hardware and model
lifecycle.

cortex never touches a GPU, never shells out to `nvidia-smi`, and never
manages a model directly. It talks only to neurons. That separation is
what lets a host be added, drained or lost without reconfiguring the
gateway.

```
                 ┌──────────────┐
   clients ─────▶│    cortex    │   :31313 API, :31314 metrics
                 └──┬────┬────┬─┘
                    │    │    │
               ┌────▼┐ ┌─▼──┐ ┌▼────┐
               │neuron│ │neuron│ │neuron│   :13131 each
               │ GPU  │ │ GPU  │ │ GPU  │
               └──────┘ └──────┘ └──────┘
```

## Requirements

- **Fedora** with systemd. SELinux enforcing is fine.
- **CUDA** on each GPU host. neuron builds are compiled per compute
  capability, so the package must match the card — an architecture
  mismatch fails at load with a PTX error rather than falling back.
- A private network between hosts. Traffic between cortex and neurons is
  plaintext and assumes the network is already trusted; put TLS at the
  edge, not between the two.
- Disk for weights. Budget generously: a 27B model in bf16 is over 50 GB,
  and hosts usually hold several.

## Install

```sh
dnf copr enable helexa/helexa

# on the gateway host
dnf install cortex
systemctl enable --now cortex

# on each GPU host
dnf install helexa-neuron
systemctl enable --now neuron
```

The package is named `helexa-neuron` because Fedora already ships an
unrelated `neuron`. The binary, service, user and config directory are
all still called `neuron`.

## Verify

Each neuron reports what it found:

```sh
curl http://gpu-host:13131/discovery   # devices, VRAM, CUDA, driver
curl http://gpu-host:13131/health      # live VRAM, utilisation, temps
curl http://gpu-host:13131/version     # which build is actually running
```

`/version` is the one to trust when asking "is my change live?" — it
reports the build identity of the running process, which a package
version alone does not.

Then, from the gateway:

```sh
cortex status                          # fleet view
curl http://localhost:31313/v1/models  # one catalogue across every node
```

If a host appears in `cortex status` but its models do not appear in
`/v1/models`, the neuron is reachable but cannot currently serve —
usually not enough free VRAM. `/health` on that host will say so.

## Where to go next

- [Configuring cortex](/docs/operating/cortex) — the gateway
- [Configuring neurons](/docs/operating/neurons) — the GPU hosts
- [Placement & displacement](/docs/operating/placement) — which host
  serves what, and what may evict what
- [Context limits](/docs/operating/context-limits) — the knobs governing
  usable context

## Building from source

```sh
cargo build --release
```

Keep the checks green — CI treats warnings as errors:

```sh
cargo fmt --check --all
cargo clippy --workspace -- -D warnings
cargo test --workspace
```

Note that a local build is **CPU-only** unless you enable the CUDA
feature; the GPU paths are compiled out, so passing tests locally does
not prove the CUDA build compiles.
