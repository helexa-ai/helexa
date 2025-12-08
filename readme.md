helexa
======

helexa is a distributed ai fabric composed of two primary node roles:

- **cortex**: a control-plane node that always participates in the mesh and can optionally
  act as an orchestrator, api gateway, and portal host.
- **neuron**: a data-plane node that runs language / vision models and exposes them to cortex.

this repository contains a cargo workspace that implements the core libraries and binaries
for running helexa nodes operated by many independent participants.

status
------

early scaffold. nothing here is ready for production. expect api breaks until there is
an initial public spec and versioned protocol semantics.

high-level topology
-------------------

- the **mesh** forms a p2p overlay between cortex nodes (and optionally neurons).
- **cortex** nodes:
  - join the mesh
  - optionally act as orchestrators (policy and provisioning)
  - optionally act as api gateways (openai-compatible entrypoint)
  - optionally serve one or more portals (front-end + billing surfaces)
- **neuron** nodes:
  - register capabilities with one or more cortex nodes
  - run model runtimes (llama.cpp, vllm, mistral.rs, etc) via a shared abstraction
  - execute inference work assigned by cortex

installation
------------

you need a recent rust toolchain (pinned via `rust-toolchain.toml`).

```bash
git clone https://github.com/helexa-ai/helexa.git
cd helexa
cargo build --workspace
```

binaries
--------

this workspace produces a single top-level binary:

- `helexa` — multi-role entrypoint with subcommands:
  - `helexa cortex` — run a cortex node
  - `helexa neuron` — run a neuron node

examples
--------

### run a simple cortex

```bash
helexa cortex \
  --orchestrator-socket 0.0.0.0:8040 \
  --gateway-socket      0.0.0.0:8080 \
  --portal-socket       0.0.0.0:8091
```

this starts a cortex node that joins the mesh and:

- exposes an orchestrator control api on port 8040
- exposes an openai-compatible api gateway on port 8080
- serves a default portal on port 8091

### run a simple neuron

```bash
helexa neuron \
  --control-socket 0.0.0.0:9050 \
  --api-socket     127.0.0.1:8060 \
  --models-dir     /var/lib/helexa/models
```

this starts a neuron that:

- listens for control messages from cortex on port 9050
- exposes a local api for model serving on port 8060
- uses `/var/lib/helexa/models` as its model store

development
-----------

the workspace is organised as a set of small, composable crates:

- `helexa-cli` — entrypoint binary and cli parsing
- `cortex` — control-plane node logic
- `neuron` — data-plane node logic
- `mesh` — p2p membership, discovery, and identity
- `protocol` — shared types and wire formats
- `model-runtime` — abstraction over concrete model runtimes
- `config` — layered configuration loader
- `util` — logging, metrics and misc utilities

see `agents.md` for guidance on where new code and tests belong.

documentation
-------------

design documentation lives under `docs/`:

- `docs/architecture.md` — describes topology, responsibilities and flows

licensing
---------

the intended licensing model is to keep the core of helexa open source to build trust
with operators. exact license details are still to be determined.
