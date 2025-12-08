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

helexa is provided under a **source-available** model with a scheduled transition to
a fully open source license.

- until **january 1st, 2028**, the codebase is licensed under the **polyform shield license 1.0**.
- on and after **january 1st, 2028**, this repository and all contributions made prior
  to that date are automatically and irrevocably relicensed to **apache-2.0**.

see `license.md` for the full terms and the exact legal text, including the spdx
identifiers to use in source files.

# 📝 why this license?

the helexa codebase is public because **trust requires transparency**.
our network, our operators, and the people who depend on distributed ai infrastructure deserve the ability to **audit the code**, verify our intentions, and build confidence in how helexa works.

at the same time, early-stage infrastructure projects are fragile. a fully permissive license would allow a better-funded competitor to **lift the entire platform and outrun the project before it has a chance to stand on its own**. that would undermine the long-term vision of a **diverse, decentralised, globally accessible ai mesh**, especially across regions traditionally left out of centralised ai growth.

for this reason, helexa uses the **polyform shield license** until **january 1st, 2028**, after which it automatically becomes **apache 2.0**.

### this approach gives us:

* **auditability from day one**
  the code is public, reviewable, and modifiable for non-commercial purposes.

* **protection during the fragile early years**
  commercial rights remain with helexa while foundational work is being built, tested, and stabilised.

* **a guaranteed path to full open-source freedom**
  on january 1st, 2028, the project relicenses to **apache 2.0**—a fully permissive, industry-standard open source license. no surprises, no bait-and-switch.

* **community alignment**
  contributors know exactly how their work will be used today and how it will evolve.
  operators and integrators get long-term clarity and stability.

this model ensures that helexa can grow fast enough to serve its mission while still committing to the **open, decentralised future** we want the ecosystem to inherit.

contributing
------------

we welcome contributions from operators, implementors, and researchers who share the
goal of building a trustworthy, decentralised ai fabric.

before contributing, please:

1. **read the license**

   - understand that, until **2028-01-01**, contributions are under the
     **polyform shield 1.0** terms.
   - after that date, the project transitions to **apache-2.0** for all prior
     and future contributions.
   - see `license.md` for the definitive legal details and spdx identifiers.

2. **follow the ci and code quality gates**

   changes must pass the workspace ci pipeline, which runs on pushes and pull
   requests to `main`:

   - `cargo fmt --all -- --check`
   - `cargo clippy --workspace --all-targets --all-features -- -D warnings`
   - `cargo test --workspace --all-features`

   locally, you should run:

   ```bash
   cargo fmt --all
   cargo clippy --workspace --all-targets --all-features
   cargo test --workspace --all-features
   ```

   fix issues rather than suppressing them. if you must silence a lint, scope it
   as narrowly as possible and document why.

3. **include spdx headers in rust sources**

   all rust source files in this workspace are expected to declare the license
   explicitly via an spdx header:

   ```rust
   // SPDX-License-Identifier: PolyForm-Shield-1.0
   ```

   the ci workflow enforces this for pull requests by checking that all `*.rs`
   files under `crates/` contain this header. if you add a new rust file and
   omit the header, ci will fail with a message listing the offending paths.

4. **keep scaffolds explicit**

   when adding new fields or modules that are not yet fully implemented:

   - use them in real code paths (e.g. logging, routing decisions).
   - prefer `todo!()` / `unimplemented!()` over silent no-ops, so incomplete
     behaviour is obvious during development.
   - avoid long-lived `#[allow(dead_code)]` attributes; instead, make the
     intention explicit and fail loudly until the implementation is ready.

5. **follow architectural boundaries**

   - keep business logic out of `helexa-cli`; it should only parse cli, load
     config, and delegate to `cortex::run` / `neuron::run`.
   - put shared protocol types and traits in `crates/protocol`.
   - keep crates small and focused as described in `agents.md` and
     `docs/architecture.md`.

for more detailed guidance on contributing, including scaffolding patterns and
workflow expectations, see `agents.md`.
