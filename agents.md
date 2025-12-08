agents guide
============

this file is for humans and machine agents contributing to the helexa workspace.
it describes where responsibilities live and how to keep the repository tidy.

overall principles
------------------

- small crates with single responsibilities beat huge grab-bags.
- shared behaviour should live in libraries, not in the binary crate.
- public apis should be explicit; prefer small, focussed traits over giant ones.
- tests should live as close as possible to the code they exercise.

workspace structure
-------------------

- `crates/helexa-cli`
  - owns the `helexa` binary.
  - only responsibilities:
    - parse cli flags and subcommands
    - initial configuration loading (config file + env + cli)
    - wire subcommands into `cortex::run` and `neuron::run`
  - **no business logic** should be implemented here.

- `crates/cortex`
  - control-plane node library.
  - always joins the mesh.
  - optional roles:
    - orchestrator: model policy, worker selection, provisioning
    - gateway: openai-compatible api ingress and request routing
    - portal: http(s) front-end and billing hooks
  - main entrypoint is `cortex::run(config)`, called from the cli crate.
  - internal modules:
    - `mesh.rs` — glue between cortex and the `mesh` crate.
    - `orchestrator.rs` — orchestration traits and implementations.
    - `gateway.rs` — http routing, request classification and dispatch.
    - `portal.rs` — front-end server(s), multi-tenant portals.
    - `shutdown.rs` — graceful shutdown wiring.

- `crates/neuron`
  - data-plane node library.
  - runs models, exposes inference services and reports capabilities.
  - main entrypoint is `neuron::run(config)`.
  - internal modules:
    - `runtime.rs` — integration with `model-runtime` crate.
    - `control_plane.rs` — protocol handling for commands from cortex.
    - `registry.rs` — local model inventory, capabilities and health.

- `crates/mesh`
  - p2p membership, identity and discovery.
  - provides a `MeshHandle` used by cortex (and optionally neuron).
  - hides concrete transport primitives (libp2p, quic, rosenpass, etc).

- `crates/protocol`
  - shared message types and traits for control-plane communication:
    - cortex ↔ neuron
    - cortex ↔ cortex
  - should be transport-agnostic (no direct dependency on http, grpc, etc).

- `crates/model-runtime`
  - abstraction over concrete model runtimes.
  - defines traits such as:
    - `TextInference`
    - `ChatInference`
    - `EmbeddingInference`
    - `VisionInference`
  - neuron owns concrete adapters and wiring to processes or libraries.

- `crates/config`
  - layered configuration loader (defaults, config files, env, cli).
  - responsible for schema validation and helpful error messages.

- `crates/util`
  - shared helpers (logging setup, metrics helpers, error types).

module and trait boundaries
---------------------------

### cortex orchestrator

- inputs:
  - topology and mesh events (new neuron discovered, lost, health changes).
  - capability announcements from neurons.
  - user-facing requests arriving via the gateway.
- outputs:
  - scheduling decisions: which neuron(s) should handle a request.
  - provisioning actions: load/unload models, adjust concurrency.

core traits live in `cortex::orchestrator` and `protocol`:

- `protocol::ModelCapability`
- `protocol::WorkloadClass`
- `cortex::orchestrator::Scheduler`
- `cortex::orchestrator::Provisioner`

orchestrator should not know about concrete runtimes. it only deals in:

- model identifiers
- resource constraints (vram, cpu, bandwidth)
- historical performance metrics (latency, error rates)

### cortex gateway

- owns http/websocket termination.
- translates incoming requests into internal request types defined in `protocol`.
- delegates routing decisions to an injected `Scheduler`.
- does not directly call neuron apis using ad-hoc types.

### neuron

- implements `protocol::NeuronControl` and related traits.
- exposes a control-channel api (grpc-like or custom) for cortex.
- internally, delegates to `model-runtime` via traits, not via hard-coded commands.

request routing decisions
-------------------------

routing should be arranged as a pipeline within `cortex::gateway`:

1. `classify_request`  
   classify into a `WorkloadClass` based on:
   - endpoint (`/v1/chat/completions`, `/v1/embeddings`, etc.)
   - model name (user-specified or default)
   - request size (prompt tokens, max output tokens)

2. `select_model`  
   orchestrator chooses a concrete model id suitable for:
   - requested behaviour
   - operator policy
   - resource availability

3. `select_neuron`  
   orchestrator selects one or more target neurons based on:
   - capabilities advertised via `ModelCapability`
   - health and load reports
   - latency hints and topology awareness

4. `dispatch_request`  
   gateway calls into the selected neuron(s) via protocol traits.

agent guidelines
----------------

- new protocol-level concepts belong in `crates/protocol`.
- if you need a new kind of workload (e.g. audio transcription), extend:
  - `WorkloadClass`
  - `ModelCapability`
  - `model-runtime` traits
- never let the `helexa-cli` crate accumulate real logic.
- prefer adding new small crates over bloating existing ones when crossing domains.
- if you are unsure where a change belongs, prefer:
  - types/interfaces in `protocol` or domain crate (`cortex`, `neuron`)
  - glue code and wiring in the crate that owns the side-effect (e.g. network calls).

scaffolding and placeholders
----------------------------

when introducing new fields, structs, or modules that are not fully implemented yet, avoid leaving them as silent dead code. instead:

- **use fields in live code paths**  
  wire them into logging, routing decisions, or helper methods so that their presence is clearly intentional (e.g. log mesh node id, log models directory, etc).

- **prefer explicit `todo!()` / `unimplemented!()` over hidden no-ops**  
  if you cannot provide a real implementation yet, call `todo!()` or `unimplemented!()` from the code path that uses the field. this ensures:
  - the field cannot be accidentally relied on in production without being noticed.
  - callers see a loud failure until the behaviour is defined.

- **keep placeholders small and well-documented**  
  - keep placeholder methods minimal (log + `todo!()`/`unimplemented!()`).
  - add a concise `TODO` comment describing the intended behaviour and where the real implementation should live.

- **avoid `#[allow(dead_code)]` for long-lived scaffolding**  
  short-lived `allow` attributes are acceptable for very local work-in-progress, but for anything that is expected to live across multiple changes:
  - prefer making the usage explicit and failing loudly.
  - remove `allow` attributes as soon as the code is wired into real execution paths.

- **examples of acceptable scaffolds**  
  - a scheduler that logs the mesh node id and then returns a trivial routing decision, with a `TODO` pointing to future scheduling logic.
  - a registry method that logs model lookups and then calls `unimplemented!()` until the real model-runtime binding is wired up.
  - a control-plane handler that constructs an implementation struct, logs that it is unimplemented, and then calls `todo!()`.

this strategy keeps the workspace free of quietly ignored dead code while making it obvious which pieces are intentionally incomplete and need future work.

continuous integration and workflow expectations
-------------------------------------------------

the main branch is expected to stay buildable, formatted, and lint-clean at all times. a github workflow runs on pushes and pull requests targeting `main` and enforces a basic set of checks:

- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo test --workspace --all-features`

### what the ci does

the ci pipeline:

- checks out the repository
- installs the rust toolchain as pinned in `rust-toolchain.toml`
- runs rustfmt in check mode across the entire workspace
- runs clippy on all workspace crates and targets, treating all warnings as errors
- runs the full workspace test suite

if any of these steps fail, the workflow fails and the change cannot be merged into `main` (assuming branch protection is enabled).

### contributor workflow

to avoid breaking ci when you submit or update a change:

1. **format before committing**

   always run:

   - `cargo fmt --all`

   this should produce no diffs when re-run.

2. **lint locally with clippy**

   run:

   - `cargo clippy --workspace --all-targets --all-features`

   address all warnings rather than silencing them. if you genuinely need to silence a lint, prefer the narrowest possible scope (e.g. a single expression or function) and document why with a short comment.

3. **run the full test suite**

   before pushing:

   - `cargo test --workspace --all-features`

   add or update tests alongside your changes, and keep tests close to the code they exercise.

4. **license headers in rust files**

   - all new `*.rs` files under `crates/` must start with:

     `// SPDX-License-Identifier: PolyForm-Shield-1.0`

   - ci enforces this for pull requests and will fail with a list of offending
     files if the header is missing.

5. **keep scaffolds explicit**

   when introducing placeholders (see *scaffolding and placeholders* above):

   - ensure any new fields are used in real code paths (logging, routing decisions, or helper methods)
   - use `todo!()` / `unimplemented!()` in those paths so incomplete behaviour is obvious
   - avoid adding `#[allow(dead_code)]` or broad `allow` attributes that could hide issues from clippy

5. **iterate with ci feedback**

   when ci fails on your pull request:

   - read the failing step’s log (format, clippy, or tests)
   - reproduce the failing command locally
   - fix the underlying issue rather than muting it
   - push a new commit and wait for ci to go green

by following this workflow, contributors help ensure that `main` remains stable, consistent, and pleasant to work in for both humans and automated agents.
