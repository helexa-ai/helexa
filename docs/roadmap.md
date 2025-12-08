helexa/docs/roadmap.md
```
# helexa roadmap

This document outlines a staged implementation plan for helexa, aligned with the architecture and agent guidelines. It is written to support incremental delivery while preserving clean boundaries between crates and modules.

The roadmap is organised into phases:

1. [Phase 0 — Foundations & Skeleton](#phase-0--foundations--skeleton)
2. [Phase 1 — Minimal Single-Operator Deployment](#phase-1--minimal-single-operator-deployment)
3. [Phase 2 — Robust Control Plane & Scheduling](#phase-2--robust-control-plane--scheduling)
4. [Phase 3 — Production-Ready Gateway & Portals](#phase-3--production-ready-gateway--portals)
5. [Phase 4 — Mesh & Multi-Operator Topology](#phase-4--mesh--multi-operator-topology)
6. [Phase 5 — Ecosystem, Tooling & Operations](#phase-5--ecosystem-tooling--operations)
7. [Phase 6 — Cortex Spec & Demand-Driven Provisioning](#phase-6--cortex-spec--demand-driven-provisioning)

Each phase defines goals, scope boundaries, concrete milestones, and acceptance criteria.

---

## Phase 0 — Foundations & Skeleton

**Goal:** Establish a compilable workspace with correct crate boundaries, shared configuration, and basic wiring, but minimal business logic.

### Scope

- [x] Create core crates:
  - `helexa-cli`
  - `cortex`
  - `neuron`
  - `mesh`
  - `protocol`
  - `model-runtime`
  - `config`
  - `util`
- [x] Implement shared configuration, logging, and error primitives.
- [x] Provide no-op or stubbed implementations of the main entrypoints:
  - `cortex::run(config)`
  - `neuron::run(config)`

### Work items

1. **Workspace structure**
   - [x] Define `Cargo.toml` workspace members, features, and dependency graph.
   - [x] Ensure each crate compiles independently with minimal dependencies.
   - [x] Add `rust-toolchain.toml` (already present) and basic CI checks (fmt, clippy, test).

2. **`util` crate**
   - [x] Logging helpers:
     - Structured logging facade (e.g. via `tracing`).
     - Simple logging configuration helper (env-driven log level).
   - [ ] Error types:
     - Common error enum or error trait pattern usable across crates.
     - Helpers/macros for attaching context.
   - [ ] Metrics stubs:
     - Trait(s) for metrics emission (counters, histograms, gauges).
     - No-op implementation used by default.

3. **`config` crate**
   - [x] Config struct definitions for:
     - `CortexConfig`
     - `NeuronConfig`
     - Shared fields:
       - network sockets
       - mesh configuration
       - logging / metrics configuration
   - [ ] Layered loading:
     - Defaults
     - Config file (YAML/TOML/JSON; pick one and document)
     - Environment variables
     - CLI overrides (later passed from `helexa-cli`)
   - [ ] Validation:
     - `validate()` methods returning descriptive errors.
     - Helpful error messages for common misconfigurations (port conflicts, missing endpoints, etc).

4. **`protocol` crate (v0 skeleton)**
   - [x] Basic type definitions (no complex semantics yet):
     - `ModelId`
     - `ModelCapability` (minimal fields)
     - `WorkloadClass` (minimal set of variants)
     - `NeuronDescriptor`
   - [x] Ensure types are serialisable (`serde`) and versionable (e.g. include a protocol version constant).

5. **`model-runtime` crate (v0 traits)**
   - [x] Define core traits:
     - `ChatInference`
   - [x] Provide:
     - Trait method signatures with clear semantics (async).
     - A process-backed `ProcessRuntime` that speaks OpenAI-style HTTP chat completions.
   - [ ] Extend with:
     - `TextInference`
     - `EmbeddingInference`
     - `VisionInference`
   - [ ] Provide an in-memory / dummy implementation for testing:
       - Returns canned responses or echoes input.

6. **`helexa-cli`**
   - [x] Implement CLI parsing:
     - Subcommands: `cortex`, `neuron`.
     - Common flags: `--config`, `--log-level`, and role-specific flags.
   - [x] Wiring:
     - Load configuration via `config` crate.
     - Initialise logging via `util`.
     - Call `cortex::run(config)` or `neuron::run(config)`.

7. **`cortex` and `neuron` skeletons**
   - For `cortex`:
     - [x] Define module layout: `mesh.rs`, `orchestrator.rs`, `gateway.rs`, `portal.rs`, `shutdown.rs`, `control_plane.rs`.
     - [x] Implement a `run(config)` that:
       - Sets up a basic runtime/executor (tokio).
       - Calls stub functions for mesh, orchestrator, gateway, portal.
       - Starts the control-plane websocket server when configured.
       - Integrates a clean shutdown path (`shutdown.rs`).
   - For `neuron`:
     - [x] Define module layout: `runtime.rs`, `control_plane.rs`, `registry.rs`.
     - [x] Implement `run(config)` with:
       - Control-plane websocket client startup.
       - Stub API server startup.
       - A `ctrl_c`-based shutdown so the process stays alive.

### Acceptance criteria

- `cargo build --workspace` succeeds.
- `helexa --help`, `helexa cortex --help`, and `helexa neuron --help` work.
- Running `helexa cortex` and `helexa neuron` with default configs logs startup and shutdown messages without panics.
- Types in `protocol` and traits in `model-runtime` are documented with rustdoc.

---

## Phase 1 — Minimal Single-Operator Deployment

**Goal:** Achieve a minimal but functional deployment: one cortex, one neuron, no mesh, with a simple local control protocol and a trivial gateway.

### Scope

- [x] Implement a direct cortex ↔ neuron control channel via websocket (without full mesh).
- [ ] Implement minimal orchestration: single neuron target, static model mapping for real requests.
- [ ] Implement a minimal OpenAI-compatible gateway surface:
  - `/v1/chat/completions`
  - `/v1/embeddings`
- [x] Integrate `model-runtime` with at least one concrete backend placeholder (`ProcessRuntime` HTTP bridge).

### Work items

1. **`protocol` v1: control-plane MVP**
   - [x] Define request/response messages for:
     - Provisioning commands (`UpsertModelConfig`, `LoadModel`, `UnloadModel`).
   - [x] Define `NeuronControl` trait used by cortex/neuron boundary for provisioning.
   - [x] Define DTOs for:
     - `ModelId`, `ModelConfig`, `ProvisioningCommand`, `ProvisioningResponse`
   - [ ] Define request/response messages for:
     - `announce_capabilities`
     - `execute_request` (chat + embedding for now)
     - `report_health` (very basic health events).
   - [ ] Define DTOs for:
     - `EmbeddingRequest`, `EmbeddingResponse`

2. **`neuron` — MVP implementation**
   - `registry.rs`:
     - [x] Store an in-memory map of models to `ChatRuntimeHandle` and worker identifiers.
     - [x] Provide APIs to register and unregister chat-capable models.
   - `runtime.rs`:
     - [x] Implement adapter from provisioning commands to `ProcessManager` and `ModelRegistry`.
     - [x] Maintain a cache-backed `ModelConfigState` so neuron remembers model configs across restarts.
   - `control_plane.rs`:
     - [x] Implement a websocket client that:
       - Connects to cortex.
       - Sends `Register` and periodic `Heartbeat` messages.
       - Receives `CortexToNeuron::Provisioning` and applies them via `NeuronControlImpl`.
   - [ ] Periodically publish capabilities (if needed in this phase) or allow on-demand query.

3. **`cortex` — direct control-channel integration**
   - `orchestrator.rs`:
     - [x] Implement a naïve `Scheduler`:
       - Single hard-coded behavior selecting a default model id per `WorkloadClass`.
     - [ ] Implement `Provisioner` that uses the websocket control-plane to drive dynamic load/unload.
   - `control_plane.rs`:
     - [x] Implement a websocket server that:
       - Accepts neuron connections.
       - Tracks registrations and heartbeats in `NeuronRegistry`.
       - Receives provisioning responses.
       - Maintains per-neuron outbound senders for `CortexToNeuron` messages.
     - [x] Provide a helper `send_provisioning_to_neuron` for admin/orchestrator use.
   - `gateway.rs`:
     - [ ] Implement HTTP server with OpenAI-like endpoints:
       - `/v1/chat/completions` → translates to `ChatRequest`.
       - `/v1/embeddings` → translates to `EmbeddingRequest`.
     - [ ] Basic request classification to `WorkloadClass`:
       - `ChatInteractive` for chat.
       - `Embedding` for embeddings.
   - [ ] Implement translation between gateway types and `protocol` types.
   - [ ] Integrate a `NeuronControl`/control-plane client from cortex to neurons (for non-websocket transports, if needed later).

4. **Config & wiring**
   - `config` crate:
     - [ ] Extend config structs to include:
       - Cortex: a `--spec` path and any bootstrap policy knobs.
       - Neuron: additional runtime configuration as it grows.
   - `helexa-cli`:
     - [x] Add control-plane and role-specific flags (e.g. `--control-plane-socket`, `--cortex-control-endpoint`, `--node-id`).
     - [x] Ensure CLI flags override config for key ports.

5. **Basic observability**
   - [x] Use `util` logging helpers to:
     - Log key lifecycle events (startup, shutdown).
     - Log control-plane events (registration, heartbeat, provisioning, process spawn).
   - [ ] Log each incoming gateway request (rate-limited to avoid floods).
   - [ ] Expose a basic `/metrics` endpoint (even if only a stub) on cortex and neuron.

### Acceptance criteria

- A single cortex and single neuron can be started with documented commands.
- `curl` or OpenAI-compatible clients can call `/v1/chat/completions` and `/v1/embeddings` and receive deterministic dummy responses.
- If the neuron is down, cortex returns well-defined error responses and does not panic.
- Integration tests exist that:
  - Spin up a neuron and cortex within the same process (or test harness).
  - Exercise a small set of chat and embedding flows end-to-end.

---

## Phase 2 — Robust Control Plane & Scheduling

**Goal:** Introduce proper orchestration, richer capabilities, health tracking, and early provisioning logic, still in a single-operator context.

### Scope

- Extend `protocol` with richer concepts for capabilities and workloads.
- Implement a `Scheduler` with awareness of:
  - Model capabilities
  - Neuron health and simple load metrics
- Implement early `Provisioner` support for load/unload (even if mapped to lightweight operations initially).
- Improve `neuron` registry and health reporting.

### Work items

1. **`protocol` v2: capabilities & workloads**
   - [ ] Extend `ModelCapability` to include:
     - Workload types supported (chat, completion, embeddings, vision).
     - Max context length, throughput hints.
     - Resource hints (e.g. approximate VRAM/CPU footprint).
   - [x] Extend `WorkloadClass` with variants for:
     - `ChatInteractive`
     - `ChatBulk`
     - `Embedding`
     - `VisionCaption` (placeholder).
   - [x] Define:
     - `RoutingDecision` type containing:
       - `ModelId`
       - `NeuronDescriptor` list
       - routing mode (simple for now: single target).
   - [x] Add messages/traits for:
     - `load_model`
     - `unload_model`

2. **`neuron` — registry & health**
   - `registry.rs`:
     - [ ] Store current model states:
       - `Loaded`, `Unloaded`, `Loading`, `Unloading`, `Failed`.
     - [x] APIs for:
       - Registering and unregistering chat runtimes by model id.
       - Applying model configuration payloads received from cortex at runtime (no requirement for static per-model TOML files on disk).
   - `control_plane.rs`:
     - [x] Implement handlers for:
       - `UpsertModelConfig` (updates `ModelConfigState`).
       - `LoadModel` (spawns backend processes and registers runtimes).
       - `UnloadModel` (terminates workers and unregisters runtimes).
     - [ ] Implement `announce_capabilities` response including current model states.
     - [ ] Implement periodic `report_health` push or on-demand status.
   - Health model:
     - [ ] Track basic metrics:
       - Recent failures
       - Concurrent request count
       - Simple moving average latency.
       - Configuration- and provisioning-related health (e.g. failed model config application, failed process spawn for a given model).

3. **`cortex` — scheduler & provisioner**
   - `orchestrator.rs`:
     - [ ] Implement `Scheduler` that:
       - Uses `ModelCapability` and `WorkloadClass` to find compatible models.
       - Filters neurons by health (e.g. avoid unhealthy or overloaded).
       - Performs simple load balancing (round-robin, least-loaded, or random among healthy).
     - [ ] Implement `Provisioner` that:
       - Ensures required models are loaded on at least one neuron.
       - Drives dynamic model configuration into neurons over the websocket control channel so neurons never require a static on-disk model catalog.
       - Can pre-load configured "hot" models by sending configuration + load directives to neurons at startup.
   - [ ] Provide configuration knobs for:
     - Minimum number of replicas per model.
     - Cooldown periods for unloading unused models.

4. **Gateway improvements**
   - Improve classification:
     - Use request size (prompt tokens, `max_tokens`) to distinguish `ChatInteractive` vs `ChatBatch`.
   - Introduce basic rate limiting hooks (can be stubbed) with clear interfaces.

5. **Testing & simulation**
   - Add simulation tests or load tests that:
     - Spin up multiple logical neurons (in-process mocks).
     - Validate `Scheduler` decisions under different capabilities and health states.
   - Add property-based tests for routing decisions where possible.

### Acceptance criteria

- Cortex can manage multiple neurons and make non-trivial scheduling decisions.
- Models can be dynamically (un)loaded on neurons through the provisioner path.
- If a neuron becomes unhealthy (simulated), cortex avoids routing new requests to it.
- End-to-end tests cover:
  - Multi-neuron routing.
  - Recovery when a neuron is restarted or returns to healthy state.

---

## Phase 3 — Production-Ready Gateway & Portals

**Goal:** Enhance the external API surface, introduce portal abstraction, and prepare for multi-tenant, operator-facing deployments.

### Scope

- Harden the OpenAI-compatible gateway:
  - Streaming, timeouts, structured error handling.
- Introduce portal concept with basic multi-tenant separation.
- Implement initial hooks for authentication and billing integration.

### Work items

1. **Gateway hardening**
   - Request validation:
     - Check model names, rate limits, payload sizes.
   - Streaming support:
     - Implement streaming responses for chat/completions.
     - Ensure backpressure is handled correctly.
   - Timeouts & cancellation:
     - Per-request timeout configs.
     - Propagate cancellation to neurons where possible.
   - Error model:
     - Map internal errors to OpenAI-like error formats.
     - Provide stable error codes for clients.

2. **Portal abstraction in `cortex::portal`**
   - Define `PortalId` and `PortalConfig`:
     - Domain/host
     - Bound sockets
     - Auth configuration
     - Per-portal routing or policy hints.
   - Implement ability to run multiple HTTP servers or virtual hosts:
     - Each associated with a `PortalId`.
   - Define an abstraction for:
     - Tenant identification (API keys, headers, or OIDC claims).
     - Policy hooks:
       - Per-portal model allow/deny lists.
       - Per-portal quotas (requests/minute, tokens/day).

3. **Auth & billing hooks**
   - Design traits/interfaces for:
     - `Authenticator`
     - `Authorizer`
     - `BillingSink`
   - Provide:
     - A simple API key based authenticator implementation.
     - A no-op billing sink with structured events.

4. **Configuration & docs**
   - Extend `config` crate to support:
     - Per-portal configuration.
     - Auth and billing plugin configuration.
   - Update docs and examples to show:
     - Running multiple portals on one cortex.
     - Different API keys or policies per portal.

5. **Operational resilience**
   - Add structured access logs (gateway and portals).
   - Increase test coverage for:
     - Large payloads.
     - Streaming under load.
     - Timeout and cancellation propagation.

### Acceptance criteria

- Cortex exposes a reasonably complete OpenAI-compatible surface for chat & embeddings, including streaming.
- Operators can configure multiple portals, each with its own API keys and policies.
- Auth and billing hooks exist and can be integrated by implementers without modifying core crates.
- Documentation includes runnable examples for common deployment patterns.

---

## Phase 4 — Mesh & Multi-Operator Topology

**Goal:** Introduce the mesh overlay, enabling multiple cortex nodes (and optionally neurons) across operators, with controlled inter-operator routing.

### Scope

- Implement a mesh overlay for cortex nodes (and optional neuron participation).
- Extend `protocol` for cortex ↔ cortex communication.
- Define policies and mechanisms for cross-operator work routing.

### Work items

1. **`mesh` crate implementation**
   - Identity:
     - Key generation and management for nodes.
   - Membership:
     - Join/leave semantics.
     - Peer discovery (bootstrap nodes, static peers).
   - Messaging:
     - Reliable messaging primitive between cortex nodes.
     - Optional direct messaging to neurons (if enabled).
   - Abstract transport details:
     - Provide traits that can be implemented using different backends.

2. **Cortex ↔ cortex protocol**
   - Extend `protocol` with:
     - `CortexDescriptor` (similar to `NeuronDescriptor`).
     - Messages for:
       - Sharing neuron capabilities summaries.
       - Routing requests between cortex nodes (proxy mode).
   - Implement:
     - Simple inter-cortex routing where:
       - Local cortex can ask remote cortex to handle a request when:
         - The remote has capabilities the local does not (e.g. specific model).
         - Or policy explicitly prefers remote resources.

3. **Cortex mesh integration**
   - `cortex::mesh`:
     - Integrate with `mesh` crate:
       - Join mesh at startup.
       - Publish local capabilities summaries.
   - `orchestrator.rs`:
     - Extend scheduler to:
       - Consider both local and remote options.
       - Apply policy to prefer:
         - Local capacity when available.
         - Remote capacity when local cannot serve or is saturated.
   - Policy configuration:
     - Allow operators to configure:
       - Which remote operators are trusted.
       - What classes of work may be routed externally.
       - Limits on external usage.

4. **Security & trust primitives (initial)**
   - Define identity & auth concepts for cortex nodes:
     - Node-level identity tied to mesh keys.
   - Hooks for:
     - Verifying remote cortex identity.
     - Applying allow/deny lists.

5. **Testing & simulation**
   - Multi-cortex test harness:
     - Simulate N cortex nodes and M neurons.
     - Validate:
       - Topology discovery.
       - Cross-operator routing decisions.
       - Failure modes when mesh partitions occur.

### Acceptance criteria

- Two or more cortex nodes can form a mesh and share information about their neurons and capabilities.
- A request received by one cortex can be legally and successfully routed to another cortex, which then routes it to its neurons.
- Mesh failures (e.g. losing a peer) degrade gracefully; local cortex continues serving with local resources.
- Operators can configure strict policies to prevent unintended cross-operator routing.

---

## Phase 5 — Ecosystem, Tooling & Operations

**Goal:** Make helexa practically operable and extensible: tooling, packaging, observability, and developer experience.

### Scope

- Operational tooling for operators (debugging, introspection, deployment).
- Developer tooling and extension points.
- Documentation and examples.

### Work items

1. **Operational tooling**
   - Admin endpoints:
     - Status endpoints for cortex and neuron:
       - Current models and states
       - Neuron list and health
       - Mesh peers
   - CLI commands:
     - `helexa cortex inspect` (or similar) for:
       - Printing current routing table.
       - Printing loaded models and neurons.
     - `helexa neuron inspect` for:
       - Printing registry contents and model states.
   - Packaging:
     - Container images.
     - Systemd unit files or equivalent examples.

2. **Observability**
   - Metrics:
     - Define a metrics vocabulary:
       - Request counts, latencies, error rates, utilisation.
     - Implement metrics integration (e.g. Prometheus exporters).
   - Tracing:
     - Propagate trace IDs across:
       - Gateway
       - Cortex orchestrator
       - Neuron control-plane
       - Model runtime.
     - Provide example configs for distributed tracing backends.

3. **Developer extensibility**
   - Stable traits for plugins:
     - `Scheduler` strategies (e.g. pluggable via features or config).
     - `Provisioner` heuristics.
     - Custom `model-runtime` backends.
   - Document how to:
     - Add a new workload type (e.g. audio transcription):
       - Extend `WorkloadClass`.
       - Extend `ModelCapability`.
       - Implement new traits in `model-runtime`.
       - Wire into `gateway` and `neuron`.

4. **Examples and reference deployments**
   - Minimal local demo:
     - Single cortex + single neuron + dummy models.
     - Documentation with copy-pasteable commands.
   - More advanced demo:
     - Multiple neurons
     - Multiple portals with distinct keys
     - Optional mesh with two cortex nodes.

5. **Hardening & polish**
   - Run sustained load tests and record recommended production tunings.
   - Use static analysis and fuzzing where appropriate (especially around protocol parsing).
   - Stabilise protocol version and document compatibility guarantees.

### Acceptance criteria

- Operators can deploy helexa with clear documentation, collect metrics, and debug issues using provided tooling.
- Developers can extend the system with new workloads and runtime backends following documented patterns.
- The system behaves reliably under realistic load for extended periods in test environments.

---

## Phase 6 — Cortex Spec & Demand-Driven Provisioning

**Goal:** Allow cortex to bootstrap its understanding of model demand and provisioning policy from a declarative “spec” file (similar to chainspecs in some blockchains), and evolve that understanding at runtime based on observed traffic and metrics. Persist this demand state to the filesystem cache so it survives restarts.

### Scope

- [ ] Define a `CortexSpec` JSON schema that can be passed via `--spec /path/to/spec.json`:
  - Includes:
    - Initial `ModelConfig` entries (per model).
    - Optional per-model weights / priorities.
    - Optional per-model min/max replica hints.
    - Optional global policy hints (e.g. max concurrent models per neuron).
- [ ] Extend `cortex::Config` and CLI to accept `--spec` and load it at startup.
- [ ] Introduce a cache-backed demand state in cortex:
  - Backed by the `cache` crate.
  - Stores:
    - Per-model demand metrics (e.g. smoothed request rates).
    - Learned capacity / concurrency hints per backend + environment.
  - Seeded from the spec on first run, then updated at runtime.
- [ ] Integrate the spec- and metrics-backed demand state into the `Provisioner`:
  - At startup, when neurons register, use the spec to:
    - Decide which models to `UpsertModelConfig` and `LoadModel` on which neurons.
  - At runtime:
    - Adjust load/unload decisions based on:
      - Incoming traffic.
      - Historical performance.
      - Capacity metrics reported by neurons.
- [ ] Document the distinction between:
  - `--spec` (bootstrapping and long-lived policy/demand hints).
  - Runtime configuration and metrics (dynamic, may override or refine spec guidance).

---

## Contribution guidelines alignment

Throughout all phases:

- Follow `agents.md` guidelines:
  - Keep responsibilities narrow per crate.
  - Place shared message types and traits in `protocol`.
  - Keep `helexa-cli` free of business logic: it only parses, configures, and delegates.
- Prefer adding small, well-scoped crates/modules rather than bloating existing ones.
- Co-locate tests with the code they exercise.
- Maintain explicit and well-documented public APIs, especially for traits and protocol types.

This roadmap is intentionally incremental: each phase can be delivered and released independently while building towards a full multi-operator, production-ready distributed AI fabric.