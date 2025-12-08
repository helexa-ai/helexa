architecture
============

overview
--------

helexa is a distributed ai fabric built from two primary node roles:

- **cortex** — control-plane node that:
  - always participates in the p2p mesh
  - may act as an orchestrator
  - may act as an api gateway
  - may host one or more portals (front-end + billing)
  - exposes a websocket-based control-plane endpoint that neurons connect to
- **neuron** — data-plane node that:
  - runs one or more model runtimes
  - exposes capabilities and health to cortex
  - executes inference work routed by cortex
  - connects to cortex over a websocket control-plane client for:
    - registration and identity
    - periodic heartbeats and lightweight metrics
    - dynamic provisioning commands (model config updates, load/unload)

the system is designed so that a single cortex and a single neuron can form a minimal,
fully functional deployment, while also scaling out to many operators and nodes.

crates and boundaries
---------------------

### helexa-cli

- binary entrypoint (`helexa`).
- exposes subcommands:
  - `helexa cortex`
  - `helexa neuron`
- parses cli flags and config files.
- constructs configuration objects and delegates to:
  - `cortex::run(config)`
  - `neuron::run(config)`

### cortex

- owns the control-plane for a particular operator or site.
- joins the mesh via the `mesh` crate.
- maintains knowledge of:
  - local neurons (same operator)
  - remote neurons accessible through peer cortex nodes
- exposes a websocket-based control-plane endpoint that neurons connect to.
- key modules:

  - `mesh.rs`
    - starts and manages participation in the p2p overlay.
    - exposes a `MeshHandle` for sending and receiving control-plane messages.

  - `orchestrator.rs`
    - contains orchestration logic:
      - scheduling
      - placement
      - provisioning (load/unload models on neurons)
    - uses types and traits from `protocol` to reason about:
      - model capabilities
      - workloads
      - resource constraints.

  - `gateway.rs`
    - owns the openai-compatible http api surface.
    - performs request classification into workload classes.
    - asks the orchestrator for routing decisions.
    - dispatches requests to neuron(s) via the control-plane protocol or directly to
      neuron HTTP endpoints, depending on the chosen transport for inference.

  - `portal.rs`
    - serves front-end portals over http(s).
    - integrates with billing and account systems (to be defined).
    - may host multiple portals on different sockets / domains for white-label use.

  - `shutdown.rs`
    - centralised graceful shutdown handling for all roles.

### neuron

- runs model runtimes, provides inference services.
- learns about model configurations and provisioning directives dynamically from cortex
  over the websocket control-plane.
- key modules:

  - `runtime.rs`
    - integrates with the `model-runtime` crate.
    - manages process lifecycles for external runtimes where needed.
    - exposes a simple interface to execute inference requests.

  - `control_plane.rs`
    - implements the protocol required by cortex.
    - listens on a control socket for:
      - scheduling assignments
      - provisioning commands (load/unload model)
      - dynamic model configuration updates received from cortex
      - health checks
    - does **not** require a preconfigured, on-disk model catalog at startup.
      all model definitions and process wiring information are learned at runtime
      from cortex over the control channel (e.g. websocket or similar transport).

  - `registry.rs`
    - tracks locally available models, their states and capabilities.
    - is populated and updated by configuration and provisioning messages from cortex
      rather than static files (e.g. no requirement for per-model TOML files on disk).
    - periodically publishes capability summaries to cortex via protocol messages.

### mesh

- abstract p2p overlay.
- provides:

  - node identity
  - discovery (peer set)
  - message routing (cortex ↔ cortex, optionally cortex ↔ neuron)

- hides the details of the underlying transport (libp2p, quic, rosenpass, etc).

### protocol

- shared data types and traits for:

  - cortex ↔ neuron control-plane
  - cortex ↔ cortex interactions

- must be transport-agnostic:

  - serialisable with serde
  - usable over different transports (http, grpc, quic, etc)

- currently defines:
  - `ModelId`, `ModelConfig`, `ModelCapability`, `WorkloadClass`.
  - provisioning-related enums:
    - `ProvisioningCommand` (e.g. `UpsertModelConfig`, `LoadModel`, `UnloadModel`).
    - `ProvisioningResponse`.
  - a `NeuronControl` trait that neuron implements to apply provisioning commands
    in-process, while the websocket transport is responsible for carrying
    `CortexToNeuron` / `NeuronToCortex` messages.

### model-runtime

- abstraction layer over concrete model backends.

- core traits:

  - `TextInference`
  - `ChatInference`
  - `EmbeddingInference`
  - `VisionInference`

- neuron owns concrete implementations for each backend and maps models in the
  registry to one or more `model-runtime` implementations.

control-plane protocol
----------------------

protocol traits and types are defined in `crates/protocol`. key concepts:

- `ModelId`
  - logical identifier for a model (opaque string plus optional metadata).

- `ModelCapability`
  - describes what a model can do:
    - workload types (chat, completion, embedding, vision)
    - performance hints (max context length, throughput estimates)
    - resource requirements (approximate vram / cpu usage)

- `WorkloadClass`
  - classification of a request by behaviour and resource footprint:
    - e.g. `ChatInteractive`, `ChatBulk`, `Embedding`, `VisionCaption`, etc.

- `ModelConfig`
  - describes how to start and talk to a model backend:
    - `backend_kind` (e.g. `"vllm"`, `"llama_cpp"`, `"openai_proxy"`).
    - `command`, `args`, `env` for spawning backend processes.
    - optional `listen_endpoint` (base URL for OpenAI-style HTTP).
    - free-form `metadata` for backend-specific parameters.

- `NeuronDescriptor`
  - high-level description of a neuron node:
    - network endpoints
    - operator metadata
    - cost / pricing hints (if published)

- `Scheduler` (trait, implemented by cortex)
  - given a `WorkloadClass` and current state (capabilities + health), chooses:
    - `ModelId`
    - target `NeuronDescriptor`(s)

- `Provisioner` (trait, implemented by cortex)
  - responsible for ensuring that necessary models are loaded on neurons.
  - will use the websocket control-plane to:
    - push `ModelConfig` values to neurons (`UpsertModelConfig`).
    - request `LoadModel` and `UnloadModel` for specific models on specific neurons.

- `NeuronControl` (trait, implemented by neuron)
  - operations invoked by cortex:
    - `announce_capabilities`
    - `load_model`
    - `unload_model`
    - `execute_request`
    - `report_health`

request routing flow
--------------------

1. **ingress (gateway)**

   - a request arrives at the cortex gateway on `/v1/chat/completions` or similar.
   - the gateway:
     - parses the request
     - extracts user-requested model (if specified)
     - constructs an internal `WorkloadClass` using helpers from `protocol`.

2. **scheduling (orchestrator)**

   - the gateway calls into the orchestrator:

     ```text
     Scheduler::schedule(workload, user_hints) -> RoutingDecision
     ```

   - `RoutingDecision` contains:
     - selected `ModelId`
     - one or more `NeuronDescriptor` targets
     - routing mode (single, replicated, sharded — to be defined later).

3. **dispatch (gateway + protocol)**

   - the gateway uses `NeuronControl` to send the request to the chosen neuron.
   - the neuron executes the request via `model-runtime` traits.
   - results are streamed back to gateway and then to the client.

4. **feedback (health & metrics)**

   - latency, errors and utilisation are collected by cortex.
   - orchestrator updates its internal state so future decisions can use:
     - rolling latency histograms
     - error rates
     - capacity utilisation signals.
   - neurons can also report provisioning-related outcomes back to cortex via
     `NeuronToCortex::ProvisioningResponse`, allowing the provisioner to react
     to failed spawns, unsupported configs, etc.

multi-portal support
--------------------

a cortex node may expose multiple portal sockets:

- each portal can represent a branded surface or tenant.
- all portals share the same gateway and orchestrator.
- policy (quotas, pricing, routing preferences) can be per-portal.

this is implemented by:

- running multiple http servers in `cortex::portal` on different sockets.
- associating each portal with:
  - an identifier
  - a set of routes
  - optional auth / billing hooks.

minimal deployment
------------------

- **single cortex + single neuron** (same machine or different machines):
  - neuron connects to the cortex control-plane websocket endpoint.
  - neuron resolves its identity via:
    - CLI `--node-id` if supplied, otherwise
    - `/etc/machine-id` (or fails fast if neither is available).
  - cortex runs mesh + orchestrator + gateway + portal + control-plane server.
  - mesh can operate in a degenerate mode with a single node.

- **multi-operator deployment**:
  - each operator runs one or more cortex nodes.
  - mesh connects cortex nodes together.
  - neurons primarily talk to their operator's cortex, which may proxy work
    to other operators via inter-cortex protocol extensions.

open questions
--------------

- exact transport(s) for:
  - cortex ↔ neuron (grpc, quic, custom).
  - cortex ↔ cortex.
- runtime integration strategies (embedded libraries vs subprocesses).
- how pricing and billing metadata travels over the protocol.
- multi-operator trust and slashing mechanisms, if any.

these are left as future design work once the basic scaffolding is proven.
