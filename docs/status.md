# helexa status — websocket control-plane and provisioning

_last updated: websocket control-plane handshake working between cortex and neuron_

This document describes the current state of in-flight work around:

- the **websocket-based control-plane** between cortex and neuron
- **dynamic model provisioning** (UpsertModelConfig / LoadModel / UnloadModel)
- related runtime and process management behaviour

It is intended to help contributors understand what is done, what is scaffolded, and what remains to be wired up or cleaned up.

---

## Overview

The current direction is:

- **cortex** exposes a **websocket endpoint** that neurons connect to:
  - neurons register themselves,
  - send heartbeats and metrics,
  - receive provisioning commands (e.g. load/unload model).
- **neuron** runs a **websocket client**:
  - connects to cortex on startup,
  - sends a registration message and periodic heartbeats,
  - listens for control messages and applies them via `NeuronControlImpl`.
- **model provisioning** is driven by `protocol::ProvisioningCommand`:
  - `UpsertModelConfig(ModelConfig)`
  - `LoadModel { model_id }`
  - `UnloadModel { model_id }`
- **runtime and process management**:
  - `RuntimeManager` holds:
    - a `ModelRegistry` for `model_id -> ChatRuntimeHandle`,
    - a `ProcessManager` that spawns and tracks backend processes per model,
    - a cache-backed `ModelConfigState` (per-node config store),
  - `LoadModel`:
    - spawns a backend process,
    - constructs a `ProcessRuntime` that talks HTTP to the backend,
    - registers a chat runtime handle in the registry.
  - `UnloadModel`:
    - terminates backend workers for the model,
    - unregisters the model from the registry.

The **design principle** is that cortex is “smart” and neuron is deliberately “dumb but robust”:

- cortex decides **what** to run and **how** (command, args, env, backend_kind),
- neuron faithfully executes those directives, chooses ports, and exposes a simple internal abstraction over HTTP-backed runtimes.

---

## Cortex control-plane (websocket server)

### Implemented

- New module: `cortex::control_plane` (code exists and is mostly wired).
- `cortex::Config` has:
  - `control_plane_socket: Option<SocketAddr>` — address for the websocket listener.
- CLI support:
  - `helexa cortex --control-plane-socket 0.0.0.0:9040` starts the websocket server.

### Behaviour (current)

- Cortex binds a TCP listener on `control_plane_socket`.
- Each incoming TCP connection is upgraded to a websocket.
- Cortex expects the first message to be:

  ```json
  { "kind": "register", "neuron": { "node_id": "...", "label": "...", "metadata": { ... } } }
  ```

- Cortex maintains a simple in-memory `NeuronRegistry`:

  - `ConnectedNeuron`:
    - `descriptor: NeuronDescriptor`
    - `last_heartbeat: Instant`
  - `NeuronRegistry`:
    - `upsert_neuron(descriptor)`
    - `update_heartbeat(neuron_id)`
    - `prune_stale(timeout)`

- Cortex currently:

  - Example log of a successful handshake and registration:

    ```text
    control-plane accepted TCP connection from 127.0.0.1:56878 on 0.0.0.0:9040
    attempting websocket upgrade for neuron control-plane connection from 127.0.0.1:56878
    neuron connection successfully upgraded to websocket from 127.0.0.1:56878
    cortex received first websocket message from neuron peer 127.0.0.1:56878: Text("{\"kind\":\"register\",...}")
    registered neuron_id=neuron-1 from 127.0.0.1:56878
    ```

- Cortex can receive:
  - `heartbeat` messages with arbitrary metrics:

    ```json
    {
      "kind": "heartbeat",
      "neuron_id": "...",
      "metrics": { ... }
    }
    ```

  - `provisioning_response` messages with results of provisioning commands.

- A background task periodically prunes stale neurons based on `last_heartbeat`.

### Not yet implemented / open items

- **Sending provisioning commands** from cortex to neuron:

  - Message form will be:

    ```json
    { "kind": "provisioning", "cmd": { ...ProvisioningCommand... } }
    ```

  - The send path is not yet fully implemented: the server currently only reads and logs incoming messages.

- **Capability requests**:

  - `CortexToNeuron::RequestCapabilities` exists in the type, but cortex is not yet sending these, and neuron currently only logs them.

- **Integration with orchestrator/provisioner**:

  - `NeuronRegistry` is not yet surfaced into `cortex::orchestrator`.
  - Provisioning responses are only logged; they are not yet fed into any scheduling/provisioning strategy.

- **Refinement of registry structure**:

  - `NeuronRegistry` is a simple `Vec` inside a `RwLock`.
  - As the system grows, this may migrate to an `Arc`-based or map-based structure keyed by `node_id`.

---

## Neuron control-plane (websocket client)

### Implemented (current)

- `neuron::Config` now includes:

  - `cortex_control_endpoint: String` — URL of cortex’s websocket control-plane endpoint (e.g. `ws://127.0.0.1:9040`).

- CLI support:

  - `helexa neuron --cortex-control-endpoint ws://127.0.0.1:9040 ...`

- On startup, `neuron::run`:

  - Constructs a `RuntimeManager` (registry + process manager + config store).
  - Calls `control_plane::spawn(_, runtime.clone())`:
    - The `_addr` is currently unused in the new design (legacy placeholder), since the neuron now acts as a **client** of cortex rather than hosting a control-plane server.

- Neuron process lifetime:
  - After starting the control-plane client and the (stub) API server, the neuron now waits on a shutdown signal (`ctrl_c`) before exiting.
  - This ensures websocket connections remain active and handshakes can complete instead of the process exiting immediately.

- `NeuronControlImpl` implements `protocol::NeuronControl`:

  - `apply_provisioning(&self, cmd: ProvisioningCommand) -> ProvisioningResponse`
  - Internally, routes to:
    - `handle_upsert_model_config`
    - `handle_load_model`
    - `handle_unload_model`

- Websocket client logic (neuron → cortex):

  - Connects to `cortex_control_endpoint` via `tokio-tungstenite`.
  - On connect:
    - Sends a `register` message with a `NeuronDescriptor`:

      ```json
      {
        "kind": "register",
        "neuron": {
          "node_id": "<config.node_id / or anonymous>",
          "label": "<same as node_id for now>",
          "metadata": { "backend": "neuron" }
        }
      }
      ```

    - Spawns a heartbeat task that periodically sends:

      ```json
      {
        "kind": "heartbeat",
        "neuron_id": "<node_id or anonymous>",
        "metrics": {}
      }
      ```

  - Main receive loop:
    - Receives text frames, decodes as `CortexToNeuron`:

      - `provisioning { cmd: ProvisioningCommand }`:
        - Calls `NeuronControlImpl::apply_provisioning(cmd)`.
        - Wraps the result into a `provisioning_response` and sends it back to cortex.
      - `request_capabilities`:
        - Logged as a TODO; capability reporting is not implemented yet.

### Not yet implemented / open items
- The websocket client code currently uses a split sink/stream; the send/receive wiring has been refactored to:
 - Use a single writer task that owns the websocket sink.
 - Use an internal unbounded channel for all outbound messages (register, heartbeats, provisioning responses).
 - Use the split receive half exclusively for incoming messages from cortex.

- TODOs remain around:

  - Use `SinkExt` and `StreamExt` consistently.
  - Ensure heartbeats and responses don’t interfere with the main receive loop.
- Capability reporting (e.g. listing models, backends, approximate capacity) is not yet implemented; `RequestCapabilities` events are logged, not actioned.

- Error handling / reconnection strategy is minimal:

  - On websocket error or close, the client logs and exits.
  - It does not currently attempt reconnection.

---

## Neuron runtime and provisioning semantics

### Model configuration state (dynamic, cortex-driven)

- `protocol::ModelConfig` describes what cortex knows about a model:

  - `id: ModelId`
  - `backend_kind: String` (e.g. `"vllm"`, `"llama_cpp"`)
  - `command: Option<String>`
  - `args: Vec<String>`
  - `env: Vec<EnvVar>`
  - `listen_endpoint: Option<String>`
  - `metadata: serde_json::Value`
  - Plus some optional display/metadata fields.

- `RuntimeManager` owns:

  - `ModelConfigState` — in-memory `HashMap<String, ModelConfig>`.
  - `JsonStore` (`neuron-model-configs.json`) under `${HOME}/.cache/helexa/`.

- On startup:

  - `ModelConfigState` is loaded from cache; this lets neurons “remember” configs between restarts (though cortex remains the source of truth).

- `UpsertModelConfig`:

  - `NeuronControlImpl::handle_upsert_model_config` updates `ModelConfigState` in memory.
  - Does not yet auto-persist; `RuntimeManager::persist_model_config_state()` is available to be called by higher layers (e.g. on shutdown or after a batch of changes).

### Process management

- `neuron::process::ProcessManager`:

  - Spawns external processes via `Command` using `command`, `args`, `env` from `ModelConfig`.
  - Tracks workers by PID and model id.
  - Supports:
    - `spawn_worker_with_env(cmd, args, model_id, env) -> WorkerHandle`
    - `terminate_workers_for_model(model_id: &str)`
    - `terminate_worker_by_pid(pid)`, etc.

- `RuntimeManager` adds a naive backend port allocator:

  - Starts from `9100`.
  - `allocate_backend_port()` increments monotonically.
  - `derive_listen_endpoint(&ModelConfig)`:
    - If `listen_endpoint` is provided, use it directly.
    - Else:
      - For `backend_kind == "vllm"` or `"llama_cpp"`:
        - Choose a port, return `http://127.0.0.1:<port>`.
      - Otherwise, error.

  - **Note:** neuron currently does not yet modify the `args` to append `--host`/`--port` for these backends; it only derives the URL. Aligning the actual backend invocation (so that the process listens on the derived port) is a TODO.

### HTTP-backed runtime (`ProcessRuntime`)

- `model-runtime::ProcessRuntime` implements `ChatInference` by calling OpenAI-style `/v1/chat/completions`:

  - Uses `reqwest::Client` with a configured timeout.
  - Request shape is a minimal OpenAI-compatible body built from `ChatRequest`.
  - Response:
    - Expects `choices[0].message.content`.

This is the abstraction neuron uses per loaded model.

### Load / Unload semantics

- `LoadModel`:

  - Validates that `ModelConfig` exists (requires a prior `UpsertModelConfig`).
  - Calls `derive_listen_endpoint` to get a base URL.
  - Ensures `command` is present; otherwise returns an error.
  - Spawns the backend process via `ProcessManager::spawn_worker_with_env`.
  - Logs:

    - `model_id`, `backend_kind`, `pid`.

  - Constructs a `ProcessRuntime` with:
    - `base_url = derived listen URL`,
    - `timeout = 30s` (currently hard-coded),
    - `model = Some(model_id_string)`.

  - Wraps this in a `ChatRuntimeHandle` and registers it with `ModelRegistry` under `model_id`.

- `UnloadModel`:

  - Calls `ProcessManager::terminate_workers_for_model(model_id)`.
  - Calls `ModelRegistry::unregister_chat_model(model_id)` to prevent new routing to that model.
  - Leaves `ModelConfigState` intact (model config remains known, but not actively served).

### Not yet implemented / open items

- **Appending host/port flags to backend commands**:

  - Right now, neuron does not modify `args`; it assumes cortex has configured processes to listen in a compatible way.
  - For vLLM & llama.cpp, we likely want neuron to:
    - Inspect `backend_kind`,
    - Extend `args` with `--host 127.0.0.1 --port <allocated_port>` or backend-specific equivalents,
    - Then spawn the process.

- **Metrics emission**:

  - No metrics sink or flow back to cortex is wired yet.
  - Future work:
    - Add a `MetricsSink` abstraction,
    - Emit per-request latency and error metrics from `ProcessRuntime::chat` (or a wrapper),
    - Forward those metrics to cortex to inform capacity decisions.

- **Graceful in-flight completion**:

  - Current `UnloadModel` terminates processes using `kill`.
  - There is no per-request in-flight tracking or graceful shutdown handshake with the backend processes.
  - For now, this is acceptable scaffolding; future refinements should:
    - Support graceful backend shutdown,
    - Or track in-flight work per model before termination.

---

## Compilation state and known issues (to be fixed next)

At the time of this status snapshot, the repository still has some **compile-time and clippy issues** due to the breadth of recent changes:

- **Ownership / move issues** in `neuron::run` vs `RuntimeManager::new`:
  - `RuntimeManager::new` currently takes ownership of `Config`, but `neuron::run` still uses `config` after the call (e.g. for `api_socket`).
  - Resolution options:
    - Make `Config: Clone` and pass `config.clone()` into `RuntimeManager::new`.
    - Or change `RuntimeManager::new` to take a reference or a subset of the config instead.

- **Duplicate methods**:
  - `RuntimeManager` currently has duplicate `registry()` method definitions that must be consolidated.

- **Missing imports and trait bounds**:
  - Cortex control-plane:
    - Needs to import `protocol::ProvisioningCommand`.
  - Neuron control-plane client:
    - Needs `futures::SinkExt` imported to use `.send()`.
    - Needs consistent usage of `tx` (sink) and `rx` (stream) rather than mixing them.

- **Neuron websocket client read/write split**:
  - The current code splits the websocket into sink/stream and then reuses the sink for reading in the main loop.
  - This must be refactored so:
    - One task handles heartbeats (using the sink),
    - Another handles incoming messages (using the stream),
    - Or a small internal abstraction coordinates them cleanly.

- **Cortex NeuronRegistry cloning**:
  - The current use of `NeuronRegistry::clone()` is inconsistent; instead:
    - `NeuronRegistry` should be wrapped in an `Arc`, and that should be cloned where needed, or
    - A simpler pattern for sharing the registry should be adopted.

These are primarily mechanical and structural issues; the underlying design is in place and can be made to compile once we (a) fix ownership and imports, and (b) simplify the concurrency model in the control-plane client/server.

---

## Next steps

The immediate next steps to get the workspace compiling and CI-clean are:

1. **Stabilise neuron config ownership** — **DONE**:
   - `neuron::Config` now derives `Clone`.
   - `RuntimeManager::new` receives a cloned config so `neuron::run` can still use the original for sockets.

2. **Clean up `RuntimeManager` API** — **DONE**:
   - Only one `registry()` accessor remains.
   - Unused locals have been removed or explicitly marked where intentional.

3. **Fix cortex `control_plane` imports and registry semantics** — **MOSTLY DONE**:
   - `protocol::ProvisioningCommand` is imported where needed.
   - `NeuronRegistry` is backed by `Arc<RwLock<...>>` and clone helpers were simplified.
   - A small `registry_list_clone` helper remains but now just clones the inner `Arc` instead of copying data.

4. **Refactor neuron websocket client** — **DONE**:
   - `SinkExt` is imported and `.send()` is used via a single dedicated writer task.
   - Outbound messages (register, heartbeats, provisioning responses) are funneled through an internal `mpsc::unbounded_channel<Message>`.
   - The receive half of the websocket is used exclusively for incoming messages from cortex.

5. **Manual integration harness**:
   - Current manual test pattern:
     - Start cortex with `--control-plane-socket`:
       - Observe logs for:
         - `cortex control-plane websocket listening on ...`
     - Start neuron with `--cortex-control-endpoint ws://...` and `--node-id ...`:
       - Observe logs for:
         - `neuron successfully completed websocket handshake with cortex at ...`
         - `neuron websocket connected to cortex control-plane`
     - On cortex:
       - Observe:
         - `control-plane accepted TCP connection from ...`
         - `neuron connection successfully upgraded to websocket from ...`
         - `cortex received first websocket message from neuron peer ...: Text("{\"kind\":\"register\",...}")`
         - `registered neuron_id=...`
         - periodic `heartbeat from neuron_id=... metrics={}`.
   - A more formal test harness for provisioning commands remains future work.

Once these fixes are in place, the project should:

- Compile successfully,
- Run clippy clean with the existing CI configuration, and
- Have a functional (if minimal) websocket control-plane path between cortex and neuron ready to be iterated on for capabilities, metrics, and smarter provisioning logic.