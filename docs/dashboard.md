# helexa dashboard websocket protocol

This document describes the initial websocket protocol exposed by a cortex node for use by operator dashboards and other observability clients.

For now there is a single dashboard-oriented websocket endpoint:

- `ws://<cortex-host>:<dashboard-socket>` (the exact port is configured via `--dashboard-socket`)

This endpoint is **read-only** from cortex’s perspective:

- Clients receive:
  - A **snapshot** of current state immediately after connection.
  - A **stream of events** describing control-plane activity between cortex and neurons.
- Clients are free to send messages, but current implementations ignore them (they are only used to detect disconnects). Future work will add operator commands on this channel.

The protocol is JSON-based. All messages are encoded as UTF‑8 text frames.

---

## 1. Connecting to the dashboard websocket

### URL

Given a cortex process started with:

```bash
helexa cortex \
  --dashboard-socket 0.0.0.0:9051 \
  ...
```

Dashboard clients should connect to:

- `ws://<cortex-host>:9051`

For example:

```ts
const ws = new WebSocket("ws://localhost:9051");
```

No query parameters or subprotocol negotiation are required at this time.

---

## 2. Message envelope

All outbound messages from cortex to the client share a common envelope type:

```json
{
  "kind": "<string>",
  ...
}
```

Where:

- `kind` is one of:
  - `"snapshot"` — the initial state, sent exactly once per connection.
  - `"event"` — one or more events streamed after the snapshot is sent.

### 2.1 Snapshot message

This is the first message the client will receive after the websocket is successfully upgraded.

```json
{
  "kind": "snapshot",
  "snapshot": {
    "neurons": [
      {
        "node_id": "785b03f697304c88baf539e50e15e44a",
        "label": "785b03f697304c88baf539e50e15e44a",
        "metadata": {
          "backend": "neuron"
        }
      }
    ]
  }
}
```

#### Shape

```ts
type ObserveSnapshot = {
  neurons: NeuronDescriptor[];
};

type SnapshotMessage = {
  kind: "snapshot";
  snapshot: ObserveSnapshot;
};
```

Where:

```ts
type NeuronDescriptor = {
  node_id: string | null;  // machine-id or CLI --node-id, if known
  label: string | null;    // human-friendly label; currently same as node_id
  metadata: any;           // reserved for OS/arch/GPU/etc; currently minimal
};
```

### 2.2 Event message

After the snapshot, cortex sends a stream of `event` messages:

```json
{
  "kind": "event",
  "event": {
    "type": "<event-type>",
    ...
  }
}
```

#### Shape

```ts
type EventMessage = {
  kind: "event";
  event: ObserveEvent;
};
```

---

## 3. Event types

`ObserveEvent` is a tagged union (discriminated by `type`):

```ts
type ObserveEvent =
  | { type: "neuron_registered"; neuron: NeuronDescriptor }
  | { type: "neuron_heartbeat"; neuron_id: string; metrics: any }
  | { type: "provisioning_sent"; neuron_id: string; cmd: ProvisioningCommand }
  | {
      type: "provisioning_response";
      neuron_id: string;
      response: ProvisioningResponse;
    };
```

Each event type is described below.

### 3.1 `neuron_registered`

Emitted whenever cortex registers a neuron via the control-plane websocket (first register, or re‑register with updated metadata).

#### Example

```json
{
  "kind": "event",
  "event": {
    "type": "neuron_registered",
    "neuron": {
      "node_id": "785b03f697304c88baf539e50e15e44a",
      "label": "785b03f697304c88baf539e50e15e44a",
      "metadata": {
        "backend": "neuron"
      }
    }
  }
}
```

#### Schema

```ts
type NeuronRegisteredEvent = {
  type: "neuron_registered";
  neuron: NeuronDescriptor;
};
```

### 3.2 `neuron_heartbeat`

Emitted when cortex receives a `Heartbeat` message from a neuron via the control-plane websocket.

#### Example

```json
{
  "kind": "event",
  "event": {
    "type": "neuron_heartbeat",
    "neuron_id": "785b03f697304c88baf539e50e15e44a",
    "metrics": {}
  }
}
```

The `metrics` field is a free-form JSON object; current implementations send `{}` but future versions may include:

- basic load indicators,
- error counters,
- resource utilisation hints.

#### Schema

```ts
type NeuronHeartbeatEvent = {
  type: "neuron_heartbeat";
  neuron_id: string;
  metrics: any; // JSON-serialisable object
};
```

### 3.3 `provisioning_sent`

Emitted whenever cortex sends a provisioning command to a neuron (e.g., `UpsertModelConfig`, `LoadModel`, `UnloadModel`) over the control-plane websocket.

This includes **bootstrap provisioning** driven by the spec/demand state as well as future manual or algorithmic provisioning commands.

#### Example

```json
{
  "kind": "event",
  "event": {
    "type": "provisioning_sent",
    "neuron_id": "785b03f697304c88baf539e50e15e44a",
    "cmd": {
      "kind": "upsert_model_config",
      "config": {
        "id": { "0": "example-chat-model" },
        "display_name": "Example Chat Model (vLLM)",
        "backend_kind": "vllm",
        "command": "uvx",
        "args": [
          "--python",
          "3.13",
          "vllm@latest",
          "serve",
          "--model",
          "QuantTrio/Qwen3-Coder-30B-A3B-Instruct-GPTQ-Int8"
        ],
        "env": [],
        "listen_endpoint": null,
        "metadata": {}
      }
    }
  }
}
```

#### Schema

```ts
type ProvisioningSentEvent = {
  type: "provisioning_sent";
  neuron_id: string;
  cmd: ProvisioningCommand;
};
```

Where `ProvisioningCommand` mirrors the control-plane protocol:

```ts
type ProvisioningCommand =
  | { kind: "upsert_model_config"; config: ModelConfig }
  | { kind: "load_model"; model_id: ModelId }
  | { kind: "unload_model"; model_id: ModelId };

type ModelId = { 0: string }; // current serde shape for ModelId(pub String)

type ModelConfig = {
  id: ModelId;
  display_name: string | null;
  backend_kind: string;      // e.g. "vllm", "llama_cpp", "openai_proxy"
  command: string | null;    // program to exec (e.g. "uvx", "llama-server")
  args: string[];            // argv[] for the command
  env: { key: string; value: string }[];
  listen_endpoint: string | null; // base URL, if provided; otherwise derived
  metadata: any;                  // backend-specific configuration
};
```

> Note: The `ModelId` encoding (`{ "0": "example-chat-model" }`) reflects the current `ModelId(pub String)` serde derivation. A more ergonomic JSON mapping may be introduced later; clients should treat it as an opaque identifier for now.

### 3.4 `provisioning_response`

Emitted whenever cortex receives a `ProvisioningResponse` from a neuron (acknowledging or rejecting a provisioning command).

#### Example

```json
{
  "kind": "event",
  "event": {
    "type": "provisioning_response",
    "neuron_id": "785b03f697304c88baf539e50e15e44a",
    "response": {
      "Ok": {
        "model_id": { "0": "example-chat-model" },
        "message": "configuration updated"
      }
    }
  }
}
```

The exact shape of `ProvisioningResponse` mirrors the Rust enum:

```rust
pub enum ProvisioningResponse {
    Ok { model_id: ModelId, message: Option<String> },
    Error { model_id: ModelId, error: String },
}
```

Under serde’s default enum encoding, the JSON will look like:

```json
{ "Ok": { "model_id": { "0": "..." }, "message": "..." } }
```

or:

```json
{ "Error": { "model_id": { "0": "..." }, "error": "..." } }
```

#### Schema

```ts
type ProvisioningResponseEvent = {
  type: "provisioning_response";
  neuron_id: string;
  response: any; // serde-encoded enum: { Ok: {...} } | { Error: {...} }
};
```

Clients should:

- Detect whether `response.Ok` or `response.Error` is present.
- Extract `model_id` and `message`/`error` fields accordingly.

---

## 4. Client responsibilities and expectations

### 4.1 Connection lifecycle

- Connect to `ws://<cortex-host>:<dashboard-socket>`.
- Expect the first message to be a `snapshot`.
- Then process `event` messages until:
  - The server closes the connection,
  - The client closes it, or
  - An error occurs.

If the connection drops, the client should:

- Consider its local view stale, and
- Reconnect to rebuild state via a new snapshot.

### 4.2 Message ordering and de-duplication

- Snapshot reflects the state at the moment of connection.
- Events are **not guaranteed to be strictly ordered across connections**.
- Within a single connection, order reflects the order seen by cortex, but network delays or reconnects may cause replays or gaps.
- Clients should be resilient:
  - Use `neuron_id` and `model_id` as stable identifiers.
  - Treat events as *observations* rather than strict commands.

### 4.3 Client → server messages

Currently:

- Any messages sent from the dashboard to the `/observe` websocket are ignored and only used to detect disconnects.
- Future revisions will use client messages for:
  - Adjusting config,
  - Updating weights and policies,
  - Other operator actions.

Clients should:

- Not rely on any particular effect from sending messages at this time.
- Expect that any non-close messages may be silently ignored.

---

## 5. Example TypeScript typings for the SPA

For a Vite/React TS dashboard, a minimal typings layer might look like:

```ts
export type NeuronDescriptor = {
  node_id: string | null;
  label: string | null;
  metadata: any;
};

export type ModelId = { 0: string };

export type ModelConfig = {
  id: ModelId;
  display_name: string | null;
  backend_kind: string;
  command: string | null;
  args: string[];
  env: { key: string; value: string }[];
  listen_endpoint: string | null;
  metadata: any;
};

export type ProvisioningCommand =
  | { kind: "upsert_model_config"; config: ModelConfig }
  | { kind: "load_model"; model_id: ModelId }
  | { kind: "unload_model"; model_id: ModelId };

export type ProvisioningResponseWire = any; // { Ok: {...} } | { Error: {...} }

export type ObserveEvent =
  | { type: "neuron_registered"; neuron: NeuronDescriptor }
  | { type: "neuron_heartbeat"; neuron_id: string; metrics: any }
  | { type: "provisioning_sent"; neuron_id: string; cmd: ProvisioningCommand }
  | { type: "provisioning_response"; neuron_id: string; response: ProvisioningResponseWire };

export type ObserveSnapshot = {
  neurons: NeuronDescriptor[];
};

export type ObserveMessage =
  | { kind: "snapshot"; snapshot: ObserveSnapshot }
  | { kind: "event"; event: ObserveEvent };
```

Client logic can:

- On `snapshot`:
  - Replace its local `neurons` state with `snapshot.neurons`.
- On `event`:
  - Switch on `event.type` and update state accordingly:
    - Add/update neurons on `neuron_registered`.
    - Update last-seen heartbeat timestamp/metrics on `neuron_heartbeat`.
    - Log provisioning flows on `provisioning_sent` / `provisioning_response`.

---

This document describes the **current** `/observe` protocol as implemented in the cortex codebase. As the system evolves (e.g. richer capabilities, demand state, operator commands), this document should be updated to reflect new message types and semantics.