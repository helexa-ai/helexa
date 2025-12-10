# helexa dashboard websocket protocol

This document describes the websocket protocol exposed by a cortex node for use by
operator dashboards and other observability clients.

The endpoint is **read-only** from cortex’s perspective:

- Clients receive:
  - A **snapshot** of current state immediately after connection.
  - A **stream of events** describing control-plane activity between cortex and neurons.
- Clients may send messages (used only to detect disconnects for now); they are otherwise
  ignored. Future work may introduce operator commands on this channel.

All messages are JSON, encoded as UTF‑8 text websocket frames.

---

## 1. Endpoint

Given a cortex process started with:

```/dev/null/dashboard-example-run.md#L1-6
helexa cortex \
  --dashboard-socket 0.0.0.0:8090 \
  ...
```

Dashboard clients should connect to:

- `ws://<cortex-host>:8090`

For example (TypeScript):

```/dev/null/dashboard-example-connect.ts#L1-5
const ws = new WebSocket("ws://localhost:8090");

ws.onmessage = (ev) => {
  const msg = JSON.parse(ev.data);
  // handle snapshot/event
};
```

There is no subprotocol negotiation or required query parameters.

---

## 2. Top-level message envelope

All outbound messages from cortex share a common envelope:

```/dev/null/dashboard-envelope.json#L1-6
{
  "kind": "snapshot" | "event",
  // if kind == "snapshot":
  "snapshot": { ... },
  // if kind == "event":
  "event": { ... }
}
```

Where:

- `kind: "snapshot"` — the initial state, sent **exactly once per connection**.
- `kind: "event"` — subsequent streaming events sent after the snapshot.

TypeScript-style typings:

```/dev/null/dashboard-types-top-level.ts#L1-15
export type ObserveMessage =
  | { kind: "snapshot"; snapshot: ObserveSnapshot }
  | { kind: "event"; event: ObserveEvent };
```

---

## 3. Snapshot message

Immediately after the websocket is upgraded, cortex sends a **single** snapshot:

```/dev/null/dashboard-snapshot-example.json#L1-30
{
  "kind": "snapshot",
  "snapshot": {
    "neurons": [
      {
        "descriptor": {
          "node_id": "785b03f697304c88baf539e50e15e44a",
          "label": "785b03f697304c88baf539e50e15e44a",
          "metadata": {
            "backend": "neuron"
          }
        },
        "last_heartbeat_at": "2025-12-10T03:43:25.600000Z",
        "health": "healthy",
        "models": [
          {
            "model_id": { "0": "QuantTrio/Qwen3-Coder-30B-A3B-Instruct-GPTQ-Int8" },
            "last_cmd_kind": "load_model",
            "last_response": {
              "Ok": {
                "model_id": { "0": "QuantTrio/Qwen3-Coder-30B-A3B-Instruct-GPTQ-Int8" },
                "message": "model loaded and serving at http://127.0.0.1:8060"
              }
            },
            "effective_status": "loaded"
          }
        ]
      }
    ]
  }
}
```

### 3.1 Snapshot schema

```/dev/null/dashboard-snapshot-types.ts#L1-42
export type ModelId = { 0: string }; // serde encoding of ModelId(pub String)

export type ModelProvisioningStatus = {
  model_id: ModelId;

  /**
   * Last provisioning command kind cortex sent for this (neuron, model).
   * Examples:
   *  - "upsert_model_config"
   *  - "load_model"
   *  - "unload_model"
   */
  last_cmd_kind: string;

  /**
   * Most recent provisioning response from the neuron, if any.
   *
   * This mirrors the Rust enum:
   *   enum ProvisioningResponse {
   *     Ok { model_id: ModelId, message: Option<String> },
   *     Error { model_id: ModelId, error: String },
   *   }
   *
   * Under serde’s default representation this becomes:
   *   { "Ok": { "model_id": { "0": "..." }, "message": "..." } }
   *   { "Error": { "model_id": { "0": "..." }, "error": "..." } }
   */
  last_response: any | null;

  /**
   * Coarse derived status as seen by cortex, e.g.:
   *  - "configured"  (after successful UpsertModelConfig)
   *  - "loading"     (LoadModel sent, no response yet)
   *  - "loaded"      (LoadModel + Ok)
   *  - "unloading"   (UnloadModel sent, no response yet)
   *  - "unloaded"    (UnloadModel + Ok)
   *  - "failed"      (any Error response)
   *  - "unknown"     (insufficient information)
   */
  effective_status: string;
};

export type NeuronDescriptor = {
  node_id: string | null;  // machine-id or CLI --node-id
  label: string | null;    // human-friendly label; currently often same as node_id
  metadata: any;           // backend-specific and host metadata (os/arch/gpu/etc)
};

export type ObserveNeuron = {
  descriptor: NeuronDescriptor;

  /**
   * Best-effort wall-clock timestamp of the last heartbeat observed by cortex.
   * May be null if:
   *  - no heartbeat has been observed yet, or
   *  - conversion from internal clocks to SystemTime underflowed.
   */
  last_heartbeat_at: string | null;

  /**
   * Coarse health classification derived from heartbeat recency:
   *  - "healthy"  : recent heartbeat (<= 60s)
   *  - "degraded" : heartbeat present but a bit old (<= 5min)
   *  - "stale"    : no heartbeat yet, or last heartbeat older than 5min
   */
  health: "healthy" | "degraded" | "stale" | string;

  /**
   * Whether cortex currently considers this neuron online. This will be `false`
   * when the neuron has been explicitly removed (e.g. via a clean Shutdown
   * message) or pruned due to missing heartbeats.
   */
  offline: boolean;

  /**
   * Model provisioning state as seen by cortex for this neuron.
   * Includes configured, loaded, unloaded and failed models.
   */
  models: ModelProvisioningStatus[];
};

export type ObserveSnapshot = {
  neurons: ObserveNeuron[];
};
```

### 3.2 Snapshot semantics

- The snapshot is **per websocket connection**:
  - Sent exactly once, immediately after upgrade.
  - Reflects cortex’s current view at that moment.
- It is **authoritative** for dashboard state:
  - Clients should treat the snapshot as “source of truth as of now”.
  - After applying it, clients fold incremental `event` messages on top.

If the websocket disconnects, the client’s state is considered stale; upon reconnection, the new snapshot replaces any in-memory state.

---

## 4. Event stream

After the snapshot, cortex sends a stream of `event` messages:

```/dev/null/dashboard-event-envelope.json#L1-6
{
  "kind": "event",
  "event": {
    "type": "<event-type>",
    ...
  }
}
```

### 4.1 Event union

```/dev/null/dashboard-event-types.ts#L1-100
import type { NeuronDescriptor, ModelId, ModelProvisioningStatus } from "./dashboard-snapshot-types";

export type ProvisioningCommand =
  | { kind: "upsert_model_config"; config: any }   // see protocol docs for full shape
  | { kind: "load_model"; model_id: ModelId }
  | { kind: "unload_model"; model_id: ModelId };

/**
 * ProvisioningResponse is encoded using serde’s default enum representation:
 *
 *  { "Ok":    { "model_id": { "0": "..." }, "message": "..." } }
 *  { "Error": { "model_id": { "0": "..." }, "error": "..." } }
 */
export type ProvisioningResponseWire = any;

export type ObserveEvent =
  | { type: "neuron_registered"; neuron: NeuronDescriptor }
  | { type: "neuron_removed"; neuron_id: string }
  | { type: "neuron_heartbeat"; neuron_id: string; metrics: any }
  | { type: "provisioning_sent"; neuron_id: string; cmd: ProvisioningCommand }
  | {
      type: "provisioning_response";
      neuron_id: string;
      response: ProvisioningResponseWire;
    }
  | {
      type: "model_state_changed";
      neuron_id: string;
      models: ModelProvisioningStatus[];
    }
  | {
      type: "cortex_shutdown_notice";
      reason: string | null;
    };
    };
```

The following sections describe each event type.

---

## 5. Event types

### 5.1 `neuron_registered`

Emitted whenever cortex registers a neuron via the control-plane websocket, or when a neuron re-registers to refresh its metadata.

Example:

```/dev/null/dashboard-event-neuron-registered.json#L1-16
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

Schema:

```/dev/null/dashboard-event-neuron-registered.ts#L1-6
export type NeuronRegisteredEvent = {
  type: "neuron_registered";
  neuron: NeuronDescriptor;
};
```

Notes:

- This event is **incremental**; dashboards still rely on the snapshot to know the full neuron set at connect time.
- A re-register updates the descriptor in cortex and may be used to reflect changes in host metadata.

---

### 5.2 `neuron_removed`

Emitted when cortex considers a neuron to have left the cluster. This is typically
triggered when the neuron is pruned from the registry due to missing heartbeats
for longer than a configured timeout.

Example:

```/dev/null/dashboard-event-neuron-heartbeat.json#L1-12
{
  "kind": "event",
  "event": {
    "type": "neuron_heartbeat",
    "neuron_id": "785b03f697304c88baf539e50e15e44a",
    "metrics": {}
  }
}
```

Schema:

```/dev/null/dashboard-event-neuron-heartbeat.ts#L1-7
export type NeuronHeartbeatEvent = {
  type: "neuron_heartbeat";
  neuron_id: string;
  /**
   * Free-form JSON metrics object. Current implementations send `{}` but
   * future versions may include load, error counts, resource utilisation, etc.
   */
  metrics: any;
};
```

Correlation with snapshot:

- When a neuron is pruned and a `neuron_removed` event is emitted, future snapshots
  will no longer include that neuron in `snapshot.neurons`.
- Dashboards that maintain their own in-memory neuron list should remove entries
  when they receive `neuron_removed` for the corresponding `neuron_id`.

---

### 5.3 `neuron_heartbeat`

Emitted when cortex receives a `Heartbeat` message from a neuron via the control-plane websocket.

Example:

```/dev/null/dashboard-event-provisioning-sent.json#L1-38
{
  "kind": "event",
  "event": {
    "type": "provisioning_sent",
    "neuron_id": "785b03f697304c88baf539e50e15e44a",
    "cmd": {
      "kind": "upsert_model_config",
      "config": {
        "id": { "0": "QuantTrio/Qwen3-Coder-30B-A3B-Instruct-GPTQ-Int8" },
        "display_name": "Qwen3 Coder 30B GPTQ Int8",
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

Schema:

```/dev/null/dashboard-event-provisioning-sent.ts#L1-24
export type ProvisioningSentEvent = {
  type: "provisioning_sent";
  neuron_id: string;
  cmd: ProvisioningCommand;
};

export type ProvisioningCommand =
  | {
      kind: "upsert_model_config";
      config: {
        id: ModelId;
        display_name: string | null;
        backend_kind: string;
        command: string | null;
        args: string[];
        env: { key: string; value: string }[];
        listen_endpoint: string | null;
        metadata: any;
      };
    }
  | { kind: "load_model"; model_id: ModelId }
  | { kind: "unload_model"; model_id: ModelId };
```

Correlation with snapshot:

- The snapshot contains `last_heartbeat_at` and `health` derived from heartbeat timing.
- Heartbeat events are primarily for **live** UIs (e.g. streaming logs, “last seen” timers) rather than as a durable state source.

---

### 5.4 `provisioning_response`

Emitted whenever cortex receives a `ProvisioningResponse` from a neuron, acknowledging or rejecting a command.

Example:

```/dev/null/dashboard-event-provisioning-response.json#L1-20
{
  "kind": "event",
  "event": {
    "type": "provisioning_response",
    "neuron_id": "785b03f697304c88baf539e50e15e44a",
    "response": {
      "Ok": {
        "model_id": { "0": "QuantTrio/Qwen3-Coder-30B-A3B-Instruct-GPTQ-Int8" },
        "message": "model loaded and serving at http://127.0.0.1:8060"
      }
    }
  }
}
```

Error example:

```/dev/null/dashboard-event-provisioning-response-error.json#L1-14
{
  "kind": "event",
  "event": {
    "type": "provisioning_response",
    "neuron_id": "785b03f697304c88baf539e50e15e44a",
    "response": {
      "Error": {
        "model_id": { "0": "QuantTrio/Qwen3-Coder-30B-A3B-Instruct-GGPTQ-Int8" },
        "error": "failed to spawn backend process: <details>"
      }
    }
  }
}
```

Schema:

```/dev/null/dashboard-event-provisioning-response.ts#L1-12
export type ProvisioningResponseEvent = {
  type: "provisioning_response";
  neuron_id: string;
  response: ProvisioningResponseWire; // { Ok: {...} } | { Error: {...} }
};

export type ProvisioningResponseWire = any;
```

Relationship to snapshot and model_state_changed:

- Cortex updates its internal `ModelProvisioningStore` on every response.
- Snapshot `neurons[*].models[*].last_response` and `effective_status` expose
  the **latest** provisioning outcome, so late subscribers still see current
  model status without needing historical events.
- After recording a response, cortex also emits a `model_state_changed` event
  for the affected neuron, carrying the updated `models` array. Dashboards that
  need live per-model updates can fold this event into their in-memory state.

On clean neuron shutdown:

- The neuron sends a `shutdown` control-plane message to cortex before exiting.
- Cortex removes the neuron from its registry and emits a `neuron_removed` event.
- Cortex also clears model state for that neuron and emits a `model_state_changed`
  event with an empty `models` array so dashboards can immediately reflect that
  no models are active on that neuron.

---

### 5.5 `model_state_changed`

Emitted whenever cortex updates its internal view of model provisioning state for
a particular neuron. This typically happens immediately after handling a
`provisioning_response` from that neuron.

Example:

```/dev/null/dashboard-event-model-state-changed.json#L1-24
{
  "kind": "event",
  "event": {
    "type": "model_state_changed",
    "neuron_id": "785b03f697304c88baf539e50e15e44a",
    "models": [
      {
        "model_id": { "0": "QuantTrio/Qwen3-Coder-30B-A3B-Instruct-GPTQ-Int8" },
        "last_cmd_kind": "load_model",
        "last_response": {
          "Ok": {
            "model_id": { "0": "QuantTrio/Qwen3-Coder-30B-A3B-Instruct-GPTQ-Int8" },
            "message": "model loaded and serving at http://127.0.0.1:8060"
          }
        },
        "effective_status": "loaded"
      }
    ]
  }
}
```

Schema:

```/dev/null/dashboard-event-model-state-changed.ts#L1-10
import type { ModelProvisioningStatus } from "./dashboard-snapshot-types";

export type ModelStateChangedEvent = {
  type: "model_state_changed";
  neuron_id: string;
  models: ModelProvisioningStatus[];
};
```

Semantics:

- `models` is the **full set** of model provisioning statuses currently known
  for that neuron at the time of the event.
- The payload is derived from the same internal store that powers the snapshot:
  for a given neuron, `models` in `model_state_changed` matches
  `snapshot.neurons[*].models` for that neuron at that moment.
- Dashboards that only care about eventual consistency can ignore this event
  and rely solely on snapshots.
- Dashboards that want live per-model state should:
  - locate the corresponding neuron by `neuron_id`,
  - replace that neuron’s `models` array with the one from the event.

### 5.6 `cortex_shutdown_notice`

Emitted when this cortex instance is performing a planned shutdown or restart.

Example:

```/dev/null/dashboard-event-cortex-shutdown-notice.json#L1-10
{
  "kind": "event",
  "event": {
    "type": "cortex_shutdown_notice",
    "reason": "cortex is shutting down"
  }
}
```

Schema:

```/dev/null/dashboard-event-cortex-shutdown-notice.ts#L1-6
export type CortexShutdownNoticeEvent = {
  type: "cortex_shutdown_notice";
  reason: string | null;
};
```

Semantics:

- Indicates that the current `/observe` connection will close shortly as this cortex instance shuts down or restarts.
- Dashboards should:
  - Treat the current snapshot as about to become stale.
  - Expect the websocket to close soon after this event.
  - Attempt to reconnect to `/observe` after the connection closes, using a suitable backoff, to obtain a fresh snapshot from the new cortex instance.

---

## 6. Late subscribers and model state

Late subscribers (dashboards that connect after provisioning has happened) may have missed:

- `provisioning_sent` events,
- `provisioning_response` events.

However:

- Cortex tracks per-model, per-neuron provisioning state in-memory.
- The **snapshot** includes the current state for each neuron:

  - For each `ObserveNeuron`:
    - `models[*].last_cmd_kind` — last command cortex sent.
    - `models[*].last_response` — most recent `Ok` / `Error` from the neuron.
    - `models[*].effective_status` — derived status (`configured`, `loaded`, `failed`, …).

Because of this:

- A late-subscribing dashboard can always render:
  - Which models are configured on which neurons.
  - Which models are currently loaded/unloaded.
  - Which models failed to load/unload (and why), based on the `Error` payload.

The incremental events (`provisioning_sent`, `provisioning_response`) are primarily useful for:

- Live streaming logs / activity views.
- Real-time feedback as an operator triggers provisioning.

---

## 7. Client responsibilities and patterns

### 7.1 Connection lifecycle

A typical dashboard lifecycle:

```/dev/null/dashboard-client-pattern.ts#L1-40
const ws = new WebSocket("ws://localhost:8090");

type State = {
  snapshot: ObserveSnapshot | null;
  events: ObserveEvent[];
};

const state: State = { snapshot: null, events: [] };

ws.onmessage = (ev) => {
  const msg = JSON.parse(ev.data) as ObserveMessage;
  if (msg.kind === "snapshot") {
    state.snapshot = msg.snapshot;
    state.events = [];
  } else if (msg.kind === "event") {
    state.events.push(msg.event);
    // Optional: fold incremental updates into derived UI state
  }
};

ws.onclose = () => {
  // Consider state.snapshot stale; reconnect to rebuild.
};
```

Guidelines:

- Always handle the **first message** as a snapshot.
- Treat any disconnect as losing source-of-truth:
  - On reconnect, discard previous snapshot and events.
  - Replace with the new snapshot.

### 7.2 Folding events

The simplest (and robust) approach:

- Use the snapshot as the **only** source of truth for structured state.
- Use events only for:
  - logging / timelines,
  - transient UI updates (e.g. a toast when provisioning succeeds/fails).

More advanced clients may:

- Maintain an in-memory view of neurons and models.
- Apply incremental changes based on:
  - `neuron_registered` (add/update a neuron),
  - `neuron_heartbeat` (update last-seen timestamp client-side),
  - `provisioning_sent` / `provisioning_response` (update a model’s status).

Even in that case, the snapshot remains the canonical “reset point” used on reconnect.

---

## 8. Client → server messages

Currently:

- Any messages sent by the client on this websocket are:
  - ignored by cortex for business logic,
  - used only to detect disconnects / liveness at the transport level.

Future protocol revisions may introduce:

- Operator-initiated provisioning,
- Demand / capacity adjustments,
- Administrative actions on neurons or models.

Dashboards should:

- Not depend on any server-side effect from sending messages at this time.
- Be prepared for new `event.type` variants and extended payloads as the system evolves.

---

This document describes the current `/observe` protocol as implemented in the cortex
codebase, including enriched neuron snapshots with heartbeat-based health and per-model
provisioning state. As the platform grows, new event types and snapshot fields may be
introduced; dashboard implementations should handle unknown fields and extra `event.type`
variants gracefully.