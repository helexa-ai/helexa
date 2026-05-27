# CLAUDE.md — cortex

## Project overview

cortex is a Rust reverse-proxy that sits in front of multiple
mistral.rs inference nodes and presents a unified OpenAI + Anthropic
compatible API surface. It handles model routing, lifecycle management
(load/unload/evict), request translation, and metrics collection.

## Repository layout

```
cortex/
├── Cargo.toml              # workspace root
├── cortex.toml      # example gateway config
├── README.md
├── CLAUDE.md               # ← you are here
├── crates/
│   ├── cortex-core/            # shared types, config, envelopes
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── config.rs       # figment-based config structs
│   │       ├── node.rs         # NodeState, ModelStatus
│   │       ├── openai.rs       # OpenAI request/response types
│   │       ├── anthropic.rs    # Anthropic request/response types
│   │       ├── translate.rs    # OpenAI <-> Anthropic translation
│   │       └── metrics.rs      # RequestMetrics, histogram helpers
│   ├── cortex-gateway/         # the HTTP proxy server
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── state.rs        # CortexState: Arc<RwLock<...>>
│   │       ├── router.rs       # model -> node routing logic
│   │       ├── proxy.rs        # streaming HTTP proxy to backends
│   │       ├── evictor.rs      # LRU/priority eviction logic
│   │       ├── poller.rs       # background task polling node status
│   │       ├── handlers.rs     # axum handlers (chat, completions, models, etc.)
│   │       └── metrics.rs      # prometheus exporter endpoint
│   ├── cortex-agent/           # per-node sidecar (future: defrag, restart)
│   │   └── src/
│   │       ├── lib.rs
│   │       └── agent.rs        # local node management
│   └── cortex-cli/             # CLI entrypoint
│       └── src/
│           └── main.rs
└── tests/                  # integration tests (future)
```

## Key design decisions

### mistral.rs HTTP API for model lifecycle
mistral.rs (v0.8+) supports dynamic model loading/unloading at runtime:
- `POST /v1/models/unload {"model_id": "..."}` — frees VRAM, preserves config
- `POST /v1/models/reload {"model_id": "..."}` — explicitly reload
- `POST /v1/models/status {"model_id": "..."}` — loaded/unloaded/reloading
- `GET /v1/models` — lists all models with status field
- Lazy loading: requests to unloaded models trigger automatic reload

The gateway does NOT manage systemd units for model swaps. It calls these
HTTP endpoints directly. The only systemd interaction is for full-process
restarts after VRAM fragmentation accumulates (defrag_after_cycles).

### Streaming proxy
Chat completions are proxied as SSE streams. The gateway must:
1. Parse the inbound request to extract the model name
2. Route to the correct backend node
3. Stream the response back, capturing token timing for metrics
4. NOT buffer the full response — true streaming passthrough

### Anthropic translation
When a request arrives at `/v1/messages` (Anthropic format), the gateway
translates it to OpenAI format before proxying to mistral.rs, then
translates the response back. This is stateless envelope transformation.

### Eviction
The evictor runs as a background task. Before loading a model on a node
where VRAM is tight:
1. Check if the model is already loaded elsewhere → route there instead
2. Find the LRU model on the target node (excluding pinned models)
3. Call `/v1/models/unload` on that model
4. The incoming request's lazy-load triggers the new model load

### Metrics
Per-request: model, node, prompt_tokens, completion_tokens, total_tokens,
tok_per_sec, time_to_first_token_ms, total_latency_ms.
Exposed as Prometheus histograms/counters on a separate port.

### Per-device worker thread (neuron)
The neuron daemon dedicates one OS thread per CUDA device it loads
onto. That thread binds the device's `CudaContext` once at startup and
owns it for the daemon's lifetime; every model load, forward step,
KV-cache reset, VRAM query, NCCL init/sanity, NCCL all_reduce, and
model drop on that device routes through this thread via a
`std::sync::mpsc` job channel. Replies cross back via
`tokio::sync::oneshot`.

Three properties this gives us, in order of weight:

1. **Context locality.** cudarc binds the CUDA context per OS thread
   via `cuCtxSetCurrent`. Before this refactor, ad-hoc
   `tokio::task::spawn_blocking` calls bound the context onto a
   different thread per request — and `device_vram_mb()` from an
   async task bound it onto whichever tokio worker happened to be
   running. Pinning the context to one named thread ends that.
2. **Drop safety.** Every `CudaSlice` in a `Tensor`, every
   `cudarc::nccl::Comm`, and the `CudaContext` itself call `cuMemFree` /
   `ncclCommDestroy` / `cuCtxDestroy` during `Drop` — and require the
   right context current. With the worker owning the model slab,
   `Drop` always runs on the right thread. The cudarc Drop constraint
   is structurally enforced.
3. **Poisoning blast radius.** When a CUDA driver error makes the
   context unrecoverable, the poison flag lives on the
   `DeviceWorkerHandle` itself. Subsequent `submit()` calls fast-reject
   at the channel boundary with a clear "device worker is poisoned"
   error before any further CUDA work is attempted. The thread doesn't
   exit (dropping the slab would re-touch the broken context) — it
   enters a drain-only mode and replies error to everything until the
   daemon restarts.

Tensors never escape the worker thread alive. Inference replies carry
`Vec<f32>` CPU-side logits; the async caller wraps them in a CPU
candle tensor and runs `apply_repeat_penalty` + `LogitsProcessor::sample`
without ever rebinding the device context. Sampled tokens come back as
`u32`; VRAM queries as `(u64, u64)`. The opaque `ArchHandle(u64)` and
`TpHandle(u64)` are the only "references" callers hold to loaded
models — they're indices into the worker's state slab, not pointers.

The TP worker subprocesses in `harness/tp/worker.rs` are the same
pattern out-of-process — a dedicated context-owning process per
non-zero NCCL rank. The in-process worker in `harness/device_worker/`
brings the discipline to rank 0.

CPU loads (`Device::Cpu` fallback when CUDA is unavailable) keep the
legacy `tokio::task::spawn_blocking + Arc<Mutex<ModelArch>>` path —
there's no context to own and the channel hop would only add latency.
Four `spawn_blocking` references in `harness/candle.rs` are deliberate
CPU fallback.

Canonical narrative lives in
`crates/neuron/src/harness/device_worker/mod.rs`'s module
doc-comment; touch points (the `Job` enum, the dispatch handlers, the
`DeviceWorkerState` struct) are in the sibling `jobs.rs` and
`dispatch.rs`.

## Tech stack

- **Rust 2024 edition** — workspace with 4 crates
- **Axum 0.8** — HTTP framework (same as mistral.rs itself)
- **reqwest** — HTTP client for proxying to backends
- **figment** — config loading (TOML + env vars)
- **tokio** — async runtime
- **metrics + metrics-exporter-prometheus** — observability
- **tracing** — structured logging

## Build commands

```sh
cargo build --release           # build all crates
cargo run -p cortex-cli -- serve    # run the gateway
cargo test                      # run all tests
cargo clippy --workspace        # lint
```

## CI

Gitea Actions runs on every push to any branch. All three checks must
pass before merging:

```sh
cargo fmt --check --all                    # formatting
cargo clippy --workspace -- -D warnings   # lint (warnings are errors)
cargo test --workspace                     # tests
```

Run these locally before pushing. `cargo fmt --all` fixes formatting
automatically. Clippy warnings must be resolved, not suppressed with
`#[allow(...)]` unless there is a clear rationale.

## Environment

- Targets Fedora 43 (systemd, SELinux enforcing)
- Nodes communicate over a private network (e.g. WireGuard mesh)
  - One or more GPU nodes running mistral.rs on port 8080
  - Optionally a metrics-only node (no GPU) for Prometheus/Grafana
- Each node runs `mistralrs serve` on port 8080
- Gateway listens on port 31313 (API) and 31314 (metrics)
- neuron listens on port 13131 on each GPU host
- TLS terminated at gateway or via nginx; internal traffic is plaintext over WireGuard

## Conventions

- Error handling: `anyhow` for binaries, `thiserror` for library crates
- No `unwrap()` in library code; `expect()` only with clear rationale
- All public types derive `Debug, Clone, Serialize, Deserialize` where sensible
- Config structs use `figment` with TOML as primary source, env vars as override
- Prefer `Arc<RwLock<...>>` for shared fleet state; minimize lock duration
- SSE streaming uses `tokio_stream` + `eventsource-stream` for parsing
- Log at `info` for request routing, `debug` for proxy details, `warn` for
  eviction and node health, `error` for proxy failures

## mistral.rs API gotchas

These are sharp edges Claude Code will hit when implementing the proxy.
Read before touching `proxy.rs` or `handlers.rs`.

### Model name validation

mistral.rs validates that the `model` field in every request matches the
model that was actually loaded. If the names don't match, the request is
rejected outright. The special model name `"default"` bypasses this
validation entirely.

**Implication for cortex:** The gateway must ensure the `model` field in
the proxied request body matches what mistral.rs expects. Two strategies:

1. **Passthrough** — the client uses the exact HuggingFace model ID
   (e.g. `Qwen/Qwen3-Coder-30B-A3B-Instruct`) and cortex routes based
   on that. This is the simplest approach and should be the default.
2. **Rewrite to `"default"`** — if cortex introduces its own model
   aliases, it must rewrite the `model` field to `"default"` before
   proxying. This is a future feature, not phase 1.

### Lazy loading latency

When a request hits an unloaded model, mistral.rs automatically reloads
it before processing. This can take 10-60+ seconds for large models. The
gateway must:
- Set a generous HTTP client timeout (already 300s in the scaffold).
- Mark the request as `cold_start: true` in metrics.
- Not retry or time out prematurely — the upstream is busy loading, not dead.

### SSE stream format

mistral.rs streams use standard OpenAI SSE format:
```
data: {"id":"...","choices":[{"delta":{"content":"token"},...}]}\n\n
data: [DONE]\n\n
```
The proxy must forward these chunks verbatim. Do not attempt to parse
or re-serialize each chunk — that adds latency and risks breaking the
stream. Parse only for metrics extraction (token counts from the final
`usage` object, timing from chunk arrival).

### Multi-model mode

`mistralrs serve` can load multiple models when started with a selector
config or multiple `--text-model` / `--vision-model` flags. The
`/v1/models` response lists all of them with a `status` field. When
sending requests, the `model` field must match one of the listed model
IDs — `"default"` only works if you don't care which model handles it.

### Unload preserves config

`POST /v1/models/unload` frees VRAM but keeps the model's config in
memory. A subsequent request to that model (or explicit `reload`) will
reload from disk/HF cache — not re-download. This is fast relative to
initial download but still involves loading weights into VRAM.


## Implementation plan

Each phase is a branch → PR. CI must pass (fmt, clippy, test) before merge.
Phases are sequential — each builds on the previous.

### Phase 1: Compile and proxy a basic request ✅

Completed. 6 integration tests in `cortex-gateway/tests/proxy_basic.rs`:
chat completion proxy, health endpoint, list models, model not found,
no healthy nodes, missing model field. Test helpers in `tests/common/mod.rs`
provide `spawn_mock_backend()` and `spawn_gateway()` using axum as the
mock mistral.rs backend.

### Phase 2: Streaming SSE passthrough ✅

Completed. The existing `Body::from_stream(bytes_stream())` proxy works
for SSE out of the box. 2 integration tests in `cortex-gateway/tests/streaming.rs`:
- `test_streaming_sse_passthrough` — 5 chunks with 50ms delays, verifies
  incremental delivery (time spread between first and last chunk)
- `test_streaming_done_terminator` — verifies `data: [DONE]` is forwarded

### Phase 3: Poller + live `/v1/models` ✅

Completed. Extracted `poll_once()` from `poll_loop()` for testability.
4 tests in `cortex-gateway/tests/poller.rs`:
- `test_poller_discovers_models` — 2 models (loaded + unloaded) discovered with correct status
- `test_poller_updates_gateway_models_endpoint` — `/v1/models` reflects polled state with node attribution
- `test_poller_marks_unreachable_node_unhealthy` — unreachable node flipped to unhealthy
- `test_poller_removes_stale_models` — model removed from upstream is pruned from state

### Phase 4: Eviction ✅

Completed. Added `last_accessed` tracking in handlers (`touch_model`
called after routing). 5 tests in `cortex-gateway/tests/eviction.rs`:
- `test_evict_lru_model` — older model evicted, unload call verified on mock
- `test_eviction_skips_pinned_models` — pinned model protected, newer model evicted instead
- `test_eviction_nothing_to_evict` — all models pinned, returns None
- `test_eviction_increments_lifecycle_cycles` — counter incremented after eviction
- `test_last_accessed_updated_on_request` — `last_accessed` set after proxied request

Router-triggered eviction (automatic eviction on VRAM pressure during
request routing) deferred — requires per-model VRAM tracking which is
not yet populated. The `evict_lru_on_node` function is callable and
tested for when that integration is added.

### Phase 5: Anthropic translation ✅

Completed. Non-streaming Anthropic round-trip implemented: handler
buffers upstream OpenAI response, translates via `openai_to_anthropic`,
returns Anthropic-format JSON. 5 tests in `cortex-gateway/tests/anthropic.rs`:
- `test_anthropic_to_openai_round_trip` — full request/response translation
  with stop_reason mapping ("stop" → "end_turn") and usage field names
- `test_anthropic_with_system_prompt` — system field translated to system message
- `test_anthropic_with_content_blocks` — array content blocks handled
- `test_anthropic_model_not_found` — 404 for unknown model
- `test_anthropic_invalid_request` — 400 for malformed request

Streaming Anthropic SSE translation (OpenAI SSE → Anthropic SSE event
types) deferred as a follow-up.

### Phase 6: Metrics instrumentation ✅

Completed. Added `proxy_with_metrics` helper in handlers that wraps
every proxy call with timing and counters. All three handler paths
(chat completions, completions, Anthropic messages) instrumented.

Metrics emitted per request (with `model` and `node` labels):
- `cortex_requests_total` — incremented on every proxy attempt
- `cortex_request_duration_seconds` — histogram of successful request latency
- `cortex_request_errors_total` — incremented on proxy failures
- `cortex_cold_starts_total` — incremented when routing to an unloaded model

Added `install_test_recorder()` for testing without the HTTP listener.
1 test in `cortex-gateway/tests/metrics.rs` verifies counters and
histograms appear after a proxied request.

Token-level metrics (tok/s, TTFT) deferred — requires parsing the
response body or final SSE chunk, which is Phase 6b work.

## 2026-04-15 addendum

**Phases 1–6 complete.** The gateway proxies requests (streaming and
non-streaming), routes by model name to the correct node, polls node
`/v1/models` for live state, evicts LRU models with pinning, translates
Anthropic ↔ OpenAI envelopes, and emits Prometheus metrics. CI is green.

**Phase 7 onward** introduces `neuron` — the per-node daemon that replaces
the placeholder `cortex-agent` crate — along with hardware discovery,
a harness abstraction (so cortex is not permanently wedded to mistral.rs),
and a model catalogue for placement decisions.


### Architecture: cortex + neuron

cortex is the **control plane**. It exposes the unified API, routes
requests, manages model lifecycle across the fleet, and collects metrics.

neuron is the **node plane**. One instance runs on every GPU host. It:
- **Discovers** local hardware (GPU count, types, VRAM, CUDA compute
  capability, driver version) and reports it to cortex.
- **Manages harnesses** — inference engines like mistral.rs, llama.cpp,
  or ComfyUI. Each harness is a trait implementation. neuron starts,
  stops, health-checks, and proxies to whichever harness is serving a
  given model.
- **Manages model lifecycle** — load, unload, status — abstracting the
  differences between harnesses (mistral.rs has HTTP lifecycle endpoints;
  llama.cpp may need process management).
- **Reports runtime state** — per-device VRAM usage, GPU utilisation,
  temperature, loaded models with actual VRAM consumption.

cortex never shells out to `nvidia-smi`, never touches systemd units,
and never talks directly to a harness. It talks only to neurons.

```
                    ┌─────────────────────┐
                    │      cortex         │
                    │  (cortex-gateway)   │
                    │  Router · Evictor   │
                    │  Metrics · Translate│
                    └──┬──────┬────────┬──┘
                       │      │        │
            ┌──────────▼┐  ┌──▼─────┐  ┌▼──────────┐
            │  neuron   │  │ neuron │  │  neuron   │
            │  beast    │  │ benjy  │  │ quadbrat  │
            │           │  │        │  │           │
            │ harness:  │  │harness:│  │ harness:  │
            │ mistralrs │  │mistral │  │ mistralrs │
            │ (+ comfy) │  │rs      │  │           │
            └───────────┘  └────────┘  └───────────┘
```


## The Harness trait

Defined in `cortex-core` so both cortex and neuron share the type
definitions. neuron provides the runtime implementations.

```rust
/// What an inference harness must do, from neuron's perspective.
#[async_trait]
pub trait Harness: Send + Sync {
    /// Human-readable name (e.g. "mistralrs", "llamacpp", "comfyui").
    fn name(&self) -> &str;

    /// Start the harness process if it is not already running.
    async fn start(&self, config: &HarnessConfig) -> Result<()>;

    /// Stop the harness process gracefully.
    async fn stop(&self) -> Result<()>;

    /// Health check. Returns the harness process status.
    async fn health(&self) -> HarnessHealth;

    /// List models the harness knows about (loaded + unloaded).
    async fn list_models(&self) -> Result<Vec<ModelInfo>>;

    /// Load a model with the given spec (quant, TP, device assignment).
    async fn load_model(&self, spec: &ModelSpec) -> Result<()>;

    /// Unload a model, freeing device memory.
    async fn unload_model(&self, model_id: &str) -> Result<()>;

    /// Return the URL where inference requests for this model should
    /// be sent. None if the model is not loaded.
    async fn inference_endpoint(&self, model_id: &str) -> Option<String>;
}
```

The mistral.rs implementation wraps the HTTP API:
- `list_models` → `GET /v1/models`
- `load_model` → `POST /v1/models/reload`
- `unload_model` → `POST /v1/models/unload`
- `inference_endpoint` → returns the base URL (the model name routes
  internally within mistral.rs)
- `start`/`stop` → manage the `mistralrs.service` systemd unit

A future llama.cpp implementation would manage per-model `llama-server`
processes (one process per loaded model, each on its own port).


## neuron API

neuron exposes an HTTP API on port 13131 that cortex polls and calls.

```
GET  /discovery
     → {
         hostname, os, kernel,
         cuda_version, driver_version,
         devices: [{ index, name, vram_total_mb, compute_capability }],
         harnesses: ["mistralrs", ...]
       }

GET  /health
     → {
         uptime_secs,
         devices: [{ index, vram_used_mb, vram_free_mb, utilization_pct, temp_c }]
       }

GET  /models
     → [{ id, harness, status, devices: [int], vram_used_mb }]

POST /models/load
     ← { model_id, harness, quant, tensor_parallel, devices: [int] }
     → { status: "loaded" | "loading" }

POST /models/unload
     ← { model_id }
     → { status: "unloaded" }

GET  /models/{model_id}/endpoint
     → { url: "http://localhost:8080" }
```

cortex never constructs a harness-specific URL. It asks neuron for the
inference endpoint and proxies there.


## Discovery replaces static device config

cortex.toml no longer contains device types, VRAM sizes, or CUDA
architectures. That information comes from neuron's `/discovery`
endpoint. cortex.toml shrinks to:

```toml
[gateway]
listen = "0.0.0.0:31313"
metrics_listen = "0.0.0.0:31314"

[eviction]
strategy = "lru"
defrag_after_cycles = 50

[[neurons]]
name = "beast"
endpoint = "http://beast.hanzalova.internal:13131"

[[neurons]]
name = "benjy"
endpoint = "http://benjy.hanzalova.internal:13131"

[[neurons]]
name = "quadbrat"
endpoint = "http://quadbrat.hanzalova.internal:13131"
```

On startup and periodically, cortex calls `GET /discovery` and
`GET /health` on each neuron to build its topology map. The router
uses this topology — not config — to make placement decisions.


## Model catalogue

Model serving profiles live in a separate file (`models.toml`) because
they describe how to serve a model, not where. cortex matches these
profiles against the discovered topology to determine valid placements.

```toml
[[models]]
id = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
harness = "mistralrs"
quant = "Q4_K_M"
vram_mb = 19000
min_devices = 2
min_device_vram_mb = 10000
pinned_on = ["beast"]       # optional: never evict from these neurons

[[models]]
id = "Qwen/Qwen3-VL-8B"
harness = "mistralrs"
quant = "Q8_0"
vram_mb = 10000
min_devices = 1

[[models]]
id = "Qwen/Qwen2.5-Coder-14B-Instruct"
harness = "mistralrs"
quant = "Q6_K"
vram_mb = 12000
min_devices = 1
pinned_on = ["benjy"]
```

The router consults the catalogue to answer: "model X needs 2 devices
with ≥10GB each; beast has 2× RTX 5090 at 32GB each; that's a valid
placement." This replaces the current per-node `pinned` list in config
and the hardcoded `vram_mb` per node.


## Revised repository layout

```
cortex/
├── Cargo.toml
├── cortex.toml                 # gateway config (neurons only)
├── models.toml                 # model catalogue
├── README.md
├── CLAUDE.md
├── crates/
│   ├── cortex-core/            # shared types
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── config.rs       # GatewayConfig, NeuronEndpoint
│   │       ├── catalogue.rs    # ModelProfile, placement matching
│   │       ├── discovery.rs    # DeviceInfo, DiscoveryResponse
│   │       ├── harness.rs      # Harness trait, HarnessConfig, HarnessHealth
│   │       ├── node.rs         # NodeState, ModelEntry, ModelStatus
│   │       ├── openai.rs       # OpenAI envelope types
│   │       ├── anthropic.rs    # Anthropic envelope types
│   │       ├── translate.rs    # OpenAI <-> Anthropic translation
│   │       └── metrics.rs      # RequestMetrics
│   ├── cortex-gateway/         # control plane (existing, modified)
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── state.rs        # CortexState (updated: discovery topology)
│   │       ├── router.rs       # updated: catalogue + discovery placement
│   │       ├── proxy.rs        # streaming proxy (unchanged)
│   │       ├── evictor.rs      # updated: talks to neuron, not mistralrs
│   │       ├── poller.rs       # updated: polls neuron, not mistralrs
│   │       ├── handlers.rs     # axum handlers (unchanged API surface)
│   │       └── metrics.rs      # prometheus exporter (unchanged)
│   ├── neuron/                 # node plane (replaces cortex-agent)
│   │   └── src/
│   │       ├── main.rs         # binary entrypoint, axum server on :13131
│   │       ├── discovery.rs    # nvidia-smi, device enumeration
│   │       ├── health.rs       # runtime GPU polling
│   │       ├── api.rs          # HTTP handlers for /discovery, /models, etc.
│   │       ├── harness/
│   │       │   ├── mod.rs      # Harness trait re-export, registry
│   │       │   ├── mistralrs.rs  # mistral.rs HTTP API wrapper
│   │       │   └── llamacpp.rs   # stub for future llama.cpp support
│   │       └── models.rs       # local model lifecycle orchestration
│   └── cortex-cli/             # CLI entrypoint (unchanged)
│       └── src/
│           └── main.rs
└── tests/
```

The `cortex-agent` crate is deleted. Its replacement is `neuron/`.


## Implementation plan (phases 7+)

Phases 1–6 are merged and passing CI. Each subsequent phase is a
branch → PR. CI (fmt, clippy, test) must pass before merge.

### Phase 7: neuron scaffold and discovery ✅

Completed. Deleted `cortex-agent`, created `crates/neuron/` (binary:
`neuron`). Added shared types to cortex-core: `discovery.rs`
(DeviceInfo, DiscoveryResponse, DeviceHealth, HealthResponse) and
`harness.rs` (Harness async trait, HarnessConfig, ModelSpec, ModelInfo).

neuron discovers GPUs via nvidia-smi, caches health readings, and
serves `GET /discovery` and `GET /health`. Pure parsing functions
separated from command execution for testability. 9 unit tests for
nvidia-smi CSV parsing, 3 integration tests for the HTTP endpoints.

### Phase 8: neuron harness — mistral.rs implementation ✅

Completed. Full `Harness` trait implementation for mistral.rs in
`neuron/src/harness/mistralrs.rs`: list_models, load_model, unload_model,
inference_endpoint, health, start/stop (systemd). `HarnessRegistry` in
`harness/mod.rs` maps harness name → `Box<dyn Harness>`, built from
`neuron.toml` config. Four new neuron API endpoints: `GET /models`,
`POST /models/load`, `POST /models/unload`, `GET /models/:id/endpoint`.

Config via `neuron.toml` (figment + env override). Integration test
covers full model lifecycle through neuron → mock mistral.rs backend.

### Phase 9: cortex talks to neurons ✅

Completed. Full refactor of cortex-gateway to talk to neurons:

- **Config**: `NodeConfig { endpoint, vram_mb, pinned }` replaced with
  `NeuronEndpoint { name, endpoint }`. Hardware info comes from neuron
  discovery, pinning from `models.toml` catalogue.
- **catalogue.rs**: `ModelProfile` with `pinned_on`, `ModelCatalogue`
  with `is_pinned()` for eviction decisions.
- **Poller**: polls neuron's `GET /models` (ModelInfo format) instead
  of mistralrs `/v1/models`.
- **Router**: asks neuron `GET /models/{id}/endpoint` for the inference
  URL before proxying. Decouples cortex from knowing harness ports.
- **Evictor**: calls `POST {neuron}/models/unload` instead of
  mistralrs directly. Uses catalogue for pinning.
- **Tests**: all 22 gateway tests updated to mock neuron API instead
  of raw mistralrs. 36 total tests passing.

Topology-aware placement (min_devices, min_device_vram_mb) deferred —
the router currently routes based on polled model status. Catalogue
placement matching can be added incrementally.

### Phase 10: RPM packaging ✅

Completed. Both packages have RPM specs, systemd units, and example configs.
CI builds parallel SRPMs on tag push and publishes to separate COPR repos.

- `cortex.spec` — installs the `cortex` binary. Package name keeps the
  short `cortex` because no Fedora package collides with it.
- `helexa-neuron.spec` — installs the `neuron` binary under package name
  `helexa-neuron`. Renamed from bare `neuron` to avoid collision with
  Fedora's NEURON neural-simulation package
  (https://src.fedoraproject.org/rpms/neuron); binary, systemd unit,
  system user, and config dir all stay named `neuron` since those are
  project-local contexts.
- `data/cortex.service`, `data/neuron.service` — systemd units
- `cortex.example.toml`, `neuron.example.toml`, `models.example.toml`
- CI: parallel `srpm-cortex` + `srpm-neuron` jobs, then parallel COPR
  publish to a single project `helexa/helexa` hosting both packages.

Install:
```sh
dnf copr enable helexa/helexa
dnf install cortex                # gateway host
dnf install helexa-neuron         # GPU nodes
```

## 2026-05-18 addendum: candle-native pivot

Phases 11 (llama.cpp harness) and 12 (mistral.rs COPR) below are
**superseded**. The project no longer treats mistral.rs or llama.cpp as
dependencies — both are conceptually out of scope. neuron becomes a
candle-native inference daemon, with `Harness` retained as an
internal seam for adding future engines (vision/audio/diffusion) but
its only implementation being in-process candle.

The full staged plan for this pivot lives at
`~/.claude/plans/create-a-more-aggressive-calm-naur.md`. Summary:

- **Stage 1 (this commit):** delete `mistralrs.rs` and `llamacpp.rs`,
  scaffold inert `CandleHarness`, drop `endpoint`/`systemd_unit` from
  `HarnessConfig`, default no-op `start`/`stop` on the `Harness` trait.
- **Stages 2–4:** wire up candle model load/unload (quantized Qwen3
  first), add OpenAI-compatible inference endpoint in neuron, then SSE
  streaming.
- **Stages 5–6:** load-on-activation (default models in config) and
  unload-on-deactivation (graceful shutdown).
- **Stages 7–8:** multi-GPU tensor parallelism and broader model/quant
  coverage.

Sections of this document that describe mistral.rs HTTP behaviour
("mistral.rs API gotchas") are retained as historical context for
Phases 1–10 — they document what was true while the project depended
on mistral.rs. They do not describe current behaviour.

---

### Phase 11 (superseded): llama.cpp harness stub

~~Originally planned as a second engine to prove the harness
abstraction.~~ Replaced by the candle harness work in the 2026-05-18
addendum above. llama.cpp's any-model/any-hardware breadth is no
longer in scope for helexa.

### Phase 12 (superseded): mistral.rs COPR packaging

~~Originally planned to ship CUDA-versioned mistral.rs RPMs.~~ Replaced
by the candle harness work in the 2026-05-18 addendum above. With
mistral.rs out of the dependency tree, there is nothing to package.

## 2026-05-27 addendum: per-device worker thread

Replaced the ad-hoc `tokio::task::spawn_blocking` pattern that drove
every leader-side CUDA op with one dedicated OS thread per CUDA device,
permanently bound to that device's `CudaContext`. All leader-side
inference work (GGUF + dense + TP shard load, forward, kv-cache clear,
NCCL init/sanity, NCCL all_reduce, VRAM query, model drop) routes
through the worker via a `std::sync::mpsc` channel; tensors never
escape the worker thread alive. See "Per-device worker thread (neuron)"
above and `crates/neuron/src/harness/device_worker/mod.rs` for the
canonical narrative.

Motivated by the 2026-05-26 silent-hang on beast: a CUDA OOM cascade
poisoned the device context on whichever spawn_blocking thread caught
it, and subsequent requests stalled invisibly on the pool lock. After
the refactor, the same failure mode shows up in journalctl as
`prefill sample failed; logits unhealthy nan: 248320/248320` followed
by `failed, model marked poisoned`. The thread stays alive and rejects
subsequent requests at the channel boundary.

Landed in four PRs:

- **Phase 1** (`081b532`) — device_worker module + 8 VRAM-query sites
  route through the worker. CPU build only; smoke on beast confirmed
  a persistent `cuda-dev-0` thread.
- **Phase 2** (`b179204`) — single-GPU forward + clear_kv + drop via
  the worker. `LoadedModel.arch_handle: Option<ArchHandle>` replaces
  `Arc<Mutex<ModelArch>>` for CUDA loads. CPU keeps the legacy path.
- **Phase 3** (`76ab24d`) — TP forward + NCCL init/sanity + leader
  KV-clear routed through the worker. `WorkerPool.leader_nccl` moves
  into the worker's state. `TpLoadedModel.leader_handle: TpHandle`
  replaces `Arc<Mutex<TpLeaderModel>>`. CUDA-only TP smoke deferred to
  next deploy.
- **Phase 4** (`b4f3576`) — GGUF + dense + TP shard loads move onto
  the worker. The `Job::TransferIn` / `Job::CloneLeaderComm` bridges
  from Phases 2/3 deleted; `SendComm` newtype no longer needed in the
  load path. `grep -rn spawn_blocking crates/neuron/src/harness/`
  returns only deliberate CPU-fallback hits after this PR.
