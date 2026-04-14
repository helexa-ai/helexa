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
- Gateway listens on port 8000 (API) and 9100 (metrics)
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

### Phase 4: Eviction

**Goal:** When a request targets a model that requires loading and the
node is at capacity, cortex evicts the LRU non-pinned model first.

**Files to change:**
- `cortex-gateway/src/evictor.rs` — `evict_lru_on_node` is implemented;
  integrate it into the request path
- `cortex-gateway/src/router.rs` — add a `resolve_with_eviction` path
  that calls the evictor when the target model is unloaded and the node
  has no free VRAM headroom
- `cortex-gateway/src/handlers.rs` — update `last_accessed` on
  `ModelEntry` for every successful request (drives LRU ordering)
- `tests/` — eviction test:
  1. Mock node reports 2 loaded models, 0 free VRAM
  2. Request arrives for a 3rd model (unloaded on that node)
  3. Assert cortex calls `POST /v1/models/unload` on the LRU model
  4. Assert the original request is then forwarded (lazy load)
  5. Assert pinned models are never evicted

**Done when:** Eviction test passes. `lifecycle_cycles` increments.
Defrag warning fires at threshold.

### Phase 5: Anthropic translation

**Goal:** `POST /v1/messages` accepts Anthropic-format requests, proxies
to mistral.rs in OpenAI format, returns Anthropic-format responses.

**Files to change:**
- `cortex-core/src/translate.rs` — the scaffold has a working
  `anthropic_to_openai` and `openai_to_anthropic`. Extend to handle:
  - Multi-block content (images, tool use, tool results)
  - `stop_reason` mapping (`end_turn`, `max_tokens`, `tool_use`)
  - Usage token counts
- `cortex-gateway/src/handlers.rs` — the `anthropic_messages` handler
  currently has TODO comments for response translation and streaming.
  Implement non-streaming first (buffer upstream response, translate,
  return). Then streaming (convert OpenAI SSE to Anthropic SSE event
  types: `message_start`, `content_block_start`, `content_block_delta`,
  `content_block_stop`, `message_delta`, `message_stop`).
- `tests/` — round-trip test:
  1. Send Anthropic-format request to cortex
  2. Assert the proxied request to mock backend is valid OpenAI format
  3. Assert the response back to the client is valid Anthropic format

**Done when:** Non-streaming Anthropic round-trip test passes. Streaming
is a bonus — flag it as a follow-up if complex.

### Phase 6: Metrics instrumentation

**Goal:** Every proxied request emits Prometheus metrics. `/metrics`
on port 9100 returns valid Prometheus text format.

**Files to change:**
- `cortex-gateway/src/proxy.rs` or `cortex-gateway/src/handlers.rs` —
  wrap each proxy call with timing instrumentation:
  - `Instant::now()` before the request, compute duration after
  - Parse `usage` from the response (non-streaming) or final chunk
    (streaming) for token counts
  - Emit: `metrics::histogram!("cortex_request_duration_seconds", ...)`
    with labels `model` and `node`
  - Emit: `metrics::counter!("cortex_requests_total", ...)` 
  - Emit cold start, eviction, and error counters
- `cortex-gateway/src/metrics.rs` — already installs the exporter;
  verify the described metrics appear
- `tests/` — test that after a proxied request, the `/metrics`
  endpoint contains the expected metric names

**Done when:** `curl localhost:9100/metrics` shows request counters
and duration histograms after proxying a test request.

### Phase 7 (lower priority): Agent sidecar

**Goal:** Per-node binary that handles VRAM defrag restarts and
reports real VRAM usage via `nvidia-smi`.

This is deferred. The gateway handles the critical path (model
lifecycle) entirely via the mistral.rs HTTP API. The agent adds
operational polish: automatic process restart when `lifecycle_cycles`
exceeds threshold, real VRAM reporting (vs. estimates), and
potentially GPU temperature/power monitoring.

**Defer until:** Phases 1-6 are merged and running in production.
