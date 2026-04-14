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

### Phase 7 (lower priority): Agent sidecar

**Goal:** Per-node binary that handles VRAM defrag restarts and
reports real VRAM usage via `nvidia-smi`.

This is deferred. The gateway handles the critical path (model
lifecycle) entirely via the mistral.rs HTTP API. The agent adds
operational polish: automatic process restart when `lifecycle_cycles`
exceeds threshold, real VRAM reporting (vs. estimates), and
potentially GPU temperature/power monitoring.

**Defer until:** Phases 1-6 are merged and running in production.
