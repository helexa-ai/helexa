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

## Current status

**Scaffold phase** — crate structure, types, and handler stubs are in place.
The following needs implementation:

1. **cortex-core**: Flesh out OpenAI/Anthropic envelope types with all fields
   needed for chat completions (streaming + non-streaming)
2. **cortex-gateway/proxy.rs**: Implement streaming HTTP proxy with SSE passthrough
3. **cortex-gateway/router.rs**: Model-to-node routing with fallback to least-loaded
4. **cortex-gateway/evictor.rs**: LRU eviction with pinning support
5. **cortex-gateway/poller.rs**: Background polling of node `/v1/models` endpoints
6. **cortex-gateway/handlers.rs**: Wire up axum routes to proxy logic
7. **cortex-core/translate.rs**: OpenAI <-> Anthropic request/response translation
8. **cortex-agent**: Sidecar for VRAM defrag restarts (lower priority)
9. **Integration tests**: Mock mistralrs backends, test routing + eviction
