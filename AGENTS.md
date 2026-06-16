# AGENTS.md — helexa/cortex

## Project Overview

helexa is a self-hosted LLM serving stack for multi-node GPU inference clusters. It has two components:

- **cortex** — the per-operator control plane and LLM proxy. A Rust reverse-proxy that sits in front of the fleet and presents a unified OpenAI + Anthropic compatible API surface. It handles model routing, lifecycle management (load/unload/evict), request translation, and metrics collection.
- **neuron** — the per-host LLM harness. One instance runs on every GPU host, serving candle-based in-process inference and managing local hardware discovery and model lifecycle.

## Repository Layout

```
cortex/
├── Cargo.toml              # workspace root (Rust 2024 edition, GPL-3.0)
├── cortex.example.toml     # example gateway config
├── models.example.toml     # example model catalogue
├── neuron.example.toml     # example neuron config
├── README.md               # public-facing documentation
├── CLAUDE.md               # detailed design rationale and implementation history
├── AGENTS.md               # ← you are here
├── cortex.spec             # RPM spec for cortex
├── helexa-neuron.spec      # RPM spec for neuron (renamed to avoid Fedora collision)
├── rpm/                    # prerelease RPM specs
│   ├── cortex-prerelease.spec
│   ├── helexa-neuron-prerelease.spec
│   └── helexa-bench-prerelease.spec
├── data/                   # systemd units and example configs for packaging
│   ├── cortex.service
│   ├── neuron.service
│   ├── cortex.example.toml
│   ├── neuron.example.toml
│   └── models.example.toml
└── crates/
    ├── cortex-core/            # shared types, config, envelopes
    │   └── src/
    │       ├── lib.rs
    │       ├── build_info.rs   # BuildInfo type for /version endpoint
    │       ├── config.rs       # figment-based config structs
    │       ├── catalogue.rs    # ModelProfile, placement matching
    │       ├── discovery.rs    # DeviceInfo, DiscoveryResponse
    │       ├── harness.rs      # Harness trait, HarnessConfig, HarnessHealth
    │       ├── node.rs         # NodeState, ModelStatus
    │       ├── openai.rs       # OpenAI request/response types
    │       ├── anthropic.rs    # Anthropic request/response types
    │       ├── translate.rs    # OpenAI <-> Anthropic translation
    │       └── metrics.rs      # RequestMetrics, histogram helpers
    ├── cortex-gateway/         # the HTTP proxy server
    │   └── src/
    │       ├── lib.rs
    │       ├── state.rs        # CortexState: Arc<RwLock<...>>
    │       ├── router.rs       # model -> node routing logic
    │       ├── proxy.rs        # streaming HTTP proxy to backends
    │       ├── evictor.rs      # LRU/priority eviction logic
    │       ├── poller.rs       # background task polling neuron status
    │       ├── handlers.rs     # axum handlers (chat, completions, models, etc.)
    │       └── metrics.rs      # prometheus exporter endpoint
    ├── cortex-cli/             # CLI entrypoint
    │   └── src/main.rs         # binary: `cortex`
    ├── neuron/                 # per-host LLM daemon (replaces cortex-agent)
    │   ├── Cargo.toml          # features: cuda, cudnn, flash-attn, cuda-integration
    │   ├── build.rs            # compiles CUDA kernels, emits build metadata
    │   └── src/
    │       ├── main.rs         # binary: `neuron`
    │       ├── discovery.rs    # nvidia-smi parsing, device enumeration
    │       ├── health.rs       # runtime GPU polling
    │       ├── api.rs          # HTTP handlers for /discovery, /models, etc.
    │       ├── version.rs      # GET /version endpoint with BuildInfo
    │       ├── models.rs       # local model lifecycle orchestration
    │       └── harness/        # in-process candle inference
    │           ├── device_worker/  # per-device CUDA worker threads
    │           │   ├── mod.rs      # canonical narrative for worker architecture
    │           │   ├── jobs.rs     # Job enum, dispatch handlers
    │ │           └── dispatch.rs   # DeviceWorkerState struct
    │           ├── candle.rs       # candle model implementation
    │           └── tp/             # tensor parallelism
    │               └── worker.rs   # TP worker subprocesses
    ├── helexa-acp/             # Agent Client Protocol bridge (Apache-2.0)
    │   └── src/main.rs         # binary: `helexa-acp`, self-contained (no workspace deps)
    └── helexa-bench/           # benchmark harness
        └── src/main.rs         # binary: `helexa-bench`, SQLite-backed, version-aware
```

## Key Design Decisions

### Architecture
- **cortex** is the control plane. It exposes the unified API, routes requests, manages model lifecycle across the fleet, and collects metrics.
- **neuron** is the node plane. One instance runs on every GPU host. It discovers local hardware, manages in-process candle inference, handles NCCL tensor parallelism, and reports runtime state.
- cortex never shells out to `nvidia-smi`, never touches systemd units, and never talks directly to a harness. It talks only to neurons via HTTP API on port 13131.

### Per-device worker thread (neuron)
Every CUDA device gets one dedicated OS thread that owns its `CudaContext` for the daemon's lifetime. All CUDA operations route through this thread via a `std::sync::mpsc` job channel. Tensors never escape the worker thread alive. Inference replies carry `Vec<f32>` CPU-side logits; sampled tokens come back as `u32`. The opaque `ArchHandle(u64)` and `TpHandle(u64)` are indices into the worker's state slab, not pointers.

CPU loads (`Device::Cpu` fallback) keep the legacy `tokio::task::spawn_blocking + Arc<Mutex<ModelArch>>` path — there's no context to own and the channel hop would only add latency. Four `spawn_blocking` references in `harness/candle.rs` are deliberate CPU fallback.

### candle-native (not mistral.rs)
neuron builds directly on [candle](https://github.com/huggingface/candle). Every model architecture it serves is implemented in this repository, ported against the HuggingFace reference. No external inference server to babysit. The Harness trait remains as an internal seam for adding future engines (vision/audio/diffusion) but its only implementation is in-process candle.

### Streaming proxy
Chat completions are proxied as SSE streams. The gateway must:
1. Parse the inbound request to extract the model name
2. Route to the correct backend neuron
3. Stream the response back, capturing token timing for metrics
4. NOT buffer the full response — true streaming passthrough

### Anthropic translation
When a request arrives at `/v1/messages` (Anthropic format), the gateway translates it to OpenAI format before proxying to neuron, then translates the response back. This is stateless envelope transformation. Non-streaming round-trip is implemented; streaming SSE translation deferred.

### Eviction
The evictor runs as a background task. Before loading a model on a node where VRAM is tight:
1. Check if the model is already loaded elsewhere → route there instead
2. Find the LRU model on the target node (excluding pinned models)
3. Call `POST {neuron}/models/unload` on that model
4. The incoming request's lazy-load triggers the new model load

### Metrics
Per-request: model, node, prompt_tokens, completion_tokens, total_tokens, tok_per_sec, time_to_first_token_ms, total_latency_ms. Exposed as Prometheus histograms/counters on a separate port (31314).

## Tech Stack

- **Rust 2024 edition** — workspace with 6 crates
- **Axum 0.8** — HTTP framework
- **reqwest** — HTTP client for proxying to backends
- **figment** — config loading (TOML + env vars)
- **tokio** — async runtime
- **metrics + metrics-exporter-prometheus** — observability
- **tracing** — structured logging
- **candle** — in-process inference engine (neuron only, with CUDA support)
- **cudarc** — patched for neuron's needs (see workspace `[patch]`)
- **clap** — CLI parsing
- **rusqlite** (bundled) — helexa-bench SQLite system-of-record

## Build Commands

```sh
cargo build --release           # build all crates
cargo run -p cortex-cli -- serve    # run the gateway
cargo test                      # run all tests
cargo clippy --workspace        # lint
```

### neuron Features
- `cuda`: Enables CUDA acceleration in candle and cudarc/nccl bindings. Without it, falls back to CPU.
- `cudnn`: Use cuDNN for convolution/attention kernels (requires `cuda`).
- `flash-attn`: FlashAttention kernels (requires `cuda`).
- `cuda-integration`: Reserved for GPU-only integration tests (requires multiple CUDA devices + libnccl).

### Build Scripts
- `neuron/build.rs`: Compiles CUDA kernels (`src/cuda/*.cu`) using `cudaforge::KernelBuilder` when `cuda` feature is enabled. Handles compute capability checks (sm_<80 disables bf16 intrinsics). Also captures build metadata: git SHA, dirty flag, timestamp, rustc version, profile, features, candle-core version.

## CI

Gitea Actions runs on every push to any branch. All three checks must pass before merging:

```sh
cargo fmt --check --all                    # formatting
cargo clippy --workspace -- -D warnings   # lint (warnings are errors)
cargo test --workspace                     # tests
```

Run these locally before pushing. `cargo fmt --all` fixes formatting automatically. Clippy warnings must be resolved, not suppressed with `#[allow(...)]` unless there is a clear rationale.

Tagged releases (`v*`) build SRPMs for `cortex`, `helexa-neuron`, and `helexa-bench` and publish to COPR (`helexa/helexa`). Build metadata SHA injection: CI sets `HELEXA_BUILD_SHA=$(git rev-parse HEAD)`.

## Environment

- Targets Fedora 43 (systemd, SELinux enforcing)
- Nodes communicate over a private network (e.g. WireGuard mesh)
- cortex listens on port 31313 (API) and 31314 (metrics)
- neuron listens on port 13131 on each GPU host
- TLS terminated at gateway or via nginx; internal traffic is plaintext over WireGuard

## Conventions

- Error handling: `anyhow` for binaries, `thiserror` for library crates
- No `unwrap()` in library code; `expect()` only with clear rationale
- All public types derive `Debug, Clone, Serialize, Deserialize` where sensible
- Config structs use `figment` with TOML as primary source, env vars as override
- Prefer `Arc<RwLock<...>>` for shared fleet state; minimize lock duration
- SSE streaming uses `tokio_stream` + `eventsource-stream` for parsing
- Log at `info` for request routing, `debug` for proxy details, `warn` for eviction and node health, `error` for proxy failures

## Testing

### Gateway tests
Use mock neurons spawned via axum in `crates/cortex-gateway/tests/common/mod.rs`. Helpers: `spawn_mock_backend()`, `spawn_gateway()`.

### neuron integration tests
- Numerical reference tests (`numerical_reference.rs`) require `NEURON_REF_MODEL_PATH` env var pointing to a HF snapshot directory. Fixtures are f32-based for precision validation against HuggingFace transformers.
- CUDA integration tests (`tp_worker_lifecycle_cuda.rs`) gated behind `cuda-integration` feature; requires 2+ CUDA devices (e.g., 2x RTX 5090).

### Metrics testing
Use `install_test_recorder()` in test code to capture metrics without the HTTP listener.

## helexa-bench

A continuous, version-aware benchmark harness. Hits each neuron directly on `:13131`, exercises each warm model with a Scenario suite (chat-latency family), and records results into SQLite stamped with the neuron's full `BuildInfo`. The loop is version-aware: skips any (target, build SHA, model, scenario) cell already at `samples_per_version`.

Packaged as `helexa-bench` RPM (prebuilt-binary spec). One systemd unit, typically on the metrics host.

## helexa-acp

Agent Client Protocol bridge — connects ACP editors (Zed, etc.) to any OpenAI-compatible endpoint, cortex by default. Intentionally self-contained: no workspace crate dependencies. Uses `agent-client-protocol` with `unstable_session_model` feature for Zed model picker support. Licensed Apache-2.0 (workspace is GPL-3.0).

## RPM Packaging

- `cortex.spec` — installs the `cortex` binary
- `helexa-neuron.spec` — installs the `neuron` binary under package name `helexa-neuron` (renamed to avoid Fedora's NEURON neural-simulation package collision)
- Systemd units in `data/cortex.service`, `data/neuron.service`
- Example configs: `cortex.example.toml`, `neuron.example.toml`, `models.example.toml`

Install:
```sh
dnf copr enable helexa/helexa
dnf install cortex                # gateway host
dnf install helexa-neuron         # GPU nodes
```

## Configuration Files

### cortex.toml (gateway)
```toml
[gateway]
listen = "0.0.0.0:31313"
metrics_listen = "0.0.0.0:31314"

[eviction]
strategy = "lru"          # lru | priority
defrag_after_cycles = 50

[[neurons]]
name = "beast"
endpoint = "http://beast.internal:13131"
```

### models.toml (catalogue)
```toml
[[models]]
id = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
harness = "candle"
quant = "Q4_K_M"
vram_mb = 19000
min_devices = 2
min_device_vram_mb = 10000
pinned_on = ["beast"]       # optional: never evict from these neurons
```

### neuron.toml (per-host)
Configured via figment + env override. See `neuron.example.toml` for reference.

## neuron API Endpoints

```
GET  /discovery        → hardware discovery (hostname, OS, CUDA, devices, harnesses)
GET  /health           → runtime GPU stats (VRAM, utilization, temperature)
GET  /models           → loaded/unloaded models with VRAM usage
POST /models/load      → load a model with spec (quant, TP, devices)
POST /models/unload    → unload a model, freeing device memory
GET  /models/{id}/endpoint → inference URL for a model
GET  /version          → build metadata (SHA, features, candle version, etc.)
```

## Sources of Truth

When prose documentation conflicts with code, trust:
1. Executable configuration (`*.toml`, `Cargo.toml` features)
2. Type definitions in `cortex-core/`
3. Test files in `crates/*/tests/` and `*/src/**/*_test.rs`
4. `CLAUDE.md` for historical design rationale