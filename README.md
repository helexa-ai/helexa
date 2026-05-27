# cortex

A Rust reverse-proxy and fleet management layer for multi-node GPU inference
clusters. Cortex sits in front of one or more `neuron` daemons (each running
candle-based inference on a local GPU host) and presents a unified OpenAI +
Anthropic compatible API surface.

## Problem

Running local LLMs across multiple GPU nodes (different VRAM tiers, different
model affinities) requires a unified API surface that:

- Presents a **single `/v1/models` catalogue** merging every model that can be
  served by any neuron in the fleet.
- **Routes requests** to the correct node based on where a model is loaded
  (or can be loaded), handling cold-load and eviction transparently.
- Manages **model lifecycle** — load on demand, unload cold models, pin
  critical ones — by calling each neuron's `/models/{load,unload}` API.
- Translates between **OpenAI and Anthropic** request/response envelopes so
  every client speaks whichever dialect it prefers.
- Captures **per-request metrics** (tokens, tok/s, TTFT, latency) and exposes
  them as Prometheus counters/histograms.

## Architecture

```
┌──────────────┐  ┌──────────┐  ┌────────────┐  ┌────────────┐
│ Claude Code  │  │ Zed/IDE  │  │ Tidal / mm │  │ curl / etc │
└──────┬───────┘  └─────┬────┘  └──────┬─────┘  └──────┬─────┘
       │                │              │               │
       └────────────────┴──────┬───────┴───────────────┘
                               │
                    ┌──────────▼──────────┐
                    │      cortex         │
                    │  (cortex-gateway)   │
                    │                     │
                    │  Router · Metrics   │
                    │  Evictor · Translate│
                    └──┬──────┬────────┬──┘
                       │      │        │
            ┌──────────▼┐  ┌──▼─────┐  ┌▼──────────┐
            │  neuron   │  │ neuron │  │  neuron   │
            │  :13131   │  │ :13131 │  │  :13131   │
            │  candle   │  │ candle │  │  candle   │
            └───────────┘  └────────┘  └───────────┘
                  private network (.internal)
```

### Crates

| Crate | Purpose |
|---|---|
| `cortex-core` | Shared types: config, node/model state, metrics, OpenAI/Anthropic envelopes, harness trait, discovery types |
| `cortex-gateway` | Axum HTTP server: proxy, router, evictor, poller, metrics exporter |
| `neuron` | Per-node daemon: GPU discovery, in-process candle inference, model lifecycle API |
| `cortex-cli` | CLI entrypoint (`cortex serve`, `cortex status`, etc.) |

## Node setup

Each GPU node runs `neuron` (listening on `:13131`). Neuron uses
huggingface/candle for in-process inference — there is no external
inference subprocess to manage.

Inside the daemon, every CUDA device gets one dedicated OS thread
(named `cuda-dev-N`) that owns the device's CUDA context for the
daemon's lifetime. Model loads, forward passes, KV-cache resets,
NCCL collectives, VRAM queries, and unloads all route through that
thread via a job channel; tensors never escape it alive. This pins
context binding to a known thread, makes the CUDA Drop contract
structurally safe, and isolates driver-error poisoning to one worker
rather than the whole process. See `CLAUDE.md` for the design
rationale and `crates/neuron/src/harness/device_worker/` for the code.

The neuron RPM (`helexa-neuron`) ships a systemd unit:

```sh
dnf copr enable helexa/helexa
dnf install helexa-neuron
systemctl enable --now neuron
```

## Gateway config

```toml
# /etc/cortex/cortex.toml
[gateway]
listen = "0.0.0.0:31313"
metrics_listen = "0.0.0.0:31314"

[eviction]
strategy = "lru"        # lru | priority
defrag_after_cycles = 50

[[neurons]]
name = "beast"
endpoint = "http://beast.internal:13131"

[[neurons]]
name = "benjy"
endpoint = "http://benjy.internal:13131"
```

Model placement profiles live in `models.toml` — see `models.example.toml`.

## Building

```sh
cargo build --release
```

## CI

Every push triggers format, lint, and test checks. Ensure these pass
locally before pushing:

```sh
cargo fmt --check --all                    # must be clean
cargo clippy --workspace -- -D warnings   # warnings are errors
cargo test --workspace                     # all tests must pass
```

Tagged releases (`v*`) additionally build SRPMs for both `cortex` and
`helexa-neuron` and publish to COPR.

## Running

```sh
# start the gateway
cortex serve --config /etc/cortex/cortex.toml

# check fleet status
cortex status

# list all models across nodes
curl http://localhost:31313/v1/models
```

## License

GPL-3.0
