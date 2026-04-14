# cortex

A Rust reverse-proxy and fleet management layer for multi-node
[mistral.rs](https://github.com/EricLBuehler/mistral.rs) inference clusters.

## Problem

Running local LLMs across multiple GPU nodes (different VRAM tiers, different
model affinities) requires a unified API surface that:

- Presents a **single `/v1/models` catalogue** merging every model across every
  node.
- **Routes requests** to the correct node based on where a model is loaded (or
  *can* be loaded).
- Manages **model lifecycle** — unload cold models, reload on demand, pin
  critical ones — using the mistral.rs
  `/v1/models/{unload,reload,status}` HTTP API (PR #1828+).
- Translates between **OpenAI and Anthropic** request/response envelopes so
  every client in the homelab speaks whichever dialect it prefers.
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
                    │   cortex     │
                    │   (cortex-gateway)      │
                    │                     │
                    │  Router · Metrics   │
                    │  Evictor · Translate│
                    └──┬──────┬────────┬──┘
                       │      │        │
            ┌──────────▼┐  ┌──▼─────┐  ┌▼──────────┐
            │ gpu-large │  │gpu-med │  │ gpu-small │
            │ mistralrs │  │mistral │  │ mistralrs │
            │ serve     │  │rs serve│  │ serve     │
            │ :8080     │  │ :8080  │  │  :8080    │
            └───────────┘  └────────┘  └───────────┘
                  private network (.internal)
```

### Crates

| Crate | Purpose |
|---|---|
| `cortex-core` | Shared types: config, node/model state, metrics, OpenAI/Anthropic request/response envelopes |
| `cortex-gateway` | Axum HTTP server: proxy, router, evictor, metrics exporter |
| `cortex-agent` | Per-node sidecar: polls local mistralrs, reports to gateway, handles restart/defrag |
| `cortex-cli` | CLI entrypoint (`cortex serve`, `cortex status`, etc.) |

## Node setup

Each GPU node runs `mistralrs serve` with a multi-model config. Models are
declared but start **unloaded** — mistral.rs lazy-loads on first request and
the gateway can explicitly unload/reload via the HTTP API.

Example node systemd unit:

```ini
# /etc/systemd/system/mistralrs.service
[Unit]
Description=mistral.rs inference server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=/usr/local/bin/mistralrs serve \
    --from-config /etc/mistralrs/config.toml \
    --port 8080
Restart=on-failure
RestartSec=5
Environment=CUDA_VISIBLE_DEVICES=0,1

[Install]
WantedBy=multi-user.target
```

## Gateway config

```toml
# cortex.toml
[gateway]
listen = "0.0.0.0:8000"
metrics_listen = "0.0.0.0:9100"

[eviction]
strategy = "lru"        # lru | priority
defrag_after_cycles = 50

[[nodes]]
name = "gpu-large"
endpoint = "http://gpu-large.internal:8080"
vram_mb = 49_152        # e.g. 2x RTX 4090
pinned = ["your-org/large-model"]

[[nodes]]
name = "gpu-medium"
endpoint = "http://gpu-medium.internal:8080"
vram_mb = 24_576        # e.g. RTX 4090
pinned = ["your-org/medium-model"]

[[nodes]]
name = "gpu-small"
endpoint = "http://gpu-small.internal:8080"
vram_mb = 12_288        # e.g. RTX 3060
pinned = ["your-org/embedding-model"]
```

## Building

```sh
cargo build --release
```

## Running

```sh
# start the gateway
cortex serve --config cortex.toml

# check fleet status
cortex status

# list all models across nodes
curl http://localhost:8000/v1/models
```

## License

GPL-3.0
