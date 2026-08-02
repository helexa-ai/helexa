# helexa

**Near-frontier AI for mortals.**

helexa is a self-hosted LLM serving stack, written in Rust, for people
who run open-weight models on their own consumer GPUs. It has two
components:

- **cortex** — the per-operator control plane and LLM proxy. It sits in
  front of your GPU fleet and presents a unified OpenAI + Anthropic
  compatible API surface, handling model routing, lifecycle management
  (load / unload / evict), request translation, and metrics.
- **neuron** — the per-host LLM harness. One instance runs on every GPU
  host, serving candle-based in-process inference and managing local
  hardware discovery and model lifecycle.

## Why

Two principles constrain everything in this repository:

1. **Frontier or close to it.** helexa serves the open-weight models
   that get nearest to frontier capability — not every architecture
   ever published.
2. **Consumer hardware.** Everything must run on the cards mortals can
   actually buy: a 3060 here, a 4090 there, a 5090 if you got lucky.
   Mixed VRAM tiers across mismatched boxes are the expected topology,
   not a degraded case.

GPU acquisition is harder than it was a year ago, and the gap between
what cloud providers charge and what your own silicon costs keeps
widening. The intersection of those two principles — near-frontier
models, squeezed onto hardware you own — is helexa's entire niche.

The secondary objective is **predictable consumption**. If you own the
hardware, your tooling shouldn't break because a cloud provider changed
billing, deprecated a model, or reshaped an API. cortex's OpenAI and
Anthropic surfaces are a stability contract: point your editor, agent,
or CLI at it once, and it keeps working.

## What helexa is not

This is an intentionally different path from vLLM, SGLang, and peers —
not a smaller version of them. Out of scope, permanently:

- Any-model breadth. Architectures are ported because they're at or
  near the frontier, not to complete a compatibility matrix.
- Datacenter-class scheduling. No sophisticated continuous-batching /
  paged-attention machinery — the workload is a handful of operators
  and their agents, not 200 QPS.
- Wrapping external inference engines. neuron builds directly on
  [candle](https://github.com/huggingface/candle); every model
  architecture it serves is implemented in this repository, ported
  against the HuggingFace reference.

One thing that is *not* a principle: CUDA exclusivity. All high-end
consumer hardware is in scope. helexa is CUDA-only today because
that's the hardware on the bench — nothing ships untested — and ROCm
or other consumer accelerators join as soon as there's real hardware
to build against.

In scope, and where the engineering effort goes: aggressive
quantization (GGUF Q4_K_M / Q6_K / Q8_0), NCCL tensor parallelism
across heterogeneous consumer GPUs, careful CUDA failure handling, and
single-request latency — the performance that one operator at a
keyboard actually feels.

## Architecture

```
┌──────────────┐  ┌──────────┐  ┌────────────┐  ┌────────────┐
│ Claude Code  │  │ Zed/IDE  │  │ Tidal / mm │  │ curl / etc │
└──────┬───────┘  └─────┬────┘  └──────┬─────┘  └──────┬─────┘
       │                │              │               │
       └────────────────┴──────┬───────┴───────────────┘
                               │  OpenAI + Anthropic APIs
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

cortex discovers each neuron's hardware (devices, VRAM, compute
capability) at runtime and matches it against a model catalogue
(`models.toml`) to decide placement: which models fit where, what to
evict when VRAM is tight, where to route a request right now. Adding a
GPU host to the fleet is one `[[neurons]]` entry — no device specs in
config.

### Crates

| Crate | Purpose |
|---|---|
| `cortex-core` | Shared types: config, node/model state, metrics, OpenAI/Anthropic envelopes, harness trait, discovery types |
| `cortex-gateway` | Axum HTTP server: proxy, router, evictor, poller, metrics exporter |
| `neuron` | Per-host daemon: GPU discovery, in-process candle inference, NCCL tensor parallelism, model lifecycle API |
| `cortex-cli` | CLI entrypoint (`cortex serve`, `cortex status`, etc.) |
| `helexa-acp` | Agent Client Protocol bridge — connects ACP editors (Zed, etc.) to any OpenAI-compatible endpoint, cortex by default |

## The engine

neuron runs inference in-process on candle — there is no external
inference server to babysit. The parts that earn their keep:

- **Per-device worker threads.** Every CUDA device gets one dedicated
  OS thread that owns its CUDA context for the daemon's lifetime. All
  loads, forward passes, KV-cache resets, NCCL collectives, VRAM
  queries, and unloads route through it; tensors never escape it
  alive. Context binding is pinned to a known thread, the CUDA `Drop`
  contract is structurally safe, and a driver error poisons one worker
  — visibly — instead of hanging the whole process.
- **Tensor parallelism on consumer cards.** Megatron-style row/column
  parallel layers with NCCL all-reduce, spanning the mismatched GPUs
  you actually have. A step watchdog aborts wedged collectives instead
  of letting a request hang forever.
- **Text-to-image.** Z-Image-Turbo (6B S3-DiT, Apache 2.0) served
  candle-native through the same device-worker discipline: OpenAI
  `/v1/images/generations` end to end, ~11 s for a 1024x1024 image on
  an RTX 4090, metered in megapixel-steps.
- **Current model focus: the Qwen3 family** — dense and GGUF-quantized,
  including the hybrid linear-attention (Gated DeltaNet) generation.
  Vision support is in progress. Each architecture is ported against
  its HuggingFace reference implementation.

See `CLAUDE.md` for design rationale and
`crates/neuron/src/harness/device_worker/` for the worker narrative.

## Install

Pre-built RPMs for Fedora:

```sh
dnf copr enable helexa/helexa
dnf install cortex            # on the gateway host
dnf install helexa-neuron     # on each GPU host
systemctl enable --now cortex   # or neuron, respectively
```

## Configure

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

Model placement profiles (VRAM requirements, quant, device minimums,
pinning) live in `models.toml` — see `models.example.toml`.

## Run

```sh
# start the gateway
cortex serve --config /etc/cortex/cortex.toml

# check fleet status
cortex status

# one catalogue across every node
curl http://localhost:31313/v1/models
```

## Tailoring model behaviour

System prompts are **application-owned**. Send yours through the standard
field for whichever API you speak, and the serving chain — edge → router →
cortex → neuron — passes it to the model **verbatim**.

```sh
# OpenAI chat completions
curl http://localhost:31313/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model": "helexa/balanced",
       "messages": [{"role": "system", "content": "Reply only in French."},
                    {"role": "user",   "content": "Good morning"}]}'

# OpenAI responses — the `instructions` field is the system slot
curl http://localhost:31313/v1/responses \
  -H 'content-type: application/json' \
  -d '{"model": "helexa/balanced",
       "instructions": "Reply only in French.",
       "input": "Good morning"}'

# Anthropic messages — top-level `system`, string or content-block array
curl http://localhost:31313/v1/messages \
  -H 'content-type: application/json' \
  -d '{"model": "helexa/balanced", "max_tokens": 256,
       "system": "Reply only in French.",
       "messages": [{"role": "user", "content": "Good morning"}]}'
```

### The passthrough guarantee

helexa will never, to any request you proxy through it:

- inject a system prompt, house style or preamble of its own,
- rewrite, truncate or reorder the one you sent,
- supply a default when you send none.

If you send no system prompt, the model receives none. Behaviour you did
not ask for is a bug — please report it.

Streaming and non-streaming behave identically, and the guarantee holds on
every surface above. Two details worth knowing:

- **Several system messages** are all forwarded, unmerged and in order.
  The model sees the last one last, so in practice the last instruction
  wins. Send one message if you want certainty.
- **`/no_think`** in a prompt is a Qwen-family convention the *model*
  interprets to skip its reasoning block. It is the model's feature, not
  ours — helexa neither adds nor strips it. Note it currently suppresses
  reasoning on `/v1/chat/completions` but **not** on `/v1/responses`,
  where a reasoning model may think regardless (#223); with a small
  `max_output_tokens` the whole budget can go on reasoning and the reply
  comes back empty with `status: "incomplete"`. Give Responses requests
  room (a few hundred tokens) when the model reasons.

### Why it works this way

The ecosystem serves an unenumerable diversity of workloads through
OpenAI- and Anthropic-compatible APIs. An operator cannot curate prompts
for use cases they will never see, and a proxy that quietly edits your
payload makes model behaviour impossible to reason about. So the split is:
**applications own the prompt, operators own the fleet.**

If you specifically want centrally-managed prompts across your own
workloads, run your own helexa mesh — it is open source, and that is a
different deployment from the shared helexa.ai ecosystem.

Operators: this is a contract, not a default you may flip. cortex and
helexa-router proxy inference bodies without adding to them; nothing in
the chain is a place to put prompt content.

## Build from source

```sh
cargo build --release
```

CI runs on every push; keep it green locally:

```sh
cargo fmt --check --all                    # must be clean
cargo clippy --workspace -- -D warnings   # warnings are errors
cargo test --workspace                     # all tests must pass
```

Tagged releases (`v*`) build SRPMs for `cortex` and `helexa-neuron`
and publish to COPR.

## Status

Pre-1.0 and moving fast. The gateway path (routing, eviction,
translation, metrics) is stable and tested; the candle-native engine
is under active development — expect the supported-model list to track
the open-weight frontier, deliberately narrowly.

Development happens at <https://git.lair.cafe/helexa/helexa>;
<https://github.com/helexa-ai/helexa> is a read-only mirror.

## License

GPL-3.0
