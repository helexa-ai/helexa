---
title: Configuring neurons
sidebar_label: neurons
description: The per-host daemon — weights cache, default models, concurrency and VRAM guards.
---

# Configuring neurons

A neuron owns one host's GPUs. It discovers the hardware, fetches and
loads weights, serves inference in-process, and reports what it is
doing. Everything below lives in `/etc/neuron/neuron.toml`.

```toml
port = 13131

[[harnesses]]
name = "candle"

[harness.candle]
# see below
```

Environment variables override file settings with a `NEURON_` prefix,
which is the right way to do per-host tuning — a systemd drop-in keeps
the shared config file identical across the fleet.

## Where weights live

The weights cache is the single most important setting on a GPU host,
because it is large and because putting it on the wrong disk makes every
cold load slow.

Resolution order, first hit wins:

1. `hf_cache` in `neuron.toml`
2. `HF_HUB_CACHE`
3. `HF_HOME` (with `/hub` appended)
4. `~/.cache/huggingface/hub`

`HF_HUB_CACHE` uses the same convention as the Python `huggingface_hub`
library, so an existing cache can be shared with other tooling rather
than duplicated.

```ini
# /etc/systemd/system/neuron.service.d/local.conf
[Service]
Environment=HF_HUB_CACHE=/fast-nvme/llm-cache
```

Two things that bite:

- **The service user must be able to read the cache.** neuron runs as
  its own system user, so a cache populated by a human account needs
  permissions that survive new downloads — set a *default* ACL on the
  directory, not just on what is already there. Weights that arrive
  unreadable fail the load with a permission error that reads like a
  missing file.
- **Cold loads are disk-bound.** A 50 GB model off spinning rust is a
  very different experience from the same model on NVMe.

## Default models

Models listed as defaults load when the daemon activates, so the host
comes back serving rather than waiting for the first request to pay the
load cost:

```toml
[[default_models]]
model_id = "Qwen/Qwen3.6-27B"
harness = "candle"
quant = "q6k"
tensor_parallel = 2
devices = [0, 1]
```

`quant` selects in-situ quantisation at load — the weights on disk stay
as distributed, and the quantised form exists only in VRAM. `q6k` is a
good default where bf16 does not fit; `q8_0` is closer to bf16 quality
at a smaller saving.

`tensor_parallel` splits one model across several GPUs, which is how a
model too large for a single card is served at all. It requires
safetensors weights, and support is architecture-specific — not every
model that loads on one GPU can be split across two.

## Concurrency

Each loaded model has bounded admission control:

```toml
[harness.candle.admission]
max_in_flight = 8        # concurrent requests actually running
max_queue_depth = 8      # waiting before rejection
max_wait_secs = 30       # how long a queued request waits
max_per_principal = 2    # fair share, so one caller cannot monopolise
```

Over-capacity requests are rejected promptly with `429`/`503` and a
`Retry-After` rather than being allowed to queue invisibly. A fast
rejection a client can retry is far better than a request that hangs
until something times out.

`max_in_flight` above 1 enables batched decode on architectures that
support it, so concurrent requests share a decode step instead of
serialising. Raise it on big cards, leave it at 1 on small ones — the
KV cache for several concurrent long contexts is itself substantial.

`max_per_principal` is what stops one busy client consuming the whole
model.

### Anonymous callers

Fair share is keyed on the account and key cortex resolves from the
bearer token. A caller that sends no credential — or one that does not
resolve, which `require_auth = false` deliberately tolerates — has no
key to be shared out, so `max_per_principal` cannot bind it.

Anonymous traffic is therefore served from the capacity left over once
identified traffic is satisfied. It never takes a seat while an
authenticated request is waiting for one, whoever arrived first, and it
cannot hold every seat at once:

```toml
anon_max_in_flight = 7   # unset: max_in_flight - 1, floored at 1
anon_max_pending = 15    # unset: max_in_flight + max_queue_depth - 1
```

The defaults hold back one seat and one queue place, which is what
bounds how long an authenticated request can be made to wait: a request
already running cannot be preempted, so without a reserved seat an
anonymous burst arriving during an idle moment locks the model for as
long as all of it takes to finish. On a single-seat model the floor
keeps anonymous callers served — priority alone decides who wins there.

Set `anon_max_in_flight = 0` to refuse anonymous traffic outright, or
`require_auth = true` on the gateway to make attribution universal.
Anonymous callers still contend with each other, and under sustained
authenticated load they are refused rather than served slowly — the
refusal is a retryable `503`, so a client that backs off gets served
when the load passes.

Watch it with `cortex_model_anon_in_flight` against
`cortex_model_in_flight` (how much of a model's load is unattributable)
and `cortex_model_rejections_total{reason="anon_yield"}`. That last
counter rising while `reason="wait_timeout"` stays flat means the
reservation is doing its job, not that the model is overloaded.

## VRAM guards

neuron refuses work it cannot finish rather than crashing partway:

- a **static floor** of free VRAM below which prefill is refused
  (`NEURON_MIN_FREE_VRAM_MB`, default 1500)
- a **length-aware check** that estimates the KV cache the request will
  actually need and refuses if it will not fit

The second is what protects against long prompts specifically: a request
can pass the static floor and still be impossible, because KV grows with
prompt *and* generation length. See [context
limits](/docs/operating/context-limits) for how the numbers relate.

A host that cannot currently serve a model reports that on `/health`,
and cortex stops routing to it — an unservable replica is not a routing
target.

## Prompt caps

`NEURON_MAX_PROMPT_TOKENS` sets a hard ceiling; larger prompts are
rejected before any GPU work, so a runaway client costs nothing.

## Checking on it

```sh
curl http://localhost:13131/health    # VRAM, utilisation, temps, per-model load
curl http://localhost:13131/models    # what is loaded, and can it serve
curl http://localhost:13131/version   # exactly which build is running
journalctl -u neuron -f
```

When a model has gone wrong, `/models` is usually more informative than
the logs: it reports whether each model can currently be served and, if
not, why.
