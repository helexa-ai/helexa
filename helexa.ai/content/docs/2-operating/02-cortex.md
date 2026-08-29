---
title: Configuring cortex
sidebar_label: cortex
description: The gateway config, the model catalogue, and what cortex decides.
---

# Configuring cortex

cortex reads two files. `cortex.toml` describes the fleet — where the
neurons are. `models.toml` describes the models — what they need and
where they may run.

They are separate because they change for different reasons and at
different rates. Hardware facts are discovered from neurons rather than
configured; you do not list device types or VRAM anywhere.

## cortex.toml

```toml
# /etc/cortex/cortex.toml
[gateway]
listen = "0.0.0.0:31313"
metrics_listen = "0.0.0.0:31314"

[eviction]
strategy = "lru"          # lru | priority
defrag_after_cycles = 50

[[neurons]]
name = "beast"
endpoint = "http://beast.internal:13131"

[[neurons]]
name = "benjy"
endpoint = "http://benjy.internal:13131"
```

A neuron entry is a name and an address. Everything else about that host
— how many GPUs, how much VRAM, which CUDA version, what is loaded right
now — comes from polling it. Adding a GPU to a host requires no gateway
change.

`defrag_after_cycles` counts load/unload cycles on a host and warns when
VRAM fragmentation has had time to accumulate. It is a signal to restart
that neuron, not an automatic action.

## secrets.toml

Credentials do not belong in the file CI deploys. Put them in a
`secrets.toml` beside the main config and the service will merge it
last, so its values win:

```
/etc/cortex/cortex.toml     structure — deployed by CI, safe to diff
/etc/cortex/secrets.toml    credentials — yours, root-only, never in git
```

```toml
# /etc/cortex/secrets.toml   (root:cortex 0640)
[[entitlements.keys]]
key = "sk-…"
account_id = "acct-a"
key_id = "a"

[upstream]
bearer = "…"
```

`helexa-upstream` and `helexa-angels` read a `secrets.toml` beside their
own configs the same way — the DB URL, JWT secret and SMTP URL belong
there.

**An absent `secrets.toml` is not an error.** A single-file deployment
keeps working exactly as before, so this is opt-in per host and can be
adopted one service at a time.

### Why bother

Because a config CI cannot write is a config no check can defend, and
the cost of that is not hypothetical. `helexa-router.toml` was kept out
of git for sitting beside secret-bearing files. Nothing could diff it
against the catalogue, so its `helexa/balanced` alias went on pointing
at a model that had been retired — and since both models were
deliberately given equal residency priority so they could displace each
other, every authenticated chat turn evicted the resident flagship to
cold-load the retired one. Four of those overlapped and the node ended
up holding 32 GB of GPU memory with no model loaded and nothing
serving.

Splitting secrets from structure is what lets the structure be
version-controlled, reviewed and checked. The ordering matters too: the
operator's file merges last, so a deploy can never replace a live
credential with a placeholder.

### Moving existing secrets

One-time, per host, and reversible — the service reads both files, so
you can move a value and restart before moving the next:

1. Create `secrets.toml` beside the config, owned `root:<service>` mode
   `0640` (`0600` where the service runs as root).
2. Move the credential blocks across. Leave everything else in place.
3. Restart the service and confirm it still authenticates — for cortex,
   a request with a real key should be billed rather than treated as
   anonymous.
4. Delete the moved lines from the main file only once step 3 passes.

Until step 4 the value exists in both files, which is safe: the
operator's copy wins.

## models.toml

The catalogue says what each model needs and where it may run:

```toml
[[models]]
id = "Qwen/Qwen3-8B"
harness = "candle"
vram_mb = 18000
min_devices = 1
min_device_vram_mb = 16000
residency_priority = 200
cost.input = 0.10
cost.output = 0.40
```

cortex matches these constraints against what it discovered. A model
needing two devices of 24 GB each will only ever be offered to a host
that has them.

`residency_priority` decides what may displace what when a host runs out
of VRAM. It repays reading [placement &
displacement](/docs/operating/placement) before setting it — the common
mistake, giving two models that should alternate different numbers,
silently makes their swap one-directional.

### Tier aliases

Clients should not hardcode model ids. Aliases let them ask for a role:

```toml
[aliases]
"helexa/small" = "Qwen/Qwen3-1.7B"
"helexa/balanced" = "Qwen/Qwen3-8B"
"helexa/large" = "Qwen/Qwen3.8-27B"
"helexa/image" = "Tongyi-MAI/Z-Image-Turbo"
```

You can then re-point a tier at a better model without a single client
changing anything.

### Pricing

`cost.input` and `cost.output` are USD per million tokens, surfaced
verbatim on `GET /v1/models` so clients can show spend.

Omitting the block and setting it to zero mean different things: absent
is "not declared", `0.0` is "deliberately free". The advertised rate has
to match what metering bills, so treat a change here as a change to
billing.

## What cortex decides

**Routing.** For a model already loaded in more than one place, cortex
picks the least-busy replica using live in-flight and queue-depth
figures polled from each neuron. It is load-aware, not first-match.

**Placement.** For a model that is not loaded anywhere, cortex ranks
candidate hosts: one the model is pinned to, then one with free VRAM
already, then one that could fit it after evicting something it
outranks, then simply the most free. Free-fit beats evict-fit, so
nothing is displaced while a host with room exists.

**Eviction.** When a host is chosen that needs room, cortex unloads the
least-recently-used model that the incoming one is permitted to
displace.

**Translation.** Anthropic-shaped requests are converted to the internal
form and the response converted back. This is envelope transformation
only — no prompt content is added, removed or reordered at any point.

## Metrics

Prometheus metrics on `metrics_listen` (31314 by default): request
counts, latency histograms, error and cold-start counters, per-model
load and per-device health, all labelled by model and node.

Scrape the gateway rather than individual neurons — it already has the
fleet-wide view.
