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

### Where weights come from

By default a bare `org/name` id is fetched from HuggingFace. Operators
running a mirror, or serving models from more than one place, can
declare sources explicitly:

```toml
[harness.candle]
default_source = "huggingface"

[harness.candle.sources.huggingface]
endpoint = "https://huggingface.co"
auth_env = "HF_TOKEN"          # env var name, never the token itself
cache_dir = "/archive/llm"     # optional; per-source cache
```

`auth_env` names an environment variable rather than holding a token, so
credentials stay out of the config file — worth keeping to even when the
file is not obviously shared, because config is exactly what gets copied
between hosts.

Give a source its own `cache_dir` when two sources can serve the same
`org/name`. The cache tree is keyed on the model id alone, so sharing a
directory between sources that disagree about what `org/name` means will
collide on disk.

## Default models

Models listed as defaults load when the daemon activates, so the host
comes back serving rather than waiting for the first request to pay the
load cost:

```toml
[[default_models]]
model_id = "Qwen/Qwen3.8-27B"
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
max_wait_secs = 30       # how long a queued request waits for a slot
kv_max_wait_secs = 300   # how long it waits for enough KV budget
max_per_principal = 2    # fair share, so one caller cannot monopolise
```

`kv_max_wait_secs` is deliberately much longer than `max_wait_secs`.
The first waits for a slot to turn over; the second waits for a whole
long-context sequence to finish, which on a 27B can be minutes. A
request that needs KV budget nobody has yet is not a request to reject
in thirty seconds.

Over-capacity requests are rejected promptly with `429`/`503` and a
`Retry-After` rather than being allowed to queue invisibly. A fast
rejection a client can retry is far better than a request that hangs
until something times out.

`max_in_flight` above 1 enables batched decode on architectures that
support it, so concurrent requests share a decode step instead of
serialising.

**`max_in_flight` does not partition the KV budget.** KV is a shared
pool, and each request reserves what its own prompt actually needs when
it is admitted — which is why `kv_max_wait_secs` above exists at all. A
request larger than what is currently free waits for bytes to come back
rather than being handed a fixed 1/N slice, and a single long session
can legitimately hold most of the pool.

So the slot count and the context window are separate questions. What
bounds a busy moment is the pool: divide `kv_budget_mb` by the model's
KV cost per token (`GET /health` reports both) to see how many tokens
are available to share. Many ordinary chat turns fit at once; a few very
long agentic sessions will queue on bytes.

Size `max_in_flight` against the concurrency you actually want to serve
concurrently, not against context. Anything above it queues on admission
with a `Retry-After` rather than being refused, so setting it too low
converts capacity you have into latency your callers feel. Setting it
very high mainly costs batch-step efficiency, since the engine pads to
the widest active sequence.

Measure before assuming you need a high number — but measure the traffic
you will have, not the traffic you had. On one production host, seven
days of spans showed 89.6% of busy time at a single request in flight;
that host's chat tier was at the time being routed to a *different*
model, and the figure stopped describing it the moment that was
corrected.

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
  (`NEURON_MIN_FREE_VRAM_MB`, or `min_free_floor_mb` under
  `[harness.candle.context_limit]`; default 1500)
- a **length-aware check** that estimates the KV cache the request will
  actually need and refuses if it will not fit

The second is what protects against long prompts specifically: a request
can pass the static floor and still be impossible, because KV grows with
prompt *and* generation length. See [context
limits](/docs/operating/context-limits) for how the numbers relate.

`activation_headroom_mb` reserves VRAM per card for prefill activations
on top of the weights and the KV cache, before the context ceiling is
computed. Raise it if prefill OOMs on a host whose steady-state
occupancy looks fine — activations are transient and do not show up in
the resident figure.

A host that cannot currently serve a model reports that on `/health`,
and cortex stops routing to it — an unservable replica is not a routing
target.

## Image models

```toml
[harness.candle.image]
te_device = "cpu"     # where the text encoder runs
te_resident = true    # keep it loaded between generations
max_dim = 2048        # resolution ceiling per side
```

The text encoder is the placement decision. On CPU the ~8 GB encoder
never touches the GPU — prompt encoding is one short forward and the
features are tiny — which is what lets a 24 GB card serve a BF16 DiT at
all. Move it to `cuda` only on a card with room to spare.

`te_resident` trades memory for latency: resident costs ~16 GB of system
RAM on CPU (or ~8 GB of VRAM on CUDA) and saves a few seconds per
request. CPU defaults to resident, CUDA to rebuilding per request, which
are the right defaults for where each is scarce.

`max_dim` rejects oversized requests **before admission**, so an
impossible resolution costs nothing. That matters more than it sounds:
an OOM on the device poisons the worker context, so refusing early is
the difference between one rejected request and a model that needs
restarting.

## Prompt caps

`NEURON_MAX_PROMPT_TOKENS` sets a hard ceiling; larger prompts are
rejected before any GPU work, so a runaway client costs nothing.

## Sampling

Every model publishes its own sampling defaults in
`generation_config.json`, and a caller can override them per request.
Between those two sits an operator override, for the cases where the
model's published defaults are wrong for how you serve it:

```toml
[[default_models]]
model_id = "Qwen/Qwen3.8-27B"

[default_models.sampling]
temperature = 0.6
presence_penalty = 1.0
```

Precedence is **request > operator > model > built-in**. Anything you
leave unset changes nothing, so an empty table behaves exactly as no
table at all.

Set the same values in the gateway's `models.toml` under
`[models.sampling]`. A model that is cold-loaded on demand gets its
sampling from the catalogue, not from this file, so the two must agree
or the same model samples differently depending on how it arrived —
`script/check-config-consistency.py` fails the deploy if they diverge.

### Which knob for which symptom

| symptom | knob |
|---|---|
| Structurally malformed output — truncated JSON, unbalanced braces | `temperature` |
| The think block restates the same passage over and over | `presence_penalty` |
| Common tokens over-represented across a long answer | `frequency_penalty` |
| Short immediate loops — "no, no, no…" | `repeat_penalty` |
| You need two runs to be comparable | `seed` |

**`temperature`** is the blunt one. Qwen3.8-27B publishes `1.0`, which
measured a 20% structural-defect rate on long agentic turns against 0/60
at `0.6`; the fleet therefore overrides it. Note that `0.6` is also what
Qwen recommend for their thinking variants, so this is not fighting the
model.

**`presence_penalty`** and **`frequency_penalty`** are scored over the
*whole* generated sequence. They are the ones that matter for reasoning
models, and Qwen name `presence_penalty` (0–2, 1.5 when severe) as the
remedy for the endless repetition their models fall into during long
thinking. Both default to `0.0`, which is off.

**`repeat_penalty`** (default `1.1`) and **`repeat_last_n`** (default
`64`) are the older, *windowed* form: the penalty only sees the last
`repeat_last_n` tokens. That is fine for chat and close to useless for a
think block — a 20,000-token deliberation restating a passage 5,000
tokens later is entirely outside a 64-token window. Measured across
recorded sessions, **86% of think blocks over 60k characters contained
verbatim repeated sentences**, one of them repeating 76% of its own
content.

Widening the window is the obvious move and the wrong one: the penalty
is token-level, and source code legitimately repeats tokens constantly
(`const`, `function`, brace runs), so a wide window at a meaningful
penalty degrades exactly the output you care about. Reach for
`presence_penalty` first — what recurs is passages, not tokens.

**`seed`** pins the sampler's RNG for every request to this model. It
exists because parameter search needs a control and the clients that
would otherwise pin it often cannot. Leaving it set is a product
decision, not just a testing one: every identical prompt then returns an
identical answer to every caller, which is predictable and cacheable but
removes the variation a user may expect from retrying.

## Reasoning

```toml
[default_models.sampling]
# nothing here — reasoning settings live in their own places, below
```

**`preserve_thinking`** controls whether prior turns keep their think
blocks when the conversation is re-rendered. It is the model's own chat
template control, not ours:

```toml
[[default_models]]
preserve_thinking = false   # keep reasoning only for the turn in progress
```

Unset is the default and means the template decides. On Qwen3.8-27B that
is `true` — every prior turn's reasoning is replayed. Full replay is
what makes prompt growth track *total* completion rather than visible
output: one measured session produced 28,672 reasoning tokens on turn 1
and carried all of them into turn 2's 31,269-token prompt, at ~22 s of
prefill on every turn thereafter.

Setting it to `false` is a legitimate experiment, not a recommendation.
Whether narrowing replay *helps* the model is unmeasured. Set it in both
`models.toml` and this file, like sampling, and the value is reported on
`/v1/models` and logged at load so a session can be attributed to the
arm it ran under.

**`reasoning_budget.answer_reserve_tokens`** (default `4096`) holds back
part of the caller's output budget for the answer:

```toml
[harness.candle.reasoning_budget]
answer_reserve_tokens = 4096
```

A reasoning model with no stopping criterion fills whatever budget it is
given. Measured on Qwen3.8-27B: given 16,384 tokens it used 16,384;
given 32,768 it used 32,768 — and both times returned no answer at all.
The reserve bounds the think block at `max_tokens - answer_reserve` so
the rest survives for the reply.

Deliberately *not* derived from reasoning effort. Effort is a prompt
instruction the model is trained to follow; making it also set a
guillotine is how a model came to be told to reason at `xhigh` and then
cut off at `medium`'s budget. Set `0` to disable, restoring unbounded
reasoning.

The trade to know: when the reserve fires, the model is interrupted
mid-deliberation and its answer phase tends to state an *intention*
rather than a conclusion. That is better than returning nothing, but it
is not free.

## The prefix cache

Snapshots of previously-served prefixes, so a follow-up turn re-uses the
KV cache instead of re-prefilling the whole conversation:

```toml
[harness.candle.prefix_cache]
enabled = true
budget_mb = 2048    # snapshot bytes per loaded model
max_entries = 8     # live snapshots per model, regardless of budget
```

**Reach for `budget_mb` when `cached_tokens` stops growing mid-session.**
That is the symptom: the prompt keeps growing, the cached prefix does
not, and every turn re-prefills the difference. Observed on a 1 GiB
default — `cached_tokens` pinned at 2,016 for ten consecutive turns
while the prompt reached 52,869, so ~50k tokens were re-prefilled every turn
at roughly 37 s each, against ~20 s of actual decode.

The number is not free, and the direction of the trade is the opposite
of what it looks like. **The snapshot budget is reserved before the KV
budget, not from what is left over.** On a 2×32 GB host serving a 27B,
measured 1:1 — the token column is the pool all concurrent requests
share, not a per-slot allowance:

| `budget_mb` | KV budget | tokens in the pool | holds a 53k snapshot |
|---|---|---|---|
| 1024 | 4713 MB | 150,800 | no |
| 2048 | 3689 MB | 118,000 | yes |
| 3072 | 2665 MB | 85,300 | yes, but a session cannot reach 53k |

So raising it too far starves the KV cache and shortens how long a
single session can run — the model can no longer reach the context whose
snapshot the larger cache was meant to hold. Size it against the deepest
session you actually serve, and check `kv_budget_mb` in the journal
after a reload.

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
