# Context-window & token-limit settings

How the numeric knobs that govern usable context fit together, what the
valid ranges are, and where they live. Getting these out of sync is the
difference between "the agent has room to think" and "it compacts every
few turns and reasons from a corrupted summary."

The tables below document the **manual** reasoning that
[#62](https://git.lair.cafe/helexa/helexa/issues/62) and then
[#67](https://git.lair.cafe/helexa/helexa/issues/67) automate. As of #67
the neuron **computes** this limit itself from model architecture + live
VRAM + a self-measured throughput ceiling and advertises it on
`GET /models`; operators no longer hand-derive it. Read the rules below
as the *why* behind the derivation — see
[After #67](#after-67-the-neuron-computes-its-own-limit) for what the
daemon now does automatically.

## The knobs

| Knob | Where | What it bounds |
|---|---|---|
| `max_position_embeddings` | model `config.json` (fixed per model) | the model's native context ceiling — quality wall |
| `NEURON_MAX_PROMPT_TOKENS` | neuron systemd drop-in (env) | hard **prompt** cap; neuron rejects larger prompts with `400 context_length_exceeded` before any device work |
| `NEURON_MIN_FREE_VRAM_MB` | neuron systemd drop-in (env, default 1500) | static free-VRAM floor below which prefill is refused (`503 service_unavailable` / `InsufficientVram`) |
| request `max_tokens` | per request; neuron default 8192 | generation length; KV grows by prompt **+** generation |
| `limit.context` | `opencode.json` `provider.models.<id>.limit` | the wall opencode tracks for compaction % |
| `limit.input` | same | compaction trigger — opencode compacts to keep the prompt at/under this |
| `limit.output` | same | generation reserve opencode leaves below the wall |

## How they must relate

For a single model on a single neuron, all of these must hold:

```
1.  limit.input + limit.output  ≤  limit.context          (opencode internal; convention: input = context − output)
2.  limit.context               ≤  max_position_embeddings (model quality wall)
3.  limit.input                 ≤  NEURON_MAX_PROMPT_TOKENS (else neuron 400s a prompt opencode thought was fine)
4.  NEURON_MAX_PROMPT_TOKENS + max_tokens  ≤  max_position_embeddings
5.  KV(limit.context)/card + activation + NEURON_MIN_FREE_VRAM_MB  ≤  free VRAM on the tightest card
```

Notes:
- **Keep a margin on rule 3.** Set `NEURON_MAX_PROMPT_TOKENS` a bit above
  `limit.input` (e.g. one `output`-worth) so opencode↔neuron tokenizer
  counting differences don't trip a spurious 400 mid-session.
- **Convention:** mirror `limit.context` to `NEURON_MAX_PROMPT_TOKENS`
  and set `limit.input = context − output`. opencode then compacts to
  keep the prompt one `output` below the neuron wall — there is always
  generation headroom under the cap.
- **Rule 5 is the one with teeth at scale.** Today only the *static*
  floor (`NEURON_MIN_FREE_VRAM_MB`) guards the text path; it does **not**
  scale with prompt length. A long-but-under-cap prompt can clear the
  floor and then OOM mid-prefill (poisoning the device context). Tracked
  in [#65](https://git.lair.cafe/helexa/helexa/issues/65) — until that
  lands, treat the VRAM-safe ceiling in rule 5 as a hard limit you set
  `NEURON_MAX_PROMPT_TOKENS` below, not something the daemon enforces.

## VRAM cost of context (Qwen3.6-27B on beast)

Qwen3.6-27B is a hybrid linear-attention model: of its 64 layers only
every 4th is full-attention (`full_attention_interval = 4` → **16**
full-attn layers); the rest are `linear_attention` with constant-size
recurrent state. KV cache grows **only** on the 16 full-attn layers
(GQA, `num_key_value_heads = 4`, `head_dim = 256`, F16):

```
kv_per_token (total) = 2 (K+V) × 16 layers × 4 kv_heads × 256 head_dim × 2 B = 65536 B = 64 KiB/token
kv_per_token (per card, TP=2)                                                            = 32 KiB/token
```

beast = 2× RTX 5090 (32607 MiB each). KV per card and headroom against
the **measured idle free of the tighter card (GPU 1: 9254 MiB)**:

| `limit.context` | KV / card | Free left on GPU 1 (after KV) | Verdict |
|---|---|---|---|
| 49152 (≈49k, prior default) | ~1.5 GiB | ~7.7 GiB | very safe |
| **131072 (128k, recommended)** | **~4.0 GiB** | **~5.2 GiB** | **safe** |
| 196608 (192k, stretch) | ~6.0 GiB | ~3.1 GiB | plausible; wants #65 guard |
| 262144 (256k, model max) | ~8.0 GiB | ~1.1 GiB | unsafe at current free (under the 1500 MiB floor) |

`ConcatKvCache` is lazy — it allocates nothing at idle and resets between
requests — so raising the cap costs zero until a session actually uses
the longer window. The numbers above are upper bounds at *measured idle
free*; real usable headroom is lower under fragmentation and whatever
else is resident. Leave margin.

Reaching 256k (or running concurrent long sessions) needs more free VRAM
than this load leaves — KV quantization or a fixed/paged KV allocator —
none of which is required for 128k.

## Recommended profile: 128k

**neuron** — `NEURON_MAX_PROMPT_TOKENS` is **deploy-managed**, not
hand-edited. It lives in the `deploy-neurons` matrix in
`.gitea/workflows/deploy.yml` (`max_prompt_tokens` per host) and is
written to `/etc/systemd/system/neuron.service.d/model.conf` on each run.
A change to that value restarts the neuron **even when no new RPM ships**
(the deploy gates on package version *or* drop-in change), so the cap
rolls out alongside the rest of the service config. To change it, edit
the matrix value and let the deploy apply it:

```yaml
# .gitea/workflows/deploy.yml → jobs.deploy-neurons.strategy.matrix.include
- host: beast.hanzalova.internal
  flavour: blackwell
  load_timeout: 900
  max_prompt_tokens: 131072
```

The drop-in it writes:

```ini
# /etc/systemd/system/neuron.service.d/model.conf  (managed by deploy.yml)
[Service]
Environment=NEURON_MAX_PROMPT_TOKENS=131072
```

Verify after a deploy:

```sh
curl -s http://beast:13131/discovery | jq .max_prompt_tokens   # expect 131072
```

`model.conf` sorts after any manual `local.conf`, so the deploy-managed
value wins over a hand override of the same variable. Use `local.conf`
only for genuinely host-local, transient experiments — and remember a
later deploy will re-assert `model.conf`.

**opencode** — `opencode.json`, `provider.models."Qwen/Qwen3.6-27B".limit`:

```json
{ "context": 131072, "input": 122880, "output": 8192 }
```

(`input = context − output = 131072 − 8192`; `NEURON_MAX_PROMPT_TOKENS`
131072 sits one `output` above `input`, the tokenizer-drift margin.)

## After #62: single source of truth (superseded by #67)

[#62](https://git.lair.cafe/helexa/helexa/issues/62) moved `limit
{ context, input, output }` (and `cost`) onto `GET /models`, sourced from
the operator-declared catalogue (`models.toml`). That was the right
plumbing but the wrong *source*: a per-model catalogue limit goes stale
the moment cortex hot-swaps a neuron's resident model, and forces the
hand-tuning fight (the tables above) to be re-run on every change.

## After #67: the neuron computes its own limit

[#67](https://git.lair.cafe/helexa/helexa/issues/67) makes the limit a
**computed function of live state**, not an operator-declared fact. Per
loaded model, the neuron derives:

```
output  = output_reserve_tokens                       (config; default 8192)
kv/token/card = 2(K+V) · n_full_attn_layers · (n_kv_heads / tp) · head_dim · dtype_bytes
vram_ceiling       = (free_tightest − activation_headroom − min_free_floor) / kv_per_token_per_card
throughput_ceiling = target_prefill_latency_secs · measured_prefill_tok_per_sec
context = min(max_position_embeddings, vram_ceiling, throughput_ceiling)
          clamped by NEURON_MAX_PROMPT_TOKENS only if explicitly set (backstop)
input   = context − output
```

- `free_tightest` is the **minimum free VRAM across the model's
  devices** — the tightest card, often a non-leader TP rank.
- `measured_prefill_tok_per_sec` is **self-measured** (an EMA over real
  requests; a configured bootstrap until the first sample). Because it
  reads live state, the advertised `limit` **rises automatically** as
  prefix caching (#11) or other efficiency work frees VRAM / speeds
  prefill — no operator action.
- Knobs live in `[harness.candle.context_limit]` (see
  `neuron.example.toml`). The catalogue `limit` is **no longer
  consulted** (the field is inert/deprecated); `cost` stays
  operator-set in the catalogue.
- **opencode**: remove any hand-entered `limit` block from
  `opencode.json` — discovery is authoritative.

`NEURON_MAX_PROMPT_TOKENS` is demoted from authority to an **optional
clamp-only backstop** (applied only when explicitly set). The
deploy-managed drop-in still pins a per-host ceiling, but the derivation
binds below it in practice. (Enforcement of the *derived* prompt cap —
rejecting prompts above the computed `input` rather than the static
`NEURON_MAX_PROMPT_TOKENS` — is the remaining hardening; until it lands,
the static cap remains the enforced backstop.)

## Operational note

`GET /discovery` reports the live `max_prompt_tokens` the running neuron
process actually uses — check it rather than assuming the drop-in took
effect. A drop-in change only applies after `daemon-reload` + a neuron
restart, which the deploy performs; if `/discovery` doesn't match the
`max_prompt_tokens` in the deploy matrix, the host hasn't been
re-deployed since the value changed (or a higher-sorting drop-in is
overriding it). Re-run the `deploy` workflow to reconcile.
