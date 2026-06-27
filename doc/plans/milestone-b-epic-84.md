# Milestone B — Reasoning frontier (80B-A3B MoE) · epic #84

> **Session hand-off.** Written 2026-06-27 at the close of the session that
> built and shipped all of Milestone A (Performance observability, epic #83,
> now closed). This document is the brief for the *next* session, which opens
> Milestone B. Read it top-to-bottom before touching code.

---

## Start-here prompt (paste/adapt for the next session)

> We're starting Milestone B (Gitea epic helexa/helexa#84, "Reasoning
> frontier 80B-A3B MoE on beast"). Milestone A (the bench observability
> foundation, epic #83) is fully merged and partly deployed — the bench can
> now measure everything we need to data-gate this work. Begin with **F1
> (#92)**: wire the high-sparsity MoE FFN into neuron's `qwen3_5` hybrid
> attention layer stack, with TP expert sharding, variant-agnostic. This is
> the gating CUDA-heavy piece; scope it carefully first (read the qwen3_5 and
> qwen3_moe code, plan the HF-reference logits validation on beast), don't
> dive straight into edits. Follow the per-issue branch→local-CI→push→watch-
> CUDA-gate→merge rhythm from the last session (see "How to work" below). The
> CUDA type-check is the real gate for neuron changes — the local build is
> CPU-only and does **not** compile the `cfg(cuda)` paths.

---

## Why this milestone exists (the strategic finding)

beast (2× RTX 5090 = 64 GB) currently serves **`Qwen/Qwen3.6-27B`** — the
*dense* member of the `qwen3_5` family (Qwen3-Next: **Gated DeltaNet**
linear-attention + sparse **Gated** full-attention hybrid), Q6K/ISQ, TP-2,
vision-capable. It writes correct compiling code but is short on the
reasoning/implementation-planning capacity of larger models.

The capability ceiling for beast is the **MoE sibling of the same
architecture**: **Qwen3-Next-80B-A3B** / **Qwen3-Coder-Next** — 80B total /
**3B active**, 512 experts (10 active) + shared expert, *same* Gated DeltaNet
hybrid attention. Why it's the right target:

- **~3× total parameters** → the reasoning/planning lever the user feels missing.
- **3B active per token** → decode is bandwidth-bound on 3B, so it should be
  **faster** than today's dense 27B, not slower.
- **Fits 64 GB at Q4_K_M** (~45 GB weights → ~22.5 GB/GPU under TP-2).
- The hybrid has only ~12 full-attention layers with **2 KV heads**, so the
  KV cache is tiny (~3 GB @ 256k) — **weights, not KV, are the only real
  constraint**. The "~2/3 VRAM used" observation on the 27B is unspent
  capacity this model would use.

Full background: Gitea epic **#84**; the original planning doc at
`~/.claude/plans/enchanted-yawning-octopus.md`; auto-memory
`project_frontier_observability_plan.md`.

---

## Decisions already made (do NOT re-litigate)

1. **Vision stays via cold-swap.** A frontier text model (~45 GB) and the
   27B-VL (~22 GB) can't co-reside on 64 GB, so vision is preserved by
   evicting/cold-loading on demand — *not* co-tenancy. This makes model-swap
   cost a first-class metric (already built: O6 `swap-cost`). → F4e (#99).
2. **MoE support is built variant-agnostic.** Don't hard-wire to one
   checkpoint. The A/B bench (F3 #94) + the capability rubric (O7) pick the
   resident model from real numbers — Coder-Next vs Next-80B-A3B-Thinking vs
   keep-27B.
3. **Measure-first.** Every engine lever is gated on bench numbers. Milestone
   A exists precisely so F-work is data-driven. Use it.

---

## What's DONE (Milestone A — closed) and how to use it

Epic #83 closed; 7 PRs merged (#100–#106). The `helexa-bench` crate now
measures, per (target, model, build SHA, scenario):

| Capability | How to read it |
|---|---|
| Server-measured **prefill vs decode tok/s** (not client-inferred) | `usage.helexa_timing` on the wire; `helexa-bench report` columns |
| **p50/p95/p99** tail latency | `report` columns / `/api/summary` |
| **VRAM used + node total (headroom)**, GPU util/temp | `report` VRAM column / `/api/summary` |
| **Context-length scaling** + decode-flatness (the GDN O(1) check) | `helexa-bench report --scaling` / `/api/scaling` |
| **Throughput under concurrency** + queue-wait + admission rejects | `concurrency:<n>` scenarios (opt-in) / `/api/summary` |
| **Cold-load / model-swap cost** | `helexa-bench swap-cost` then `report --swap` / `/api/swap` |
| **Capability quality** (planning/reasoning) | `capability:<name>` probes (opt-in) → `score --id <n> --score <x>` → `report --capability` |

Deploy status: **O1–O3 + O5 are live on the fleet** (validated against agent
zero — a0 on the Responses path works; the additive `usage.helexa_timing`
didn't disturb existing clients). **O4/O6/O7 land on the next deploy.**

**Key implication for this milestone:** F3 (the A/B decision gate) is *already
tooled*. Once the 80B model loads, comparing it to the 27B is: run the bench
(incl. a `capability_probes` planning prompt and the `concurrency:<n>`
scenarios), `helexa-bench report --scaling/--swap/--capability`, and read the
table. You do **not** need to build new measurement for F3 — you need to wire
the model (F1/F2) and then drive the bench.

---

## The todo list (Milestone B / epic #84)

Dependency order matters. F1 gates everything; F3 needs F2 + the bench; the
F4 levers are bench-gated and mostly independent.

- [ ] **F1 #92 — neuron: MoE FFN into the `qwen3_5` hybrid layer stack + TP
  expert sharding (variant-agnostic).** *Gating impl — the hard CUDA piece.*
- [ ] **F2 #93 — catalogue + load Qwen3-Next-80B-A3B family @ Q4_K_M on
  beast** (Coder-Next + Thinking); context auto-derive (#67); expose
  `max_model_len` (relate #78). Depends on F1.
- [ ] **F3 #94 — A/B decision gate.** Run the bench: 80B-A3B-Q4 variants vs
  27B-Q6 across prefill/decode tok/s, VRAM headroom, p95-under-concurrency,
  swap cost, and the capability rubric. Pick the resident model. Depends on
  F2 + Milestone A (done).
- [ ] **F4a #95 — enable flash-attention on the full-attn layers** (cheap
  TTFT/prefill win; `flash-attn` feature exists, defaults off). Bench-gated.
- [ ] **F4b #96 — wire speculative decoding into generation** (core exists in
  `harness/speculative.rs`, not connected). **Relates / likely supersedes
  #25, #79.** Expectation: *low* value for A3B-active decode (already cheap),
  higher for the dense 27B path — let the bench decide ordering.
- [ ] **F4c #97 — KV-cache quantization** (`p3-later`). Measurement-gated;
  likely unnecessary for this hybrid (KV is already tiny).
- [ ] **F4d #98 — continuous batching for agentic fan-out.** Highest-value
  *throughput* lever for the a0/hermes/opencode workload; scoped by the **O5
  concurrency numbers** (#89). A3B-MoE batches especially well.
- [ ] **F4e #99 — vision cold-swap policy** (cortex): catalogue + eviction so
  27B-VL ⇄ frontier swaps cleanly; gated by the **O6 swap-cost numbers**
  (#90).

**Suggested first arc for the next session:** F1 only (or F1 scoping + the
first slice). It's multi-session on its own; don't try to chain F2/F3 in the
same session.

---

## F1 deep-dive (the gating item)

**The shape of the work.** beast already has BOTH halves, as *separate* code
paths:
- `qwen3_5` hybrid **attention** (Gated DeltaNet + Gated full-attention) —
  this is what the dense 27B uses today.
- `qwen3_moe` **MoE FFN** — used by other Qwen3 MoE variants.

Qwen3-Next-80B **fuses them**: the `qwen3_5` hybrid attention layer stack with
a **high-sparsity MoE FFN** (router + 10-of-512 experts + a shared expert) in
place of the dense MLP. F1 wires the MoE FFN into the `qwen3_5` layer stack,
**including TP expert sharding** alongside the existing attention sharding.

**Where to look (read before editing):**
- `crates/neuron/src/harness/arch/qwen3_5/` — the hybrid arch (attention,
  GatedDeltaNet, the layer struct that currently holds a dense FFN).
- `crates/neuron/src/harness/tp/tp_qwen3_5.rs` — the TP sharding of qwen3_5
  (column/row-parallel attention + AllReduce); expert sharding goes here.
- The existing `qwen3_moe` implementation (grep `qwen3_moe`) — reuse its
  router + expert FFN logic; don't reinvent.
- `crates/neuron/src/harness/candle.rs` — arch dispatch (where model type →
  arch is chosen) and `tp/isq.rs` (in-situ quantization).
- Canonical device-worker narrative:
  `crates/neuron/src/harness/device_worker/mod.rs` doc-comment (all leader
  CUDA ops — load/forward/drop — route through the per-device worker thread;
  tensors never escape it alive).

**Validation approach (decide this up front).** Correctness = logits match a
cached HF reference on beast, within tolerance. This is the project's
established pattern (cf. the vision spatial-position work — HF ref cached on
beast; see auto-memory `vision spatial position encoding` and
`doc/vision-qwen3_6-spec.md`). Plan: pick a small prompt, capture HF
`transformers` logits for the 80B-A3B checkpoint on beast, diff against
neuron's. Get this harness sketched before/while implementing — it's how you'll
know F1 actually works, since the CUDA path can't run locally.

**Watch out for:**
- TP expert sharding: experts must shard across ranks without breaking the
  router's top-k selection or the AllReduce pattern. Fused tensors in qwen3_5
  (`in_proj_qkv`, `conv1d`) already do per-region slicing per rank — the MoE
  expert weights need analogous care.
- Transient VRAM peaks during sharded load (one full-tensor allocation per
  layer during construction is the existing pattern).
- `ModelSpec` is `{ model_id, harness, quant?, tensor_parallel?, devices? }` —
  `quant` left `None` lets neuron resolve from catalogue.

---

## How to work (the rhythm that worked all of Milestone A)

This is the CLAUDE.md workflow, confirmed effective across 8 PRs this session:

1. **One issue = one branch = one PR.** `feat/<issue#>-<slug>` (e.g.
   `feat/92-qwen3_5-moe`). Branch off `main`.
2. **Run the CI triad locally before pushing** (CPU-only):
   `cargo fmt --check --all`, `cargo clippy --workspace -- -D warnings`,
   `cargo test --workspace`. For neuron changes this is necessary-but-not-
   sufficient.
3. **The CUDA type-check is the real gate for neuron.** The local build does
   NOT compile `#[cfg(feature="cuda")]` paths — only the branch CI's "CUDA
   type-check" job validates them. This session that job caught a real
   TP-scope bug (variables declared inside a `'work` block but read after it
   exited) that was invisible to a green local build. **Treat the push as the
   validation, not a rubber stamp.**
4. **Push on local-green** (don't ask first), then background-watch the branch
   CI via the `gitea-mcp` `actions_run_read` tools (`list_runs` →
   `list_run_jobs` → on failure `get_job_log_preview`).
5. **Merge to `main` when the FOUR validation jobs are green** — Format,
   Clippy, Test, CUDA type-check. The SRPM/COPR/version-bump jobs are the
   deploy pipeline (run on `main`); don't wait on them. Merging triggers
   auto-deploy.
6. **Commit/PR hygiene:** every commit subject ends with `(#NN)`; PR body
   `Closes #NN` (auto-closes the issue on merge); after merge, sync `main`,
   delete the branch, tick the epic checklist.
7. **Git remote quirk:** this repo pins the working SSH key —
   `git config core.sshCommand "ssh -i ~/.ssh/id_grenade -o IdentitiesOnly=yes"`
   is already set; pushes/pulls just work.

CI timing observed this session: Format ~40s, Clippy ~2.5min, CUDA type-check
~1.5–2min (fast-fails in ~90s on a compile error), **Test ~6min** (the long
pole). A full green run is ~6–7 min wall-clock.

---

## Open / incomplete items carried over (none block F1)

- **Baseline-gate capture (was local task #4):** the enriched 27B baseline
  auto-populates as the continuous bench sweeps the new neuron+bench SHA. It's
  not a code task — just confirm it's populated (run `helexa-bench report` /
  `report --scaling` on the metrics host, or hit `/api/summary`) before F3
  consumes it. If you want the *capability* and *concurrency* dimensions of
  the baseline, those scenarios are opt-in — enable `capability_probes` and
  `concurrency_levels` in the bench config (and run `swap-cost` once) so the
  27B has comparison data ready for F3.
- **External bench UI (bench.internal):** a *separate frontend* (not in this
  repo) consuming the JSON API. All new metrics are in `/api/{summary,scaling,
  swap,capability}` but the UI renders a fixed column set, so it currently
  looks unchanged. Updating it is a frontend task in that other repo — file/
  do whenever; not blocking.
- **O7 LLM-judge auto-scoring:** the schema (`quality_score`, `scorer`) and
  the manual `score` CLI are done; wiring an LLM-judge to auto-score
  capability artifacts is deferred (the user chose "manual now, judge later").
  Pick this up if/when F3's capability comparison needs scale.

Working tree is clean, `main` is synced, all session branches deleted.
Nothing is in flight.

---

## Quick reference

- **Epic:** helexa/helexa#84 (milestone "Reasoning frontier (80B-A3B)", id 11).
  Children #92–#99. Observability epic #83 is **closed** (#85–#91 merged).
- **Related issues:** #25/#79 (spec decoding), #78 (`max_model_len`), #67
  (context auto-derive), #11 (prefix KV), #23 (chunked prefill), #53
  (admission), #26 (release writeup), #22 (bench).
- **Bench crate:** `crates/helexa-bench/` — `scenario.rs` (the `Scenario`
  trait + families), `store.rs` (SQLite + migrations), `report.rs`,
  `sweep.rs`, `api.rs`, `client.rs`, `config.rs`. Example config:
  `helexa-bench.example.toml`.
- **neuron arch:** `crates/neuron/src/harness/{arch/qwen3_5, tp, device_worker,
  candle.rs}`.
- **Project memory (auto-loaded):** `project_frontier_observability_plan.md`,
  `tensor_parallelism`, `vision_spatial_position_encoding`,
  `fleet_validation_workflow`, `phase_branch_ci_workflow`.
- **Note on this file's location:** the repo convention is `doc/plan/`
  (singular); this was written to `doc/plans/` per explicit request. Consider
  `git mv doc/plans/milestone-b-epic-84.md doc/plan/` for consistency.
