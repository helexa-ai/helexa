# Benchmarks

Batch-1 numbers for the helexa fleet — what one operator at a keyboard
feels. Produced by [`script/bench.py`](../script/bench.py), which works
against any OpenAI-compatible `/v1` endpoint so the same table can be
extended with llama.cpp / Ollama / vLLM columns by pointing it at their
servers (issue #22 tracks adding those baselines).

## Method

- **Workload**: streamed `chat/completions`, one request at a time
  (helexa's regime is operators and their agents, not QPS — see
  README "What helexa is not").
- **TTFT**: request send → first SSE content chunk. For thinking
  models this includes any visible-token delay; the bench prompt
  appends Qwen's `/no_think` soft switch so the budget isn't burned
  invisibly.
- **decode tok/s**: visible completion tokens over the first→last
  chunk window. neuron emits exactly one SSE chunk per generated
  token, so the chunk count is engine-truth (streaming
  `stream_options.include_usage` is not implemented yet). Reported
  only when the window exceeds 200 ms — short coalesced replies don't
  produce an honest rate.
- **Prompts**: synthetic filler at ~128 and ~4096 tokens plus a
  ~300-word generation task (`--max-tokens 256`, temperature 0).
- **Runs**: median of 3 after 1 unmeasured warmup, per cell.
- Requests go through the cortex gateway (`hanzalova:31313`), so
  numbers include the proxy hop — the path real clients use. The
  gateway also exports the same quantities per-request as Prometheus
  histograms (`cortex_time_to_first_token_seconds`,
  `cortex_tokens_per_second`, see #21).

## Fleet

| host | GPU(s) | model under test | quant / placement |
|---|---|---|---|
| beast | 2× RTX 5090 (32 GB, cc 12.0) | Qwen/Qwen3.6-27B | Q6K, TP=2 |
| benjy | RTX 4090 (24 GB, cc 8.9) | Qwen/Qwen3-8B | BF16, single GPU |
| quadbrat | RTX 3060 (12 GB, cc 8.6) | Qwen/Qwen3-1.7B | BF16, single GPU |

Driver 580.159, CUDA 13.0, Fedora 43. Models as configured in each
host's `default_models`.

## Results — 2026-06-12 (`8f6f1d3`)

| engine | model | prompt tok | TTFT (s) | decode tok/s | total (s) |
|---|---|---:|---:|---:|---:|
| helexa | Qwen/Qwen3-1.7B | ~128 | 0.685 | 81.3 | 3.741 |
| helexa | Qwen/Qwen3-1.7B | ~4096 | 2.743 | 35.4 | 9.884 |
| helexa | Qwen/Qwen3-8B | ~128 | 0.884 | 62.4 | 4.938 |
| helexa | Qwen/Qwen3-8B | ~4096 | 1.818 | 46.5 | 7.27 |
| helexa | Qwen/Qwen3.6-27B | ~128 | 1.658 | 35.0 | 8.981 |
| helexa | Qwen/Qwen3.6-27B | ~4096 | 7.067 | 33.7 | 14.63 |

Reading the table:

- Long-context decode degradation (81→35 tok/s on the 1.7B) is the
  attention cost of a fuller KV cache — expected, and the kind of
  number the prefix-KV-cache work (#11) and chunked prefill (#23)
  exist to improve at the TTFT end.
- The 27B rows are the headline case: a near-frontier hybrid
  linear-attention model decoding at a steady ~35 tok/s on two
  consumer cards, with essentially no decode degradation from 128 to
  4k context (the Gated DeltaNet recurrent state is O(1) in sequence
  length — this is the architecture doing what it promises). The
  4k-prompt TTFT (7.1 s) is dominated by the recurrent, non-chunked
  delta-rule prefill — issue #23 tracks the fix, and this row is its
  before number.

## Reproducing

```sh
# the whole fleet (all loaded models), defaults shown
./script/bench.py --base-url http://hanzalova.internal:31313/v1 \
    --runs 3 --prompt-tokens 128,4096 --max-tokens 256 \
    --json bench-results.jsonl

# a competitor engine for comparison columns
./script/bench.py --base-url http://localhost:8080/v1 \
    --label llama.cpp --model <model-id>
```

Append-only JSON rows (`--json`) keep a longitudinal record across
commits; the `engine` label column makes cross-engine tables a
concatenation, not a merge.

## Known gaps

- **No competitor baselines yet** — requires llama.cpp / Ollama
  serving the same checkpoints on the same hosts; the harness is
  ready for them.
- **Cold-load time** is not yet measured here; it is visible per
  deploy in the `loaded default model … elapsed_ms=…` journal line
  and the deploy workflow's validation step, and is tracked as #1.
- **Streaming usage**: neuron does not emit a final usage frame on
  SSE streams yet, so token counts rely on the chunk-per-token
  invariant.
