#!/usr/bin/env python3
"""Reproducible batch-1 benchmark harness for helexa (#22).

Measures what one operator at a keyboard feels, per model:

  - TTFT      time from request send to the first SSE content chunk
  - decode    completion tokens per second over the first->last chunk
              window (token count from the final `usage` object when
              the server sends one, else the content-chunk count)
  - total     wall-clock for the whole request

Works against ANY OpenAI-compatible /v1 endpoint (helexa's cortex,
llama.cpp's llama-server, Ollama's compat endpoint, vLLM, ...), so the
same invocation produces comparable columns across engines:

  ./script/bench.py --base-url http://hanzalova.internal:31313/v1
  ./script/bench.py --base-url http://localhost:8080/v1 --model qwen3:8b

stdlib-only on purpose: no venv, no pip, runs from any Fedora host.
Results print as a markdown table; --json appends machine-readable
rows for longitudinal tracking (doc/benchmarks.md records the method).
"""

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request

# A paragraph of filler re-used to synthesise prompts of a target
# approximate token count (~4 chars/token heuristic — close enough for
# bucketing; real token counts are read back from the usage object).
FILLER = (
    "The quick brown fox jumps over the lazy dog while the band plays "
    "a slow waltz in the background and somebody counts the beats. "
)

# /no_think: Qwen3-family soft switch rendered by the chat template;
# keeps thinking models from burning the token budget invisibly
# (reasoning deltas are not on the wire by default). Harmless for
# non-thinking models.
QUESTION = (
    "\n\nRetell the scene above as a vivid story of about 300 words. /no_think"
)


def build_prompt(approx_tokens: int) -> str:
    target_chars = max(approx_tokens, 16) * 4
    body = (FILLER * (target_chars // len(FILLER) + 1))[:target_chars]
    return body + QUESTION


def one_run(base_url: str, model: str, prompt: str, max_tokens: int, timeout: float):
    """Single streamed request. Returns dict with ttft, decode_tps,
    total_s, completion_tokens, prompt_tokens (None where unknown)."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
    )
    start = time.monotonic()
    first = last = None
    chunk_count = 0
    prompt_tokens = completion_tokens = None
    tail = ""
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        buf = b""
        while True:
            block = resp.read(8192)
            if not block:
                break
            now = time.monotonic()
            buf += block
            while b"\n\n" in buf:
                event, buf = buf.split(b"\n\n", 1)
                line = event.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    continue
                tail = data  # last data frame wins (usage rides there)
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                delta = (choices[0].get("delta") or {}) if choices else {}
                if delta.get("content"):
                    if first is None:
                        first = now
                    last = now
                    chunk_count += 1
                usage = obj.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens")
                    completion_tokens = usage.get("completion_tokens")
    end = time.monotonic()

    if first is None:
        raise RuntimeError(f"no content chunks received (last frame: {tail[:200]})")
    # neuron emits exactly one SSE chunk per generated visible token,
    # so chunk count is an engine-truth count when no usage frame is
    # sent (streaming include_usage is not implemented yet).
    tokens = completion_tokens if completion_tokens else chunk_count
    # decode rate is only meaningful over a real inter-chunk window;
    # short replies can arrive coalesced into one TCP read (window=0).
    window = (last - first) if (last and last > first) else 0.0
    return {
        "ttft_s": first - start,
        "decode_tps": tokens / window if window > 0.2 else None,
        "total_s": end - start,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": tokens,
    }


def discover_models(base_url: str, timeout: float) -> list[str]:
    with urllib.request.urlopen(f"{base_url}/models", timeout=timeout) as resp:
        data = json.load(resp).get("data", [])
    # helexa extension: prefer loaded models; plain OpenAI lists lack
    # the field, in which case take everything.
    loaded = [m["id"] for m in data if m.get("loaded")]
    return loaded or [m["id"] for m in data]


def median(values):
    vals = [v for v in values if v is not None]
    return statistics.median(vals) if vals else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://hanzalova.internal:31313/v1")
    ap.add_argument("--model", action="append", help="repeatable; default: all loaded models")
    ap.add_argument("--runs", type=int, default=3, help="measured runs per cell (after 1 warmup)")
    ap.add_argument(
        "--prompt-tokens",
        default="128,4096",
        help="comma-separated approximate prompt sizes",
    )
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--json", help="append JSON rows to this file")
    ap.add_argument("--label", default="helexa", help="engine label for the output rows")
    args = ap.parse_args()

    models = args.model or discover_models(args.base_url, args.timeout)
    sizes = [int(s) for s in args.prompt_tokens.split(",")]
    rows = []

    for model in models:
        for size in sizes:
            prompt = build_prompt(size)
            try:
                one_run(args.base_url, model, prompt, args.max_tokens, args.timeout)  # warmup
                runs = [
                    one_run(args.base_url, model, prompt, args.max_tokens, args.timeout)
                    for _ in range(args.runs)
                ]
            except (RuntimeError, urllib.error.URLError, TimeoutError) as e:
                print(f"!! {model} @~{size} tok: {e}", file=sys.stderr)
                continue
            row = {
                "engine": args.label,
                "model": model,
                "approx_prompt_tokens": size,
                "actual_prompt_tokens": runs[0]["prompt_tokens"],
                "runs": args.runs,
                "ttft_s_median": round(median(r["ttft_s"] for r in runs), 3),
                "decode_tps_median": round(median(r["decode_tps"] for r in runs), 1),
                "total_s_median": round(median(r["total_s"] for r in runs), 3),
                "completion_tokens": runs[0]["completion_tokens"],
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            rows.append(row)
            print(f".. {model} @~{size} tok done", file=sys.stderr)

    print(f"\n| engine | model | prompt tok | TTFT (s) | decode tok/s | total (s) |")
    print("|---|---|---:|---:|---:|---:|")
    for r in rows:
        ptok = r["actual_prompt_tokens"] or f"~{r['approx_prompt_tokens']}"
        print(
            f"| {r['engine']} | {r['model']} | {ptok} | {r['ttft_s_median']} "
            f"| {r['decode_tps_median']} | {r['total_s_median']} |"
        )

    if args.json:
        with open(args.json, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
