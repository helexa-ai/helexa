#!/usr/bin/env python3
"""Decide what a newly-released model costs us to serve, before downloading it.

When weights drop, the only question that matters is whether the
architecture is one the harness already builds. That answer is entirely
contained in a handful of small files -- `config.json` and the
safetensors index are a few hundred kilobytes between them, against tens
of gigabytes of tensors -- so it can be had in seconds rather than after
a long download.

The report covers the three things that decide the work:

  supported   whether `model_type` is already in the harness's dense and
              tensor-parallel allow-lists. Those lists are read out of
              candle.rs rather than copied here, so this cannot claim
              support the code does not have.
  shape       a field-level diff against a reference model already
              served. A short diff means an alias; a long one means a
              port.
  fit         parameter count summed from the safetensors index, VRAM
              at each precision, and the per-token KV cost used to
              derive context limits.

Usage:
  script/arch-triage.py <repo-id> [--reference <repo-id>] [--json]
  script/arch-triage.py --local <path-to-snapshot>

Requires network access for repo ids; `--local` works offline against an
already-downloaded snapshot.
"""

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CANDLE_RS = REPO_ROOT / "crates" / "neuron" / "src" / "harness" / "candle.rs"

# Files worth pulling. Small enough that fetching all of them costs less
# than a second; between them they determine architecture, tokenisation,
# prompt formatting and total size.
SMALL_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "model.safetensors.index.json",
]

BYTES_PER_PARAM = {"bf16": 2.0, "q8_0": 1.0625, "q4_k_m": 0.5625}


def supported_model_types():
    """Read the harness allow-lists from source.

    Duplicating these constants is how a triage tool starts lying: the
    code grows support, the script does not, and the report says
    'port required' for something that already loads. Parsing the real
    declaration keeps the two from drifting.
    """
    out = {}
    try:
        src = CANDLE_RS.read_text()
    except OSError:
        return out
    for const in ("DENSE_SUPPORTED_MODEL_TYPES", "TP_SUPPORTED_MODEL_TYPES"):
        m = re.search(rf"{const}[^=]*=\s*&\[(.*?)\]", src, re.S)
        if m:
            out[const] = re.findall(r'"([^"]+)"', m.group(1))
    return out


def fetch(repo, filename, token=None):
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read().decode("utf-8")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return None


def load_files(repo=None, local=None, token=None):
    files = {}
    for name in SMALL_FILES:
        if local:
            p = Path(local) / name
            files[name] = p.read_text() if p.is_file() else None
        else:
            files[name] = fetch(repo, name, token)
    return files


def text_config(cfg):
    """Hyperparameters live under `text_config` in some layouts and at the
    top level in others. Both appear within the same model family, so
    neither can be assumed."""
    tc = cfg.get("text_config")
    return tc if isinstance(tc, dict) else cfg


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, f"{key}."))
        elif isinstance(v, list):
            # Long uniform lists (layer_types, rope scaling tables) are
            # summarised: the interesting fact is the composition, not
            # the hundred entries.
            if len(v) > 8 and all(isinstance(x, str) for x in v):
                counts = {x: v.count(x) for x in sorted(set(v))}
                out[key] = f"<{len(v)} entries: {counts}>"
            else:
                out[key] = json.dumps(v)
        else:
            out[key] = v
    return out


def diff_configs(new, ref):
    a, b = flatten(new), flatten(ref)
    added = {k: a[k] for k in a.keys() - b.keys()}
    removed = {k: b[k] for k in b.keys() - a.keys()}
    changed = {k: (b[k], a[k]) for k in a.keys() & b.keys() if a[k] != b[k]}
    return added, removed, changed


def param_count(index_json):
    """Total parameters, from the safetensors index.

    `total_size` is a byte count, so it needs the on-disk dtype to become
    a parameter count. The index does not record the dtype, but the
    checkpoint dtype in config.json does, and bf16 is the near-universal
    distribution format.
    """
    if not index_json:
        return None
    try:
        meta = json.loads(index_json).get("metadata", {})
        total = meta.get("total_size")
        return int(total) if total else None
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def count_full_attention_layers(tc):
    """KV grows only on full-attention layers; hybrid models interleave
    them with linear/recurrent layers that carry fixed-size state.
    Mirrors `profile_from_qwen3_5_config`."""
    layer_types = tc.get("layer_types")
    if isinstance(layer_types, list) and layer_types:
        counted = sum(1 for t in layer_types if t == "full_attention")
        if counted:
            return counted, "layer_types"
    n = tc.get("num_hidden_layers")
    if not n:
        return None, None
    interval = tc.get("full_attention_interval")
    if interval:
        return n // max(int(interval), 1), "full_attention_interval"
    return n, "all layers (no hybrid markers)"


def kv_bytes_per_token(n_full, n_kv_heads, head_dim, world_size=1):
    per_rank = max(n_kv_heads // max(world_size, 1), 1)
    return 2 * n_full * per_rank * head_dim * 2


def render_tool_demo(template):
    """Render the chat template over a tool-calling exchange.

    A model's tool-call wire format is not described anywhere except in
    its chat template, and getting it wrong is expensive: the parser has
    to recognise whatever the template emits, and the mismatch only
    shows up as tools silently never being called. Rendering a canned
    exchange makes the format readable in seconds.

    This uses jinja2, not the minijinja the harness actually renders
    with, so it says nothing about whether the template will *work* in
    production -- the Rust test over the verbatim template is what
    proves that. It answers a narrower question: what do tool calls look
    like on the wire.
    """
    try:
        import jinja2
    except ImportError:
        return None, "jinja2 not installed"

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the weather in Tbilisi?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"location": "Tbilisi"},
                    },
                }
            ],
        },
        {"role": "tool", "name": "get_weather", "content": "18C, clear"},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Look up the weather for a place.",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }
    ]

    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = lambda v, **kw: json.dumps(v, ensure_ascii=False)

    def raise_exception(msg):
        raise RuntimeError(msg)

    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: "2026-01-01"
    try:
        out = env.from_string(template).render(
            messages=messages,
            tools=tools,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return out, None
    except Exception as e:  # noqa: BLE001 - any template error is just "did not render"
        return None, f"{type(e).__name__}: {e}"


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("repo", nargs="?", help="HuggingFace repo id")
    ap.add_argument("--local", help="path to an already-downloaded snapshot")
    ap.add_argument(
        "--reference",
        default="Qwen/Qwen3.6-27B",
        help="repo id to diff the config against (default: %(default)s)",
    )
    ap.add_argument("--token", help="HF token for gated repos")
    ap.add_argument("--world-size", type=int, default=2, help="TP degree for KV math")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument(
        "--render",
        action="store_true",
        help="render the chat template over a tool-calling exchange",
    )
    args = ap.parse_args()

    if not args.repo and not args.local:
        ap.error("give a repo id or --local")

    files = load_files(args.repo, args.local, args.token)
    if not files.get("config.json"):
        where = args.local or args.repo
        print(f"no config.json for {where} — not public yet, gated, or wrong id")
        return 2

    cfg = json.loads(files["config.json"])
    tc = text_config(cfg)
    model_type = cfg.get("model_type", "")
    archs = cfg.get("architectures", [])

    supported = supported_model_types()
    dense_ok = model_type in supported.get("DENSE_SUPPORTED_MODEL_TYPES", [])
    tp_ok = model_type in supported.get("TP_SUPPORTED_MODEL_TYPES", [])

    total_bytes = param_count(files.get("model.safetensors.index.json"))
    dtype = str(cfg.get("torch_dtype") or cfg.get("dtype") or "bfloat16")
    bytes_per = 2 if "16" in dtype else 4
    params = total_bytes // bytes_per if total_bytes else None

    n_full, basis = count_full_attention_layers(tc)
    n_kv = tc.get("num_key_value_heads")
    head_dim = tc.get("head_dim")
    if head_dim is None and tc.get("hidden_size") and tc.get("num_attention_heads"):
        head_dim = tc["hidden_size"] // tc["num_attention_heads"]

    kv = None
    if n_full and n_kv and head_dim:
        kv = kv_bytes_per_token(n_full, n_kv, head_dim, args.world_size)

    report = {
        "model": args.local or args.repo,
        "model_type": model_type,
        "architectures": archs,
        "dense_supported": dense_ok,
        "tp_supported": tp_ok,
        "params": params,
        "checkpoint_bytes": total_bytes,
        "max_position_embeddings": tc.get("max_position_embeddings"),
        "num_hidden_layers": tc.get("num_hidden_layers"),
        "full_attention_layers": n_full,
        "full_attention_basis": basis,
        "num_key_value_heads": n_kv,
        "head_dim": head_dim,
        "kv_bytes_per_token_per_card": kv,
        "has_vision": "vision_config" in cfg,
        "has_chat_template": bool(files.get("chat_template.jinja"))
        or "chat_template" in json.loads(files.get("tokenizer_config.json") or "{}"),
    }

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    print(f"\n=== {report['model']} ===\n")
    print(f"  model_type      {model_type or '(missing!)'}")
    print(f"  architectures   {', '.join(archs) or '(none listed)'}")
    print(f"  dense path      {'SUPPORTED' if dense_ok else 'NOT SUPPORTED'}")
    print(f"  tensor parallel {'SUPPORTED' if tp_ok else 'NOT SUPPORTED'}")
    print(f"  vision tower    {'yes' if report['has_vision'] else 'no'}")
    print(f"  chat template   {'yes' if report['has_chat_template'] else 'no'}")

    print("\n  -- size --")
    if params:
        print(f"  parameters      {params / 1e9:.1f}B")
        print(f"  checkpoint      {human(total_bytes)} ({dtype})")
        for name, bp in BYTES_PER_PARAM.items():
            print(f"  {name:<15} {human(params * bp)} of weights")
    else:
        print("  (no safetensors index — size unknown, may be GGUF-only)")

    print("\n  -- context --")
    print(f"  max positions   {tc.get('max_position_embeddings')}")
    print(f"  layers          {tc.get('num_hidden_layers')}")
    print(f"  full-attn       {n_full} (via {basis})")
    if kv:
        print(f"  kv/token/card   {kv} B  (world_size={args.world_size})")
        print(f"  128k context    {human(kv * 131072)} of KV per card")
    else:
        print("  kv/token/card   could not derive")

    tmpl = files.get("chat_template.jinja")
    if not tmpl:
        # Older repos inline the template in tokenizer_config.json
        # instead of shipping it as a standalone file.
        tk = json.loads(files.get("tokenizer_config.json") or "{}")
        tmpl = tk.get("chat_template") if isinstance(tk.get("chat_template"), str) else None
    if tmpl:
        print(f"\n  -- chat template ({len(tmpl)} bytes) --")
        print(f"  mentions tools  {'yes' if 'tool' in tmpl.lower() else 'no'}")
        if args.render:
            rendered, err = render_tool_demo(tmpl)
            if err:
                print(f"  tool-call render did not complete: {err}")
            else:
                print("  rendered tool-call exchange (jinja2, not minijinja):")
                print("  " + "-" * 60)
                for line in rendered.splitlines():
                    print(f"  | {line}")
                print("  " + "-" * 60)

    ref_cfg_text = fetch(args.reference, "config.json", args.token)
    if ref_cfg_text:
        added, removed, changed = diff_configs(cfg, json.loads(ref_cfg_text))
        print(f"\n  -- config diff vs {args.reference} --")
        n = len(added) + len(removed) + len(changed)
        if n == 0:
            print("  identical")
        for k, v in sorted(added.items()):
            print(f"  + {k} = {v}")
        for k in sorted(removed):
            print(f"  - {k}")
        for k, (was, now) in sorted(changed.items()):
            print(f"  ~ {k}: {was} -> {now}")
        print(f"\n  {n} field(s) differ")

    print("\n  -- verdict --")
    if dense_ok and tp_ok:
        print("  Already-supported model_type. Expect catalogue entry + load,")
        print("  with the config diff above as the risk surface.")
    elif dense_ok:
        print("  Dense path supports this; TP does not. Single-GPU only until")
        print("  a harness/tp/tp_<family>.rs module is added.")
    else:
        print(f"  NEW model_type '{model_type}'. Needs a ModelArch variant and")
        print("  dispatch in candle.rs. Judge the true cost from the diff:")
        print("  a near-empty diff means the arch is a rename.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
