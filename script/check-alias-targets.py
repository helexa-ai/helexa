#!/usr/bin/env python3
"""Fail when a product-tier alias names a model the catalogue does not define.

Two files describe one routing decision:

  * `models.toml`         — the catalogue cortex serves from, plus its own
                            `[aliases]` table.
  * `helexa-router.toml`  — the public edge's `[aliases]` table, which is
                            what an unauthenticated or web-chat caller
                            actually resolves against.

Nothing forced them to agree, and on 2026-08-27 they did not: the router
still mapped `helexa/balanced` to `Qwen/Qwen3.6-27B` after that model had
been retired. Because both 27Bs sat at equal `residency_priority` — a
deliberate choice so they could displace each other for an A/B — every
authenticated web-chat turn evicted beast's resident flagship and
cold-loaded the retired one, ~70 s of TP layer loading each way. Four of
those overlapped and left the node holding 32 GB on one card with no
model registered and nothing serving.

The failure needed no bad code, only two files and no check. This is that
check. It is the same shape as `check-config-consistency.py` (#252), which
exists because `models.toml` and a neuron config could disagree about
`quant`; that pairing now has a guard and this one did not.

Deliberately narrow: it asks only whether an alias target exists in the
catalogue. It does not judge *which* model a tier should point at — that
is a product decision, and a check that guessed at it would be wrong more
often than the operator.

Exit status is 0 when every alias resolves, 1 when any does not, 2 when a
file is missing or unparseable.
"""

from __future__ import annotations

import argparse
import difflib
import os
import sys
import tomllib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(path: str) -> dict:
    try:
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except FileNotFoundError:
        print(f"error: {path} not found", file=sys.stderr)
        raise SystemExit(2)
    except tomllib.TOMLDecodeError as exc:
        print(f"error: {path} is not valid TOML: {exc}", file=sys.stderr)
        raise SystemExit(2)


def catalogue_ids(path: str) -> set[str]:
    """Every model id `models.toml` defines."""
    return {m["id"] for m in load(path).get("models", []) if "id" in m}


def alias_table(path: str) -> dict[str, str]:
    """The `[aliases]` table, or an empty one if the file defines none."""
    table = load(path).get("aliases", {})
    return {k: v for k, v in table.items() if isinstance(v, str)}


def label(path: str) -> str:
    """Repo-relative when inside the tree, basename otherwise.

    Deploy runs pass paths from outside the checkout, and a `../../..`
    prefix makes a CI failure harder to read than the bare name.
    """
    rel = os.path.relpath(path, REPO)
    return rel if not rel.startswith("..") else os.path.basename(path)


def check(alias_path: str, catalogue_path: str, ids: set[str]) -> list[str]:
    """One human-readable line per alias that does not resolve."""
    problems = []
    for alias, target in sorted(alias_table(alias_path).items()):
        if target in ids:
            continue
        # Name the likely replacement. A retired model is normally
        # superseded by one with a near-identical id (3.6-27B -> 3.8-27B),
        # so a close match turns a puzzle into an edit. Substring matching
        # was tried first and listed every Qwen in the catalogue, which is
        # noise dressed as help — difflib ranks by actual similarity and a
        # 0.6 cutoff keeps it to genuine neighbours.
        near = difflib.get_close_matches(target, sorted(ids), n=2, cutoff=0.6)
        hint = f"\n      did you mean: {', '.join(near)}" if near else ""
        problems.append(
            f'{label(alias_path)}: "{alias}" -> "{target}"\n'
            f"    no such model in {label(catalogue_path)}{hint}"
        )
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--models",
        default=os.path.join(REPO, "models.toml"),
        help="path to models.toml (default: repo root)",
    )
    ap.add_argument(
        "--aliases",
        nargs="*",
        default=[os.path.join(REPO, "helexa-router.toml"), os.path.join(REPO, "models.toml")],
        help="config(s) carrying an [aliases] table (default: router config + models.toml)",
    )
    args = ap.parse_args()

    ids = catalogue_ids(args.models)
    if not ids:
        print(f"error: {label(args.models)} defines no models", file=sys.stderr)
        return 2

    problems = []
    checked = 0
    for path in args.aliases:
        table = alias_table(path)
        checked += len(table)
        problems.extend(check(path, args.models, ids))

    if problems:
        print("alias targets missing from the catalogue:\n", file=sys.stderr)
        for p in problems:
            print(f"  {p}\n", file=sys.stderr)
        print(
            "An alias pointing at a model the catalogue does not define does not\n"
            "fail loudly — cortex cold-loads it, which on a busy node means\n"
            "evicting whatever is resident. Point it at a served model, or\n"
            "restore the model to models.toml.",
            file=sys.stderr,
        )
        return 1

    print(f"alias targets OK: {checked} alias(es) resolve to catalogued models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
