#!/usr/bin/env python3
"""Fail when the two files describing one model disagree.

`models.toml` (cortex's catalogue, used for cold loads) and
`asset/neuron/<host>.toml` (`[[default_models]]`, used for models a host
holds resident) both describe *how a model is served*. Nothing forced
them to agree, and #252 is what that costs: models.toml omitted `quant`,
so a cortex cold-load served bf16 and could not prefill, while the same
model was fine when the host loaded it from its own config.

#283 adds a second such field — the operator sampling override — so this
check exists before the second instance of the bug rather than after it.

Checked, per model id present in both files:

  * ``quant``      — the #252 field
  * ``sampling``   — the #283 field, compared field-by-field

A model in only one file is not an error: plenty of catalogue entries are
never resident anywhere, and a host may hold a model the catalogue does
not list. Only *disagreement* fails.

Usage::

    check-config-consistency.py                       # repo defaults
    check-config-consistency.py --models /etc/cortex/models.toml \\
                               --neuron asset/neuron/beast.toml

Exit status is 0 when consistent, 1 on any disagreement, 2 on bad input.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    print("error: needs Python 3.11+ for tomllib", file=sys.stderr)
    raise SystemExit(2)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Fields that must agree wherever both files mention the same model.
# Extend this list whenever a serving-critical field lands in both.
SCALAR_FIELDS = ["quant"]
TABLE_FIELDS = ["sampling"]


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


def catalogue_entries(path: str) -> dict[str, dict]:
    """model id -> profile, from models.toml's `[[models]]`."""
    return {m["id"]: m for m in load(path).get("models", []) if "id" in m}


def resident_entries(path: str) -> dict[str, dict]:
    """model id -> spec, from a neuron config's `[[default_models]]`."""
    return {
        m["model_id"]: m
        for m in load(path).get("default_models", [])
        if "model_id" in m
    }


def label(path: str) -> str:
    """Repo-relative when the file is in the repo, basename otherwise.

    Deploy runs pass paths outside the tree (a models.toml fetched off
    the gateway), and `../../../..`-prefixed relpaths make the diff
    unreadable in a CI log.
    """
    rel = os.path.relpath(path, REPO)
    return rel if not rel.startswith("..") else os.path.basename(path)


def compare(model_id: str, cat: dict, res: dict, catalogue_path: str, neuron_path: str):
    """Yield one human-readable line per disagreement."""
    cat_name = label(catalogue_path)
    neu_name = label(neuron_path)

    for field in SCALAR_FIELDS:
        a, b = cat.get(field), res.get(field)
        # Absent on one side is "unspecified", not "different" -- but for
        # quant specifically that IS the #252 bug, so say so loudly.
        if a != b:
            if a is None or b is None:
                detail = (
                    "  (one side leaves it unset -- this is exactly #252: "
                    "an unset quant makes a cold load serve bf16)"
                    if field == "quant"
                    else "  (one side leaves it unset)"
                )
            else:
                detail = ""
            yield (
                f"{model_id}: {field} differs\n"
                f"    {cat_name}: {a!r}\n"
                f"    {neu_name}: {b!r}{detail}"
            )

    for field in TABLE_FIELDS:
        a = cat.get(field) or {}
        b = res.get(field) or {}
        if a == b:
            continue
        keys = sorted(set(a) | set(b))
        rows = "\n".join(
            f"      {k}: {cat_name}={a.get(k)!r}  {neu_name}={b.get(k)!r}"
            for k in keys
            if a.get(k) != b.get(k)
        )
        yield (
            f"{model_id}: {field} differs\n"
            f"    a model must sample the same way whether cortex cold-loads it\n"
            f"    or the host holds it resident:\n{rows}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=os.path.join(REPO, "models.toml"),
                    help="path to models.toml (default: repo root)")
    ap.add_argument("--neuron", nargs="*", default=None,
                    help="neuron config(s) (default: asset/neuron/*.toml)")
    args = ap.parse_args()

    neuron_paths = args.neuron
    if not neuron_paths:
        neuron_paths = sorted(glob.glob(os.path.join(REPO, "asset", "neuron", "*.toml")))
    if not neuron_paths:
        print("error: no neuron configs found", file=sys.stderr)
        return 2

    catalogue = catalogue_entries(args.models)
    problems: list[str] = []
    compared = 0

    for npath in neuron_paths:
        for model_id, spec in resident_entries(npath).items():
            profile = catalogue.get(model_id)
            if profile is None:
                # A host may hold a model the catalogue does not list.
                # Not an error, but worth seeing -- cortex cannot route
                # to what it has no profile for.
                print(f"note: {model_id} is resident in {label(npath)} "
                      f"but absent from {label(args.models)}")
                continue
            compared += 1
            problems.extend(compare(model_id, profile, spec, args.models, npath))

    if problems:
        print(f"\nconfig inconsistency: {len(problems)} disagreement(s)\n",
              file=sys.stderr)
        for p in problems:
            print(f"  {p}\n", file=sys.stderr)
        print("Two config files describe one model. Make them agree, or the\n"
              "model is served one way on a cold load and another when\n"
              "resident -- see #252 and #283.", file=sys.stderr)
        return 1

    print(f"config consistency OK: {compared} model/host pairing(s) agree "
          f"on {', '.join(SCALAR_FIELDS + TABLE_FIELDS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
