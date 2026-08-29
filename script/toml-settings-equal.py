#!/usr/bin/env python3
"""Exit 0 when two TOML files carry the same settings, ignoring comments.

A deploy needs to answer two questions about a config it is about to
ship, and they are not the same question:

  * has anything changed at all?  → decides whether to **sync**
  * have the settings changed?    → decides whether to **restart**

Byte comparison answers only the first. Conflating them costs requests:
the first run of deploy.yml's router sync bounced helexa-router because
a comment header had changed, dropping whatever federation requests were
in flight. The router reads its aliases at startup, so a comment is not a
reason to interrupt anyone.

Usage:
    toml-settings-equal.py <a.toml> <b.toml>

Exit status:
    0  the parsed settings are identical (a restart would achieve nothing)
    1  they differ, or either file is missing or unparseable

Failing "differ" on an unparseable file is deliberate. Being wrong that
way costs one restart; being wrong the other way leaves a service running
settings nobody deployed, which is the failure this whole area exists to
prevent.
"""

from __future__ import annotations

import sys
import tomllib


def load(path: str) -> dict | None:
    try:
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        print(f"{path}: {exc}", file=sys.stderr)
        return None


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(f"usage: {argv[0]} <a.toml> <b.toml>", file=sys.stderr)
        return 1

    a, b = load(argv[1]), load(argv[2])
    if a is None or b is None:
        print("treating as changed: a file could not be parsed", file=sys.stderr)
        return 1

    if a == b:
        print("settings identical")
        return 0

    # Name the top-level sections that moved. Enough to make a deploy log
    # say *why* it restarted something, without dumping values — these
    # files are non-secret today, but a log is the wrong place to learn
    # that assumption has changed.
    moved = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    print(f"settings differ in: {', '.join(moved)}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
