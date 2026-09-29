#!/usr/bin/env python3
"""End-to-end check of /v1/systemone decision serving (#341).

Replays every case of the recorded Laya reference corpus
(`crates/neuron/src/harness/testdata/laya/reference.json`) against a
serving endpoint -- a neuron directly, a cortex, or the public router --
and compares each response with what the reference server returned:

  - `routing.model` (which checkpoint answered) and `usage` must match
    exactly;
  - each answer's type, and its chosen option or score level, must match,
    except where the reference itself was within `--tie` of a tie (reduced
    precision may break a near-tie either way);
  - probabilities and confidences must be within `--tolerance`. The
    reference is f32; a bf16 GPU path drifts from it by about as much as
    the reference's own bf16 GPU run does.

It prints each case's latency and a summary, and exits non-zero on any
mismatch. Nothing is loaded or unloaded: point it at an endpoint that
already serves a decision model.

Usage:
  script/validate-decision.py URL [--model M] [--bearer KEY]

  URL      http://quadbrat.hanzalova.internal:13131   (neuron)
           http://hanzalova.internal:31313             (cortex)
           https://router.example                      (public chain)
  --model  override every request's `model` (e.g. helexa/one); by
           default each case sends what it was recorded with, which
           includes checkpoint pins and a foreign Jev model name.
"""
import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "crates", "neuron", "src",
                       "harness", "testdata", "laya", "reference.json")


def post(url, body, bearer):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    if bearer:
        req.add_header("Authorization", "Bearer " + bearer)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            payload = json.load(r)
            status = r.status
    except urllib.error.HTTPError as e:
        payload = json.loads(e.read() or b"{}")
        status = e.code
    return status, payload, (time.perf_counter() - started) * 1e3


def top_gap(probs):
    p = sorted(probs, reverse=True)
    return p[0] - p[1] if len(p) > 1 else 1.0


def compare(name, got, want, tol, tie):
    errs = []
    if got.get("routing", {}).get("model") != want["routing"]["model"]:
        errs.append("routing.model %r != %r" % (got.get("routing", {}).get("model"), want["routing"]["model"]))
    if got.get("usage") != want["usage"]:
        errs.append("usage %r != %r" % (got.get("usage"), want["usage"]))
    ga, wa = got.get("answers", {}), want["answers"]
    if list(ga) != list(wa):
        errs.append("answer ids %r != %r" % (list(ga), list(wa)))
        return errs, 0.0
    worst = 0.0
    for qid, w in wa.items():
        g = ga[qid]
        if g.get("type") != w["type"]:
            errs.append("%s: type %r != %r" % (qid, g.get("type"), w["type"]))
            continue
        wp = w.get("probabilities")
        near_tie = wp is not None and top_gap(list(wp.values())) < tie
        if w["type"] == "choice" and g.get("choice") != w["choice"] and not near_tie:
            errs.append("%s: choice %r != %r" % (qid, g.get("choice"), w["choice"]))
        if wp:
            for opt, p in wp.items():
                worst = max(worst, abs(g["probabilities"][opt] - p))
        for key in ("noul", "confidence", "answer_confidence"):
            if key in w:
                worst = max(worst, abs(g[key] - w[key]))
    if worst > tol:
        errs.append("max |Δ| %.4f > %.4f" % (worst, tol))
    return errs, worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--model")
    ap.add_argument("--bearer")
    ap.add_argument("--tolerance", type=float, default=0.1)
    ap.add_argument("--tie", type=float, default=0.05)
    args = ap.parse_args()
    endpoint = args.url.rstrip("/") + "/v1/systemone"
    cases = json.load(open(FIXTURE))["cases"]

    failures, latencies, worst_all = 0, [], 0.0
    for case in cases:
        body = dict(case["request"])
        if args.model:
            body["model"] = args.model
        status, got, ms = post(endpoint, body, args.bearer)
        latencies.append(ms)
        if status != 200:
            failures += 1
            print("FAIL %-40s HTTP %d %s" % (case["name"], status, json.dumps(got)[:200]))
            continue
        errs, worst = compare(case["name"], got, case["response"], args.tolerance, args.tie)
        worst_all = max(worst_all, worst)
        mark = "ok  " if not errs else "FAIL"
        failures += bool(errs)
        print("%s %-40s %7.1f ms  %-15s max|Δ| %.4f  %s" % (
            mark, case["name"], ms, got.get("routing", {}).get("model", "?"), worst, "; ".join(errs)))

    print("\n%d/%d cases ok; latency median %.1f ms, p90 %.1f ms; worst |Δ| %.4f" % (
        len(cases) - failures, len(cases), statistics.median(latencies),
        sorted(latencies)[int(0.9 * (len(latencies) - 1))], worst_all))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
