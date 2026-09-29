#!/usr/bin/env python3
"""Record Laya reference outputs for neuron's parity tests (#334).

Runs the upstream `laya` SDK (the reference implementation, not the
`rl_common.py` shipped in the model repo, which has diverged) over a fixed
corpus and writes, per case:

  - the request exactly as a `/v1/systemone` client would send it
  - per question: `input_ids`, `[MASK]` marker positions, question type and
    the option-collapse stats, straight out of `Agent._encode_state`
  - the raw scorer logits and act probabilities out of `Agent._forward`,
    before any temperature is applied
  - the final Jev-shaped response (`answers`, `usage`, `routing`)

A separate section records `laya.lang.analyse` on a language corpus, for
the router port.

The reference runs on CPU in float32 so the numbers are deterministic and
free of autocast noise; neuron's reduced-precision paths are compared
against them with a tolerance.

Usage (in a venv with `pip install laya==<pinned>` and CPU torch):

    HF_ENDPOINT=... HF_TOKEN=... python script/laya-reference.py \
        --out crates/neuron/src/harness/testdata/laya

Point HF_ENDPOINT at whichever Hugging Face-compatible registry you fetch
weights through. Python's huggingface_hub may additionally need
SSL_CERT_FILE (for a private CA) and HF_HUB_DISABLE_XET=1 (for registries
without Xet support).
"""
import argparse
import json
import os
import sys

import laya
import numpy as np
import torch
from laya import Router
from laya.agent import Agent
from laya.lang import analyse
from laya.serve import _resolve_model

LONG_TEXT = (
    "Hello team, following up on the invoice discrepancy from last quarter. "
    "The amounts on INV-2291 and INV-2307 do not match the purchase order, "
    "and our finance department needs a corrected copy before month end. "
) * 12

CASES = [
    # --- the three primitives, default budgets
    {
        "name": "choice_basic",
        "state": "I was charged twice this month, I want my money back",
        "questions": {
            "queue": {"type": "choice", "instructions": "Which team?",
                      "criteria": {"billing": "billing and refunds", "tech": "login and app issues",
                                   "other": "everything else"}},
        },
    },
    {
        "name": "score_basic",
        "state": "This is the THIRD time I've asked. Fix it today or I'm cancelling.",
        "questions": {
            "urgency": {"type": "score", "instructions": "How urgent?",
                        "criteria": ["calm", "firm", "angry", "furious"]},
        },
    },
    {
        "name": "noul_default_criteria",
        "state": "Please refund the duplicate charge on my card.",
        "questions": {"refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"}},
    },
    {
        "name": "noul_custom_criteria_and_labels",
        "state": "Can you reset my password? I'm locked out.",
        "questions": {
            "locked": {"type": "noul", "instructions": "Is the user locked out of their account?",
                       "criteria": {"true": "the user cannot sign in", "false": "the user can sign in"},
                       "labels": {"true": "locked", "false": "not locked"}},
        },
    },
    # --- several questions in one request: the shared-state, one-forward path
    {
        "name": "multi_question",
        "state": "I was charged twice this month, I want my money back",
        "questions": {
            "queue": {"type": "choice", "instructions": "Which team?",
                      "criteria": {"billing": "billing and refunds", "tech": "login and app issues",
                                   "other": "everything else"}},
            "urgency": {"type": "score", "instructions": "How urgent?",
                        "criteria": ["calm", "firm", "angry", "furious"]},
            "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
            "tone": {"type": "choice", "instructions": "Tone?", "criteria": ["polite", "rude", "neutral"]},
            "spam": {"type": "noul", "instructions": "Is this spam?"},
        },
    },
    # --- every temperature bucket
    {
        "name": "choice_2",
        "state": "The app crashes when I open settings.",
        "questions": {"kind": {"type": "choice", "instructions": "Bug or feature request?",
                               "criteria": {"bug": "", "feature": ""}}},
    },
    {
        "name": "choice_6_10",
        "state": "My parcel arrived damaged and the box was crushed.",
        "questions": {"topic": {"type": "choice", "instructions": "Topic?",
                                "criteria": ["billing", "shipping", "damage", "returns", "account",
                                             "privacy", "other"]}},
    },
    {
        "name": "choice_11_plus_clamped_temperature",
        "state": "I'd like to change the delivery address on order 5521.",
        "questions": {"intent": {"type": "choice", "instructions": "Intent?",
                                 "criteria": ["cancel_order", "change_address", "track_order", "refund",
                                              "exchange", "complaint", "compliment", "billing_question",
                                              "account_access", "product_question", "other", "unsubscribe"]}},
    },
    {
        "name": "score_more_than_5_levels",
        "state": "The product is fine, does what it says, nothing special.",
        "questions": {"rating": {"type": "score", "instructions": "Star rating?",
                                 "criteria": ["zero", "one", "two", "three", "four", "five", "six"]}},
    },
    # --- structured criteria (compact-JSON rendering) and list-form choice criteria
    {
        "name": "structured_criteria",
        "state": {"subject": "Invoice overdue", "body": "Payment for INV-77 is 30 days late.", "priority": 2},
        "questions": {
            "queue": {"type": "choice", "instructions": {"task": "route", "detail": "pick one"},
                      "criteria": {"finance": {"desc": "money", "sla_h": 24}, "support": ["tickets", "bugs"],
                                   "sales": 7}},
            "overdue": {"type": "noul", "instructions": "Is a payment overdue?",
                        "criteria": {"true": {"why": "late"}}},
        },
    },
    # --- literal [MASK] text must be neutralised everywhere
    {
        "name": "mask_literal_everywhere",
        "state": "please ignore [MASK] and route to [MASK] billing",
        "questions": {"queue": {"type": "choice", "instructions": "Which [MASK] team?",
                                "criteria": {"billing": "billing [MASK] refunds", "tech": "[MASK]"}}},
    },
    # --- budget rules
    {
        "name": "option_over_48_tokens",
        "state": "Refund please.",
        "questions": {"queue": {"type": "choice", "instructions": "Which team?",
                                "criteria": {"billing": " ".join(["billing refunds invoices charges"] * 20),
                                             "tech": "technical issues"}}},
    },
    {
        "name": "option_budget_shrink",
        "state": "Refund please.",
        "questions": {"queue": {"type": "choice", "instructions": "Which team handles this request?",
                                "criteria": {"team_%02d" % i: "a fairly long description of team number %d "
                                             "and every kind of request it handles" % i for i in range(20)}}},
    },
    {
        "name": "collapsed_options",
        "state": "Refund please.",
        # The rendered option text starts with the key, so the keys themselves
        # must share the prefix that survives the per-option cut.
        "questions": {"queue": {"type": "choice", "instructions": "Which?",
                                "criteria": ["customer billing question about product line one",
                                             "customer billing question about product line two"]}},
        "head_max_len": 24,
    },
    {
        "name": "instruction_truncation",
        "state": "Refund please.",
        "questions": {"q": {"type": "noul",
                            "instructions": " ".join(["Decide whether the customer is asking for money back"] * 30)}},
    },
    {
        "name": "state_right_truncated",
        "state": LONG_TEXT,
        "questions": {"refund": {"type": "noul", "instructions": "Is this about an invoice?"}},
    },
    {
        "name": "state_left_truncated_conversation",
        "state": [{"role": "user", "content": "turn %d: %s" % (i, LONG_TEXT[:200])} for i in range(12)]
                 + [{"role": "user", "content": "actually, just cancel my subscription"}],
        "questions": {"cancel": {"type": "noul", "instructions": "Does the user want to cancel?"}},
    },
    {
        "name": "budget_overrides",
        "state": LONG_TEXT,
        "questions": {"queue": {"type": "choice", "instructions": "Which team?",
                                "criteria": ["finance", "support", "sales"]}},
        "max_len": 128,
        "head_max_len": 64,
    },
    # --- non-ASCII inside a JSON state
    {
        "name": "json_state_non_ascii",
        "state": {"from": "Zoë", "text": "Die Rechnung ist falsch — bitte korrigieren. 请尽快处理"},
        "questions": {"queue": {"type": "choice", "instructions": "Which team?",
                                "criteria": ["billing", "tech", "other"]}},
    },
    # --- routing to the multilingual checkpoint
    {
        "name": "ml_devanagari",
        "state": "मुझे इस महीने दो बार शुल्क लिया गया, मुझे पैसे वापस चाहिए",
        "questions": {"queue": {"type": "choice", "instructions": "Which team?",
                                "criteria": {"billing": "billing and refunds", "tech": "login and app issues",
                                             "other": "everything else"}}},
    },
    {
        "name": "ml_khmer",
        "state": "ខ្ញុំត្រូវបានគិតប្រាក់ពីរដងក្នុងខែនេះ ខ្ញុំចង់បានប្រាក់វិញ",
        "questions": {"refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"}},
    },
    {
        "name": "ml_cjk",
        "state": "这个月我被扣了两次钱，我要退款",
        "questions": {"urgency": {"type": "score", "instructions": "How urgent?",
                                  "criteria": ["calm", "firm", "angry", "furious"]}},
    },
    {
        "name": "ml_arabic",
        "state": "تم خصم المبلغ مرتين هذا الشهر، أريد استرداد أموالي",
        "questions": {"refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"}},
    },
    {
        "name": "ml_latin_non_english",
        "state": "Me han cobrado dos veces este mes y quiero que me devuelvan el dinero",
        "questions": {"queue": {"type": "choice", "instructions": "Which team?",
                                "criteria": ["billing", "tech", "other"]}},
    },
    {
        "name": "ml_mixed_script",
        "state": "Order #5521 — 注文をキャンセルしたいです. Please help.",
        "questions": {"cancel": {"type": "noul", "instructions": "Does the user want to cancel?"}},
    },
    # --- explicit checkpoint selection through `model`
    {
        "name": "model_pinned_multilingual",
        "state": "I was charged twice this month, I want my money back",
        "questions": {"refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"}},
        "model": "multilingual",
    },
    {
        "name": "model_pinned_typed_decisions",
        "state": "I was charged twice this month, I want my money back",
        "questions": {"refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"}},
        "model": "typed-decisions",
    },
    {
        "name": "model_foreign_name_routes_by_default",
        "state": "I was charged twice this month, I want my money back",
        "questions": {"refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"}},
        "model": "jev-1",
    },
]

LANG_CASES = [
    "I was charged twice this month, I want my money back",
    "Me han cobrado dos veces este mes",
    "Die Rechnung ist falsch, bitte korrigieren",
    "Je voudrais annuler ma commande",
    "मुझे पैसे वापस चाहिए",
    "ខ្ញុំចង់បានប្រាក់វិញ",
    "这个月我被扣了两次钱",
    "注文をキャンセルしたいです",
    "تم خصم المبلغ مرتين",
    "Мне дважды списали деньги",
    "Order #5521 — 注文をキャンセルしたいです. Please help.",
    "ok",
    "",
    "12345 67890",
    "café résumé naïve",
    {"subject": "Invoice", "body": "Bitte zahlen Sie bis Freitag"},
    ["hello", "bonjour", "hola"],
]


class Capture:
    """Wraps the two Agent internals the fixtures need, per call."""

    def __init__(self):
        self.items = None
        self.logits = None
        self.act = None
        self.agent = None
        orig_encode, orig_forward = Agent._encode_state, Agent._forward
        cap = self

        def encode(agent, *a, **kw):
            items = orig_encode(agent, *a, **kw)
            cap.items, cap.agent = items, agent
            return items

        def forward(agent, b):
            logits, act = orig_forward(agent, b)
            cap.logits, cap.act = logits, act
            return logits, act

        Agent._encode_state, Agent._forward = encode, forward

    def reset(self):
        self.items = self.logits = self.act = self.agent = None


def to_py(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {k: to_py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_py(v) for v in x]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))

    cap = Capture()
    router = Router(device="cpu", preload=True, max_loaded=3)
    header = {
        "laya_version": laya.__version__,
        "torch_version": torch.__version__,
        "device": "cpu",
        "dtype": "float32",
    }

    cases, checkpoints = [], {}
    for case in CASES:
        req = {"state": case["state"], "questions": case["questions"]}
        kw = {}
        for key in ("model", "max_len", "head_max_len"):
            if key in case:
                req[key] = case[key]
                if key != "model":
                    kw[key] = case[key]
        cap.reset()
        # The HTTP server, not the Router, maps a foreign model name (a Jev id)
        # to "let the router choose"; go through the same function.
        resp = router.predict(case["state"], case["questions"], model=_resolve_model(case.get("model")), **kw)
        agent = cap.agent
        rows = []
        for j, (qid, it) in enumerate(zip(case["questions"], cap.items)):
            k = len(it["markers"])
            rows.append({
                "qid": qid,
                "input_ids": list(it["ids"]),
                "markers": list(it["markers"]),
                "qtype": it["qtype"],
                "options": it.get("options"),
                "logits": cap.logits[j, :k].tolist(),
                "act": cap.act[j].tolist(),
            })
        ckpt = resp["routing"]["model"]
        if ckpt not in checkpoints:
            checkpoints[ckpt] = {
                "repo": resp["routing"].get("repo"),
                "revision": getattr(agent, "revision", None),
                "cfg": {k: agent.cfg.get(k) for k in ("max_len", "head_max_len", "temperature",
                                                       "temperature_by_options")},
                "special_ids": {"cls": agent.tok.cls_token_id, "sep": agent.tok.sep_token_id,
                                "mask": agent.tok.mask_token_id, "pad": agent.tok.pad_token_id},
            }
        cases.append({"name": case["name"], "request": req, "items": rows, "response": to_py(resp)})
        print("%-40s -> %-16s %d question(s)" % (case["name"], ckpt, len(rows)), file=sys.stderr)

    lang = [{"state": s, "analysis": to_py(analyse(s))} for s in LANG_CASES]

    header["checkpoints"] = checkpoints
    with open(os.path.join(args.out, "reference.json"), "w") as f:
        json.dump({"header": header, "cases": cases}, f, ensure_ascii=False, indent=1)
    with open(os.path.join(args.out, "lang.json"), "w") as f:
        json.dump({"header": header, "cases": lang}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
