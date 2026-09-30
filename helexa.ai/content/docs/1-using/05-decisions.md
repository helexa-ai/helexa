---
title: Using decisions
sidebar_label: Decisions
description: Typed questions over text or JSON, answered with calibrated probabilities — the System-One endpoint and its playground.
---

# Using decisions

A **decision model** answers typed questions about a piece of text or
JSON — the *state* — in a single pass. It never generates text: every
answer is a probability distribution over options you define in the
request. That makes it fast (tens of milliseconds), cheap, and
predictable enough to put in front of a workflow: routing a ticket,
screening a message, deciding whether to escalate.

Try it without writing code in [the playground](/playground).

The endpoint is `POST https://helexa.ai/v1/systemone`, and it speaks the
TypeSafe Jev protocol, so existing Jev clients work unchanged.

## A request

```json
{
  "state": "I was charged twice this month and want the duplicate refunded.",
  "questions": {
    "queue": {
      "type": "choice",
      "instructions": "Which team should handle this ticket?",
      "criteria": {
        "billing": "payments, invoices and refunds",
        "technical": "login, bugs and app issues",
        "other": "anything else"
      }
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is this ticket?",
      "criteria": ["low", "normal", "high", "critical"]
    },
    "refund": {
      "type": "noul",
      "instructions": "Is the customer asking for money back?"
    }
  }
}
```

Every question over the same state is answered in the same request, and
`answers` comes back in the order you asked.

## The three question types

| Type | Asks | `criteria` | Answer |
|---|---|---|---|
| `noul` | Is a statement true? | optional `{"true": …, "false": …}` descriptions | `noul`: P(true) |
| `score` | Where does this fall on a scale? | a list of levels, lowest first | `score`: the expected level, which can fall between two |
| `choice` | Which of these options? | an object of `label: description`, or a list of labels | `choice`: the most likely label |

A `noul` question also takes `labels`, e.g. `{"true": "late", "false":
"on time"}`, to name its two answers.

Every answer carries `probabilities` for each option (choice and score),
plus two confidence numbers that are **not** interchangeable:

- `answer_confidence` is the probability of the chosen answer. It is the
  calibrated number: set thresholds on this one.
- `confidence` is how decided the whole distribution is — one minus the
  normalised entropy for choice and score, the larger probability for
  noul. It is not comparable across question types.

## The state

- A **string** is read as written.
- An **object** is serialised as JSON, keys in the order you wrote them.
- A **list** is read as a conversation. When it is too long, it is
  trimmed from the start, so the newest turns survive.

Order matters: the model reads question ids, option labels and object
keys in the order they arrive.

## Languages and checkpoints

The model family has three checkpoints: English, multilingual and
typed-decisions. By default the service picks one from the language of
the state — Hindi, Japanese, Arabic, Russian and non-English Latin text
all go to the multilingual checkpoint — and the response's `routing`
block says which answered and why.

Send `"model": "multilingual"` (or `english`, `typed-decisions`) to pin
one. `helexa/one` is the class alias, and any model name the service
doesn't recognise — such as a Jev client's own — falls back to it.

## Limits and budgets

- At most 64 questions, 100 options per choice, 32 levels per score,
  and 512 options across the request.
- The state is at most 50,000 characters.
- Each question and its options share a token budget with the state.
  Many or long options squeeze each other; if they are cut to the same
  tokens, `usage.options` says so, and the model can no longer tell
  them apart.

## Metering

Decisions are metered in **input tokens**, reported in
`usage.input_tokens`; `output_tokens` is always zero. Each question is
encoded together with the state, so every extra question over a long
state costs that state again.

## Errors

Errors use the same envelope as the rest of the API, plus a `detail`
string with the exact reason:

- `400` / `422`: the request needs fixing. `detail` names the question.
- `413`: over one of the limits above.
- `429` / `503`: slow down or retry; honour `Retry-After`. The public
  endpoint allows 10 requests a minute from each address without an API
  key.
