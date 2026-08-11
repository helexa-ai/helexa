---
title: API reference
sidebar_label: API
description: The OpenAI- and Anthropic-compatible endpoints helexa serves, and where it differs.
---

# API reference

Base URL: `https://helexa.ai/v1`

helexa speaks three request shapes so existing clients work unmodified.
Pick whichever your SDK already speaks — they reach the same models.

| Endpoint | Shape |
|---|---|
| `POST /v1/chat/completions` | OpenAI chat completions, streaming or not |
| `POST /v1/responses` | OpenAI Responses API, including tool calls |
| `POST /v1/messages` | Anthropic Messages |
| `POST /v1/images/generations` | OpenAI images |
| `GET /v1/models` | catalogue, with per-model limits and pricing |

## Authentication

A bearer token on every request:

```
Authorization: Bearer <your-api-key>
```

Keys are created and revoked in [your account](/account/keys). A key is
shown once, at creation. If you lose it, revoke it and make another.

## Models and tiers

`GET /v1/models` returns everything currently servable, including a
context limit and price per million tokens for each entry. The tier
aliases (`helexa/small`, `helexa/balanced`, `helexa/large`,
`helexa/image`) resolve to concrete models and are the right choice
unless you specifically need one model to stay put.

The list reflects what the fleet can actually serve right now, not a
static catalogue — a model whose host is unavailable is not advertised.

## Streaming

Set `"stream": true` for token-by-token delivery over SSE. The response
is a standard event stream terminated by `data: [DONE]`. Responses-API
streams carry `type` and `sequence_number` inside each event payload, as
that API requires.

## System prompts belong to you

**helexa never adds to your prompt.** No injected system message, no
house style, no default persona when you send none, and no reordering of
what you sent. Every system slot maps straight through:

- `/v1/chat/completions` — every `messages` entry with `role: "system"`,
  in the order you sent them.
- `/v1/responses` — `instructions`, plus `input` items with
  `role: "system"`.
- `/v1/messages` — top-level `system`, in both its string and
  content-block-array forms.

This is a contract, not a current default. Model behaviour is yours to
define.

## Images

```sh
curl https://helexa.ai/v1/images/generations \
  -H "Authorization: Bearer $HELEXA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "helexa/image",
    "prompt": "a lighthouse in fog, painted",
    "size": "1024x1024",
    "response_format": "b64_json"
  }'
```

`n` is 1 — ask again for another image. Output is PNG as base64.

Beyond the OpenAI fields, helexa accepts:

| Field | Effect |
|---|---|
| `seed` | fixes the noise, so the same prompt reproduces the same image |
| `num_steps` | denoising steps; more is slower, not always better |
| `negative_prompt` | what to avoid — **enables CFG, doubling time and cost** |
| `guidance_scale` | how strictly to follow the prompt when CFG is on |

Both dimensions must be **multiples of 16**, and each model has a
maximum. Requests that break either rule are rejected before any GPU
work, so they cost nothing.

Images are metered in **megapixel-steps** — `width × height × steps ÷
1,000,000`, doubled when CFG is on — rather than tokens, because that is
what actually consumes the GPU. The figure comes back in
`usage.helexa_image_units`.

## Errors

Errors use a consistent envelope:

```json
{ "error": { "code": "invalid_image_params", "message": "..." } }
```

Worth handling specifically:

| Status | Meaning |
|---|---|
| `400` | malformed request — including image dimensions that are not multiples of 16 |
| `401` | missing or revoked key |
| `422` | right endpoint, wrong modality — e.g. a chat request against an image model |
| `429` | rate limited, or the model is at capacity; honour `Retry-After` |
| `503` | no host can currently serve that model; also carries `Retry-After` |

`429` and `503` are normal under load rather than faults. They always
carry `Retry-After`, and respecting it is the difference between backing
off and making things worse.

## Timeouts

Allow at least 300 seconds. A cold model load happens before any tokens
are produced, and a large image with CFG is genuinely slow. A client
that gives up after 30 seconds will appear to fail against a perfectly
healthy fleet.
