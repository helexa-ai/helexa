---
title: Getting started
sidebar_label: Getting started
description: What helexa serves, how to sign in, and how to make your first request.
---

# Getting started

helexa serves open-weight models on hardware we own and operate in the
EU. You can use it two ways: through the web app, or through the API
with any OpenAI- or Anthropic-compatible client.

## In the browser

Nothing is required to start a conversation — open
[the chat](/chat) and type. Conversations are stored **in your browser**,
not on a server, so they survive reloads on the same device and go no
further.

Creating an account adds an API key, usage visibility, and higher
limits. It does not move your conversations anywhere.

## From code

Point any OpenAI-compatible client at `https://helexa.ai/v1` and use an
API key from [your account](/account/keys).

```sh
curl https://helexa.ai/v1/chat/completions \
  -H "Authorization: Bearer $HELEXA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "helexa/balanced",
    "messages": [{"role": "user", "content": "Say hello."}]
  }'
```

The same key works against the Anthropic-shaped `/v1/messages` endpoint
and the OpenAI `/v1/responses` endpoint. See [the API
reference](/docs/using/api) for the full surface.

## Choosing a model

Ask for a **tier** rather than a specific model, and you get whatever
currently fills that role:

| Tier | For |
|---|---|
| `helexa/small` | short, cheap, fast turns |
| `helexa/balanced` | general use — the sensible default |
| `helexa/large` | long context, harder reasoning, agentic work |
| `helexa/image` | text-to-image |

Tiers are stable; the model behind a tier is not. That is the point —
you get upgrades without changing code. When you need a specific model
and want it to stay that way, use its full id from `GET /v1/models`.

## Limits

Public API requests are rate-limited to **10 per minute per IP**, with a
short burst allowance. That suits interactive use and normal application
traffic. If you are running a batch job or a busy service against
helexa, talk to us rather than working around the limit.

Models load on demand. A request for a model that is not currently
resident on a GPU may take tens of seconds while it loads, then run at
full speed. Clients should allow a generous timeout — 300 seconds is
what our own tooling uses.

## What runs where

Every request is served on hardware helexa operates. Nothing is
forwarded to a third-party inference provider, and prompts are not used
for training. See [privacy](/privacy) for the specifics.
