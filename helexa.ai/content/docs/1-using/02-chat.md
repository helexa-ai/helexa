---
title: Chat
sidebar_label: Chat
description: How the chat workspace works, where your conversations live, and what the model can reach.
---

# Chat

[The chat workspace](/chat) is a normal conversation interface over the
same models the API serves. There is nothing different about the
inference — the difference is where your history lives.

## Your conversations stay in your browser

Messages, conversations and projects are stored in your browser's local
database. They are **not** sent to a server, not backed up by us, and
not readable by us.

Two consequences worth knowing before you rely on it:

- **Clearing site data deletes your history.** There is no server-side
  copy to restore from.
- **History does not follow you between devices or browsers.** Signing
  in on a phone gives you your account, not your conversations.

Only the messages needed to answer are sent, at the moment you send
them, and they are not retained afterwards.

## Projects

Conversations can be grouped into projects, which is worth doing once
you have more than a handful. Ungrouped conversations live under
*Unsorted*. Pinning keeps a conversation at the top of the list.

## Choosing a model

The model picker offers tiers — small, balanced, large — rather than
model names. Balanced is the right default. Reach for large when the
task needs long context or harder reasoning, and expect the first
message to take longer if that model has to load.

You can switch models mid-conversation. The new model sees the existing
history.

## Web search

When a question needs current information, the assistant can search the
web and read pages, then cite what it used. Citations appear beneath the
reply — follow them, particularly for anything time-sensitive.

Search is a capability of the larger models. Smaller ones will answer
from training data instead of reaching for a tool, which is the usual
reason a small model gives a confidently stale answer.

## Attaching images

Models with vision accept images alongside text — paste or attach one
and ask about it. Where a model has no vision tower, the option is not
offered rather than silently ignored.

## When something is slow

A first message to a model that is not currently loaded waits for it to
load, which can take tens of seconds. Subsequent messages are fast. The
interface shows that it is working rather than pretending to be stuck.

Under heavy load you may see a message about capacity. That is the fleet
protecting response times for requests already in flight rather than
degrading everything at once — waiting a moment and retrying is the
right response.
