---
title: Placement & displacement
description: Which neuron a model lands on, and what it may push aside to get there.
---

# Model placement & displacement

Which neuron a model lands on, and what it is allowed to push aside to
get there. Two settings in `models.toml` govern this, and they answer
different questions — conflating them is the most common way to end up
with a fleet that either refuses to load a model or quietly evicts the
one you most wanted to keep.

## The two knobs

| Knob | Question it answers | Default |
|---|---|---|
| `pinned_on` | **Where** may this model run? | empty = anywhere its device constraints allow |
| `residency_priority` | **What** may it push aside once there — and what may push *it* aside? | 100 (but see [legacy](#legacy-catalogues) below) |

They are independent on purpose. A model can be confined to one neuron
without being protected there, and protected everywhere without being
confined anywhere.

```toml
[[models]]
id = "Qwen/Qwen3.6-27B"
harness = "candle"
min_devices = 2
min_device_vram_mb = 24000
pinned_on = ["beast"]        # only beast has two big cards
residency_priority = 300     # and this is what protects it there
```

> **`pinned_on` does not protect anything.** It once did — it used to
> mean both "run only here" and "never evict here" — but those were
> split so the two could be set separately. If you are reading an older
> catalogue, or older notes, assume the word "pinned" means immunity and
> check whether the entry now carries a priority.

## Priority is a class, not a ranking

A model may displace residents **of its own class or below**, and
nothing above it.

Equal rank permits displacement, in both directions. This is the part
that surprises people, and it is deliberate: two models that share a
node and take turns on it each need to evict the other on demand. That
is what a cold-swap *is*.

So:

- **Models that should take turns on a node share a number.**
- **A model is protected by being ranked _above_ whatever must not evict
  it** — never by being given a number nobody else has.

### The trap

Giving two peers different numbers makes their swap one-directional.
The higher one takes the node and the lower one can never take it back.
Nothing errors; the model simply stops coming back, which usually gets
reported as "the text tier disappeared after I generated an image" or
"the old model is gone since we added the new one".

If two models are meant to alternate, give them the *same* priority.

## A worked fleet

Three rules that a single pinned/unpinned flag cannot express together,
because two of them protect the same model and disagree about who is
doing the pushing:

1. Image generation may take the mid tier's node when someone asks for
   an image — and the next text request takes it back.
2. Image generation must **never** take the flagship's node.
3. The frontier coder model **may** take the flagship's node.

Expressed as two classes:

| Priority | Models | Behaviour |
|---|---|---|
| 300 | flagship 27B, frontier coder, frontier thinking | share the big node; any may displace any other, both ways |
| 200 | image generator, mid-tier text | share the smaller node; take turns on it |
| 100 | small models (default) | displace nothing above themselves |

Rule 1 works because the image generator and the mid tier share `200`.
Rule 2 works because `200` never reaches `300`, however idle the
flagship is — worth checking, since the image model's device
constraints alone would happily let it land on the big node. Rule 3
works because the frontier model shares `300` with the flagship.

## Displacement is permitted, not required

Priority decides who **may** be displaced. It never decides whether a
displacement is **needed**.

The router prefers, in order: a neuron the model is pinned to; one whose
free VRAM already fits it; one that could fit it after evicting models
it outranks; then simply the one with the most free VRAM. Free-fit
outranks evict-fit, so **a model never displaces anything while a
neuron with room exists** — regardless of how the two rank.

A high priority is therefore not an instruction to be aggressive. It is
only an answer to "if something has to go, may it be this one?"

## Models missing from the catalogue

A model can be resident on a neuron without a catalogue profile — loaded
directly, or left behind by an earlier catalogue. Those rank at the
default rather than being treated as protected, so an unlisted model
cannot wedge a neuron permanently.

## Legacy catalogues

A profile that has `pinned_on` and **no** `residency_priority` defaults
to `1000` rather than `100`. This preserves the immunity `pinned_on`
used to grant on its own, so a catalogue written before the split does
not silently start allowing its flagship to be evicted.

Set an explicit priority when you touch such an entry — `1000` is a
compatibility floor, not a recommendation, and every legacy-pinned model
sharing it means they can all displace each other.

## Where the settings live

`models.toml` on the gateway host — operator-owned and not tracked in
git. `models.example.toml` in the repository carries the full field
reference. Changes take effect when cortex reloads the catalogue, so a
priority change is a config deployment, not a rebuild.
