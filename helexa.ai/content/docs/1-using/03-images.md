---
title: Images
sidebar_label: Images
description: Generating images in the browser — sizes, seeds, steps, and what they cost.
---

# Images

[The image workspace](/images) generates images from a text prompt on
the same fleet that serves chat.

## The basics

Describe what you want and generate. The first image after a quiet
period may take longer, because the image model has to load onto a GPU
before it can start.

Generated images are stored **in your browser**, like conversations,
along with the settings that produced them. Clearing site data deletes
them, so download anything you want to keep.

## Size and orientation

Square, portrait and landscape sizes are offered. Both dimensions must
be multiples of 16 — the interface only offers valid sizes, so this
matters mainly if you are calling [the API](/docs/using/api) directly.

Larger is not automatically better. Cost and time scale with the pixel
count, and each model has a maximum it can render well; beyond that,
results tend to degrade before they fail.

## Seeds

A seed fixes the random noise a generation starts from. The same prompt
with the same seed and settings gives the same image.

This is the single most useful control here. Without a fixed seed,
changing a word in your prompt changes *everything*, and you cannot tell
whether the difference came from your edit or from new noise. With one
fixed, you are actually comparing prompts.

Leave it unset to get something new each time.

## Steps

Steps control how many denoising passes the model makes. The default is
tuned for the model — more steps cost proportionally more and stop
helping fairly quickly, and fewer produce softer, less coherent images.

## Negative prompts

A negative prompt says what to avoid. It is worth knowing that this
turns on classifier-free guidance, which **doubles both the time and the
cost** of a generation, because the model effectively renders the prompt
and its negation at each step.

Useful when you need it. Not something to set by default.

## What it costs

Images are metered in **megapixel-steps** rather than tokens:

```
width × height × steps ÷ 1,000,000     (doubled when a negative prompt is used)
```

That reflects what actually occupies the GPU, so a small quick image
costs a small fraction of a large slow one rather than both counting as
"an image". A 512×512 at 4 steps is about 1 unit; a 1024×1024 at 9 steps
is about 9.4.
