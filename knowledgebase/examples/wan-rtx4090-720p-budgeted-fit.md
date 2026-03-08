---
id: wan_rtx4090_720p_budgeted_fit
kind: example
title: WAN RTX 4090 720p Needed A Budgeted mmgp Fit-First Path
type: ""
category: video-generation
summary: On a 24 GB RTX 4090, the WAN 4-step resident path that won at 480p no longer fit at 720p. A budgeted `mmgp` branch around `12.5 GB` per transformer became the first working 720p path.
support_level: measured
reliability: measured
review_status: reviewed
url: ""
tags:
  - wan
  - rtx4090
  - 720p
  - mmgp
  - fit-first
source_ids:
  - huggingface-diffusers-group-offloading
  - flexgen-offload-compression
gpu_ids:
  - rtx4090
workloads:
  - video-generation
operators:
  - transformer
  - attention
  - offload
precision:
  - bf16
  - int8
---

## Measured Behavior

- Target shape: `720p`, `21` requested frames, `4` steps.
- Unrestricted resident path: OOM.
- Same high-VRAM profile with `--offload-model`: still OOM on the tested shape.
- `WAN_MMGP_BUDGETS=transformer1=12000,transformer2=12000`: about `37.65s`
- `WAN_MMGP_BUDGETS=transformer1=12500,transformer2=12500`: about `37.29s`
- `WAN_MMGP_BUDGETS=transformer1=12750,transformer2=12750`: OOM.

## Interpretation

- On this 24 GB Ada card, `720p` is not just a larger version of the `480p` winner.
- The first viable `720p` branch is an explicit fit-first `mmgp` budget, not the unrestricted resident path and not a plain model-offload flag.
- This is exactly the case where a staged-prefetch branch can be useful even though it loses at smaller shapes.

## Practical Guidance

- Keep the unrestricted resident path for `480p`.
- For `720p` on this host, start with a budgeted branch around `12500 MB` per transformer before jumping to a different model family or a new kernel project.
