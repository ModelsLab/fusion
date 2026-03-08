---
id: wan_rtx4090_mmgp_budget_sweep
kind: example
title: WAN RTX 4090 mmgp Budget Sweep Did Not Beat The Unrestricted Resident Winner
type: ""
category: video-generation
summary: On an RTX 4090 running the WAN qint8 T2V 4-step path, explicit `mmgp` transformer budgets enabled per-block async preloading, but every safe budget tested was slower than the unrestricted resident winner and the near-resident edge OOM'd.
support_level: measured
reliability: measured
review_status: reviewed
url: ""
tags:
  - wan
  - rtx4090
  - mmgp
  - prefetch
  - h2d
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

- Baseline unrestricted resident winner on the same host: about `21.61s` for `480p`, `21` requested frames, `4` steps.
- `WAN_MMGP_BUDGETS=transformer1=12750,transformer2=12750`: about `23.10s`
- `WAN_MMGP_BUDGETS=transformer1=12500,transformer2=12500`: about `23.32s`
- `WAN_MMGP_BUDGETS=transformer1=12000,transformer2=12000`: about `23.61s`
- `WAN_MMGP_BUDGETS=transformer1=10000,transformer2=10000`: about `25.28s`
- `13000 MB` per transformer OOM'd on the tested shape.

## Interpretation

- The budgeted branch is technically valid and makes `mmgp` split each WAN transformer into recurrent blocks with an async preload plan.
- On this 24 GB Ada host and this smaller `480p` workload, the unrestricted resident path is still better than budgeted staged prefetch.
- Budgeted prefetch is therefore a fit-first branch, not the default throughput winner.

## Practical Guidance

- Keep the unrestricted resident path as the promoted `480p` winner on this class of GPU.
- Reach for explicit transformer budgets only when the target resolution or frame shape stops fitting the resident path.
