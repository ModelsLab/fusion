---
id: wan_rtx3090_clock_locks_regressed
kind: example
title: WAN RTX 3090 Clock Locks Did Not Beat Auto Boost
type: ""
category: video-generation
summary: On the measured WAN 4-step resident path for RTX 3090, locking graphics and memory clocks to the card's maximum supported values did not improve throughput versus the default auto-boost behavior.
support_level: measured
reliability: measured
review_status: reviewed
url: ""
tags:
  - wan
  - rtx3090
  - clocks
  - auto-boost
  - negative-example
gpu_ids:
  - rtx3090
workloads:
  - video-generation
operators:
  - transformer
  - attention
precision:
  - bf16
  - int8
---

## Measured Setup

- workload: WAN text-to-video, 480p, 21 requested frames, 4 steps
- runtime: `WAN_OFFLOAD_PROFILE=highram-highvram`, `WAN_EMPTY_CACHE_POLICY=never`, `--no-offload-model`
- metric: steady-state `gen_s` after one warmup and three measured runs

## Results

- auto boost baseline: `29.1537s`
- fixed graphics lock `2100 MHz`: `29.2340s`
- fixed graphics `2100 MHz` plus memory `9751 MHz`: `29.3314s`

## Conclusion

- power was already at the card maximum `350 W`
- forcing max supported clocks did not improve this workload
- the path is still better described as transfer and staging limited than as pure compute limited
