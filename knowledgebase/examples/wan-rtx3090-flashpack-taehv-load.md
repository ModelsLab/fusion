---
id: wan_rtx3090_flashpack_taehv_load
kind: example
title: WAN RTX 3090 FlashPack Helped Cold Start, Not Steady-State
type: ""
category: video-generation
summary: On the WAN RTX 3090 environment, FlashPack reduced a representative TAEHV checkpoint load path, but it does not address the measured steady-state H2D-heavy generation bottleneck.
support_level: measured
reliability: measured
review_status: reviewed
url: ""
tags:
  - wan
  - rtx3090
  - flashpack
  - cold-start
  - startup
source_ids:
  - fal-ai-flashpack
  - pytorch-torch-load-mmap
gpu_ids:
  - rtx3090
workloads:
  - video-generation
operators:
  - vae
  - general
precision:
  - bf16
  - int8
---

## Measured Setup

- representative load path: WAN `TAEHV` checkpoint `models/taew2_1.pth`
- baseline: `torch.load(..., mmap=True)` plus `load_state_dict`
- candidate: FlashPack `pack_to_file` once, then `assign_from_file` into the same module shape

## Results

- baseline mean load/init time: about `80.3 ms`
- FlashPack mean assign time: about `59.0 ms`

## Conclusion

- FlashPack is a real cold-start optimization branch for this stack
- it is still not the main answer for WAN steady-state throughput on the RTX 3090 because the measured generation profile remains dominated by host-to-device transfer
