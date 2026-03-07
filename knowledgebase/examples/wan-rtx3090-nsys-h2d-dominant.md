---
id: wan_rtx3090_nsys_h2d_dominant
kind: example
title: WAN RTX 3090 Nsight Systems Transfer-Dominant Trace
type: ""
category: video-generation
summary: A steady-state-only `nsys` trace of the current WAN 4-step resident path on an RTX 3090 showed that host-to-device movement and `cudaMemcpyAsync` still dominate after load and warmup are excluded, so residency and staging remain higher-value branches than new kernels.
support_level: measured
reliability: measured
review_status: reviewed
url: ""
tags:
  - wan
  - rtx3090
  - nsys
  - h2d
  - offload
source_ids:
  - huggingface-diffusers-group-offloading
  - flexgen-offload-compression
gpu_ids:
  - rtx3090
workloads:
  - video-generation
operators:
  - transformer
  - attention
  - vae
precision:
  - bf16
  - int8
---

## Measured Signals

- `cudaMemcpyAsync` consumed about `89.5%` of traced CUDA API time in the steady-state-only profile
- host-to-device copies consumed about `94.4%` of traced GPU memcpy time
- about `51.6 GB` of host-to-device traffic still occurred during the measured phase
- the top CUTLASS GEMM kernel consumed about `57.9%` of traced kernel time once work was on device
- `_attn_fwd` still consumed about `8.7%` of traced kernel time

## Interpretation

- the path is not purely compute-bound even after moving to the resident high-VRAM profile and excluding warmup
- transfer and staging pressure still matter enough that residency, offload policy, and prefetch strategy come before fresh kernel work
- `torch.load(..., mmap=True)` is not the main fix here because the hot path is loading qint8 safetensors via `mmgp`, not large `.pt` weights
- checkpoint-packing formats like FlashPack belong to the cold-start branch, not the steady-state throughput branch

## Practical Next Steps

- profile steady-state generation separately from one-time load
- inspect `mmgp` pinning and staged prefetch options before adding custom kernels
- keep weight-only INT4 as a fit-first branch if it unlocks more residency on 24 GB cards
