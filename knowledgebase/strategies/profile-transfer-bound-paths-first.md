---
id: profile_transfer_bound_paths_first
kind: strategy
title: Profile Transfer-Bound Paths Before Kernel Search
category: profiling
summary: When profiling shows large host-device transfer or synchronization costs, prioritize residency, offload policy, and staging fixes before spending time on custom kernels.
support_level: stable
workloads:
  - text-generation
  - image-generation
  - image-editing
  - video-generation
  - audio-generation
  - multimodal
operators:
  - general
  - attention
  - transformer
  - unet
  - vae
gpu_families:
  - all
precision:
  - any
bottlenecks:
  - memory
  - mixed
goals:
  - latency
  - throughput
  - memory
priority: 94
preconditions:
  - baseline benchmark exists
  - profile shows heavy cudaMemcpyAsync, host-device transfer, or synchronization cost
actions:
  - separate load and warmup from steady-state profiling
  - reduce offload churn and keep the dominant modules resident when the workload fits
  - test pinned-buffer reuse or staged prefetch before writing custom kernels
  - use fit-first quantization if residency is blocked by VRAM
tradeoffs:
  - can increase VRAM residency and reduce flexibility on larger shapes
  - may expose a second compute bottleneck only after transfer pressure is removed
metrics:
  - cuda_memcpy_api_pct
  - h2d_memcpy_time_pct
  - steady_state_latency
  - clips_per_hour
source_ids:
  - huggingface-diffusers-group-offloading
  - flexgen-offload-compression
  - pytorch-torch-load-mmap
  - fal-ai-flashpack
---

## Actions

- separate one-time load behavior from steady-state generation before drawing conclusions
- if transfer dominates, prioritize residency and staged prefetch over kernel rewrites
- use `torch.load(..., mmap=True)` only on real `.pt` load paths, not as a generic safetensors optimization
- use FlashPack or similar checkpoint-packing flows only when startup latency matters; do not mistake them for fixes to a transfer-heavy steady-state trace

## Why

- transfer-bound traces often hide the real kernel bottleneck until residency is improved
- the first 50 percent of speedup often comes from moving less data rather than running the same kernels slightly faster
