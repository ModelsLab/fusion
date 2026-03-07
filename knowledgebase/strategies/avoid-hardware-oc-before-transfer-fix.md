---
id: avoid_hardware_oc_before_transfer_fix
kind: strategy
title: Avoid Hardware Clock Tuning Before Fixing Transfer Bottlenecks
category: hardware-tuning
summary: When profiles show host-device transfer and staging dominate the hot path, power or clock tuning should be deferred until residency and movement issues are reduced.
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
priority: 70
preconditions:
  - baseline benchmark exists
  - profile shows heavy host-device transfer or synchronization cost
actions:
  - keep default GPU boost behavior unless profiling proves a compute-bound steady-state phase
  - try residency, offload policy, quantization, and transfer overlap before clock tuning
  - only test clock locks after the path is known to be compute-bound or close to power-limited
tradeoffs:
  - can leave a small hardware-only win unexplored until later
  - reduces time spent on low-leverage tuning branches
metrics:
  - cuda_memcpy_api_pct
  - h2d_memcpy_time_pct
  - gen_s
  - clips_per_hour
source_ids:
  - flexgen-offload-compression
  - deepspeed-zero-inference
---

## Why

- clock tuning does not reduce PCIe or host staging costs
- if the runtime is waiting on movement, higher SM clocks usually add little or nothing
- the highest-value branch is to expose a compute-bound steady-state path first
