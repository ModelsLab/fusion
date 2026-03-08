---
id: ada_q8f_before_triton_fp8
kind: strategy
title: On Ada, Test Q8F Before Assuming Triton FP8 Is Best
category: backend-selection
summary: If a model runtime offers both Triton quantized inference and an Ada-oriented `q8f` backend, benchmark `q8f` early instead of assuming Triton FP8 will be the fastest option.
support_level: recommended
reliability: curated
review_status: reviewed
tags:
  - ada
  - q8f
  - fp8
  - triton
  - backend-selection
workloads:
  - image-generation
  - video-generation
  - multimodal
operators:
  - attention
  - matmul
  - quantization
gpu_families:
  - Ada
gpu_ids:
  - rtx4090
  - l40
  - l40s
  - rtx6000ada
precision:
  - int8
  - fp8
bottlenecks:
  - throughput
  - mixed
goals:
  - latency
  - throughput
priority: 92
preconditions:
  - the runtime exposes both Triton and `q8f` quantized inference paths
  - quantized checkpoints are available for the target model
actions:
  - benchmark `int8-q8f`, `fp8-q8f`, `int8-triton`, and `fp8-triton` under the same prompt, seed, frame count, and step count
  - rank kernel families before concluding that FP8 support on the GPU makes Triton FP8 the right branch
  - record load time separately from steady-state generation time because backend choice can invert the cold-start ranking
metrics:
  - gen_s
  - video_frames_per_sec
  - init_s
  - peak_alloc_gb
tradeoffs:
  - q8f may require building extra CUDA extensions instead of using a pure Python Triton stack
  - FP8 can still lose to INT8 if the backend family and fused kernels are not the best match for the workload
source_ids:
  - lightx2v-quantization-q8f
path: strategies/ada-q8f-before-triton-fp8.md
---

## Summary

Ada-class hardware having FP8 tensor cores is not enough to declare a Triton FP8 runtime the best option. If the runtime has an Ada-specialized backend such as `q8f`, benchmark that family early and compare both INT8 and FP8 inside it.
