---
id: rtx6000ada
kind: gpu
title: NVIDIA RTX 6000 Ada Generation
type: ""
category: ""
summary: ""
support_level: ""
reliability: ""
review_status: ""
url: ""
tags: []
aliases:
  - rtx 6000 ada
  - rtx6000 ada
  - nvidia rtx 6000 ada
  - rtx 6000 ada generation
family: Ada
market: workstation
compute_capability: "8.9"
memory_gb: 48
memory_bandwidth_gbps: 960
preferred_precisions:
  - fp16
  - bf16
  - fp8
  - int8
  - int4
experimental_precisions: []
strengths:
  - 48 GB Ada workstation card for larger local inference workloads
  - strong fit for Triton development when you need more memory than a 4090
constraints:
  - workstation deployment path is less representative than passive server cards
  - still far below Hopper HBM bandwidth on memory-bound decode
source_ids:
  - nvidia-cuda-gpus
  - nvidia-rtx-6000-ada
  - nvidia-rtx-blackwell-pro-architecture
workloads: []
operators: []
gpu_families: []
gpu_ids: []
precision: []
bottlenecks: []
goals: []
priority: 0
preconditions: []
actions: []
metrics: []
tradeoffs: []
preferred_backends: []
required_tools: []
steps: []
verification: []
benchmark_rubric: []
failure_recovery: []
artifacts_to_save: []
runtime_adapters: []
reference_source_ids: []
backend: ""
runtimes: []
use_cases: []
notes: []
reference_paths: []
backends: []
path: ""
---

## Strengths

- 48 GB Ada workstation card for larger local inference workloads
- strong fit for Triton development when you need more memory than a 4090

## Constraints

- workstation deployment path is less representative than passive server cards
- still far below Hopper HBM bandwidth on memory-bound decode

## Precisions

- fp16
- bf16
- fp8
- int8
- int4
