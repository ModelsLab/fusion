---
id: rtxpro6000blackwellworkstation
kind: gpu
title: NVIDIA RTX PRO 6000 Blackwell Workstation Edition
type: ""
category: ""
summary: ""
support_level: ""
reliability: ""
review_status: ""
url: ""
tags: []
aliases:
  - rtx pro 6000 blackwell
  - 6000 pro blackwell
  - rtx pro 6000 blackwell workstation
  - nvidia rtx pro 6000 blackwell workstation edition
family: Blackwell
market: workstation
compute_capability: "12.0"
memory_gb: 96
memory_bandwidth_gbps: 1792
preferred_precisions:
  - fp16
  - bf16
  - fp8
  - int8
  - int4
experimental_precisions:
  - fp4
  - nvfp4
strengths:
  - 96 GB workstation Blackwell target is excellent for local large-model iteration
  - bandwidth and VRAM make it a strong bridge between consumer and datacenter deployment
constraints:
  - workstation cooling and deployment assumptions differ from server cards
  - Blackwell low-precision serving stacks still need validation per framework
source_ids:
  - nvidia-cuda-gpus
  - nvidia-rtx-pro-6000-blackwell-workstation
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

- 96 GB workstation Blackwell target is excellent for local large-model iteration
- bandwidth and VRAM make it a strong bridge between consumer and datacenter deployment

## Constraints

- workstation cooling and deployment assumptions differ from server cards
- Blackwell low-precision serving stacks still need validation per framework

## Precisions

- fp16
- bf16
- fp8
- int8
- int4
- fp4
- nvfp4
