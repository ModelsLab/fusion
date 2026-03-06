---
id: rtxpro6000blackwellserver
kind: gpu
title: NVIDIA RTX PRO 6000 Blackwell Server Edition
type: ""
category: ""
summary: ""
support_level: ""
reliability: ""
review_status: ""
url: ""
tags: []
aliases:
  - rtx pro 6000 blackwell server
  - rtx pro 6000 server
  - nvidia rtx pro 6000 blackwell server edition
family: Blackwell
market: datacenter
compute_capability: "12.0"
memory_gb: 96
memory_bandwidth_gbps: 1597
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
  - server-ready Blackwell professional card with large memory and strong low-precision throughput
  - useful for inference fleets that want Blackwell features outside the HGX class
constraints:
  - less bandwidth than B200-class HBM platforms for the heaviest memory-bound decode cases
  - serving software coverage is still maturing for newer Blackwell professional SKUs
source_ids:
  - nvidia-cuda-gpus
  - nvidia-rtx-pro-6000-blackwell-server
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

- server-ready Blackwell professional card with large memory and strong low-precision throughput
- useful for inference fleets that want Blackwell features outside the HGX class

## Constraints

- less bandwidth than B200-class HBM platforms for the heaviest memory-bound decode cases
- serving software coverage is still maturing for newer Blackwell professional SKUs

## Precisions

- fp16
- bf16
- fp8
- int8
- int4
- fp4
- nvfp4
