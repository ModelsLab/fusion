---
id: per_gpu_specialization
kind: strategy
title: Maintain Per-GPU Kernel Variants
type: ""
category: workflow
summary: Different GPUs deserve different tile sizes, launch parameters, and sometimes entirely different kernels.
support_level: stable
reliability: ""
review_status: ""
url: ""
tags: []
aliases: []
family: ""
market: ""
compute_capability: ""
memory_gb: 0
memory_bandwidth_gbps: 0
preferred_precisions: []
experimental_precisions: []
strengths: []
constraints: []
source_ids:
  - doublegraph
  - doubleai-warpspeed
  - rightnow-qwen35-triton
workloads:
  - all
operators:
  - all
gpu_families:
  - Ampere
  - Ada
  - Hopper
  - Blackwell
gpu_ids:
  - rtx3090
  - rtx4090
  - rtx5090
  - h100
  - b200
  - l4
precision:
  - all
bottlenecks:
  - all
goals:
  - throughput
  - latency
  - portability
priority: 86
preconditions:
  - benchmark infrastructure can compare variants on the target GPU
actions:
  - avoid assuming one kernel configuration is optimal across Ampere, Ada, Hopper, and Blackwell
  - store winning variants by GPU family and workload shape
  - treat hardware specialization as a normal outcome, not a last resort
metrics:
  - best-of-GPU throughput
  - latency variance by architecture
tradeoffs:
  - more variants mean more maintenance
  - benchmark coverage needs to grow with the matrix of GPUs and workloads
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

## Actions

- avoid assuming one kernel configuration is optimal across Ampere, Ada, Hopper, and Blackwell
- store winning variants by GPU family and workload shape
- treat hardware specialization as a normal outcome, not a last resort

## Tradeoffs

- more variants mean more maintenance
- benchmark coverage needs to grow with the matrix of GPUs and workloads

## Metrics

- best-of-GPU throughput
- latency variance by architecture
