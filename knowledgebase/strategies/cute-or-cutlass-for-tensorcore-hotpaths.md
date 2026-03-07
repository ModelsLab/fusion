---
id: cute_or_cutlass_for_tensorcore_hotpaths
kind: strategy
title: Use CuTe Or CUTLASS For Tensor-Core Hot Paths
type: ""
category: kernel
summary: When the bottleneck is a tensor-core-heavy GEMM, attention core, or grouped GEMM, CuTe or CUTLASS should be the primary backend.
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
  - nvidia-cute-dsl
  - nvidia-cutlass-overview
  - nvidia-blackwell-cutlass
workloads:
  - prefill
  - decode
  - serving
operators:
  - matmul
  - gemm
  - attention
  - moe
gpu_families:
  - Ampere
  - Ada
  - Hopper
  - Blackwell
gpu_ids:
  - a100
  - rtx4090
  - l40s
  - h100
  - h200
  - b200
precision:
  - bf16
  - fp16
  - fp8
  - int8
  - int4
  - nvfp4
bottlenecks:
  - compute
goals:
  - throughput
  - latency
priority: 92
preconditions:
  - profiling shows the hot path is actually tensor-core dominated
  - the team is willing to accept a slower iteration loop than Triton
actions:
  - start from CuTe DSL or CUTLASS templates instead of freehand CUDA
  - specialize launch and tile shapes per GPU family rather than forcing one universal kernel
  - keep correctness and performance gates after each generated variant
metrics:
  - tensor core utilization
  - GFLOP/s
  - kernel latency
tradeoffs:
  - template complexity is higher than Triton
  - authoring speed is slower for simple pointwise fusions
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

- start from CuTe DSL or CUTLASS templates instead of freehand CUDA
- specialize launch and tile shapes per GPU family rather than forcing one universal kernel
- keep correctness and performance gates after each generated variant

## Tradeoffs

- template complexity is higher than Triton
- authoring speed is slower for simple pointwise fusions

## Metrics

- tensor core utilization
- GFLOP/s
- kernel latency
