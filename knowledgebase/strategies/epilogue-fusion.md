---
id: epilogue_fusion
kind: strategy
title: Fuse Epilogues And Adjacent Pointwise Work
type: ""
category: kernel
summary: Operator fusion is still one of the fastest ways to cut HBM traffic on transformer blocks.
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
  - triton-tutorials
  - rightnow-qwen35-triton
  - rightnow-tile
workloads:
  - prefill
  - decode
  - serving
operators:
  - matmul
  - norm
  - rope
  - activation
gpu_families: []
gpu_ids: []
precision:
  - bf16
  - fp16
  - fp8
  - int8
  - int4
bottlenecks:
  - memory
  - mixed
goals:
  - throughput
  - latency
  - memory
priority: 84
preconditions:
  - the fused region is performance-critical and numerically stable under fusion
actions:
  - fuse bias, activation, norm, or rope steps when they currently materialize full intermediates
  - keep fusion decisions aligned with the downstream runtime so kernels remain callable
  - verify the fused kernel numerically against the reference path
metrics:
  - kernel count
  - HBM traffic
  - end-to-end latency
tradeoffs:
  - larger kernels can become harder to tune and maintain
  - fusion can complicate framework integration
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

- fuse bias, activation, norm, or rope steps when they currently materialize full intermediates
- keep fusion decisions aligned with the downstream runtime so kernels remain callable
- verify the fused kernel numerically against the reference path

## Tradeoffs

- larger kernels can become harder to tune and maintain
- fusion can complicate framework integration

## Metrics

- kernel count
- HBM traffic
- end-to-end latency
