---
id: triton_for_memory_bound_fusions
kind: strategy
title: Use Triton First For Memory-Bound Fusions
type: ""
category: kernel
summary: Triton should usually be the first backend for fused norms, dequant, softmax, and layout-heavy memory-bound kernels.
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
  - flashinfer-docs
  - rightnow-qwen35-triton
workloads:
  - prefill
  - decode
  - serving
operators:
  - layernorm
  - rmsnorm
  - softmax
  - dequantization
  - kv-cache
  - attention
gpu_families:
  - Ampere
  - Ada
  - Hopper
  - Blackwell
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
priority: 90
preconditions:
  - the target operator is fusion-friendly rather than an opaque runtime black box
actions:
  - prototype the fused kernel in Triton before dropping to CUDA C++
  - benchmark Triton against runtime-native kernels and CuTe only if the operator stays hot
  - keep separate variants for prefill and decode if access patterns differ
metrics:
  - kernel latency
  - HBM throughput
  - occupancy
tradeoffs:
  - final peak performance may still require CuTe or CUDA C++
  - runtime packaging can be rougher than a pure C++ path
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

- prototype the fused kernel in Triton before dropping to CUDA C++
- benchmark Triton against runtime-native kernels and CuTe only if the operator stays hot
- keep separate variants for prefill and decode if access patterns differ

## Tradeoffs

- final peak performance may still require CuTe or CUDA C++
- runtime packaging can be rougher than a pure C++ path

## Metrics

- kernel latency
- HBM throughput
- occupancy
