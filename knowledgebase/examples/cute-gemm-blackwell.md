---
id: cute_gemm_blackwell
kind: example
title: CuTe DSL GEMM And Tensor-Core Kernel Skeleton
type: ""
category: kernel-skeleton
summary: CuTe DSL reference path for tensor-core-heavy GEMM kernels across Ampere, Hopper, and Blackwell.
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
gpu_families:
  - Ampere
  - Hopper
  - Blackwell
gpu_ids:
  - a100
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
backend: cute_dsl
runtimes:
  - vllm
  - tensorrt-llm
  - transformers
  - sglang
use_cases:
  - compute
  - tensor-core-hotpath
notes:
  - Prefer this when Triton prototypes are no longer enough and the hot path is tensor-core dominated.
  - Maps well to CUTLASS-grade tiling and AOT packaging.
reference_paths:
  - docs:cutlass/pythonDSL/cute_dsl
  - docs:cutlass/cpp/blackwell_functionality
backends: []
path: ""
---

## Use Cases

- compute
- tensor-core-hotpath

## Notes

- Prefer this when Triton prototypes are no longer enough and the hot path is tensor-core dominated.
- Maps well to CUTLASS-grade tiling and AOT packaging.

## Reference Paths

- docs:cutlass/pythonDSL/cute_dsl
- docs:cutlass/cpp/blackwell_functionality
