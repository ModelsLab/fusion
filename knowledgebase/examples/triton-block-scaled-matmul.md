---
id: triton_block_scaled_matmul
kind: example
title: Triton Block-Scaled Matmul Reference
type: ""
category: kernel-skeleton
summary: Reference path for block-scaled FP4 or FP8-style GEMMs on newer NVIDIA GPUs.
support_level: experimental
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
  - triton-block-scaled-matmul
  - nvidia-blackwell-cutlass
workloads:
  - prefill
  - serving
operators:
  - matmul
  - gemm
gpu_families:
  - Blackwell
gpu_ids:
  - b200
  - rtx5090
precision:
  - fp8
  - fp4
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
backend: triton
runtimes:
  - tensorrt-llm
  - transformers
use_cases:
  - compute
  - memory
  - low-precision
notes:
  - Use FP8 as the control before testing block-scaled FP4 or NVFP4 variants.
  - Treat accuracy validation as a first-class gate.
reference_paths:
  - tutorials:triton/block-scaled-matmul
backends: []
path: ""
---

## Use Cases

- compute
- memory
- low-precision

## Notes

- Use FP8 as the control before testing block-scaled FP4 or NVFP4 variants.
- Treat accuracy validation as a first-class gate.

## Reference Paths

- tutorials:triton/block-scaled-matmul
