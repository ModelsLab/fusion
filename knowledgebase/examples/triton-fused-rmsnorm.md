---
id: triton_fused_rmsnorm
kind: example
title: Triton Fused Norm And Pointwise Fusion Pattern
type: ""
category: kernel-skeleton
summary: Canonical Triton path for RMSNorm, LayerNorm, softmax, and similar memory-bound fused operators.
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
workloads:
  - prefill
  - decode
  - serving
operators:
  - layernorm
  - rmsnorm
  - softmax
  - dequantization
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
  - transformers
  - vllm
  - sglang
use_cases:
  - memory
  - fusion
  - rapid-iteration
notes:
  - Strong first stop for bandwidth-heavy fusions before CUDA C++.
  - Keep separate decode and prefill variants if access patterns diverge.
reference_paths:
  - tutorials:triton/getting-started
backends: []
path: ""
---

## Use Cases

- memory
- fusion
- rapid-iteration

## Notes

- Strong first stop for bandwidth-heavy fusions before CUDA C++.
- Keep separate decode and prefill variants if access patterns diverge.

## Reference Paths

- tutorials:triton/getting-started
