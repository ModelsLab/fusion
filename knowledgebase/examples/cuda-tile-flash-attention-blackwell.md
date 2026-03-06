---
id: cuda_tile_flash_attention_blackwell
kind: example
title: Blackwell CUDA Tile FlashAttention Reference
type: ""
category: attention-template
summary: Official Blackwell-oriented flash-attention tuning material for CUDA Tile and cuTile kernels.
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
  - nvidia-cuda-tile-flash-attention
  - nvidia-blackwell-cutlass
  - rightnow-tile
workloads:
  - prefill
  - decode
  - serving
operators:
  - attention
  - paged-attention
gpu_families:
  - Blackwell
gpu_ids:
  - b200
  - rtx5090
  - rtxpro6000blackwellserver
  - rtxpro6000blackwellworkstation
precision:
  - bf16
  - fp16
  - fp8
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
backend: cuTile
runtimes:
  - vllm
  - tensorrt-llm
  - sglang
use_cases:
  - compute
  - memory
  - blackwell-attention
notes:
  - Use it as a tuning and kernel-shape reference, not as a drop-in production recipe.
  - Best paired with Nsight-guided attention diagnosis and correctness gates.
reference_paths:
  - blog:tuning-flash-attention-for-peak-performance-in-nvidia-cuda-tile
backends: []
path: ""
---

## Use Cases

- compute
- memory
- blackwell-attention

## Notes

- Use it as a tuning and kernel-shape reference, not as a drop-in production recipe.
- Best paired with Nsight-guided attention diagnosis and correctness gates.

## Reference Paths

- blog:tuning-flash-attention-for-peak-performance-in-nvidia-cuda-tile
