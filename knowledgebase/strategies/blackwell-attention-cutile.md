---
id: blackwell_attention_cutile
kind: strategy
title: Use CUDA Tile Or CuTe First For Blackwell Attention
type: ""
category: attention
summary: Blackwell attention tuning should start from CUDA Tile or CuTe-based paths before generic CUDA rewrites.
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
  - fp4
  - nvfp4
bottlenecks:
  - compute
  - memory
  - mixed
goals:
  - throughput
  - latency
priority: 96
preconditions:
  - the attention workload is actually hot in Nsight or benchmark data
  - the runtime can tolerate Blackwell-specific kernel specialization
actions:
  - establish a bf16 or fp8 baseline before introducing CUDA Tile variants
  - try CUDA Tile or cuTile tuning before writing raw CUDA from scratch
  - validate attention-specific changes with numerical tolerance checks and real decode benchmarks
metrics:
  - tokens/sec
  - TTFT
  - inter-token latency
  - tensor utilization
tradeoffs:
  - portability to older GPUs is limited
  - tooling and public examples are newer than Triton-era flows
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

- establish a bf16 or fp8 baseline before introducing CUDA Tile variants
- try CUDA Tile or cuTile tuning before writing raw CUDA from scratch
- validate attention-specific changes with numerical tolerance checks and real decode benchmarks

## Tradeoffs

- portability to older GPUs is limited
- tooling and public examples are newer than Triton-era flows

## Metrics

- tokens/sec
- TTFT
- inter-token latency
- tensor utilization
