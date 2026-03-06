---
id: block_scaled_low_precision
kind: strategy
title: Try Block-Scaled Low-Precision Kernels On Blackwell
type: ""
category: precision
summary: Blackwell makes block-scaled low-precision paths materially more attractive than on earlier generations.
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
  - nvidia-blackwell-architecture
workloads:
  - prefill
  - decode
operators:
  - matmul
  - gemm
gpu_families:
  - Blackwell
gpu_ids:
  - rtx5090
  - b200
precision:
  - fp8
  - fp6
  - fp4
  - nvfp4
bottlenecks:
  - compute
  - memory
goals:
  - throughput
  - cost
  - memory
priority: 83
preconditions:
  - toolchain and runtime support the targeted low-precision format
  - model quality is validated against the reduced precision path
actions:
  - start from Triton block-scaled examples instead of inventing a layout from scratch
  - benchmark against an FP8 baseline so the incremental value is measurable
  - treat correctness and calibration drift as first-class gates
metrics:
  - GEMM throughput
  - quality drift
  - memory footprint
tradeoffs:
  - tooling coverage is still evolving
  - portability to non-Blackwell stacks is limited
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

- start from Triton block-scaled examples instead of inventing a layout from scratch
- benchmark against an FP8 baseline so the incremental value is measurable
- treat correctness and calibration drift as first-class gates

## Tradeoffs

- tooling coverage is still evolving
- portability to non-Blackwell stacks is limited

## Metrics

- GEMM throughput
- quality drift
- memory footprint
