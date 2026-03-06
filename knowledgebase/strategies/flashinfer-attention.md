---
id: flashinfer_attention
kind: strategy
title: Prefer FlashInfer Or Similar Serving Attention Kernels
type: ""
category: kernel
summary: Serving-focused attention kernels usually beat generic framework paths once batch mixing and paged KV are involved.
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
  - flashinfer-docs
  - flashinfer-repo
  - rightnow-qwen35-triton
workloads:
  - decode
  - prefill
  - serving
operators:
  - attention
  - kv-cache
  - paged-attention
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
  - compute
  - mixed
goals:
  - throughput
  - latency
priority: 90
preconditions:
  - the model runtime can call a serving-optimized attention backend
  - sequence layout matches the backend kernel contracts
actions:
  - benchmark serving-focused attention backends before writing a custom kernel
  - compare prefill and decode separately because the winning backend may differ
  - keep paged KV layout and backend configuration aligned
metrics:
  - attention kernel time
  - tokens/sec
  - dram throughput
tradeoffs:
  - backend-specific layouts can reduce portability
  - operator coverage differs across serving stacks
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

- benchmark serving-focused attention backends before writing a custom kernel
- compare prefill and decode separately because the winning backend may differ
- keep paged KV layout and backend configuration aligned

## Tradeoffs

- backend-specific layouts can reduce portability
- operator coverage differs across serving stacks

## Metrics

- attention kernel time
- tokens/sec
- dram throughput
