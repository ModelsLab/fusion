---
id: moe_grouped_gemm
kind: strategy
title: Use Grouped GEMM For MoE Hot Paths
type: ""
category: kernel
summary: MoE experts usually need grouped GEMM or equivalent batching to keep tensor cores busy.
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
  - tensorrt-llm-docs
workloads:
  - prefill
  - decode
  - serving
operators:
  - moe
  - matmul
  - gemm
gpu_families: []
gpu_ids: []
precision:
  - bf16
  - fp16
  - fp8
  - int8
  - int4
bottlenecks:
  - compute
  - mixed
goals:
  - throughput
  - latency
priority: 80
preconditions:
  - the model has expert routing or sparse expert activation
actions:
  - replace many tiny per-expert GEMMs with grouped or batched GEMM paths
  - tune the schedule separately for prefill and decode because token routing sparsity changes
metrics:
  - expert kernel time
  - tensor core utilization
  - tokens/sec
tradeoffs:
  - routing imbalance can still dominate
  - kernel schedules become more model-specific
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

- replace many tiny per-expert GEMMs with grouped or batched GEMM paths
- tune the schedule separately for prefill and decode because token routing sparsity changes

## Tradeoffs

- routing imbalance can still dominate
- kernel schedules become more model-specific

## Metrics

- expert kernel time
- tensor core utilization
- tokens/sec
