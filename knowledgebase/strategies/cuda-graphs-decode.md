---
id: cuda_graphs_decode
kind: strategy
title: Capture Steady-State Decode With CUDA Graphs
type: ""
category: runtime
summary: Steady-state decode often benefits from graph capture once shapes and buffers are stabilized.
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
  - pytorch-cuda-graphs
  - flashinfer-docs
workloads:
  - decode
  - serving
operators:
  - attention
  - kv-cache
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
bottlenecks:
  - latency
  - mixed
goals:
  - latency
  - throughput
priority: 82
preconditions:
  - runtime shapes can be bucketed or stabilized
  - memory addresses are reusable across decode steps
actions:
  - warm up and capture the repeatable decode subgraph after allocations settle
  - avoid dynamic allocations and shape churn inside the captured step
  - compare graph replay latency against the eager decode baseline
metrics:
  - inter-token latency
  - kernel launch count
  - tokens/sec
tradeoffs:
  - dynamic shapes reduce graph capture coverage
  - debugging captured graphs is less ergonomic than eager execution
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

- warm up and capture the repeatable decode subgraph after allocations settle
- avoid dynamic allocations and shape churn inside the captured step
- compare graph replay latency against the eager decode baseline

## Tradeoffs

- dynamic shapes reduce graph capture coverage
- debugging captured graphs is less ergonomic than eager execution

## Metrics

- inter-token latency
- kernel launch count
- tokens/sec
