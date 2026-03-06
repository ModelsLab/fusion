---
id: torch_compile_reduce_overhead_small_batch
kind: strategy
title: Use torch.compile Reduce-Overhead For Small Batch Serving
type: ""
category: runtime
summary: For small-batch serving and iterative decode, reduce-overhead is often the first torch.compile mode worth validating.
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
  - pytorch-torch-compile
  - pytorch-cuda-graphs
workloads:
  - decode
  - serving
operators:
  - attention
  - matmul
  - layernorm
  - rmsnorm
gpu_families: []
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
  - the workload has reasonably stable shapes
  - the runtime allows graph-style compile and warmup
actions:
  - benchmark eager vs torch.compile(mode=reduce-overhead)
  - compare decode latency with and without graph capture
  - only retain the mode if warmup cost is acceptable for the deployment shape mix
metrics:
  - inter-token latency
  - steady-state tokens/sec
  - warmup time
tradeoffs:
  - dynamic shapes can reduce wins
  - compile warmup can hurt short-lived jobs
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

- benchmark eager vs torch.compile(mode=reduce-overhead)
- compare decode latency with and without graph capture
- only retain the mode if warmup cost is acceptable for the deployment shape mix

## Tradeoffs

- dynamic shapes can reduce wins
- compile warmup can hurt short-lived jobs

## Metrics

- inter-token latency
- steady-state tokens/sec
- warmup time
