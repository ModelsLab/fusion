---
id: torch_compile_max_autotune_prefill
kind: strategy
title: Use torch.compile Max-Autotune For Prefill Sweeps
type: ""
category: runtime
summary: Compute-heavy prefill often justifies max-autotune or max-autotune-no-cudagraphs sweeps before hand-written kernels.
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
  - pytorch-attention-docs
workloads:
  - prefill
operators:
  - matmul
  - attention
  - moe
gpu_families: []
gpu_ids: []
precision:
  - bf16
  - fp16
  - fp8
bottlenecks:
  - compute
  - mixed
goals:
  - throughput
priority: 80
preconditions:
  - the benchmark is long enough to amortize compile time
  - the model stack uses supported PyTorch graph regions
actions:
  - benchmark default vs max-autotune vs max-autotune-no-cudagraphs
  - keep the best compile mode as the control before kernel generation work
  - record compile time and steady-state throughput separately
metrics:
  - prefill tokens/sec
  - compile duration
  - tensor core utilization
tradeoffs:
  - compile cost can be large
  - unsupported graph regions may limit wins
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

- benchmark default vs max-autotune vs max-autotune-no-cudagraphs
- keep the best compile mode as the control before kernel generation work
- record compile time and steady-state throughput separately

## Tradeoffs

- compile cost can be large
- unsupported graph regions may limit wins

## Metrics

- prefill tokens/sec
- compile duration
- tensor core utilization
