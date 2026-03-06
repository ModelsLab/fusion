---
id: transformers_torch_compile_recipe
kind: example
title: Transformers torch.compile Sweep Recipe
type: ""
category: runtime-patch
summary: Reference path for eager vs torch.compile sweeps in a raw Transformers stack.
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
  - prefill
  - decode
  - serving
operators:
  - attention
  - matmul
  - layernorm
  - rmsnorm
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
backend: torch_compile
runtimes:
  - transformers
use_cases:
  - runtime-integration
  - latency
  - throughput
notes:
  - Use it as a model-level baseline branch before custom kernel generation.
  - Benchmark reduce-overhead and max-autotune separately because they optimize different workload shapes.
reference_paths:
  - docs:torch.compile
backends: []
path: ""
---

## Use Cases

- runtime-integration
- latency
- throughput

## Notes

- Use it as a model-level baseline branch before custom kernel generation.
- Benchmark reduce-overhead and max-autotune separately because they optimize different workload shapes.

## Reference Paths

- docs:torch.compile
