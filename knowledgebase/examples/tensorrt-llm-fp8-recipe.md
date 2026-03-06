---
id: tensorrt_llm_fp8_recipe
kind: example
title: TensorRT-LLM FP8 And Quantization Recipe
type: ""
category: runtime-patch
summary: Reference integration path for mature FP8 or weight-quantized deployment in TensorRT-LLM.
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
  - tensorrt-llm-docs
  - nvidia-transformer-engine
  - nvidia-blackwell-cutlass
workloads:
  - prefill
  - decode
  - serving
operators:
  - matmul
  - attention
gpu_families:
  - Hopper
  - Blackwell
gpu_ids:
  - h100
  - h200
  - b200
precision:
  - fp8
  - int4
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
backend: runtime
runtimes:
  - tensorrt-llm
use_cases:
  - runtime-integration
  - precision
  - serving
notes:
  - Use it as the mature control path before custom tensor-core kernel work.
  - Pair it with quality checks and tokens-per-second benchmarking.
reference_paths:
  - docs:tensorrt-llm
backends: []
path: ""
---

## Use Cases

- runtime-integration
- precision
- serving

## Notes

- Use it as the mature control path before custom tensor-core kernel work.
- Pair it with quality checks and tokens-per-second benchmarking.

## Reference Paths

- docs:tensorrt-llm
