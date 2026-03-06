---
id: fp8_transformer_engine
kind: strategy
title: Use FP8 Paths First On Hopper And Blackwell
type: ""
category: precision
summary: Hopper and Blackwell should usually validate mature FP8 inference paths before custom kernel work.
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
  - nvidia-h100
  - nvidia-blackwell-architecture
  - tensorrt-llm-docs
  - nvidia-tensorrt-model-optimizer
  - llm-compressor
  - nvidia-transformer-engine
  - torchao
workloads:
  - prefill
  - decode
  - serving
operators:
  - matmul
  - gemm
  - attention
gpu_families:
  - Hopper
  - Blackwell
gpu_ids:
  - h100
  - b200
precision:
  - fp8
bottlenecks:
  - compute
  - memory
  - mixed
goals:
  - throughput
  - memory
  - cost
priority: 94
preconditions:
  - the model tolerates FP8 calibration and validation
  - runtime stack supports the chosen FP8 serving path
actions:
  - benchmark a packaged FP8 path in TensorRT-LLM, vLLM, SGLang, or an equivalent mature serving runtime first when one already exists
  - if no packaged FP8 checkpoint exists, synthesize one with NVIDIA Model Optimizer, Transformer Engine, torchao float8 flows, or llm-compressor when the runtime and GPU support it
  - run calibration, save the quantization recipe, and validate output quality before spending time on custom kernels
  - validate accuracy before investing in more custom kernel work
  - only hand-write kernels after the FP8 baseline is known
metrics:
  - tokens/sec
  - quality drift
  - tensor core utilization
tradeoffs:
  - calibration quality matters
  - some model components may need fallback higher-precision paths
  - FP8 conversion can improve throughput and memory use, but unsupported operators or unstable layers may force mixed-precision fallbacks
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

- benchmark a packaged FP8 path in TensorRT-LLM, vLLM, SGLang, or an equivalent mature serving runtime first when one already exists
- if no packaged FP8 checkpoint exists, synthesize one with NVIDIA Model Optimizer, Transformer Engine, torchao float8 flows, or llm-compressor when the runtime and GPU support it
- run calibration, save the quantization recipe, and validate output quality before spending time on custom kernels
- validate accuracy before investing in more custom kernel work
- only hand-write kernels after the FP8 baseline is known

## Tradeoffs

- calibration quality matters
- some model components may need fallback higher-precision paths
- FP8 conversion can improve throughput and memory use, but unsupported operators or unstable layers may force mixed-precision fallbacks

## Metrics

- tokens/sec
- quality drift
- tensor core utilization
