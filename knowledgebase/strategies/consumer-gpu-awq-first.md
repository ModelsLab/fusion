---
id: consumer_gpu_awq_first
kind: strategy
title: Try AWQ Or INT4 Early On Consumer GPUs
type: ""
category: precision
summary: Consumer Ada and Blackwell cards usually benefit from weight-only INT4 or AWQ before deeper kernel work.
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
  - awq-activation-aware-weight-quantization
  - tensorrt-llm-docs
  - vllm-docs
workloads:
  - prefill
  - decode
  - serving
operators:
  - quantization
  - matmul
  - attention
gpu_families:
  - Ada
  - Blackwell
gpu_ids:
  - rtx4090
  - rtx5090
  - rtx6000ada
  - rtxpro6000blackwellworkstation
precision:
  - int4
  - awq-int4
  - bf16
  - fp16
bottlenecks:
  - memory
  - mixed
goals:
  - memory
  - cost
  - throughput
priority: 88
preconditions:
  - the model fits a supported quantization flow
  - a quality gate exists for the target model family
actions:
  - benchmark an AWQ or equivalent INT4 track before building new high-byte kernels
  - prefer fused dequant plus GEMM paths instead of standalone dequant stages
  - treat prefill and decode as separate benchmark tracks
metrics:
  - tokens/sec
  - VRAM footprint
  - quality drift
tradeoffs:
  - quality depends on calibration and runtime support
  - compute-heavy prefill can still require custom kernels after quantization
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

- benchmark an AWQ or equivalent INT4 track before building new high-byte kernels
- prefer fused dequant plus GEMM paths instead of standalone dequant stages
- treat prefill and decode as separate benchmark tracks

## Tradeoffs

- quality depends on calibration and runtime support
- compute-heavy prefill can still require custom kernels after quantization

## Metrics

- tokens/sec
- VRAM footprint
- quality drift
