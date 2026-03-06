---
id: fused_dequant_gemm
kind: strategy
title: Fuse Dequantization Into GEMM For Quantized Paths
type: ""
category: kernel
summary: Quantized inference often wins by fusing dequant work directly into GEMM instead of materializing higher-precision weights.
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
  - triton-tutorials
workloads:
  - prefill
  - decode
  - serving
operators:
  - matmul
  - gemm
  - quantization
gpu_families:
  - Ampere
  - Ada
  - Hopper
  - Blackwell
gpu_ids: []
precision:
  - int4
  - int8
bottlenecks:
  - memory
  - mixed
goals:
  - throughput
  - memory
  - cost
priority: 88
preconditions:
  - weights are quantized or can be quantized without violating quality constraints
actions:
  - benchmark fused dequant GEMM paths before materializing fp16 or bf16 weights
  - keep epilogue fusion close to the GEMM when activations remain bandwidth-sensitive
  - compare decode and prefill independently because the best quantization path may differ
metrics:
  - GEMM time
  - HBM read volume
  - tokens/sec
tradeoffs:
  - calibration and accuracy checks are mandatory
  - kernel variants multiply once group size and quant format vary
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

- benchmark fused dequant GEMM paths before materializing fp16 or bf16 weights
- keep epilogue fusion close to the GEMM when activations remain bandwidth-sensitive
- compare decode and prefill independently because the best quantization path may differ

## Tradeoffs

- calibration and accuracy checks are mandatory
- kernel variants multiply once group size and quant format vary

## Metrics

- GEMM time
- HBM read volume
- tokens/sec
