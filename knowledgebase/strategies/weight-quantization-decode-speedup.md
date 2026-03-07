---
id: weight_quantization_decode_speedup
kind: strategy
title: Weight Quantization For Decode Speedup
category: quantization
summary: Quantize model weights to INT4 or FP8 to reduce memory reads during decode, achieving 2-4x speedup on memory-bound token generation.
support_level: stable
source_ids:
  - awq-activation-aware-weight-quantization
workloads:
  - decode
  - serving
operators:
  - matmul
  - gemm
gpu_families:
  - Ampere
  - Ada
  - Hopper
  - Blackwell
precision:
  - fp8
  - int8
  - int4
  - fp4
bottlenecks:
  - memory
goals:
  - throughput
  - latency
priority: 95
preconditions:
  - decode GEMM is memory-bandwidth bound (M is small)
  - quantization quality is acceptable for use case
actions:
  - evaluate AWQ INT4 (g128) as first choice for consumer GPUs
  - evaluate FP8 as first choice for GPUs with native FP8 tensor cores (Ada, Hopper, Blackwell)
  - benchmark with Marlin kernel for INT4 (fastest W4A16 GEMM)
  - measure perplexity degradation to ensure acceptable quality
  - use FP4 on Blackwell for maximum throughput
metrics:
  - decode tokens per second
  - perplexity change vs baseline
  - memory bandwidth utilization
tradeoffs:
  - INT4 has more quality loss than FP8 but more speedup
  - FP8 is near-lossless but only 2x speedup vs FP16
  - group_size affects quality vs speed (smaller groups = better quality, more overhead)
---

## Actions
- evaluate AWQ INT4 or FP8 for Ada consumer GPUs, FP8 for Hopper/Blackwell datacenter
- benchmark with specialized kernels (Marlin for INT4, cuBLAS for FP8)
- measure perplexity to validate quality

## Decision Matrix
| GPU | Best Quantization | Expected Speedup |
|-----|------------------|-----------------|
| RTX 3090 (Ampere) | AWQ INT4 (g128) | 3-4x decode |
| RTX 4090 (Ada) | AWQ INT4 or FP8 | 2-4x decode |
| A100 (Ampere) | AWQ INT4 | 2-4x decode |
| H100 | FP8 (native) | 2x decode |
| B200 | FP4 (native) | 4x decode |
