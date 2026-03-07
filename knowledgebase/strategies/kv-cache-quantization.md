---
id: kv_cache_quantization
kind: strategy
title: KV Cache Quantization For Memory And Bandwidth Savings
category: attention
summary: Quantize KV cache to FP8 or INT4 to reduce memory usage and attention bandwidth, enabling larger batch sizes and longer contexts.
support_level: stable
source_ids: []
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
precision:
  - fp8
  - int4
bottlenecks:
  - memory
goals:
  - throughput
  - memory-efficiency
priority: 80
preconditions:
  - KV cache is a significant fraction of GPU memory
  - batch sizes are limited by KV cache memory
actions:
  - start with FP8 KV cache (minimal quality loss, 2x memory savings)
  - evaluate INT4 KV cache if memory is very tight (4x savings but more quality loss)
  - measure attention accuracy and end-to-end quality
  - benchmark throughput improvement from larger batch sizes
metrics:
  - max batch size achievable
  - throughput improvement
  - quality degradation
tradeoffs:
  - FP8 KV is nearly lossless for most models
  - INT4 KV may require per-head calibration for best results
  - attention kernel must support quantized KV access
---

## Actions
- enable FP8 KV cache as default for GPUs with FP8 tensor cores (Ada, Hopper, Blackwell)
- measure quality impact on representative benchmarks
- calculate memory savings and resulting batch size increase

## Impact Example
LLaMA-70B, batch=64, seq_len=4096:
- FP16 KV: 80 GB → cannot fit with 70B FP8 weights on H100
- FP8 KV: 40 GB → fits with 70B FP8 weights on H100
- INT4 KV: 20 GB → room for even larger batches
