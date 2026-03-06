---
id: speculative_decoding
kind: strategy
title: Speculative Decoding For Faster Token Generation
category: inference
summary: Use a draft model or self-speculation to generate multiple candidate tokens and verify them in parallel, achieving 2-4x decode speedup.
support_level: stable
source_ids: []
workloads:
  - decode
  - serving
operators:
  - attention
  - matmul
gpu_families:
  - Ampere
  - Ada
  - Hopper
  - Blackwell
precision:
  - fp16
  - bf16
  - fp8
  - int4
bottlenecks:
  - memory
goals:
  - latency
priority: 85
preconditions:
  - decode is the latency bottleneck (not prefill)
  - a suitable draft model or self-speculation method is available
  - acceptance rate is above 60% for meaningful speedup
actions:
  - profile decode latency to confirm memory-bound single-token generation is the bottleneck
  - select speculation method (draft model, Medusa, Eagle, LayerSkip)
  - tune number of speculative tokens (k=3-7 typical)
  - benchmark end-to-end to measure actual speedup including draft overhead
metrics:
  - acceptance rate
  - effective tokens per second
  - per-token latency (TPOT)
  - draft overhead as fraction of target forward time
tradeoffs:
  - draft model adds memory overhead for weights
  - acceptance rate drops with more speculative tokens
  - some methods require fine-tuning extra heads (Medusa, Eagle)
---

## Actions
- profile decode latency to confirm memory-bound single-token generation is the bottleneck
- select speculation method based on available resources and acceptable overhead
- tune number of speculative tokens (k=3-7 typical)
- benchmark end-to-end to measure actual speedup

## Expected Speedup
- Draft model (good match): 2-3x with k=5, 70-80% acceptance
- Medusa: 2-2.5x, requires training heads
- Eagle: 2.5-3.5x, best quality among head-based methods
- LayerSkip: 1.5-2x, no extra model needed
