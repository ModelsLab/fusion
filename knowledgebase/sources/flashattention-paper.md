---
id: flashattention-paper
kind: source
title: FlashAttention Papers (v1, v2, v3)
type: paper
category: attention
summary: The FlashAttention paper series by Tri Dao introducing IO-aware exact attention with tiling, online softmax, and Hopper-specific optimizations.
reliability: authoritative
review_status: reviewed
url: https://arxiv.org/abs/2205.14135
tags:
  - flash-attention
  - attention
  - tiling
  - io-aware
  - hopper
  - fp8
---

## Papers
- FlashAttention v1 (2022): IO-aware attention with tiling, O(N) memory
- FlashAttention-2 (2023): Better work partitioning, 2x faster than v1
- FlashAttention-3 (2024): Hopper-specific with warp specialization, TMA, FP8, intra-warp pipelining

## Key Insight
Standard attention materializes O(N^2) attention matrix. FlashAttention tiles the computation and uses online softmax to never materialize the full matrix, reducing memory to O(N) and achieving near-optimal IO complexity.
