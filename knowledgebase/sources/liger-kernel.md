---
id: liger-kernel
kind: source
title: Liger Kernel - Efficient Triton Kernels for LLM Training
type: code
category: kernel
summary: Collection of optimized Triton kernels for LLM training including fused cross-entropy, RMSNorm, RoPE, SwiGLU, and GeGLU with significant memory savings.
reliability: community
review_status: reviewed
url: https://github.com/linkedin/Liger-Kernel
tags:
  - liger
  - triton
  - fused-kernel
  - training
  - cross-entropy
  - rmsnorm
  - swiglu
---

## Key Kernels
- Fused Cross-Entropy + Softmax: 4x memory reduction (no materialization of logits)
- Fused RMSNorm: ~2x faster with residual fusion
- Fused SwiGLU: eliminates intermediate tensor
- Fused RoPE: applies rotation in-place
- Fused GeGLU: for models using GeGLU activation

## Integration
Drop-in replacement for HuggingFace Transformers layers.
Saves 20-60% GPU memory during training.
