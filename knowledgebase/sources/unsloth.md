---
id: unsloth
kind: source
title: Unsloth - Fast LLM Fine-Tuning
type: code
category: training
summary: Unsloth provides 2x+ speedup for LLM fine-tuning through custom Triton kernels, memory optimization, and efficient QLoRA implementation.
reliability: community
review_status: reviewed
url: https://github.com/unslothai/unsloth
tags:
  - unsloth
  - fine-tuning
  - qlora
  - triton
  - memory-efficient
---

## Key Optimizations
- Custom Triton kernels for RoPE, cross-entropy, RMSNorm
- Efficient QLoRA with fused dequant
- Memory-optimized backward pass
- 2x faster than standard HuggingFace training
- 60% less memory usage
