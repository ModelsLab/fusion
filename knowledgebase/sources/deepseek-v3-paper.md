---
id: deepseek-v3-paper
kind: source
title: "DeepSeek-V3: Scaling Mixture-of-Experts with Multi-Head Latent Attention"
category: paper
url: https://arxiv.org/abs/2412.19437
summary: DeepSeek-V3 architecture paper detailing MLA (Multi-Head Latent Attention) for KV cache compression, auxiliary-loss-free load balancing for MoE, FP8 mixed-precision training, and multi-token prediction.
tags:
  - deepseek
  - mla
  - moe
  - fp8-training
  - multi-token-prediction
  - load-balancing
source_ids: []
operators:
  - attention
  - matmul
gpu_families:
  - Hopper
key_contributions:
  - MLA reduces KV cache by 64x vs standard MHA through learned compression
  - Auxiliary-loss-free expert load balancing using per-expert bias terms
  - FP8 training at scale (671B parameters) with minimal quality loss
  - Multi-token prediction for faster inference via speculative-style generation
  - 2048 H800 GPUs, 2.788M GPU hours total training cost
---

## Key Technical Details

### Multi-Head Latent Attention (MLA)
- Compresses KV into low-dimensional latent vectors (d_compress << n_heads * d_head)
- Absorbs decompression matrices into query projection for inference efficiency
- KV cache per token: d_compress + d_rope (vs 2 * n_heads * d_head for standard)

### Auxiliary-Loss-Free Load Balancing
- Traditional: add auxiliary loss to encourage balanced expert selection (degrades quality)
- DeepSeek: per-expert bias added to routing scores, adjusted dynamically
- Achieves balanced routing without quality loss from auxiliary objectives

### FP8 Training
- Forward: FP8 E4M3 for activations and weights in GEMM
- Backward: FP8 E5M2 for gradients
- 1-bit scaling factors per 128 elements (block-wise quantization)
- Attention computation remains in BF16 (quantization-sensitive)
