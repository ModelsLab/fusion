---
id: flex-attention
kind: source
title: FlexAttention - PyTorch Composable Attention API
type: official-doc
category: attention
summary: PyTorch's API for defining custom attention patterns (causal, sliding window, document masking) that compile to efficient fused kernels via block masks and score modifications.
reliability: official
review_status: reviewed
url: https://pytorch.org/blog/flexattention/
tags:
  - flex-attention
  - pytorch
  - attention
  - block-mask
  - score-mod
  - compiled
---

## Key Features
- create_block_mask() for defining attention patterns
- score_mod for custom attention score modifications
- Compiles to fused Triton kernels (no mask materialization)
- Block-sparse computation (skips fully-masked blocks)
- Supports: causal, sliding window, prefix-LM, document masking, ALiBi, custom patterns
