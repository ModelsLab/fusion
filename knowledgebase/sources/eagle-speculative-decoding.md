---
id: eagle-speculative-decoding
kind: source
title: "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"
category: paper
url: https://arxiv.org/abs/2401.15077
summary: Training-free-ish speculative decoding that uses a lightweight draft head trained on the target model's hidden states, achieving 2-3x speedup with lossless generation quality via tree-structured verification.
tags:
  - speculative-decoding
  - eagle
  - draft-model
  - tree-attention
  - lossless
source_ids: []
operators:
  - attention
  - sampling
gpu_families:
  - Ampere
  - Ada
  - Hopper
key_contributions:
  - Autoregressively predicts next-token features (hidden states) instead of tokens
  - Lightweight draft head (single transformer layer) trained on target model features
  - Tree-structured speculative sampling for higher acceptance rates
  - EAGLE-2 adds confidence-aware dynamic draft tree construction
  - EAGLE-3 extends to multi-model serving with shared draft
  - 2-3x speedup (EAGLE-1), 3-6.5x (EAGLE-2/3) on various LLMs
github: https://github.com/SafeAILab/EAGLE
---

## Usage with vLLM
```bash
# EAGLE speculative decoding in vLLM
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --speculative-model yuhuili/EAGLE-LLaMA3-Instruct-8B \
  --speculative-draft-tensor-parallel-size 1 \
  --num-speculative-tokens 5
```
