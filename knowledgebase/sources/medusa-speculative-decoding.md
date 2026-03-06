---
id: medusa-speculative-decoding
kind: source
title: "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"
category: paper
url: https://arxiv.org/abs/2401.10774
summary: Adds multiple lightweight prediction heads to an LLM to propose several future tokens simultaneously, verified via tree attention. Medusa-1 requires fine-tuning heads; Medusa-2 is training-free with self-distillation.
tags:
  - speculative-decoding
  - medusa
  - multi-head
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
  - Multiple parallel prediction heads (no separate draft model needed)
  - Tree-based attention for efficient verification of multiple candidates
  - Medusa-1 trains heads on task data (2.2-3.6x speedup)
  - Medusa-2 uses self-distillation (training-free variant, 1.5-2.5x speedup)
  - No quality degradation with rejection sampling
github: https://github.com/FasterDecoding/Medusa
---

## Usage
```bash
pip install medusa-llm
# Medusa heads available on HuggingFace for popular models
# e.g., FasterDecoding/medusa-vicuna-7b-v1.3
```
