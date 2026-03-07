---
id: huggingface-quantstack-wan22-gguf
kind: source
title: QuantStack WAN2.2 GGUF Collection
type: community-model-card
category: quantization
summary: QuantStack publishes GGUF conversions of WAN 2.2 text-to-video checkpoints, including multiple 4-bit variants such as Q4_K_S, Q4_0, Q4_1, and Q4_K_M for ComfyUI GGUF-style runtimes.
reliability: community
review_status: reviewed
url: https://huggingface.co/QuantStack/Wan2.2-T2V-A14B-GGUF
tags:
  - wan
  - gguf
  - q4
  - video-generation
  - quantization
---

## Key Features

- Packaged low-bit WAN 2.2 artifacts exist, including several 4-bit GGUF variants
- Relevant when the optimization question is "is there already a smaller packaged model family to test?"
- Runtime compatibility matters because GGUF paths target different inference stacks than the native PyTorch server

## Notes

- Treat these as a distinct packaged-model branch, not as proof that the current PyTorch runtime can execute the same quantization path efficiently
- Useful for Fusion search because they can outperform ad hoc conversion work if the serving stack can adopt the alternate runtime
