---
id: flexgen-offload-compression
kind: source
title: FlexGen Offloading And Compression For Single-GPU Inference
type: paper
category: inference-systems
summary: FlexGen is a single-GPU inference system that uses offloading, compression, and cost-aware scheduling to push large models through limited GPU memory. The core lesson is to schedule data movement deliberately instead of letting transfers dominate the runtime.
reliability: primary
review_status: reviewed
url: https://arxiv.org/abs/2303.06865
tags:
  - arxiv
  - single-gpu
  - offload
  - compression
  - scheduling
---

## Why It Matters

- limited-VRAM systems can still run larger models when transfer, placement, and compression are co-designed
- the paper is LLM-oriented, but the scheduling lesson transfers to diffusion and video pipelines
- it is most useful when the alternative is repeated host-device shuttling of large weights or activations
