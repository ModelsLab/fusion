---
id: huggingface-diffusers-group-offloading
kind: source
title: Hugging Face Diffusers Group Offloading
type: official-doc
category: memory-offload
summary: Diffusers documents layer and block offloading with optional CUDA stream overlap so models can reduce VRAM pressure while hiding some transfer cost behind computation.
reliability: official
review_status: reviewed
url: https://huggingface.co/docs/diffusers/optimization/memory
tags:
  - diffusers
  - offload
  - group-offloading
  - cuda-streams
  - memory
---

## Key Takeaways

- offloading can be organized at block or leaf granularity instead of whole-model swaps
- stream overlap can hide part of transfer time when the runtime prepares the next block early
- pinned memory and stream-aware buffer lifetime matter when host-device transfer dominates

## Fusion Notes

- even outside Diffusers, the reusable idea is staged prefetch with overlap rather than coarse CPU/GPU bouncing
- this is most relevant when profiles show large `cudaMemcpyAsync` cost or repeated host-to-device movement
