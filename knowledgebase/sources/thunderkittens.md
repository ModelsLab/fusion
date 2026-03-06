---
id: thunderkittens
kind: source
title: ThunderKittens - Stanford GPU Kernel DSL
type: code
category: kernel-lib
summary: A GPU kernel DSL from Stanford focused on register-tile-first programming, providing higher-level abstractions than raw CUDA while maintaining performance.
reliability: research
review_status: reviewed
url: https://github.com/HazyResearch/ThunderKittens
tags:
  - thunderkittens
  - dsl
  - kernel
  - register-tile
  - stanford
---

## Key Ideas
- Register tiles as the primary abstraction (not threads or warps)
- Operations on tiles: load, store, mma, reduce, etc.
- Compile-time layout optimization
- Designed for attention and GEMM-like operations
- ~100 lines of ThunderKittens ≈ ~1000 lines of CUDA
