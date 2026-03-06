---
id: cutlass-library
kind: source
title: NVIDIA CUTLASS - CUDA Templates for Linear Algebra Subroutines
category: library
url: https://github.com/NVIDIA/cutlass
summary: Template library for high-performance GEMM and convolution kernels on NVIDIA GPUs, featuring CuTe layout abstraction, TMA support, warp specialization, EVT epilogue fusion, and FP4/FP8 tensor core support up to Blackwell.
tags:
  - cutlass
  - cute
  - gemm
  - tensor-cores
  - tma
  - warp-specialization
  - epilogue-fusion
source_ids: []
operators:
  - matmul
  - gemm
  - convolution
gpu_families:
  - Ampere
  - Ada
  - Hopper
  - Blackwell
key_contributions:
  - CuTe DSL for composable layout and copy abstractions
  - CUTLASS 3.x kernel architecture with scheduling policies
  - EVT (Epilogue Visitor Tree) for arbitrary epilogue fusion
  - Persistent kernels with TMA and warp specialization (Hopper)
  - FP4 tensor core support (Blackwell, CUTLASS 3.7+)
  - Python CuTe DSL for kernel development
  - Grouped GEMM for MoE workloads
---

## Architecture Evolution
- CUTLASS 2.x: template-heavy, Ampere/Ada, SIMT + tensor core
- CUTLASS 3.x: CuTe-based, Hopper/Blackwell, TMA + WGMMA + warp specialization
- Python CuTe: experimental Python DSL mirroring C++ CuTe abstractions
