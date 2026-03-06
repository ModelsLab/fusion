---
id: persistent_kernels_decode
kind: strategy
title: Persistent Kernels For Decode Optimization
category: kernel
summary: Use persistent kernels that stay resident on SMs to eliminate kernel launch overhead, particularly effective for small-batch decode where launch costs are a significant fraction of total time.
support_level: experimental
source_ids: []
workloads:
  - decode
gpu_families:
  - Hopper
  - Blackwell
operators:
  - matmul
  - attention
precision:
  - fp16
  - bf16
  - fp8
bottlenecks:
  - memory
goals:
  - latency
priority: 60
preconditions:
  - kernel launch overhead is measurable (>5% of per-token time)
  - GPU has enough SMs to keep persistent kernels resident
actions:
  - measure kernel launch overhead in decode step using Nsight Systems
  - implement persistent GEMM kernel using CUTLASS persistent scheduler
  - benchmark against CUDA graphs (often simpler alternative)
metrics:
  - kernel launch overhead reduction
  - per-token latency improvement
tradeoffs:
  - more complex kernel code
  - CUDA graphs may achieve similar benefits with less complexity
  - requires careful work distribution logic
---

## CUDA Graphs vs Persistent Kernels
- CUDA graphs: simpler, captures entire decode step, widely supported
- Persistent kernels: more flexible, works with dynamic shapes, better for grouped GEMM
- Start with CUDA graphs, use persistent kernels when CUDA graphs can't work
