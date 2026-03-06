---
id: operator_fusion_inference
kind: strategy
title: Operator Fusion For Reduced Memory Traffic
category: kernel
summary: Fuse adjacent operators to eliminate intermediate memory round-trips, particularly effective for memory-bound operations like norms, residuals, and activations.
support_level: stable
source_ids:
  - triton-tutorials
workloads:
  - prefill
  - decode
  - serving
operators:
  - rmsnorm
  - layernorm
  - softmax
  - attention
  - swiglu
  - rope
gpu_families:
  - Ampere
  - Ada
  - Hopper
  - Blackwell
precision:
  - fp16
  - bf16
  - fp8
bottlenecks:
  - memory
goals:
  - latency
  - throughput
priority: 88
preconditions:
  - adjacent operators are memory-bound
  - fusion is semantically valid (same access pattern)
actions:
  - fuse RMSNorm + residual add (saves one full memory pass)
  - fuse gate+up projection into single GEMM
  - fuse SwiGLU activation after gate+up
  - fuse GEMM epilogue (bias + activation) via cuBLASLt or CUTLASS EVT
  - fuse softmax + causal masking in attention
  - consider torch.compile for automatic pointwise fusion
metrics:
  - kernel launch count reduction
  - memory bandwidth savings
  - end-to-end latency improvement
tradeoffs:
  - fused kernels are more complex to maintain
  - may need separate fused variants for prefill vs decode
  - torch.compile can handle many fusions automatically
---

## Key Fusion Opportunities

1. **RMSNorm + Residual**: ~2x speedup for the norm operation
2. **QKV Projection**: 3 GEMMs → 1 GEMM (larger, more efficient)
3. **Gate+Up Projection**: 2 GEMMs → 1 GEMM (SwiGLU models)
4. **GEMM + Bias + Activation**: via epilogue fusion (zero extra cost)
5. **Attention + RoPE**: avoid separate RoPE kernel launch
