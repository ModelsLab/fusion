---
id: marlin-kernel
kind: source
title: Marlin - Fastest W4A16 GEMM Kernel
type: code
category: kernel
summary: Marlin is a highly optimized CUDA kernel for 4-bit weight, 16-bit activation GEMM that achieves near-optimal memory bandwidth utilization.
reliability: authoritative
review_status: reviewed
url: https://github.com/IST-DASLab/marlin
tags:
  - marlin
  - w4a16
  - gemm
  - quantization
  - int4
  - kernel
---

## Summary
Marlin achieves near-peak memory bandwidth for W4A16 GEMM by:
- Asynchronous global→shared memory with cp.async
- Register-level INT4→FP16 dequantization
- Full tensor core utilization during memory latency
- Specialized for batch=1 and small-batch decode

Achieves 3.5-4x speedup over FP16 cuBLAS for decode (batch=1).
Integrated into vLLM as the primary GPTQ/AWQ kernel backend.
