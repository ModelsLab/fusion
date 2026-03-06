---
id: vllm-project
kind: source
title: vLLM - High-Throughput LLM Serving Engine
category: library
url: https://github.com/vllm-project/vllm
summary: Production LLM serving engine featuring PagedAttention for efficient KV cache management, continuous batching, speculative decoding, FP8/AWQ/GPTQ quantization support, tensor parallelism, and CUDA graph optimization.
tags:
  - vllm
  - serving
  - paged-attention
  - continuous-batching
  - speculative-decoding
  - tensor-parallelism
source_ids: []
operators:
  - attention
  - matmul
  - sampling
gpu_families:
  - Ampere
  - Ada
  - Hopper
  - Blackwell
key_contributions:
  - PagedAttention eliminates KV cache memory waste (near-zero fragmentation)
  - Continuous batching for maximum GPU utilization
  - Automatic prefix caching for repeated prompts
  - Supports 50+ model architectures out of the box
  - Custom CUDA kernels for attention, sampling, and quantized ops
  - Chunked prefill for mixed long/short request handling
  - OpenAI-compatible API server
---

## Key Custom Kernels
- `csrc/attention/`: PagedAttention v1/v2, FlashAttention integration
- `csrc/quantization/`: AWQ GEMM, GPTQ GEMM, Marlin, FP8, SqueezeLLM
- `csrc/moe/`: Fused MoE (topk gating + permute + GEMM)
- `csrc/activation/`: Fused SiLU-and-mul
- `csrc/layernorm/`: Fused RMSNorm
- `csrc/pos_encoding/`: Fused RoPE
