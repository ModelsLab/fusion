---
id: sglang-paper
kind: source
title: "SGLang: Efficient Execution of Structured Language Model Programs"
category: paper
url: https://arxiv.org/abs/2312.07104
summary: SGLang serving framework with RadixAttention for automatic KV cache reuse, constrained decoding with jump-forward, and compressed finite state machines for structured output generation.
tags:
  - sglang
  - radix-attention
  - constrained-decoding
  - kv-cache-reuse
  - serving
source_ids: []
operators:
  - attention
  - sampling
gpu_families:
  - Ampere
  - Ada
  - Hopper
key_contributions:
  - RadixAttention for automatic prefix caching using radix tree data structure
  - Compressed FSM for efficient constrained decoding (grammar-guided generation)
  - Jump-forward decoding skips model evaluation for deterministic tokens
  - FlashInfer integration for high-performance attention kernels
  - Achieves 2-5x higher throughput than vLLM on structured output tasks
---

## Key Technical Details

### RadixAttention
- Stores KV cache in a radix tree indexed by token sequences
- Automatic cache hit detection for shared prefixes across requests
- LRU eviction policy for cache management
- Enables efficient multi-turn conversation and few-shot prompt caching

### Constrained Decoding
- Compresses grammar FSM states to reduce branching overhead
- Jump-forward: when only one valid next token exists, emit without model call
- Achieves near-zero overhead for JSON/regex-constrained generation
