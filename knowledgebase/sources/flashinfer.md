---
id: flashinfer
kind: source
title: FlashInfer - Kernel Library for LLM Serving
category: library
url: https://github.com/flashinfer-ai/flashinfer
summary: High-performance attention kernel library for LLM serving supporting paged KV cache, ragged tensors, GQA/MQA/MHA, FP8 attention, cascade attention, and multiple attention variants optimized for both prefill and decode.
tags:
  - flashinfer
  - attention
  - paged-kv
  - ragged-tensor
  - cascade-attention
  - fp8
source_ids: []
operators:
  - attention
gpu_families:
  - Ampere
  - Ada
  - Hopper
key_contributions:
  - Paged KV cache attention without padding overhead (ragged tensors)
  - Cascade attention for shared prefix computation
  - FP8 KV cache support with per-channel quantization
  - Plan-then-execute API for zero-overhead dynamic shapes
  - Default attention backend for SGLang
---

## Key APIs
```python
import flashinfer

# Paged KV cache attention (decode)
wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer)
wrapper.plan(indptr, indices, last_page_len, num_heads_q, num_heads_kv, head_dim)
output = wrapper.run(query, paged_kv_cache)

# Prefill with ragged tensors
wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace_buffer)
wrapper.plan(qo_indptr, kv_indptr, num_heads_q, num_heads_kv, head_dim)
output = wrapper.run(query, key, value)
```
