---
id: attention_mechanisms_kernel_guide
kind: document
title: Attention Mechanisms - Complete Kernel Implementation Guide
category: attention
summary: Deep technical guide to every attention mechanism variant and their GPU kernel implementations, from FlashAttention to MLA to sparse attention.
tags:
  - attention
  - flash-attention
  - flashinfer
  - paged-attention
  - mla
  - gqa
  - sparse-attention
  - kv-cache
source_ids:
  - flashinfer-docs
  - flashinfer-repo
  - pagedattention-paper
  - pytorch-attention-docs
gpu_families:
  - Ampere
  - Ada
  - Hopper
  - Blackwell
operators:
  - attention
  - softmax
  - kv-cache
  - paged-attention
precision:
  - fp16
  - bf16
  - fp8
---

# Attention Mechanisms - Complete Kernel Implementation Guide

## Standard Multi-Head Attention

### Mathematical Formulation
```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

Where:
  Q: (B, H, S, D)  - queries
  K: (B, H, T, D)  - keys (T = source/KV sequence length)
  V: (B, H, T, D)  - values
  Output: (B, H, S, D)

  B = batch size
  H = number of heads
  S = query sequence length
  T = key/value sequence length
  D = head dimension (typically 64 or 128)
```

### Naive Implementation Memory Problem
```
# This materializes the full S x T attention matrix:
scores = Q @ K.transpose(-2, -1) / sqrt(d_k)  # (B, H, S, T) - O(S*T) memory!
attn_weights = softmax(scores, dim=-1)          # (B, H, S, T)
output = attn_weights @ V                       # (B, H, S, D)

# For S=T=128K: attention matrix = 128K * 128K * 2 bytes = 32 GB per head!
```

## FlashAttention Algorithm

### Key Insight: Online Softmax with Tiling
Instead of materializing the full attention matrix, compute attention in tiles while maintaining a running softmax.

### FlashAttention v1 Algorithm
```
# Tile Q into blocks of size B_r, tile K,V into blocks of size B_c
# For each Q block:
#   For each K,V block:
#     1. Compute local attention scores: S_ij = Q_i @ K_j^T
#     2. Track running max: m_i = max(m_i_prev, rowmax(S_ij))
#     3. Update softmax normalization: l_i = exp(m_i_prev - m_i) * l_i_prev + rowsum(exp(S_ij - m_i))
#     4. Update output: O_i = diag(exp(m_i_prev - m_i)) * l_i_prev / l_i * O_i_prev
#                            + 1/l_i * exp(S_ij - m_i) @ V_j

# Memory: O(N) instead of O(N^2)
# No materialization of full attention matrix!
```

### Online Softmax Trick (Milakov & Gimelshein 2018)
```python
# Standard softmax requires two passes:
# Pass 1: max_val = max(x)
# Pass 2: softmax = exp(x - max_val) / sum(exp(x - max_val))

# Online softmax - single pass with running statistics:
m = -inf  # running max
d = 0     # running sum of exp
for x_i in x:
    m_new = max(m, x_i)
    d = d * exp(m - m_new) + exp(x_i - m_new)
    m = m_new
# Final: softmax(x_i) = exp(x_i - m) / d
```

### FlashAttention v2 Improvements
1. **Better work partitioning**: Parallelize over Q blocks (not K/V blocks) in the outer loop
2. **Reduced non-matmul FLOPs**: Minimize rescaling operations
3. **Better occupancy**: Support variable-length sequences without padding
4. **2x speedup over v1**: Better utilization of tensor cores

```
# v2 loop structure (swapped inner/outer):
for each Q block i (PARALLEL across thread blocks):
    for each K,V block j (SEQUENTIAL within thread block):
        compute local attention and update running statistics
```

### FlashAttention-3 (Hopper-Specific)

**Key innovations for H100**:

1. **Warp Specialization**:
   - Producer warps: load K,V tiles using TMA
   - Consumer warps: compute attention using WGMMA
   - Overlaps memory and compute

2. **Asynchronous Pipeline (TMA)**:
   ```
   # TMA handles bulk data movement without warp involvement
   tma_load_async(smem_K[stage], gmem_K[block])
   tma_load_async(smem_V[stage], gmem_V[block])
   # Consumer warps can compute on previously loaded stages
   ```

3. **FP8 Attention**:
   - Q, K in FP8 for score computation
   - Accumulate in FP32
   - V in FP8 for output computation
   - ~2x throughput vs FP16 attention

4. **Intra-Warp Pipelining**:
   - Within a single warp group, pipeline softmax with the next GEMM
   - softmax(S_j) happens concurrently with S_{j+1} = Q @ K_{j+1}^T

**Performance**: 1.5-2x faster than FlashAttention-2 on H100.

## FlashDecoding and FlashDecoding++

### Problem: Decode Attention is Underutilized
During decode, S=1 (single new token) but T is large (full context). The standard FlashAttention parallelizes over Q blocks, but with S=1, there's only 1 Q block → only 1 thread block → massive underutilization.

### FlashDecoding Solution
Parallelize across the KV sequence length:
```
# Split KV into chunks, each chunk processed by a separate thread block
# Each thread block computes partial attention over its KV chunk
# Final reduction combines partial results using the online softmax trick

for each KV chunk k (PARALLEL across thread blocks):
    partial_output[k], partial_lse[k] = flash_attention(Q, K[chunk_k], V[chunk_k])

# Reduction kernel:
output = reduce_attention(partial_output[], partial_lse[])
```

### FlashDecoding++
Further optimizations:
- Unified max value across chunks (avoids costly reduction)
- Asynchronous softmax computation
- Better memory access patterns for KV cache

## FlashInfer

### Architecture
FlashInfer is a kernel library specifically for LLM serving attention:

**Key Features**:
1. **Multiple KV layouts**: Paged (block tables), Ragged (packed variable-length), Contiguous
2. **JIT Compilation**: Generates specialized kernels at runtime using CUDA/Triton
3. **Cascade Attention**: For prefix caching (shared prefix + unique suffix)
4. **Batch Operations**: Efficiently handle batches with different sequence lengths

### Paged KV Cache Layout
```python
# vLLM-style paged KV cache:
# Physical blocks of fixed size (e.g., 16 tokens)
# Block tables map logical positions to physical blocks

# block_table[batch_idx][logical_block_idx] = physical_block_idx
# kv_data[physical_block_idx] shape: (2, page_size, num_heads, head_dim)
# 2 for K and V

# During attention:
for each query token:
    for each KV block in block_table[batch]:
        phys_block = block_table[batch][block_idx]
        K_block = kv_data[phys_block, 0]  # K
        V_block = kv_data[phys_block, 1]  # V
        compute_attention_with_block(Q, K_block, V_block)
```

### Ragged Tensor Layout
```python
# Variable-length sequences packed contiguously
# indptr array marks boundaries: seq_i occupies data[indptr[i]:indptr[i+1]]

# Example for 3 sequences of lengths [5, 3, 7]:
# indptr = [0, 5, 8, 15]
# data = [seq0_tok0, ..., seq0_tok4, seq1_tok0, ..., seq1_tok2, seq2_tok0, ..., seq2_tok6]
```

### Cascade Attention (for Prefix Caching)
```python
# Many requests share a common prefix (system prompt)
# Instead of duplicating KV cache for shared prefix:

# Level 1: Compute attention over shared prefix KV (read once, apply to all)
partial_out_shared, lse_shared = attention(Q_batch, K_shared, V_shared)

# Level 2: Compute attention over unique suffix KV per request
partial_out_unique[i], lse_unique[i] = attention(Q[i], K_unique[i], V_unique[i])

# Merge: combine using log-sum-exp
output[i] = merge_attention(partial_out_shared, lse_shared, partial_out_unique[i], lse_unique[i])
```

## PagedAttention (vLLM)

### Virtual Memory for KV Cache
Inspired by OS virtual memory / paging:

```
# Physical KV blocks: fixed-size blocks in GPU memory
# Block table: maps logical block index to physical block
# Each block stores page_size tokens (typically 16)

# Benefits:
# 1. No memory fragmentation (blocks are uniform size)
# 2. Memory sharing (copy-on-write for beam search, parallel sampling)
# 3. Dynamic allocation (grow KV cache as needed)
# 4. Near-zero memory waste (only last block partially filled)
```

### Kernel Implementation
```cuda
// PagedAttention kernel (simplified):
__global__ void paged_attention_kernel(
    float* output,          // (num_seqs, num_heads, head_dim)
    const float* query,     // (num_seqs, num_heads, head_dim)
    const float* key_cache, // (num_blocks, num_heads, page_size, head_dim)
    const float* val_cache, // (num_blocks, num_heads, page_size, head_dim)
    const int* block_tables,// (num_seqs, max_num_blocks)
    const int* seq_lens,    // (num_seqs,)
    int page_size
) {
    int seq_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int num_blocks = (seq_lens[seq_idx] + page_size - 1) / page_size;

    float max_score = -INFINITY;
    float exp_sum = 0;
    float acc[HEAD_DIM] = {0};

    // Iterate over KV blocks
    for (int block = 0; block < num_blocks; block++) {
        int phys_block = block_tables[seq_idx * max_blocks + block];
        // Load K from this physical block, compute QK^T
        // Update running softmax statistics
        // Accumulate weighted V
    }
    // Write output
}
```

### PagedAttention v2
Splits the sequence dimension across multiple thread blocks (like FlashDecoding):
```
# v1: one thread block per (sequence, head) pair - underutilized for long sequences
# v2: multiple thread blocks per (sequence, head), partition across KV blocks
#     then reduce partial results
```

## Attention Variants

### Grouped-Query Attention (GQA)
```
# Instead of H key/value heads, use G groups (G < H)
# Multiple query heads share the same KV head
# LLaMA-2-70B: 64 query heads, 8 KV heads (8:1 ratio)

# Kernel implications:
# - KV cache is G * head_dim * seq_len instead of H * head_dim * seq_len
# - Multiple query heads read the same K,V → can broadcast
# - Memory savings for KV cache: H/G reduction (8x for 70B)
```

### Multi-Query Attention (MQA)
```
# Extreme case of GQA: G=1, all query heads share one KV head
# Minimal KV cache but potentially lower quality
# Used in: Falcon, PaLM
```

### Multi-Latent Attention (MLA) - DeepSeek
```
# Key innovation: compress KV into low-rank latent space
# Instead of caching full K,V, cache compressed representations

# Standard: cache K (H, T, D) and V (H, T, D)
# MLA: cache C_kv (T, D_c) where D_c << H*D

# During attention:
# K = C_kv @ W_uk  # decompress keys
# V = C_kv @ W_uv  # decompress values
# Then standard attention with decompressed K, V

# Memory savings: D_c / (2 * H * D) compression ratio
# DeepSeek-V2: 512 latent dim vs 2 * 128 * 128 = 32768 → 64x compression!
```

### Sliding Window Attention
```
# Each token only attends to the last W tokens (window size)
# Used in: Mistral (W=4096)

# Kernel: mask attention scores where |i - j| > W
# KV cache: only need to store last W tokens (circular buffer)

# Implementation:
# - During prefill: apply causal mask AND window mask
# - During decode: only load last W KV entries
# - Memory: O(W) instead of O(T) per sequence
```

### Differential Attention (Microsoft)
```
# Compute attention as difference of two softmax attention maps
# attn = softmax(Q1 @ K1^T) - lambda * softmax(Q2 @ K2^T)
# Reduces noise in attention, improves quality

# Kernel: need to compute two attention maps and subtract
# Can fuse both computations in one pass
```

### Native Sparse Attention (NSA) - DeepSeek
```
# Dynamically select which tokens to attend to
# Three components:
# 1. Compressed attention: attend to block-compressed representations
# 2. Selected attention: attend to top-k important tokens (selected by routing)
# 3. Sliding window: attend to recent tokens

# Kernel needs:
# - Token selection kernel (top-k routing)
# - Gather-based attention for selected tokens
# - Standard sliding window attention
# - Merge results from all three components
```

### SageAttention
```
# Quantized attention computation:
# 1. Quantize Q, K to INT8 for score computation: S = Q_int8 @ K_int8^T
# 2. Dequantize scores, apply softmax in FP16/FP32
# 3. Use FP8/FP16 for V matmul: O = softmax(S) @ V

# Achieves ~2x speedup with minimal accuracy loss
# Key: per-head quantization of Q,K is sufficient
```

## KV Cache Optimization Strategies

### StreamingLLM (Attention Sinks)
```
# Observation: first few tokens get disproportionate attention (attention sinks)
# Strategy: always keep first few tokens + sliding window of recent tokens

# KV cache layout:
# [sink_tokens (4-8)] + [recent_tokens (window_size)]
# Enables infinite-length generation with fixed memory

# Kernel: attention over two non-contiguous segments
```

### H2O (Heavy-Hitter Oracle)
```
# Dynamically evict KV cache entries based on cumulative attention scores
# Keep tokens that consistently receive high attention

# Each decoding step:
# 1. Compute attention as normal
# 2. Track cumulative attention score per token
# 3. If cache is full, evict token with lowest cumulative score
# 4. Store new token's KV

# Kernel: need online tracking + eviction decision
```

### SnapKV
```
# Observe attention patterns during prefill
# Identify important tokens per head
# Compress KV cache by keeping only important tokens + recent window

# After prefill:
# 1. Compute attention patterns over "observation window" (last few layers)
# 2. Pool attention scores over observation tokens
# 3. Select top-k tokens per head based on pooled scores
# 4. Keep selected KV + recent window for decode phase
```

### XKV (Cross-Layer KV Cache Sharing)
```
# Observation: KV caches across adjacent layers are often very similar
# Strategy: share KV cache between layers, only store differences

# Group layers into sharing groups
# Lead layer computes full KV, follower layers reuse (with optional corrections)
# Can reduce KV cache by 2-4x across layers
```

## Attention Kernel Performance Analysis

### Prefill Attention (Compute-Bound)
```
FLOPs = 4 * B * H * S * T * D  (QK^T and softmax@V, each 2*S*T*D)
Bytes = B * H * (S*D + T*D + T*D + S*D) * dtype_size  (Q, K, V, O)
AI = 4*S*T*D / ((S+2T+S)*D * dtype_size) ≈ 2*T / (2*dtype_size) for large S≈T

For S=T=4096, FP16: AI ≈ 2048 → deeply compute-bound
```

### Decode Attention (Memory-Bound)
```
S = 1, T = seq_len (possibly 4K-128K)
FLOPs = 4 * B * H * 1 * T * D
Bytes = B * H * (1*D + T*D + T*D + 1*D) * dtype_size ≈ B * H * 2*T*D * dtype_size

AI = 4*T*D / (2*T*D * dtype_size) = 2 / dtype_size

For FP16: AI = 1 → severely memory-bound
For FP8 KV cache: AI ≈ 2 → still memory-bound but 2x faster
```

This is why:
1. Prefill benefits from larger tile sizes and max tensor core utilization
2. Decode benefits from KV cache quantization, batching (increases M), and bandwidth optimization
3. Different kernels should be used for prefill vs decode

### Choosing the Right Attention Kernel

| Scenario | Recommended | Why |
|----------|------------|-----|
| Prefill, standard MHA | FlashAttention-2/3 | Best compute-bound attention |
| Prefill, Hopper | FlashAttention-3 | Warp specialization, TMA |
| Decode, serving | FlashInfer / PagedAttention | Paged KV, batched decode |
| Decode, long context | FlashDecoding++ | KV-parallel decomposition |
| GQA decode | FlashInfer (split-kv) | Handles GQA natively |
| Prefix caching | FlashInfer cascade | Avoids recomputing shared prefix |
| Sliding window | Custom or FlashInfer | Native window support |
| FP8 attention | FlashAttention-3 | Native FP8 on Hopper |
| Blackwell | cuTile attention (CUTLASS) | Fifth-gen tensor cores |
| Variable-length batch | FlashInfer (ragged) | No padding needed |
