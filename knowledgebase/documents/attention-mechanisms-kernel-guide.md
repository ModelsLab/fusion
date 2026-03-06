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

## Practical Attention Recipes

### Recipe 1: Use FlashAttention in PyTorch (3 Ways)

```python
import torch
import time

# Setup: common inputs for all methods
B, H, S, D = 4, 32, 4096, 128
device = "cuda"
dtype = torch.float16

Q = torch.randn(B, H, S, D, device=device, dtype=dtype)
K = torch.randn(B, H, S, D, device=device, dtype=dtype)
V = torch.randn(B, H, S, D, device=device, dtype=dtype)

def benchmark(fn, name, warmup=10, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters * 1000
    print(f"{name}: {elapsed:.2f} ms")


# --------------------------------------------------------------------------
# Method 1: torch.nn.functional.scaled_dot_product_attention (built-in, easiest)
# --------------------------------------------------------------------------
# Available since PyTorch 2.0. Automatically selects the best backend
# (FlashAttention, Memory-Efficient, or Math) based on inputs and hardware.
import torch.nn.functional as F

def method1():
    return F.scaled_dot_product_attention(Q, K, V, is_causal=True)

benchmark(method1, "SDPA (auto backend)")


# --------------------------------------------------------------------------
# Method 2: flash_attn package (fastest, pip install flash-attn --no-build-isolation)
# --------------------------------------------------------------------------
# Requires Ampere+ GPU. Inputs must be (B, S, H, D) layout, not (B, H, S, D).
from flash_attn import flash_attn_func

Q_flash = Q.transpose(1, 2).contiguous()  # (B, S, H, D)
K_flash = K.transpose(1, 2).contiguous()
V_flash = V.transpose(1, 2).contiguous()

def method2():
    return flash_attn_func(Q_flash, K_flash, V_flash, causal=True)

benchmark(method2, "flash-attn package")


# --------------------------------------------------------------------------
# Method 3: xformers (pip install xformers)
# --------------------------------------------------------------------------
# Supports a wider range of GPUs including Turing (T4).
from xformers.ops import memory_efficient_attention
from xformers.ops import LowerTriangularMask

Q_xf = Q.transpose(1, 2)  # (B, S, H, D)
K_xf = K.transpose(1, 2)
V_xf = V.transpose(1, 2)

def method3():
    return memory_efficient_attention(Q_xf, K_xf, V_xf, attn_bias=LowerTriangularMask())

benchmark(method3, "xformers")


# --------------------------------------------------------------------------
# Typical results (A100 80GB, S=4096, D=128, FP16, causal):
#   SDPA (auto backend):  ~3.1 ms  (selects flash backend internally)
#   flash-attn package:   ~2.8 ms  (fastest, minimal overhead)
#   xformers:             ~3.0 ms  (close to flash-attn)
#
# At S=16384:
#   SDPA:                 ~48 ms
#   flash-attn:           ~44 ms
#   xformers:             ~46 ms
# --------------------------------------------------------------------------
```

### Recipe 2: Check Which Attention Backend is Active

```python
import torch
import torch.nn.functional as F

B, H, S, D = 2, 8, 1024, 64
Q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
K = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
V = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)

# --- Detect which backend SDPA would select ---
from torch.backends.cuda import (
    flash_sdp_enabled,
    mem_efficient_sdp_enabled,
    math_sdp_enabled,
)

print(f"Flash SDP available:            {flash_sdp_enabled()}")
print(f"Memory-efficient SDP available: {mem_efficient_sdp_enabled()}")
print(f"Math SDP available:             {math_sdp_enabled()}")


# --- Force a specific backend using the context manager ---
from torch.nn.attention import sdpa_kernel, SDPBackend

# Force FlashAttention only
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    out_flash = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
    print("Forced FlashAttention backend")

# Force memory-efficient (xformers-like) only
with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
    out_efficient = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
    print("Forced Memory-Efficient backend")

# Force math (unfused) backend -- slow but always works
with sdpa_kernel(SDPBackend.MATH):
    out_math = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
    print("Forced Math backend")

# Force CuDNN attention (PyTorch 2.2+, Hopper)
# with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
#     out_cudnn = F.scaled_dot_product_attention(Q, K, V, is_causal=True)


# --- Verify with profiling (most reliable method) ---
# Run with: python -m torch.utils.bottleneck your_script.py
# Or use the PyTorch profiler:
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
) as prof:
    F.scaled_dot_product_attention(Q, K, V, is_causal=True)

# Look for kernel names:
#   "flash_fwd"            -> FlashAttention
#   "efficient_attention"  -> Memory-Efficient
#   "sdp_math"             -> Math fallback
for event in prof.key_averages():
    if "attention" in event.key.lower() or "flash" in event.key.lower() or "sdp" in event.key.lower():
        print(f"  Kernel: {event.key}  CUDA time: {event.cuda_time_total:.0f} us")
```

### Recipe 3: FlashAttention Installation Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `No module named 'flash_attn'` | Package not installed | `pip install flash-attn --no-build-isolation` |
| `FlashAttention only supports Ampere GPUs or newer` | GPU compute capability < 8.0 (e.g., T4, V100) | Use `xformers` or the SDPA math backend instead |
| Build fails with `nvcc fatal: Unsupported gpu architecture` | CUDA toolkit too old for your GPU or missing arch flag | Install CUDA toolkit >= 11.6; for Ada/Hopper ensure >= 11.8 |
| `RuntimeError: FlashAttention does not support head_dim > 256` | Head dimension exceeds kernel limit | Reshape to split heads: `(B, H, S, 512)` -> `(B, 2*H, S, 256)` |
| `No matching distribution found for flash-attn` | Python version or platform mismatch | Use Python 3.8-3.11 on Linux x86_64; flash-attn has no macOS/Windows wheels |
| `error: subprocess-exited-with-error` during build | Missing build dependencies or ninja | `pip install packaging ninja setuptools wheel` then retry |
| `torch.cuda.OutOfMemoryError` during import/first call | GPU memory fragmented or near-full | Free other tensors; set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| `undefined symbol: _ZN2at...` | flash-attn built against different PyTorch version | Reinstall: `pip install flash-attn --no-build-isolation --force-reinstall` matching your PyTorch |
| `CUDA driver version is insufficient` | System NVIDIA driver too old | Update NVIDIA driver to >= 525.60 for CUDA 12.x support |

**Quick compatibility check:**

```python
import torch
print(f"PyTorch:          {torch.__version__}")
print(f"CUDA available:   {torch.cuda.is_available()}")
print(f"CUDA version:     {torch.version.cuda}")
print(f"GPU:              {torch.cuda.get_device_name(0)}")
print(f"Compute cap:      {torch.cuda.get_device_capability(0)}")

cap = torch.cuda.get_device_capability(0)
if cap >= (8, 0):
    print("-> FlashAttention-2 supported (Ampere+)")
if cap >= (9, 0):
    print("-> FlashAttention-3 supported (Hopper+)")
if cap < (8, 0):
    print("-> Use xformers or SDPA math/efficient backend")
```

### Recipe 4: Measure Attention Performance

```python
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
import time
import json

def measure_attention(B, H, S, D, backend, causal=True, warmup=20, iters=100):
    """Measure latency and peak memory for a given attention backend."""
    Q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    K = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    V = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)

    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.max_memory_allocated()

    try:
        with sdpa_kernel(backend):
            # Warmup
            for _ in range(warmup):
                _ = F.scaled_dot_product_attention(Q, K, V, is_causal=causal)
            torch.cuda.synchronize()

            # Timed run
            start = time.perf_counter()
            for _ in range(iters):
                _ = F.scaled_dot_product_attention(Q, K, V, is_causal=causal)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) / iters * 1000

        mem_after = torch.cuda.max_memory_allocated()
        mem_delta_mb = (mem_after - mem_before) / 1024**2
        return {"latency_ms": round(elapsed_ms, 3), "peak_mem_mb": round(mem_delta_mb, 1)}

    except RuntimeError as e:
        return {"latency_ms": None, "peak_mem_mb": None, "error": str(e)[:80]}


# --- Benchmark across sequence lengths ---
backends = {
    "Flash":     SDPBackend.FLASH_ATTENTION,
    "Efficient": SDPBackend.EFFICIENT_ATTENTION,
    "Math":      SDPBackend.MATH,
}

seq_lengths = [512, 1024, 2048, 4096, 8192, 16384]
B, H, D = 4, 32, 128

results = {}
for name, backend in backends.items():
    results[name] = {}
    for S in seq_lengths:
        r = measure_attention(B, H, S, D, backend)
        results[name][S] = r
        status = f"{r['latency_ms']:.2f} ms, {r['peak_mem_mb']:.0f} MB" if r["latency_ms"] else r.get("error", "N/A")
        print(f"{name:>10} | S={S:>6} | {status}")
    torch.cuda.empty_cache()

# --- Plot results (requires matplotlib) ---
try:
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for name in backends:
        seqs = [s for s in seq_lengths if results[name][s]["latency_ms"] is not None]
        lats = [results[name][s]["latency_ms"] for s in seqs]
        mems = [results[name][s]["peak_mem_mb"] for s in seqs]

        ax1.plot(seqs, lats, marker="o", label=name)
        ax2.plot(seqs, mems, marker="s", label=name)

    ax1.set_xlabel("Sequence Length")
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("Attention Latency vs Sequence Length")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log", base=10)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Sequence Length")
    ax2.set_ylabel("Peak Memory (MB)")
    ax2.set_title("Attention Memory vs Sequence Length")
    ax2.set_xscale("log", base=2)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("attention_benchmark.png", dpi=150)
    print("Plot saved to attention_benchmark.png")
except ImportError:
    print("Install matplotlib to generate plots: pip install matplotlib")
    print("Raw results:")
    print(json.dumps(results, indent=2))
```

### Recipe 5: Enable FlashAttention in vLLM/SGLang

**vLLM: Attention Backend Selection**

```bash
# vLLM automatically selects the best attention backend.
# Override with the VLLM_ATTENTION_BACKEND environment variable:

# Use FlashAttention (default on Ampere+)
VLLM_ATTENTION_BACKEND=FLASH_ATTN vllm serve meta-llama/Llama-3.1-8B-Instruct

# Use FlashInfer (recommended for decode-heavy workloads)
VLLM_ATTENTION_BACKEND=FLASHINFER vllm serve meta-llama/Llama-3.1-8B-Instruct

# Use xformers (wider GPU compatibility)
VLLM_ATTENTION_BACKEND=XFORMERS vllm serve meta-llama/Llama-3.1-8B-Instruct

# Verify in startup logs -- look for:
#   "Using attention backend: FLASH_ATTN"
#   or "Using attention backend: FLASHINFER"
```

**SGLang: FlashInfer Configuration**

```bash
# SGLang uses FlashInfer by default for attention kernels.

# Launch with default FlashInfer attention:
python -m sglang.launch_server --model meta-llama/Llama-3.1-8B-Instruct

# Explicitly set attention backend:
python -m sglang.launch_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --attention-backend flashinfer

# Use triton backend instead (useful for debugging or unsupported configs):
python -m sglang.launch_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --attention-backend triton

# Verify FlashInfer is active -- check startup logs for:
#   "Attention backend: flashinfer"
#   "FlashInfer version: x.y.z"
```

**How to Verify the Attention Backend is Being Used**

```bash
# Method 1: Check server startup logs
# vLLM prints: "Using attention backend: FLASH_ATTN"
# SGLang prints: "Attention backend: flashinfer"

# Method 2: Run with CUDA profiling
nsys profile -o attn_trace python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct &
# Send a request, then Ctrl+C
# Open in Nsight Systems and search for kernel names:
#   "flash_fwd" = FlashAttention
#   "BatchPrefillWithPagedKVCache" = FlashInfer
#   "paged_attention" = vLLM PagedAttention

# Method 3: Python-level check in vLLM
python -c "
from vllm import LLM
llm = LLM(model='meta-llama/Llama-3.1-8B-Instruct', enforce_eager=True)
print(llm.llm_engine.model_config.dtype)
# Check logs output during initialization for backend info
"
```

### When to Use Which Attention

| Scenario | Best Choice | Why |
|----------|-------------|-----|
| Short sequences (<2K), any GPU | SDPA Math / Efficient | Low kernel launch overhead, no special requirements |
| Long sequences (>4K), Ampere+ GPU | FlashAttention-2 | O(N) memory instead of O(N^2), no attention matrix materialization |
| Hopper GPU (H100/H200) | FlashAttention-3 | Exploits TMA async copies and WGMMA tensor core instructions |
| Paged KV cache in serving | FlashInfer (paged) | Native support for block tables and non-contiguous KV memory |
| Variable-length batches | FlashInfer (ragged) | Packed tensor layout eliminates padding waste entirely |
| Prefix caching (shared system prompts) | FlashInfer cascade | Computes shared-prefix attention once, merges per-request suffixes |
| Decode-only (single token generation) | FlashDecoding / FlashInfer split-KV | Parallelizes across KV length to avoid thread block underutilization |
| GQA models (LLaMA-3, Mistral) | FlashInfer or FlashAttention-2 | Both handle grouped-query natively with KV head broadcasting |
| FP8 KV cache quantization | FlashAttention-3 / FlashInfer | Native FP8 dequantize-on-the-fly during attention |
| Sliding window models (Mistral) | FlashInfer or custom kernel | Native window mask support, bounded KV cache |
| Turing GPUs (T4, RTX 2080) | xformers or SDPA efficient | FlashAttention requires Ampere+; these work on compute cap 7.5 |
| Training with custom attention bias | SDPA with `attn_mask` param | Flexible masking support; Flash backend handles causal/no mask |
| Speculative decoding (draft + verify) | FlashInfer batch prefill | Efficiently handles mixed-length verification sequences |
| Multi-node / tensor parallel | FlashAttention-2 + NCCL ring | Attention is local per-head; TP splits heads across GPUs |
