---
id: memory_optimization_complete_guide
kind: document
title: GPU Memory Optimization - Complete Guide
category: memory
summary: Deep technical guide to GPU memory hierarchy, optimization techniques, KV cache management, memory budgets, and practical memory calculations for LLM inference.
tags:
  - memory
  - kv-cache
  - hbm
  - shared-memory
  - coalescing
  - memory-management
gpu_families:
  - Ampere
  - Ada
  - Hopper
  - Blackwell
---

# GPU Memory Optimization - Complete Guide

## GPU Memory Hierarchy

### Register File (Fastest)
- **Speed**: ~0 cycles latency, ~10+ TB/s effective bandwidth
- **Size per SM**: 256 KB (65,536 x 32-bit registers)
- **Per thread**: up to 255 registers (each 32-bit = 4 bytes)
- **Total per thread**: up to 1020 bytes in registers
- **Key**: More registers per thread → lower occupancy

Register allocation examples:
```
Simple element-wise kernel: ~16 registers → high occupancy
RMSNorm with accumulator: ~32-48 registers → good occupancy
GEMM tile in registers: ~128-200 registers → lower occupancy but necessary
Flash attention: ~128-180 registers → carefully tuned
```

### L1 Cache / Shared Memory (SMEM)
- **Speed**: ~20-30 cycles latency, ~10+ TB/s effective bandwidth
- **Configurable split** (some architectures allow L1/SMEM ratio tuning)

| GPU | Total L1+SMEM per SM | Max SMEM | L2 Cache |
|-----|---------------------|----------|----------|
| A100 | 192 KB | 164 KB | 40 MB |
| RTX 3090 | 128 KB | 100 KB | 6 MB |
| H100 | 256 KB | 228 KB | 50 MB |
| RTX 4090 | 128 KB | 100 KB | 72 MB |
| B200 | 256 KB | ~228 KB | 192 MB |

**Shared Memory Banks**: 32 banks, 4-byte stride per bank
- Bank conflict occurs when 2+ threads in a warp access different addresses in the same bank
- Broadcast: all threads accessing same address → no conflict
- Multicast (Ampere+): subset accessing same address → no conflict

### L2 Cache
- **Speed**: ~200 cycles latency, ~4-6 TB/s effective bandwidth
- Shared across all SMs
- Critical for GEMM: tiles of A and B may hit L2
- L2 cache residency control (Ampere+):
```cuda
// Tell the GPU to keep certain data in L2
cudaAccessPolicyWindow policy;
policy.base_ptr = device_ptr;
policy.num_bytes = size;
policy.hitRatio = 1.0f;  // try to keep 100% in L2
policy.hitProp = cudaAccessPropertyPersisting;
policy.missProp = cudaAccessPropertyStreaming;
cudaCtxSetAccessPolicyWindow(&policy);
```

### HBM (Global Memory)
- **Speed**: ~400-600 cycles latency
- Effective bandwidth typically 80-90% of theoretical

Achievable bandwidth (practical):
| GPU | Theoretical | Achievable | How to measure |
|-----|------------|------------|---------------|
| A100 SXM | 2039 GB/s | ~1800 GB/s | bandwidthTest or ncu |
| H100 SXM | 3350 GB/s | ~2900 GB/s | |
| RTX 4090 | 1008 GB/s | ~900 GB/s | |
| B200 | 8000 GB/s | ~7000 GB/s | |

## Memory Coalescing Deep Dive

### What Makes an Access Coalesced
A warp (32 threads) issues a memory request. The memory controller services it in **32-byte sectors**.

```
Ideal: 32 threads each access consecutive 4-byte floats
  Thread 0 → addr 0x1000
  Thread 1 → addr 0x1004
  ...
  Thread 31 → addr 0x107C
  = 128 bytes = 4 sectors → perfect coalescing

Bad: 32 threads access with stride 2
  Thread 0 → addr 0x1000
  Thread 1 → addr 0x1008  (skip 4 bytes)
  ...
  = 256 bytes = 8 sectors → 2x overhead

Terrible: 32 threads access random addresses
  = up to 32 sectors → 8x overhead!
```

### Vectorized Loads
```cuda
// Load 1 float at a time (4 bytes per thread):
float val = input[idx];  // 4 sectors per warp for 128 bytes

// Load float4 (16 bytes per thread, aligned):
float4 val4 = reinterpret_cast<float4*>(input)[idx];  // Same 4 sectors but 4x data!

// For FP16, use half2:
half2 val2 = reinterpret_cast<half2*>(input)[idx];
```

**When to vectorize**: When consecutive threads access consecutive memory and alignment is guaranteed.

### AoS vs SoA
```
// Array of Structures (AoS) - BAD for GPU:
struct Particle { float x, y, z, w; };
Particle particles[N];
// Thread i accesses particles[i].x → stride of 16 bytes → poor coalescing for single field

// Structure of Arrays (SoA) - GOOD for GPU:
float x[N], y[N], z[N], w[N];
// Thread i accesses x[i] → consecutive → perfect coalescing
```

## KV Cache Memory Management

### Memory Requirements Calculation

```python
def kv_cache_memory(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    batch_size: int,
    dtype_bytes: int = 2,  # 2 for FP16, 1 for FP8, 0.5 for INT4
) -> float:
    """Returns KV cache memory in bytes"""
    # 2 for K and V
    return 2 * num_layers * num_kv_heads * head_dim * seq_len * batch_size * dtype_bytes
```

### KV Cache for Popular Models

| Model | Layers | KV Heads | Head Dim | KV per token (FP16) | KV per token (FP8) |
|-------|--------|----------|----------|--------------------|--------------------|
| LLaMA-7B | 32 | 32 | 128 | 512 KB | 256 KB |
| LLaMA-13B | 40 | 40 | 128 | 800 KB | 400 KB |
| LLaMA-70B | 80 | 8 | 128 | 320 KB | 160 KB |
| Mistral-7B | 32 | 8 | 128 | 128 KB | 64 KB |
| Mixtral-8x7B | 32 | 8 | 128 | 128 KB | 64 KB |
| Qwen2.5-72B | 80 | 8 | 128 | 320 KB | 160 KB |
| DeepSeek-V3 | 61 | MLA | ~512 | ~62 KB* | ~31 KB* |

*DeepSeek-V3 uses MLA with compressed KV, dramatically less KV cache.

### KV Cache at Scale

LLaMA-70B, batch=128, seq_len=4096, FP16:
```
KV cache = 2 * 80 * 8 * 128 * 4096 * 128 * 2 bytes
         = 2 * 80 * 8 * 128 * 4096 * 128 * 2
         = ~85 GB (!)

With FP8 KV cache: ~42 GB
With INT4 KV cache: ~21 GB
```

This is why KV cache quantization and efficient management are critical for high-throughput serving.

## Model Memory Footprint

### Weight Memory
```python
def model_memory(num_params: int, bits_per_param: float) -> float:
    """Returns memory in GB"""
    return num_params * bits_per_param / 8 / 1e9

# Examples:
# LLaMA-7B:  7e9 params
# FP16: 7e9 * 2 / 1e9 = 14 GB
# FP8:  7e9 * 1 / 1e9 = 7 GB
# INT4: 7e9 * 0.5 / 1e9 = 3.5 GB (+ scales ~0.5 GB)
# INT4 AWQ g128: ~4 GB total with scales

# LLaMA-70B: 70e9 params
# FP16: 140 GB (needs 2x H100 or 8x RTX 4090)
# FP8:  70 GB (fits on 1x H100)
# INT4: 35 GB + ~5 GB scales = ~40 GB (fits on 1x A100-80GB)
```

### Total GPU Memory Budget

```
Total GPU Memory = Weights + KV Cache + Activations + Framework Overhead

For serving (inference):
  Activations ≈ small (batch_size * max_seq_len * hidden_dim * ~4 bytes)
  Framework overhead ≈ 500MB - 2GB

Example: LLaMA-70B on H100 (80GB), FP8 weights, FP8 KV cache:
  Weights: 70 GB (FP8)
  Framework: ~1 GB
  Available for KV: 80 - 70 - 1 = 9 GB
  KV per token: 160 KB (FP8)
  Max tokens in cache: 9 GB / 160 KB = ~57,000 tokens
  At avg 2K seq_len: ~28 concurrent sequences

Same model, INT4 weights, FP8 KV cache:
  Weights: ~40 GB (INT4 AWQ)
  Framework: ~1 GB
  Available for KV: 80 - 40 - 1 = 39 GB
  Max tokens: 39 GB / 160 KB = ~249,000 tokens
  At avg 2K seq_len: ~124 concurrent sequences → 4.4x more throughput!
```

### Can It Fit? Quick Reference

| Model | GPU | Precision | Fits? | Max Batch (4K ctx) |
|-------|-----|-----------|-------|--------------------|
| 7B | RTX 3090 (24GB) | FP16 | Yes | ~15 |
| 7B | RTX 3090 (24GB) | INT4 | Yes | ~50 |
| 13B | RTX 3090 (24GB) | INT4 | Yes | ~25 |
| 13B | RTX 4090 (24GB) | FP16 | Barely | ~5 |
| 34B | RTX 4090 (24GB) | INT4 | Yes | ~10 |
| 70B | A100-80GB | INT4 | Yes | ~20 |
| 70B | H100-80GB | FP8 | Tight | ~10 |
| 70B | 2x H100 | FP8/TP2 | Yes | ~60 |
| 70B | 4x H100 | FP8/TP4 | Yes | ~200 |
| 405B | 8x H100 | FP8/TP8 | Yes | ~30 |

## CUDA Memory Management

### Memory Allocation Strategies

```python
# PyTorch caching allocator: caches freed blocks for reuse
# CRITICAL: torch.cuda.memory_allocated() vs torch.cuda.memory_reserved()
# allocated = actually used by tensors
# reserved = total held by caching allocator (includes free cached blocks)

# Tune the allocator:
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join([
    "expandable_segments:True",    # Better memory utilization
    "max_split_size_mb:512",       # Max size for split blocks
    "garbage_collection_threshold:0.8",  # GC when 80% fragmented
])
```

### Memory-Efficient Patterns

```python
# 1. In-place operations (save memory):
x.add_(y)        # in-place add
x.mul_(scale)    # in-place multiply

# 2. Avoid unnecessary intermediate tensors:
# Bad:
a = x @ W1
b = F.gelu(a)
c = b @ W2
# Memory: x, W1, a, b, W2, c all alive

# Good (fused):
c = F.linear(F.gelu(F.linear(x, W1)), W2)
# Or better: let torch.compile fuse these

# 3. Delete tensors when done:
del intermediate_tensor
# Note: doesn't free GPU memory immediately, just marks for caching allocator

# 4. Empty cache (last resort):
torch.cuda.empty_cache()  # Returns cached blocks to CUDA, breaks caching
# Only do this when you need memory for non-PyTorch operations
```

### Pinned Memory for Fast Transfers
```python
# Regular transfer: pageable → GPU
# ~12 GB/s on PCIe Gen4 x16

# Pinned transfer: pinned → GPU
# ~25 GB/s on PCIe Gen4 x16 (2x faster!)

# Allocate pinned memory:
tensor_pinned = torch.empty(size, pin_memory=True)

# Or in DataLoader:
loader = DataLoader(dataset, pin_memory=True)
```

## Activation Checkpointing

### How It Works
```
# Normal: store all activations for backward pass
# Forward: compute and store a1, a2, a3, ..., aN
# Backward: use stored activations to compute gradients
# Memory: O(N) activations

# With checkpointing: store only checkpointed activations
# Forward: compute all, store only every k-th activation
# Backward: recompute non-checkpointed activations from nearest checkpoint
# Memory: O(N/k) stored + O(k) recomputed
# Compute: ~33% more forward compute (one extra forward pass per segment)
```

### PyTorch Implementation
```python
from torch.utils.checkpoint import checkpoint

class TransformerBlock(nn.Module):
    def forward(self, x):
        # Checkpoint this block: don't store intermediate activations
        return checkpoint(self._forward_impl, x, use_reentrant=False)

    def _forward_impl(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
```

### Selective Checkpointing
```python
# Don't checkpoint everything - only the memory-heavy parts
# Attention: QKV projection activations are large (B*S*3*H*D)
# MLP: intermediate activations are large (B*S*intermediate_size)

# Checkpoint attention but not MLP (or vice versa based on profiling)
```

## Multi-GPU Memory Patterns

### Tensor Parallelism Memory Savings
```
# TP splits model weights across GPUs
# For TP=4 on LLaMA-70B (FP16):
# Per-GPU weights: 140 GB / 4 = 35 GB
# KV cache: NOT split (each GPU has full KV for its heads)
# Activations: split for most ops

# Communication: AllReduce after each TP-split layer
# Bandwidth needed: 2 * hidden_dim * batch_size * dtype_size per layer
```

### Pipeline Parallelism Memory
```
# PP splits model layers across GPUs
# For PP=4 on LLaMA-70B (80 layers):
# GPU 0: layers 0-19 (weights for 20 layers)
# GPU 1: layers 20-39
# GPU 2: layers 40-59
# GPU 3: layers 60-79

# Each GPU only loads its layers' weights
# But needs to store activations for micro-batches in flight
# Memory per GPU: weights/4 + activations * num_micro_batches
```

## Memory Optimization Decision Tree

```
1. Model doesn't fit on GPU?
   → Try quantization: FP8 (2x), INT4 (4x compression)
   → Try tensor parallelism across multiple GPUs
   → Try CPU offloading (ZeRO-3 / llama.cpp partial offload)

2. Model fits but KV cache is limiting batch size?
   → Quantize KV cache (FP8 → 2x, INT4 → 4x)
   → Use GQA/MQA model variant (less KV heads)
   → Consider MLA (DeepSeek-style compressed KV)
   → Reduce max_seq_len if possible
   → Use KV cache eviction (H2O, StreamingLLM)

3. Training OOM?
   → Enable activation checkpointing
   → Use gradient accumulation (smaller micro-batch)
   → Mixed precision (FP16/BF16 instead of FP32)
   → Use FSDP / ZeRO to shard optimizer states
   → CPU offload optimizer states

4. Fragmentation issues?
   → Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   → Avoid mixing large/small allocations
   → Pre-allocate KV cache at startup
```
