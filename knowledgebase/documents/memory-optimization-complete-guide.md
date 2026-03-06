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

## Practical Memory Recipes

### Recipe 1: Measure Your Model's Memory Usage

Use this script to load any HuggingFace model and get a complete memory breakdown:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def measure_model_memory(model_name: str, dtype=torch.float16, device="cuda"):
    """Load a model and measure exactly where GPU memory goes."""

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    mem_before = torch.cuda.memory_allocated()
    print(f"Baseline GPU memory: {mem_before / 1e9:.2f} GB")

    # --- Phase 1: Load weights ---
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device
    )
    model.eval()

    mem_after_load = torch.cuda.memory_allocated()
    weight_memory = mem_after_load - mem_before
    print(f"Weight memory:       {weight_memory / 1e9:.2f} GB")
    print(f"Parameter count:     {sum(p.numel() for p in model.parameters()) / 1e9:.2f} B")

    # --- Phase 2: Run a forward pass to measure activations ---
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(device)

    torch.cuda.reset_peak_memory_stats()
    mem_before_fwd = torch.cuda.memory_allocated()

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)

    mem_after_fwd = torch.cuda.memory_allocated()
    peak_during_fwd = torch.cuda.max_memory_allocated()

    activation_memory = peak_during_fwd - mem_before_fwd
    print(f"Activation peak:     {activation_memory / 1e6:.2f} MB (seq_len={inputs['input_ids'].shape[1]})")

    # --- Phase 3: Measure KV cache ---
    past_kv = outputs.past_key_values
    kv_memory = 0
    for layer_kv in past_kv:
        for tensor in layer_kv:
            kv_memory += tensor.nelement() * tensor.element_size()
    print(f"KV cache memory:     {kv_memory / 1e6:.2f} MB (for {inputs['input_ids'].shape[1]} tokens)")
    print(f"KV per token:        {kv_memory / inputs['input_ids'].shape[1] / 1024:.2f} KB")

    # --- Phase 4: Full summary ---
    print("\n" + "=" * 60)
    print(torch.cuda.memory_summary(abbreviated=True))

    return {
        "weight_gb": weight_memory / 1e9,
        "activation_peak_mb": activation_memory / 1e6,
        "kv_per_token_kb": kv_memory / inputs["input_ids"].shape[1] / 1024,
    }


# Usage:
# stats = measure_model_memory("meta-llama/Llama-3.1-8B")
```

**Interpreting `torch.cuda.memory_summary()` output:**

```
|                   | Cur Usage  | Peak Usage | Allocs  |
| Allocated memory  |  14.02 GiB |  14.25 GiB |    1842 |   <-- actual tensor data
| Active memory     |  14.02 GiB |  14.25 GiB |    1842 |   <-- same minus freed-not-returned
| Requested memory  |  14.00 GiB |  14.23 GiB |    1842 |   <-- what you asked for (before alignment)
| GPU reserved mem  |  14.50 GiB |  14.50 GiB |      12 |   <-- caching allocator total pool
| Non-releasable    |   0.48 GiB |   0.83 GiB |      34 |   <-- fragmented, can't return to CUDA
```

- **Allocated vs Reserved gap**: memory the caching allocator holds but is not actively used. Normal.
- **Non-releasable**: fragmented blocks. If this is large (>10% of reserved), you have fragmentation issues -- set `expandable_segments:True`.
- **Peak vs Current**: shows how much transient memory your forward pass needs.

**"Can model X fit on GPU Y?" calculation:**

```python
def can_it_fit(
    param_billions: float,
    bits_per_param: float,   # 16 for FP16, 8 for FP8, 4 for INT4
    gpu_memory_gb: float,
    batch_size: int = 1,
    seq_len: int = 2048,
    kv_per_token_kb: float = 0,  # look up from table above
) -> dict:
    weight_gb = param_billions * bits_per_param / 8
    kv_gb = kv_per_token_kb * seq_len * batch_size / 1024 / 1024
    overhead_gb = 1.5  # framework, CUDA context, caching allocator
    activation_gb = 0.5 * batch_size  # rough estimate for inference

    total_gb = weight_gb + kv_gb + activation_gb + overhead_gb
    fits = total_gb <= gpu_memory_gb
    headroom = gpu_memory_gb - total_gb

    print(f"Weights:      {weight_gb:.1f} GB")
    print(f"KV cache:     {kv_gb:.2f} GB")
    print(f"Activations:  {activation_gb:.2f} GB (estimate)")
    print(f"Overhead:     {overhead_gb:.1f} GB")
    print(f"TOTAL:        {total_gb:.1f} GB")
    print(f"GPU:          {gpu_memory_gb:.0f} GB")
    print(f"{'FITS' if fits else 'DOES NOT FIT'} (headroom: {headroom:+.1f} GB)")
    return {"fits": fits, "total_gb": total_gb, "headroom_gb": headroom}

# Example:
# can_it_fit(param_billions=70, bits_per_param=4, gpu_memory_gb=80,
#            batch_size=32, seq_len=4096, kv_per_token_kb=320)
```

### Recipe 2: Memory Calculator

```python
def memory_calculator(
    model_name: str,
    num_params_b: float,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    hidden_dim: int,
    intermediate_size: int,
    weight_dtype: str = "fp16",     # fp32, fp16, bf16, fp8, int4
    kv_dtype: str = "fp16",         # fp16, fp8, int4
    batch_size: int = 1,
    seq_len: int = 2048,
    training: bool = False,
    optimizer: str = "adamw",       # adamw, sgd, adam-8bit
):
    """Complete memory breakdown for any model configuration."""

    dtype_bytes = {"fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1, "int4": 0.5}
    w_bytes = dtype_bytes[weight_dtype]
    kv_bytes = dtype_bytes[kv_dtype]

    # 1. Weight memory
    weight_mem = num_params_b * 1e9 * w_bytes
    # INT4 has quantization scales (~10% overhead for group_size=128)
    if weight_dtype == "int4":
        weight_mem *= 1.1

    # 2. KV cache memory
    kv_per_token = 2 * num_layers * num_kv_heads * head_dim * kv_bytes
    kv_mem = kv_per_token * seq_len * batch_size

    # 3. Activation memory (inference)
    # Per-layer: input (B*S*H) + attention scores (B*heads*S*S) + MLP intermediate
    act_per_layer = (
        batch_size * seq_len * hidden_dim * 2         # input + residual
        + batch_size * seq_len * intermediate_size * 2 # MLP up + gate
    )
    activation_mem = act_per_layer * num_layers * 2  # 2 bytes for fp16 activations
    # For inference with no grad, only need ~1 layer's activations at a time
    if not training:
        activation_mem = act_per_layer * 2  # only 1 layer active

    # 4. Optimizer states (training only)
    optimizer_mem = 0
    if training:
        if optimizer == "adamw":
            # fp32 master weights + fp32 momentum + fp32 variance
            optimizer_mem = num_params_b * 1e9 * (4 + 4 + 4)  # 12 bytes/param
        elif optimizer == "sgd":
            optimizer_mem = num_params_b * 1e9 * 4  # momentum only
        elif optimizer == "adam-8bit":
            optimizer_mem = num_params_b * 1e9 * (4 + 1 + 1)  # fp32 master + 8-bit states

    # 5. Gradient memory (training only)
    gradient_mem = 0
    if training:
        gradient_mem = num_params_b * 1e9 * 2  # fp16/bf16 gradients

    # 6. Framework overhead
    overhead = 1.5 * 1e9  # ~1.5 GB for CUDA context, PyTorch, etc.

    total = weight_mem + kv_mem + activation_mem + optimizer_mem + gradient_mem + overhead

    # Print table
    print(f"\n{'=' * 65}")
    print(f"  Memory Breakdown: {model_name}")
    print(f"  {weight_dtype.upper()} weights | {kv_dtype.upper()} KV | batch={batch_size} | seq={seq_len}")
    print(f"  {'Training' if training else 'Inference'} mode")
    print(f"{'=' * 65}")
    print(f"  {'Component':<30} {'Memory':>10} {'% Total':>10}")
    print(f"  {'-' * 50}")

    components = [
        ("Model weights", weight_mem),
        ("KV cache", kv_mem),
        ("Activations", activation_mem),
    ]
    if training:
        components.append(("Optimizer states", optimizer_mem))
        components.append(("Gradients", gradient_mem))
    components.append(("Framework overhead", overhead))

    for name, mem in components:
        pct = mem / total * 100
        if mem >= 1e9:
            print(f"  {name:<30} {mem / 1e9:>8.2f} GB {pct:>8.1f}%")
        else:
            print(f"  {name:<30} {mem / 1e6:>8.1f} MB {pct:>8.1f}%")

    print(f"  {'-' * 50}")
    print(f"  {'TOTAL':<30} {total / 1e9:>8.2f} GB {'100.0':>9}%")
    print(f"{'=' * 65}\n")

    return {"total_gb": total / 1e9, "weight_gb": weight_mem / 1e9, "kv_gb": kv_mem / 1e9}


# --- Example usage for popular models ---

# LLaMA 3.1 8B
memory_calculator("LLaMA-3.1-8B", num_params_b=8.03, num_layers=32,
    num_kv_heads=8, head_dim=128, hidden_dim=4096, intermediate_size=14336,
    weight_dtype="fp16", batch_size=16, seq_len=4096)

# LLaMA 3.1 70B
memory_calculator("LLaMA-3.1-70B", num_params_b=70.6, num_layers=80,
    num_kv_heads=8, head_dim=128, hidden_dim=8192, intermediate_size=28672,
    weight_dtype="int4", kv_dtype="fp8", batch_size=32, seq_len=4096)

# Qwen2.5-72B
memory_calculator("Qwen2.5-72B", num_params_b=72.7, num_layers=80,
    num_kv_heads=8, head_dim=128, hidden_dim=8192, intermediate_size=29568,
    weight_dtype="int4", kv_dtype="fp8", batch_size=32, seq_len=4096)
```

### Recipe 3: Reduce Memory Step-by-Step

A complete worked example showing how to take LLaMA 70B from requiring a multi-GPU setup down to a single consumer-grade GPU:

```
=== LLaMA 70B Memory Reduction Roadmap ===

Starting point: FP16 (no optimization)
  Weights:      140.0 GB
  KV cache:      10.0 GB (batch=1, seq=4096, FP16, 320 KB/token)
  Activations:    ~0.5 GB
  Overhead:       ~1.5 GB
  TOTAL:        ~152.0 GB
  Requires:     2x H100 80GB with tensor parallelism

Step 1: FP8 weight quantization (NVIDIA FP8 or llm-compressor)
  Weights:       70.0 GB   (-70 GB, 2x compression)
  KV cache:      10.0 GB
  TOTAL:         ~82.0 GB
  Requires:     1x H100 80GB (tight) or 2x A100 40GB
  Quality:      <0.5% perplexity increase on most benchmarks

Step 2: INT4 AWQ quantization (autoawq, group_size=128)
  Weights:       ~38.5 GB  (-101.5 GB from baseline, ~3.6x compression)
  KV cache:      10.0 GB
  TOTAL:         ~50.5 GB
  Requires:     1x A100 80GB (comfortable) or 1x A6000 48GB (tight)
  Quality:      ~1-2% perplexity increase, minimal task degradation

Step 3: FP8 KV cache (vLLM --kv-cache-dtype fp8)
  Weights:       ~38.5 GB
  KV cache:       5.0 GB   (-5 GB, 50% KV reduction)
  TOTAL:         ~45.5 GB
  Requires:     1x A100 40GB (fits!) or 1x RTX 4090 (need lower batch)
  Quality:      negligible impact on output quality

Step 4: Activation checkpointing (for training/fine-tuning)
  Weights:       ~38.5 GB
  KV cache:       5.0 GB
  Activations:   ~0.2 GB   (-60% activation memory)
  TOTAL:         ~45.2 GB
  Trade-off:    ~33% slower forward pass due to recomputation

Step 5: INT4 GPTQ weights + INT4 KV cache (aggressive)
  Weights:       ~38.5 GB
  KV cache:       2.5 GB   (-75% from FP16 baseline)
  TOTAL:         ~43.0 GB
  Requires:     1x A100 40GB with room for batch=8+
  Quality:      monitor carefully -- INT4 KV can degrade long-context tasks

=== Summary Table ===

| Step | Technique                 | Weight GB | KV GB | Total GB | Fits On                |
|------|---------------------------|-----------|-------|----------|------------------------|
| 0    | FP16 baseline             | 140.0     | 10.0  | ~152     | 2x H100 80GB           |
| 1    | FP8 weights               |  70.0     | 10.0  |  ~82     | 1x H100 80GB           |
| 2    | INT4 AWQ weights          |  38.5     | 10.0  |  ~50     | 1x A100 80GB           |
| 3    | + FP8 KV cache            |  38.5     |  5.0  |  ~45     | 1x A100 40GB           |
| 4    | + Activ. checkpointing    |  38.5     |  5.0  |  ~45     | 1x A100 40GB (train)   |
| 5    | + INT4 KV cache           |  38.5     |  2.5  |  ~43     | 1x A100 40GB + batch   |
```

### Recipe 4: Monitor Memory During Inference

**Track peak memory and allocation patterns:**

```python
import torch
import time
from contextlib import contextmanager

@contextmanager
def track_gpu_memory(label=""):
    """Context manager to track GPU memory for a code block."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    mem_start = torch.cuda.memory_allocated()
    t_start = time.perf_counter()

    yield

    torch.cuda.synchronize()
    t_end = time.perf_counter()
    mem_end = torch.cuda.memory_allocated()
    mem_peak = torch.cuda.max_memory_allocated()

    print(f"[{label}]")
    print(f"  Duration:     {t_end - t_start:.3f}s")
    print(f"  Mem start:    {mem_start / 1e9:.3f} GB")
    print(f"  Mem end:      {mem_end / 1e9:.3f} GB")
    print(f"  Mem peak:     {mem_peak / 1e9:.3f} GB")
    print(f"  Mem delta:    {(mem_end - mem_start) / 1e6:+.1f} MB")
    print(f"  Peak above start: {(mem_peak - mem_start) / 1e6:.1f} MB")


# Usage:
# with track_gpu_memory("prefill batch=16"):
#     outputs = model.generate(**inputs, max_new_tokens=1)
#
# with track_gpu_memory("decode 256 tokens"):
#     outputs = model.generate(**inputs, max_new_tokens=256)
```

**Detect memory leaks across requests:**

```python
def detect_memory_leak(model, tokenizer, prompt, num_iterations=20, device="cuda"):
    """Run repeated inference and check if memory grows over time."""
    memories = []
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Warm-up
    for _ in range(3):
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=50)

    torch.cuda.empty_cache()

    for i in range(num_iterations):
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated()

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50)

        del outputs
        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated()
        memories.append(mem_after)

        if i % 5 == 0:
            print(f"  Iter {i:3d}: {mem_after / 1e6:.1f} MB allocated")

    # Check for growth
    growth = memories[-1] - memories[0]
    avg_growth_per_iter = growth / num_iterations

    if abs(avg_growth_per_iter) > 1e6:  # more than 1 MB/iter
        print(f"\nWARNING: Memory leak detected!")
        print(f"  Growth: {growth / 1e6:.1f} MB over {num_iterations} iterations")
        print(f"  Rate:   {avg_growth_per_iter / 1e6:.2f} MB/iteration")
        print(f"  Common causes:")
        print(f"    - KV cache not being freed (check model.generate kwargs)")
        print(f"    - Tensors accidentally stored in a list/dict")
        print(f"    - Gradient computation enabled (missing torch.no_grad)")
    else:
        print(f"\nNo memory leak detected. Stable at {memories[-1] / 1e6:.1f} MB")
```

**vLLM memory monitoring flags:**

```bash
# Launch vLLM with memory visibility
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --dtype float16 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 4096 \
    --enable-prefix-caching \
    2>&1 | tee vllm_server.log

# Key vLLM startup log lines to watch:
#   "GPU memory: XX.XX GiB total, XX.XX GiB free"
#   "Maximum number of batched tokens: XXXXX"
#   "Number of GPU blocks: XXXX, Number of CPU blocks: XXXX"
#   "KV cache memory: XX.XX GiB"

# Monitor during serving:
# GET /metrics endpoint exposes:
#   vllm:gpu_cache_usage_perc    - KV cache utilization (target: 50-90%)
#   vllm:num_requests_running    - concurrent requests
#   vllm:num_requests_waiting    - queued requests (>0 means memory-bound)

# Useful environment variables:
export VLLM_LOGGING_LEVEL=DEBUG           # verbose memory logs
export CUDA_VISIBLE_DEVICES=0             # restrict to specific GPU
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"  # reduce fragmentation

# Check memory pressure in real time:
watch -n 1 nvidia-smi                     # basic
nvidia-smi dmon -s um -d 1                # utilization + memory every 1s
```

### "Can It Fit?" Quick Reference

This table covers popular models across popular GPUs for inference. "YES" means fits with comfortable headroom for at least batch=1 at 4K context. "TIGHT" means it fits but with minimal headroom for KV cache. "QUANT" means it requires quantization (INT4/AWQ) to fit. "NO" means it does not fit even with INT4 quantization.

| Model | Params | RTX 3090 (24 GB) | RTX 4090 (24 GB) | A100 40 GB | A100 80 GB | H100 80 GB |
|-------|--------|-------------------|-------------------|------------|------------|------------|
| **Mistral 7B** | 7.2B | FP16: YES | FP16: YES | FP16: YES | FP16: YES | FP16: YES |
| | | INT4: YES (batch 50+) | INT4: YES (batch 50+) | INT4: YES (batch 100+) | INT4: YES (batch 200+) | INT4: YES (batch 200+) |
| **LLaMA 3.1 8B** | 8.0B | FP16: YES | FP16: YES | FP16: YES | FP16: YES | FP16: YES |
| | | INT4: YES (batch 40+) | INT4: YES (batch 40+) | INT4: YES (batch 80+) | INT4: YES (batch 200+) | INT4: YES (batch 200+) |
| **Mixtral 8x7B** | 46.7B | FP16: NO | FP16: NO | FP16: TIGHT | FP16: YES | FP16: YES |
| | | INT4: YES (batch 1-4) | INT4: YES (batch 1-4) | INT4: YES (batch 20+) | INT4: YES (batch 60+) | INT4: YES (batch 60+) |
| **LLaMA 3.1 70B** | 70.6B | FP16: NO | FP16: NO | FP16: NO | FP16: TIGHT | FP16: TIGHT |
| | | INT4: TIGHT (batch 1) | INT4: TIGHT (batch 1) | INT4: YES (batch 1-4) | INT4: YES (batch 20+) | FP8: YES (batch 10+) |
| **Qwen2.5 72B** | 72.7B | FP16: NO | FP16: NO | FP16: NO | FP16: TIGHT | FP16: TIGHT |
| | | INT4: TIGHT (batch 1) | INT4: TIGHT (batch 1) | INT4: YES (batch 1-4) | INT4: YES (batch 20+) | FP8: YES (batch 10+) |
| **LLaMA 3.1 405B** | 405B | ANY: NO | ANY: NO | ANY: NO | INT4: NO (need 4+) | FP8: NO (need 8x) |
| | | -- | -- | -- | 4x A100 80: INT4 YES | 8x H100: FP8 YES |

**Multi-GPU configurations for large models:**

| Model | Configuration | Precision | Batch Capacity (4K ctx) |
|-------|--------------|-----------|------------------------|
| LLaMA 70B | 2x RTX 4090 (TP=2) | INT4 AWQ | batch 8-12 |
| LLaMA 70B | 2x A100 80GB (TP=2) | FP16 | batch 30-40 |
| LLaMA 70B | 4x H100 (TP=4) | FP8 | batch 150-200 |
| Mixtral 8x7B | 2x RTX 4090 (TP=2) | INT4 AWQ | batch 10-15 |
| LLaMA 405B | 8x H100 (TP=8) | FP8 | batch 20-30 |
| LLaMA 405B | 16x H100 (TP=8, PP=2) | FP8 | batch 60-80 |

**How to read this table:**
- Check your model row and GPU column
- If FP16 shows YES, no quantization needed
- If only INT4 shows YES, you must quantize (use AutoAWQ or AutoGPTQ)
- "batch N+" means you can serve N concurrent sequences at 4096 context length
- For longer contexts (8K, 16K, 32K), divide batch capacity roughly proportionally
- Multi-GPU setups use tensor parallelism (TP) which splits weights evenly but adds inter-GPU communication overhead
