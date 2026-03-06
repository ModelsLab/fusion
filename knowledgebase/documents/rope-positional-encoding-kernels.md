---
id: rope_positional_encoding_kernels
kind: document
title: RoPE and Positional Encoding Kernel Optimization
category: kernel
summary: Deep dive into RoPE implementation, optimization, and kernel patterns including fused RoPE, NTK-aware scaling, YaRN, and ALiBi alternatives.
tags:
  - rope
  - positional-encoding
  - rotary
  - alibi
  - ntk-scaling
  - yarn
source_ids: []
operators:
  - rope
  - embedding
gpu_families:
  - Ampere
  - Ada
  - Hopper
  - Blackwell
---

## 1. RoPE Mathematical Formulation

### Core Idea

Rotary Position Embedding encodes absolute position information into query and key vectors such that their dot product naturally captures relative position. Given a query vector q at position m and a key vector k at position n, applying RoPE ensures:

```
<RoPE(q, m), RoPE(k, n)> = g(q, k, m - n)
```

The dot product depends only on the relative distance (m - n), not on absolute positions.

### Rotation Matrix Form

For a d-dimensional embedding, RoPE partitions dimensions into d/2 pairs and applies a 2D rotation to each pair. For dimensions (2i, 2i+1):

```
            [ cos(m * theta_i)   -sin(m * theta_i) ] [ q_{2i}   ]
RoPE(q,m) = [ sin(m * theta_i)    cos(m * theta_i) ] [ q_{2i+1} ]
```

The full rotation matrix R(m) is block-diagonal with d/2 such 2x2 rotation blocks:

```
R(m) = diag(R_0(m), R_1(m), ..., R_{d/2-1}(m))
```

### Frequency Basis

Each dimension pair i uses a frequency:

```
theta_i = base^(-2i/d)
```

where `base = 10000` by default (from the original RoFormer paper). This creates a geometric progression of wavelengths from `2*pi` (high frequency, short range) to `2*pi * base` (low frequency, long range).

### Complex Number Interpretation

RoPE has an elegant complex number form. Treating consecutive pairs (q_{2i}, q_{2i+1}) as complex numbers z_i = q_{2i} + j*q_{2i+1}:

```
RoPE(z_i, m) = z_i * e^{j * m * theta_i}
```

This is simply multiplication by a unit complex number -- a rotation in the complex plane. The implementation reduces to:

```
q_rope_{2i}   = q_{2i} * cos(m * theta_i) - q_{2i+1} * sin(m * theta_i)
q_rope_{2i+1} = q_{2i} * sin(m * theta_i) + q_{2i+1} * cos(m * theta_i)
```

### Frequency Spectrum Intuition

Low-frequency dimensions (large i) change slowly with position, encoding coarse long-range relationships. High-frequency dimensions (small i) oscillate rapidly, capturing fine-grained local position. This multi-scale encoding is what makes RoPE effective and why frequency manipulation is key to context extension.

## 2. Naive vs Optimized RoPE Kernel Implementations

### Naive PyTorch Implementation

```python
def rope_naive(x, seq_len, dim, base=10000.0):
    """x: (batch, seq_len, num_heads, head_dim)"""
    # Compute frequency basis
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    # Position indices
    t = torch.arange(seq_len, dtype=torch.float32)
    # Outer product: (seq_len, dim//2)
    angles = torch.outer(t, freqs)
    cos_cached = angles.cos()
    sin_cached = angles.sin()

    # Split into pairs and apply rotation
    x1 = x[..., ::2]   # even indices
    x2 = x[..., 1::2]  # odd indices
    out_even = x1 * cos_cached - x2 * sin_cached
    out_odd  = x1 * sin_cached + x2 * cos_cached
    return torch.stack([out_even, out_odd], dim=-1).flatten(-2)
```

Problems with the naive version: multiple intermediate tensors, non-coalesced memory from stack+flatten, and redundant cos/sin recomputation.

### Optimized Triton Kernel

```python
@triton.jit
def rope_fwd_kernel(
    Q_ptr, COS_ptr, SIN_ptr, OUT_ptr,
    seq_len, num_heads, head_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (batch, seq_pos, head) slice
    pid = tl.program_id(0)
    batch_idx = pid // (seq_len * num_heads)
    rem = pid % (seq_len * num_heads)
    seq_idx = rem // num_heads
    head_idx = rem % num_heads

    half_dim = head_dim // 2

    # Base offset into Q for this (batch, seq, head)
    base_off = (batch_idx * seq_len * num_heads * head_dim
                + seq_idx * num_heads * head_dim
                + head_idx * head_dim)

    # Load cos/sin for this position -- shape (seq_len, half_dim)
    cs_off = seq_idx * half_dim + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < half_dim

    cos_val = tl.load(COS_ptr + cs_off, mask=mask, other=0.0)
    sin_val = tl.load(SIN_ptr + cs_off, mask=mask, other=0.0)

    # Load even and odd elements with stride-2 pattern
    even_off = base_off + tl.arange(0, BLOCK_SIZE) * 2
    odd_off  = base_off + tl.arange(0, BLOCK_SIZE) * 2 + 1

    q_even = tl.load(Q_ptr + even_off, mask=mask, other=0.0)
    q_odd  = tl.load(Q_ptr + odd_off, mask=mask, other=0.0)

    # Apply rotation
    out_even = q_even * cos_val - q_odd * sin_val
    out_odd  = q_even * sin_val + q_odd * cos_val

    # Store results
    tl.store(OUT_ptr + even_off, out_even, mask=mask)
    tl.store(OUT_ptr + odd_off, out_odd, mask=mask)
```

### CUDA Pseudocode (Vectorized)

```cpp
__global__ void rope_kernel(
    const half2* __restrict__ qk,   // interleaved pairs
    const half2* __restrict__ cos_sin, // precomputed (cos, sin) pairs
    half2* __restrict__ out,
    int seq_len, int num_heads, int half_dim
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * num_heads * half_dim;
    if (tid >= total) return;

    int pair_idx = tid % half_dim;
    int seq_idx  = (tid / (num_heads * half_dim)) % seq_len;

    // Load q pair as half2: (q_even, q_odd)
    half2 q = qk[tid];
    // Load (cos, sin) for this position and dimension
    half2 cs = cos_sin[seq_idx * half_dim + pair_idx];

    // Rotation: out = (q.x*cos - q.y*sin, q.x*sin + q.y*cos)
    half2 result;
    result.x = __hfma(q.x, cs.x, __hneg(__hmul(q.y, cs.y)));
    result.y = __hfma(q.x, cs.y, __hmul(q.y, cs.x));

    out[tid] = result;
}
```

Key optimization: using `half2` packs the even/odd pair into a single register and enables vectorized load/store, cutting memory transactions in half.

## 3. Fused RoPE

### Fused with QKV Projection

The most impactful fusion is applying RoPE directly after the QKV linear projection, avoiding a separate kernel launch and intermediate memory round-trip.

```python
@triton.jit
def fused_qkv_rope_kernel(
    X_ptr, W_ptr, COS_ptr, SIN_ptr,
    Q_out_ptr, K_out_ptr, V_out_ptr,
    M, N, K,  # matmul dimensions
    head_dim: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Step 1: standard GEMM tile to produce QKV
    # ... (tiled matmul logic) ...
    # qkv_tile now in registers, shape conceptually (BLOCK_M, head_dim)

    # Step 2: apply RoPE in-register to Q and K portions only
    # V passes through unchanged
    half = head_dim // 2
    for i in range(half):
        cos_val = tl.load(COS_ptr + seq_offset + i)
        sin_val = tl.load(SIN_ptr + seq_offset + i)

        # Rotate Q
        q_even = q_tile[..., 2*i]
        q_odd  = q_tile[..., 2*i+1]
        q_tile[..., 2*i]   = q_even * cos_val - q_odd * sin_val
        q_tile[..., 2*i+1] = q_even * sin_val + q_odd * cos_val

        # Rotate K (same logic)
        k_even = k_tile[..., 2*i]
        k_odd  = k_tile[..., 2*i+1]
        k_tile[..., 2*i]   = k_even * cos_val - k_odd * sin_val
        k_tile[..., 2*i+1] = k_even * sin_val + k_odd * cos_val

    # Step 3: store Q, K (with RoPE), V (without)
```

Benefits of fused RoPE:
- Eliminates one full read+write of Q and K tensors from global memory.
- The rotation is pure ALU and adds negligible compute to a memory-bound GEMM epilogue.
- Reduces kernel launch overhead (one launch instead of three).

### Fused with Flash Attention

In FlashAttention-2 and FlashAttention-3, RoPE can be applied on-the-fly as Q/K tiles are loaded into SRAM, before the dot product. FlashInfer and vLLM implement this pattern:

```
for each Q tile loaded into SRAM:
    apply_rope_in_sram(Q_tile, position_offset)
    for each K tile:
        apply_rope_in_sram(K_tile, position_offset)
        S_tile = Q_tile @ K_tile^T
        ... softmax, accumulate with V ...
```

This avoids ever materializing the rotated Q/K in global memory.

## 4. RoPE Scaling Methods

### 4a. Linear Scaling (Position Interpolation)

The simplest extension method. Scale down all positions by a factor s so that a context of length L*s maps to positions [0, L):

```
theta_i(m) = base^(-2i/d)
m_scaled = m / s
```

For a model trained at 4K context extended to 16K, s = 4. Position 16000 maps to effective position 4000.

Pros: trivial to implement, zero parameter changes.
Cons: compresses the position space, reducing resolution between adjacent tokens. Requires fine-tuning to recover quality.

### 4b. NTK-Aware Scaling

Instead of compressing all frequencies uniformly, NTK-aware scaling adjusts the frequency base to spread the interpolation burden across the frequency spectrum:

```
base_new = base * s^(d / (d - 2))
```

where s is the scaling factor. For s=4, d=128:

```
base_new = 10000 * 4^(128/126) = 10000 * 4.088 = 40880
```

Mathematical justification: the Neural Tangent Kernel theory suggests that high-frequency components are harder to learn, so they should be perturbed less. By increasing the base, low-frequency dimensions (which are already slowly varying) absorb most of the interpolation, while high-frequency dimensions remain nearly unchanged.

```python
def compute_ntk_freqs(dim, base=10000.0, scale=4.0):
    base_new = base * (scale ** (dim / (dim - 2)))
    freqs = 1.0 / (base_new ** (torch.arange(0, dim, 2).float() / dim))
    return freqs
```

### 4c. YaRN (Yet Another RoPE extensioN)

YaRN combines NTK-aware frequency scaling with an attention scaling factor and a per-dimension interpolation strategy. It partitions dimensions into three groups:

1. High-frequency dimensions (small wavelength relative to original context): no interpolation, keep original frequencies.
2. Low-frequency dimensions (wavelength much larger than original context): full linear interpolation.
3. Medium-frequency dimensions: smooth ramp between no interpolation and full interpolation.

```python
def yarn_freqs(dim, base=10000.0, original_max_pos=4096, target_max_pos=131072, beta_fast=32, beta_slow=1):
    scale = target_max_pos / original_max_pos
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    wavelengths = 2 * math.pi / freqs  # wavelength per dimension

    # Compute interpolation ramp
    low = max(math.floor(dim * math.log(original_max_pos / (beta_fast * 2 * math.pi))
                         / (2 * math.log(base))), 0)
    high = min(math.ceil(dim * math.log(original_max_pos / (beta_slow * 2 * math.pi))
                          / (2 * math.log(base))), dim // 2 - 1)

    ramp = torch.zeros(dim // 2)
    for i in range(dim // 2):
        if i < low:
            ramp[i] = 0.0   # no interpolation (high freq)
        elif i > high:
            ramp[i] = 1.0   # full interpolation (low freq)
        else:
            ramp[i] = (i - low) / (high - low)  # smooth transition

    # Apply per-dimension scaling
    freqs_scaled = freqs / scale   # linear interpolation frequencies
    freqs_yarn = freqs * (1 - ramp) + freqs_scaled * ramp

    # Attention scaling factor (sqrt of ratio)
    attn_scale = 0.1 * math.log(scale) + 1.0

    return freqs_yarn, attn_scale
```

YaRN achieves strong results with minimal fine-tuning (often 400-1000 steps) and is the default strategy in many long-context models.

### 4d. Dynamic NTK

Dynamic NTK adjusts the scaling factor at inference time based on the actual sequence length seen so far, rather than using a fixed scale:

```python
def dynamic_ntk_freqs(dim, seq_len, original_max_pos=4096, base=10000.0):
    if seq_len <= original_max_pos:
        # No scaling needed within original context
        return 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

    # Scale dynamically based on how far we exceed training length
    scale = seq_len / original_max_pos
    base_new = base * (scale ** (dim / (dim - 2)))
    return 1.0 / (base_new ** (torch.arange(0, dim, 2).float() / dim))
```

This is useful for autoregressive generation where context grows incrementally and you want to avoid pre-committing to a maximum extended length.

## 5. Long Context Extensions

### How Scaling Enables 4K to 128K+

The fundamental challenge: a model trained with max position 4096 has never seen frequencies corresponding to positions beyond 4096. Extrapolation to unseen positions causes attention score explosion and quality collapse.

Scaling methods address this by ensuring all position-dependent frequencies remain within the trained range:

| Method            | Mechanism                                  | Fine-tune needed | Quality at 128K |
|-------------------|--------------------------------------------|------------------|-----------------|
| Linear Scaling    | Compress all positions by factor s         | Yes (1-2K steps) | Moderate        |
| NTK-Aware         | Increase base to spread interpolation      | Minimal          | Good            |
| YaRN              | Per-dimension ramp + attention scaling     | Minimal (400+)   | Very good       |
| Dynamic NTK       | Runtime base adjustment                    | No               | Good            |
| Trained from scratch | Large context during pretraining        | N/A (expensive)  | Best            |

### Practical Considerations

- KV cache memory grows linearly with context: at 128K with GQA (8 KV heads), head_dim=128, bf16, each layer needs `128K * 8 * 128 * 2 bytes = 256MB`. A 32-layer model: 8GB just for KV cache.
- Attention compute grows quadratically: Flash Attention is essential.
- RoPE scaling is nearly free computationally; the bottleneck is always memory and attention.

## 6. ALiBi as Alternative

### Concept

Attention with Linear Biases (ALiBi) adds no positional embedding to token representations. Instead, it adds a static linear bias to attention scores based on distance:

```
attention_score(i, j) = q_i . k_j - m * |i - j|
```

where m is a head-specific slope. For h heads, slopes are set to geometric sequence:

```
m_k = 2^(-8k/h)    for k = 1, 2, ..., h
```

### Implementation

```python
def alibi_bias(seq_len, num_heads):
    """Returns bias tensor of shape (num_heads, seq_len, seq_len)"""
    slopes = 2.0 ** (-8.0 * torch.arange(1, num_heads + 1) / num_heads)
    positions = torch.arange(seq_len)
    # Relative distance matrix
    dist = positions.unsqueeze(0) - positions.unsqueeze(1)  # (seq, seq)
    # Causal: only attend to past, so use negative distances
    dist = dist.float()
    # Apply slopes per head
    bias = slopes.unsqueeze(1).unsqueeze(2) * dist.unsqueeze(0)  # (heads, seq, seq)
    return bias
```

### ALiBi vs RoPE Comparison

| Property              | RoPE                           | ALiBi                       |
|-----------------------|--------------------------------|-----------------------------|
| Embedding location    | Applied to Q and K vectors     | Added to attention scores   |
| Parameters            | None (deterministic)           | None (fixed slopes)         |
| Extrapolation         | Poor without scaling           | Moderate out-of-the-box     |
| Flash Attention compat| Native in most implementations | Requires bias support       |
| Adoption              | LLaMA, Mistral, Qwen, Gemma   | BLOOM, MPT                  |
| KV cache reuse        | Position must match            | Bias recomputed cheaply     |

ALiBi has fallen out of favor for frontier models. RoPE with YaRN scaling provides better long-context quality while remaining compatible with the Flash Attention ecosystem.

## 7. Performance Considerations

### Precomputing cos/sin Tables

The cos/sin values depend only on position and dimension, not on the data. Precompute once and cache:

```python
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position=8192, base=10000.0):
        super().__init__()
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("freqs", freqs, persistent=False)
        self._build_cache(max_position)

    def _build_cache(self, max_pos):
        t = torch.arange(max_pos, dtype=self.freqs.dtype)
        angles = torch.outer(t, self.freqs)
        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)

    def forward(self, x, seq_len):
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]
```

Table size is small: for dim=128, max_pos=131072, each table is `131072 * 64 * 2 bytes (bf16) = 16MB`. Negligible compared to model weights.

### Half-Precision Issues

Computing cos/sin in fp16 introduces visible error for large position indices. The product `m * theta_i` can overflow fp16 range (max ~65504) when m is large. Best practice:

- Compute frequencies and angles in fp32.
- Store cos/sin tables in bf16 or fp16 (the values are in [-1, 1], safe for half precision).
- Apply the rotation in the compute precision of the model (bf16/fp16).

### Memory Access Patterns

RoPE's main cost is memory bandwidth, not compute (4 multiplies + 2 adds per pair). Optimization targets:

- Coalesced access: ensure threads in a warp access contiguous memory. The interleaved (even/odd) layout used by LLaMA is suboptimal; the "half-rotated" layout (first half = x1, second half = x2) enables contiguous loads.
- Vectorized loads: use 128-bit loads (float4, half8) to maximize memory throughput.
- Minimize global memory trips: fuse RoPE with adjacent operations.

## 8. Kernel Optimization Patterns

### Vectorized Loads for RoPE

```cpp
// Load 8 half values at once (128-bit load)
// Process 4 rotation pairs per thread
__global__ void rope_vec4_kernel(
    const half* __restrict__ q,
    const half* __restrict__ cos_table,
    const half* __restrict__ sin_table,
    half* __restrict__ out,
    int stride_seq, int half_dim
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Each thread processes 4 consecutive pairs
    int pair_start = tid * 4;
    if (pair_start >= half_dim) return;

    int seq_idx = blockIdx.y;
    int head_idx = blockIdx.z;

    // Vectorized load: 4 pairs = 8 halfs = 128 bits
    float4 q_vec = *reinterpret_cast<const float4*>(
        &q[seq_idx * stride_seq + head_idx * half_dim * 2 + pair_start * 2]);
    float2 cs0 = *reinterpret_cast<const float2*>(
        &cos_table[seq_idx * half_dim + pair_start]);
    float2 cs1 = *reinterpret_cast<const float2*>(
        &sin_table[seq_idx * half_dim + pair_start]);

    // Unpack and rotate...
    // Store with vectorized write
}
```

### Fusing RoPE with Transpose

Many architectures store QKV as `(batch, seq, 3, num_heads, head_dim)` and need `(batch, num_heads, seq, head_dim)` for attention. Fuse the transpose with RoPE:

```python
@triton.jit
def rope_transpose_kernel(
    QKV_ptr,        # input: (batch, seq, 3, num_heads, head_dim)
    Q_out_ptr,      # output: (batch, num_heads, seq, head_dim) with RoPE
    K_out_ptr,      # output: (batch, num_kv_heads, seq, head_dim) with RoPE
    V_out_ptr,      # output: (batch, num_kv_heads, seq, head_dim) no RoPE
    COS_ptr, SIN_ptr,
    # dimensions...
):
    # Load from (batch, seq, qkv_idx, head, dim) layout
    # Apply RoPE to Q and K in registers
    # Store to (batch, head, seq, dim) layout
    # Single kernel replaces: split + rope + transpose
    pass
```

This pattern eliminates two intermediate buffers and three separate kernel launches.

### Batch Processing for Decode

During autoregressive decoding, each request may be at a different position. Use a position index array:

```python
@triton.jit
def rope_decode_kernel(
    Q_ptr, COS_ptr, SIN_ptr, POS_ptr, OUT_ptr,
    batch_size, num_heads, head_dim: tl.constexpr,
    HALF_DIM: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    # Each sequence in the batch has its own position
    pos = tl.load(POS_ptr + batch_idx)

    dim_offs = tl.arange(0, HALF_DIM)
    cos_val = tl.load(COS_ptr + pos * HALF_DIM + dim_offs)
    sin_val = tl.load(SIN_ptr + pos * HALF_DIM + dim_offs)

    base = batch_idx * num_heads * head_dim + head_idx * head_dim
    q_even = tl.load(Q_ptr + base + dim_offs * 2)
    q_odd  = tl.load(Q_ptr + base + dim_offs * 2 + 1)

    tl.store(OUT_ptr + base + dim_offs * 2,     q_even * cos_val - q_odd * sin_val)
    tl.store(OUT_ptr + base + dim_offs * 2 + 1, q_even * sin_val + q_odd * cos_val)
```

## 9. Implementation in Major Frameworks

### vLLM

vLLM applies RoPE inside the paged attention kernel. Key design choices:
- cos/sin tables precomputed on initialization and stored on GPU.
- During prefill, a fused Triton kernel applies RoPE to the entire Q/K tensor.
- During decode, RoPE is applied per-token as part of the cache write operation.
- Supports NTK scaling and YaRN via configurable `rope_scaling` dict.
- Source: `vllm/model_executor/layers/rotary_embedding.py`

### llama.cpp

llama.cpp implements RoPE in the `ggml` tensor library:
- Custom C/CUDA kernels for each quantization format (Q4_0, Q8_0, fp16, etc.).
- Supports all scaling methods via `llama_rope_scaling_type` enum.
- The CUDA kernel `rope_norm` / `rope_neox` handles two RoPE layout variants:
  - "Normal" layout: pairs are (dim[0], dim[1]), (dim[2], dim[3])...
  - "NeoX" layout: pairs are (dim[0], dim[d/2]), (dim[1], dim[d/2+1])...
- YaRN implemented with per-dimension frequency correction factors and attention scaling.
- Source: `ggml/src/ggml-cuda/rope.cu`

### HuggingFace Transformers

Transformers provides modular RoPE classes:
- `RotaryEmbedding`: base implementation with cos/sin caching.
- `LinearScalingRotaryEmbedding`: position interpolation.
- `DynamicNTKScalingRotaryEmbedding`: dynamic base adjustment.
- Applied in model-specific attention modules (e.g., `LlamaAttention.forward`).
- RoPE is a separate operation, not fused (relies on torch.compile for fusion).
- Source: `transformers/models/llama/modeling_llama.py`

### FlashInfer

FlashInfer provides the most aggressively optimized RoPE path:
- RoPE applied inside the Flash Attention kernel during Q/K tile loading.
- Supports both prefill and decode with paged KV cache.
- CUDA kernels use `half2` vectorized loads with in-register rotation.
- Eliminates all intermediate RoPE buffers.
- Source: `flashinfer/rope.py`, `include/flashinfer/pos_enc.cuh`

## 10. Comparison Table of Positional Encoding Methods

| Method     | Type           | Params | Relative PE | Max Context    | Extrapolation | Compute Cost  | Adoption              |
|------------|----------------|--------|-------------|----------------|---------------|---------------|-----------------------|
| Sinusoidal | Additive       | 0      | No          | Fixed          | Poor          | Negligible    | Original Transformer  |
| Learned    | Additive       | L*d    | No          | Fixed          | None          | Negligible    | GPT-2, BERT           |
| RoPE       | Multiplicative | 0      | Yes         | Scalable       | Poor (raw)    | Low           | LLaMA, Mistral, Qwen  |
| RoPE+NTK   | Multiplicative | 0      | Yes         | ~8x training   | Good          | Low           | CodeLlama             |
| RoPE+YaRN  | Multiplicative | 0      | Yes         | ~32x training  | Very good     | Low           | Nous, Yi, DeepSeek    |
| ALiBi      | Attention bias | 0      | Yes         | Moderate extrap| Moderate      | Low           | BLOOM, MPT            |
| CoPE       | Contextual     | Small  | Yes         | Scalable       | Good          | Moderate      | Research              |
| FIRE       | Functional     | Small  | Yes         | Scalable       | Good          | Moderate      | Research              |

### Performance Impact Summary

RoPE is almost never the bottleneck in transformer inference:
- Compute: 4 FLOPs per element (2 muls + 1 add for each of 2 outputs). For head_dim=128, that is 512 FLOPs per (position, head). Negligible compared to attention (seq_len * head_dim FLOPs).
- Memory: one read + one write of Q and K. Dominates over compute for small batch sizes, but dwarfed by attention and FFN memory traffic.
- Latency: unfused RoPE adds 5-15us per layer on modern GPUs. Fused RoPE adds effectively zero.

The real optimization priority order for positional encoding:
1. Fuse RoPE with adjacent operations (QKV projection or attention) to eliminate memory round-trips.
2. Precompute cos/sin tables; never recompute per forward pass.
3. Use the half-rotated dimension layout (not interleaved) for coalesced memory access.
4. Compute frequency basis in fp32, store and apply in model precision.
5. For long context, choose YaRN over linear scaling for better quality-per-fine-tuning-step.
