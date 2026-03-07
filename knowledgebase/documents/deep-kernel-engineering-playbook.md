---
id: deep-kernel-engineering-playbook
kind: document
title: "Deep Kernel Engineering Playbook: From Profile to Production"
category: kernel-engineering
summary: Step-by-step methodology for the AI agent to write, optimize, and integrate custom CUDA/Triton/CuTe kernels for every hot operator in model inference
tags:
  - kernel-engineering
  - triton
  - cuda
  - cute
  - optimization
  - profiling
gpu_families:
  - Ampere
  - Ada
  - Hopper
  - Blackwell
workloads:
  - prefill
  - decode
  - serving
operators:
  - attention
  - matmul
  - rmsnorm
  - layernorm
  - softmax
  - rope
  - silu
  - geglu
  - embedding
  - lm-head
  - moe-routing
  - kv-cache
precision:
  - fp32
  - bf16
  - fp16
  - fp8
  - int8
  - int4
backends:
  - triton
  - cuda
  - cute_dsl
  - cutlass
---

## Purpose

This playbook gives the agent a complete, executable methodology for writing custom GPU kernels that replace default operator implementations in LLM inference. It covers profiling, decomposition, kernel authoring, autotuning, correctness verification, and integration into serving runtimes. Every section is designed to be followed literally, not interpreted loosely.

---

## Phase 1: Operator-Level Decomposition

### 1.1 Decomposing a Transformer Into Operators

A single transformer decoder layer consists of the following discrete operators, in execution order:

```
input_hidden_states
  |
  +-- RMSNorm (pre-attention norm)
  +-- Q/K/V Projection (3x Linear or fused QKV Linear)
  +-- RoPE (applied to Q and K)
  +-- Attention (dot-product, softmax, value aggregation)
  +-- Output Projection (Linear)
  +-- Residual Add
  +-- RMSNorm (pre-FFN norm)
  +-- Gate Projection (Linear)
  +-- Up Projection (Linear)
  +-- SiLU Activation (applied to gate)
  +-- Element-wise Multiply (gate * up)
  +-- Down Projection (Linear)
  +-- Residual Add
  |
output_hidden_states
```

For MoE models (Mixtral, DeepSeek-V3, DBRX), the FFN block becomes:

```
  +-- RMSNorm (pre-FFN norm)
  +-- Router Linear (hidden_dim -> num_experts)
  +-- Top-K Gating + Softmax
  +-- Expert Dispatch (permute tokens to experts)
  +-- Per-Expert FFN (gate_proj, up_proj, SiLU, down_proj) via Grouped GEMM
  +-- Expert Combine (un-permute and weighted sum)
```

Additionally, the first layer is preceded by an Embedding lookup and the last layer is followed by a final RMSNorm and LM Head (Linear to vocab size).

### 1.2 Hot Operators by Workload

| Operator | Prefill (long seq) | Decode (single token) | Bottleneck Type |
|---|---|---|---|
| Attention | 40-60% of time | 25-40% of time | Compute (prefill), Memory (decode) |
| GEMM (QKV, O, Gate, Up, Down) | 30-45% | 40-55% | Compute (large batch), Memory (small batch) |
| RMSNorm / LayerNorm | 2-5% | 5-10% | Memory |
| RoPE | 1-2% | 2-4% | Memory |
| SiLU / GeGLU activation | <1% | 1-3% | Memory |
| Residual Add | <1% | 2-4% | Memory |
| Softmax (standalone) | 1-3% | 1-2% | Memory |
| Embedding | <1% (amortized) | <1% | Memory |
| LM Head | 2-5% | 5-15% | Compute (large vocab) |
| KV Cache append | 1-2% | 3-8% | Memory |
| MoE Routing | N/A or 1-2% | 2-5% | Memory |

### 1.3 Reading a Profiler Trace

#### nsys (NVIDIA Nsight Systems) -- Timeline-Level

Run a profile:

```bash
nsys profile --trace=cuda,nvtx,osrt --output=model_trace \
  python run_inference.py --prompt "Hello" --max_tokens 128
```

Key things to look for in the nsys report:

```bash
nsys stats --report cuda_gpu_kern_sum model_trace.nsys-rep
```

This produces a table sorted by total GPU time. Example output:

```
 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Kernel Name
 --------  ---------------  ---------  ---------  ---------  -----------
    35.2      1,250,000,000       2048   610,351    608,000   flash_fwd_kernel<...>
    28.1        998,000,000       6144   162,435    161,280   sm80_xmma_gemm_bf16bf16_...
     8.3        295,000,000       2048   144,042    143,360   void rmsnorm_kernel<...>
     6.1        217,000,000       2048   105,957    105,472   void rope_kernel<...>
     5.5        195,000,000       2048    95,214     94,720   void silu_mul_kernel<...>
     ...
```

What to extract:
- The top 3-5 kernels by total time are optimization targets
- Instance count tells you how many times the kernel is launched (per layer * per token)
- If avg time is very close to med time, the kernel is stable (no outliers from launch overhead)
- If a kernel has high instance count but low avg time, fusion is the strategy (combine with neighbors)

#### ncu (NVIDIA Nsight Compute) -- Kernel-Level

Profile a single kernel invocation:

```bash
ncu --set full --kernel-name "rmsnorm_kernel" --launch-skip 10 --launch-count 1 \
  python run_inference.py --prompt "Hello" --max_tokens 1
```

Key metrics to extract from ncu output:

```
Section: GPU Speed Of Light Throughput
  SM [%]:                           12.5
  Memory [%]:                       78.3

Section: Memory Workload Analysis
  Memory Throughput [GB/s]:         2,145.2
  L2 Hit Rate [%]:                  45.2
  Mem Busy [%]:                     81.4

Section: Compute Workload Analysis
  SM Busy [%]:                      12.1
  Executed Ipc Active:              0.82

Section: Occupancy
  Achieved Occupancy [%]:           62.5
  Theoretical Occupancy [%]:        100.0
  Block Limit SM:                   Registers

Section: Launch Statistics
  Threads:                          256
  Registers Per Thread:             48
  Shared Memory [bytes]:            8,192
  Blocks:                           128
  Waves Per SM:                     0.94
```

How to interpret:
- **Memory [%] >> SM [%]**: kernel is memory-bound. Optimize data access, fusion, reduce reads/writes.
- **SM [%] >> Memory [%]**: kernel is compute-bound. Optimize math, tiling, tensor core usage.
- **Both low**: kernel is latency-bound (launch overhead, synchronization, low occupancy). Fix occupancy or fuse.
- **Achieved Occupancy < 50%**: register pressure or shared memory is limiting parallelism. Reduce register usage or tune block size.
- **Memory Throughput vs HBM peak**: H100 SXM peak is 3.35 TB/s. If you see 2.1 TB/s for a memory-bound kernel, you are at 63% efficiency -- there is room to improve.

### 1.4 Operator-to-Backend Decision Table

| Operator | Bottleneck | Recommended Backend | Rationale |
|---|---|---|---|
| Attention (prefill) | Compute | FlashAttention-3 / CuTe | Tensor core tiling critical; matured implementations exist |
| Attention (decode) | Memory | FlashInfer / Triton | Memory-bound paged read; Triton can fuse RoPE |
| GEMM (large M) | Compute | cuBLAS / CUTLASS | Highly optimized; hard to beat for standard shapes |
| GEMM (small M, decode) | Memory | Triton / Marlin | Small-M GEMMs are bandwidth-limited; quantized kernels win |
| RMSNorm | Memory | Triton | Simple memory-bound kernel; easy to fuse |
| LayerNorm | Memory | Triton | Same rationale as RMSNorm |
| RoPE | Memory | Triton | Small kernel, benefits from fusion with attention |
| SiLU / GeGLU | Memory | Triton | Fuse with GEMM output or preceding load |
| Softmax | Memory | Triton | Online softmax well-suited to Triton |
| Embedding | Memory | PyTorch default | Rarely a bottleneck |
| LM Head | Compute | cuBLAS + chunking | Large GEMM; chunking reduces memory peak |
| KV Cache ops | Memory | Triton / CUDA | Custom memory layout requires custom kernels |
| MoE Routing | Memory | Triton | Small ops, fuse gating + dispatch |
| MoE Grouped GEMM | Compute | CUTLASS Grouped GEMM | Tensor core efficiency critical |

---

## Phase 2: Operator-by-Operator Optimization Recipes

### 2.1 Attention (Prefill)

#### Current State of the Art

- **FlashAttention-2**: Tiled attention with online softmax, O(N) memory. Standard for Ampere/Ada.
- **FlashAttention-3**: Hopper-optimized with TMA loads, warp-specialization, FP8 support. ~75% MMA utilization on H100.
- **FlashInfer**: Flexible attention with page table support, custom masks, JIT compilation.
- **xformers**: `memory_efficient_attention` with multiple backends.

#### When to Write Custom

- Non-standard head dimensions (not 64, 128, or 256)
- Custom causal masks (sliding window + global tokens, block-sparse)
- Fused QKV projection + attention in a single kernel
- FP8 attention with non-standard scaling
- ALiBi or other non-RoPE position encodings baked into the kernel

#### Triton Attention Kernel Template (Forward, Causal)

```python
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
    ],
    key=['seq_len', 'head_dim'],
)
@triton.jit
def _fwd_attention_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    seq_len, head_dim: tl.constexpr,
    sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // tl.num_programs(2)  # batch
    pid_h = pid_bh % tl.num_programs(2)   # head

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, head_dim)

    # Load Q block: [BLOCK_M, head_dim]
    q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + \
             offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seq_len, other=0.0)

    # Initialize accumulators
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)

    # Iterate over K, V blocks (online softmax)
    for start_n in range(0, (pid_m + 1) * BLOCK_M, BLOCK_N):  # causal: only up to current block
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # Load K block: [BLOCK_N, head_dim]
        k_ptrs = K + pid_b * stride_kb + pid_h * stride_kh + \
                 offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=offs_n[:, None] < seq_len, other=0.0)

        # Compute QK^T: [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, tl.trans(k)) * sm_scale

        # Apply causal mask
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        qk = tl.where(causal_mask, qk, float('-inf'))

        # Online softmax update
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])

        # Update accumulator
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        # Load V block and accumulate
        v_ptrs = V + pid_b * stride_vb + pid_h * stride_vh + \
                 offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=offs_n[:, None] < seq_len, other=0.0)
        acc += tl.dot(p.to(v.dtype), v)

        m_i = m_new

    # Normalize
    acc = acc / l_i[:, None]

    # Store output
    out_ptrs = Out + pid_b * stride_ob + pid_h * stride_oh + \
               offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < seq_len)
```

#### Key Tuning Parameters

| Parameter | Prefill (long seq) | Small Seq (<512) |
|---|---|---|
| BLOCK_M | 128 | 64 |
| BLOCK_N | 64 or 128 | 64 |
| num_warps | 4-8 | 4 |
| num_stages | 2-4 (Hopper: 4) | 2 |
| head_dim | constexpr (64, 128) | constexpr |

#### Memory Access Pattern

- Q is loaded once per program, reused across all K/V blocks (temporal reuse in registers)
- K and V are streamed block-by-block (spatial locality via tiling)
- Output is written once at the end (write-back after full reduction)
- For causal attention, roughly half the K/V blocks are skipped (triangular iteration)

#### Expected Performance

| GPU | Dtype | Seq Len | Head Dim | Expected TFLOPS |
|---|---|---|---|---|
| H100 SXM | BF16 | 4096 | 128 | 500-620 (FlashAttention-3) |
| A100 SXM | BF16 | 4096 | 128 | 200-260 (FlashAttention-2) |
| RTX 4090 | FP16 | 4096 | 128 | 140-180 |
| H100 SXM | FP8 | 4096 | 128 | 800-990 (FlashAttention-3 FP8) |

### 2.2 Attention (Decode)

#### Why Decode Is Different

In decode, each new token generates a single query vector (or a small batch of queries). The attention computation is:

```
score = Q[1, D] @ K[S, D]^T    -> [1, S]     (S = sequence length so far)
attn  = softmax(score)          -> [1, S]
out   = attn @ V[S, D]          -> [1, D]
```

This is a matrix-vector product, not matrix-matrix. It is entirely memory-bound: the bottleneck is reading the entire KV cache from HBM. Compute utilization will be very low (<5% SM utilization is normal).

#### Optimization Strategies

1. **PagedAttention** (vLLM): KV cache stored in non-contiguous pages. Kernel gathers from page table. Eliminates memory fragmentation.

2. **FlashInfer Paged Decode**: Optimized decode kernel with page-table indirection, supports variable-length sequences in a batch.

3. **Split-K for small batch**: When batch size is small, split the sequence dimension across multiple thread blocks, then reduce. This increases parallelism.

4. **Persistent kernel approach**: Launch one kernel that persists across multiple layers or attention heads, avoiding launch overhead.

5. **Fused RoPE + Attention**: Apply rotary embeddings inside the decode attention kernel rather than as a separate launch. Saves one HBM read/write of Q and K.

#### Decode Attention Performance Targets

For decode attention, the metric is HBM bandwidth utilization, not TFLOPS:

| GPU | KV Cache Read BW Target | Achieved by FlashInfer |
|---|---|---|
| H100 SXM | >2.8 TB/s (84% of 3.35 TB/s) | ~2.5-3.0 TB/s |
| A100 SXM | >1.6 TB/s (80% of 2.0 TB/s) | ~1.5-1.7 TB/s |
| RTX 4090 | >0.85 TB/s (83% of 1.0 TB/s) | ~0.8 TB/s |

### 2.3 GEMM / Linear Layers

#### When to Use Each Backend

| Scenario | Best Backend | Why |
|---|---|---|
| Standard BF16 GEMM, M >= 128 | cuBLAS | Highly tuned, hard to beat |
| Standard BF16 GEMM, M < 128 (decode) | Triton or custom | cuBLAS underperforms at small M |
| INT4 weight-only quant (W4A16) | Marlin / CUTLASS | Dequant fused into GEMM |
| FP8 GEMM (W8A8) | cuBLAS FP8 / CUTLASS | Native FP8 tensor core support on Ada/Hopper/Blackwell |
| GEMM + activation fused | Triton / CUTLASS epilogue | Avoid extra memory round-trip |
| Grouped GEMM (MoE) | CUTLASS Grouped GEMM | Batch multiple small GEMMs efficiently |
| Extremely small M (M=1, pure decode) | Triton / custom CUDA | Treat as GEMV, not GEMM |

#### Quantized GEMM: Marlin (INT4)

Marlin is the gold standard for W4A16 GEMM. Key properties:
- Weights stored as INT4 in a specific layout optimized for GPU global memory access
- Dequantization happens in registers during the GEMM
- Achieves near-ideal memory bandwidth (close to reading weights at INT4 size while computing at FP16)
- On A100: ~3.8x speedup over FP16 cuBLAS for decode (M=1)

#### FP8 GEMM with Delayed Scaling

On Ada, Hopper, and Blackwell:
- Per-tensor or per-channel scaling factors stored separately
- GEMM computed in FP8 on tensor cores, accumulated in FP32
- Delayed scaling: scale factors computed from the previous iteration (avoids synchronization)
- cuBLAS and CUTLASS both support FP8 natively

```python
# Using cuBLAS FP8 via PyTorch (Ada/Hopper/Blackwell)
import torch
import torch._scaled_mm

out = torch._scaled_mm(
    a_fp8,               # [M, K] in float8_e4m3fn
    b_fp8.t(),            # [K, N] in float8_e4m3fn (transposed)
    scale_a=scale_a,      # [1] or [M, 1] per-row scale
    scale_b=scale_b,      # [1] or [1, N] per-col scale
    out_dtype=torch.bfloat16,
)
```

#### Fused GEMM + Activation (gate_proj * up_proj * SiLU)

The LLaMA-style FFN computes:

```python
output = down_proj(silu(gate_proj(x)) * up_proj(x))
```

Naive execution: 3 GEMMs + 1 SiLU + 1 element-wise multiply = 5 kernel launches plus intermediate tensors.

Optimized approach:
1. Fuse gate_proj and up_proj into a single GEMM with concatenated weights: `[gate; up] = [x] @ [W_gate | W_up]`
2. Fuse SiLU + element-wise multiply into the GEMM epilogue (CUTLASS) or a follow-up Triton kernel
3. Net: 2 kernel launches (fused_gate_up_gemm + silu_mul, then down_proj_gemm)

#### Tile Size Selection Guide

| Problem Shape (M, N, K) | Recommended Tile | Backend |
|---|---|---|
| M >= 1024, N >= 1024 | 128x256x64 or 256x128x64 | cuBLAS/CUTLASS |
| M = 1-16 (decode) | 16xNx128 | Triton (treat as GEMV) |
| M = 16-128 | 64x128x64 | Triton or CUTLASS |
| INT4, M=1 | Marlin layout | Marlin |
| Grouped GEMM (MoE) | 64x128x64 per expert | CUTLASS Grouped |

### 2.4 RMSNorm / LayerNorm

#### Why These Are Pure Memory-Bound

RMSNorm for a hidden_dim=4096 vector:
- Read input: 4096 * 2 bytes = 8 KB
- Read weight: 4096 * 2 bytes = 8 KB
- Compute: 4096 multiplies, 1 reduction (variance), 4096 multiplies = ~12K FLOPs
- Write output: 4096 * 2 bytes = 8 KB
- Total data movement: 24 KB, total compute: 12K FLOPs
- Arithmetic intensity: 12K / 24K = 0.5 FLOPs/byte
- H100 ridge point: ~300 FLOPs/byte for BF16 tensor cores
- This is 600x below the ridge point. Purely memory-bound.

#### Fused RMSNorm + Residual Add (Triton)

```python
@triton.jit
def _rms_norm_fused_residual_kernel(
    X,          # input tensor [N, D]
    Residual,   # residual tensor [N, D] (read and updated in-place)
    Weight,     # RMSNorm weight [D]
    Out,        # output [N, D]
    stride_x,
    stride_r,
    stride_o,
    N, D: tl.constexpr,
    eps: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, D)

    # Load input and residual
    x = tl.load(X + row * stride_x + offs, mask=offs < D, other=0.0).to(tl.float32)
    res = tl.load(Residual + row * stride_r + offs, mask=offs < D, other=0.0).to(tl.float32)

    # Fused residual add
    hidden = x + res

    # Store updated residual in-place
    tl.store(Residual + row * stride_r + offs, hidden.to(Residual.dtype.element_ty), mask=offs < D)

    # RMSNorm
    variance = tl.sum(hidden * hidden, axis=0) / D
    rstd = 1.0 / tl.sqrt(variance + eps)
    w = tl.load(Weight + offs, mask=offs < D, other=1.0).to(tl.float32)
    out = hidden * rstd * w

    tl.store(Out + row * stride_o + offs, out.to(Out.dtype.element_ty), mask=offs < D)
```

#### Fused RMSNorm + Quantization

For FP8 inference, fuse RMSNorm output directly to FP8:

```python
@triton.jit
def _rms_norm_fp8_kernel(
    X, Weight, Out_fp8, Scale_out,
    stride_x, stride_o,
    N, D: tl.constexpr, eps: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, D)

    x = tl.load(X + row * stride_x + offs, mask=offs < D, other=0.0).to(tl.float32)

    variance = tl.sum(x * x, axis=0) / D
    rstd = 1.0 / tl.sqrt(variance + eps)
    w = tl.load(Weight + offs, mask=offs < D, other=1.0).to(tl.float32)
    normed = x * rstd * w

    # Dynamic per-row quantization to FP8
    amax = tl.max(tl.abs(normed), axis=0)
    scale = amax / 448.0  # E4M3 max value
    scale = tl.where(scale > 0, scale, 1.0)
    quantized = normed / scale

    tl.store(Out_fp8 + row * stride_o + offs, quantized.to(tl.float8e4nv), mask=offs < D)
    tl.store(Scale_out + row, scale.to(tl.float32))
```

#### Expected Bandwidth Targets

| GPU | Kernel | Hidden Dim | Target BW | Target Time |
|---|---|---|---|---|
| H100 SXM | RMSNorm BF16 | 4096 | >2.7 TB/s (80% peak) | ~8.9 us |
| H100 SXM | RMSNorm + Residual BF16 | 4096 | >2.5 TB/s (75% peak) | ~12.8 us |
| A100 SXM | RMSNorm BF16 | 4096 | >1.6 TB/s (80% peak) | ~15 us |
| RTX 4090 | RMSNorm FP16 | 4096 | >0.8 TB/s (80% peak) | ~30 us |

### 2.5 Rotary Position Embedding (RoPE)

#### Standalone RoPE Triton Kernel

```python
@triton.jit
def _rope_kernel(
    Q, K,           # [batch, seq, heads, head_dim]
    COS, SIN,       # [seq, head_dim // 2] precomputed
    stride_qb, stride_qs, stride_qh, stride_qd,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_cs, stride_cd,
    seq_len, n_heads, head_dim: tl.constexpr,
    HALF_DIM: tl.constexpr,
):
    pid = tl.program_id(0)
    # Decode pid into (batch, seq, head)
    pid_b = pid // (seq_len * n_heads)
    pid_s = (pid // n_heads) % seq_len
    pid_h = pid % n_heads

    offs_half = tl.arange(0, HALF_DIM)

    # Load first and second halves of Q
    q_base = Q + pid_b * stride_qb + pid_s * stride_qs + pid_h * stride_qh
    q_first = tl.load(q_base + offs_half * stride_qd)
    q_second = tl.load(q_base + (offs_half + HALF_DIM) * stride_qd)

    # Load cos, sin
    cos = tl.load(COS + pid_s * stride_cs + offs_half * stride_cd)
    sin = tl.load(SIN + pid_s * stride_cs + offs_half * stride_cd)

    # Apply rotation
    q_out_first = q_first * cos - q_second * sin
    q_out_second = q_second * cos + q_first * sin

    tl.store(q_base + offs_half * stride_qd, q_out_first)
    tl.store(q_base + (offs_half + HALF_DIM) * stride_qd, q_out_second)

    # Same for K
    k_base = K + pid_b * stride_kb + pid_s * stride_ks + pid_h * stride_kh
    k_first = tl.load(k_base + offs_half * stride_kd)
    k_second = tl.load(k_base + (offs_half + HALF_DIM) * stride_kd)

    k_out_first = k_first * cos - k_second * sin
    k_out_second = k_second * cos + k_first * sin

    tl.store(k_base + offs_half * stride_kd, k_out_first)
    tl.store(k_base + (offs_half + HALF_DIM) * stride_kd, k_out_second)
```

#### Fused RoPE in Attention

The preferred approach is to fuse RoPE into the attention kernel itself. Apply rotation to Q after loading the Q block and to K after loading each K block inside the attention loop. This eliminates a separate kernel launch and avoids writing rotated Q/K to HBM.

#### Scaling Variants

- **NTK-aware**: Scale the frequency basis `theta_i = base^(-2i/d)` by modifying `base`. The kernel uses the same structure but with a different `COS`/`SIN` table.
- **YaRN**: Combines NTK scaling with attention scaling and temperature. Precompute modified `COS`/`SIN` tables; kernel code is unchanged.
- **Dynamic NTK**: `base` changes with sequence length. Recompute tables when sequence length crosses thresholds.

### 2.6 Activation Functions (SiLU, GELU, GeGLU)

#### Fused SiLU + Element-wise Multiply

The LLaMA FFN pattern `silu(gate) * up` is the primary target. Always fuse this into a single kernel:

```python
@triton.jit
def _silu_mul_kernel(
    Gate, Up, Out,
    stride_g, stride_u, stride_o,
    N, D: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, D)

    gate = tl.load(Gate + row * stride_g + offs, mask=offs < D, other=0.0).to(tl.float32)
    up = tl.load(Up + row * stride_u + offs, mask=offs < D, other=0.0).to(tl.float32)

    # SiLU = x * sigmoid(x)
    sigmoid_gate = 1.0 / (1.0 + tl.exp(-gate))
    result = gate * sigmoid_gate * up

    tl.store(Out + row * stride_o + offs, result.to(Out.dtype.element_ty), mask=offs < D)
```

#### Fusing with Preceding GEMM

With CUTLASS, define a custom epilogue that applies SiLU + multiply after the GEMM output is computed in the register file:

```
GEMM(x, W_gate) -> gate  (in registers)
GEMM(x, W_up)   -> up    (requires separate GEMM or fused column-split)
epilogue: out = silu(gate) * up  (avoids HBM write/read of gate and up)
```

This requires either:
1. A dual-GEMM CUTLASS kernel (computes two GEMMs with shared input, fused epilogue)
2. Fused QKV-style GEMM with column splitting, then epilogue

The Triton approach of a separate fused elementwise kernel after two GEMMs is simpler and nearly as fast since the fused kernel achieves high bandwidth.

### 2.7 Embedding / LM Head

#### Why LM Head Is Expensive

For LLaMA 3 with vocab_size=128256:
- Weight matrix: [hidden_dim, vocab_size] = [4096, 128256] * 2 bytes = 1.0 GB in BF16
- Decode (M=1): Must read entire 1 GB weight matrix for a single output vector
- At H100 bandwidth (3.35 TB/s), minimum time = 1.0 GB / 3.35 TB/s = ~300 us
- This is a significant fraction of per-token decode time

#### Optimization Strategies

1. **Chunked LM Head**: Split vocab dimension into chunks (e.g., 4 chunks of 32K). Process each chunk, find local top-K, then merge. Reduces peak memory by chunk factor.

```python
def chunked_lm_head(hidden, lm_weight, chunk_size=32768):
    vocab_size = lm_weight.shape[0]
    logits_chunks = []
    for i in range(0, vocab_size, chunk_size):
        chunk_w = lm_weight[i:i+chunk_size]
        logits_chunks.append(hidden @ chunk_w.T)
    return torch.cat(logits_chunks, dim=-1)
```

2. **FP8 LM Head**: Quantize LM head weight to FP8. Halves memory read, nearly halving latency. On Ada/Hopper/Blackwell, use `torch._scaled_mm`.

3. **Speculative Decoding LM Head**: When using speculative decoding, the LM head is called on draft model outputs. Keep the draft model LM head in FP8/INT8.

4. **Skip LM Head When Possible**: For intermediate tokens in speculative verify, you may only need the top-1 logit. Fuse argmax into the GEMM epilogue to avoid materializing full logits.

### 2.8 KV Cache Operations

#### Paged KV Cache Structure

```
Physical pages: [num_pages, page_size, num_kv_heads, head_dim]
Page table:     [num_sequences, max_pages_per_seq]  (maps logical -> physical page index)
```

#### Paged KV Cache Append (Triton)

```python
@triton.jit
def _kv_cache_append_kernel(
    K_new, V_new,       # [batch, num_kv_heads, head_dim] -- new token's KV
    K_cache, V_cache,   # [num_pages, page_size, num_kv_heads, head_dim]
    Page_table,          # [batch, max_pages_per_seq]
    Seq_lens,            # [batch] -- current sequence length per sequence
    stride_kb, stride_kh, stride_kd,
    stride_cb, stride_cs, stride_ch, stride_cd,
    stride_pb, stride_pm,
    page_size: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch index
    pid_h = tl.program_id(1)  # kv head index

    seq_len = tl.load(Seq_lens + pid_b)
    page_idx = seq_len // page_size
    slot_idx = seq_len % page_size

    # Look up physical page
    phys_page = tl.load(Page_table + pid_b * stride_pb + page_idx * stride_pm)

    # Load new K, V
    offs_d = tl.arange(0, head_dim)
    k_new = tl.load(K_new + pid_b * stride_kb + pid_h * stride_kh + offs_d * stride_kd)
    v_new = tl.load(V_new + pid_b * stride_kb + pid_h * stride_kh + offs_d * stride_kd)

    # Write to cache page
    cache_base_k = K_cache + phys_page * stride_cb + slot_idx * stride_cs + pid_h * stride_ch
    cache_base_v = V_cache + phys_page * stride_cb + slot_idx * stride_cs + pid_h * stride_ch
    tl.store(cache_base_k + offs_d * stride_cd, k_new)
    tl.store(cache_base_v + offs_d * stride_cd, v_new)
```

#### KV Cache Quantization

- **FP8 KV Cache**: Quantize K and V to FP8 (E4M3) with per-head or per-page scaling. Halves KV cache memory, allowing 2x longer sequences or 2x batch size.
- **INT8 KV Cache**: Similar to FP8 but uses symmetric INT8 quantization. Slightly more complex dequant but wider hardware support.
- Fuse quantization into the KV cache append kernel: after computing K_new and V_new, quantize in-register before writing to cache.
- Dequantization happens inside the attention kernel when loading K/V blocks.

#### Fused KV Cache Update + RoPE

Apply RoPE to K before writing to KV cache, in a single kernel:

```python
# Inside the append kernel, after loading k_new:
offs_half = tl.arange(0, head_dim // 2)
cos = tl.load(COS + seq_len * (head_dim // 2) + offs_half)
sin = tl.load(SIN + seq_len * (head_dim // 2) + offs_half)
k_first = k_new[offs_half]      # first half (via slice, conceptual)
k_second = k_new[offs_half + head_dim // 2]
k_rot_first = k_first * cos - k_second * sin
k_rot_second = k_second * cos + k_first * sin
# Store rotated K to cache
```

### 2.9 MoE Routing and Expert Dispatch

#### Top-K Gating Kernel

```python
@triton.jit
def _topk_gating_kernel(
    Logits,        # [num_tokens, num_experts]
    TopK_vals,     # [num_tokens, K] output
    TopK_ids,      # [num_tokens, K] output
    stride_l, stride_tv, stride_ti,
    num_experts: tl.constexpr,
    K: tl.constexpr,
):
    pid = tl.program_id(0)  # token index
    offs_e = tl.arange(0, num_experts)

    logits = tl.load(Logits + pid * stride_l + offs_e, mask=offs_e < num_experts, other=float('-inf'))

    # Softmax for gating weights
    max_logit = tl.max(logits, axis=0)
    exp_logits = tl.exp(logits - max_logit)
    sum_exp = tl.sum(exp_logits, axis=0)
    probs = exp_logits / sum_exp

    # Top-K selection (for K=2, unroll)
    # First top
    idx1 = tl.argmax(probs, axis=0)
    val1 = tl.max(probs, axis=0)

    # Mask out first and find second
    probs_masked = tl.where(offs_e == idx1, float('-inf'), probs)
    idx2 = tl.argmax(probs_masked, axis=0)
    val2 = tl.max(probs_masked, axis=0)

    # Renormalize
    total = val1 + val2
    tl.store(TopK_vals + pid * stride_tv + 0, val1 / total)
    tl.store(TopK_vals + pid * stride_tv + 1, val2 / total)
    tl.store(TopK_ids + pid * stride_ti + 0, idx1)
    tl.store(TopK_ids + pid * stride_ti + 1, idx2)
```

#### Grouped GEMM for Expert Execution

After routing, tokens are grouped by expert. Each expert runs an FFN (gate/up/down projections). The key optimization is executing all experts as a single Grouped GEMM:

```python
# Using CUTLASS Grouped GEMM (conceptual interface)
problem_sizes = []  # List of (M_i, N, K) for each expert
for expert_id in range(num_experts):
    num_tokens_for_expert = token_counts[expert_id]
    problem_sizes.append((num_tokens_for_expert, intermediate_dim, hidden_dim))

# CUTLASS groups these into a single kernel launch
grouped_gemm(
    A_ptrs=expert_input_ptrs,     # per-expert input pointers
    B_ptrs=expert_weight_ptrs,    # per-expert weight pointers
    C_ptrs=expert_output_ptrs,    # per-expert output pointers
    problem_sizes=problem_sizes,
)
```

Alternatively, in Triton, use a persistent kernel that iterates over expert assignments:

```python
# Each thread block picks up work items from a global work queue
# Work item = (expert_id, token_start, token_end)
# Block loads the expert's weight tile and processes assigned tokens
```

#### Load Balancing

- **Capacity factor**: Limit tokens per expert to `capacity = ceil(num_tokens * topk / num_experts * capacity_factor)`. Typical capacity_factor = 1.25.
- **Auxiliary loss**: Training concern, not inference. But at inference, dropped tokens degrade quality. Monitor and log.
- **Expert parallelism**: For very large MoE (e.g., DeepSeek-V3 with 256 experts), distribute experts across GPUs. Routing decides which GPU to send each token to (all-to-all communication).

### 2.10 Fused Residual Chains

#### Why Multi-Op Fusion Matters for Decode

In decode, every operator is memory-bound. The transformer layer has this chain:

```
hidden -> residual_add -> rmsnorm -> qkv_proj -> ...
```

Without fusion, `residual_add` reads hidden (8 KB) + residual (8 KB), writes result (8 KB). Then `rmsnorm` reads the result (8 KB), writes normed output (8 KB). Total: 40 KB of HBM traffic.

With fusion (`residual_add_rmsnorm`): reads hidden (8 KB) + residual (8 KB) + weight (8 KB), writes updated residual (8 KB) + normed output (8 KB). Total: 40 KB but only 2 kernel launches instead of 2, and intermediate never materializes in HBM. In practice this saves ~20-30% on these operations due to reduced launch overhead and better cache behavior.

#### Triton Fused Residual + RMSNorm + Optional Quant

```python
@triton.jit
def _fused_residual_rmsnorm_quant_kernel(
    X,              # [N, D] -- layer output (attention or FFN output)
    Residual,       # [N, D] -- running residual (in-place updated)
    Weight,         # [D]    -- RMSNorm weight
    Out,            # [N, D] -- normed output (BF16 or FP8)
    Out_scale,      # [N]    -- per-row FP8 scale (if quantizing)
    stride_x, stride_r, stride_o,
    N, D: tl.constexpr,
    eps: tl.constexpr,
    QUANTIZE_FP8: tl.constexpr,  # 1 = output FP8, 0 = output BF16
):
    row = tl.program_id(0)
    offs = tl.arange(0, D)
    mask = offs < D

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    res = tl.load(Residual + row * stride_r + offs, mask=mask, other=0.0).to(tl.float32)

    # Residual add
    hidden = x + res
    tl.store(Residual + row * stride_r + offs, hidden.to(Residual.dtype.element_ty), mask=mask)

    # RMSNorm
    sq_mean = tl.sum(hidden * hidden, axis=0) / D
    rstd = 1.0 / tl.sqrt(sq_mean + eps)
    w = tl.load(Weight + offs, mask=mask, other=1.0).to(tl.float32)
    normed = hidden * rstd * w

    if QUANTIZE_FP8:
        amax = tl.max(tl.abs(normed), axis=0)
        scale = amax / 448.0
        scale = tl.where(scale > 0, scale, 1.0)
        quantized = normed / scale
        tl.store(Out + row * stride_o + offs, quantized.to(tl.float8e4nv), mask=mask)
        tl.store(Out_scale + row, scale)
    else:
        tl.store(Out + row * stride_o + offs, normed.to(Out.dtype.element_ty), mask=mask)
```

---

## Phase 3: Kernel Writing Methodology

This is the step-by-step process the agent must follow when optimizing any operator. Do not skip steps.

### Step 1: Profile the Operator in Isolation

```bash
# Wrap the operator in a benchmark harness
python -c "
import torch

x = torch.randn(1, 4096, device='cuda', dtype=torch.bfloat16)
w = torch.randn(4096, device='cuda', dtype=torch.bfloat16)

def rmsnorm(x, w, eps=1e-6):
    return w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

# Warmup
for _ in range(100):
    rmsnorm(x, w)
torch.cuda.synchronize()

# Profile
import time
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(10000):
    rmsnorm(x, w)
torch.cuda.synchronize()
t1 = time.perf_counter()
print(f'Avg: {(t1-t0)/10000*1e6:.1f} us')
"
```

Then profile with ncu:

```bash
ncu --set full --kernel-name-base demangled \
    --launch-skip 50 --launch-count 5 \
    python benchmark_rmsnorm.py
```

### Step 2: Calculate Theoretical Peak

For memory-bound operators:

```
bytes_read = input_elements * dtype_size + weight_elements * dtype_size
bytes_written = output_elements * dtype_size
total_bytes = bytes_read + bytes_written
theoretical_min_time = total_bytes / peak_hbm_bandwidth
```

For compute-bound operators:

```
total_flops = 2 * M * N * K  (for GEMM)
theoretical_min_time = total_flops / peak_tensor_core_flops
```

Peak values by GPU:

| GPU | HBM BW (TB/s) | BF16 Tensor Core TFLOPS | FP8 Tensor Core TFLOPS |
|---|---|---|---|
| A100 SXM | 2.0 | 312 | N/A |
| RTX 4090 | 1.0 | 330 (FP16) | 660 |
| H100 SXM | 3.35 | 989 | 1979 |
| H200 SXM | 4.8 | 989 | 1979 |
| B200 SXM | 8.0 | 2250 | 4500 |

### Step 3: Calculate Achieved Efficiency

```
memory_efficiency = total_bytes / (measured_time * peak_bandwidth)
compute_efficiency = total_flops / (measured_time * peak_flops)
```

Example: RMSNorm on H100, hidden_dim=4096, BF16
- total_bytes = 4096 * 2 (input) + 4096 * 2 (weight) + 4096 * 2 (output) = 24,576 bytes = 24 KB
- peak_bandwidth = 3.35 TB/s = 3.35e12 B/s
- theoretical_min_time = 24,576 / 3.35e12 = 7.3 ns
- But kernel launch overhead dominates at this scale (~2-5 us minimum)
- For batched RMSNorm (batch=256, hidden=4096): total_bytes = 256 * 24 KB = 6.14 MB
- theoretical_min_time = 6.14e6 / 3.35e12 = 1.83 us
- If measured time is 2.5 us: efficiency = 1.83 / 2.5 = 73%

### Step 4: Determine If Optimization Is Worthwhile

Decision thresholds:
- **Memory-bound kernel at <70% HBM bandwidth utilization**: Room to optimize. Common causes: uncoalesced access, unnecessary reads, register spills.
- **Compute-bound kernel at <50% tensor core utilization**: Room to optimize. Common causes: bad tile sizes, insufficient occupancy, data type mismatch.
- **Both at <30%**: Kernel is likely latency-bound (launch overhead, synchronization). Fusion or persistent kernels needed.
- **Memory-bound at >85%**: Close to hardware limits. Further gains require algorithmic changes (fewer bytes total) or fusion.
- **Compute-bound at >70%**: Near optimal. Only micro-optimization (instruction scheduling, register allocation) will help.

### Step 5: Choose Backend

| Condition | Backend | Rationale |
|---|---|---|
| Memory-bound, simple elementwise/reduction | Triton | Fastest iteration, good codegen for memory ops |
| Compute-bound GEMM, standard shapes | cuBLAS | Already optimal; do not rewrite |
| Compute-bound GEMM, custom epilogue needed | CUTLASS / CuTe DSL | Epilogue fusion critical for perf |
| Compute-bound attention | FlashAttention / CuTe | Tiling + online softmax is complex |
| Need NVIDIA-specific features (TMA, wgmma) | CuTe DSL | Only way to access Hopper hardware features from Python |
| Need maximum control | CUDA C++ | Last resort; highest effort |

### Step 6: Write Kernel from Template

Never write from scratch. Start from the templates in Phase 2 of this document, or from:
- Triton tutorial kernels (github.com/triton-lang/triton/tree/main/python/tutorials)
- CUTLASS examples (github.com/NVIDIA/cutlass/tree/main/examples)
- vLLM kernels (github.com/vllm-project/vllm/tree/main/csrc)
- SGLang kernels (github.com/sgl-project/sglang/tree/main/python/sglang)
- FlashInfer (github.com/flashinfer-ai/flashinfer)
- Liger Kernel (github.com/linkedin/Liger-Kernel) -- Triton kernels for common LLM ops

Modify the template for the specific problem shape, dtype, and fusion pattern needed.

### Step 7: Tile Size Selection

Start with the Triton autotuner, then narrow down:

```python
@triton.autotune(
    configs=[
        # Sweep BLOCK_SIZE and num_warps
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        # For Hopper, also sweep num_stages
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=4),
    ],
    key=['D'],  # re-autotune when D changes
)
@triton.jit
def _kernel(..., BLOCK_SIZE: tl.constexpr):
    ...
```

After autotuning selects a winner, verify with ncu that:
- Achieved occupancy is reasonable (>50%)
- Register count per thread is not maxed out (check "Block Limit SM: Registers")
- Shared memory usage fits within SM limits

### Step 8: Verify Correctness

```python
def verify_kernel(custom_fn, reference_fn, input_shapes, dtype, atol=None, rtol=None):
    """Verify custom kernel against PyTorch reference."""
    if atol is None:
        atol = {
            torch.float32: 1e-5,
            torch.float16: 1e-3,
            torch.bfloat16: 1e-2,
            torch.float8_e4m3fn: 5e-2,
        }[dtype]
    if rtol is None:
        rtol = {
            torch.float32: 1e-5,
            torch.float16: 1e-3,
            torch.bfloat16: 1.6e-2,
            torch.float8_e4m3fn: 1e-1,
        }[dtype]

    inputs = [torch.randn(s, device='cuda', dtype=dtype) for s in input_shapes]
    ref_out = reference_fn(*inputs)
    custom_out = custom_fn(*inputs)

    if not torch.allclose(ref_out.float(), custom_out.float(), atol=atol, rtol=rtol):
        max_diff = (ref_out.float() - custom_out.float()).abs().max().item()
        mean_diff = (ref_out.float() - custom_out.float()).abs().mean().item()
        raise AssertionError(
            f"Kernel mismatch: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}, "
            f"atol={atol}, rtol={rtol}"
        )
    print(f"PASSED: max_diff={(ref_out.float() - custom_out.float()).abs().max().item():.6e}")
```

Always test:
- Multiple input shapes (powers of 2 and non-powers of 2)
- Edge cases: seq_len=1, batch=1, very large inputs
- Multiple random seeds (at least 3)
- Both contiguous and non-contiguous (transposed) inputs

### Step 9: Benchmark

```python
import torch
import triton

def benchmark_kernel(fn, args, warmup=100, rep=1000):
    """Benchmark using Triton's built-in timer for accurate GPU timing."""
    ms = triton.testing.do_bench(lambda: fn(*args), warmup=warmup, rep=rep)
    return ms

def report_kernel_perf(name, ms, total_bytes=None, total_flops=None, gpu='H100'):
    """Print kernel performance report."""
    peak_bw = {
        'A100': 2.0e12, 'RTX4090': 1.008e12,
        'H100': 3.35e12, 'H200': 4.8e12, 'B200': 8.0e12,
    }
    peak_flops_bf16 = {
        'A100': 312e12, 'RTX4090': 330e12,
        'H100': 989e12, 'H200': 989e12, 'B200': 2250e12,
    }

    print(f"\n{'='*60}")
    print(f"Kernel: {name}")
    print(f"Time: {ms*1000:.1f} us")

    if total_bytes:
        bw = total_bytes / (ms / 1000)
        bw_pct = bw / peak_bw[gpu] * 100
        print(f"Bandwidth: {bw/1e12:.3f} TB/s ({bw_pct:.1f}% of {gpu} peak)")

    if total_flops:
        tflops = total_flops / (ms / 1000) / 1e12
        flops_pct = total_flops / (ms / 1000) / peak_flops_bf16[gpu] * 100
        print(f"Throughput: {tflops:.1f} TFLOPS ({flops_pct:.1f}% of {gpu} BF16 peak)")

    print(f"{'='*60}")
```

### Step 10: Compare Against Best-Known Implementation

Always benchmark against:
1. **PyTorch eager** (baseline, usually slow)
2. **torch.compile** (free speedup from fusion/codegen)
3. **Best known library** (FlashAttention, cuBLAS, Liger Kernel, etc.)

If the custom kernel does not beat option 3, do not integrate it. Document why:
- "Custom kernel achieves 78% HBM BW; FlashInfer achieves 82%. Gap is due to [specific reason]. Not worth maintaining custom code."

### Step 11: Integrate or Document

If the custom kernel wins:
1. Package it (see Phase 5)
2. Add correctness tests to CI
3. Add benchmarks to regression suite
4. Integrate into the serving runtime

If it loses:
1. Document the comparison results
2. Record the specific gap and root cause
3. Keep the kernel code in a `experiments/` directory for future reference
4. Move to the next operator

---

## Phase 4: Tile Size and Autotuning Guide

### 4.1 Triton Autotune Decorator Patterns

#### For Element-wise / Reduction Kernels (Memory-Bound)

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 256}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_D': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_D': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_D': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_D': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_D': 8192}, num_warps=16, num_stages=2),
    ],
    key=['D'],
)
```

#### For 2D Tiled Kernels (Matmul-Like)

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
```

### 4.2 BLOCK_SIZE Selection Heuristics

For memory-bound kernels processing a dimension D:
- If D <= 1024: BLOCK_D = D (process entire dimension in one block)
- If D = 2048-4096: BLOCK_D = D or D/2
- If D = 8192+: BLOCK_D = 4096 or 8192, multiple rows per program

For 2D tiled kernels:
- Total tile elements (BLOCK_M * BLOCK_N) should be 4096-16384 per block
- BLOCK_K = 32 or 64 (fits in shared memory pipeline stages)
- Increase BLOCK_M for better output reuse, BLOCK_N for better weight reuse

### 4.3 num_warps Selection

| Problem Size per Block | Recommended num_warps |
|---|---|
| < 1024 elements | 1-2 |
| 1024-4096 elements | 4 |
| 4096-8192 elements | 4-8 |
| > 8192 elements | 8-16 |

Rule: each warp processes 32 threads. If you have fewer than 32 independent operations per warp, you waste threads. If you have too many warps, register pressure increases and occupancy may drop.

### 4.4 num_stages for Software Pipelining

- **Ampere/Ada**: num_stages = 2-3 is typical. Each stage doubles shared memory usage.
- **Hopper**: num_stages = 3-5 recommended. Hopper has more shared memory (228 KB vs 164 KB on A100) and TMA supports deeper pipelining.
- **Blackwell**: num_stages = 4-6. Even more shared memory (256 KB).
- If increasing num_stages causes occupancy to drop (shared memory limit), reduce stages.

### 4.5 Occupancy Analysis

Check with ncu:

```
Block Limit SM: Registers     -> too many registers per thread
Block Limit SM: Shared Mem    -> too much shared memory per block
Block Limit SM: Warps         -> too many warps per block
Block Limit SM: Blocks        -> max blocks per SM reached
```

Common fixes:
- **Register-limited**: Reduce BLOCK_SIZE, simplify kernel (fewer variables), use `tl.constexpr` aggressively
- **Shared memory limited**: Reduce num_stages, reduce tile size, use smaller dtypes in shared memory
- **Low waves**: Not enough blocks to fill GPU. Increase parallelism (more blocks) or use persistent kernels.

### 4.6 GPU-Specific Tuning Notes

#### Ampere (A100, RTX 3090)

- Max shared memory per SM: 164 KB (A100 GA100), 96 KB (RTX 3090 GA102)
- Max registers per SM: 65536
- Tensor core instruction: `mma.m16n8k16` for BF16
- No TMA, no warp-specialization
- Triton configs: num_stages=2-3, focus on register pressure

#### Ada (RTX 4090, L40S)

- Max shared memory per SM: 100 KB (RTX 4090), 100 KB (L40S)
- Max registers per SM: 65536
- Tensor core: `mma.m16n8k16` for FP16/BF16, FP8 support via `mma.m16n8k32`
- Similar to Ampere tuning but FP8 available
- RTX 4090 has high shader clock (2520 MHz boost) but only 1 TB/s HBM BW

#### Hopper (H100, H200)

- Max shared memory per SM: 228 KB
- Max registers per SM: 65536
- Tensor core: `wgmma` instructions (warp group matrix multiply-accumulate)
- TMA (Tensor Memory Accelerator) for async global-to-shared copies
- Warp specialization: producer warps load data, consumer warps compute
- Triton configs: num_stages=3-5, use `tl.async_copy` hints
- CuTe DSL can directly target wgmma + TMA

#### Blackwell (B200, GB200)

- Max shared memory per SM: 256 KB
- Tensor core: 5th generation, 2x throughput vs Hopper
- FP4 tensor core support (NVFP4 format)
- Enhanced TMA with multicast
- Triton support still maturing; CUTLASS 3.x + CuTe DSL recommended
- Focus on FP8 and FP4 quantized paths

---

## Phase 5: Integration into Serving Runtimes

### 5.1 Replacing an Operator in vLLM

vLLM uses custom ops registered via `torch.library`:

```python
# In your_custom_ops.py
import torch
from torch.library import custom_op

@custom_op("myops::fused_rmsnorm", mutates_args=("residual",))
def fused_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    # Call your Triton kernel here
    output = torch.empty_like(x)
    _rms_norm_fused_residual_kernel[grid](
        x, residual, weight, output,
        x.stride(0), residual.stride(0), output.stride(0),
        x.shape[0], x.shape[1],
        eps=eps,
    )
    return output

# Register fake tensor impl for torch.compile
@fused_rmsnorm.register_fake
def _(x, residual, weight, eps):
    return torch.empty_like(x)
```

Then monkey-patch the model layer:

```python
# In your model modification script
original_forward = model.layers[0].input_layernorm.forward

def patched_forward(hidden_states, residual):
    return torch.ops.myops.fused_rmsnorm(
        hidden_states, residual,
        model.layers[0].input_layernorm.weight,
        model.layers[0].input_layernorm.variance_epsilon,
    )

for layer in model.layers:
    layer.input_layernorm.forward = patched_forward
```

### 5.2 Replacing an Operator in SGLang

SGLang follows a similar pattern. Custom kernels are placed in `sglang/srt/layers/` and registered:

```python
# sglang/srt/layers/custom_rmsnorm.py
from sglang.srt.layers.rmsnorm import RMSNorm

class FusedRMSNorm(RMSNorm):
    def forward(self, x, residual=None):
        if residual is not None:
            output = fused_rmsnorm_triton(x, residual, self.weight, self.variance_epsilon)
            return output, residual  # residual updated in-place
        return rmsnorm_triton(x, self.weight, self.variance_epsilon)
```

### 5.3 Using torch.compile Custom Ops

For kernels that should work with `torch.compile`:

```python
# 1. Register with torch.library (as above)
# 2. Provide a fake tensor implementation
# 3. Optionally provide a backward (for training)

@torch.library.custom_op("myops::custom_attention", mutates_args=())
def custom_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return _triton_attention(q, k, v)

@custom_attention.register_fake
def _(q, k, v):
    return torch.empty_like(q)

# Now torch.compile will treat this as an opaque op and not try to trace into it
```

### 5.4 Packaging as a PyTorch Extension

For CUDA C++ kernels:

```python
# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custom_kernels',
    ext_modules=[
        CUDAExtension(
            name='custom_kernels._C',
            sources=[
                'csrc/rmsnorm_kernel.cu',
                'csrc/attention_kernel.cu',
                'csrc/bindings.cpp',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-gencode=arch=compute_80,code=sm_80',  # Ampere
                    '-gencode=arch=compute_89,code=sm_89',  # Ada
                    '-gencode=arch=compute_90,code=sm_90',  # Hopper
                ],
            },
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
```

For Triton kernels, no compilation step is needed. Just package as Python:

```python
# custom_kernels/__init__.py
from .rmsnorm import fused_rmsnorm
from .attention import custom_attention
from .rope import fused_rope
```

### 5.5 Handling Fallbacks

Always implement fallbacks:

```python
def rmsnorm_with_fallback(x, residual, weight, eps):
    try:
        return fused_rmsnorm_triton(x, residual, weight, eps)
    except Exception as e:
        import warnings
        warnings.warn(f"Custom kernel failed: {e}. Falling back to PyTorch.")
        if residual is not None:
            x = x + residual
            residual.copy_(x)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        return x * weight
```

Check for:
- Unsupported GPU architecture (e.g., Triton kernel targeting Hopper features on Ampere)
- Unsupported tensor shapes (e.g., head_dim not a power of 2)
- Unsupported dtypes (e.g., FP8 kernel on Ampere)
- CUDA OOM in shared memory allocation

---

## Phase 6: End-to-End Optimization Checklist

Follow this numbered checklist for every model optimization engagement. Do not skip steps.

### Checklist

1. **Profile the full model end-to-end** with nsys:
   ```bash
   nsys profile --trace=cuda,nvtx -o full_model_trace python serve.py --input "test" --max-tokens 128
   nsys stats --report cuda_gpu_kern_sum full_model_trace.nsys-rep > kernel_summary.txt
   ```

2. **Extract the top-10 hottest kernels** by total GPU time from the summary. Record kernel name, total time, instance count, and average time per invocation.

3. **Map each hot kernel to its operator**: e.g., `sm80_xmma_gemm_bf16bf16_...` is a cuBLAS GEMM (likely a linear layer), `flash_fwd_kernel` is FlashAttention, `void rmsnorm_kernel` is RMSNorm.

4. **For each hot kernel, classify as compute-bound or memory-bound** using ncu:
   ```bash
   ncu --set full --kernel-name "<kernel_name>" --launch-skip 20 --launch-count 3 \
       python serve.py --input "test" --max-tokens 1
   ```
   Check SM % vs Memory %. If Memory % > 2x SM %, it is memory-bound. If SM % > 2x Memory %, it is compute-bound.

5. **Calculate the efficiency gap** for each hot kernel:
   - Memory-bound: `gap = 1.0 - (achieved_bandwidth / peak_bandwidth)`
   - Compute-bound: `gap = 1.0 - (achieved_flops / peak_flops)`

6. **Sort kernels by optimization opportunity**: `opportunity = gap * total_time_pct`. This prioritizes kernels that are both slow and inefficient.

7. **For the top opportunity**, select the optimization strategy from Phase 2. Write or adapt a kernel from the templates.

8. **Verify correctness** (Step 8 of Phase 3). If the kernel fails correctness, debug before proceeding.

9. **Benchmark** the custom kernel against the original (Step 9-10 of Phase 3). Record:
   - Original time (us)
   - Custom kernel time (us)
   - Speedup factor
   - Efficiency (% of theoretical peak)

10. **If the custom kernel wins**: integrate into the runtime (Phase 5). If it loses, document why and move to the next opportunity.

11. **Re-profile the full model** after each integration to verify end-to-end improvement. Check for:
    - Did overall latency decrease?
    - Did any other kernel regress (e.g., due to changed memory layout)?
    - Are there new bottlenecks exposed by the optimization?

12. **Repeat from step 2** until no kernel has more than a 2x gap from its theoretical peak, or until the top opportunity's `total_time_pct` is below 3%.

### Termination Criteria

Stop optimizing when ANY of these conditions is true:
- All top-5 kernels are within 30% of their theoretical peak
- The hottest single kernel accounts for less than 15% of total GPU time (well-balanced)
- End-to-end latency improvement from the last round was less than 2%
- All kernels are using best-known implementations (FlashAttention, cuBLAS) and gaps are <20%

### Common Pitfalls

1. **Optimizing a kernel that is not in the critical path**: Always re-profile end-to-end. A kernel taking 10% of GPU time but overlapping with CPU work contributes 0% to latency.

2. **Winning on kernel benchmark but losing end-to-end**: Custom kernels may have higher launch overhead, incompatible memory layouts, or break torch.compile graphs.

3. **Ignoring dtype mismatches**: A custom BF16 kernel may force dtype conversions that negate the speedup.

4. **Over-fusing**: Fusing 5 operators into one kernel increases register pressure and may reduce occupancy below acceptable levels. Fuse 2-3 operators at a time.

5. **Not testing across batch sizes**: A kernel optimized for batch=1 (decode) may be suboptimal for batch=64 (prefill). Use autotune with batch-size as a key.

6. **Ignoring GPU thermals**: Sustained benchmarks may throttle the GPU. Always report stable (post-warmup) numbers.

7. **Forgetting the LM Head**: The LM head GEMM is often overlooked because it only runs once per token, but for large vocabularies it can dominate decode time.

---

## Appendix A: Quick Reference Tables

### A.1 Bytes Per Element by Dtype

| Dtype | Bytes | Range | Use Case |
|---|---|---|---|
| FP32 | 4 | 3.4e38 | Reference, accumulators |
| BF16 | 2 | 3.4e38 (low precision) | Default training/inference |
| FP16 | 2 | 6.5e4 | Inference (watch for overflow) |
| FP8 E4M3 | 1 | 448 | Ada/Hopper/Blackwell inference |
| FP8 E5M2 | 1 | 5.7e4 | Gradients (training) |
| INT8 | 1 | -128 to 127 | Weight/activation quantization |
| INT4 | 0.5 | -8 to 7 | Weight-only quantization |
| FP4 E2M1 | 0.5 | 6 | Blackwell NVFP4 |

### A.2 Correctness Tolerances by Dtype

| Dtype | atol | rtol | Notes |
|---|---|---|---|
| FP32 | 1e-5 | 1e-5 | Gold standard |
| BF16 | 1e-2 | 1.6e-2 | 7 mantissa bits |
| FP16 | 1e-3 | 1e-3 | 10 mantissa bits |
| FP8 E4M3 | 5e-2 | 1e-1 | 3 mantissa bits |
| INT8 | N/A | N/A | Compare dequantized output |
| INT4 | N/A | N/A | Compare dequantized output |

### A.3 SM and Memory Specs by GPU

| GPU | SMs | CUDA Cores | HBM Capacity | HBM BW | Shared Mem/SM | L2 Cache |
|---|---|---|---|---|---|---|
| A100 SXM | 108 | 6912 | 80 GB | 2.0 TB/s | 164 KB | 40 MB |
| RTX 3090 | 82 | 10496 | 24 GB | 936 GB/s | 100 KB | 6 MB |
| RTX 4090 | 128 | 16384 | 24 GB | 1.0 TB/s | 100 KB | 72 MB |
| L40S | 142 | 18176 | 48 GB | 864 GB/s | 100 KB | 96 MB |
| H100 SXM | 132 | 16896 | 80 GB | 3.35 TB/s | 228 KB | 50 MB |
| H200 SXM | 132 | 16896 | 141 GB | 4.8 TB/s | 228 KB | 50 MB |
| B200 SXM | 192 | 18432 | 192 GB | 8.0 TB/s | 256 KB | 96 MB |

### A.4 Tensor Core TFLOPS by GPU and Dtype

| GPU | FP32 | BF16/FP16 | FP8 | INT8 | FP4 |
|---|---|---|---|---|---|
| A100 SXM | 19.5 | 312 | N/A | 624 | N/A |
| RTX 4090 | 82.6 | 330 | 660 | 660 | N/A |
| H100 SXM | 66.9 | 989 | 1979 | 1979 | N/A |
| H200 SXM | 66.9 | 989 | 1979 | 1979 | N/A |
| B200 SXM | 90 | 2250 | 4500 | 4500 | 9000 |

---

## Appendix B: Operator Fusion Dependency Graph

This shows which operators can be fused together in a transformer decoder layer:

```
Fusion Group 1 (Pre-Attention):
  residual_add + rmsnorm [+ optional FP8 quant]
  -> Single Triton kernel, template in Section 2.10

Fusion Group 2 (QKV + RoPE + Attention):
  qkv_projection is a GEMM (leave as cuBLAS unless quantized)
  rope can be fused INTO the attention kernel (apply to Q/K after load)
  attention is a single FlashAttention/FlashInfer call
  -> Net: 2 kernels (QKV GEMM + fused_rope_attention)

Fusion Group 3 (Post-Attention):
  output_projection is a GEMM (leave as cuBLAS)
  residual_add + rmsnorm (same as Fusion Group 1)
  -> Net: 2 kernels (O GEMM + fused_residual_rmsnorm)

Fusion Group 4 (FFN):
  gate_up_projection: fuse into single GEMM with concatenated weights
  silu + elementwise_mul: fuse into single Triton kernel
  down_projection: GEMM (cuBLAS)
  -> Net: 3 kernels (fused_gate_up GEMM + silu_mul + down GEMM)

Fusion Group 5 (Post-FFN):
  residual_add (can fuse with next layer's rmsnorm)
  -> Folds into next layer's Fusion Group 1

Total per layer (optimized):
  Fusion Group 1: 1 kernel (fused_residual_rmsnorm)
  Fusion Group 2: 2 kernels (QKV GEMM + attention_with_rope)
  Fusion Group 3: 2 kernels (O GEMM + fused_residual_rmsnorm)
  Fusion Group 4: 3 kernels (gate_up GEMM + silu_mul + down GEMM)
  = 8 kernels per layer (vs ~14-16 unfused)
```

---

## Appendix C: Profiling Command Cheatsheet

```bash
# Full model timeline
nsys profile --trace=cuda,nvtx,osrt --cuda-memory-usage=true \
  -o model_timeline python inference.py

# Kernel summary sorted by time
nsys stats --report cuda_gpu_kern_sum model_timeline.nsys-rep

# Kernel summary sorted by instance count
nsys stats --report cuda_gpu_kern_sum --format csv model_timeline.nsys-rep | sort -t, -k3 -rn

# Single kernel deep analysis
ncu --set full --kernel-name "rmsnorm" --launch-skip 50 --launch-count 5 \
  -o rmsnorm_analysis python inference.py

# Roofline analysis
ncu --set roofline --kernel-name "rmsnorm" --launch-skip 50 --launch-count 1 \
  python inference.py

# Memory access pattern analysis
ncu --set memory --kernel-name "rmsnorm" --launch-skip 50 --launch-count 1 \
  python inference.py

# Compare two kernel implementations
ncu --set full --kernel-name "rmsnorm_v1|rmsnorm_v2" --launch-skip 50 --launch-count 5 \
  -o comparison python benchmark_both.py

# Triton kernel compilation analysis (see generated PTX/SASS)
TRITON_PRINT_AUTOTUNING=1 python run_triton_kernel.py

# Check Triton cache for compiled kernels
ls ~/.triton/cache/

# Export Triton kernel to PTX for manual inspection
TRITON_CACHE_DIR=/tmp/triton_debug python run_kernel.py
find /tmp/triton_debug -name "*.ptx" | head -5
```

---

## Appendix D: Decision Flowchart

When the agent encounters a hot operator, follow this decision tree:

```
1. Is there an existing optimized library kernel?
   (FlashAttention, FlashInfer, cuBLAS, Marlin, Liger)
   |
   +-- YES: Is it already being used?
   |   +-- YES: Profile with ncu. Is it >70% efficient?
   |   |   +-- YES: Move to next operator.
   |   |   +-- NO: Check if wrong config (tile size, dtype). Fix config first.
   |   +-- NO: Switch to it. Re-benchmark.
   |
   +-- NO: Is the operator memory-bound?
       +-- YES: Write a Triton kernel from template.
       |   Can it be fused with adjacent operators?
       |   +-- YES: Write fused kernel. Bigger win.
       |   +-- NO: Write standalone kernel.
       +-- NO (compute-bound):
           Is it a GEMM variant?
           +-- YES: Use CUTLASS or cuBLAS with custom epilogue.
           +-- NO: Write CuTe DSL kernel for tensor core ops,
                   or Triton kernel for non-tensor-core compute.
```

This document is a living reference. As new hardware (Blackwell, Rubin), new libraries (FlashAttention-4, Triton 3.x), and new techniques emerge, update the templates and performance targets accordingly.
