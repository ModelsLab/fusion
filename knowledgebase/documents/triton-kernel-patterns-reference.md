---
id: triton_kernel_patterns_reference
kind: document
title: Triton Kernel Patterns - Complete Reference with Code
category: kernel
summary: Comprehensive reference of Triton kernel patterns for LLM inference and training, including fused norms, attention, GEMM, quantization, sampling, and activation kernels with complete code examples.
tags:
  - triton
  - kernel
  - fused-norm
  - softmax
  - attention
  - gemm
  - rope
  - quantization
  - sampling
source_ids:
  - triton-tutorials
  - triton-block-scaled-matmul
gpu_families:
  - Ampere
  - Ada
  - Hopper
  - Blackwell
---

# Triton Kernel Patterns - Complete Reference

## Triton Programming Fundamentals

### Core Concepts
```python
import triton
import triton.language as tl

@triton.jit
def kernel(
    input_ptr,        # Pointer to input tensor
    output_ptr,       # Pointer to output tensor
    n_elements,       # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Compile-time constant
):
    # Each program instance processes one block
    pid = tl.program_id(axis=0)  # Which block am I?

    # Compute pointer offsets for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask for boundary handling
    mask = offsets < n_elements

    # Load, compute, store
    x = tl.load(input_ptr + offsets, mask=mask)
    y = x * 2.0
    tl.store(output_ptr + offsets, y, mask=mask)

# Launch
grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
kernel[grid](input, output, n_elements, BLOCK_SIZE=1024)
```

### Key Triton Operations
```python
# Memory
tl.load(ptr, mask=None, other=0.0)   # Load from global memory
tl.store(ptr, val, mask=None)         # Store to global memory
tl.atomic_add(ptr, val)              # Atomic addition

# Math
tl.dot(a, b)                         # Matrix multiply (uses tensor cores!)
tl.sum(x, axis=0)                    # Reduction
tl.max(x, axis=0)                    # Max reduction
tl.minimum(a, b)                     # Element-wise min
tl.maximum(a, b)                     # Element-wise max
tl.where(cond, a, b)                 # Conditional select
tl.exp(x)                            # Element-wise exp
tl.log(x)                            # Element-wise log
tl.sqrt(x)                           # Element-wise sqrt
tl.sigmoid(x)                        # Sigmoid

# Casting
x.to(tl.float16)                     # Type conversion
x.to(tl.float32)

# Scan
tl.cumsum(x, axis=0)                 # Cumulative sum
tl.associative_scan(combine_fn, x)   # General associative scan
```

## Pattern 1: Fused RMSNorm

```python
@triton.jit
def rmsnorm_kernel(
    X,          # input: (M, N)
    W,          # weight: (N,)
    Y,          # output: (M, N)
    stride_x,   # row stride of X
    stride_y,   # row stride of Y
    N,          # number of columns
    eps,        # epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row
    row = tl.program_id(0)
    X += row * stride_x
    Y += row * stride_y

    # Compute variance: mean(x^2)
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        _var += x * x
    var = tl.sum(_var, axis=0) / N

    # Compute rrms = 1/sqrt(var + eps)
    rrms = 1.0 / tl.sqrt(var + eps)

    # Normalize and multiply by weight
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        y = x * rrms * w
        tl.store(Y + cols, y.to(tl.float16), mask=mask)
```

### Fused RMSNorm + Residual Add
```python
@triton.jit
def rmsnorm_residual_kernel(
    X, Residual, W, Y, ResidualOut,
    stride, N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride
    Residual += row * stride
    Y += row * stride
    ResidualOut += row * stride

    # Add residual first, then normalize
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(Residual + cols, mask=mask, other=0.0).to(tl.float32)
        x = x + r  # residual addition
        tl.store(ResidualOut + cols, x.to(tl.float16), mask=mask)  # save for next residual
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rrms = 1.0 / tl.sqrt(var + eps)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(ResidualOut + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(Y + cols, (x * rrms * w).to(tl.float16), mask=mask)
```

## Pattern 2: Fused Softmax

```python
@triton.jit
def softmax_kernel(
    output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols

    # Load row
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

    # Online softmax in one pass:
    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    # Store
    output_row_start = output_ptr + row_idx * output_row_stride
    tl.store(output_row_start + col_offsets, softmax_output, mask=mask)
```

### Fused Softmax + Causal Mask
```python
@triton.jit
def causal_softmax_kernel(
    output_ptr, input_ptr, stride_row, n_cols,
    query_pos,  # current query position (for causal mask)
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Causal mask: only attend to positions <= query_pos
    causal_mask = col_offsets <= query_pos
    combined_mask = mask & causal_mask

    row = tl.load(input_ptr + row_idx * stride_row + col_offsets,
                  mask=combined_mask, other=-float('inf'))

    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    tl.store(output_ptr + row_idx * stride_row + col_offsets,
             softmax_output, mask=combined_mask)
```

## Pattern 3: Rotary Position Embedding (RoPE)

```python
@triton.jit
def rope_kernel(
    Q, K,            # (batch, seq_len, num_heads, head_dim)
    cos_cache,       # (max_seq, head_dim // 2)
    sin_cache,       # (max_seq, head_dim // 2)
    seq_len,
    head_dim: tl.constexpr,
    HALF_DIM: tl.constexpr,
):
    # Each program handles one (batch, seq, head) triple
    pid = tl.program_id(0)
    batch_head = pid // seq_len
    seq_idx = pid % seq_len

    # Load cos and sin for this position
    cos_offsets = tl.arange(0, HALF_DIM)
    cos = tl.load(cos_cache + seq_idx * HALF_DIM + cos_offsets)
    sin = tl.load(sin_cache + seq_idx * HALF_DIM + cos_offsets)

    # Load Q (first half and second half of head_dim)
    base = batch_head * seq_len * head_dim + seq_idx * head_dim
    q_first = tl.load(Q + base + cos_offsets)        # x[..., :dim//2]
    q_second = tl.load(Q + base + HALF_DIM + cos_offsets)  # x[..., dim//2:]

    # Apply rotation: [x1*cos - x2*sin, x1*sin + x2*cos]
    q_out_first = q_first * cos - q_second * sin
    q_out_second = q_first * sin + q_second * cos

    tl.store(Q + base + cos_offsets, q_out_first)
    tl.store(Q + base + HALF_DIM + cos_offsets, q_out_second)

    # Same for K (if K is not None)
    # ... similar code for K
```

## Pattern 4: SwiGLU Activation Fusion

```python
@triton.jit
def swiglu_fused_kernel(
    gate_ptr,    # output of gate projection: (M, N)
    up_ptr,      # output of up projection: (M, N)
    output_ptr,  # result: (M, N)
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    gate = tl.load(gate_ptr + offsets, mask=mask).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask).to(tl.float32)

    # SwiGLU: silu(gate) * up
    # silu(x) = x * sigmoid(x)
    silu_gate = gate * tl.sigmoid(gate)
    result = silu_gate * up

    tl.store(output_ptr + offsets, result.to(tl.float16), mask=mask)
```

### Fused Gate+Up Projection + SwiGLU (if both projections share input)
```python
@triton.jit
def fused_gate_up_swiglu_kernel(
    # Input X: (M, K), Gate weight: (K, N), Up weight: (K, N)
    # Computes: SwiGLU(X @ W_gate, X @ W_up)
    # This is better done as: compute both GEMMs, then fuse SwiGLU
    # The GEMM should be: X @ [W_gate; W_up] = [gate_out, up_out]
    # Then apply SwiGLU element-wise (this kernel)
    gate_up_ptr,  # (M, 2*N) - concatenated gate and up projections
    output_ptr,   # (M, N)
    M, N,
    stride_gu,     # row stride of gate_up
    stride_out,    # row stride of output
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N

        # Gate output is first N columns
        gate = tl.load(gate_up_ptr + row * stride_gu + cols, mask=mask).to(tl.float32)
        # Up output is second N columns
        up = tl.load(gate_up_ptr + row * stride_gu + N + cols, mask=mask).to(tl.float32)

        result = gate * tl.sigmoid(gate) * up
        tl.store(output_ptr + row * stride_out + cols, result.to(tl.float16), mask=mask)
```

## Pattern 5: Fused Cross-Entropy Loss

```python
@triton.jit
def cross_entropy_fwd_kernel(
    logits_ptr,    # (M, V) - raw logits
    labels_ptr,    # (M,) - target labels
    loss_ptr,      # (M,) - per-sample loss
    V,             # vocabulary size
    stride,        # row stride
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    logits_ptr += row * stride
    label = tl.load(labels_ptr + row)

    # Online softmax over vocabulary (can be very large, e.g., 128K)
    # Pass 1: find max
    _max = -float('inf')
    for off in range(0, V, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < V
        logit = tl.load(logits_ptr + cols, mask=mask, other=-float('inf'))
        _max = tl.maximum(_max, tl.max(logit, axis=0))

    # Pass 2: compute log-sum-exp
    _sum = 0.0
    for off in range(0, V, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < V
        logit = tl.load(logits_ptr + cols, mask=mask, other=-float('inf'))
        _sum += tl.sum(tl.exp(logit - _max), axis=0)

    log_sum_exp = _max + tl.log(_sum)

    # Loss = log_sum_exp - logit[label]
    target_logit = tl.load(logits_ptr + label)
    loss = log_sum_exp - target_logit
    tl.store(loss_ptr + row, loss)
```

## Pattern 6: Triton GEMM (Matrix Multiplication)

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8},
                      num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8},
                      num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)

    # Swizzle for L2 cache locality
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Block start pointers
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # Main loop
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        accumulator += tl.dot(a, b)  # Uses tensor cores!
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Store result
    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)
```

## Pattern 7: Dequantization Kernels

### INT4 Dequantization
```python
@triton.jit
def dequant_int4_kernel(
    packed_ptr,    # (N, K // 8) uint32 packed weights
    scales_ptr,    # (N, K // group_size) fp16 scales
    zeros_ptr,     # (N, K // group_size) int4 zero points (packed)
    output_ptr,    # (N, K) fp16 output
    N, K,
    group_size: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    # Load packed INT4 (8 values per uint32)
    packed_offs = offs_k // 8
    shift = (offs_k % 8) * 4
    packed = tl.load(packed_ptr + offs_n[:, None] * (K // 8) + packed_offs[None, :])
    # Extract 4-bit values
    int4_val = (packed >> shift[None, :]) & 0xF  # 0-15

    # Load scale and zero point for this group
    group_idx = offs_k // group_size
    scale = tl.load(scales_ptr + offs_n[:, None] * (K // group_size) + group_idx[None, :])
    zero = 8  # symmetric quantization around 8 for uint4

    # Dequantize
    fp_val = (int4_val.to(tl.float16) - zero) * scale

    tl.store(output_ptr + offs_n[:, None] * K + offs_k[None, :], fp_val,
             mask=(offs_n[:, None] < N) & (offs_k[None, :] < K))
```

## Pattern 8: Top-p (Nucleus) Sampling

```python
@triton.jit
def top_p_sampling_kernel(
    logits_ptr,    # (B, V) - logits
    output_ptr,    # (B,) - sampled token IDs
    temperature,
    top_p,
    V,
    stride,
    BLOCK_SIZE: tl.constexpr,
):
    batch = tl.program_id(0)
    logits_ptr += batch * stride

    # Apply temperature
    # Note: for full top-p, you need sorting which is hard in Triton
    # This shows the temperature + top-p threshold concept
    # Real implementations typically use CUB sort on GPU

    # Simplified: just do temperature scaling and argmax
    # (Full top-p requires sorted cumsum which needs different approach)
    max_logit = -float('inf')
    for off in range(0, V, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < V
        logit = tl.load(logits_ptr + cols, mask=mask, other=-float('inf'))
        logit = logit / temperature
        block_max = tl.max(logit, axis=0)
        max_logit = tl.maximum(max_logit, block_max)

    # This is a simplified version - real top-p needs sorting
    best_idx = 0
    best_val = -float('inf')
    for off in range(0, V, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < V
        logit = tl.load(logits_ptr + cols, mask=mask, other=-float('inf'))
        local_max = tl.max(logit, axis=0)
        local_argmax = tl.argmax(logit, axis=0)
        if local_max > best_val:
            best_val = local_max
            best_idx = off + local_argmax

    tl.store(output_ptr + batch, best_idx)
```

## Pattern 9: Fused Adam Optimizer

```python
@triton.jit
def fused_adam_kernel(
    param_ptr, grad_ptr, exp_avg_ptr, exp_avg_sq_ptr,
    lr, beta1, beta2, eps, weight_decay,
    step,  # current step number (for bias correction)
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load
    param = tl.load(param_ptr + offsets, mask=mask).to(tl.float32)
    grad = tl.load(grad_ptr + offsets, mask=mask).to(tl.float32)
    exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask).to(tl.float32)
    exp_avg_sq = tl.load(exp_avg_sq_ptr + offsets, mask=mask).to(tl.float32)

    # Weight decay (decoupled)
    if weight_decay > 0:
        param = param * (1.0 - lr * weight_decay)

    # Update biased first moment estimate
    exp_avg = beta1 * exp_avg + (1.0 - beta1) * grad
    # Update biased second moment estimate
    exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * grad * grad

    # Bias correction
    bias_correction1 = 1.0 - beta1 ** step
    bias_correction2 = 1.0 - beta2 ** step

    # Update parameters
    denom = tl.sqrt(exp_avg_sq / bias_correction2) + eps
    param = param - lr * (exp_avg / bias_correction1) / denom

    # Store
    tl.store(param_ptr + offsets, param.to(tl.float16), mask=mask)
    tl.store(exp_avg_ptr + offsets, exp_avg, mask=mask)
    tl.store(exp_avg_sq_ptr + offsets, exp_avg_sq, mask=mask)
```

## Pattern 10: Block-Scaled MatMul (MX Format)

```python
@triton.jit
def block_scaled_matmul_kernel(
    A, B, C,
    scale_A, scale_B,  # per-block scales
    M, N, K,
    block_size: tl.constexpr,  # scaling block size (e.g., 32)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Standard GEMM structure but with block-level rescaling
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        # Load A tile and its scales
        a_tile = tl.load(...)  # BLOCK_M x BLOCK_K in low precision
        a_scales = tl.load(...)  # BLOCK_M x (BLOCK_K // block_size) scales

        # Load B tile and its scales
        b_tile = tl.load(...)  # BLOCK_K x BLOCK_N in low precision
        b_scales = tl.load(...)  # (BLOCK_K // block_size) x BLOCK_N scales

        # Dequantize using block scales
        # Each group of block_size elements shares one scale
        for kb in range(0, BLOCK_K, block_size):
            a_block = a_tile[:, kb:kb+block_size].to(tl.float32)
            b_block = b_tile[kb:kb+block_size, :].to(tl.float32)
            sa = a_scales[:, kb // block_size][:, None]
            sb = b_scales[kb // block_size, :][None, :]
            accumulator += (a_block * sa) @ (b_block * sb)

    # Store result
    tl.store(C + ..., accumulator.to(tl.float16))
```

## Triton Autotuning Best Practices

```python
@triton.autotune(
    configs=[
        # For compute-bound (large M, N):
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64},
                      num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64},
                      num_stages=3, num_warps=8),
        # For memory-bound (small M):
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32},
                      num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],  # Re-tune when these change
    # Can also use restore_value to reset modified tensors between configs
)
```

### Tuning Guidelines:
1. **num_warps**: More warps (8) for compute-bound, fewer (2-4) for memory-bound
2. **num_stages**: More stages hide memory latency but use more shared memory
3. **BLOCK sizes**: Larger blocks improve reuse but reduce parallelism
4. **GROUP_M (L2 swizzle)**: 8 is usually good, helps L2 cache hit rate
5. **Key parameter**: tells autotuner which args affect optimal config

## Triton vs CUDA: When to Use Each

| Criteria | Triton | CUDA/CUTLASS |
|----------|--------|-------------|
| Development speed | Fast (Python) | Slow (C++) |
| Memory-bound kernels | Excellent | Good |
| Compute-bound GEMM | Good (80-95% of cuBLAS) | Best (CUTLASS matches cuBLAS) |
| Shared memory control | Limited | Full control |
| Register management | Automatic | Manual (__launch_bounds__) |
| Bank conflict avoidance | Automatic (usually) | Manual swizzle |
| Warp specialization | Not supported | Full control (Hopper) |
| TMA usage | Limited/experimental | Full support via CuTe |
| Rapid prototyping | Best | Slow |
| Production kernels | Good for most ops | Best for hot-path GEMM |

## End-to-End Kernel Development Workflow

### Step 1: Write Your First Triton Kernel

The simplest useful Triton kernel is vector addition. This teaches you the core pattern: program ID, offsets, mask, load, compute, store.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)

def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    vector_add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output

# --- Run it ---
if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(100_000, device='cuda', dtype=torch.float16)
    y = torch.randn(100_000, device='cuda', dtype=torch.float16)

    # Correctness
    triton_out = vector_add(x, y)
    torch_out = x + y
    assert torch.allclose(triton_out, torch_out), "Mismatch!"
    print("Correctness: PASSED")

    # Benchmark
    ms_triton = triton.testing.do_bench(lambda: vector_add(x, y))
    ms_torch = triton.testing.do_bench(lambda: x + y)
    print(f"Triton: {ms_triton:.4f} ms | PyTorch: {ms_torch:.4f} ms")
```

### Step 2: Write a Useful Kernel - Fused RMSNorm

RMSNorm is used in every modern LLM (Llama, Mistral, Qwen). This kernel fuses the normalization into a single GPU pass instead of multiple PyTorch ops.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_fwd_kernel(
    X, W, Y,
    stride_x, stride_y,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride_x
    Y += row * stride_y

    # Pass 1: compute variance = mean(x^2)
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rrms = 1.0 / tl.sqrt(var + eps)

    # Pass 2: normalize and scale
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        y = x * rrms * w
        tl.store(Y + cols, y.to(tl.float16), mask=mask)

def triton_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    M, N = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)
    BLOCK_SIZE = min(BLOCK_SIZE, 8192)
    grid = (M,)
    rmsnorm_fwd_kernel[grid](
        x, weight, y,
        x.stride(0), y.stride(0),
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y
```

**PyTorch reference implementation for testing:**

```python
def pytorch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return (x_normed * weight).to(x.dtype)
```

**Correctness test script:**

```python
if __name__ == "__main__":
    torch.manual_seed(42)
    for M in [1, 4, 64, 256]:
        for N in [128, 512, 2048, 4096]:
            x = torch.randn(M, N, device='cuda', dtype=torch.float16)
            w = torch.randn(N, device='cuda', dtype=torch.float16)

            ref = pytorch_rmsnorm(x, w)
            out = triton_rmsnorm(x, w)
            torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
    print("All correctness tests PASSED")

    # Performance benchmark
    x = torch.randn(2048, 4096, device='cuda', dtype=torch.float16)
    w = torch.randn(4096, device='cuda', dtype=torch.float16)
    ms_triton = triton.testing.do_bench(lambda: triton_rmsnorm(x, w))
    ms_pytorch = triton.testing.do_bench(lambda: pytorch_rmsnorm(x, w))
    print(f"Triton RMSNorm: {ms_triton:.4f} ms")
    print(f"PyTorch RMSNorm: {ms_pytorch:.4f} ms")
    print(f"Speedup: {ms_pytorch / ms_triton:.2f}x")
```

### Step 3: Integrate Into PyTorch

**torch.autograd.Function wrapper (complete code):**

```python
class TritonRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps=1e-6):
        M, N = x.shape
        y = torch.empty_like(x)
        BLOCK_SIZE = min(triton.next_power_of_2(N), 8192)
        rmsnorm_fwd_kernel[(M,)](
            x, weight, y,
            x.stride(0), y.stride(0),
            N, eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        ctx.BLOCK_SIZE = BLOCK_SIZE
        return y

    @staticmethod
    def backward(ctx, dy):
        # For training, implement backward kernel or fall back to PyTorch
        x, weight = ctx.saved_tensors
        variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        rrms = torch.rsqrt(variance + ctx.eps)
        x_hat = x * rrms
        dw = (dy * x_hat).sum(dim=0)
        dx = dy * weight * rrms
        dx -= x_hat * (dy * weight * x_hat).mean(dim=-1, keepdim=True) * rrms
        return dx, dw, None

def triton_rmsnorm_autograd(x, weight, eps=1e-6):
    return TritonRMSNorm.apply(x, weight, eps)
```

**Drop-in replacement in a model:**

```python
class TritonRMSNormModule(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        orig_shape = x.shape
        x = x.view(-1, orig_shape[-1])
        y = TritonRMSNorm.apply(x, self.weight, self.eps)
        return y.view(orig_shape)

# Swap into an existing model:
# model.layers[0].input_layernorm = TritonRMSNormModule(hidden_size=4096)
```

**torch.compile compatibility (Meta-style implementation):**

```python
# For torch.compile, register a custom op instead of autograd.Function:
@torch.library.custom_op("mylib::rmsnorm", mutates_args=())
def rmsnorm_op(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return triton_rmsnorm(x.view(-1, x.shape[-1]), weight, eps).view(x.shape)

@rmsnorm_op.register_fake
def rmsnorm_fake(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.empty_like(x)

# Now works with torch.compile:
# compiled_model = torch.compile(model)
```

### Step 4: Autotuning Your Kernel

```python
@triton.autotune(
    configs=[
        # Config 1: Small inputs, memory-bound
        # Few warps because there is not enough compute to keep them busy.
        # More pipeline stages to hide memory latency.
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2, num_stages=4),

        # Config 2: Medium inputs, balanced
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=3),

        # Config 3: Large inputs, compute-bound
        # More warps saturate the SMs. Fewer stages because compute dominates.
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),

        # Config 4: Very large inputs
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),

        # Config 5: Max block size
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=1),
    ],
    key=['N'],  # Re-tune when N changes. This is the "shape key".
)
@triton.jit
def autotuned_rmsnorm_kernel(
    X, W, Y,
    stride_x, stride_y,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # ... kernel body identical to rmsnorm_fwd_kernel above ...
    pass
```

**How to choose BLOCK_SIZE, num_warps, num_stages:**

| Parameter | What It Controls | Rule of Thumb |
|-----------|-----------------|---------------|
| BLOCK_SIZE | Elements per program instance | Start at `next_power_of_2(N)`. For reductions over a row, set >= N so one program handles one full row. For element-wise ops, 1024 is a safe default. |
| num_warps | Parallel thread groups per SM (each warp = 32 threads) | 2 for memory-bound (small tensors, simple ops). 4 for balanced. 8 for compute-bound (large matmuls). |
| num_stages | Software pipelining depth (overlaps loads with compute) | 2-3 for compute-bound. 4-5 for memory-bound. Hopper supports more stages than Ampere. More stages = more shared memory usage. |

**How to interpret autotuning results:**

```python
# After running your kernel once, Triton prints the best config:
# "Best config for N=4096: BLOCK_SIZE=1024, num_warps=4, num_stages=2"
#
# To see all configs tested and their timings:
# Set environment variable: TRITON_PRINT_AUTOTUNING=1
#
# To cache results across runs (avoids re-tuning every time):
# Set environment variable: TRITON_CACHE_DIR=~/.triton/cache
#
# If autotuning is slow, reduce configs to 2-3 candidates after
# initial exploration, then hardcode the winner.
```

### Step 5: Debug and Profile

**Common Triton errors and what they mean:**

| Error Message | Cause | Fix |
|--------------|-------|-----|
| `triton.compiler.errors.CompilationError: BLOCK_SIZE must be a power of 2` | `tl.arange` requires power-of-2 size | Use `triton.next_power_of_2(N)` |
| `RuntimeError: Triton Error: CUDA error: misaligned address` | Pointer arithmetic produced unaligned access | Check stride calculations, ensure inputs are contiguous |
| `triton.compiler.errors.CompilationError: incompatible types` | Mixing types in an operation (e.g., fp16 + fp32) | Explicitly cast with `.to(tl.float32)` before math |
| `CUDA error: an illegal memory access was encountered` | Out-of-bounds load or store | Check your `mask` covers all edge cases |
| `triton.compiler.errors.CompilationError: ... is not a tl.constexpr` | Using a runtime value where a compile-time constant is needed | Add `: tl.constexpr` to the parameter |
| `Kernel produced nan values` | Numerical overflow in fp16, or missing eps in division | Upcast to fp32 for accumulation, add eps |
| `error: operation scheduled before its operands` | Compiler scheduling bug (rare) | Try different `num_stages`, update Triton version |

**How to print values inside a kernel (for debugging):**

```python
@triton.jit
def debug_kernel(X, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask)

    # Print from a single program instance to avoid flooding output
    if pid == 0:
        tl.device_print("x values", x)
        tl.device_print("max x", tl.max(x, axis=0))

    # WARNING: tl.device_print is very slow and serializes execution.
    # Only use for debugging, never in production.
    # Remove all prints before benchmarking.
```

**How to profile with ncu (NVIDIA Nsight Compute):**

```bash
# Basic profiling - shows key metrics
ncu --set full python my_kernel_script.py

# Target a specific kernel by name
ncu --kernel-name rmsnorm_fwd_kernel python my_kernel_script.py

# Save report for GUI analysis
ncu -o profile_report --set full python my_kernel_script.py
# Open with: ncu-ui profile_report.ncu-rep

# Key metrics to look at in ncu output:
#   - Memory Throughput (% of peak): >60% is good for memory-bound kernels
#   - Compute (SM) Throughput: >60% is good for compute-bound kernels
#   - Achieved Occupancy: fraction of maximum warps on the SM
#   - L2 Cache Hit Rate: higher is better, affected by GROUP_M swizzle
```

**Performance checklist:**

```
1. Memory throughput
   - Is the kernel memory-bound? Check: compute intensity < SM flops / bandwidth
   - Are you achieving >60% of peak memory bandwidth?
   - Fix: increase BLOCK_SIZE, ensure coalesced access patterns

2. Occupancy
   - Are there enough warps to hide latency?
   - Fix: reduce shared memory usage (fewer num_stages), reduce register pressure

3. Unnecessary memory traffic
   - Are you reading the same data multiple times from global memory?
   - Fix: fuse operations, use single-pass algorithms (e.g., online softmax)

4. Data type
   - Are you accumulating in fp16? Switch to fp32 for accuracy AND speed
     (tensor cores output fp32 anyway)
   - Is the input contiguous? Non-contiguous inputs kill performance

5. Launch overhead
   - For tiny tensors, kernel launch overhead (~5us) dominates
   - Fix: batch operations, use CUDA graphs, or stay on CPU for tiny inputs
```

## Common Mistakes in Triton

| # | Mistake | Error / Symptom | Fix |
|---|---------|----------------|-----|
| 1 | Non-power-of-2 BLOCK_SIZE | `CompilationError: ... must be power of 2` | Always use `triton.next_power_of_2(N)`. Example: `BLOCK_SIZE = min(triton.next_power_of_2(N), 8192)` |
| 2 | Accumulating in fp16 | Silent numerical errors, nan/inf in output | Cast to fp32 before reduction: `x = tl.load(...).to(tl.float32)`, accumulate in fp32, cast back on store |
| 3 | Forgetting the mask on boundary blocks | `CUDA error: illegal memory access` | Always compute `mask = offsets < n_elements` and pass to both `tl.load` and `tl.store` |
| 4 | Wrong stride calculation | Silently wrong results (reads wrong rows/cols) | Use `tensor.stride(0)` from PyTorch, never hardcode. Print shapes and strides to verify. |
| 5 | Using Python `if` instead of `tl.where` for data-dependent branching | `CompilationError` or wrong results | `tl.where(condition, true_val, false_val)` for element-wise conditionals. Python `if` only works on `tl.constexpr` values. |
| 6 | Missing `.to(tl.float32)` before `tl.exp` or `tl.log` | inf/nan in softmax or cross-entropy | Always upcast: `x = x.to(tl.float32)` before transcendental functions |
| 7 | Grid size mismatch | Only part of the output is computed (rest is zeros or garbage) | Grid must cover all work: `grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))`. Print grid size to verify. |
| 8 | Modifying input tensors in an autotuned kernel | Different configs see different inputs, autotuning gives wrong winner | Use `restore_value=['output_ptr']` in `@triton.autotune` for any tensor the kernel writes to |
| 9 | Forgetting that `tl.arange` is unsigned | Negative index bugs when doing `tl.arange(0, BLOCK) - offset` | Cast to signed: `offs = tl.arange(0, BLOCK).to(tl.int32) - offset` |
| 10 | Not making the kernel input-contiguous | Massive slowdown (strided memory access) | Call `x = x.contiguous()` before passing to kernel, or handle strides explicitly in pointer arithmetic |

## Triton Kernel Testing Template

Complete pytest template for testing any Triton kernel with parametrized shapes, dtypes, and integrated performance benchmarking.

```python
"""
test_triton_kernel.py - Template for testing any Triton kernel.

Usage:
    pytest test_triton_kernel.py -v               # correctness tests
    pytest test_triton_kernel.py -v -k benchmark   # performance benchmarks
"""
import pytest
import torch
import triton

# ---------- Import your kernel and reference ----------
# from my_kernels import triton_rmsnorm
# from my_kernels import pytorch_rmsnorm

def triton_rmsnorm(x, weight, eps=1e-6):
    """Replace with your actual Triton kernel wrapper."""
    # ... your implementation ...
    pass

def pytorch_rmsnorm(x, weight, eps=1e-6):
    """Reference implementation in pure PyTorch."""
    variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    return (x * torch.rsqrt(variance + eps) * weight).to(x.dtype)

# ---------- Correctness Tests ----------
SHAPES = [
    (1, 128),
    (1, 4096),
    (4, 1024),
    (32, 2048),
    (128, 4096),
    (256, 8192),
]

DTYPES = [torch.float16, torch.bfloat16, torch.float32]

@pytest.mark.parametrize("M,N", SHAPES, ids=[f"{M}x{N}" for M, N in SHAPES])
@pytest.mark.parametrize("dtype", DTYPES, ids=[str(d) for d in DTYPES])
def test_rmsnorm_correctness(M, N, dtype):
    torch.manual_seed(42)
    x = torch.randn(M, N, device='cuda', dtype=dtype)
    w = torch.ones(N, device='cuda', dtype=dtype)

    ref = pytorch_rmsnorm(x, w)
    out = triton_rmsnorm(x, w)

    # Tolerances: fp16 needs looser tolerance than fp32
    atol = 1e-2 if dtype == torch.float16 else 1e-4
    rtol = 1e-2 if dtype == torch.float16 else 1e-4

    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)

@pytest.mark.parametrize("M,N", SHAPES, ids=[f"{M}x{N}" for M, N in SHAPES])
def test_rmsnorm_random_weights(M, N):
    """Test with non-trivial weight values."""
    torch.manual_seed(123)
    x = torch.randn(M, N, device='cuda', dtype=torch.float16)
    w = torch.randn(N, device='cuda', dtype=torch.float16)

    ref = pytorch_rmsnorm(x, w)
    out = triton_rmsnorm(x, w)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

def test_rmsnorm_zero_input():
    """Edge case: all-zero input should produce all-zero output."""
    x = torch.zeros(4, 128, device='cuda', dtype=torch.float16)
    w = torch.ones(128, device='cuda', dtype=torch.float16)
    out = triton_rmsnorm(x, w)
    assert torch.all(out == 0), "Zero input should produce zero output"

def test_rmsnorm_large_values():
    """Edge case: large values should not overflow in fp16."""
    x = torch.full((4, 128), 100.0, device='cuda', dtype=torch.float16)
    w = torch.ones(128, device='cuda', dtype=torch.float16)
    out = triton_rmsnorm(x, w)
    assert not torch.any(torch.isnan(out)), "Should not produce NaN"
    assert not torch.any(torch.isinf(out)), "Should not produce Inf"

# ---------- Performance Benchmarks ----------
BENCH_SHAPES = [
    (1, 4096),       # Single token (inference decode)
    (32, 4096),      # Small batch
    (512, 4096),     # Medium batch
    (2048, 4096),    # Large batch (training)
    (2048, 8192),    # Large hidden dim
]

@pytest.mark.parametrize("M,N", BENCH_SHAPES, ids=[f"{M}x{N}" for M, N in BENCH_SHAPES])
def test_benchmark_rmsnorm(M, N):
    """Benchmark Triton kernel vs PyTorch baseline."""
    torch.manual_seed(42)
    x = torch.randn(M, N, device='cuda', dtype=torch.float16)
    w = torch.ones(N, device='cuda', dtype=torch.float16)

    ms_triton = triton.testing.do_bench(lambda: triton_rmsnorm(x, w))
    ms_pytorch = triton.testing.do_bench(lambda: pytorch_rmsnorm(x, w))

    speedup = ms_pytorch / ms_triton
    print(f"\n  [{M}x{N}] Triton: {ms_triton:.4f}ms | PyTorch: {ms_pytorch:.4f}ms | Speedup: {speedup:.2f}x")

    # Assert Triton is not dramatically slower (sanity check)
    assert speedup > 0.5, f"Triton kernel is more than 2x slower than PyTorch ({speedup:.2f}x)"
```
