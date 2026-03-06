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
