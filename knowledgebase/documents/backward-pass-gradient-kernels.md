---
id: backward_pass_gradient_kernels
kind: document
title: Backward Pass and Gradient Computation Kernels
category: kernel
summary: Deep dive into GPU kernels for backward pass computation including gradient GEMM, attention backward, activation recomputation, mixed-precision gradient handling, and LoRA/QLoRA gradient patterns.
tags:
  - backward-pass
  - gradient
  - training
  - fine-tuning
  - lora
  - activation-checkpointing
  - mixed-precision
source_ids: []
operators:
  - matmul
  - attention
  - layernorm
  - softmax
gpu_families:
  - Ampere
  - Ada
  - Hopper
  - Blackwell
---

# Backward Pass and Gradient Computation Kernels

> A comprehensive reference on GPU kernels that power the backward pass in LLM training and fine-tuning: gradient GEMMs, attention backward, normalization gradients, mixed-precision handling, activation checkpointing, LoRA/QLoRA gradient flow, distributed gradient reduction, and optimizer kernels.

---

## Table of Contents

1. [Backward Pass GEMM Patterns](#1-backward-pass-gemm-patterns)
2. [Attention Backward Kernel](#2-attention-backward-kernel)
3. [LayerNorm and RMSNorm Backward](#3-layernorm-and-rmsnorm-backward)
4. [Softmax Backward](#4-softmax-backward)
5. [Mixed-Precision Training Gradients](#5-mixed-precision-training-gradients)
6. [Activation Checkpointing](#6-activation-checkpointing)
7. [LoRA Gradient Computation](#7-lora-gradient-computation)
8. [QLoRA: Gradients Through Quantization](#8-qlora-gradients-through-quantization)
9. [Gradient Accumulation and All-Reduce](#9-gradient-accumulation-and-all-reduce)
10. [Optimizer Kernels](#10-optimizer-kernels)

---

## 1. Backward Pass GEMM Patterns

For a linear layer `Y = X @ W^T + b` with input `X` of shape `[B, M, K]`, weight `W` of shape `[N, K]`, and output `Y` of shape `[B, M, N]`, the backward pass requires two GEMMs:

### 1.1 Weight Gradient: dW = X^T @ dY

```
dW [N, K] = dY^T [N, M] @ X [M, K]      (accumulated over batch)
```

This is a **reduction GEMM** -- it sums contributions across the batch and sequence dimensions. Memory access pattern: `dY` is read column-major (transposed), `X` is read row-major. The result `dW` has the same shape as `W`.

### 1.2 Input Gradient: dX = dY @ W

```
dX [B, M, K] = dY [B, M, N] @ W [N, K]
```

This is a **propagation GEMM** -- it broadcasts `W` across the batch. Memory access: both `dY` and `W` are read row-major. `dX` propagates gradients to the previous layer.

### 1.3 Triton Backward GEMM Kernel

```python
@triton.jit
def matmul_backward_dw_kernel(
    X_ptr, dY_ptr, dW_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_dym, stride_dyn,
    stride_dwn, stride_dwk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Compute dW = X^T @ dY, tiled over [N, K] output."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    # Accumulate in FP32 for numerical stability
    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for m_start in range(0, M, BLOCK_M):
        m_offs = m_start + tl.arange(0, BLOCK_M)
        n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        k_offs = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        # Load dY^T block: [BLOCK_N, BLOCK_M] from dY [M, N] transposed
        dy_ptrs = dY_ptr + m_offs[None, :] * stride_dym + n_offs[:, None] * stride_dyn
        dy_block = tl.load(dy_ptrs, mask=(m_offs[None, :] < M) & (n_offs[:, None] < N), other=0.0)
        # Load X block: [BLOCK_M, BLOCK_K]
        x_ptrs = X_ptr + m_offs[:, None] * stride_xm + k_offs[None, :] * stride_xk
        x_block = tl.load(x_ptrs, mask=(m_offs[:, None] < M) & (k_offs[None, :] < K), other=0.0)
        # Accumulate: [BLOCK_N, BLOCK_M] @ [BLOCK_M, BLOCK_K]
        acc += tl.dot(dy_block, x_block)
    # Store dW block
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offs = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    dw_ptrs = dW_ptr + n_offs[:, None] * stride_dwn + k_offs[None, :] * stride_dwk
    tl.store(dw_ptrs, acc.to(tl.float16), mask=(n_offs[:, None] < N) & (k_offs[None, :] < K))
```

### 1.4 Memory Analysis

| GEMM | Reads | Writes | Arithmetic Intensity |
|------|-------|--------|---------------------|
| dW = X^T @ dY | `2*B*M*K + 2*B*M*N` bytes (FP16) | `2*N*K` bytes | `2*B*M*N*K / (2*B*M*(K+N) + 2*N*K)` |
| dX = dY @ W | `2*B*M*N + 2*N*K` bytes | `2*B*M*K` bytes | `2*B*M*N*K / (2*B*M*(N+K) + 2*N*K)` |

The dW kernel has higher arithmetic intensity because `W` is much smaller than the activation tensors. The dX kernel broadcasts `W` across the batch, making it more memory-bound for small N.

---

## 2. Attention Backward Kernel

### 2.1 FlashAttention Backward Algorithm

FlashAttention avoids materializing the full `S = Q @ K^T` attention matrix in the forward pass, storing only the output `O`, log-sum-exp `L`, and using **recomputation** in the backward pass.

**Saved from forward:** `Q, K, V, O, L` where `L[i] = log(sum_j(exp(s_ij - m_i)))` and `m_i = max_j(s_ij)`.

**Backward inputs:** `dO` (gradient of output)

**Backward outputs:** `dQ, dK, dV`

### 2.2 Algorithm (Tiled, Two-Pass)

```
# Precompute D_i = rowsum(dO_i * O_i)  -- needed for dS computation
for each block i:
    D_i = sum_j(dO[i,j] * O[i,j])    # shape [Br]

# Main backward loop: iterate over K/V blocks (outer), Q blocks (inner)
for each KV block j:
    Load K_j, V_j                      # [Bc, d]
    Initialize dK_j = 0, dV_j = 0     # [Bc, d]
    for each Q block i:
        Load Q_i, O_i, dO_i, L_i, D_i
        # Recompute attention: S_ij = Q_i @ K_j^T
        S_ij = Q_i @ K_j^T             # [Br, Bc]
        P_ij = exp(S_ij - L_i)         # Recomputed softmax probabilities
        # Gradient of V: dV_j += P_ij^T @ dO_i
        dV_j += P_ij^T @ dO_i          # [Bc, d]
        # Gradient of P: dP_ij = dO_i @ V_j^T
        dP_ij = dO_i @ V_j^T           # [Br, Bc]
        # Gradient of S (softmax backward): dS_ij = P_ij * (dP_ij - D_i)
        dS_ij = P_ij * (dP_ij - D_i)   # [Br, Bc]
        # Gradient of Q: dQ_i += dS_ij @ K_j
        dQ_i += dS_ij @ K_j            # [Br, d]
        # Gradient of K: dK_j += dS_ij^T @ Q_i
        dK_j += dS_ij^T @ Q_i          # [Bc, d]
    Store dK_j, dV_j
```

### 2.3 Memory Savings

| Approach | Forward Memory | Backward Extra Memory | Total |
|----------|---------------|----------------------|-------|
| Standard attention | O(N^2) for S, P | O(N^2) for dS, dP | O(N^2) |
| FlashAttention | O(N) for L, m | O(N) for D + block recompute | O(N) |

For sequence length N=8192, d=128, FP16: standard stores 8192^2 * 2 = 128 MB per head; FlashAttention stores 8192 * 4 = 32 KB for L per head. The backward recomputation cost is roughly 25% additional FLOPs (one extra Q@K^T matmul per tile).

### 2.4 FlashAttention-2 Backward Optimizations

- **Reduced non-matmul FLOPs**: The D precomputation pass avoids redundant work in the main loop.
- **Better parallelism**: Outer loop over KV blocks allows parallelism across both batch and KV-block dimensions.
- **Warp-level partitioning**: On Ampere, 4 warps split the d dimension for the matmuls; on Hopper, warp specialization separates producer (TMA loads) from consumer (tensor core matmuls) warps.

---

## 3. LayerNorm and RMSNorm Backward

### 3.1 LayerNorm Backward Formulas

Forward: `y = (x - mu) / sigma * gamma + beta` where `mu = mean(x)`, `sigma = sqrt(var(x) + eps)`.

Backward (given `dy`):

```
dgamma = sum_batch(dy * (x - mu) / sigma)        # [D]
dbeta  = sum_batch(dy)                             # [D]
dx     = (1/sigma) * (dy * gamma - mean(dy * gamma) - (x_hat) * mean(dy * gamma * x_hat))
       where x_hat = (x - mu) / sigma
```

### 3.2 Fused LayerNorm Backward Triton Kernel

```python
@triton.jit
def layernorm_backward_kernel(
    dY_ptr, X_ptr, gamma_ptr, Mean_ptr, Rstd_ptr,
    dX_ptr, dgamma_ptr, dbeta_ptr,
    N, D,
    stride_n, stride_d,
    BLOCK_D: tl.constexpr,
):
    """Fused LayerNorm backward: computes dX, dgamma, dbeta in one pass."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    # Load row data
    dy = tl.load(dY_ptr + row * stride_n + cols * stride_d, mask=mask, other=0.0).to(tl.float32)
    x  = tl.load(X_ptr + row * stride_n + cols * stride_d, mask=mask, other=0.0).to(tl.float32)
    g  = tl.load(gamma_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(Mean_ptr + row)
    rstd = tl.load(Rstd_ptr + row)
    # Normalized input
    x_hat = (x - mean) * rstd
    # Partial gradients for dgamma, dbeta (atomically accumulated across rows)
    tl.atomic_add(dgamma_ptr + cols, (dy * x_hat).to(tl.float32), mask=mask)
    tl.atomic_add(dbeta_ptr + cols, dy.to(tl.float32), mask=mask)
    # dx computation: requires two reductions over D
    dy_g = dy * g
    mean_dy_g = tl.sum(dy_g, axis=0) / D
    mean_dy_g_xhat = tl.sum(dy_g * x_hat, axis=0) / D
    dx = rstd * (dy_g - mean_dy_g - x_hat * mean_dy_g_xhat)
    tl.store(dX_ptr + row * stride_n + cols * stride_d, dx.to(tl.float16), mask=mask)
```

### 3.3 RMSNorm Backward

RMSNorm omits the mean subtraction: `y = x / rms * gamma` where `rms = sqrt(mean(x^2) + eps)`.

```
dx = (1/rms) * (dy * gamma - x_hat * mean(dy * gamma * x_hat))
   where x_hat = x / rms
dgamma = sum_batch(dy * x / rms)
```

RMSNorm backward is simpler (one fewer reduction) and saves ~10-15% kernel time vs LayerNorm backward.

---

## 4. Softmax Backward

### 4.1 The Jacobian Trick

For `y = softmax(x)`, the Jacobian is `J_ij = y_i * (delta_ij - y_j)`.

Naively: `dx_i = sum_j(J_ij * dy_j)` requires O(N^2) work.

**The trick**: factor the computation:
```
s = sum_j(y_j * dy_j)          # scalar dot product
dx_i = y_i * (dy_i - s)        # elementwise, O(N)
```

This reduces the Jacobian-vector product from O(N^2) to O(N).

### 4.2 Fused Softmax + Cross-Entropy Backward

When softmax is followed by cross-entropy loss, the combined gradient simplifies dramatically:

```
L = -sum_i(t_i * log(y_i))     # cross-entropy where t is one-hot target
dy_i = y_i - t_i                # gradient w.r.t. logits -- no Jacobian needed!
```

This is one of the most important fusions in training. It avoids:
1. Storing the full softmax output (can be vocab_size = 128K+ per token)
2. Computing the Jacobian-vector product entirely
3. Numerical instability from dividing by small softmax values

```python
@triton.jit
def fused_cross_entropy_backward_kernel(
    logits_ptr, targets_ptr, dlogits_ptr, loss_scale,
    N, V,  # N = num tokens, V = vocab size
    stride_n,
    BLOCK_V: tl.constexpr,
):
    """Fused softmax + cross-entropy backward: dL/d(logits) = softmax(logits) - one_hot(target)."""
    row = tl.program_id(0)
    target = tl.load(targets_ptr + row)
    # Online softmax computation over vocabulary
    m = -float('inf')
    s = 0.0
    for v_start in range(0, V, BLOCK_V):
        v_offs = v_start + tl.arange(0, BLOCK_V)
        mask = v_offs < V
        logit = tl.load(logits_ptr + row * stride_n + v_offs, mask=mask, other=-float('inf'))
        new_m = tl.maximum(m, tl.max(logit, axis=0))
        s = s * tl.exp(m - new_m) + tl.sum(tl.exp(logit - new_m), axis=0)
        m = new_m
    log_sum = m + tl.log(s)
    # Second pass: compute and store gradient
    for v_start in range(0, V, BLOCK_V):
        v_offs = v_start + tl.arange(0, BLOCK_V)
        mask = v_offs < V
        logit = tl.load(logits_ptr + row * stride_n + v_offs, mask=mask, other=0.0)
        p = tl.exp(logit - log_sum)       # softmax probability
        is_target = (v_offs == target).to(tl.float32)
        grad = (p - is_target) * loss_scale
        tl.store(dlogits_ptr + row * stride_n + v_offs, grad, mask=mask)
```

---

## 5. Mixed-Precision Training Gradients

### 5.1 The Three-Copy Pattern

Standard mixed-precision training maintains:

| Copy | Precision | Purpose | Size (7B model) |
|------|-----------|---------|-----------------|
| Master weights | FP32 | Optimizer updates | 28 GB |
| Working weights | FP16/BF16 | Forward + backward | 14 GB |
| Gradients | FP16/BF16 | Backward output, accumulated in FP32 | 14 GB (transient) |

### 5.2 Loss Scaling and Gradient Flow

```
Forward:   X_fp16 --[W_fp16]--> Y_fp16 --[loss]--> L_fp32
                                                      |
Scale:                                          L_scaled = L * scale
                                                      |
Backward:  dX_fp16 <--[W_fp16^T]-- dY_fp16 <-------- |
                                                      |
Unscale:   dW_fp32 = dW_fp16 / scale    (before optimizer)
                                                      |
Update:    W_fp32_master -= lr * dW_fp32
           W_fp16 = cast(W_fp32_master)
```

### 5.3 Gradient Accumulation in FP32

When accumulating gradients across micro-batches, FP16 accumulation loses precision rapidly. The kernel pattern:

```python
@triton.jit
def gradient_accumulate_fp32_kernel(
    grad_fp16_ptr, grad_acc_fp32_ptr, N,
    BLOCK: tl.constexpr,
):
    """Accumulate FP16 gradients into FP32 buffer."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    g16 = tl.load(grad_fp16_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    acc = tl.load(grad_acc_fp32_ptr + offs, mask=mask, other=0.0)
    tl.store(grad_acc_fp32_ptr + offs, acc + g16, mask=mask)
```

### 5.4 BF16 vs FP16 for Backward Pass

| Property | FP16 | BF16 |
|----------|------|------|
| Dynamic range | 6e-8 to 65504 | Same as FP32 |
| Requires loss scaling | Yes (gradients underflow) | Usually no |
| Gradient precision | Higher (10-bit mantissa) | Lower (7-bit mantissa) |
| Accumulation concern | Overflow risk | Precision loss |
| Recommendation | Use with GradScaler | Preferred for LLM training |

---

## 6. Activation Checkpointing

### 6.1 The Memory-Compute Tradeoff

Without checkpointing, all intermediate activations must be stored for the backward pass. For a transformer with L layers, sequence length S, hidden dim H, batch B:

```
Memory per layer (approximate):
  - Attention input:           2 * B * S * H bytes (FP16)
  - QKV projections:           3 * 2 * B * S * H
  - Attention output (pre-proj): 2 * B * S * H
  - MLP input:                 2 * B * S * H
  - MLP intermediate:          2 * B * S * 4H  (for SwiGLU: 2 * B * S * (8/3)H)
  - Residuals, norms:          ~4 * B * S * H
  Total per layer:            ~24 * B * S * H bytes  (FP16)
```

For Llama-70B (H=8192, L=80), B=1, S=4096: ~24 * 1 * 4096 * 8192 * 80 = 60 GB of activations.

### 6.2 Segment-Level vs Layer-Level Checkpointing

**Layer-level** (checkpoint every layer): Discard all intermediate activations within each layer. Recompute during backward. Memory = O(L * B * S * H) for just the layer inputs. Extra compute = 1 full forward pass.

**Segment-level** (checkpoint every k layers): Store activations at segment boundaries. Memory reduction = k-fold. Extra compute = (k-1)/k of a forward pass.

```
Optimal segment size k = sqrt(L)

For L=80 layers:
  k = 9 segments of ~9 layers each
  Memory: 80/9 ~ 9x reduction in activation memory
  Extra compute: ~33% of forward pass (only recompute within segments)
```

### 6.3 Selective Checkpointing

Modern implementations selectively recompute only cheap operations (norms, activations) while storing expensive ones (matmul outputs):

```python
# PyTorch selective checkpointing policy
def checkpoint_policy(ctx, op, *args, **kwargs):
    # Keep matmul results, recompute everything else
    if op in (torch.ops.aten.mm, torch.ops.aten.addmm, torch.ops.aten.bmm):
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE
```

---

## 7. LoRA Gradient Computation

### 7.1 Low-Rank Gradient Flow

LoRA replaces a weight update `dW` with low-rank factors: `W' = W + alpha/r * B @ A` where `A` is `[r, K]`, `B` is `[N, r]`, and `W` is frozen `[N, K]`.

Forward: `Y = X @ W^T + (alpha/r) * X @ A^T @ B^T`

Backward (only A and B require gradients):

```
dB = (alpha/r) * dY^T @ X @ A^T     # [N, r] -- gradient for B
   = (alpha/r) * dY^T @ Z           # where Z = X @ A^T [B*M, r] is cached

dA = (alpha/r) * B^T @ dY^T @ X     # [r, K] -- gradient for A
   = (alpha/r) * (B^T @ dY^T) @ X   # compute B^T @ dY^T first: [r, B*M]
```

### 7.2 Memory Savings Analysis

| Component | Full Fine-Tuning | LoRA (r=16) |
|-----------|-----------------|-------------|
| Trainable params (7B model) | 7B | ~10M (0.14%) |
| Gradient memory (FP16) | 14 GB | ~20 MB |
| Optimizer state (Adam FP32) | 56 GB (2x FP32 moments) | ~80 MB |
| Activation memory | Full | Same (activations still needed) |
| Total training memory | ~98 GB+ | ~28 GB (frozen W in FP16 + activations) |

### 7.3 Rank Selection and Gradient Dynamics

The effective gradient rank determines information capacity. For rank r:

```
Gradient update per step: dW_eff = B @ A   (rank at most r)
Full gradient dW:          rank up to min(N, K)

Information bottleneck: r << min(N, K)
  - r=4:  Good for simple adaptations (single task)
  - r=16: Standard for instruction tuning
  - r=64: Approaches full fine-tuning quality for complex tasks
  - r=256: Diminishing returns, consider full fine-tuning
```

### 7.4 LoRA Backward Kernel

```python
@triton.jit
def lora_backward_dB_kernel(
    dY_ptr, Z_ptr, dB_ptr,  # Z = X @ A^T, precomputed
    M, N, R,  # M = seq*batch, N = out_dim, R = rank
    alpha_over_r,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_R: tl.constexpr,
):
    """Compute dB = (alpha/r) * dY^T @ Z, output [N, R]."""
    pid_n = tl.program_id(0)
    pid_r = tl.program_id(1)
    acc = tl.zeros((BLOCK_N, BLOCK_R), dtype=tl.float32)
    for m_start in range(0, M, BLOCK_M):
        m_offs = m_start + tl.arange(0, BLOCK_M)
        n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        r_offs = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        # dY^T block: [BLOCK_N, BLOCK_M]
        dy = tl.load(dY_ptr + m_offs[None, :] * N + n_offs[:, None],
                     mask=(m_offs[None, :] < M) & (n_offs[:, None] < N), other=0.0)
        # Z block: [BLOCK_M, BLOCK_R]
        z = tl.load(Z_ptr + m_offs[:, None] * R + r_offs[None, :],
                    mask=(m_offs[:, None] < M) & (r_offs[None, :] < R), other=0.0)
        acc += tl.dot(dy, z)
    acc *= alpha_over_r
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    r_offs = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    tl.store(dB_ptr + n_offs[:, None] * R + r_offs[None, :],
             acc.to(tl.float16), mask=(n_offs[:, None] < N) & (r_offs[None, :] < R))
```

---

## 8. QLoRA: Gradients Through Quantization

### 8.1 Architecture

QLoRA keeps the base model in 4-bit NormalFloat (NF4) quantization and trains FP16 LoRA adapters on top:

```
Forward:
  W_dequant = dequantize_nf4(W_4bit, scale, zero_point)   # FP16
  Y = X @ W_dequant^T + (alpha/r) * X @ A^T @ B^T

Backward:
  dY flows through both paths, but only LoRA adapters get gradient updates.
  The frozen W path: dX = dY @ W_dequant  (dequantize on the fly, no W gradient)
  The LoRA path: dA, dB computed as in standard LoRA
```

### 8.2 Double Quantization

QLoRA applies quantization to the quantization constants themselves:

```
Standard:  W_4bit + FP16 scale per block (64 elements)
           Overhead: 16 bits / 64 = 0.25 bits/param

Double:    W_4bit + FP8 scale per block + FP32 scale per 256 blocks
           Overhead: 8/64 + 32/(64*256) = 0.127 bits/param
           Savings: ~0.4 GB for a 65B model
```

### 8.3 Gradient Flow Through Dequantization

The key insight: gradients do NOT flow through the quantized weights. The quantized `W` is treated as a frozen constant. Only LoRA parameters receive gradients:

```
dL/dB = dL/dY * dY/dB    (standard LoRA gradient)
dL/dA = dL/dY * dY/dA    (standard LoRA gradient)
dL/dW_4bit = 0            (frozen, no gradient)
```

The dequantization is performed on-the-fly during both forward and backward to avoid storing the full FP16 weight matrix:

```
Memory for 70B model:
  Full FP16:    140 GB
  QLoRA (NF4):  ~35 GB (4 bits/param + quantization overhead)
  LoRA adapters: ~100 MB (r=16)
  Optimizer:     ~400 MB (Adam states for LoRA only)
  Total:         ~40 GB (fits on single 48GB GPU)
```

### 8.4 Paged Optimizers in QLoRA

When GPU memory is tight, QLoRA uses CUDA unified memory to page optimizer states to CPU:

```
GPU memory spike during backward -> optimizer states paged to CPU
Optimizer step -> states paged back to GPU on demand
Cost: ~10% throughput overhead from PCIe transfers
```

---

## 9. Gradient Accumulation and All-Reduce

### 9.1 Gradient Accumulation Over Micro-Batches

When the effective batch size exceeds GPU memory, split into micro-batches:

```python
optimizer.zero_grad()
for i, micro_batch in enumerate(micro_batches):
    loss = model(micro_batch) / num_micro_batches  # normalize loss
    loss.backward()    # gradients accumulate in .grad buffers
    # No optimizer.step() or zero_grad() between micro-batches
optimizer.step()
```

The `.backward()` call adds to existing `.grad` tensors. This is a simple `grad += new_grad` elementwise kernel. With FP16 gradients, accumulation should use FP32 buffers to avoid precision loss over many micro-steps.

### 9.2 Overlapping Backward with All-Reduce

In data-parallel training, gradient all-reduce can overlap with backward computation using bucketed communication:

```
Timeline (without overlap):
  |--- backward pass ---|--- all-reduce ---|--- optimizer ---|

Timeline (with overlap):
  |--- backward layer N ---|--- backward layer N-1 ---|--- backward ... ---|
       |--- all-reduce bucket 1 ---|--- all-reduce bucket 2 ---|
```

### 9.3 Bucketed All-Reduce

PyTorch DDP groups gradients into buckets (default 25 MB) and triggers all-reduce as soon as a bucket is full:

```
Bucket formation (reverse parameter order, matching backward execution):
  Bucket 0: [layer_N.weight.grad, layer_N.bias.grad, ...]     (first to complete)
  Bucket 1: [layer_N-1.weight.grad, ...]
  ...

All-reduce algorithm per bucket (ring all-reduce for large tensors):
  Communication volume per GPU: 2 * (P-1)/P * bucket_size
  For P=8 GPUs, 25MB bucket: 2 * 7/8 * 25MB = 43.75 MB per bucket
  With NVLink (600 GB/s on H100): 43.75MB / 600GB/s = 73 us per bucket
```

### 9.4 Gradient Compression

For bandwidth-limited clusters, gradient compression reduces communication:

```
FP16 gradients:         2 bytes/param
PowerSGD (rank 4):      ~8*d / (d1*d2) compression ratio
1-bit Adam:             1 bit/param + error feedback (2x compression)
Top-K sparsification:   k/N compression + indices overhead
```

---

## 10. Optimizer Kernels

### 10.1 Fused AdamW Kernel

The standard Adam update requires 5 elementwise operations per parameter. A fused kernel does it in one pass:

```python
@triton.jit
def fused_adamw_kernel(
    param_ptr, grad_ptr, exp_avg_ptr, exp_avg_sq_ptr,
    N, lr, beta1, beta2, eps, weight_decay, step,
    BLOCK: tl.constexpr,
):
    """Fused AdamW: single kernel for full optimizer step."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    # Load everything
    p = tl.load(param_ptr + offs, mask=mask).to(tl.float32)
    g = tl.load(grad_ptr + offs, mask=mask).to(tl.float32)
    m = tl.load(exp_avg_ptr + offs, mask=mask)
    v = tl.load(exp_avg_sq_ptr + offs, mask=mask)
    # Update moments
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g * g
    # Bias correction
    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step
    m_hat = m / bc1
    v_hat = v / bc2
    # Weight decay (decoupled) + parameter update
    p = p * (1 - lr * weight_decay) - lr * m_hat / (tl.sqrt(v_hat) + eps)
    # Store
    tl.store(param_ptr + offs, p.to(tl.float16), mask=mask)
    tl.store(exp_avg_ptr + offs, m, mask=mask)
    tl.store(exp_avg_sq_ptr + offs, v, mask=mask)
```

**Performance**: The fused kernel reads 4 tensors and writes 3 tensors in a single pass. Unfused PyTorch requires 5+ kernel launches and 10+ global memory round trips. Measured speedup: 2-3x for optimizer step.

### 10.2 8-bit Optimizers (bitsandbytes)

bitsandbytes stores optimizer states in INT8 with block-wise dynamic quantization:

```
Standard Adam states:  2 * 4 bytes/param = 8 bytes/param  (FP32 m, v)
8-bit Adam states:     2 * 1 byte/param + scales = ~2.1 bytes/param

Memory savings for 70B model:
  Standard: 70B * 8 = 560 GB
  8-bit:    70B * 2.1 = 147 GB  (3.8x reduction)
```

The quantization is block-wise with dynamic ranges:

```
For each block of 2048 elements:
  absmax = max(|values|)
  int8_values = round(values / absmax * 127)
  # Store absmax as FP32 scale (4 bytes per 2048 elements = 0.002 bytes/param)
```

During the optimizer step, states are dequantized to FP32, the update is computed, and states are re-quantized. The dequantize-compute-quantize is fused into a single kernel.

### 10.3 Optimizer State Partitioning (ZeRO)

ZeRO Stage 1 partitions optimizer states across data-parallel ranks:

```
Per-GPU optimizer memory with P GPUs:
  Full Adam:  2 * 4 * num_params bytes  (m + v in FP32)
  ZeRO-1:     2 * 4 * num_params / P bytes

After local optimizer step on partitioned states:
  All-gather updated parameters: communication = 2 * num_params bytes (FP16)
```

ZeRO Stage 2 additionally partitions gradients, and Stage 3 partitions parameters themselves. Each stage trades communication for memory:

| ZeRO Stage | Memory per GPU | Communication Overhead |
|------------|---------------|----------------------|
| None | 2P (params) + 2P (grads) + 12P (Adam states) = 16P | 0 |
| Stage 1 | 2P + 2P + 12P/N_gpu | ~0 (optimizer states gathered rarely) |
| Stage 2 | 2P + 2P/N_gpu + 12P/N_gpu | Scatter-reduce replaces all-reduce |
| Stage 3 | (2P + 2P + 12P) / N_gpu | All-gather params in forward + backward |

Where P = model parameters in bytes (FP16), N_gpu = number of GPUs.

---

## Summary: Backward Pass Kernel Performance Characteristics

| Kernel | Compute Bound? | Key Optimization | Typical % of Backward Time |
|--------|---------------|------------------|---------------------------|
| dW GEMM | Yes (large M) | Tile size tuning, split-K | 25-30% |
| dX GEMM | Depends on N | Persistent kernels on Hopper | 20-25% |
| Attention backward | Yes | FlashAttention recompute | 25-35% |
| LayerNorm backward | Memory bound | Fused with residual add | 3-5% |
| Softmax + CE backward | Memory bound | Fused into single kernel | 1-2% |
| Optimizer step | Memory bound | Fused multi-tensor apply | 5-8% |
| All-reduce | Communication | Overlap with backward | (overlapped) |

The backward pass typically takes 2x the time of the forward pass due to the two GEMM operations per layer plus the attention recomputation cost from FlashAttention. With full overlap of communication, the backward pass wall time approaches the pure compute cost.
