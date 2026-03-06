---
id: triton_fused_attention_example
kind: example
title: Triton Fused Multi-Head Attention
category: kernel
summary: Complete Triton implementation of fused multi-head attention with causal masking, demonstrating the online softmax trick used in FlashAttention.
tags:
  - triton
  - attention
  - flashattention
  - online-softmax
  - causal
source_ids:
  - flashattention-paper
operators:
  - attention
  - softmax
gpu_families:
  - Ampere
  - Ada
  - Hopper
precision:
  - fp16
  - bf16
---

## Fused Attention Kernel (FlashAttention-style)

```python
import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_attention_kernel(
    Q, K, V, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_z = tl.program_id(1) // H
    pid_h = tl.program_id(1) % H

    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to Q block (loaded once)
    q_ptrs = Q + pid_z * stride_qz + pid_h * stride_qh + \
             offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < N_CTX) & (offs_k[None, :] < BLOCK_K), other=0.0)

    # Scale
    sm_scale = 1.0 / tl.sqrt(BLOCK_K * 1.0)

    # Online softmax accumulators
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)

    # Loop over K, V blocks
    for start_n in range(0, N_CTX, BLOCK_N):
        cur_offs_n = start_n + offs_n

        # Causal mask: skip blocks entirely above diagonal
        if IS_CAUSAL:
            if start_n > pid_m * BLOCK_M + BLOCK_M - 1:
                break

        # Load K block
        k_ptrs = K + pid_z * stride_kz + pid_h * stride_kh + \
                 cur_offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=(cur_offs_n[:, None] < N_CTX) & (offs_k[None, :] < BLOCK_K), other=0.0)

        # Compute QK^T
        qk = tl.dot(q, tl.trans(k)) * sm_scale  # [BLOCK_M, BLOCK_N]

        # Causal mask
        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= cur_offs_n[None, :]
            qk = tl.where(causal_mask, qk, float('-inf'))

        # Online softmax update
        m_ij = tl.max(qk, axis=1)           # new block max
        m_new = tl.maximum(m_i, m_ij)        # running max
        alpha = tl.exp(m_i - m_new)          # rescale factor for old
        p = tl.exp(qk - m_new[:, None])      # softmax numerator

        # Update accumulator: rescale old + add new
        l_new = alpha * l_i + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]  # rescale old output

        # Load V block and accumulate
        v_ptrs = V + pid_z * stride_vz + pid_h * stride_vh + \
                 cur_offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=(cur_offs_n[:, None] < N_CTX) & (offs_k[None, :] < BLOCK_K), other=0.0)
        acc += tl.dot(p.to(v.dtype), v)

        m_i = m_new
        l_i = l_new

    # Final normalization
    acc = acc / l_i[:, None]

    # Store output
    out_ptrs = Out + pid_z * stride_oz + pid_h * stride_oh + \
               offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty),
             mask=(offs_m[:, None] < N_CTX) & (offs_k[None, :] < BLOCK_K))


def flash_attention_triton(q, k, v, causal=True):
    """
    q, k, v: (batch, heads, seq_len, head_dim)
    """
    B, H, N, D = q.shape
    assert D in [64, 128], "head_dim must be 64 or 128"

    out = torch.empty_like(q)
    BLOCK_M = 64
    BLOCK_N = 64

    grid = (triton.cdiv(N, BLOCK_M), B * H)
    _fwd_attention_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, N,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=D,
        IS_CAUSAL=causal,
    )
    return out
```

## Usage
```python
B, H, N, D = 2, 32, 2048, 128
q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)

out = flash_attention_triton(q, k, v, causal=True)
# Verify against PyTorch SDPA
ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
print(f"Max diff: {(out - ref).abs().max().item():.6f}")
```
