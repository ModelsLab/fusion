---
id: model_architectures_kernel_implications
kind: document
title: LLM Architectures and Their Kernel-Level Implications
category: architecture
summary: Detailed analysis of every major LLM architecture (LLaMA, Mistral, DeepSeek, Qwen, etc.) and the specific kernel requirements, fusion opportunities, and optimization strategies each demands.
tags:
  - architecture
  - llama
  - mistral
  - deepseek
  - moe
  - mla
  - gqa
  - rope
  - swiglu
  - mamba
source_ids: []
gpu_families:
  - Ampere
  - Ada
  - Hopper
  - Blackwell
operators:
  - attention
  - matmul
  - rmsnorm
  - rope
  - swiglu
  - softmax
  - topk
---

# LLM Architectures and Their Kernel-Level Implications

## Standard Transformer Block (LLaMA-style)

### Layer Structure
```
Input (B, S, H)
  ├── RMSNorm → (B, S, H)
  ├── QKV Projection: Linear(H, H+2*H_kv)  [GEMM]
  ├── RoPE on Q, K
  ├── GQA Attention (FlashAttention / PagedAttention)
  ├── Output Projection: Linear(H, H)  [GEMM]
  ├── Residual Add
  ├── RMSNorm → (B, S, H)
  ├── Gate+Up Projection: Linear(H, 2*I)  [GEMM, fused gate+up]
  ├── SwiGLU: silu(gate) * up
  ├── Down Projection: Linear(I, H)  [GEMM]
  └── Residual Add
Output (B, S, H)
```

### Kernel Count Per Layer (Unfused)
```
1. RMSNorm (1 kernel)
2. QKV GEMM (1 kernel, fused Q+K+V)
3. RoPE (1 kernel)
4. Attention (1 kernel - FlashAttention)
5. Output GEMM (1 kernel)
6. Residual add (1 kernel)
7. RMSNorm (1 kernel)
8. Gate+Up GEMM (1 kernel, fused)
9. SwiGLU (1 kernel)
10. Down GEMM (1 kernel)
11. Residual add (1 kernel)
Total: ~11 kernel launches per layer
```

### Fusion Opportunities
```
Fused kernels (ideal):
1. RMSNorm + Residual Add → 1 kernel (saves 1 launch + memory pass)
2. QKV GEMM (already fused Q+K+V into single GEMM)
3. Attention + RoPE → 1 kernel (apply RoPE inside attention)
4. Output GEMM + Residual → via epilogue fusion
5. RMSNorm + Residual → 1 kernel
6. Gate+Up GEMM (already fused)
7. SwiGLU → 1 kernel
8. Down GEMM + Residual → via epilogue fusion

Fused total: ~6-8 kernel launches per layer (vs 11 unfused)
```

## Model-Specific Architectures

### LLaMA 3 / LLaMA 3.1 (Meta)
```
Sizes: 8B, 70B, 405B
Key features:
- GQA: 8 KV heads for 70B (8:1 ratio), 8 for 405B
- RoPE with extended context (128K for 3.1)
- SwiGLU activation: intermediate = 8/3 * hidden (rounded)
- Vocabulary: 128K tokens (larger than LLaMA 2's 32K)

Kernel implications:
- Large vocabulary → LM head GEMM is (B*S, H) x (H, 128K) → significant
- GQA reduces KV cache but attention kernel must handle head grouping
- Long context (128K) → FlashAttention essential, KV cache optimization critical
- 405B needs TP=8+ → heavy communication

Model dimensions:
| | 8B | 70B | 405B |
|---|---|---|---|
| hidden_dim | 4096 | 8192 | 16384 |
| num_layers | 32 | 80 | 126 |
| num_heads | 32 | 64 | 128 |
| num_kv_heads | 8 | 8 | 8 |
| intermediate | 14336 | 28672 | 53248 |
| vocab | 128256 | 128256 | 128256 |
```

### Mistral 7B / Mistral Large
```
Key features:
- Sliding window attention (W=4096 in original 7B)
- GQA: 8 KV heads
- RoPE, SwiGLU

Kernel implications:
- Sliding window: only attend to last W tokens during decode
  → KV cache can be circular buffer of size W
  → Attention kernel needs window masking
- FlexAttention can express sliding window natively
```

### Mixtral 8x7B / 8x22B (MoE)
```
Key features:
- Sparse MoE: 8 experts, top-2 routing
- Each expert is a standard FFN (gate+up+down projections)
- Shared attention across all experts
- Total params: 46.7B (8x7B) but only 12.9B active per token

Kernel requirements:
1. Router kernel: softmax + top-k selection
   - Input: (B*S, expert_dim) → routing scores
   - top-2 selection per token
   - Load balancing (auxiliary loss during training)

2. Token permutation:
   - Scatter tokens to their assigned experts
   - Input: (B*S, H) → sorted by expert assignment
   - Output: (num_tokens_expert_i, H) for each expert i

3. Grouped GEMM:
   - Each expert processes different number of tokens
   - Grouped GEMM: [(M_0, N, K), (M_1, N, K), ..., (M_7, N, K)]
   - CUTLASS GroupedGemm or batched GEMM

4. Token unpermutation:
   - Gather expert outputs back to original token order
   - Weighted by routing scores

5. All-to-All communication (multi-GPU expert parallelism):
   - Scatter tokens across GPUs, each GPU hosts different experts
   - AllToAll collective (expensive!)

Performance challenge: load imbalance
- If all tokens route to same 2 experts → those GPUs are bottleneck
- Need balancing: capacity factor, jitter, auxiliary loss
```

### DeepSeek-V2 / V3 / R1
```
Key features:
- MLA (Multi-Latent Attention): compressed KV projections
- DeepSeekMoE: fine-grained experts (more but smaller)
- Auxiliary-loss-free load balancing (V3)

MLA kernel implications:
Standard GQA KV cache per token: 2 * num_kv_heads * head_dim = 2 * 8 * 128 = 2048 bytes (FP16)
MLA KV cache per token: compressed_dim = 512 bytes (FP16)
→ 4x KV cache compression!

MLA computation:
1. Compress: c_kv = hidden @ W_dkv  # (H,) → (D_c,) compression
2. Store c_kv in KV cache (not full K, V)
3. During attention:
   K = c_kv @ W_uk  # decompress keys
   V = c_kv @ W_uv  # decompress values
   # Then standard attention

Kernel: need to fuse decompression with attention
Or: absorb decompression into QKV projection (algebraic trick)

DeepSeekMoE:
- 256 experts (vs 8 in Mixtral)
- Top-6 routing per token
- Shared experts (always active) + routed experts
- Kernel: more fine-grained token routing, more experts per GEMM group
```

### Qwen2.5 (Alibaba)
```
Sizes: 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B
Key features:
- Standard transformer with GQA
- Large vocabulary: 152K tokens
- 128K context support

Kernel implications:
- Very large vocab: LM head GEMM for 152K tokens
- Standard optimization path (same as LLaMA)
```

### Phi-4 (Microsoft)
```
Key features:
- Small but capable (14B)
- Optimized for reasoning
- Standard transformer with some architectural tweaks

Kernel implication:
- Small model → fits easily on single GPU
- Optimization: focus on decode latency (not model fit)
- INT4 quantization → ultra-fast on consumer GPUs
```

## Specific Operator Kernels

### RMSNorm Kernel
```
Formula: y = x * weight / sqrt(mean(x^2) + eps)

Implementation:
1. Compute variance: sum(x^2) across hidden dimension
2. Compute scaling: 1/sqrt(var/N + eps)
3. Apply: y[i] = x[i] * scale * weight[i]

Memory-bound: reads x (2N bytes), writes y (2N bytes), reads weight (2N bytes)
Total: ~6N bytes for 2N FLOPs → AI ≈ 0.33 (severely memory-bound)

Fusion: fuse with residual add for free (both memory-bound, same access pattern)
```

### RoPE (Rotary Position Embedding)
```
Formula: apply 2D rotation to pairs of dimensions based on position

For each pair (x_2i, x_{2i+1}) at position pos:
  x_2i' = x_2i * cos(pos * freq_i) - x_{2i+1} * sin(pos * freq_i)
  x_{2i+1}' = x_2i * sin(pos * freq_i) + x_{2i+1} * cos(pos * freq_i)

where freq_i = 1 / (base^(2i/d))

Implementation options:
1. Separate kernel: load Q/K, apply rotation, store
2. Fused into QKV projection epilogue
3. Fused into attention kernel (best: no extra memory pass)

For long context (128K+): use NTK-aware RoPE or YaRN for extended positions
```

### SwiGLU Activation
```
Formula: output = silu(gate) * up = gate * sigmoid(gate) * up

Where gate = x @ W_gate, up = x @ W_up

Optimization: fuse gate+up projections into single GEMM:
combined = x @ [W_gate; W_up]  # (B*S, H) @ (H, 2*I) → (B*S, 2*I)
gate, up = combined.chunk(2, dim=-1)
output = silu(gate) * up  # fused pointwise kernel

Total: 1 GEMM + 1 pointwise kernel (vs 2 GEMM + 1 pointwise)
```

### MoE Router (Top-K Gating)
```python
# Router: select top-k experts per token
def moe_router(hidden_states, gate_weight, k=2):
    # Gate scores: (B*S, num_experts)
    scores = hidden_states @ gate_weight  # GEMM
    # Softmax: normalize scores
    probs = softmax(scores, dim=-1)
    # Top-k: select k experts per token
    topk_probs, topk_indices = torch.topk(probs, k, dim=-1)
    # Normalize selected probs
    topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
    return topk_probs, topk_indices

# Kernel: fuse softmax + topk for efficiency
# Avoid materializing full probability matrix
```

## State Space Models

### Mamba / Mamba-2 (Selective Scan)
```
Not attention-based - uses selective state space model:

Formula (simplified):
h_t = A_t * h_{t-1} + B_t * x_t  # state update (recurrence)
y_t = C_t * h_t + D * x_t         # output

Where A_t, B_t, C_t are input-dependent (selective)

Kernel challenge: sequential recurrence is hard to parallelize
Solution: parallel scan algorithm (work-efficient)

Parallel scan:
1. Up-sweep: compute partial products in tree structure
2. Down-sweep: propagate to get all prefix sums
Total: O(N) work, O(log N) depth → GPU-friendly

Mamba-2 simplification:
- Restricts A to scalar-times-identity
- Allows more efficient parallelization
- Can be expressed as matrix multiply chunks
```

### Hybrid Models (Jamba, Zamba)
```
Mix attention and SSM layers:
- Some layers use attention (for precise long-range)
- Some layers use Mamba (for efficient local processing)
- Get benefits of both

Kernel implications:
- Need both attention kernels AND selective scan kernels
- Different optimization strategies per layer type
```

## Diffusion Transformer (DiT) Architecture

### Structure
```
DiT replaces U-Net with Transformer for diffusion models:

Input: noisy latent (B, C, H, W) + timestep + text conditioning
  ├── Patchify: (B, C, H, W) → (B, N, D) where N = H*W/patch_size^2
  ├── For each DiT block:
  │   ├── AdaLN (Adaptive Layer Norm): modulated by timestep
  │   ├── Self-Attention (standard, with RoPE or sinusoidal PE)
  │   ├── Cross-Attention (attend to text embeddings)
  │   ├── Pointwise FFN
  │   └── Scale+Shift modulation
  ├── Unpatchify: (B, N, D) → (B, C, H, W)
  └── Output: predicted noise

Kernel implications:
- AdaLN: needs per-token modulation parameters → fused norm kernel with modulation
- Cross-attention: Q from image, K/V from text → different sequence lengths
- Sequence length can be large: 1024x1024 image with patch=2 → 262K tokens
- Need FlashAttention for these lengths
```
