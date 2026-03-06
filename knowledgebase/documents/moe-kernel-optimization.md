---
id: moe_kernel_optimization
kind: document
title: Mixture of Experts (MoE) Kernel Optimization Guide
category: architecture
summary: Complete guide to MoE kernel optimization covering routing, token permutation, grouped GEMM, expert parallelism, load balancing, and model-specific patterns for Mixtral, DeepSeek, and other MoE architectures.
tags:
  - moe
  - mixture-of-experts
  - grouped-gemm
  - routing
  - expert-parallelism
  - mixtral
  - deepseek
  - load-balancing
source_ids: []
gpu_families:
  - Ampere
  - Ada
  - Hopper
  - Blackwell
operators:
  - matmul
  - topk
  - softmax
  - permutation
precision:
  - fp16
  - bf16
  - fp8
  - int4
---

# Mixture of Experts (MoE) Kernel Optimization

## MoE Architecture Overview

### How MoE Works in Transformers
```
Standard FFN (dense):
  output = down_proj(activation(up_proj(x)))  # ALL parameters used for ALL tokens

MoE FFN (sparse):
  router_scores = softmax(x @ gate_weight)     # (B*S, num_experts)
  top_k_experts = topk(router_scores, k)       # select k experts per token
  for each token:
    output = sum(weight_i * expert_i(x) for i in selected_experts)
```

Key numbers for popular MoE models:
| Model | Total Experts | Active (k) | Expert Size | Total Params | Active Params |
|-------|--------------|------------|-------------|-------------|---------------|
| Mixtral-8x7B | 8 | 2 | 7B FFN | 46.7B | 12.9B |
| Mixtral-8x22B | 8 | 2 | 22B FFN | ~141B | ~39B |
| DeepSeek-V3 | 256 (+1 shared) | 8 | Small | 671B | ~37B |
| Grok-1 | 8 | 2 | ~50B FFN | ~314B | ~86B |
| DBRX | 16 | 4 | 12B FFN | 132B | ~36B |

## The Five Kernel Stages of MoE

### Stage 1: Router (Gating) Kernel
```python
# Input: hidden_states (B*S, hidden_dim)
# Output: expert_indices (B*S, k), expert_weights (B*S, k)

@triton.jit
def moe_router_kernel(
    hidden_ptr, gate_weight_ptr,
    expert_indices_ptr, expert_weights_ptr,
    B_S, hidden_dim, num_experts, k,
    BLOCK_E: tl.constexpr,
):
    token_idx = tl.program_id(0)

    # Compute router scores: x @ gate_weight^T
    # (1, H) @ (H, E) → (1, E)
    scores = tl.zeros([BLOCK_E], dtype=tl.float32)
    for h in range(0, hidden_dim, 128):
        x_chunk = tl.load(hidden_ptr + token_idx * hidden_dim + h + tl.arange(0, 128))
        for e in range(num_experts):
            w_chunk = tl.load(gate_weight_ptr + e * hidden_dim + h + tl.arange(0, 128))
            scores[e] += tl.sum(x_chunk * w_chunk)

    # Softmax over experts
    max_score = tl.max(scores)
    exp_scores = tl.exp(scores - max_score)
    sum_exp = tl.sum(exp_scores)
    probs = exp_scores / sum_exp

    # Top-k selection (simplified - real implementation uses CUB)
    # Store top-k indices and normalized weights
```

**Optimization notes**:
- Router GEMM is small (hidden_dim x num_experts) - often memory-bound
- Can fuse softmax + top-k into single kernel
- For small num_experts (8-16): thread-level top-k
- For large num_experts (256+): block-level CUB sort

### Stage 2: Token Permutation
```python
# Reorder tokens so each expert's tokens are contiguous
# This is necessary for efficient grouped GEMM

# Input: token_expert_assignments (B*S, k)
# Output: permuted indices, expert_offsets

def permute_tokens(hidden_states, expert_indices, expert_weights, num_experts):
    # Count tokens per expert
    expert_counts = torch.zeros(num_experts, dtype=torch.int32)
    for e in range(num_experts):
        expert_counts[e] = (expert_indices == e).sum()

    # Compute offsets (prefix sum)
    expert_offsets = torch.cumsum(expert_counts, dim=0)
    expert_offsets = torch.cat([torch.zeros(1), expert_offsets[:-1]])

    # Scatter tokens to expert-sorted order
    permuted = torch.empty_like(hidden_states.repeat(k, 1))  # k copies
    for token, experts in enumerate(expert_indices):
        for rank, expert_id in enumerate(experts):
            dest_idx = expert_offsets[expert_id]
            permuted[dest_idx] = hidden_states[token]
            expert_offsets[expert_id] += 1

    return permuted, expert_counts, expert_offsets
```

**Optimization**: Use CUB radix sort to sort by expert ID, or scatter with atomic counters.

### Stage 3: Grouped GEMM (Expert Execution)
```
# Each expert processes its assigned tokens
# Problem: different experts may get different numbers of tokens
# Solution: grouped GEMM - execute multiple GEMMs in one kernel

problems = []
for expert_id in range(num_experts):
    M_i = expert_counts[expert_id]  # varies!
    problems.append({
        'M': M_i, 'N': intermediate_size, 'K': hidden_dim,
        'A': permuted_tokens[expert_offset[expert_id]:],
        'B': expert_weights[expert_id],
    })

# CUTLASS GroupedGemm handles this:
cutlass_grouped_gemm(problems)
```

**CUTLASS GroupedGemm**:
```cpp
using GemmGrouped = cutlass::gemm::device::GemmGrouped<
    cutlass::half_t,                     // ElementA
    cutlass::layout::RowMajor,           // LayoutA
    cutlass::half_t,                     // ElementB
    cutlass::layout::ColumnMajor,        // LayoutB
    cutlass::half_t,                     // ElementC
    cutlass::layout::RowMajor,           // LayoutC
    float,                               // ElementAccumulator
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>
>;
```

### Stage 4: Expert Output Combination
```python
# Weighted sum of expert outputs per token
# output[token] = sum(weight_i * expert_output_i) for selected experts

@triton.jit
def weighted_combine_kernel(
    expert_outputs_ptr, weights_ptr, output_ptr,
    token_to_expert_map_ptr,  # maps back to original token order
    hidden_dim, k,
    BLOCK_H: tl.constexpr,
):
    token_idx = tl.program_id(0)

    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for expert_rank in range(k):
        weight = tl.load(weights_ptr + token_idx * k + expert_rank)
        expert_output_idx = tl.load(token_to_expert_map_ptr + token_idx * k + expert_rank)

        for h in range(0, hidden_dim, BLOCK_H):
            offs = h + tl.arange(0, BLOCK_H)
            expert_out = tl.load(expert_outputs_ptr + expert_output_idx * hidden_dim + offs)
            acc += weight * expert_out

    # Store combined output
    tl.store(output_ptr + token_idx * hidden_dim + tl.arange(0, BLOCK_H), acc.to(tl.float16))
```

### Stage 5: Token Unpermutation
Reverse the permutation from Stage 2 to restore original token order.

## Expert Parallelism (Multi-GPU)

### All-to-All Pattern
```
# With EP=4 (4 GPUs, each hosting num_experts/4 experts):

Step 1: Each GPU computes router scores locally
Step 2: AllToAll - scatter tokens to GPUs with their assigned experts
  GPU 0 sends tokens for experts 2,3 to GPU 1
  GPU 1 sends tokens for experts 0,1 to GPU 0
  etc.
Step 3: Each GPU processes its local experts
Step 4: AllToAll - gather results back to original GPUs
Step 5: Combine expert outputs locally
```

### Communication Cost
```
AllToAll per MoE layer:
  Send: B*S * k * hidden_dim * dtype_size / EP (per GPU)
  Total: 2 * B*S * k * hidden_dim * dtype_size  (send + receive)

For Mixtral-8x7B, B=64, S=1 (decode), EP=4, FP16:
  2 * 64 * 2 * 4096 * 2 = 2 MB per MoE layer
  32 MoE layers → 64 MB per token step
  On NVLink (900 GB/s): ~0.07 ms (negligible)
  On InfiniBand (400 Gbps): ~1.3 ms (significant!)
```

### EP + TP Combinations
```
# For 8 GPUs serving DeepSeek-V3 (256 experts):
Option A: EP=8, TP=1 → 32 experts per GPU, full GEMM per expert
Option B: EP=4, TP=2 → 64 experts per GPU, half-GEMM per expert
Option C: EP=2, TP=4 → 128 experts per GPU, quarter-GEMM per expert

# Trade-off:
# More EP → less AllReduce (TP), more AllToAll (EP)
# More TP → less AllToAll, more AllReduce
# Choose based on interconnect topology
```

## Load Balancing

### The Problem
```
# If all tokens route to same 2 experts:
# Expert 0: processes 100% of tokens (bottleneck)
# Expert 1: processes 100% of tokens
# Experts 2-7: idle
# Result: no speedup from MoE, worse than dense model!
```

### Solutions

**Auxiliary Loss (standard)**:
```python
# Add load balancing loss during training
# f_i = fraction of tokens routed to expert i
# P_i = average routing probability for expert i
aux_loss = num_experts * sum(f_i * P_i)  # encourage uniform distribution
total_loss = task_loss + alpha * aux_loss  # alpha ~= 0.01
```

**Capacity Factor**:
```python
# Limit tokens per expert
capacity = int(capacity_factor * (B * S * k / num_experts))
# Typical capacity_factor = 1.25 (allow 25% overflow)
# Tokens exceeding capacity are dropped or sent to overflow expert
```

**Auxiliary-Loss-Free (DeepSeek-V3)**:
```python
# Bias-based balancing without training loss
# Each expert has a learnable bias term
# Bias is updated based on load imbalance (not gradient)
# If expert_i is overloaded: decrease bias_i
# If expert_i is underloaded: increase bias_i
```

## Model-Specific MoE Optimization

### Mixtral-8x7B Optimization
```
- 8 experts, top-2 routing
- Each expert = full 7B FFN (gate_proj, up_proj, down_proj)
- Shared attention (not MoE)
- Key optimization: fuse gate+up GEMM within each expert
- On 2xH100 (TP=2): each GPU has all 8 experts, half the attention
```

### DeepSeek-V3 Optimization
```
- 256 routed experts + 1 shared expert
- Top-8 routing per token
- Fine-grained experts (each is small)
- Key kernel challenges:
  1. Router for 256 experts → larger gating GEMM
  2. Token permutation for 256 groups
  3. Many small grouped GEMMs (each expert is small)
  4. More complex load balancing
- Shared expert runs on ALL tokens → fuse with attention output
```

## Practical MoE Deployment Recipes

### Recipe 1: Serve Mixtral 8x7B on Single GPU (AWQ)
```bash
# Mixtral 8x7B = ~90 GB in FP16 (needs 2x A100 or 1x H100)
# With AWQ INT4: ~24 GB (fits single RTX 4090!)

# Quantize (one-time)
pip install autoawq
python -c "
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
model = AutoAWQForCausalLM.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
tokenizer = AutoTokenizer.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
model.quantize(tokenizer, quant_config={'zero_point': True, 'q_group_size': 128, 'w_bit': 4, 'version': 'gemm'})
model.save_quantized('./Mixtral-8x7B-AWQ')
tokenizer.save_pretrained('./Mixtral-8x7B-AWQ')
"

# Serve
vllm serve ./Mixtral-8x7B-AWQ \
  --quantization awq \
  --dtype float16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.95
```

### Recipe 2: Serve MoE with Expert Parallelism (Multi-GPU)
```bash
# DeepSeek-V3 or large MoE: split experts across GPUs
vllm serve deepseek-ai/DeepSeek-V3 \
  --tensor-parallel-size 8 \
  --quantization fp8 \
  --max-model-len 4096 \
  --trust-remote-code

# vLLM automatically handles:
# - Expert routing across GPUs
# - All-to-All communication for token dispatch
# - Fused MoE GEMM kernel (fused_moe in vLLM)
```

### Recipe 3: MoE Memory Estimation
```python
# Quick memory calculator for MoE models
def moe_memory_gb(
    hidden_dim, intermediate_dim, num_layers,
    num_experts, num_shared_experts, num_attention_heads,
    head_dim, vocab_size, dtype_bytes=2  # 2 for FP16/BF16
):
    # Per-expert FFN: gate + up + down projections
    expert_params = 3 * hidden_dim * intermediate_dim
    total_expert_params = num_experts * expert_params * num_layers

    # Shared expert (if any)
    shared_params = num_shared_experts * expert_params * num_layers

    # Attention: Q, K, V, O projections per layer
    attn_params = 4 * hidden_dim * num_attention_heads * head_dim * num_layers

    # Router: hidden_dim → num_experts per layer
    router_params = hidden_dim * num_experts * num_layers

    # Embeddings
    embed_params = vocab_size * hidden_dim * 2  # input + output

    total_params = total_expert_params + shared_params + attn_params + router_params + embed_params
    total_gb = total_params * dtype_bytes / 1e9
    return total_gb, total_params / 1e9

# Mixtral 8x7B
gb, params = moe_memory_gb(4096, 14336, 32, 8, 0, 32, 128, 32000)
print(f"Mixtral 8x7B: {params:.1f}B params, {gb:.1f} GB in FP16")
# → 46.7B params, 93.4 GB in FP16

# DeepSeek-V3
gb, params = moe_memory_gb(7168, 2048, 61, 256, 1, 128, 128, 129280)
print(f"DeepSeek-V3: {params:.1f}B params, {gb:.1f} GB in FP16")
# → 671B params, 1342 GB in FP16
```

### Common MoE Issues and Fixes
| Issue | Cause | Fix |
|-------|-------|-----|
| OOM with Mixtral | All 8 experts loaded even though only 2 active | Use AWQ INT4 or expert offloading |
| Slow MoE inference | Token permutation overhead | Use vLLM's fused_moe kernel (automatic) |
| Load imbalance (some experts unused) | Poor routing | Check router logits distribution; this is a model issue, not deployment |
| Multi-GPU MoE slower than expected | All-to-All communication bottleneck | Ensure NVLink between GPUs; use EP only when needed |
| DeepSeek-V3 quality issues | Shared expert not handled correctly | Use SGLang (best DeepSeek support) or vLLM with --trust-remote-code |
