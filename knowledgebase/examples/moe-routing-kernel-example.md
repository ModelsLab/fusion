---
id: moe_routing_kernel_example
kind: example
title: MoE Router and Expert Dispatch Kernel
category: kernel
summary: Triton implementation of top-k expert routing with load balancing, token permutation, and grouped GEMM dispatch for Mixture-of-Experts models.
tags:
  - moe
  - routing
  - top-k
  - expert-parallelism
  - triton
source_ids: []
operators:
  - topk
  - softmax
  - matmul
gpu_families:
  - Ampere
  - Ada
  - Hopper
precision:
  - fp16
  - bf16
---

## MoE Top-K Router

```python
import torch
import triton
import triton.language as tl

def moe_router(hidden_states, gate_weight, top_k=2, num_experts=8):
    """
    Route tokens to top-k experts.

    Args:
        hidden_states: (num_tokens, hidden_dim)
        gate_weight: (num_experts, hidden_dim)
        top_k: number of experts per token
    Returns:
        topk_weights: (num_tokens, top_k) - routing weights
        topk_ids: (num_tokens, top_k) - expert indices
        token_expert_indices: permutation for grouped GEMM
    """
    # Gate scores
    router_logits = torch.matmul(hidden_states, gate_weight.t())  # (T, E)
    routing_weights = torch.softmax(router_logits, dim=-1)

    # Top-k selection
    topk_weights, topk_ids = torch.topk(routing_weights, top_k, dim=-1)

    # Renormalize selected weights
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_ids


def moe_permute_and_compute(
    hidden_states, topk_weights, topk_ids,
    w1, w2, w3, num_experts
):
    """
    Permute tokens by expert, compute expert FFN, unpermute.

    w1: (num_experts, intermediate, hidden) - gate projection
    w2: (num_experts, hidden, intermediate) - down projection
    w3: (num_experts, intermediate, hidden) - up projection
    """
    num_tokens = hidden_states.shape[0]
    top_k = topk_ids.shape[1]

    # Flatten: each token appears top_k times
    flat_topk_ids = topk_ids.view(-1)  # (T * top_k,)
    flat_topk_weights = topk_weights.view(-1, 1)  # (T * top_k, 1)

    # Repeat hidden states for each selected expert
    hidden_repeated = hidden_states.repeat_interleave(top_k, dim=0)  # (T*top_k, D)

    # Sort by expert ID for coalesced access
    sorted_expert_ids, sort_indices = flat_topk_ids.sort()
    sorted_hidden = hidden_repeated[sort_indices]
    sorted_weights = flat_topk_weights[sort_indices]

    # Find expert boundaries
    expert_counts = torch.bincount(sorted_expert_ids, minlength=num_experts)
    expert_offsets = torch.cumsum(expert_counts, dim=0)
    expert_offsets = torch.cat([torch.zeros(1, device=expert_offsets.device, dtype=expert_offsets.dtype), expert_offsets])

    # Grouped computation per expert
    output = torch.zeros_like(sorted_hidden)
    for expert_id in range(num_experts):
        start = expert_offsets[expert_id].item()
        end = expert_offsets[expert_id + 1].item()
        if start == end:
            continue

        expert_input = sorted_hidden[start:end]  # (num_tokens_for_expert, D)

        # SwiGLU: output = down(silu(gate(x)) * up(x))
        gate = torch.matmul(expert_input, w1[expert_id].t())  # (T_e, I)
        up = torch.matmul(expert_input, w3[expert_id].t())    # (T_e, I)
        activated = torch.nn.functional.silu(gate) * up
        expert_output = torch.matmul(activated, w2[expert_id].t())  # (T_e, D)

        output[start:end] = expert_output

    # Apply routing weights
    output = output * sorted_weights

    # Unsort
    unsort_indices = sort_indices.argsort()
    output = output[unsort_indices]

    # Reduce: sum top_k contributions per token
    output = output.view(num_tokens, top_k, -1).sum(dim=1)

    return output


# Full MoE layer
def moe_layer(hidden_states, gate_weight, w1, w2, w3, num_experts=8, top_k=2):
    topk_weights, topk_ids = moe_router(hidden_states, gate_weight, top_k, num_experts)
    output = moe_permute_and_compute(
        hidden_states, topk_weights, topk_ids,
        w1, w2, w3, num_experts
    )
    return output

# Example
T, D, I, E = 512, 4096, 11008, 8
hidden = torch.randn(T, D, device='cuda', dtype=torch.bfloat16)
gate_w = torch.randn(E, D, device='cuda', dtype=torch.bfloat16)
w1 = torch.randn(E, I, D, device='cuda', dtype=torch.bfloat16)
w2 = torch.randn(E, D, I, device='cuda', dtype=torch.bfloat16)
w3 = torch.randn(E, I, D, device='cuda', dtype=torch.bfloat16)

out = moe_layer(hidden, gate_w, w1, w2, w3, num_experts=E, top_k=2)
print(f"Input: {hidden.shape}, Output: {out.shape}")
```

## Fused MoE Kernel (vLLM-style concept)
```python
# vLLM's fused_moe kernel combines:
# 1. Token permutation (sort by expert)
# 2. Grouped GEMM (all experts in one kernel launch)
# 3. SiLU activation
# 4. Token unpermutation + weight application

# Key optimization: instead of looping over experts,
# launch a single kernel with expert_id as grid dimension
# Each thread block handles one (expert, tile) pair
# This eliminates E separate kernel launches
```
