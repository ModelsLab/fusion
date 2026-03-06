---
id: sampling_decoding_kernels
kind: document
title: Sampling and Decoding Kernels
category: kernel
summary: Guide to GPU kernel implementations for LLM sampling strategies including top-k, top-p (nucleus), temperature, beam search, constrained decoding, and speculative verification.
tags:
  - sampling
  - top-k
  - top-p
  - beam-search
  - temperature
  - constrained-decoding
  - speculative-decoding
source_ids: []
operators:
  - softmax
  - topk
  - sampling
gpu_families:
  - Ampere
  - Ada
  - Hopper
  - Blackwell
---

# Sampling and Decoding Kernels

## The Sampling Pipeline
```
logits (B, V)
  → temperature scaling: logits / T
  → top-k filtering: keep only top-k logits
  → top-p filtering: keep smallest set with cumulative prob >= p
  → softmax: convert to probabilities
  → multinomial sampling: draw from distribution
  → output token IDs (B,)
```

For vocabulary V=128K and batch B=64, the sampling pipeline processes 8M values per step. Efficient kernels matter.

## Temperature Scaling
```python
# Simple element-wise operation - easily fused
@triton.jit
def temperature_scale_kernel(logits_ptr, temperature, V, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    batch = pid
    for off in range(0, V, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < V
        val = tl.load(logits_ptr + batch * V + idx, mask=mask)
        tl.store(logits_ptr + batch * V + idx, val / temperature, mask=mask)
```

## Top-K Filtering

### Approach 1: Full Sort (CUB Radix Sort)
```cpp
// Sort all V values, keep top K
// CUB provides efficient GPU radix sort
cub::DeviceRadixSort::SortPairsDescending(
    temp_storage, temp_bytes,
    logits, sorted_logits,         // values
    indices, sorted_indices,       // carry indices
    V, 0, sizeof(float) * 8       // num items, begin bit, end bit
);
// Take first K entries from sorted arrays
// O(V) work, O(V log V) span
```

### Approach 2: Partial Sort (More Efficient)
```python
# Only need top-K, not full sort
# Use modified quickselect / bucket-based approach

# Bucket approach:
# 1. Histogram logits into coarse buckets
# 2. Find which bucket contains the K-th largest
# 3. Refine within that bucket
# This is O(V) work but with small constant

# Or: Use torch.topk which uses CUB's top-k internally
top_k_values, top_k_indices = torch.topk(logits, k, dim=-1)
```

### Approach 3: Fused Top-K + Softmax
```python
@triton.jit
def fused_topk_softmax_kernel(
    logits_ptr, output_ptr, indices_ptr,
    V, K,
    BLOCK: tl.constexpr,
):
    batch = tl.program_id(0)

    # Pass 1: Find K-th largest value (approximate threshold)
    # Use histogram-based approach
    max_val = -float('inf')
    for off in range(0, V, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < V
        val = tl.load(logits_ptr + batch * V + idx, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.max(val))

    # Pass 2: Count values above threshold, adjust threshold
    # (simplified - real implementation is iterative)
    threshold = max_val - 20.0  # rough estimate

    # Pass 3: Apply top-k mask and softmax
    sum_exp = 0.0
    for off in range(0, V, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < V
        val = tl.load(logits_ptr + batch * V + idx, mask=mask, other=-float('inf'))
        keep = val >= threshold
        masked_val = tl.where(keep, val - max_val, -float('inf'))
        sum_exp += tl.sum(tl.exp(masked_val))

    # Pass 4: Output normalized probabilities
    # ...
```

## Top-P (Nucleus) Sampling

### Algorithm
```python
# Keep smallest set of tokens whose cumulative probability >= p
def top_p_sampling(logits, p=0.9, temperature=1.0):
    # 1. Temperature scale
    logits = logits / temperature

    # 2. Sort descending
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)

    # 3. Compute cumulative probabilities
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(probs, dim=-1)

    # 4. Remove tokens with cumulative probability above threshold
    # Shift cumulative probs right (include the token that crosses p)
    sorted_indices_to_remove = cumulative_probs - probs > p
    sorted_logits[sorted_indices_to_remove] = -float('inf')

    # 5. Re-normalize and sample
    probs = torch.softmax(sorted_logits, dim=-1)
    token = torch.multinomial(probs, num_samples=1)

    # 6. Map back to original indices
    return sorted_indices.gather(-1, token)
```

### GPU Kernel Challenge
Top-p requires sorting → expensive for V=128K.
Optimizations:
1. Combine top-k (k=50) first to reduce candidate set, then top-p
2. Use partial sort / histogram approach
3. Fuse sort + cumsum + threshold + sample into single kernel

## Beam Search Kernels

```python
# Beam search maintains B beams (partial sequences)
# Each step: expand each beam by V candidates → B*V total
# Select top B from B*V candidates

# Key kernel: top-B selection from B*V candidates
# B=4, V=128K → select top 4 from 512K candidates
# Use CUB radix sort or multi-way merge

# Memory management:
# - Need to maintain B KV caches (one per beam)
# - On fork: copy-on-write (share KV cache blocks, copy on divergence)
# - PagedAttention's block sharing is ideal for this
```

## Constrained Decoding

### Grammar-Guided Generation (FSM-based)
```python
# Use Finite State Machine to constrain valid next tokens
# For JSON mode: FSM tracks JSON parsing state

# Each state has a set of valid next tokens
# At each step: mask out invalid tokens BEFORE sampling

# Kernel: apply token mask to logits
@triton.jit
def apply_token_mask_kernel(
    logits_ptr, mask_ptr, V,
    BLOCK: tl.constexpr
):
    batch = tl.program_id(0)
    for off in range(0, V, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        is_valid = tl.load(mask_ptr + idx, mask=idx < V)  # boolean mask
        val = tl.load(logits_ptr + batch * V + idx, mask=idx < V)
        masked = tl.where(is_valid, val, -float('inf'))
        tl.store(logits_ptr + batch * V + idx, masked, mask=idx < V)
```

### Jump-Forward Decoding (SGLang)
```
# When grammar constrains next tokens to a SINGLE possibility:
# Skip model execution entirely, emit the deterministic token
# E.g., after opening quote in JSON: `"key": ` → the `: ` is deterministic

# State machine: detect when |valid_next_tokens| == 1
# Skip expensive model forward pass for those positions
```

## Speculative Decoding Verification Kernel

```python
# Verify k draft tokens against target model probabilities
# Modified rejection sampling:

@triton.jit
def spec_decode_verify_kernel(
    target_probs_ptr,  # (k+1, V) - target model probabilities
    draft_probs_ptr,   # (k, V) - draft model probabilities
    draft_tokens_ptr,  # (k,) - draft token IDs
    random_ptr,        # (k,) - pre-generated random numbers
    accepted_ptr,      # output: number of accepted tokens
    resampled_ptr,     # output: resampled token if rejection
    V, k,
    BLOCK: tl.constexpr,
):
    # For each draft position:
    num_accepted = 0
    for i in range(k):
        draft_token = tl.load(draft_tokens_ptr + i)
        p = tl.load(target_probs_ptr + i * V + draft_token)  # target prob
        q = tl.load(draft_probs_ptr + i * V + draft_token)   # draft prob
        r = tl.load(random_ptr + i)

        acceptance_prob = tl.minimum(1.0, p / q)
        if r < acceptance_prob:
            num_accepted += 1
        else:
            # Resample from adjusted distribution: max(0, p - q)
            # Normalize and sample
            # ... (requires another pass over vocabulary)
            break

    tl.store(accepted_ptr, num_accepted)
```

## Repetition Penalty Kernel
```python
@triton.jit
def repetition_penalty_kernel(
    logits_ptr,
    generated_tokens_ptr,  # previously generated token IDs
    num_generated,
    penalty,  # > 1.0 to penalize, < 1.0 to encourage
    V,
    BLOCK: tl.constexpr,
):
    batch = tl.program_id(0)

    # For each previously generated token, apply penalty
    for i in range(num_generated):
        token_id = tl.load(generated_tokens_ptr + batch * max_gen + i)
        logit = tl.load(logits_ptr + batch * V + token_id)

        # If logit > 0, divide by penalty (make less likely)
        # If logit < 0, multiply by penalty (make more negative = less likely)
        if logit > 0:
            logit = logit / penalty
        else:
            logit = logit * penalty

        tl.store(logits_ptr + batch * V + token_id, logit)
```
