---
id: pruning_compression_complete_guide
kind: document
title: Model Pruning and Compression - Complete Guide
category: compression
summary: Comprehensive guide to all model compression techniques - structured/unstructured pruning, 2:4 sparsity, knowledge distillation, layer skipping, low-rank decomposition, and combined approaches.
tags:
  - pruning
  - sparsity
  - distillation
  - compression
  - layer-skipping
  - wanda
  - sparsegpt
  - structured-sparsity
gpu_families:
  - Ampere
  - Ada
  - Hopper
  - Blackwell
---

# Model Pruning and Compression - Complete Guide

## Pruning Overview

### Why Prune?
- Reduce model size (memory)
- Reduce compute (fewer operations)
- Increase inference speed (if hardware supports sparse execution)
- Can be combined with quantization for extreme compression

### Taxonomy

```
Pruning
├── Unstructured (element-wise)
│   ├── Magnitude Pruning
│   ├── Wanda (Weights AND Activations)
│   ├── SparseGPT (Hessian-based)
│   └── Movement Pruning
├── Structured
│   ├── Channel/Neuron Pruning
│   ├── Head Pruning (attention)
│   ├── Layer Pruning (depth)
│   │   ├── ShortGPT
│   │   ├── LaCo
│   │   └── LayerSkip
│   └── Width Pruning (SliceGPT)
├── Semi-Structured
│   └── 2:4 Sparsity (NVIDIA)
└── Dynamic
    ├── Early Exit
    ├── Token Dropping
    └── Mixture of Depths
```

## Unstructured Pruning

### Magnitude Pruning
```python
# Simplest approach: remove smallest magnitude weights
def magnitude_prune(weight, sparsity_ratio):
    threshold = torch.quantile(weight.abs(), sparsity_ratio)
    mask = weight.abs() > threshold
    return weight * mask
```

**Limitations**:
- Random sparsity pattern → no hardware acceleration (GPUs can't skip arbitrary zeros)
- Needs high sparsity (>90%) for quality models, but then accuracy degrades
- Only beneficial on specialized hardware or with sparse formats

### Wanda (Pruning by Weights AND Activations)
**Key insight**: Consider both weight magnitude AND activation magnitude. Weights connected to large activations are more important.

```python
def wanda_prune(weight, activation_norms, sparsity_ratio):
    """
    weight: (out_features, in_features)
    activation_norms: (in_features,) - norm of activations per input channel
    """
    # Importance = |weight| * ||activation||
    importance = weight.abs() * activation_norms.unsqueeze(0)

    # Prune per row (output channel)
    for row in range(weight.shape[0]):
        threshold = torch.quantile(importance[row], sparsity_ratio)
        mask = importance[row] > threshold
        weight[row] *= mask

    return weight
```

**Advantages**:
- No retraining needed (one-shot)
- Only needs a small calibration set (128 samples)
- Works well for 50% unstructured and 2:4 structured sparsity
- Very fast to compute

### SparseGPT
**Key insight**: Use approximate Hessian information (like GPTQ) to make optimal pruning decisions and compensate for pruned weights.

```python
# Algorithm (simplified per-layer):
def sparsegpt_prune(W, X, sparsity_ratio):
    """
    W: weight matrix (out, in)
    X: calibration activations (samples, in)
    """
    H = 2 * X.T @ X  # Approximate Hessian

    # Process columns (like GPTQ):
    for j in sorted_by_importance:  # process in optimal order
        if should_prune(j):
            # Set weight to zero
            error = W[:, j] / H[j, j]
            W[:, j] = 0
            # Compensate remaining weights:
            W[:, j+1:] -= error.unsqueeze(1) * H[j, j+1:].unsqueeze(0)
        else:
            # Keep weight, still apply error compensation
            pass

    return W
```

**Performance**: Achieves 50-60% unstructured sparsity with minimal perplexity loss. Can do 2:4 structured sparsity with <1 PPL increase on LLaMA models.

## 2:4 Structured Sparsity (NVIDIA)

### How It Works
For every group of 4 consecutive elements in the weight matrix, exactly 2 must be zero:

```
Group:    [a, b, c, d]
Valid:    [a, 0, 0, d]  → indices (0, 3)
Valid:    [0, b, c, 0]  → indices (1, 2)
Valid:    [a, 0, c, 0]  → indices (0, 2)
Invalid:  [a, b, c, 0]  → 3 non-zero (must be exactly 2)
```

### Storage Format
```
Original matrix (4x8, FP16):
[0.5, 0, 0, 0.8, | 0.2, 0, 0.6, 0  ]
[0, 0.3, 0.1, 0, | 0, 0.4, 0, 0.7  ]
...

Compressed storage:
Values (only non-zeros):  [0.5, 0.8, 0.2, 0.6, 0.3, 0.1, 0.4, 0.7, ...]
Metadata (2-bit indices): [0,3, 0,2, 1,2, 1,3, ...]  // which 2 of 4 are kept

Compression: ~50% for values, +metadata (~12.5% overhead)
Effective: ~56% of original size
```

### Tensor Core Acceleration
Ampere+ tensor cores have dedicated sparse MMA instructions:
```
// Sparse MMA: same throughput per instruction as dense,
// but processes 2x the K dimension (since half elements are zero)
// Net effect: 2x throughput for sparse GEMM

// A100 sparse tensor core throughput:
// FP16: 624 TFLOPS (vs 312 dense)
// INT8: 1248 TOPS (vs 624 dense)
```

### Achieving 2:4 Sparsity

**Method 1: Prune then fine-tune**
```python
# 1. Train model to convergence
# 2. Apply 2:4 mask (keep top 2 of 4 by magnitude)
# 3. Fine-tune with frozen mask
from apex.contrib.sparsity import ASP
ASP.prune_trained_model(model, optimizer)
# Fine-tune for ~10% of original training
```

**Method 2: SparseGPT + 2:4 constraint**
```python
# Modify SparseGPT to enforce 2:4 pattern instead of unstructured
# Process groups of 4 columns, select best 2 to keep using Hessian info
```

**Method 3: Wanda + 2:4 constraint**
```python
# Apply Wanda scoring, then enforce 2:4 pattern per group
for each group of 4 elements:
    scores = |weight| * activation_norm
    keep top 2 by score, zero out other 2
```

### Combining 2:4 Sparsity + Quantization

**Marlin-24 kernel**: 2:4 sparse INT4
- 50% sparsity → 2x speedup from sparse tensor cores
- INT4 quantization → 4x less memory
- Combined: ~8x compression, ~6x decode speedup
- Kernel handles both sparse metadata and INT4 dequantization

## Structured Pruning

### Layer Pruning (Depth Reduction)

**ShortGPT / LaCo**: Remove entire transformer layers
```python
# Observation: middle layers of deep transformers are often redundant
# Measure layer importance by Block Influence (BI):
# BI(layer_i) = 1 - cosine_similarity(input_to_layer_i, output_of_layer_i)
# Low BI = layer doesn't change representation much → can be removed

# For LLaMA-70B (80 layers):
# Removing 25% of layers (~20 layers) → ~1-2 PPL increase
# Remaining 60 layers → 25% less memory and latency

# Which layers to remove: typically middle layers have lowest BI
```

**LayerSkip (Meta)**: Self-speculative decoding via early exit
```python
# Train model with early exit losses at each layer
# During inference:
# 1. Draft tokens: exit after layer L_draft (e.g., layer 16 of 32)
# 2. Verify tokens: run through all layers
# Self-speculative: no separate draft model needed!

# Training:
for layer_idx, layer in enumerate(model.layers):
    hidden = layer(hidden)
    if layer_idx % exit_every == 0:
        # Apply LM head and compute loss at this layer
        loss += exit_loss_weight * compute_loss(lm_head(hidden), targets)
```

### Head Pruning (Attention)
```python
# Remove entire attention heads based on importance
# Importance metric: average attention entropy, or gradient-based scoring

# For GQA model with 8 KV heads:
# Removing 2 KV heads → 25% less KV cache, minimal quality loss
# Must retrain/fine-tune after pruning
```

### Width Pruning (SliceGPT)
```python
# Apply orthogonal transformation to weight matrices
# Then slice (remove) dimensions that contribute least

# For each layer:
# 1. Compute PCA of activations
# 2. Apply rotation matrix Q to make dimensions independent
# 3. Remove smallest-variance dimensions
# 4. Adjust next layer's weights to compensate

# Can reduce hidden dimension by 10-30% with small quality loss
```

## Knowledge Distillation

### For LLMs
```python
# Teacher: large model (e.g., LLaMA-70B)
# Student: smaller model (e.g., custom 7B)

# Standard distillation:
teacher_logits = teacher(input_ids)
student_logits = student(input_ids)

# KL divergence loss:
T = 2.0  # temperature (higher = softer distributions)
loss_kd = F.kl_div(
    F.log_softmax(student_logits / T, dim=-1),
    F.softmax(teacher_logits / T, dim=-1),
    reduction='batchmean'
) * T * T

# Combined with standard cross-entropy:
loss_ce = F.cross_entropy(student_logits, targets)
loss = alpha * loss_kd + (1 - alpha) * loss_ce
```

### Online Distillation
- Use the same model architecture but train a student with KD loss
- No need for separate inference on teacher
- Teacher and student share some parameters

## Dynamic Computation

### Mixture of Depths
```python
# Not every token needs every layer
# Learn a routing function that decides which tokens skip which layers

class MoD_Layer(nn.Module):
    def __init__(self, layer, router):
        self.layer = layer
        self.router = router  # predicts skip/compute per token

    def forward(self, x):
        routing_scores = self.router(x)  # (batch, seq_len, 1)
        top_k_mask = top_k(routing_scores, k=capacity)  # keep top-k tokens

        # Only process top-k tokens through the layer
        selected = x[top_k_mask]
        processed = self.layer(selected)
        x[top_k_mask] = processed

        return x
```

### Token Merging (ToMe)
```python
# Merge similar tokens to reduce sequence length
# Originally for ViT, applicable to LLMs

# For each layer:
# 1. Compute similarity between adjacent tokens
# 2. Merge most similar pairs (weighted average)
# 3. Process shorter sequence through next layers
# 4. Unmerge at the end (expand for output)

# Reduces sequence length by 2x → 4x less attention compute
```

## Combined Compression Techniques

### Optimal Compression Recipe

**For maximum speed with acceptable quality:**
```
1. Start with AWQ INT4 quantization (g128)
2. Apply 2:4 structured sparsity (Wanda or SparseGPT)
3. Use Marlin-24 kernel (combined sparse + quantized GEMM)
4. Result: ~8x compression, ~6x decode speedup, <1 PPL loss
```

**For maximum quality with moderate speed:**
```
1. FP8 quantization (E4M3 with per-tensor scaling)
2. No pruning (preserve all weights)
3. FP8 KV cache
4. Result: ~2x compression, ~2x speedup, <0.1 PPL loss
```

**For extreme compression (resource-constrained):**
```
1. GPTQ 3-bit quantization
2. Remove 25% of layers (ShortGPT)
3. GGUF Q3_K_M format for CPU-GPU hybrid
4. Result: ~10x compression, significant quality loss
```

### Compression Impact Summary

| Technique | Size Reduction | Speed Gain | Quality Loss (PPL) |
|-----------|---------------|-----------|-------------------|
| FP8 (from FP16) | 2x | 1.5-2x | <0.1 |
| AWQ INT4 g128 | 4x | 2-3x (decode) | 0.2-0.5 |
| GPTQ INT4 g128 | 4x | 2-3x (decode) | 0.3-0.5 |
| 2:4 Sparsity | 2x | 2x (with HW) | 0.3-0.8 |
| 2:4 + INT4 | 8x | ~6x | 0.5-1.5 |
| Layer pruning 25% | 1.3x | 1.25x | 1.0-2.0 |
| GPTQ 3-bit | 5.3x | 3-4x (decode) | 1.0-2.0 |
| GPTQ 2-bit | 8x | 4-5x (decode) | 3.0-5.0 |
| BitNet (1.58-bit) | 10x | 5-10x | Retrained from scratch |
