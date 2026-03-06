# Pruning and Model Compression Techniques: Comprehensive Reference

> A deep-dive reference covering every major pruning, sparsity, distillation, and compression technique relevant to GPU kernel optimization and efficient LLM/DNN inference.

---

## Table of Contents

1. [Unstructured Pruning](#1-unstructured-pruning)
2. [Structured Pruning](#2-structured-pruning)
3. [2:4 Structured Sparsity (NVIDIA)](#3-24-structured-sparsity-nvidia)
4. [Block Sparsity](#4-block-sparsity)
5. [Knowledge Distillation](#5-knowledge-distillation)
6. [Early Exit / Layer Skipping](#6-early-exit--layer-skipping)
7. [Model Architecture Search / Optimization](#7-model-architecture-search--optimization)
8. [Weight Sharing and Tying](#8-weight-sharing-and-tying)
9. [Low-Rank Decomposition](#9-low-rank-decomposition)
10. [Dynamic Computation](#10-dynamic-computation)
11. [Combined Approaches](#11-combined-approaches)
12. [Practical Compression Results](#12-practical-compression-results)

---

## 1. Unstructured Pruning

Unstructured pruning removes individual weight elements anywhere in the weight matrix, producing irregular sparsity patterns. The resulting sparse matrices have zeros scattered throughout, which makes them harder to accelerate on GPUs without specialized sparse computation libraries, but allows very high sparsity rates (90%+) with minimal accuracy loss.

### 1.1 Magnitude Pruning

**Core Idea**: Remove weights with the smallest absolute values, based on the assumption that low-magnitude weights contribute least to the network's function.

**Algorithm**:
1. Train the network to convergence
2. Rank all weights (or weights per layer) by absolute value
3. Set the bottom p% of weights to zero
4. Fine-tune the pruned network with a low learning rate
5. Optionally repeat steps 2-4 iteratively (iterative magnitude pruning)

**Key Details**:
- Can be applied **globally** (rank all weights across all layers together) or **locally** (rank weights within each layer independently and prune each layer to p%)
- Global pruning tends to produce better results because it allocates sparsity non-uniformly -- layers with more redundant weights get pruned more heavily
- **Iterative pruning** (prune 20%, fine-tune, prune 20% of remaining, fine-tune, repeat) outperforms one-shot pruning but requires more computation
- With weight decay during training, superfluous weights naturally decay toward zero, making magnitude a reasonable proxy for importance

**Limitations**:
- Does not account for weight interactions -- a small weight may be critical in context
- For LLMs, naive magnitude pruning causes catastrophic accuracy loss beyond 10-20% sparsity
- On OPT-175B: magnitude pruning produces unacceptable perplexity at >10% sparsity, while SparseGPT maintains quality at 60% sparsity

**Performance**: Magnitude pruning is fast and simple but typically the weakest baseline. For LLaMA-7B at 50% sparsity, magnitude pruning yields perplexity of 17.29 vs Wanda's 7.26.

### 1.2 Wanda (Pruning by Weights and Activations)

**Paper**: "A Simple and Effective Pruning Approach for Large Language Models" (Sun et al., ICLR 2024)

**Core Idea**: Instead of pruning by weight magnitude alone, Wanda uses the product of weight magnitude and the corresponding input activation norm as the pruning criterion.

**Algorithm**:
1. Collect a small calibration set (128 samples typical)
2. Run a forward pass to measure input activation norms per input feature
3. For each output neuron, compute the importance score: `S_ij = |W_ij| * ||X_j||_2`
4. Prune weights with the smallest scores on a **per-output** basis
5. No weight update or fine-tuning required

**Key Advantages**:
- **No retraining needed** -- the pruned model is used as-is
- **300x faster** than SparseGPT (minutes vs hours for large models)
- Extremely simple to implement -- just magnitude * activation norm
- Works on a per-output basis, ensuring each output neuron retains its most important connections

**Performance Results** (LLaMA-7B, 50% unstructured sparsity):
| Method | WikiText-2 Perplexity |
|--------|----------------------|
| Dense (baseline) | 5.68 |
| Magnitude Pruning | 17.29 |
| Wanda | 7.26 |
| SparseGPT | 7.22 |

- Wanda competes favorably with SparseGPT despite being orders of magnitude simpler
- Evaluated across LLaMA 7B/13B/30B/65B and LLaMA-2 7B/13B/70B

**Why It Works**: Large activation magnitudes identify "salient" input features. Weights connected to these features should be preserved even if their magnitudes are small, because they process important information. The product W*X captures this joint importance.

**Extension**: Wanda++ (Amazon, 2024) improves upon Wanda by incorporating regional gradients, achieving better accuracy at higher sparsity rates.

### 1.3 SparseGPT

**Paper**: "SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot" (Frantar & Alistarh, ICML 2023)

**Core Idea**: Use second-order (Hessian) information to optimally update remaining weights after pruning, minimizing the layer-wise reconstruction error.

**Algorithm**:
1. For each layer, compute the inverse Hessian: `H^{-1} = (XX^T + lambda*I)^{-1}` from calibration data
2. Process columns of the weight matrix sequentially in blocks of size B
3. For each block:
   a. Select which weights to prune using adaptive mask selection (blocksize Bs)
   b. Prune selected weights (set to zero)
   c. Update remaining weights in the row using the Hessian inverse to compensate for pruning error
4. The "lazy batch update" defers weight updates within a block, applying them all at once for efficiency

**Technical Innovation -- Hessian Synchronization**:
- Computing per-row Hessian inverses would be prohibitively expensive
- SparseGPT shares a single Hessian inverse across all rows of a weight matrix
- The Hessian depends only on input activations X, not the weights themselves
- Efficient rank-1 updates maintain the Hessian inverse as columns are processed

**Performance**:
- OPT-175B at 50% sparsity: negligible perplexity increase
- OPT-175B at 60% sparsity: minor perplexity increase (still usable)
- Executes on OPT-175B and BLOOM-176B in under 4.5 hours on a single GPU
- Generalizes to semi-structured 2:4 and 4:8 sparsity patterns
- Compatible with weight quantization (can jointly prune and quantize)

**Comparison with Wanda**:
| Aspect | SparseGPT | Wanda |
|--------|-----------|-------|
| Speed | Hours (large models) | Minutes |
| Quality at 50% | Slightly better | Comparable |
| Quality at 70%+ | Significantly better | Degrades faster |
| Weight updates | Yes (Hessian-based) | No |
| Calibration data | ~128 samples | ~128 samples |

**Follow-up Work**: OPTIMA (2025) improves on SparseGPT by formulating pruning as quadratic programming reconstruction, achieving better accuracy at the same sparsity levels.

### 1.4 Movement Pruning

**Paper**: "Movement Pruning: Adaptive Sparsity by Fine-Tuning" (Sanh et al., NeurIPS 2020)

**Core Idea**: Instead of pruning weights that are close to zero (magnitude), prune weights that are **moving toward zero** during fine-tuning. Retain weights that are moving away from zero, as they are becoming more important.

**Algorithm**:
1. Start with a pretrained model
2. During fine-tuning, maintain learnable score parameters S for each weight
3. The pruning mask is determined by: `M = Top_v(S)` where v is the target sparsity
4. Scores are updated via gradient descent alongside weights
5. **Soft movement pruning** uses a sigmoid on scores for differentiable masking
6. **Hard movement pruning** uses straight-through estimator

**Key Insight**: The direction of weight change during task-specific fine-tuning reveals importance. A weight moving toward zero is being "pushed away" by the loss -- it is not needed. A weight moving away from zero is being "pulled in" -- it is important for the task.

**Performance Results** (BERT fine-tuning):
- At 97% sparsity (only 3% weights remaining):
  - Movement pruning: 79.9 F1 on SQuAD
  - Magnitude pruning: 54.5 F1 on SQuAD
- At 85% sparsity: movement pruning achieves ~95% of dense BERT performance
- First-order methods excel at extreme sparsity rates (>85%)

**Integration with Distillation**: Movement pruning combines well with knowledge distillation, where a dense teacher guides the sparse student during fine-tuning.

**Block Movement Pruning**: Extended to structured blocks for better hardware efficiency, removing entire blocks of weights while using the movement criterion.

### 1.5 Lottery Ticket Hypothesis

**Paper**: "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks" (Frankle & Carlin, ICLR 2019)

**Core Hypothesis**: Dense, randomly-initialized neural networks contain sparse subnetworks ("winning tickets") that, when trained in isolation from their original initialization, reach test accuracy comparable to the full network in a similar number of training iterations.

**Finding Winning Tickets (Iterative Magnitude Pruning)**:
1. Randomly initialize network with weights `theta_0`
2. Train to convergence, reaching weights `theta_f`
3. Prune p% of weights with smallest magnitude to create mask M
4. **Reset** remaining weights to their original values `theta_0`
5. Train the subnetwork `M * theta_0` from scratch
6. Repeat steps 2-5 iteratively for higher sparsity

**Key Findings**:
- Winning tickets at 10-20% of original size match or exceed full network accuracy
- The specific initialization matters -- random re-initialization does not work
- Winning tickets generalize across different optimization procedures and datasets

**Strong Lottery Ticket Hypothesis**: Sufficiently overparameterized randomly initialized networks contain sparse subnetworks that perform well **without any training** -- they just need to be found.

**Early Bird Tickets**: Winning tickets can be identified very early in training (within 10-30% of total training epochs), enabling significant compute savings.

**Practical Limitations**:
- Finding winning tickets requires multiple rounds of training, making it expensive
- Does not scale easily to very large models (billions of parameters)
- The iterative process is compute-intensive, limiting practical deployment

---

## 2. Structured Pruning

Structured pruning removes entire groups of parameters (channels, heads, layers, neurons) rather than individual weights. The key advantage is that structured pruning produces dense submodels that run efficiently on standard hardware without sparse computation libraries.

### 2.1 Channel Pruning for CNNs

**Core Idea**: Remove entire convolutional filters/channels, reducing both the number of filters in a layer and the corresponding input channels in the next layer.

**Methods**:
- **L1-norm criterion**: Rank filters by the sum of absolute values of their weights; remove lowest-ranked filters
- **Geometric median pruning**: Remove filters closest to the geometric median (most redundant)
- **Activation-based**: Remove channels whose activations contribute least to downstream accuracy
- **Taylor expansion**: Approximate the change in loss from removing each filter using first or second-order Taylor expansion

**Implementation**:
- Removing filter k from layer l means:
  - Delete the k-th output channel (filter) from layer l's weight tensor
  - Delete the k-th input channel from layer l+1's weight tensor
  - Delete the k-th element from the batch normalization parameters (if present)
- The resulting network is a valid, dense, smaller network

**Typical Results**: Channel pruning can reduce CNN FLOPs by 30-50% with <1% accuracy loss on ImageNet. Some architectures (VGG, ResNet) tolerate aggressive pruning better than others.

### 2.2 Head Pruning for Attention

**Core Idea**: Remove entire attention heads from multi-head attention layers, as many heads are redundant.

**Key Research**:
- Michel et al. (2019) "Are Sixteen Heads Really Better Than One?" showed that in most layers, many heads can be removed after training with minimal accuracy loss
- Voita et al. (2019) identified three types of important heads: positional, syntactic, and rare-word heads
- SHRP (2024) combines head routing and pruning for encoder compression

**Pruning Criteria**:
- **Importance score**: Gradient-based scoring of each head's contribution to the loss
- **Confidence**: Heads with more uniform attention distributions are less informative
- **Taylor expansion**: First-order approximation of loss change from head removal

**Results**:
- BERT-base (12 layers, 12 heads each = 144 heads total): Can remove ~50% of heads with <1% accuracy drop on GLUE benchmarks
- Many layers function well with just 1-2 heads
- Head pruning directly reduces compute in the QKV projections and attention computation

### 2.3 Layer Pruning / Layer Skipping

**Core Idea**: Remove entire transformer blocks (self-attention + FFN), exploiting the fact that many middle-to-late layers perform nearly identity transformations.

**ShortGPT** (Men et al., 2024):
- Proposes the **Block Influence (BI) score**: `BI_i = 1 - E[cos_sim(X_i, X_{i+1})]`
- Lower BI = higher cosine similarity between layer input/output = more redundant layer
- Middle-to-later layers exhibit highest redundancy
- The final layer's FFN is crucial; its attention is less important

**ShortGPT Results**:
| Model | Layers Removed | Param Reduction | MMLU Score (Dense -> Pruned) |
|-------|---------------|-----------------|------------------------------|
| LLaMA-2-13B | 8/40 | 27.1% | 55.0 -> 54.69 |
| LLaMA-2-7B | 8/32 | 27.1% | Retains 86.3% avg performance |

**LaCo (Layer Collapse)** (Yang et al., 2024):
- Gradually merges similar layers from deep to shallow
- Uses cosine similarity between layer representations
- Sets a threshold to prevent excessive merging
- Outperforms LLM-Pruner and SliceGPT at equivalent compression ratios

**Key Insight**: Works on non-Transformer architectures too (Mamba, RWKV). Layer pruning is orthogonal to quantization -- they can be combined for multiplicative compression.

### 2.4 Width Pruning (Neuron Removal)

**Core Idea**: Remove individual neurons (columns in weight matrices) from feed-forward layers, reducing the hidden dimension.

**Methods**:
- **Activation-based**: Remove neurons with consistently low or zero activations
- **Importance scoring**: Use gradient-weighted activation magnitudes
- **LLM-Pruner** (Ma et al., 2023): Groups coupled parameters and prunes groups by first-order importance, followed by LoRA recovery fine-tuning

**Impact**: Reducing FFN hidden dimension from 4d to 3d saves 25% of FFN parameters and FLOPs. In transformers, FFN layers account for ~2/3 of total parameters, making width pruning highly effective.

### 2.5 Depth Pruning (Removing Transformer Blocks)

Depth pruning removes entire transformer blocks to create "shallower" models. This differs from layer skipping (which is dynamic) in that depth pruning permanently removes layers.

**UPDP (Unified Progressive Depth Pruner)** (2024):
- Works for both CNNs and Vision Transformers
- Progressive pruning with importance-aware layer selection
- Maintains skip connections through pruned regions

**Practical Considerations**:
- Removing too many layers causes catastrophic accuracy loss
- 20-30% depth reduction is typically feasible
- Early layers (embedding/feature extraction) and the final layer are most critical
- Middle layers are most redundant and safest to remove

### 2.6 SliceGPT

**Paper**: "SliceGPT: Compress Large Language Models by Deleting Rows and Columns" (Ashkboos et al., ICLR 2024)

**Core Idea**: Apply orthogonal rotations to the weight matrices to concentrate important information in fewer dimensions, then slice (delete) entire rows/columns of weight matrices to reduce the embedding dimension.

**Algorithm**:
1. Compute PCA of layer activations on calibration data
2. Determine an orthogonal rotation matrix Q that aligns principal components
3. Apply rotation: `W' = Q^T W Q` (absorb rotation into adjacent layers)
4. Slice: remove the least important rows/columns (those corresponding to smallest principal components)
5. The resulting model has a permanently reduced embedding dimension

**Key Innovation**: Unlike other pruning methods that produce sparse matrices, SliceGPT produces **dense, smaller** matrices. Every remaining element is nonzero, so standard dense GEMM kernels are used with no sparse computation overhead.

**Results**:
| Model | Params Removed | Zero-Shot Task Performance Retained |
|-------|---------------|-------------------------------------|
| LLaMA-2 70B | 25% | 99% |
| OPT 66B | 25% | 99% |
| Phi-2 | 25% | 90% |

**Advantages over Sparse Pruning**:
- No need for sparse computation libraries or hardware
- Directly reduces matrix dimensions, giving guaranteed speedup
- Works with any hardware that supports dense GEMM

---

## 3. 2:4 Structured Sparsity (NVIDIA)

### 3.1 How It Works

NVIDIA's 2:4 structured sparsity enforces a specific pattern: in every contiguous group of 4 elements, exactly 2 must be zero. This provides 50% sparsity with a highly regular structure that dedicated hardware can exploit.

**Pattern Example**:
```
Original:  [0.5, 0.0, 0.3, 0.0, 0.0, 0.7, 0.0, 0.2]
             |    Z    |    Z    Z    |    Z    |
           2 nonzero, 2 zero  |  2 nonzero, 2 zero
```

**Compression Format**:
- **Values**: Store only the 2 nonzero values per group (50% of original data)
- **Metadata**: 2-bit index per nonzero value indicating its position within the group of 4
- Total metadata overhead: 2 bits per nonzero element = very small
- Overall compression: close to 2x (values are halved, metadata is negligible)

### 3.2 Tensor Core Support on Ampere+

**Hardware**: NVIDIA A100 (Ampere), A6000, RTX 3090, H100 (Hopper), B100 (Blackwell)

**Sparse Tensor Cores (SpMMA)**:
- Third-generation Tensor Cores on Ampere include dedicated sparse matrix-multiply-accumulate units
- The sparse operand is decompressed on-the-fly using the metadata
- Only the matching elements from the dense operand are fetched
- **Result**: 2x effective throughput compared to dense Tensor Cores for the same hardware

**Supported Data Types**:
| Input Type | Output/Accumulator | Available Since |
|-----------|-------------------|-----------------|
| FP16 | FP32 | Ampere (A100) |
| BF16 | FP32 | Ampere (A100) |
| INT8 | INT32 | Ampere (A100) |
| FP8 | FP32 | Hopper (H100) |
| TF32 | FP32 | Ampere (A100) |

### 3.3 Training with 2:4 Sparsity

**NVIDIA's Three-Step Workflow**:
1. **Train** the dense network normally
2. **Prune** to 2:4 pattern (e.g., magnitude-based within each group of 4)
3. **Retrain** with the sparsity mask frozen to recover accuracy

**PyTorch Dynamic Pruning Approach** (torchao):
- Instead of fixing the mask, dynamically compute the optimal 2:4 mask at each training step
- Uses `apply_fake_sparsity` during training to simulate the sparsity constraint
- At inference, convert to compressed format with `to_sparse_semi_structured()`
- Achieves comparable accuracy to dense training

**Training Speedup Results**:
| Model | Task | Dense GEMM | Sparse GEMM | Speedup |
|-------|------|-----------|-------------|---------|
| ViT-L | DINOv2 Training | 538 us | 387 us | 1.39x |
| ViT-L | End-to-end Training | - | - | 1.06x (6%) |
| SAM | Inference | - | - | 1.10x |

The gap between theoretical 2x speedup and practical 1.1-1.4x is due to:
- Non-GEMM operations (normalization, activation, attention) are unchanged
- Memory bandwidth bottlenecks in batch sizes below saturation
- Amdahl's law -- GEMM is only part of total computation

### 3.4 cuSPARSELt Library

**Purpose**: High-performance CUDA library for structured sparse matrix multiplication, abstracting away the complexity of Sparse Tensor Core programming.

**API Workflow** (9 steps):
1. Initialize library handle (`cusparseLtInit`)
2. Define input/output matrix descriptors (dimensions, data types, layouts)
3. Create matmul descriptor
4. Set algorithm selection (auto-tuning available)
5. Create execution plan
6. Prune matrix A to 2:4 pattern (`cusparseLtSpMMAPrune`)
7. Compress the pruned matrix (`cusparseLtSpMMACompress`)
8. Execute: `cusparseLtMatmul` -- the actual sparse GEMM
9. Clean up resources

**Performance Benchmarks** (BERT-Large, seq_len=128, batch=128):
| Layer | Dense TFLOPs | Sparse TFLOPs | Speedup |
|-------|-------------|---------------|---------|
| QKV | 188 | 263 | 1.40x |
| FC2 | 211 | 339 | 1.61x |

**Key Features**:
- Auto-tuning selects optimal kernel configuration
- Supports row-major and column-major layouts
- Execution time for pruning + compression is <5% of total GEMM time
- Matrix pruning can be skipped if weights are already 2:4 sparse (e.g., from ASP)

### 3.5 ASP (Automatic Sparsity)

**NVIDIA ASP Library**: PyTorch utility that automates the process of inducing 2:4 sparsity in neural networks.

**Usage**:
```python
from apex.contrib.sparsity import ASP

# After model is trained, apply 2:4 sparsity
ASP.prune_trained_model(model, optimizer)

# Continue training with sparsity mask frozen
for epoch in range(recovery_epochs):
    train(model, optimizer)  # ASP ensures pruned weights stay zero
```

**How ASP Selects Which 2 of 4 to Keep**:
- Default: magnitude-based (keep the 2 largest absolute values in each group of 4)
- Custom pruning functions can be registered
- The mask is computed once and frozen during recovery training

### 3.6 Combining 2:4 Sparsity with Quantization

Combining 2:4 sparsity (2x compression) with 4-bit quantization (4x compression) yields **8x total compression**.

**Sparse-Marlin Kernel**:
- MARLIN (Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models) extends to 2:4 sparse weights
- A 4-bit quantized + 2:4 sparse matrix is half the size of a 4-bit dense matrix in the inner dimension
- Integrated with vLLM for production serving
- End-to-end inference speedups up to 2.8x

**MaskLLM** (2024):
- Uses learnable masks to achieve 2:4 sparsity
- Recovers accuracy comparable to unstructured sparsity approaches
- Specifically designed for the 2:4 constraint

**Accuracy Comparison** (LLaMA-2 7B, ~8x compression):
| Method | Compression | Relative Quality |
|--------|------------|-----------------|
| 2-bit quantization alone | 8x | Lower accuracy |
| 4-bit quant + 2:4 sparsity | 8x | Higher accuracy |

The combined approach achieves the same compression ratio with better accuracy because neither technique is pushed to its extreme.

---

## 4. Block Sparsity

### 4.1 Block-Sparse GEMM

**Core Idea**: Instead of individual element sparsity, enforce sparsity at the block level -- entire sub-matrices (e.g., 32x32, 64x64) are either all zero or all nonzero.

**Advantages**:
- Regular memory access patterns -- each block is contiguous in memory
- Naturally maps to GPU tile-based computation (CUTLASS, Triton)
- Avoids the irregular memory access overhead of unstructured sparsity
- Simpler indexing: only need a block-level mask, not element-level

**NVIDIA cuSPARSE Block-Sparse Support** (cuSPARSE 11.4+):
- High-performance block sparse matrix multiplication using dense Tensor Cores for nonzero sub-matrices
- When block size is 32, the kernel is faster than cuBLAS if density < 40% on Volta and < 50% on Ampere

**OpenAI Block-Sparse Library**:
- Efficient GPU kernels for block-sparse matrix multiplication and convolution
- Used in Sparse Transformers for long-sequence attention
- Block sizes of 8, 16, 32, 64 supported

**Performance Scaling**:
| Block Size | Density Threshold for Speedup (vs Dense) | Typical Speedup at 50% Density |
|-----------|------------------------------------------|-------------------------------|
| 8x8 | ~20% | Marginal |
| 16x16 | ~30% | 1.2-1.5x |
| 32x32 | ~40-50% | 1.5-2.0x |
| 64x64 | ~50-60% | 1.3-1.7x |

### 4.2 Block-Sparse Attention Patterns

**Sparse Transformers** (Child et al., 2019):
- Standard attention is O(n^2) -- block-sparse attention reduces to O(n*sqrt(n))
- Factorized patterns: attend to local window + strided positions
- Block size determines granularity of sparse pattern

**Longformer** (Beltagy et al., 2020):
- Sliding window attention (local) + global attention on special tokens
- Implemented as block-sparse operations

**BigBird** (Zaheer et al., 2020):
- Combines local, global, and random attention patterns
- Block-sparse implementation for efficiency

### 4.3 N:M Sparsity Generalization

N:M sparsity generalizes the 2:4 pattern: N nonzero elements in every contiguous group of M elements.

**Common Patterns**:
| Pattern | Sparsity | Hardware Support | Typical Use |
|---------|----------|-----------------|------------|
| 2:4 | 50% | NVIDIA Ampere+ (native) | Production deployment |
| 4:8 | 50% | Software (CUTLASS) | Same sparsity, larger blocks |
| 1:2 | 50% | Custom CUDA kernels | Fine-grained variant |
| 1:4 | 75% | Custom kernels | Higher compression |

**V:N:M Sparsity** (2024): Extends N:M to vector-level, where V consecutive elements are treated as a single unit. This improves cache locality and enables broader hardware support.

**GPU Kernel Design for N:M**:
- Dedicated CUDA kernels completely eliminate the dynamic pruning overhead
- Achieve 1.27-1.89x speedups over full attention under arbitrary sequence lengths
- The metadata encoding maps naturally to GPU warp-level operations

---

## 5. Knowledge Distillation

### 5.1 Teacher-Student Framework

**Core Idea**: Train a small "student" model to mimic the behavior of a large "teacher" model, transferring the teacher's "dark knowledge" (soft probability distributions) to the student.

**Classical Formulation** (Hinton et al., 2015):
```
L_distill = alpha * KL(softmax(z_t/T), softmax(z_s/T)) + (1-alpha) * L_CE(y, z_s)
```
Where:
- `z_t, z_s`: teacher and student logits
- `T`: temperature (typically 2-20, higher = softer distributions)
- `alpha`: interpolation weight (typically 0.5-0.9)
- The KL divergence term transfers teacher knowledge
- The cross-entropy term maintains ground-truth alignment

**Why Soft Labels Help**: The teacher's soft probability distribution encodes similarity relationships between classes. For example, a cat image might get probabilities [0.7 cat, 0.15 dog, 0.05 tiger, ...], telling the student that cats are similar to dogs and tigers -- information not present in hard one-hot labels.

### 5.2 Online Distillation

**Core Idea**: Teacher and student are trained simultaneously, with the teacher continuously updated.

**Approaches**:
- **Co-distillation**: Multiple models train together, each serving as teacher to others
- **Self-distillation**: A model distills knowledge from its own deeper layers to shallower ones
- **On-policy distillation** (2024-2025): Computes teacher-student divergence on **student-generated** rollouts, not fixed datasets

**Advantages over Offline**:
- No need for a pre-trained teacher
- Teacher adapts as student improves
- Better for continual learning scenarios

### 5.3 Task-Specific Distillation

**Core Idea**: Distill a large model into a smaller one for a specific downstream task, rather than general capabilities.

**Approaches**:
- **Feature-level distillation**: Match intermediate representations, not just outputs
- **Attention transfer**: Match attention patterns between teacher and student
- **Rationale distillation**: Teacher generates reasoning chains that student learns to reproduce

### 5.4 Notable Distilled Models

**DistilBERT** (Sanh et al., 2019):
- 6 transformer layers (vs 12 in BERT-base)
- 66M parameters (vs 110M in BERT-base)
- **40% smaller, 60% faster, 97% of BERT's accuracy**
- Uses triple loss: MLM loss + distillation loss + cosine embedding loss
- Every other layer initialized from teacher

**TinyBERT** (Jiao et al., 2020):
- Two-stage distillation: general + task-specific
- 14.5M parameters
- 7.5x smaller, 9.4x faster than BERT-base
- Matches BERT on GLUE with 4 layers

**TinyLlama** (Zhang et al., 2024):
- 1.1B parameters trained on 3T tokens
- Uses distillation from larger LLaMA models
- Competitive with 7B models on some benchmarks

**LLM Distillation Techniques** (2024-2025):
- **Offline distillation**: Students trained on teacher-generated trajectories
- **Off-policy distillation**: Matches teacher and student distributions
- **On-policy distillation**: Divergence computed on student-generated rollouts
- **Multi-teacher distillation**: Multiple teachers provide diverse knowledge, though knowledge conflicts must be managed
- **Reinforcement-aware distillation**: Combines RL (RLHF/RLAIF) with distillation for LLM reasoning

### 5.5 Kernel Implications

Distilled models are smaller dense models that use standard GEMM kernels with smaller matrix dimensions. The primary kernel optimization opportunity is:
- Smaller weight matrices may fall into different GEMM tile size regimes
- Batch sizes and sequence lengths become relatively larger compared to hidden dimensions
- May benefit from different CUTLASS tile configurations than the teacher model

---

## 6. Early Exit / Layer Skipping

### 6.1 Dynamic Inference with Early Exit

**Core Idea**: Not all inputs require the same amount of computation. Easy inputs can exit the model early (after fewer layers), while hard inputs use all layers.

**Architecture**:
- Add lightweight classification/prediction heads at intermediate layers
- During inference, check confidence at each exit point
- If confidence exceeds threshold, return the prediction from that layer
- Otherwise, continue to the next layer

**Confidence Criteria**:
- **Softmax entropy**: Exit when output distribution has low entropy (model is confident)
- **Max probability**: Exit when max class probability exceeds threshold
- **Learned threshold**: Train a small network to decide when to exit

### 6.2 CALM (Confident Adaptive Language Modeling)

**Paper**: Schuster et al. (2022)

**Core Idea**: Dynamically allocate different amounts of compute per input and per generation timestep in language models.

**Key Features**:
- Pretrained with early exit losses at every layer
- Optimal criteria learned for deciding exit layer
- Per-token computation allocation -- frequent/simple tokens exit early
- Calibrated confidence thresholds ensure output quality

### 6.3 LayerSkip (Meta, ACL 2024)

**Paper**: "LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding" (Elhoushi et al.)

**Three Components**:

**1. Training with Layer Dropout**:
- Dropout rate increases exponentially with layer depth: `D(l) = e^(l * ln2 / (L-1)) - 1`
- Early layers: low dropout (always computed)
- Late layers: high dropout (frequently skipped)
- Forces the model to be less reliant on later layers
- Early exit loss at every layer with quadratically higher weight for earlier layers

**2. Early Exit Inference**:
- The training recipe enables accurate prediction from early layers
- No auxiliary modules needed -- shared exit head across all layers
- Model dynamically chooses exit point based on confidence

**3. Self-Speculative Decoding**:
- **Draft stage**: Generate d tokens by exiting at layer E (using only early layers)
- **Verification stage**: Verify and correct using remaining layers L-E in parallel
- Shared KV cache between draft and verification stages
- Exit Query Cache: saves query vectors from layer E-1, enabling direct continuation from E to L

**Performance Results**:
| Task | Model | Speedup | Token Acceptance Rate |
|------|-------|---------|----------------------|
| CNN/DM Summarization | Llama2 7B | 1.86x | 77.8% |
| XSUM Summarization | Llama2 7B | 1.54x | 54.6% |
| HumanEval Coding | Llama2 7B | 1.83x | - |
| CNN/DM (from scratch) | Llama2 7B | 2.16x | - |
| TOPv2 Parsing | Llama 1.5B | 2.0x | - |

**Memory Advantage**: KV cache reuse provides 9-20ms per-token reduction. Unlike standard speculative decoding (which requires a separate draft model), self-speculative decoding uses the same model, halving memory requirements.

### 6.4 Kernel Implications

Early exit and layer skipping create unique challenges for GPU kernel design:

**Variable-Depth Execution**:
- Different tokens in a batch may exit at different layers
- Requires either: (a) padding to max depth (wastes compute), (b) splitting batches by exit layer (complex scheduling), or (c) masked computation (partial utilization)
- The most practical approach is self-speculative decoding, which maintains regular computation patterns

**KV Cache Management**:
- Early-exit tokens have partial KV caches
- Verification requires extending these caches
- Efficient cache reuse is critical for performance

**Batched Mixed-Depth Execution**:
- In batch inference, different sequences may exit at different layers
- Need efficient masking or batch reorganization
- CUDA graph compatibility is challenging with variable depths

---

## 7. Model Architecture Search / Optimization

### 7.1 Neural Architecture Search (NAS)

**Core Idea**: Automatically search for optimal neural network architectures that balance accuracy, speed, and model size.

**Search Space Components for Transformers**:
- Number of layers (depth)
- Hidden dimension (width)
- Number of attention heads
- FFN intermediate dimension
- Attention type (full, local, linear)
- Activation functions
- Normalization placement (pre-norm vs post-norm)

**Search Strategies**:
- **Reinforcement Learning**: Train a controller network that proposes architectures
- **Evolutionary Search**: Mutate and select architectures based on fitness
- **Differentiable NAS (DARTS)**: Make architecture choices differentiable, optimize via gradient descent
- **One-shot NAS**: Train a supernet containing all candidate architectures, then search for the best sub-network

**Recent Advances (2024-2025)**:
- **GPT-NAS**: Uses generative pre-trained models to propose architectures
- **NASViT**: Gradient conflict-aware supernet training for vision transformers
- **Multi-objective NAS**: Balances accuracy, latency, energy consumption, and robustness simultaneously
- **Hardware-aware NAS**: Considers GPU/TPU-specific constraints (memory bandwidth, compute units, instruction throughput)

### 7.2 Once-for-All Networks

**Paper**: Cai et al. (2020), MIT HAN Lab

**Core Idea**: Train a single large "once-for-all" (OFA) network that supports many sub-networks of different sizes. Deploy different sub-networks for different hardware constraints without retraining.

**Training**:
1. Train the largest network
2. Progressively shrink: depth, width, kernel size, resolution
3. Each sub-network shares weights with the full network
4. Knowledge distillation from larger sub-networks to smaller ones

**Deployment**:
- Given target hardware constraints (latency, memory), search for the best sub-network
- Search takes minutes on a laptop (no GPU needed)
- ImageNet accuracy within 1% of independently trained models

### 7.3 Efficient Model Design Patterns

**Key Patterns for Efficient Transformers**:
- **Grouped Query Attention (GQA)**: Share KV heads across query groups (LLaMA-2 70B uses 8 KV heads for 64 query heads)
- **Multi-Query Attention (MQA)**: Single KV head for all queries (fastest, some accuracy loss)
- **SwiGLU FFN**: Gated linear units with SiLU activation (better accuracy/FLOP)
- **Rotary Position Embeddings (RoPE)**: Efficient relative position encoding
- **Pre-Norm**: Layer normalization before attention/FFN (more stable training)
- **Mixture of Experts (MoE)**: Sparse activation -- only a subset of parameters are used per token

---

## 8. Weight Sharing and Tying

### 8.1 Embedding Weight Tying (Input/Output)

**Core Idea**: Share the same weight matrix between the input embedding layer and the output projection layer (which maps hidden states back to vocabulary logits).

**Implementation**:
```python
# Standard: separate embedding and output projection
self.embed = nn.Embedding(vocab_size, hidden_dim)  # [V, D]
self.lm_head = nn.Linear(hidden_dim, vocab_size)    # [D, V]

# With weight tying: share the same matrix
self.embed = nn.Embedding(vocab_size, hidden_dim)
self.lm_head.weight = self.embed.weight  # Shared!
```

**Savings**: For a vocabulary of 32K tokens and hidden dimension of 4096:
- Untied: 2 * 32K * 4096 = 256M parameters
- Tied: 1 * 32K * 4096 = 128M parameters (50% reduction in embedding params)
- For encoder-decoder models: full weight tying eliminates 2/3 of embedding parameters

**Impact on Quality**: Weight tying often **improves** model quality, particularly for smaller models, because:
- Regularization effect prevents overfitting
- Forces input and output spaces to be aligned
- Reduces the number of parameters that need to be learned

**Adoption**: Used in GPT-2, T5, many modern LLMs. Some very large models (GPT-3, LLaMA) do NOT tie weights because the embedding dimension is small relative to total parameters.

### 8.2 Cross-Layer Weight Sharing

**Core Idea**: Reuse the same transformer block weights across multiple layers instead of having unique parameters per layer.

**Approaches**:
- **Full sharing**: All layers use identical weights (ALBERT)
- **Cyclic sharing**: Layers 1-4 share weights, layers 5-8 share different weights, etc.
- **Sandwich sharing**: First and last few layers are unique; middle layers share weights

**ALBERT** (Lan et al., 2020):
- Shares all transformer layer parameters (attention + FFN)
- 12 "virtual" layers, 1 set of parameters
- 18x fewer parameters than BERT-large
- 70% of BERT-large accuracy with 1/18 the parameters
- Accuracy recoverable with more training data and longer training

**Subformer** (Reid et al., 2021):
- **Sandwich-style sharing**: Unique parameters for top/bottom layers, shared for middle
- Overcomes quality degradation of naive cross-layer sharing in generative models
- **SAFE (Self-Attentive Embedding Factorization)**: Factorizes embedding layer for additional parameter reduction
- Outperforms standard Transformers with significantly fewer parameters

**Universal Transformer** (Dehghani et al., 2019):
- Applies the same transformer block repeatedly (variable number of times)
- Per-token halting mechanism decides when to stop iterating
- Turing-complete (unlike standard fixed-depth transformers)
- Cross-layer sharing is the most popular weight sharing solution

### 8.3 Modern Weight Sharing: MASA (2025)

**Matrix Atom Sharing in Attention**: Represents each layer's attention weights as linear combinations of shared "matrix atoms" (a dictionary of basis matrices).

- Reduces attention module parameters by 66.7%
- Maintains competitive performance
- Different layers use different linear combinations of the same atom set

**DeltaLLM** (2025):
- Shares weights between consecutive transformer blocks
- Stores only low-rank "delta" matrices for each layer: `W_l = W_shared + Delta_l`
- Deltas are much smaller than full weight matrices
- Post-training compression -- no retraining needed

---

## 9. Low-Rank Decomposition

### 9.1 SVD-Based Weight Decomposition

**Core Idea**: Decompose a weight matrix W (m x n) into two smaller matrices using Singular Value Decomposition (SVD): `W ≈ U_r * S_r * V_r^T` where only the top r singular values are kept.

**Compression**: A matrix W of shape (m, n) with rank r approximation becomes:
- U_r: (m, r) and S_r * V_r^T: (r, n)
- Parameters: m*r + r*n instead of m*n
- Compression ratio: mn / (r*(m+n))
- For r = m/4 with square matrices: 2x compression

**Inference Speedup**: The decomposed form replaces one large matrix multiplication with two smaller ones:
```
# Original: y = Wx, one GEMM of shape (m, n)
y = W @ x

# Decomposed: two GEMMs of shapes (m, r) and (r, n)
y = U @ (SV @ x)
```
When r << min(m,n), the decomposed form is faster.

### 9.2 LoRA (Low-Rank Adaptation)

**Paper**: "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2022)

**Original Use**: Parameter-efficient fine-tuning -- keep pretrained weights frozen, add trainable low-rank matrices.

**Architecture**:
```
W' = W + alpha/r * B @ A
```
Where W is frozen (m x n), A is (r x n), B is (m x r), and r << min(m, n).

**For Compression/Inference**:
- After fine-tuning, merge: `W_merged = W + alpha/r * B @ A`
- The merged matrix is the same size as W -- no inference speedup from LoRA itself
- But LoRA enables efficient compression of multiple model variants:
  - Store base model once + small LoRA adapters per task
  - Each adapter is only r*(m+n) parameters instead of m*n

**LoRA Adapter Compression** (2024):
- Post-hoc SVD compression of LoRA adapters themselves
- Effective rank of trained LoRA adapters is often much lower than r
- Can compress adapters by 2-10x with minimal quality loss

**LoRA-XS**: Uses extremely small number of parameters by fixing A and B as random projections and only training a small (r x r) matrix between them.

### 9.3 ASVD (Activation-Aware SVD)

**Paper**: "ASVD: Activation-aware Singular Value Decomposition for Compressing Large Language Models" (Yuan et al., 2024)

**Problem with Standard SVD**: Weight matrices with activation outliers produce poor low-rank approximations because a few large singular values dominate.

**Solution**: Transform the weight matrix based on the activation distribution before applying SVD:
1. Compute activation statistics from calibration data
2. Scale weight matrix columns by activation standard deviations
3. Apply SVD to the scaled matrix
4. Absorb the scaling back into the decomposed factors

**Results**:
- 10-30% model compression with minimal quality loss (training-free)
- Additionally achieves 50% KV cache reduction without performance degradation
- Works as a plug-in improvement over standard SVD

### 9.4 SVD-LLM (ICLR 2025)

**Innovation**: Links compression loss to the singular spectrum by whitening activations via Cholesky factorization of their covariance.

**Algorithm**:
1. Compute activation covariance matrix
2. Cholesky factorization for whitening transform
3. Apply SVD in the whitened space (singular values now directly reflect loss contribution)
4. Truncate based on target rank
5. Optional: LoRA-style fine-tuning for accuracy recovery

**SVD-LLM V2** (NAACL 2025): Further improvements with better whitening and recovery strategies.

### 9.5 Delta Compression for Model Variants

**Problem**: Serving many fine-tuned variants of the same base model requires storing many full copies.

**Delta-CoMe** (NeurIPS 2024):
- Computes delta: `Delta = W_finetuned - W_base`
- SVD of delta matrix
- Mixed-precision: larger singular values stored in high-bit, smaller in low-bit
- Training-free compression

**DeltaZip**:
- Compresses model deltas by up to 10x while maintaining quality
- Achieves 2-12x throughput improvement for multi-model serving
- Enables concurrent serving of many fine-tuned models

**ImPart** (2025):
- Importance-aware delta sparsification
- Layer-wise pruning of delta weights based on importance scores
- Better compression than uniform pruning

---

## 10. Dynamic Computation

### 10.1 Mixture of Depths (MoD)

**Paper**: "Mixture-of-Depths: Dynamically allocating compute in transformer-based language models" (Raposo et al., DeepMind, 2024)

**Core Mechanism**:
- At specific routing layers, a lightweight router assigns a score to each token
- Top-k tokens (by score) proceed through the full transformer block
- Remaining tokens bypass via residual connection (identity)
- k is defined a priori, creating a static computation graph

**Router Design**:
```
r_i = w_theta^T * x_i    # Linear projection -> scalar routing score
tokens_to_compute = top_k(r, k=C)  # Select top C tokens
tokens_to_skip = remaining tokens   # Bypass via residual
```

**Optimal Configuration**:
- Route every other block (alternating with full-capacity blocks)
- Capacity C = 12.5% of sequence length (87.5% of tokens bypass)
- Learned routing only -- stochastic routing underperforms

**FLOP Savings**:
- Attention cost for routed layers: C^2/S^2 of vanilla (0.125^2 = 1.56% at C=12.5%)
- Overall per-forward-pass FLOPs reduce proportionally to capacity
- Models match baseline performance at equivalent training FLOPs

**Performance**:
- 220M MoD variant matches 220M baseline while being **60% faster** during training
- 50%+ faster during post-training sampling
- 1.5% improvement on training objective at equivalent FLOPs

**MoDE (Mixture-of-Depths-and-Experts)**:
- Combines MoD with MoE in integrated or staged configurations
- Performance improvements from MoD compound with MoE improvements

### 10.2 Token Dropping / Early Termination

**Approaches**:
- **Token pruning**: Remove unimportant tokens from the sequence after early layers
- **Token merging**: Combine similar tokens to reduce sequence length
- **Early termination**: Stop processing a token once its representation has converged

**Efficient token pruning** uses attention-based importance scoring: tokens that receive little attention are likely unimportant and can be dropped.

### 10.3 Dynamic Token Merging (ToMe)

**Paper**: "Token Merging: Your ViT But Faster" (Bolya et al., ICLR 2023, Meta)

**Algorithm**:
1. At each transformer layer, compute pairwise token similarity using the keys (K) from attention
2. Use a fast bipartite matching algorithm to pair similar tokens
3. Merge paired tokens by averaging their representations
4. The merged token inherits the combined attention pattern

**Key Properties**:
- **Training-free**: Works on any pretrained ViT without fine-tuning
- **Faster than pruning**: Merging preserves information; pruning discards it
- **Gradual reduction**: Each layer merges a few tokens, progressively reducing sequence length

**Performance** (off-the-shelf, no training):
| Model | Throughput Gain | Accuracy Drop |
|-------|----------------|---------------|
| ViT-L @ 512 | 2.0x | 0.2-0.3% |
| ViT-H @ 518 | 2.0x | 0.2-0.3% |
| ViT-L (video) | 2.2x | ~0.3% |

**Extensions**:
- **Learnable Token Merging (LTM)**: Learns optimal merging strategy
- **K-Feature Fusion (KFF) Token Merging**: Reduces similarity metric error
- **ToMA (Token Merge with Attention)**: Extends to diffusion models
- Works in image, video, and audio modalities

### 10.4 Adaptive Computation Time

**Mixture-of-Recursions (MoR)** (NeurIPS 2025):
- Combines parameter sharing (recursive transformer) with adaptive computation
- Reuses a shared stack of layers across recursion steps
- Lightweight routers dynamically assign different recursion depths per token
- Focuses attention computation only among tokens still active at each depth
- Enables per-token "thinking depth" adaptation

---

## 11. Combined Approaches

### 11.1 Quantization + Pruning + Distillation

**Deep Compression Pipeline** (Han et al., 2016):
1. **Prune**: Remove 90% of weights by magnitude
2. **Quantize**: Reduce remaining weights to 4-8 bits via k-means clustering
3. **Huffman encode**: Variable-length coding for further compression
4. Result: 35-49x compression on AlexNet/VGG with no accuracy loss

**Modern LLM Pipeline**:
1. Prune with SparseGPT or Wanda to 50% sparsity
2. Quantize with GPTQ or AWQ to 4-bit
3. Optionally distill from original model to recover accuracy
4. Result: 8-16x compression with minimal perplexity increase

### 11.2 QLoRA + Sparsity

**QLoRA** (Dettmers et al., 2023):
- 4-bit NormalFloat (NF4) quantization of base model
- Double quantization: quantize the quantization constants
- Paged optimizer with unified memory management
- Fine-tune with LoRA adapters on top of quantized weights
- Enables fine-tuning 65B models on a single 30GB GPU

**QLORAM** (2024):
- Combines LORAM (low-rank model compression) with QLoRA
- Applies pruning/sparsification alongside quantization
- Further reduces memory footprint beyond QLoRA alone

**LoftQ** (2024):
- Identifies that zero-initialized LoRA in QLoRA is suboptimal
- Initializes LoRA matrices from SVD of (original - quantized) weight difference
- Significant improvement on downstream tasks

**Q-BLoRA**:
- Simplifies adapter inputs/outputs while increasing rank
- Alleviates underfitting during quantized model fine-tuning

### 11.3 AWQ + 2:4 Sparsity (Marlin-24)

**MARLIN** (Frantar et al., 2024):
- Mixed-precision auto-regressive parallel inference kernel
- Near-optimal performance on individual LLM layers
- End-to-end speedups up to **2.8x** when integrated with vLLM

**Sparse Marlin (Marlin-24)**:
- Extends MARLIN to 2:4 structured sparse weights
- 4-bit quantized + 2:4 sparse = half the size of 4-bit dense in inner dimension
- Integrated with vLLM for production serving
- Supported on NVIDIA capability >= 8.0 (Ampere, Ada Lovelace, Hopper)

**SLiM** (2024):
- One-shot quantization and sparsity with low-rank approximation
- Uses Sparse Marlin integrated with vLLM
- Achieves notable speedups through optimized sparse+quantized matmul

**AWP** (Activation-Aware Weight Pruning and Quantization, 2025):
- Joint pruning and quantization with activation-aware scaling
- Extends AWQ's scaling strategy to handle both compression dimensions

### 11.4 Optimal Compression Recipes by Model Size

Based on comprehensive benchmarking (2024-2025):

| Model Size | Best Recipe | Compression | Quality Retention |
|-----------|------------|-------------|-------------------|
| <1B params | INT8 quantization only | 2x | 99%+ |
| 1-7B | W4A16 (GPTQ/AWQ) | 4x | 98-99% |
| 7-13B | W4A16 + 2:4 sparsity | 8x | 96-98% |
| 13-70B | W4A16 (AWQ preferred) | 4x | 99%+ |
| 70B+ | W4A16 or W8A8 + MoE routing | 4x | 99%+ |
| Multi-model serving | W4A16 + delta compression | 10-40x per variant | 98%+ |

**Key Findings**:
- For small models (<3B): be cautious with 4-bit quantization; INT8 is safer
- For large models (>13B): quantization alone often suffices; pruning is higher risk
- 2:4 sparsity is most beneficial in the 7-13B range where memory is tight but models can tolerate 50% sparsity
- Combined approaches (quant+sparse) shine when targeting specific hardware (NVIDIA Ampere+)

---

## 12. Practical Compression Results

### 12.1 Quantization Benchmarks

**ATOM** (MLSys 2024) -- LLaMA-65B:
- 4-bit quantization: only 0.3 perplexity increase on WikiText-2
- Throughput: 7.7x vs FP16, 5.5x vs W4A16, 2.5x vs W8A8

**Red Hat Large-Scale Evaluation** (2024):
- Over 500,000 evaluations across quantized LLMs
- Response quality remains highly competitive with full-precision on Arena-Hard-Auto
- AWQ generally outperforms GPTQ in weight-only quantization
- FP8 with hardware support is even more advantageous

**LLM Compressor + vLLM** (LLaMA 3.1 70B, 4xA100):
| Method | Speedup | GPU Efficiency | Accuracy Recovery |
|--------|---------|---------------|-------------------|
| W8A8 (INT8) | 1.6x at 5 QPS | 2 GPUs match FP16 on 4 GPUs | 99.6%+ |
| W4A16 | Minimal | Better memory efficiency | 99%+ |
| FP8 | ~1.5x | Native hardware support | 99.5%+ |

### 12.2 Pruning Benchmarks

**SparseGPT Results**:
| Model | Sparsity | Perplexity (Dense -> Sparse) | Time |
|-------|----------|------------------------------|------|
| OPT-175B | 50% | Negligible increase | <4.5h |
| OPT-175B | 60% | Minor increase | <4.5h |
| BLOOM-176B | 50% | Negligible increase | <4.5h |
| LLaMA-7B | 50% | 5.68 -> 7.22 | Minutes |

**Wanda Results** (LLaMA family, 50% unstructured):
| Model | Dense PPL | Wanda PPL | Magnitude PPL |
|-------|-----------|-----------|---------------|
| LLaMA-7B | 5.68 | 7.26 | 17.29 |
| LLaMA-13B | 5.09 | 6.15 | 11.43 |
| LLaMA-30B | 4.10 | 4.87 | 7.81 |

**ShortGPT Layer Removal** (LLaMA-2-13B):
- 27% layer removal: MMLU drops only 0.31 points (55.0 -> 54.69)
- Average performance retention: 91.6%

### 12.3 Combined Compression Benchmarks

**2:4 Sparsity + INT4 Quantization** (via vLLM/Sparse-Marlin):
| Model | Method | Memory Reduction | Inference Speedup |
|-------|--------|-----------------|-------------------|
| LLaMA-2 7B | W4 + 2:4 sparse | 8x | ~2x |
| LLaMA-2 13B | W4 + 2:4 sparse | 8x | ~2x |
| Various | MARLIN kernel | 4x (W4 only) | 2.8x (end-to-end) |

**cuSPARSELt Performance** (BERT-Large, batch=128):
| Layer Type | Dense TFLOPs | Sparse TFLOPs | Speedup |
|-----------|-------------|---------------|---------|
| QKV projection | 188 | 263 | 1.40x |
| FC2 (FFN) | 211 | 339 | 1.61x |

### 12.4 Distillation Results

| Student Model | Teacher | Size Reduction | Speed Improvement | Quality Retention |
|--------------|---------|---------------|-------------------|-------------------|
| DistilBERT | BERT-base | 40% smaller | 60% faster | 97% accuracy |
| TinyBERT-4L | BERT-base | 7.5x smaller | 9.4x faster | ~96% on GLUE |
| DistilGPT-2 | GPT-2 | 2x smaller | 2x faster | ~95% perplexity |

### 12.5 Early Exit / LayerSkip Results

**LayerSkip** (LLaMA-2 7B):
| Task | Speedup | Throughput (tokens/sec) |
|------|---------|------------------------|
| CNN/DM Summarization | 1.86x | 62.7 -> 127.9 |
| HumanEval Coding | 1.83x | - |
| CNN/DM (from scratch) | 2.16x | - |

### 12.6 Quality vs Speed Tradeoff Summary

```
Quality Retention vs Inference Speedup:

100% |*
     | *  INT8 quant
 98% |  *   AWQ W4
     |   *    2:4 sparse
 96% |    *     W4 + 2:4
     |     *      + distillation
 94% |      *
     |       *   Aggressive pruning
 92% |        *
     |         *  Layer removal
 90% |          *
     |           * 2-bit quant
 85% |             *
     |_____|_____|_____|_____|_____|_____|____
     1x    2x    3x    4x    6x    8x    12x
                  Inference Speedup
```

### 12.7 Key Takeaways for Practitioners

1. **Start with quantization**: W4A16 (AWQ or GPTQ) gives the best quality/speed ratio for most models
2. **Add 2:4 sparsity** if targeting NVIDIA Ampere+ and need additional speedup beyond quantization
3. **Use distillation** when you can afford training time and need a smaller architecture
4. **Layer pruning** (ShortGPT/LaCo) is a quick win for 20-30% model size reduction
5. **Early exit** (LayerSkip) provides dynamic speedup that adapts to input difficulty
6. **Combined approaches** (W4 + 2:4 sparse via Marlin-24) provide the best throughput on supported hardware
7. **Always evaluate beyond perplexity**: pruning can cause catastrophic drops on specific tasks while perplexity remains stable
8. **Larger models compress better**: A 70B model at 4-bit often outperforms a 13B model at 16-bit

---

## References and Key Papers

- **SparseGPT**: Frantar & Alistarh, ICML 2023. One-shot pruning with Hessian approximation.
- **Wanda**: Sun et al., ICLR 2024. Weight * activation pruning without retraining.
- **Movement Pruning**: Sanh et al., NeurIPS 2020. Pruning by weight movement direction.
- **Lottery Ticket Hypothesis**: Frankle & Carlin, ICLR 2019. Sparse trainable subnetworks.
- **ShortGPT**: Men et al., 2024. Layer removal via Block Influence scores.
- **LaCo**: Yang et al., 2024. Layer collapse via similarity-based merging.
- **SliceGPT**: Ashkboos et al., ICLR 2024. Rotate-then-slice weight compression.
- **LayerSkip**: Elhoushi et al., ACL 2024. Self-speculative decoding with early exit.
- **CALM**: Schuster et al., 2022. Confident adaptive language modeling.
- **Mixture of Depths**: Raposo et al., DeepMind, 2024. Dynamic compute allocation.
- **Token Merging (ToMe)**: Bolya et al., ICLR 2023. Fast token reduction for ViTs.
- **Mixture-of-Recursions**: ACL/NeurIPS 2025. Adaptive recursion depth per token.
- **MARLIN**: Frantar et al., 2024. Mixed-precision auto-regressive inference kernel.
- **cuSPARSELt**: NVIDIA. Structured sparse GEMM library for Ampere+.
- **ASP**: NVIDIA. Automatic Sparsity for PyTorch.
- **ASVD**: Yuan et al., 2024. Activation-aware SVD for LLM compression.
- **SVD-LLM**: ICLR 2025. Truncation-aware SVD with Cholesky whitening.
- **DeltaLLM**: 2025. Low-rank deltas between shared weights.
- **Delta-CoMe**: NeurIPS 2024. Mixed-precision delta compression.
- **DistilBERT**: Sanh et al., 2019. 40% smaller, 60% faster, 97% accuracy.
- **ATOM**: MLSys 2024. Low-bit quantization for efficient LLM serving.
- **LLM Compressor**: Red Hat/Neural Magic. Production compression toolkit for vLLM.
- **OPTIMA**: 2025. Optimal one-shot pruning via quadratic programming.
- **Once-for-All**: Cai et al., 2020. Train once, deploy many sub-networks.
- **Subformer**: Reid et al., 2021. Sandwich-style weight sharing.
- **MASA**: 2025. Matrix atom sharing in attention (66.7% parameter reduction).
