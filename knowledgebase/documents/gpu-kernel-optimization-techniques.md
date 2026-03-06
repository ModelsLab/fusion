# GPU Kernel Optimization: Comprehensive Knowledge Base (2025-2026)

> A deep technical reference covering cutting-edge GPU kernel techniques, from hardware-specific DSLs to AI-driven kernel generation. This document reflects the state of the art as of early 2026.

---

## Table of Contents

1. [ThunderKittens (Stanford)](#1-thunderkittens-stanford)
2. [Liger Kernel](#2-liger-kernel)
3. [Unsloth](#3-unsloth)
4. [FlexAttention (PyTorch)](#4-flexattention-pytorch)
5. [CUDA Agent / AI-Driven Kernel Generation](#5-cuda-agent--ai-driven-kernel-generation)
6. [Speculative Decoding Kernels](#6-speculative-decoding-kernels)
7. [Disaggregated Serving](#7-disaggregated-serving)
8. [Continuous Batching Optimizations](#8-continuous-batching-optimizations)
9. [FP4 and Sub-4-bit Quantization](#9-fp4-and-sub-4-bit-quantization)
10. [Persistent Kernels](#10-persistent-kernels)
11. [Warp Specialization Patterns](#11-warp-specialization-patterns)
12. [NCCL and Collective Communication Optimization](#12-nccl-and-collective-communication-optimization)
13. [Kernel Compilation Optimization](#13-kernel-compilation-optimization)
14. [Hardware-Specific Tricks](#14-hardware-specific-tricks)
15. [SageAttention and Other Recent Attention Innovations](#15-sageattention-and-other-recent-attention-innovations)

---

## 1. ThunderKittens (Stanford)

### Overview

ThunderKittens (TK) is a C++ embedded Domain-Specific Language (DSL) from Stanford's Hazy Research group for writing high-performance GPU kernels. Published at ICLR 2025, it is now used in production at ML inference providers and high-frequency trading firms. Ports exist for AMD GPUs (HipKittens) and Apple Silicon (ThunderMittens via Metal Shading Language).

### Design Philosophy: Register-Tile First

The fundamental insight behind ThunderKittens is that modern GPU hardware processes data most naturally in **16x16 tiles**. Rather than treating registers as 32-bit words (a legacy CPU abstraction), TK redefines registers as 16x16 matrix tiles that directly map to how tensor cores consume data. The framework is deliberately "small and dumb and simple" -- if the hardware does not want to handle operations smaller than 16x16, neither does TK.

This register-tile-first approach means:
- The minimum matrix dimension is constrained to 16x16
- All abstractions revolve around tile-level operations
- Memory layout, swizzling, and synchronization are managed automatically
- Hardware features (TMA, WGMMA) are exposed through tile abstractions

### Key Abstractions

TK provides four core templated data types:

| Abstraction | Location | Parameters | Description |
|---|---|---|---|
| **Register Tiles** (`kittens::rt`) | Register file | height, width, type, layout | 2D tensors split across warp threads |
| **Register Vectors** (`kittens::rv`) | Register file | length, layout | 1D tensors for reductions/aggregations |
| **Shared Tiles** (`kittens::st`) | Shared memory | height, width, layout | 2D tensors for inter-warp communication |
| **Shared Vectors** (`kittens::sv`) | Shared memory | length, layout | 1D tensors designed to avoid bank conflicts |

All types are templated by data type (e.g., `bf16`), shape, and memory layout (row-major or column-major).

### Operations and Primitives

Operations are organized at the warp level and collaborative warp-group level:

- **Initializers**: Zero-fill tiles/vectors, load from global/shared memory
- **Unary operations**: Element-wise operations (exp, log, etc.)
- **Binary operations**: Element-wise multiply, add, etc.
- **Row/Column operations**: Row summation, column max, normalization
- **Matrix operations**: MMA (matrix multiply-accumulate) via `mma_ABt`, `mma_AB`
- **Memory operations**: TMA loads/stores, async copies

### How It Differs from Triton and CUDA

**ThunderKittens vs. Triton:**
- TK embeds directly within CUDA C++; Triton provides a separate Python-based language and compiler
- TK allows "graceful failure" by dropping to raw CUDA when abstractions are insufficient
- Triton auto-tunes tile sizes and schedules; TK gives explicit control over tile scheduling
- TK directly exposes hardware features like TMA and WGMMA; Triton abstracts these away behind its compiler

**ThunderKittens vs. Raw CUDA:**
- TK abstracts memory layout complexities, swizzling patterns, and barrier synchronization
- 60-100 lines of TK code replaces hundreds of lines of raw CUDA
- TK manages data types, layout conversions, and tile movement automatically
- Raw CUDA gives maximum flexibility but requires deep hardware expertise

### Performance Results

| Kernel | Hardware | Result |
|---|---|---|
| Flash Attention Forward | H100 | ~30% faster than FlashAttention-2 |
| Flash Attention Forward | RTX 4090 | 75% hardware utilization in ~60 lines |
| Based Linear Attention | H100 | 215 TFLOPS (>300 TFLOPS with recomputation) |

### Example Kernel Code

**RTX 4090 Flash Attention (abbreviated):**
```cpp
using namespace kittens;
rt_bf_1x4<> q_reg, k_reg, v_reg;       // Register tiles: bf16, 1x4 shape
load(q_reg, _q + offset, cols);         // Load Q from global memory
mul(q_reg, q_reg, scale_factor);        // Scale Q
zero(max_vec);                          // Initialize max tracking
mma_ABt(attention_block, q_reg, k_reg, attention_block);  // QK^T
```

**H100 Flash Attention with TMA (abbreviated):**
```cpp
tma::load_async((q_smem[wg]), tma_q, barrier, tile_idx);  // Async TMA load
tma::arrive_and_wait(barrier, phase_bit);                   // Wait for data
warpgroup::mm_ABt(result, q_smem, k_smem);                 // WGMMA compute
```

### Sources

- [ThunderKittens GitHub](https://github.com/HazyResearch/ThunderKittens)
- [GPUs Go Brrr - Hazy Research Blog](https://hazyresearch.stanford.edu/blog/2024-05-12-tk)
- [ThunderKittens: A Simple Embedded DSL - Hazy Research](https://hazyresearch.stanford.edu/blog/2024-05-12-quick-tk)
- [ThunderKittens Paper (arXiv)](https://arxiv.org/html/2410.20399v1)
- [ICLR 2025 Paper](https://openreview.net/pdf/f4b2b2d3f597357551880dae1c1a4286791aadc5.pdf)
- [HipKittens for AMD](https://hazyresearch.stanford.edu/static/posts/2025-11-09-hk/hipkittens.pdf)
- [ThunderMittens for Apple Silicon](https://hazyresearch.stanford.edu/blog/2024-11-28-tk-mlx)

---

## 2. Liger Kernel

### Overview

Liger Kernel is LinkedIn's collection of Triton kernels designed specifically for LLM training. It achieves a **20% increase in multi-GPU training throughput** and **60% reduction in memory usage**, with up to **80% memory savings** for post-training alignment tasks. The project supports 20+ model architectures and integrates directly with HuggingFace Transformers.

### Fused Kernels

#### Training Kernels

| Kernel | Description | Key Benefit |
|---|---|---|
| **FusedLinearCrossEntropy** | Fuses linear projection with CrossEntropy loss | Avoids materializing full logit tensor; ~3x faster, ~5x less memory for vocab size 163840 |
| **CrossEntropy + Softmax** | Block-wise loss computation | Computes loss without full softmax materialization |
| **RMSNorm** | Root Mean Square Layer Normalization | Fused forward + backward in single kernel |
| **LayerNorm** | Standard Layer Normalization | Eliminates intermediate activations |
| **SwiGLU** | Swish-gated Linear Unit | Fuses gate, up-projection, and activation |
| **GeGLU** | GELU-gated Linear Unit | Same fusion strategy as SwiGLU |
| **RoPE** | Rotary Position Embeddings | Efficient in-place positional encoding |
| **Softmax / Sparsemax** | Attention normalization | Memory-efficient implementations |
| **Multi Token Attention** | Multi-token prediction attention | Optimized for multi-token objectives |
| **mHC (Hyper-Connections)** | Hyper-connection layers | Fused multi-head computation |

#### Post-Training / Alignment Kernels (up to 80% memory savings)

| Kernel | Use Case |
|---|---|
| **Fused Linear CPO** | Conservative Policy Optimization |
| **Fused Linear DPO** | Direct Preference Optimization |
| **Fused Linear ORPO** | Odds Ratio Preference Optimization |
| **Fused Linear SimPO** | Simple Preference Optimization |
| **Fused Linear KTO** | Kahneman-Tversky Optimization |

#### Distillation Kernels

| Kernel | Description |
|---|---|
| **KL Divergence** | Standard KL divergence loss |
| **Jensen-Shannon Divergence (JSD)** | Symmetric divergence measure |
| **Fused Linear JSD** | Fused linear + JSD computation |
| **Total Variation Distance (TVD)** | Distribution distance metric |

#### Experimental Kernels
- **Embedding optimization**: Efficient embedding lookups
- **Matmul int2xint8**: Ultra-low-precision matrix multiplication

### Memory Savings from Kernel Fusion

Liger employs three core optimization strategies:

1. **Kernel Fusion**: Multiple operations (e.g., linear projection + softmax + cross-entropy) are combined into a single GPU kernel launch, eliminating intermediate memory allocations and reducing kernel launch overhead.

2. **In-Place Replacement**: Operations modify tensors in-place rather than allocating new buffers, directly reducing peak memory.

3. **Chunking**: Large operations are processed in sequential chunks, keeping only a small working set in memory at any time. The FusedLinearCrossEntropy kernel, for example, never materializes the full logit tensor (which can be enormous for large vocabularies).

### Integration with HuggingFace Transformers

**Single-line integration:**
```python
from liger_kernel.transformers import AutoLigerKernelForCausalLM
model = AutoLigerKernelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")
```

**Model-specific patching:**
```python
from liger_kernel.transformers import apply_liger_kernel_to_llama
apply_liger_kernel_to_llama()  # Patches all Llama model instances
```

**Supported frameworks:** HuggingFace Transformers, TRL, Lightning, Axolotl, Llama-Factory, SWIFT, oumi

**Supported models:** Llama (2/3/3.2/4), Mistral, Mixtral, Pixtral, Gemma (1/2/3), Qwen (2/2.5/3/MoE variants), Phi3, Granite, OLMo, GLM-4, GPT-OSS, InternVL3, Hunyuan, and more.

### Performance Numbers

| Metric | Improvement |
|---|---|
| Multi-GPU training throughput | +20% |
| Training memory usage | -60% |
| Post-training memory (DPO, ORPO, etc.) | -80% |
| CrossEntropy execution (vocab=163840) | ~3x faster |
| CrossEntropy memory (vocab=163840) | ~5x less |

**Compatibility:** Works out of the box with Flash Attention, PyTorch FSDP, and Microsoft DeepSpeed. Requires Torch >= 2.1.2 and Triton >= 2.3.0 (CUDA) or Torch >= 2.5.0 and Triton >= 3.0.0 (ROCm).

### Sources

- [Liger Kernel GitHub](https://github.com/linkedin/Liger-Kernel)
- [Liger Kernel Paper (arXiv)](https://arxiv.org/abs/2410.10989)
- [Liger Kernel Docs](https://linkedin.github.io/Liger-Kernel/)
- [Fused Triton Kernels in LLM Optimization](https://www.emergentmind.com/topics/fused-triton-kernels)

---

## 3. Unsloth

### Overview

Unsloth is an open-source library that accelerates LLM fine-tuning by **2-3x** (up to 5x with latest kernels) while reducing VRAM usage by **30-90%**, with no accuracy loss. It achieves this through custom Triton kernels, manual backpropagation derivations, and intelligent sequence packing.

### How It Achieves 2x+ Speedup

Unsloth's speedup comes from three interrelated optimizations:

1. **Custom Triton Kernels**: All critical operations (RoPE, MLP/SwiGLU, cross-entropy, etc.) are rewritten as hand-optimized Triton kernels. By manually deriving backpropagation steps and implementing them in Triton, Unsloth avoids PyTorch's autograd overhead and memory allocations.

2. **Smart Auto Packing**: Unsloth implements "uncontaminated sequence packing" -- packing multiple sequences into a single batch without cross-contamination of attention between sequences. This eliminates padding waste while maintaining training correctness.

3. **Memory-Efficient Computation**: Operations are designed to minimize intermediate tensor allocations, with in-place operations where possible.

### Custom Triton Kernels

| Kernel | Speedup | Notes |
|---|---|---|
| **RoPE Kernel** | 2.3x (long ctx), 1.9x (short ctx) | Custom rotary position embedding with int64 indexing for long contexts |
| **MLP / SwiGLU Kernel** | Significant | Fuses gate projection, up projection, and activation function |
| **GeGLU Kernel** | Significant | GELU-gated variant, also with int64 indexing |
| **Cross-Entropy Kernel** | Significant | Chunk-based computation avoiding full logit materialization |
| **MoE Kernels** | ~12x faster training | Mixture-of-Experts specialized kernels with >35% less VRAM |

### Memory Optimization Techniques

- **Manual backprop derivation**: Instead of relying on PyTorch's autograd to store activations, Unsloth manually derives the backward pass, choosing what to recompute vs. store, dramatically reducing activation memory.
- **QLoRA integration**: Quantized base model (4-bit) with LoRA adapters in higher precision. Unsloth optimizes the QLoRA data path to avoid unnecessary dequantization steps.
- **In-place operations**: Tensor operations modify data in-place where mathematically valid.
- **Gradient checkpointing**: Strategic checkpointing combined with custom kernels minimizes peak memory.

### QLoRA Optimizations

When applied with QLoRA on all linear layers:
- **2.7x faster** training
- **74% less memory** usage
- Supports 4-bit NormalFloat (NF4) quantization with double quantization
- Optimized dequantization kernels that avoid materializing full-precision weights
- Custom backward kernels that operate directly on quantized representations

### MoE Model Support

Unsloth's newest MoE Triton kernels provide:
- ~12x faster Mixture of Experts LLM training
- >35% less VRAM
- ~6x longer context lengths
- Optimized expert routing and gate computation

### Sources

- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [3x Faster Training with Unsloth Kernels + Packing](https://docs.unsloth.ai/new/3x-faster-training-packing)
- [Unsloth + HuggingFace TRL](https://huggingface.co/blog/unsloth-trl)
- [Fine-tune MoE Models 12x Faster](https://unsloth.ai/docs/new/faster-moe)
- [Unsloth's Triton Kernels Analysis](https://www.banandre.com/blog/unsloth-triton-kernels-end-vram-arms-race)

---

## 4. FlexAttention (PyTorch)

### Overview

FlexAttention, introduced in PyTorch 2.5.0 as `torch.nn.attention.flex_attention`, allows ML researchers to implement custom attention patterns (causal, sliding window, document masking, ALiBi, etc.) without writing GPU kernel code. The user defines a `score_mod` or `mask_mod` function in Python, and PyTorch compiles it into an efficient fused attention kernel. Accepted at MLSys 2025.

### Block Mask API

The `BlockMask` is a data structure representing block-sparse attention patterns:

```python
from torch.nn.attention.flex_attention import create_block_mask

def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

block_mask = create_block_mask(causal_mask, B=1, H=1, Q_LEN=8192, KV_LEN=8192)
```

**How BlockMask works:**
- Divides the attention matrix into blocks of `KV_BLOCK_SIZE x Q_BLOCK_SIZE`
- A block is marked sparse only if **every element** within it is masked
- `torch.vmap` automatically generates the BlockMask from user-defined `mask_mod`
- Pre-computation amortizes the cost of determining sparsity patterns
- Sparse blocks are entirely skipped during attention computation

### score_mod for Custom Attention Patterns

`score_mod` is a function that transforms attention scores before softmax:

```python
def alibi_score_mod(score, b, h, q_idx, kv_idx):
    bias = (q_idx - kv_idx) * alibi_slope[h]
    return score + bias

output = flex_attention(query, key, value, score_mod=alibi_score_mod)
```

Supported patterns include:
- **Causal masking**: Standard autoregressive mask
- **Sliding window**: Local attention with fixed window size
- **Document masking**: Separate documents within a batch
- **ALiBi**: Attention with Linear Biases
- **Soft-capping**: Score clamping (as in Gemma-2)
- **Prefix LM**: Bidirectional prefix + causal suffix
- **Trainable biases**: Learned relative position biases

### Compilation to Efficient Fused Kernels

During `torch.compile`, the `score_mod` and `mask_mod` functions are:
1. **Lowered** through PyTorch's FX graph representation
2. **Code-generated** into the inner loop of a hand-written attention kernel
3. **Optimized** by TorchInductor, which sweeps kernel parameters (tile size, num_stages, etc.)
4. **Compiled** to a fused kernel that avoids materializing the full attention matrix

With `max-autotune` mode, TorchInductor benchmarks multiple configurations and selects the best:
```python
flex_attention = torch.compile(flex_attention, mode="max-autotune")
```

### Performance Comparison

| Pattern | vs. FlashAttention-2 | vs. Manual Implementation |
|---|---|---|
| Causal attention | ~90% of FA2 performance | Matches hand-written |
| Document masking | Faster (exploits sparsity) | Significant speedup |
| Sliding window | Faster (block-sparse skip) | Significant speedup |
| ALiBi | Comparable | Much simpler to implement |

FlexAttention is not always as fast as a fully hand-tuned kernel for simple patterns (like basic causal attention where FlashAttention-2 excels), but it dramatically outperforms naive implementations and provides near-optimal performance for complex, custom attention patterns.

### Recent Developments (2025-2026)

- **FlexAttention for Inference**: Extended support beyond training to efficient inference
- **Performance tuning guides**: Official documentation for max-autotune optimization
- **Trainable biases**: Support for learned attention biases during both training and inference
- **CUDAGraph compatibility**: Works with CUDA graph capture for reduced launch overhead

### Sources

- [FlexAttention PyTorch Documentation](https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html)
- [FlexAttention for Inference Blog](https://pytorch.org/blog/flexattention-for-inference/)
- [FlexAttention Paper (arXiv)](https://arxiv.org/pdf/2412.05496)
- [FlexAttention: Scalable, Flexible Attention](https://www.emergentmind.com/topics/flexattention)

---

## 5. CUDA Agent / AI-Driven Kernel Generation

### Overview

AI-driven kernel generation has emerged as a major research direction in 2025-2026, with systems using LLMs to automatically write, optimize, and verify CUDA and Triton kernels. The field spans reinforcement learning agents, multi-agent systems, evolutionary approaches, and world-model-guided search.

### CUDA Agent

**Architecture:** CUDA Agent is a large-scale agentic reinforcement learning system with three core components:

1. **Data Synthesis Pipeline (CUDA-Agent-Ops-6K)**:
   - Seed collection from `torch` and `transformers` operators
   - Combinatorial synthesis: up to 5 operators fused sequentially via LLM generation
   - Execution-driven filtering: removes stochastic operators, validates correctness, controls workload timing (1ms-100ms), eliminates high-similarity cases

2. **Skill-Augmented Execution Environment**:
   - ReAct-style iterative coding/compilation/debugging in GPU sandbox
   - Profiler-guided optimization workflow (Nsight Compute integration)
   - Milestone-based discrete rewards
   - Anti-reward-hacking: protected verification scripts, forbidden fallback calls, five-input correctness validation, synchronized warm-up profiling
   - Target: pass correctness checks and exceed 5% speedup over `torch.compile`

3. **Reinforcement Learning Training** (three-stage):
   - Stage 1: Single-turn PPO warm-up for base CUDA generation
   - Stage 2: Actor initialization via Rejection Fine-Tuning on successful trajectories
   - Stage 3: Critic initialization with value pretraining for stable advantage estimation
   - Supports 128k token contexts, 150 training turns, 200 evaluation turns

**Benchmark Results:**

| Metric | CUDA Agent | Claude Opus 4.5 | Gemini 3 Pro |
|---|---|---|---|
| Pass rate | 98.8% | ~85% | ~87% |
| Faster than torch.compile | 96.8% | 66.4% | 69.6% |
| Geomean speedup vs. compile | 2.11x | 1.42x | 1.46x |
| Speedup vs. eager | 2.60x | -- | -- |
| Level-3 pass rate | 94% | -- | -- |
| Level-3 faster rate | 90% | -- | -- |

### K-Search: Co-Evolving World Model

K-Search formulates kernel generation as a **planning problem over a structured search tree**, where the LLM itself serves as a co-evolving world model:

- The LLM maintains transition dynamics that are continuously refined by assimilating execution feedback via in-context learning
- Decouples high-level strategy from low-level code implementation
- Allows multi-step transformations even when intermediate implementations do not improve performance
- Dynamically updates prior beliefs and calibrates search strategy

**Results:**
- Average 2.10x improvement over evolutionary search methods
- Up to 14.3x gain on complex MoE kernels
- State-of-the-art on GPUMode TriMul task (1030us on H100)
- Evaluated on H100 and B200 GPUs (CUDA 12.8, FlashInfer 0.5.3, PyTorch 2.8.0)

### CudaForge

CudaForge uses a training-free, dual-agent architecture:
- **Coder Agent**: Generates and refines CUDA kernel code
- **Judge Agent**: Evaluates correctness and performance using hardware feedback
- Integrates Nsight Compute (NCU) metrics for profiling-guided optimization
- Iterative loop inspired by human expert workflows

### Other Notable Systems

| System | Approach | Key Innovation |
|---|---|---|
| **Sakana AI CUDA Engineer** | Vector database + LLM | Grounds generation in high-quality kernel examples |
| **STARK** | Multi-agent (Plan-Code-Debug) | Emulates human workflows with phased development |
| **PEAK** | Stepwise iterative refinement | Minimal executable programs reduce compilation overhead |
| **EvoEngineer / FM Agent** | Population-based evolution | Mutation/crossover with adaptive dynamics |
| **GPU Kernel Scientist** | Iterative optimization framework | Systematic profiling and bottleneck analysis |
| **KernelEvolve** | Hardware-specific knowledge base | Tailored knowledge for heterogeneous accelerators |
| **Astra** | Production-grade SGLang tuning | Focuses on real-world deployment optimization |

### Benchmarks for AI Kernel Generation

| Benchmark | Tasks | Focus |
|---|---|---|
| **KernelBench** (Stanford) | 250 | PyTorch-to-CUDA translation |
| **TritonBench** | 350+ | Triton operator generation |
| **MultiKernel-Bench** | 285 | Cross-platform synthesis |
| **BackendBench** | 271 | PyTorch core library compliance |

### Sources

- [CUDA Agent Project Page](https://cuda-agent.github.io/)
- [K-Search Paper (arXiv)](https://arxiv.org/abs/2602.19128)
- [CudaForge Paper (arXiv)](https://arxiv.org/abs/2511.01884)
- [Towards Automated Kernel Generation Survey (arXiv)](https://arxiv.org/pdf/2601.15727)
- [Sakana AI CUDA Engineer](https://sakana.ai/ai-cuda-engineer/)
- [Awesome LLM-Driven Kernel Generation](https://github.com/flagos-ai/awesome-LLM-driven-kernel-generation)
- [KernelBench (Stanford)](https://scalingintelligence.stanford.edu/blogs/kernelbench/)

---

## 6. Speculative Decoding Kernels

### Overview

Speculative decoding accelerates LLM inference by using a fast "draft" model to propose multiple tokens, then verifying them in parallel with the full "target" model. The kernel implications are significant: draft execution must be extremely fast, verification must handle tree-structured inputs efficiently, and acceptance/rejection must be implemented with minimal overhead.

### Draft Model Execution Optimization

Draft models must execute with minimal latency since they are on the critical path:

- **Lightweight architectures**: EAGLE uses a small auto-regression head attached to the target model's second-to-top layer features, avoiding a separate model entirely
- **Shared KV cache**: Draft models share the target model's KV cache where possible, reducing memory traffic
- **Quantized draft models**: INT4/INT8 draft models fit in L2 cache for faster execution
- **Batched draft generation**: Multiple draft steps are combined into a single kernel launch

### Verification Kernel Design

The verification phase processes the full draft tree with the target model:

- **Tree Attention Mask**: A specialized attention mask encodes the causal structure of the speculative tree. Each token in the tree can only attend to its ancestors, creating a block-sparse pattern.
- **Parallel Verification**: All draft tokens in the tree are verified in a single forward pass of the target model, amortizing the cost across many candidates.
- **Custom attention kernels**: Tree attention requires non-standard masking patterns. FlexAttention's `mask_mod` can encode tree structure, or custom CUDA kernels handle the tree topology directly.

### Tree-Based Speculative Decoding

#### Medusa
- Augments the base model with **multiple extra language model heads** (Medusa heads)
- Each head predicts a different future token position
- TopK sampling from each head creates a tree of candidates
- Training: Only Medusa heads are trained; base model is frozen
- Tree construction creates multiple potential paths for parallel verification

#### EAGLE (1, 2, 3)
- **EAGLE-1** (ICML 2024): Predicts next feature vector from the target model's second-to-top layer using a lightweight auto-regression head. Achieves 2-3x speedup over vanilla decoding.
- **EAGLE-2** (EMNLP 2024): Uses draft model confidence scores to approximate acceptance rates, dynamically adjusting draft tree structure per input. Context-dependent trees increase accepted tokens.
- **EAGLE-3** (NeurIPS 2025): Removes feature prediction constraint, fuses low/mid/high-level semantic features. Introduces **training-time testing** -- simulates the inference process during training so the draft head learns to recover from its own errors. Discovers a scaling law: more training data leads to proportional speedup improvements.

#### OPT-Tree
- Constructs the **optimal draft tree structure** for any input sequence
- Analytical algorithm rather than heuristic tree shapes
- Adapts tree topology to maximize expected accepted tokens

### Self-Speculative Decoding

#### LayerSkip
- Uses the target model itself as both draft and verifier
- Draft phase: Skip intermediate layers (early exit) for faster, less accurate predictions
- Verification phase: Run full model to verify
- No separate draft model needed -- zero additional memory

#### Draft Heads
- Small prediction heads attached at intermediate layers
- Multiple draft tokens predicted from partial model computation
- Verification uses full model forward pass

#### SWIFT
- On-the-fly self-speculative decoding
- Dynamically selects which layers to skip based on input difficulty
- No pre-trained draft model or additional parameters

### Kernel Requirements for Acceptance/Rejection

The acceptance/rejection step has specific kernel requirements:

1. **Token-level comparison**: Each draft token's probability under the target model is compared against the draft model's probability
2. **Stochastic acceptance**: Token is accepted with probability min(1, p_target/p_draft) -- requires random number generation and comparison in a single kernel
3. **Tree traversal**: When a token is rejected, all its descendants in the tree are also rejected. The longest accepted prefix is identified.
4. **Resampling**: A correction token is sampled from an adjusted distribution (p_target - p_draft) at the first rejection point
5. **KV cache management**: Accepted tokens' KV entries are retained; rejected tokens' entries are discarded. Efficient cache compaction is needed.

These operations are typically fused into a single kernel to avoid multiple global memory round-trips.

### Sources

- [EAGLE GitHub (EAGLE-1/2/3)](https://github.com/SafeAILab/EAGLE)
- [EAGLE-3 Paper (NeurIPS 2025)](https://openreview.net/pdf?id=4exx1hUffq)
- [NVIDIA Speculative Decoding Blog](https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/)
- [vLLM Speculative Decoding Blog](https://blog.vllm.ai/2024/10/17/spec-decode.html)
- [PyTorch Hitchhiker's Guide to Speculative Decoding](https://pytorch.org/blog/hitchhikers-guide-speculative-decoding/)
- [OPT-Tree Paper](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00735/128189)
- [TensorRT-LLM Speculative Decoding](https://nvidia.github.io/TensorRT-LLM/advanced/speculative-decoding.html)

---

## 7. Disaggregated Serving

### Overview

Disaggregated serving separates LLM inference into distinct **prefill** (prompt processing) and **decode** (token generation) phases, running them on separate GPU pools. By 2025, this architecture became the default approach across nearly every major LLM serving stack: NVIDIA Dynamo, llm-d, Ray Serve LLM, SGLang, vLLM, LMCache, and MoonCake.

### Splitwise: Separate Prefill and Decode GPUs

Splitwise (concurrent with DistServe) focuses on:
- **Heterogeneous hardware**: Uses different GPU types for prefill vs. decode (e.g., H100 for compute-intensive prefill, A100 for memory-bandwidth-bound decode)
- **Energy efficiency**: Matching hardware capabilities to workload characteristics reduces total energy consumption
- **Independent scaling**: Prefill and decode pools scale independently based on traffic patterns

### DistServe Architecture

DistServe introduced the foundational disaggregated architecture:

**Core design:**
- Physically separate prefill and decode workers on different GPU nodes
- Prefill workers process input prompts with high compute utilization
- Decode workers handle autoregressive token generation with high memory bandwidth utilization
- KV cache transfer between prefill and decode workers

**Performance:**
- 7.4x more requests served within latency constraints
- 12.6x tighter SLO compliance compared to state-of-the-art co-located systems
- > 90% of requests within latency targets

**Key innovation:** Goodput optimization -- maximizing the number of requests that meet Service Level Objectives (SLOs) rather than just raw throughput.

### Mooncake (Moonshot AI)

Mooncake is the serving platform for Moonshot AI's Kimi chatbot:

**KVCache-centric architecture:**
- Separates prefill and decoding clusters
- Efficiently utilizes **CPU, DRAM, SSD, and NIC resources** for KV cache management
- KV cache is treated as a first-class citizen with its own memory hierarchy
- Implements KV cache prefetching and caching policies

### Kernel Implications of Prefill-Decode Separation

The disaggregated architecture introduces unique kernel requirements:

**Prefill kernels (compute-bound):**
- Optimized for high arithmetic intensity
- Large batch sizes, long sequences
- Flash Attention variants with maximum compute utilization
- Can use more aggressive tiling and pipeline depth

**Decode kernels (memory-bound):**
- Optimized for memory bandwidth
- Small batch sizes (often batch=1 per request), single token generation
- Paged attention for efficient KV cache access
- MQA/GQA optimizations critical for bandwidth reduction

**KV cache transfer kernels:**
- The primary new bottleneck: moving KV tensors between prefill and decode workers
- Hundreds of megabytes per request must be transferred
- Overlapping transfer with computation is critical
- Compression of KV cache during transfer (e.g., FP8 quantization)
- RDMA-based direct GPU-to-GPU transfer where available

**Recent developments (2025-2026):**
- **TraCT**: Uses CXL shared memory for KV cache at rack scale, avoiding explicit transfers
- **DuetServe**: Harmonizes prefill and decode on the same hardware with intelligent scheduling
- **SPAD**: Proposes specialized hardware designs for disaggregated inference

### Sources

- [DistServe Retrospective - 18 Months Later](https://haoailab.com/blogs/distserve-retro/)
- [DistServe Paper (OSDI 2024)](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf)
- [Mooncake Paper (ACM)](https://dl.acm.org/doi/10.1145/3773772)
- [Disaggregated Prefill-Decode Explained](https://docs.jarvislabs.ai/blog/llm-optimization-disaggregated-prefill-decode)
- [TraCT: CXL Shared Memory KV Cache](https://arxiv.org/html/2512.18194v1)
- [DuetServe Paper](https://arxiv.org/pdf/2511.04791)

---

## 8. Continuous Batching Optimizations

### Overview

Continuous batching (also called iteration-level batching) allows new requests to be inserted into a running batch at each decode step, rather than waiting for all requests in a batch to complete. The kernel implications are significant: variable-length sequences must be handled efficiently without excessive padding waste.

### Variable Sequence Lengths in Kernels

The core challenge: different requests in a batch have different sequence lengths. Three strategies exist:

**1. Padding:**
- Pad all sequences to the maximum length in the batch
- Simple but wasteful: average utilization can drop below 50%
- Wasted computation on padding tokens
- Standard approach in early attention implementations

**2. Packing:**
- Concatenate multiple sequences end-to-end into a single "packed" tensor
- Use segment IDs or offset arrays to track sequence boundaries
- Eliminates padding waste entirely
- Requires attention masks that prevent cross-sequence attention
- Unsloth's "uncontaminated sequence packing" ensures no attention leakage

**3. Ragged/Nested Tensors:**
- Native support for variable-length dimensions
- Each sequence retains its actual length without padding
- Requires kernel support for irregular memory access patterns

### Ragged Tensors / Nested Tensors

**PyTorch Nested Tensors:**
- `torch.nested.nested_tensor()` provides a native representation
- Under active development for attention kernel integration
- Enables `torch.compile` optimizations for variable-length batches

**FlashInfer's Ragged Tensor Support:**
- Maintains three data structures for ragged KV caches:
  - `kv_page_indices`: Maps logical pages to physical pages
  - `kv_indptr`: Indirection pointer array (CSR-style format)
  - `kv_last_page_lens`: Tracks partial fill of last page per sequence
- Reinterprets attention as a **block-sparse matrix multiplication**
- Achieves 90% of dense attention throughput with vector-sparse attention (page_size=1)

### Padding vs. Packing Strategies

| Strategy | Compute Efficiency | Memory Efficiency | Kernel Complexity | Use Case |
|---|---|---|---|---|
| **Padding** | Low (wasted ops) | Low (wasted memory) | Simple | Prototyping, fixed-length |
| **Packing** | High | High | Medium | Training with variable lengths |
| **Ragged/Paged** | High | High | High | Inference with continuous batching |
| **Bucketing** | Medium | Medium | Low | Compromise approach |

**Bucketing** groups sequences of similar lengths together, reducing but not eliminating padding. Typical bucket boundaries: 128, 256, 512, 1024, 2048, 4096.

### FlashInfer's Comprehensive Support

FlashInfer provides the most complete kernel library for continuous batching:

**Paged KV Cache:**
- Fixed-size memory pages (typically 16 tokens)
- Pages allocated/deallocated as sequences grow/shrink
- Block table maps logical pages to physical GPU memory
- Efficient for continuous batching where sequence lengths change each step

**Per-Request Tiling:**
- Different sequences can have different tile counts
- Triton grid scheduler places query position loops in grid dimensions
- Nearly constant selected block counts across positions simplify optimization

**JIT Compilation:**
- Custom attention variants defined via `LogitsTransform` and `QueryTransform` functors
- PyTorch JIT loader compiles and caches kernels
- Initial compilation ~15 seconds on server CPUs; subsequent launches use cache

**CUDAGraph Support:**
- Variable-length inputs supported with CUDAGraph for speculative decoding
- Reduces kernel launch overhead for latency-sensitive decode steps

**Recent features:**
- FlashAttention-3 template integration for Hopper GPUs
- Fused MLA (Multi-head Latent Attention) decoding kernel for DeepSeek
- Block-sparse attention with any block size configuration
- Sorting-free GPU kernels for LLM sampling (March 2025)
- FlashInfer-Bench: curated benchmark dataset from production traffic

### Sources

- [FlashInfer GitHub](https://github.com/flashinfer-ai/flashinfer)
- [FlashInfer v0.2 Release](https://flashinfer.ai/2024/12/16/flashinfer-v02-release.html)
- [Dissecting FlashInfer - Systems Perspective](https://ydnyshhh.github.io/posts/flash_infer/)
- [FlashInfer Paper (arXiv)](https://arxiv.org/pdf/2501.01005)
- [FlashInfer-Bench](https://flashinfer.ai/2025/10/21/flashinfer-bench.html)
- [NVIDIA Ragged Batching](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/ragged_batching.html)

---

## 9. FP4 and Sub-4-bit Quantization

### Overview

Sub-4-bit quantization represents the frontier of model compression, enabled by new hardware support (Blackwell FP4 tensor cores) and algorithmic innovations (vector quantization, ternary networks). These techniques can halve memory footprint and double throughput compared to FP8.

### Blackwell FP4 Tensor Core Support

**NVFP4 Format Specification:**

| Component | Details |
|---|---|
| **Sign bit** | 1 bit |
| **Exponent bits** | 2 (E2) |
| **Mantissa bits** | 1 (M1) |
| **Value range** | ~{0, 0.5, 1.0, 1.5, 2, 3, 4, 6} and negatives |
| **Total bits per value** | ~4.5 (4-bit data + amortized scaling overhead) |

**Two-Level Scaling Architecture:**

1. **Micro-block level (FP8 E4M3)**: One FP8 scale per 16 values. This fine-grained scaling (vs. MXFP4's 32-element blocks) provides better dynamic range adaptation.

2. **Tensor level (FP32)**: One FP32 scalar per tensor adjusts the global distribution so micro-block scales can encode efficiently.

**Hardware Implementation:**
- Blackwell's 5th-generation Tensor Cores natively process FP4, handling grouping, dynamic scaling, and 4-bit matrix operations automatically
- FP4 doubles computational throughput over FP8
- Also supports FP6 for intermediate precision needs
- Both consumer (RTX 5090) and datacenter (B200) Blackwell GPUs support FP4

**Performance:**

| Metric | Value |
|---|---|
| Memory reduction vs. FP16 | 3.5x |
| Memory reduction vs. FP8 | 1.8x |
| Energy efficiency vs. H100 | Up to 25x per token |
| Accuracy loss (DeepSeek-R1-0528) | < 1% vs. FP8 |
| AIME 2024 benchmark | 2% improvement over FP8 |

### FP4 Kernel Patterns

FP4 kernels require specific design patterns:

1. **Micro-tensor scaling**: Each 16-element group has its own FP8 scale factor. Kernels must:
   - Load the FP4 data and its associated FP8 scales
   - Dequantize to higher precision (FP16/BF16) for accumulation
   - Accumulate in FP32 for numerical stability
   - Optionally re-quantize outputs

2. **Blackwell Transformer Engine integration**: The second-generation Transformer Engine handles micro-tensor scaling automatically, managing dynamic range at the sub-tensor level.

3. **Calibration**: Post-training quantization (PTQ) with tools like TensorRT Model Optimizer and LLM Compressor. Deployable via TensorRT-LLM and vLLM.

### 1-bit and 2-bit Quantization Approaches

#### BitNet b1.58
- **Ternary weight network**: Weights are constrained to {-1, 0, 1}
- Replaces floating-point matrix multiplication with **ternary accumulation**: additions and subtractions only
- Theoretically eliminates the need for floating-point multiply units
- Claims comparable performance to full-precision models at the same parameter count
- Implications: fundamentally different kernel design where MAC units are replaced by add/subtract/zero operations
- CPU-friendly: ternary operations map well to SIMD integer instructions

#### QuIP / QuIP#
- **Vector quantization** approach using lattice codebooks
- QuIP# achieves sub-2-bit effective storage per parameter
- Uses incoherence processing (random rotations) to spread information across dimensions
- Lookup-table-based dequantization in kernels
- Competitive accuracy at 2-bit with models quantized at higher precision

#### AQLM (Additive Quantization for Language Models)
- Multi-codebook additive quantization
- Achieves effective 2-bit precision with minimal quality loss
- Kernels use codebook lookups and accumulation
- Particularly effective for consumer GPU deployment

#### SpinQuant / QuaRot
- **Rotation-based PTQ methods**: Apply learned or random rotations to weight matrices before quantization
- Rotations spread outlier magnitudes across dimensions, making quantization more uniform
- Kernels must incorporate the rotation as part of the forward pass (typically fused into adjacent linear layers)

### Ternary Weight Networks

The BitNet paradigm envisions a "post-multiplication era" where:
- Weight storage: ~1.58 bits per parameter
- Compute: Addition/subtraction replaces multiply-accumulate
- Energy: Orders of magnitude reduction per operation
- Hardware: Could be implemented with much simpler circuits than FP tensor cores

**Kernel implications:**
- Standard GEMM is replaced by accumulation with sign selection
- Memory bandwidth becomes even more dominant (weights are tiny but activations are still FP16/FP8)
- Existing tensor core hardware cannot natively accelerate ternary operations
- Custom hardware (e.g., Microsoft's BitNet FPGA/ASIC work) would be needed for full benefit

### Sources

- [NVIDIA NVFP4 Blog](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [Blackwell NVFP4 Impact](https://www.edge-ai-vision.com/2025/10/nvidia-blackwell-the-impact-of-nvfp4-for-llm-inference/)
- [FP4 Tensor Cores Overview](https://www.emergentmind.com/topics/fp4-tensor-cores)
- [INT4 vs FP4 Comparison (HuggingFace)](https://huggingface.co/blog/onekq/nvfp4-int4)
- [FP4 Training Paper (arXiv)](https://arxiv.org/html/2501.17116v2)
- [Quantization Horizon Survey](https://uplatz.com/blog/the-quantization-horizon-navigating-the-transition-to-int4-fp4-and-sub-2-bit-architectures-in-large-language-models/)
- [NVIDIA TensorRT FP4 Image Generation](https://developer.nvidia.com/blog/nvidia-tensorrt-unlocks-fp4-image-generation-for-nvidia-blackwell-geforce-rtx-50-series-gpus)

---

## 10. Persistent Kernels

### Overview

Persistent kernels are GPU kernels that remain resident on streaming multiprocessors (SMs) for their entire lifetime, processing multiple work units rather than exiting after a single tile. Instead of launching a grid sized to the problem dimensions, persistent kernels launch a fixed number of CTAs (typically equal to the number of SMs) and use software-based tile scheduling to distribute work.

### What They Are

**Traditional kernel:**
- Grid size = number of output tiles
- Each CTA computes one tile, then exits
- New CTA launched for each tile
- Kernel launch overhead per wave

**Persistent kernel:**
- Grid size = number of SMs (fixed)
- Each CTA persists on its SM, processing multiple tiles sequentially
- CTA fetches next tile via software scheduler after completing current tile
- Single kernel launch for entire computation

### The Wave Quantization Problem

Wave quantization is the key motivation for persistent kernels. It occurs when work units do not evenly divide across available SMs:

**Example:** 10 work tiles on 4 SMs:
- Wave 1: 4 tiles (100% utilization)
- Wave 2: 4 tiles (100% utilization)
- Wave 3: 2 tiles (50% utilization -- wasted!)

On modern GPUs with 100+ SMs, this compounds dramatically:
- A problem with 115 tiles performs identically to one with 228 tiles on a 114-SM GPU
- Sharp performance drops occur when crossing wave boundaries
- Small problems relative to SM count suffer most severely

### Tile Scheduling Within Persistent Kernels

Tile schedulers encapsulate work distribution logic with three responsibilities:

1. **Grid size determination**: Fixed grid for persistent, problem-dependent for traditional
2. **Work iteration**: `get_initial_tile()`, `is_valid()`, `get_next_tile()` API
3. **Coordinate mapping**: Linear index to (M, N, K) block coordinates

**Round-robin scheduling:**
```
CTA[i] starts at tile[i], advances by gridDim.x steps
CTA 0: tiles 0, 4, 8, 12, ...
CTA 1: tiles 1, 5, 9, 13, ...
CTA 2: tiles 2, 6, 10, 14, ...
CTA 3: tiles 3, 7, 11, 15, ...
```

**Blackwell Cluster Launch Control (CLC):**
- Hardware-assisted dynamic tile scheduling for persistent GEMMs
- Eliminates software scheduling overhead on Blackwell architecture
- Enables Stream-K scheduling at hardware level

### Stream-K: Fractional Tile Distribution

Stream-K revolutionizes work distribution by assigning **fractional work tiles** to each SM:

**Concept:** With 9 tiles and 4 SMs, instead of 3 waves (4+4+1), Stream-K assigns 2.25 tiles per SM, ensuring equal total work.

**Partial tile coordination:**
- When a tile is split along the K dimension, partial results must be combined
- "Turnstile reduction": synchronization via global memory workspace and barrier objects
- Partial tiles write to workspace; final CTA reduces all partials

**Hybrid Stream-K (preferred in practice):**
1. **Stream-K phase**: 1-2 full waves with fractional tiles, handling quantization residue
2. **Data-parallel phase**: Remaining complete tiles using traditional wave scheduling

This hybrid approach balances load-balancing benefits against L2 cache efficiency (synchronized waves share operand data in L2).

### Persistent GEMM Patterns (CUTLASS)

CUTLASS 3.x provides production-ready persistent GEMM implementations:

**Key schedulers:**
- `PersistentScheduler`: Round-robin tile assignment
- `StreamKScheduler`: Stream-K with partial tile support
- `GroupedScheduler`: For grouped/batched GEMM with variable problem sizes

**Warp-specialized persistent GEMM:**
- Doubly-nested loop: outer loop over output tiles, inner loop over K dimension
- Producer warps begin fetching data for the next tile while consumer warps finish the current tile
- TMA-based data movement overlapped with WGMMA computation

### Benefits

| Benefit | Mechanism |
|---|---|
| **Eliminated wave quantization** | Fractional tiles ensure full SM utilization |
| **Reduced launch overhead** | Single kernel launch instead of multiple waves |
| **Better resource utilization** | SMs stay occupied throughout computation |
| **Complex scheduling** | Software schedulers enable Stream-K, grouped GEMM, etc. |
| **Prologue/epilogue amortization** | Setup costs paid once per CTA, not per tile |
| **L2 cache utilization** | Tile traversal order can be optimized for cache reuse |

### Sources

- [CUTLASS Persistent Kernels Tutorial (Colfax Research)](https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/)
- [Blackwell Cluster Launch Control (CUTLASS Docs)](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_cluster_launch_control.html)
- [CUTLASS 3.0 GEMM API](https://docs.nvidia.com/cutlass/media/docs/cpp/gemm_api_3x.html)
- [Blackwell Dense GEMM Persistent Example](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/dense_gemm_persistent.py)
- [CUTLASS Grouped Scheduler Docs](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html)

---

## 11. Warp Specialization Patterns

### Overview

Warp specialization is a GPU programming pattern where different warps within a thread block are assigned distinct roles -- typically "producer" warps handling data movement and "consumer" warps handling computation. This separation exploits the GPU's ability to switch between warps on stalls, overlapping memory latency with compute.

### Producer-Consumer Warp Groups

**Producer Warps:**
- Dedicated entirely to memory operations (global -> shared memory)
- Use TMA (Tensor Memory Accelerator) on Hopper/Blackwell for efficient bulk transfers
- Require fewer registers (no compute operations)
- Only designated TMA threads issue copy commands
- On Hopper, `setmaxnreg` allows producers to surrender unused registers to consumers

**Consumer Warps:**
- Focused exclusively on WGMMA (Warpgroup Matrix Multiply-Accumulate) operations
- Source operands directly from shared memory
- Register-intensive due to accumulator storage
- Benefit from registers released by producer warps

**Why this helps:** The warp scheduler can hide memory latency by switching to consumer warps when producers stall on memory access, and vice versa. Without specialization, all warps perform both memory and compute operations, leading to pipeline bubbles.

### Hopper Warp Specialization for GEMM

The canonical warp-specialized GEMM on Hopper uses:

1. **Multiple pipeline stages**: N shared memory buffers alternate between producer writes and consumer reads
2. **TMA for data movement**: Hardware-accelerated bulk copies from global to shared memory
3. **WGMMA for compute**: Warpgroup-level matrix multiply-accumulate directly from shared memory
4. **Dynamic register allocation**: `setmaxnreg` redistributes registers between producer and consumer warps

**Pipeline flow:**
```
Stage 0: Producer loads A[0],B[0] -> SMEM[0]  |  Consumer idle (waiting)
Stage 1: Producer loads A[1],B[1] -> SMEM[1]  |  Consumer computes from SMEM[0]
Stage 2: Producer loads A[2],B[2] -> SMEM[2]  |  Consumer computes from SMEM[1]
...
```

### Named Barriers for Warp Synchronization

Hopper introduces **mbarrier** (memory barrier) objects as hardware synchronization primitives:

| Barrier | Purpose | Behavior |
|---|---|---|
| `full_barrier` | Signals buffer contains valid data | Producers signal after write; consumers wait before read |
| `empty_barrier` | Signals buffer is free for reuse | Consumers signal after read; producers wait before write |

**Phase bit mechanism:**
- Each barrier maintains a phase bit (0 or 1)
- Phase toggles when the pipeline stage index wraps around (modulo N stages)
- Prevents ABA synchronization problems

**Transaction-based barriers (TMA variant):**
- TMA automatically increments arrival count based on bytes transferred
- No explicit `producer_commit()` needed -- the transaction count-based completion mechanism handles it
- Barrier releases when all producers arrive AND expected total byte count is reached

### Async Pipeline with Warp Specialization

The `PipelineState` class maintains per-thread state:
- **Index**: Current pipeline stage (modulo N stages)
- **Phase**: Binary value that flips on wrap-around

**Four core pipeline methods:**

```
producer_acquire(state) -> blocks until target stage is writable
producer_commit(state)  -> signals write complete (no-op for TMA)
consumer_wait(state)    -> blocks until target stage has valid data
consumer_release(state) -> signals read complete, stage now writable
```

**Triton GPU IR communication operations (for compiler-level warp specialization):**
- `ProducerAcquireOp`: Acquires a pipeline stage for writing
- `ProducerCommitOp`: Commits data to consumers
- `ConsumerWaitOp`: Waits for data availability
- `ConsumerReleaseOp`: Releases stage back to producers

### Recent Research (2025)

**Tawa (Automatic Warp Specialization):**
- Compiler framework that automatically generates warp-specialized code from sequential specifications
- Uses "asynchronous references" to express producer-consumer data flow
- Eliminates manual synchronization and pipeline management

**PyTorch Warp Specialization Support:**
- PyTorch has been integrating Triton GPU IR-level warp specialization primitives
- Enables Triton kernels to use producer-consumer patterns on Hopper
- Achieves up to 96% of FlashAttention-3 throughput with 1.21x speedup over standard Triton

### Sources

- [PyTorch Warp Specialization Blog](https://pytorch.org/blog/warp-specialization/)
- [CUTLASS GEMM Pipelining Tutorial (Colfax Research)](https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/)
- [Tawa: Automatic Warp Specialization (arXiv)](https://arxiv.org/html/2510.14719)
- [Persistent GEMM in CuTeDSL on Hopper](https://veitner.bearblog.dev/persistent-gemm-in-cutedsl-on-hopper/)
- [Optimal Software Pipelining and Warp Specialization (arXiv)](https://arxiv.org/html/2512.18134v1)
- [Warp Specialization Blog Post](https://ianbarber.blog/2025/02/16/warp-specialization/)
- [SIGARCH Efficient GEMM Kernel Designs](https://www.sigarch.org/efficient-gemm-kernel-designs-with-pipelining/)

---

## 12. NCCL and Collective Communication Optimization

### Overview

NVIDIA Collective Communication Library (NCCL) is the standard for multi-GPU and multi-node communication in deep learning. As model sizes grow, communication efficiency becomes as critical as compute efficiency. NCCL 2.27+ introduces significant optimizations for Blackwell systems, SHARP integration, and symmetric memory support.

### All-Reduce Algorithms

NCCL implements three algorithm families:

#### Ring Algorithm
- Classic all-reduce: data flows around a ring of GPUs
- Each GPU sends/receives to/from two neighbors
- **Latency**: O(2 * (N-1) * message_size / bandwidth) where N = number of GPUs
- **Bandwidth**: Near-optimal for large messages
- **Weakness**: High latency for small messages; 2*(N-1) steps required

#### Tree Algorithm
- Hierarchical reduce-broadcast using a binary tree topology
- **Latency**: O(2 * log(N) * message_size / bandwidth)
- **Bandwidth**: Lower than ring for large messages (not all links active simultaneously)
- **Strength**: Lower latency for small-to-medium messages

#### NVLS (NVLink SHARP)
- Designed for NVLink Switch (NVSwitch) systems (DGX, HGX, NVL72, GB200/GB300)
- Leverages hardware-accelerated reduction on NVSwitch fabric
- Two variants:
  - **NVLS**: Intra-node via NVLink SHARP + inter-node via CollNet/SHARP switches
  - **NVLSTree**: Intra-node via NVLink SHARP + inter-node via tree-based fan-out

#### PAT Algorithm
- Newer algorithm family in NCCL
- Optimized for specific topology configurations

### All-Gather and Reduce-Scatter Patterns

**All-Gather:** Each GPU contributes a chunk; all GPUs receive the full concatenated result. Critical for:
- FSDP (Fully Sharded Data Parallel): gathering sharded parameters before forward pass
- Tensor Parallel: gathering split activations

**Reduce-Scatter:** Each GPU contributes full data; each receives a reduced chunk. Used for:
- FSDP: scattering gradients after backward pass
- Ring-based gradient reduction in data parallelism

### Custom NCCL Plugins

NCCL supports plugin architecture for:
- **Network plugins**: Custom transport layers (e.g., AWS EFA, Azure InfiniBand)
- **Profiler plugins**: Custom profiling and telemetry
- **SHARP plugins**: Integration with InfiniBand SHARP for in-network reduction

### Communication-Computation Overlap

**Strategies:**
1. **Bucketed all-reduce**: Overlap gradient all-reduce for layer N with backward pass computation for layer N+1
2. **NCCL device APIs**: Launch communication from CUDA kernels directly, avoiding host synchronization
3. **Async collectives**: NCCL operations are asynchronous by default; careful stream management enables overlap
4. **Prefetch + overlap**: In FSDP, all-gather for the next layer while computing the current layer

**NCCL 2.27 symmetric memory:**
- Buffers with identical virtual addresses across GPUs
- Up to 9x latency reduction for small messages
- Enables low-latency communication within NVLink domains

### NCCL Tuning Parameters

| Environment Variable | Description | Values |
|---|---|---|
| `NCCL_ALGO` | Algorithm selection | `Ring`, `Tree`, `NVLS`, `NVLSTree`, `PAT`, `CollNetDirect`, `CollNetChain` |
| `NCCL_PROTO` | Protocol selection | `Simple`, `LL` (low-latency), `LL128` |
| `NCCL_MIN_NCHANNELS` | Minimum number of channels | Integer (more channels = more parallelism) |
| `NCCL_MAX_NCHANNELS` | Maximum number of channels | Integer |
| `NCCL_CROSS_NIC` | Cross-NIC communication | 0 (disable), 1 (enable), 2 (auto) |
| `NCCL_NET_GDR_LEVEL` | GPUDirect RDMA level | Controls when to use GPU Direct |
| `NCCL_SOCKET_IFNAME` | Network interface selection | Interface name (e.g., `eth0`, `ib0`) |
| `NCCL_IB_HCA` | InfiniBand HCA selection | HCA device names |
| `NCCL_BUFFSIZE` | Internal buffer size | Bytes (default varies by version) |
| `NCCL_P2P_LEVEL` | P2P communication level | Controls NVLink vs PCIe path selection |
| `NCCL_SHM_DISABLE` | Disable shared memory transport | 0 or 1 |
| `NCCL_WIN_COLL_SYMMETRIC` | Enable symmetric memory collectives | 0 or 1 |

### NCCL 2.27 Key Features

1. **Symmetric Memory Support**: Up to 9x latency reduction for small messages via identical-address buffers
2. **SHARP for NVLink + InfiniBand**: Offloads reduction to network switches, reducing SM usage from 16 to ~6 per GPU
3. **Direct NIC Support**: On Grace Blackwell, CX8 NIC achieves full 800 Gb/s bypassing CPU
4. **Communicator Shrink**: Dynamic GPU exclusion for resilient training (planned reconfiguration and error recovery)
5. **Enhanced Profiling**: Unified event states, GPU timestamp propagation, network retransmission tracking

### Sources

- [NCCL 2.27 Blog](https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27)
- [Demystifying NCCL (arXiv)](https://arxiv.org/html/2507.04786v1)
- [NCCL Cross-Datacenter Communication Blog](https://developer.nvidia.com/blog/nccl-deep-dive-cross-data-center-communication-and-network-topology-awareness)
- [NCCL Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [GB200 NVL Multi-Node Tuning Guide](https://docs.nvidia.com/multi-node-nvlink-systems/multi-node-tuning-guide/nccl.html)
- [MSCCL++ Paper (arXiv)](https://arxiv.org/html/2504.09014v2)

---

## 13. Kernel Compilation Optimization

### Overview

Kernel compilation optimization encompasses compile-time flags, profile-guided optimization, link-time optimization, runtime compilation, and caching strategies. Proper compilation can yield 20%+ performance gains without any source code changes.

### NVCC Optimization Flags

| Flag | Description | Impact |
|---|---|---|
| `-O3` | Maximum host-code optimization level | Standard for production builds |
| `--use_fast_math` | Enables fast math intrinsics (`__sinf`, `__cosf`, `__expf`) | 10-30% faster transcendentals; reduced accuracy |
| `--maxrregcount=N` | Limits registers per thread to N | Increases occupancy at cost of register spills |
| `--ftz=true` | Flush denormals to zero | Avoids expensive denormal handling |
| `--prec-div=false` | Less precise division | Faster division operations |
| `--prec-sqrt=false` | Less precise square root | Faster sqrt operations |
| `-arch=sm_XX` | Target specific GPU architecture | Enables architecture-specific instructions |
| `-Xptxas -v` | Verbose PTX assembler output | Shows register usage, spills, shared memory |
| `--ptxas-options=-v` | Alternative verbose flag | Same as above |

**Combined fast-math flags:**
```bash
nvcc -O3 --use_fast_math --ftz=true --prec-div=false --prec-sqrt=false -arch=sm_90a
```

### PGO (Profile-Guided Optimization) for CUDA

Profile-Guided Optimization for CUDA kernels is an emerging technique:

1. **Instrumented build**: Compile with profiling instrumentation
2. **Profile collection**: Run representative workloads to collect execution profiles
3. **Optimized rebuild**: Recompile using collected profiles to guide optimization decisions

PGO helps the compiler make better decisions about:
- Branch prediction hints
- Code layout (hot/cold path separation)
- Inlining decisions
- Loop unrolling factors

### LTO (Link-Time Optimization) for CUDA

CUDA Link-Time Optimization enables whole-program optimization across separately compiled translation units:

**Offline LTO (NVCC -dlto):**
```bash
# Compile to LTO-IR format
nvcc -dlto -dc file1.cu -o file1.o
nvcc -dlto -dc file2.cu -o file2.o
# Link with LTO
nvcc -dlto file1.o file2.o -o program
```

- Stores high-level intermediate representation at compile time
- Links and optimizes all device code together at link time
- Recovers performance of whole-program compilation while maintaining separate compilation flexibility
- Reported performance gains of **~20% or higher** in some cases
- Introduced in CUDA 11.2

**JIT LTO (nvJitLink):**
- CUDA 12.0 introduced `nvJitLink` library for runtime LTO
- Extends offline LTO benefits to applications using runtime linking
- Enables optimization across dynamically loaded code
- APIs: compile to LTO-IR, then link at runtime via nvJitLink

### NVRTC for Runtime Compilation

NVRTC (NVIDIA Runtime Compilation) compiles CUDA C++ device code at runtime:

**Key characteristics:**
- JIT compilation: source code to PTX/CUBIN during program execution
- Device code only: does not accept host code or host compiler extensions
- Enables runtime specialization based on problem parameters
- Useful for auto-tuning: generate kernels with different tile sizes, unroll factors, etc.

**API flow:**
```c
nvrtcProgram prog;
nvrtcCreateProgram(&prog, source, "kernel.cu", 0, NULL, NULL);
nvrtcCompileProgram(prog, numOpts, opts);
nvrtcGetPTX(prog, ptx);
// Load and launch via CUDA Driver API
```

**Use cases:**
- Kernel specialization based on runtime-determined problem sizes
- Template instantiation with runtime parameters
- Dynamic code generation for auto-tuning frameworks
- Avoiding pre-compilation of all possible kernel variants

### Triton Compilation Cache and AOT

**Triton's multi-stage caching:**
- Each compilation stage is cached independently
- Cache key = deterministic hash of source code, compile options, and target architecture
- Default cache location: `~/.triton/cache/`
- Base32-encoded hash determines directory name
- Startup time improvement: ~30% with preloaded cache

**Triton Autotuning (`@triton.autotune`):**
```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        # ... more configs
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, ...):
    ...
```

- Benchmarks all configurations and selects the fastest
- `cache_results=True` caches autotuning results to disk
- Key parameter determines when to re-tune (different problem sizes trigger re-tuning)

**Emerging Autotuning Approaches (2025-2026):**

| Approach | Innovation |
|---|---|
| **tritonBLAS** | Analytical autotuner using roofline models; selects parameters at JIT time without runtime benchmarking |
| **KernelBand** | Hierarchical multi-armed bandit optimization with hardware profiling features |
| **Triton-distributed** | Extends autotuning to multi-node/multi-GPU with joint communication-compute optimization |

**PyTorch 2.0 MegaCache:**
- Combines Inductor cache, Triton bundler (GPU code), AOT Autograd, Dynamo PGO, and autotune settings into a single downloadable archive
- Enables cache sharing across machines and CI systems
- Eliminates cold-start compilation overhead in production

### Sources

- [NVCC Documentation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/)
- [NVRTC Documentation](https://docs.nvidia.com/cuda/nvrtc/index.html)
- [CUDA 12.0 JIT LTO Blog](https://developer.nvidia.com/blog/cuda-12-0-compiler-support-for-runtime-lto-using-nvjitlink-library/)
- [Understanding Triton Cache (Red Hat)](https://next.redhat.com/2025/05/16/understanding-triton-cache-optimizing-gpu-kernel-compilation/)
- [Triton Autotune Documentation](https://triton-lang.org/main/python-api/generated/triton.autotune.html)
- [PyTorch PT2 Compilation Time Reduction](https://pytorch.org/blog/experience-in-reducing-pt2-compilation-time-for-meta-internal-workloads/)
- [Triton JIT and Caching (DeepWiki)](https://deepwiki.com/triton-lang/triton/2.2-core-operations)

---

## 14. Hardware-Specific Tricks

### Ampere (SM80/SM86/SM87) -- A100, RTX 3090, RTX 4090

#### Async Copy (cp.async)
- Hardware-accelerated asynchronous copies from global to shared memory
- Bypasses register file: data goes directly global -> shared
- Enables software pipelining: overlap compute with next iteration's data load
- Requires explicit `cp.async.commit_group()` and `cp.async.wait_group<N>()`

#### TF32 (TensorFloat-32)
- Default tensor core precision on Ampere
- Format: 1 sign + 8 exponent + 10 mantissa = 19 bits
- Input: Accepts FP32 operands, internally truncates to TF32
- Benefit: Near-FP32 accuracy with tensor core throughput
- Caveat: Not bit-exact with FP32; may need `torch.backends.cuda.matmul.allow_tf32 = False` for validation

#### Other Ampere Tricks
- **L2 cache residency control**: `cudaAccessPolicyWindow` to pin frequently accessed data
- **Reduced precision atomic operations**: Hardware FP16/BF16 atomics
- **Hardware-accelerated barriers**: `mbarrier` for shared memory synchronization
- **16x16x16 MMA instructions**: Standard tensor core tile size

### Hopper (SM90/SM90a) -- H100, H200

#### TMA (Tensor Memory Accelerator)
- Dedicated hardware unit for bulk data movement
- Supports 1D-5D tensor descriptors with automatic address calculation
- Handles swizzling, padding, and out-of-bounds clamping in hardware
- Eliminates complex address arithmetic from kernel code
- Integrates with async barriers for producer-consumer pipelining

#### WGMMA (Warpgroup Matrix Multiply-Accumulate)
- Operates at warpgroup level (4 warps = 128 threads)
- Sources one operand directly from shared memory (no register load required)
- Supports FP16, BF16, FP8 (E4M3, E5M2), INT8, TF32
- Instruction: `wgmma.mma_async`
- Asynchronous: issues instruction and continues; explicit fence/commit needed

#### DPX (Dynamic Programming Extensions)
- Hardware acceleration for dynamic programming algorithms
- Up to 7x faster than A100 for algorithms like Smith-Waterman, Floyd-Warshall
- Useful for genomics, robotics path planning, graph algorithms

#### Thread Block Clusters
- Groups of thread blocks that can synchronize and share data
- Extends CUDA hierarchy: threads -> blocks -> **clusters** -> grid
- Distributed shared memory: blocks in a cluster can access each other's shared memory
- Cluster-level barriers for synchronization
- Typical cluster sizes: 2-16 blocks

#### Hopper-Specific Optimization Tips
- Use `setmaxnreg` for dynamic register allocation in warp-specialized kernels
- Prefer WGMMA over MMA for better throughput
- Use TMA for all global-to-shared transfers (eliminates address computation)
- Pipeline depth of 3-4 stages typically optimal
- FP8 tensor cores: 2x throughput vs. FP16

### Blackwell (SM100/SM120) -- B200, GB200, RTX 5090

#### FP4 Tensor Cores
- Native FP4 and FP6 precision support
- FP4 doubles throughput over FP8
- Micro-tensor scaling with FP8 E4M3 scale per 16 values
- Supported on both datacenter (SM100) and consumer (SM120) variants

#### Second-Generation Transformer Engine
- Automatic micro-tensor scaling at sub-tensor level
- Dynamic range management without programmer intervention
- Handles FP4/FP6/FP8 quantization transparently
- Integrated with TensorRT-LLM and NeMo Framework

#### Decompression Engine
- Hardware decompression engine with 800 GB/s CPU-GPU bandwidth
- 50-200x speedup over software decompression
- Enables efficient loading of compressed model weights and data
- Supports multiple compression formats

#### TMEM (Tensor Memory) -- Datacenter Only (SM100)
- Dedicated 256KB per-SM memory for tensor data
- 16 TB/s read bandwidth, 8 TB/s write bandwidth
- Independent from L1/SMEM -- does not compete for bandwidth
- Optimal tile size: 64x64 elements (vs. Hopper's 32x32)
- Not available on consumer Blackwell (SM120)

#### tcgen05 Instructions -- Datacenter Only (SM100)
- Single-thread tensor core instruction (replaces warp-synchronous MMA)
- 11.0-11.4 cycle latency regardless of tile size
- FP4: 7,702.5 TFLOPS (96.3% peak utilization)
- Not available on consumer Blackwell (SM120)

### Consumer vs. Datacenter: How to Optimize

#### RTX 4090 (Consumer Ampere/Ada) vs. H100 (Datacenter Hopper)

| Aspect | RTX 4090 | H100 |
|---|---|---|
| **Memory** | 24 GB GDDR6X | 80 GB HBM3 |
| **Bandwidth** | 1008 GB/s | 3350 GB/s |
| **FP16 Tensor** | 330 TFLOPS | 990 TFLOPS |
| **Interconnect** | PCIe 4.0 x16 (no NVLink) | NVLink 4.0 (900 GB/s) |
| **Multi-GPU** | Max TP=2 (PCIe bottleneck) | TP up to 8+ (NVLink) |
| **Best for** | Single-GPU inference, kernel dev | Large-scale training/inference |

**Optimization strategies differ:**

**For RTX 4090 / consumer GPUs:**
- Maximize single-GPU performance (no NVLink)
- Memory capacity is the bottleneck -- aggressive quantization (INT4/FP4)
- Kernel development and profiling (excellent value for iteration)
- Batch size limited by 24 GB VRAM
- Cannot use Hopper-specific features (TMA, WGMMA, clusters)

**For H100 / datacenter GPUs:**
- Exploit TMA, WGMMA, warp specialization, thread block clusters
- Multi-GPU communication via NVLink is fast -- use tensor parallelism
- HBM3 bandwidth enables larger batch sizes
- FP8 tensor cores for 2x throughput with minimal accuracy loss
- Pipeline-parallel and tensor-parallel strategies viable

#### SM120 (Consumer Blackwell, RTX 5090) vs. SM100 (Datacenter Blackwell, B200)

The gap is wider in Blackwell than any previous generation:

| Feature | SM120 (RTX 5090) | SM100 (B200) |
|---|---|---|
| **TMEM** | Not available | 256KB per SM |
| **tcgen05** | Not available | Available |
| **Tensor core model** | Ampere-style `mma.sync` | Autonomous `tcgen05.mma` |
| **FP4** | Supported | Supported |
| **NVLink** | No | Yes (NVLink 5.0) |

**Kernel developers must maintain three separate code paths:**
1. **Hopper (SM90)**: TMA + WGMMA + warp specialization
2. **Datacenter Blackwell (SM100)**: TMEM + tcgen05 + Cluster Launch Control
3. **Consumer Blackwell (SM120)**: mma.sync-style + FP4 support (closer to Ampere programming model)

### Sources

- [NVIDIA Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)
- [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [Blackwell vs Hopper Comparison (IntuitionLabs)](https://intuitionlabs.ai/articles/blackwell-vs-hopper-gpu-architecture-comparison)
- [Microbenchmarking Blackwell (arXiv)](https://arxiv.org/html/2512.02189v1)
- [Inside DGX Spark: SM120 Analysis](https://www.backend.ai/blog/2026-02-is-dgx-spark-actually-a-blackwell)
- [RTX 4090 CUDA Performance Guide](https://www.rightnowai.co/guides/gpu-comparison/rtx-4090)
- [NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)

---

## 15. SageAttention and Other Recent Attention Innovations

### SageAttention: Quantized Attention with Accuracy Guarantees

SageAttention is a quantized attention mechanism from Tsinghua University that achieves **2-5x speedup** over FlashAttention-2 without losing end-to-end metrics. Recognized at ICLR 2025, ICML 2025, and NeurIPS 2025 (Spotlight).

#### SageAttention (v1) -- 8-bit
- Quantizes Q and K matrices to **INT8** for the QK^T computation
- Quantizes P and V matrices to **FP8** for the PV computation
- Smoothing technique for Q to handle outliers
- ~2.1x faster than FlashAttention-2, ~2.7x faster than xformers

#### SageAttention2 -- 4-bit
- Quantizes Q and K matrices to **INT4** with per-thread granularity
- Quantizes P and V matrices to **FP8**
- Introduces thorough outlier smoothing for INT4 QK^T accuracy
- Two-level accumulation strategy for FP8 PV precision

**Performance:**

| Configuration | Hardware | Speedup vs. FA2 | Speedup vs. xformers |
|---|---|---|---|
| SageAttention2 (4-bit) | RTX 4090 | ~3x | ~4.5x |
| SageAttention2 (8-bit) | RTX 4090 | ~2.7x | ~4x |
| SageAttention | RTX 5090 | 560T OPS (2.7x vs FA2) | -- |
| SageAttention2 | Hopper (H100) | Matches FA3 (FP8) speed | -- |

**Accuracy:** Negligible end-to-end metric loss across language (LLM), image (diffusion), and video generation models. Notably, SageAttention2 matches FlashAttention-3 FP8 speed on Hopper while delivering **much higher accuracy**.

### Native Sparse Attention (NSA) -- DeepSeek

NSA is DeepSeek's hardware-aligned, natively trainable sparse attention mechanism designed for efficient long-context modeling.

#### Three-Branch Architecture

1. **Compression Branch**: Learnable MLPs aggregate sequential token blocks into compressed representations. A block of length *l* with stride *d* reduces token count from *t* to ~floor((t-l)/d). Captures coarse-grained semantic patterns.

2. **Selection Branch**: Blockwise token selection using importance scores computed from compression attention weights. Top-*n* blocks retained for fine-grained detail. Reuses already-computed attention scores -- no redundant computation.

3. **Sliding Window Branch**: Fixed window (w=512 tokens) for local context. Prevents local patterns from dominating learning in other branches.

**Output fusion:** `o_t = sum_c(g_tc * Attn(q_t, K_tc, V_tc))` where g_tc are learned gate scores (MLP + sigmoid).

#### Hardware-Aligned Kernel Design

NSA implements specialized **Triton kernels** with three key optimizations:

1. **Query Grouping**: For GQA (Grouped-Query Attention), loads all query heads sharing a KV cache group into SRAM simultaneously. Eliminates redundant KV transfers across heads.

2. **Contiguous Block Fetching**: Selected KV blocks loaded sequentially from sorted indices, maintaining memory coalescence and maximizing Tensor Core utilization.

3. **Grid-Based Scheduling**: Query position loops placed in Triton's grid scheduler. Selected block counts are nearly constant across positions, simplifying optimization.

#### Performance

| Phase | Speedup (64k tokens) | Detail |
|---|---|---|
| Training forward | Up to 9.0x vs FA2 | Compute-bound regime |
| Training backward | Up to 6.0x vs FA2 | Compute-bound regime |
| Decoding | Up to 11.6x | Memory-bound: 5,632 vs 65,536 token access |

Evaluated on 27B parameter model (3B active) with GQA on A100 GPUs.

### MLA (Multi-Latent Attention) -- DeepSeek

Multi-head Latent Attention compresses the KV cache through low-rank factorization, achieving MQA-like cache efficiency while maintaining MHA-like quality.

#### Core Mechanism

Standard MHA stores separate K and V projections per head. MLA replaces this with:

1. **Compression**: A learned down-projection compresses K and V into a low-dimensional latent vector (e.g., 512 dimensions instead of 8192)
2. **Caching**: Only the compressed latent is stored in KV cache (massive memory reduction)
3. **Decompression**: Learned up-projections reconstruct full K and V when needed

#### Weight Absorption Trick

The key inference optimization: instead of decompressing the latent vector back to full K/V, the decompression matrices are **absorbed into the attention computation**:

- Instead of: `score = Q @ W_up @ compressed_kv`
- Compute: `score = (Q @ W_up) @ compressed_kv` where `Q @ W_up` is precomputed

This avoids materializing full K and V, operating directly on compressed representations.

**"Absorb" mode:** Defers KV expansion, caching only compressed latent representation. The wkv_b projection is absorbed into attention computation rather than applied upfront.

#### FlashMLA Kernel

DeepSeek's FlashMLA provides optimized MLA kernels:
- **5-15% performance improvement** for compute-bound workloads
- Up to **660 TFLOPS** on H800 SXM5 (compute-bound)
- Up to **3000 GB/s** in memory-bound configuration on H800 SXM5
- Integrated with FlashInfer for paged KV cache support

### Differential Attention

Differential attention calculates attention scores as the **difference between two separate softmax attention maps**:

```
DiffAttn(Q, K, V) = (softmax(Q1 @ K1^T) - lambda * softmax(Q2 @ K2^T)) @ V
```

Where Q is split into Q1, Q2 (similarly K into K1, K2), and lambda is a learnable scalar.

#### Key Properties
- Subtraction cancels noise, promoting sparse attention patterns
- Outperforms standard Transformers using **35-40% fewer parameters/tokens**
- Scales to 64K token contexts
- Reduces activation outliers, enabling lower-bit quantization

#### Kernel Implementation
- Can be implemented using two FlashAttention calls and a subtraction
- Specialized kernels fuse both softmax computations and the subtraction for better throughput
- The DINT variant adds integral terms and row-normalization
- SDT adds sparsity masking

#### Extensions
- Generalized to multimodal architectures (PaliGemma, DiffCLIP)
- Threshold Differential Attention: adds thresholding for ultra-sparse, sink-free attention

### FlashAttention-3 and FlashAttention-4

#### FlashAttention-3 (Hopper)
Three key innovations:
1. **Producer-consumer asynchrony**: Warp-specialized pipelining with TMA
2. **GEMM-softmax overlap**: Softmax computation overlapped with async WGMMA
3. **FP8 support**: Block quantization and incoherent processing for low-precision
- **Result:** 1.5-2.0x speedup over FlashAttention-2 on H100

#### FlashAttention-4 (Blackwell)
Addresses Blackwell's asymmetric hardware scaling (tensor cores scale faster than other units):
1. Fully asynchronous MMA pipeline with software-emulated exponential
2. Tensor memory (TMEM) + 2-CTA MMA mode reduces shared memory traffic
3. Halves global atomic adds
- **Result:** Up to 1.3x over cuDNN 9.13, 2.7x over Triton on B200 (BF16)

### Sources

- [SageAttention GitHub](https://github.com/thu-ml/SageAttention)
- [SageAttention2 Paper (arXiv)](https://arxiv.org/abs/2411.10958)
- [NSA Paper (arXiv)](https://arxiv.org/abs/2502.11089)
- [DeepSeek MLA Explained (planetbanatt)](https://planetbanatt.net/articles/mla.html)
- [FlashMLA GitHub](https://github.com/deepseek-ai/FlashMLA)
- [MLA Benefits (Verda)](https://verda.com/blog/multi-head-latent-attention-benefits-in-memory-and-computation)
- [Differential Transformer Paper (arXiv)](https://arxiv.org/abs/2410.05258)
- [FlashAttention-3 Paper](https://tridao.me/publications/flash3/flash3.pdf)
- [FlashAttention-4 Paper (arXiv)](https://arxiv.org/abs/2603.05451)
- [Reverse-Engineering FlashAttention-4 (Modal)](https://modal.com/blog/reverse-engineer-flash-attention-4)
- [Evolution of FlashAttention (ICLR 2026 Blog)](https://iclr-blogposts.github.io/2026/blog/2026/the-evolution-of-flashattention/)
- [SageAttention ICLR 2025 Paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/b286c344d38e10d2466c0514b78e2f36-Paper-Conference.pdf)
- [NSA in vLLM (DeepSeek-V3.2)](https://blog.vllm.ai/2025/09/29/deepseek-v3-2.html)
