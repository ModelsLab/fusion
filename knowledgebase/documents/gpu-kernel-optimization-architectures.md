# GPU Kernel Optimization for Modern AI Architectures: Comprehensive Knowledge Base

> **Scope**: Exhaustive reference covering transformer architecture kernels, LLM-specific operators, MoE systems, diffusion models, speech/audio models, state space models, vision transformers, and multi-GPU parallelism patterns.
>
> **Last Updated**: March 2026

---

## Table of Contents

1. [Transformer Architecture Kernel Breakdown](#1-transformer-architecture-kernel-breakdown)
2. [Key LLM Architectures and Their Kernel Needs](#2-key-llm-architectures-and-their-kernel-needs)
3. [Mixture of Experts (MoE) Kernel Challenges](#3-mixture-of-experts-moe-kernel-challenges)
4. [Specific Operator Kernels](#4-specific-operator-kernels)
5. [Vision Transformers and Multimodal](#5-vision-transformers-and-multimodal)
6. [Diffusion Model Kernels](#6-diffusion-model-kernels)
7. [Speech/Audio Model Kernels](#7-speechaudio-model-kernels)
8. [State Space Models](#8-state-space-models)
9. [Multi-GPU Patterns](#9-multi-gpu-patterns)

---

## 1. Transformer Architecture Kernel Breakdown

### 1.1 Pre-Norm vs Post-Norm: Kernel Implications

**Post-norm (Original Transformer, 2017)**:
```
x_{l+1} = LayerNorm(x_l + f(x_l))
```
- The normalization wraps around the residual connection
- Forces materialization of `x_l + f(x_l)` before norm can begin
- Gradient flow is attenuated through the normalization boundary at each layer, causing instability in deep networks (>24 layers)
- Kernel perspective: the residual add and LayerNorm are tightly coupled but the add result must be fully computed before norm begins -- limits pipelining

**Pre-norm (GPT-2 onward, now universal)**:
```
x_{l+1} = x_l + f(LayerNorm(x_l))
```
- Provides a cleaner identity path: the residual stream flows unimpeded, and normalization shapes only the sublayer contribution
- Kernel fusion opportunity: the sublayer input normalization can be fused with the subsequent linear projection (e.g., fused RMSNorm + QKV projection)
- The residual add at the output is a simple elementwise operation that can be fused with the *next* layer's normalization
- By 2025, pre-norm is used in 100% of frontier LLMs

**Kernel Fusion Pattern (Pre-norm)**:
```
Layer N output --> [Fused: Residual Add + RMSNorm] --> QKV Projection
                   ^--- single kernel, one memory round-trip
```
This fusion eliminates one full read-write cycle to HBM per layer, saving ~2x bandwidth for the norm operation. Measured speedups: **6.0x** for fused residual+RMSNorm vs. sequential PyTorch operations.

### 1.2 Multi-Head Attention: Kernel Decomposition

The attention mechanism consumes **70-80% of training time** and **60-70% of inference latency**. Its kernel decomposition:

#### QKV Projection
- Three (or one fused) GEMM operations: `Q = X @ W_Q`, `K = X @ W_K`, `V = X @ W_V`
- Typically fused into a single GEMM: `[Q, K, V] = X @ W_QKV` where `W_QKV` has shape `[d_model, 3*d_model]`
- For GQA: `W_QKV` has shape `[d_model, (n_q_heads + 2*n_kv_heads)*d_head]` since KV heads are fewer
- This is compute-bound for large batch sizes, memory-bound for small batches (autoregressive decoding)
- Can be fused with preceding RMSNorm and/or RoPE application

#### Attention Score Computation
```
S = Q @ K^T / sqrt(d_k)
```
- Naive: materializes full `[batch, heads, seq, seq]` tensor -- O(n^2) memory
- FlashAttention: tiles this computation, never materializing the full matrix
- Causal masking: upper triangle set to -inf before softmax; FlashAttention skips these tiles entirely for ~2x speedup

#### Softmax
- Online softmax (Milakov & Giber, 2018): computes softmax incrementally without needing the full row
- Three-pass naive: (1) find max, (2) compute exp and sum, (3) normalize
- Two-pass online: (1) fused max + exp + sum in single pass, (2) normalize
- FlashAttention: integrates softmax into the tiled attention kernel, maintaining running statistics `(m_i, l_i)` that are corrected across tiles

#### Output Projection
```
O = Attention(Q,K,V) @ W_O
```
- Standard GEMM, compute-bound for large batches
- Can be fused with the subsequent residual add + norm of the next sublayer

### 1.3 MLP / Feed-Forward Network: Kernel Decomposition

Modern LLMs use gated MLPs (SwiGLU/GeGLU) rather than simple two-layer FFNs:

#### Standard FFN (Original Transformer)
```
FFN(x) = ReLU(x @ W_1 + b_1) @ W_2 + b_2
```
- Two GEMMs with activation in between
- Up-projection: `[d_model] -> [4*d_model]` (compute-bound)
- Down-projection: `[4*d_model] -> [d_model]` (compute-bound)
- Activation (ReLU/GeLU): elementwise, memory-bound, trivially fusible

#### Gated MLP (SwiGLU -- LLaMA/modern standard)
```
SwiGLU(x) = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down
```
- Three weight matrices instead of two
- Hidden dimension adjusted to `~(8/3)*d_model` to maintain parameter count
- Gate projection + Up projection: can be fused into single GEMM with `W_gate_up = [W_gate; W_up]` concatenated
- The SiLU activation + elementwise multiply (gate * up) is a prime fusion target:
  - Memory-bound elementwise operations
  - Fused kernel reads gate_output and up_output once, applies SiLU, multiplies, writes result
  - Achieves **95-98% of cuBLAS throughput** for the GEMM while eliminating intermediate writes
- Down projection: standard GEMM

#### Fusion Opportunities in MLP
```
[Fused GEMM: gate+up projection] --> [Fused: SiLU + multiply] --> [GEMM: down projection]
```
Advanced fusion folds the activation directly into the GEMM epilogue using CUTLASS custom epilogues, eliminating the intermediate `[batch, seq, hidden]` tensor entirely.

### 1.4 Residual Connections and Fusion Opportunities

The residual stream `x_{l+1} = x_l + sublayer(x_l)` creates fusion opportunities at every layer boundary:

**Pattern: Residual Add + RMSNorm + Linear (next layer)**
```
residual = x + attn_output           # elementwise add
normed = RMSNorm(residual)           # reduction + elementwise
mlp_input = normed @ W_gate_up       # GEMM
```
Fused into: read `x` and `attn_output`, compute add, compute RMS, normalize, write `residual` (for next skip) and `normed`. This eliminates 2 memory round-trips.

**Pattern: AllReduce + Residual Add + RMSNorm (multi-GPU)**
TokenWeave-style fusion: the AllReduce communication, residual addition, and RMSNorm are fused into a single operation with compute-communication overlap. This is critical for tensor-parallel inference where AllReduce happens at every attention and MLP output.

### 1.5 Embedding and Output Head (LM Head)

#### Input Embedding
- Embedding lookup: gather operation indexed by token IDs
- Shape: `[vocab_size, d_model]` lookup -> `[batch, seq, d_model]`
- Memory-bound: each token reads one row of the embedding table
- Quantized embeddings (INT8/INT4) reduce memory footprint by 2-4x with dequantization fused into subsequent operations

#### Output Head (LM Head)
- Linear projection: `logits = hidden @ W_vocab^T`, shape `[batch*seq, d_model] @ [d_model, vocab_size]`
- For large vocabularies (32K-200K), this is a massive GEMM
- **Weight tying**: many models share `W_vocab` with the input embedding matrix (Phi-4, etc.), halving parameter count
- **FusedLinearCrossEntropy**: avoids materializing the full `[batch*seq, vocab_size]` logits tensor by:
  1. Chunking along the batch dimension
  2. Computing partial logits, applying online softmax + cross-entropy loss, accumulating gradients
  3. Never storing the full logit tensor
  - Saves **>2x memory**, prevents OOM for large vocabularies
  - Liger Kernel: **3x faster execution, 5x less memory** for vocab_size=163840

---

## 2. Key LLM Architectures and Their Kernel Needs

### 2.1 LLaMA / LLaMA 2 / LLaMA 3 / LLaMA 4

The LLaMA family established the modern LLM kernel recipe. Its architectural choices directly shaped which GPU kernels were developed and optimized.

#### Core Architecture (LLaMA 1/2/3 Dense)
| Component | Choice | Kernel Implication |
|-----------|--------|-------------------|
| Normalization | Pre-norm RMSNorm | Fused residual-add + RMSNorm kernel required |
| Activation | SwiGLU | Fused gate+activation+multiply kernel |
| Position Encoding | RoPE (base freq 500K in LLaMA 3) | Complex multiplication kernel on Q,K after projection |
| Attention | GQA (all sizes in LLaMA 3) | Modified FlashAttention supporting grouped KV heads |
| Bias | No bias anywhere | Simplifies all GEMM kernels (no bias epilogue) |
| Vocabulary | 128K tokens (LLaMA 3) | Large LM head GEMM; FusedLinearCrossEntropy critical |

#### RMSNorm Kernel
```
RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma
```
- Removes mean-centering from LayerNorm, saving ~10-15% compute
- Single reduction (sum of squares) vs. two reductions (mean + variance) in LayerNorm
- Fused Triton kernel: **8.1x speedup** over PyTorch, reaching **1365 GB/s** (88% peak) on A100
- Liger Kernel: **7x speedup, 3x memory reduction** per operation

#### SwiGLU Kernel
```
SwiGLU(x, W_gate, W_up, W_down) = (SiLU(xW_gate) * xW_up) W_down
```
Where `SiLU(x) = x * sigmoid(x)`

Implementation strategies:
1. **Separate kernels**: GEMM(gate) + GEMM(up) + fused_silu_mul + GEMM(down) -- 4 kernel launches
2. **Fused gate+up GEMM**: Concatenate `[W_gate; W_up]` into single GEMM, then fused_silu_mul -- 3 launches
3. **Fully fused (CUTLASS)**: Custom MMA_Traits atom (`SM80_16x4x16_F32BF16BF16F32_TN_GATED`) that interleaves gate/up columns, applies SiLU+multiply in the GEMM epilogue. Achieves **95-98% of cuBLAS** with ~50% activation memory reduction
4. **Triton fusion**: Triton kernels achieve **80-88% of peak bandwidth** for the elementwise portion, vs. 10-20% for PyTorch baselines

#### RoPE Kernel
Rotary Position Embeddings encode position through complex rotation:
```
q_rotated = q * cos(theta) + rotate_half(q) * sin(theta)
```
Where `rotate_half` swaps pairs and negates: `[-q1, q0, -q3, q2, ...]`

GPU kernel strategies:
- **Standalone kernel**: Reads Q,K from HBM, applies rotation, writes back -- 2 extra memory round-trips
- **Fused with QKV projection**: Apply RoPE in the GEMM epilogue or as part of the attention kernel input
- **Fused with attention**: FlashAttention variants that accept position indices and apply RoPE internally
- AWS Neuron QKV kernel: fuses matmul + bias + RMSNorm + RoPE into single operation
- Liger Kernel: **8x faster, 3x less memory** vs. naive implementation
- LLaMA 3's base frequency of 500,000 enables 128K context without additional kernel changes

#### GQA (Grouped-Query Attention) Kernel
```
n_q_heads = 32, n_kv_heads = 8 (typical for LLaMA 3 8B)
group_size = n_q_heads / n_kv_heads = 4
```
- Each KV head is shared by `group_size` query heads
- KV cache memory reduced by `group_size` factor
- FlashAttention handles GQA by broadcasting KV heads across query groups
- Inference: KV cache shape is `[batch, n_kv_heads, seq, d_head]` instead of `[batch, n_q_heads, seq, d_head]`

#### LLaMA 4 Maverick and Scout (2025)

LLaMA 4 marks Meta's transition to MoE and native multimodality:

| Model | Total Params | Active Params | Experts | Context | Key Innovation |
|-------|-------------|---------------|---------|---------|----------------|
| Scout | 109B | 17B | 16 routed | 10M tokens | Fits single H100 |
| Maverick | 400B | 17B | 128 routed + 1 shared | 1M tokens | Beats GPT-4o |

Architecture innovations:
- **Alternating dense and MoE layers**: Not every layer has experts, reducing routing overhead
- **Shared expert**: One expert always activated alongside the routed expert, ensuring baseline capability
- **Early fusion multimodality**: Text and vision tokens processed in unified backbone from pre-training
- **iRoPE**: Interleaved RoPE (some layers use RoPE, others use no positional encoding), enabling extreme context lengths
- Kernel needs: grouped GEMM for 128 experts, efficient top-1 routing, shared+routed expert fusion

### 2.2 Mistral / Mixtral / Mistral 3

#### Mistral 7B -- Sliding Window Attention
```
Attention window = 4096 tokens (local)
Theoretical span with layer stacking = 128K tokens
```
- Each layer attends only to the last W=4096 tokens
- KV cache is a rolling buffer of fixed size W, not growing with sequence length
- Kernel implication: attention mask is a band matrix, not lower-triangular
- FlashAttention can exploit this by only processing tiles within the window
- Cache eviction: oldest entries overwritten (circular buffer pattern)

#### Mixtral 8x7B -- Sparse MoE
- 8 experts per layer, top-2 routing
- Total parameters: 45B, active per token: ~13B
- Each expert is a full SwiGLU MLP block
- Router: linear layer `[d_model, n_experts]` -> softmax -> top-k selection

#### Mistral 3 Large (December 2025)
- 675B total parameters, 41B active
- NVIDIA optimizations: Wide Expert Parallelism (Wide-EP) with optimized MoE GroupGEMM kernels
- State-of-the-art Blackwell attention and MoE kernels
- Prefill/decode disaggregated serving support
- Speculative decoding collaboration between NVIDIA and Mistral

### 2.3 GPT-4 / GPT-4o (Speculated Architecture)

OpenAI has not officially confirmed GPT-4's architecture, but the consensus from multiple leaks and analyses:

| Aspect | Estimated Specification |
|--------|------------------------|
| Total parameters | ~1.8 trillion |
| Architecture | MoE with 16 experts |
| Expert size | ~111B parameters each |
| Layers | ~120 |
| Active parameters | ~280B per forward pass |
| Routing | Allegedly simple top-2 routing |
| Expert specialization | At least one safety expert, one coding expert |

Kernel implications of the rumored architecture:
- 16 experts with top-2 routing: moderate grouped GEMM requirements
- 1.8T total params: must span multiple nodes; expert parallelism essential
- Simple routing: standard softmax + top-k kernel sufficient
- GPT-5 (2025): confirmed complete shift to MoE architecture

### 2.4 Gemma / Gemma 2

Gemma 2 introduces several architectural quirks with distinct kernel implications:

#### Alternating Local-Global Attention
```
Even layers: Sliding window attention (window = 4096)
Odd layers: Global attention (span = 8192)
```
- Kernel must support two different attention masks in alternating layers
- Local layers use band-matrix masking; global layers use standard causal masking
- FlashAttention implementations need per-layer mask configuration

#### Logit Soft-Capping
```
soft_cap(x, cap) = cap * tanh(x / cap)
```
Applied at two levels:
- Self-attention logits: `cap = 50.0` (applied to QK^T / sqrt(d))
- Final output logits: `cap = 30.0` (applied to LM head output)

Kernel implications:
- Attention kernel must apply `tanh(x/50)` to attention scores before softmax
- This introduces a non-standard operation in the attention inner loop
- FlashAttention must be modified to include soft-capping in its tiled computation
- The tanh operation uses the same SFU (Special Function Unit) that computes exponentials -- potential bottleneck on Blackwell

#### Other Features
- Pre-norm with RMSNorm
- GeGLU activation (GELU-gated variant instead of SiLU-gated SwiGLU)
- RoPE positional embeddings
- Group-Query Attention

### 2.5 Qwen 2.5 / Qwen 3 / QwQ

#### Qwen 2.5 Architecture
- Dense models + MoE variants
- MoE layers replace standard FFN with multiple expert FFNs + routing mechanism
- Fine-grained expert segmentation with shared experts

#### Qwen 3 MoE (May 2025)
| Feature | Specification |
|---------|--------------|
| Total experts | 128 |
| Activated experts | 8 per token |
| Shared experts | None (unlike Qwen 2.5-MoE) |
| Load balancing | Global-batch load balancing loss |
| Training | Triton-Flash-Attention v3 kernel |
| Hardware | 4x2048 GPU clusters |

Kernel optimizations:
- Fine-grained experts (128 total): requires efficient grouped GEMM for many small experts
- 8 activated per token: top-8 routing kernel needed
- Global-batch load balancing: routing decisions consider the entire batch, not per-sequence
- Chunked processing for long-context: 3-7x prefill speedup through kernel/pipeline/scheduling co-optimization
- QwQ integration: thinking mode + non-thinking mode in single model, no architectural kernel changes needed

### 2.6 DeepSeek V2 / V3 / R1

DeepSeek represents the most innovative kernel-level architecture since FlashAttention.

#### Multi-Latent Attention (MLA)

The core insight: instead of caching full K,V per head, compress them into a low-rank latent:

```python
# Standard MHA KV cache: [batch, n_heads, seq, d_head] per K and V
# MLA: store compressed latent c_kv of dimension kv_lora_rank

# Compression (during prefill):
c_kv = x @ W_DKV           # [d_model] -> [kv_lora_rank], e.g., 512

# Decompression (during attention):
K = c_kv @ W_UK             # [kv_lora_rank] -> [n_heads * d_head]
V = c_kv @ W_UV             # [kv_lora_rank] -> [n_heads * d_head]
```

KV cache reduction: **93.3%** compared to standard MHA (DeepSeek-V2 vs their 67B dense model).

Special handling for RoPE: since RoPE breaks the low-rank structure, a separate small RoPE component `k_rope` is cached alongside `c_kv`:
```
K_full = [W_UK(c_kv), k_rope]   # concatenate decompressed K with RoPE component
```

#### FlashMLA Kernel (DeepSeek Open Source, Feb 2025)

Optimized for Hopper GPUs (H800 SXM5):
- Dense MLA decoding: **3000 GB/s** memory-bound, **660 TFLOPS** compute-bound
- Token-level sparse MLA decoding (FP8 KV + BF16 matmul): **410 TFLOPS**
- Prefill sparse attention: **640 TFLOPS**
- The kernel absorbs the KV decompression (c_kv @ W_UK/W_UV) into the attention computation, avoiding explicit K,V materialization

#### DeepSeekMoE Architecture

```
DeepSeek V3: 671B total, 37B active per token
- Fine-grained experts: more experts, each smaller (1/m hidden dim)
- m more experts activated per token to compensate
- Auxiliary-loss-free load balancing: bias terms added to routing scores
  for top-K selection but NOT included in mixture weights
  - Solves the fundamental problem where load-balancing auxiliary losses
    distort the primary training objective
```

#### FP8 Mixed Precision Training
DeepSeek V3 validated FP8 training at extreme scale:
- FP8 forward pass, BF16 backward pass
- Custom FP8 GEMM kernels with fine-grained scaling
- Nearly full computation-communication overlap through co-design of algorithms, frameworks, and hardware

### 2.7 Phi-3 / Phi-4

Small Language Models (SLMs) with distinct optimization opportunities:

#### Phi-4-mini (3.8B parameters)
| Feature | Specification |
|---------|--------------|
| Architecture | Dense decoder-only transformer |
| Vocabulary | 200K tokens |
| Attention | Grouped-Query Attention |
| Embedding | Shared input/output embedding |
| Position | LongRoPE for extended context |

Kernel optimization opportunities for small models:
- **Shared embedding**: single embedding matrix for both input lookup and LM head, halving embedding memory
- **TensorRT-LLM optimizations**: LongRoPE kernel, FP8, inflight batching
- **Quantization-friendly**: 4-bit QLoRA achieves 80% memory savings; models fit on consumer GPUs
- **ONNX Runtime**: cross-platform GPU support via DirectML
- Small model advantage: entire model fits in GPU SRAM for some operations, enabling aggressive fusion

### 2.8 Command R+ (Cohere)

Optimized for RAG (Retrieval-Augmented Generation) workloads:

- Long context window (128K tokens) for ingesting retrieved documents
- Architecture designed for grounded generation with citations
- Kernel implications:
  - Long-context attention: requires efficient FlashAttention with large sequence lengths
  - Cross-attention patterns for retrieved documents (in some variants)
  - Emphasis on prefill efficiency (processing long retrieved contexts)
  - Context parallelism (splitting long sequences across GPUs) becomes essential

### 2.9 Falcon / Falcon 2 / Falcon-H1

#### Falcon (Original) -- Multi-Query Attention Pioneer
```
MQA: Single K,V head shared across ALL query heads
- Maximum KV cache compression
- Custom CUDA kernels for end-to-end latency reduction
```

#### Falcon 2 (11B)
- Multi-modal (vision-to-language)
- 4-stage training: context length scaling from 2048 -> 4096 -> 8192
- High-quality data curation in final stage

#### Falcon-H1 (2025) -- Hybrid SSM+Attention
```
Architecture: Transformer attention + State Space Model (Mamba) layers
- Interleaved attention and SSM layers (similar to Jamba)
- GQA for attention layers
- Low-rank adaptation
- 16 MoE experts
- 256K token context
```
Kernel needs: both FlashAttention (for attention layers) and selective scan (for SSM layers), plus MoE grouped GEMM.

---

## 3. Mixture of Experts (MoE) Kernel Challenges

MoE models are the dominant paradigm for frontier LLMs in 2025-2026. Their kernel challenges are fundamentally different from dense models.

### 3.1 Top-K Routing: Softmax + TopK Kernel

```python
# Router computation
router_logits = x @ W_router           # [batch*seq, d_model] @ [d_model, n_experts]
router_probs = softmax(router_logits)   # [batch*seq, n_experts]
top_k_indices, top_k_weights = topk(router_probs, k)  # indices: [batch*seq, k]
```

Optimization opportunities:
- **Fused softmax + top-k**: both scan the same vector; fusing eliminates one memory pass
- **Auxiliary-loss-free balancing** (DeepSeek V3): add learned bias to `router_logits` before top-k selection, but use unbiased `router_probs` for the mixture weights
- Top-k for small k (1-8): simpler than full sort; can use parallel reduction to find top elements
- For k=1 (LLaMA 4): reduces to argmax, which is trivially parallelizable

### 3.2 Token Permutation / Unpermutation for Expert Dispatch

After routing, tokens must be reorganized so each expert receives its assigned tokens:

```
Input: [batch*seq, d_model] with routing decisions
Permutation: gather tokens by expert assignment
  Expert 0: tokens [3, 7, 12, ...]
  Expert 1: tokens [1, 5, 9, ...]
  ...
Expert computation: each expert processes its token subset
Unpermutation: scatter results back to original positions
```

Kernel challenges:
- **Irregular memory access**: gathering tokens creates non-contiguous reads
- **Variable workload**: different experts receive different numbers of tokens
- **CPU synchronization elimination**: DeepSeek's approach removes all CPU synchronization from Router -> Dispatcher -> Permutation -> GroupMLP pipeline
- **Fused permute/unpermute**: can be fused with the expert GEMM or with all-to-all communication

### 3.3 Grouped GEMM for Parallel Expert Execution

Each expert performs the same sequence of GEMMs (gate, up, down projections) but on different subsets of tokens. Grouped GEMM batches these into a single kernel:

```
Instead of:
  for expert_i in range(n_experts):
    output_i = tokens_i @ W_expert_i    # separate kernel launch per expert

Grouped GEMM:
  outputs = grouped_gemm([tokens_0, tokens_1, ...], [W_0, W_1, ...])  # single launch
```

**PyTorch Triton Persistent Grouped GEMM Kernel** (2025):
- **2.62x speedup** over manual PyTorch loop on H100
- Persistent kernel design: launches one program per SM (132 on H100), each processing multiple tiles
- Eliminates wave quantization (underutilization when tiles don't evenly divide SM count)
- Grouped launch ordering: tiles grouped by expert for L2 cache locality
- Single-wave execution with dynamic work distribution

Performance characteristics:
- Variable-sized matrices (different token counts per expert) handled within single kernel
- L2 cache reuse exploited when same expert's weight matrix accessed by multiple tiles
- 1.42x-2.62x speedup range across configurations for 16B DeepSeek-V3 training

### 3.4 Expert Parallelism Across GPUs

When models have hundreds of experts (DeepSeek V3: 256, LLaMA 4 Maverick: 128, Qwen 3: 128), experts must be distributed across GPUs:

```
Expert Parallelism (EP) degree 64:
  GPU 0: experts [0, 1, 2, 3]
  GPU 1: experts [4, 5, 6, 7]
  ...
  GPU 63: experts [252, 253, 254, 255]
```

Communication pattern: All-to-All
- Each GPU sends its routed tokens to the GPU hosting the target expert
- After expert computation, results are sent back via All-to-All
- This is the most communication-intensive operation in MoE inference

### 3.5 Load Balancing and Capacity Factors

**Problem**: Unbalanced routing means some experts are overloaded while others are idle.

Solutions:
1. **Capacity factor**: limit maximum tokens per expert to `capacity = ceil(k * n_tokens / n_experts * capacity_factor)`, dropping tokens that exceed capacity
2. **Auxiliary loss**: add a loss term encouraging uniform expert utilization (traditional approach)
3. **Auxiliary-loss-free** (DeepSeek V3): learned bias terms in routing, not affecting mixture weights
4. **Expert replication** (vLLM EPLB): periodically rebalance expert placement, replicate heavily-used experts across GPUs
5. **Mixture of Grouped Experts (MoGE)**: group-balanced routing ensures equal computation distribution per device

### 3.6 All-to-All Communication for Expert Parallelism

The All-to-All collective is the critical bottleneck for MoE:

```
Phase 1: All-to-All Dispatch
  Each GPU sends tokens to the GPU hosting their target expert
  Communication volume: O(batch_size * k * d_model)

Phase 2: Expert Computation (local GEMMs)

Phase 3: All-to-All Combine
  Each GPU sends results back to the originating GPU
```

Optimizations:
- **DeepEP** (DeepSeek): scales to 64+ EP degrees with optimized dispatch
- **FP8 quantized communication**: reduces All-to-All bandwidth by 2x
- **Compute-communication overlap**: pipeline expert computation with dispatch/combine
- **Wide-EP** (NVIDIA TensorRT-LLM): optimized for NVL72 coherent memory domain

### 3.7 EP + TP Combinations

Real deployments combine multiple parallelism strategies:

```
DeepSeek V3 configuration: [TP=1, PP=4, CP=1, EP=64, VP=8, FSDP=64]

Typical combinations:
- TP within node (NVLink), EP across nodes (InfiniBand)
- TP=8 (within DGX), EP=8 (across 8 nodes) -> 64 GPU cluster
```

The key constraint: TP requires high-bandwidth communication (NVLink), while EP can tolerate lower bandwidth (InfiniBand) because All-to-All transfers smaller per-token data.

vLLM parallelism configurations:
- Pure EP: all GPUs hold different experts, all handle same input
- EP + TP: experts distributed across EP groups, each group tensor-parallel
- EP + TP + PP: add pipeline stages for very large models

---

## 4. Specific Operator Kernels

### 4.1 RMSNorm: Fused Kernel Patterns

```python
def rmsnorm(x, weight, eps=1e-6):
    rms = sqrt(mean(x^2, dim=-1, keepdim=True) + eps)
    return x / rms * weight
```

**Computation profile**: reduction (sum of squares) + elementwise (divide + multiply). Entirely memory-bound.

**Naive PyTorch**: 5 separate CUDA kernels with 5-10us overhead each:
1. `x.pow(2)` -- elementwise
2. `.mean(-1)` -- reduction
3. `+ eps` -- elementwise
4. `.rsqrt()` -- elementwise
5. `x * rsqrt * weight` -- elementwise

**Fused Triton kernel** (single pass):
```
Performance: 1365 GB/s on A100 (88% peak)
vs. PyTorch: 168 GB/s (11% peak)
Speedup: 8.1x
```

**Advanced fusions**:
- Residual Add + RMSNorm: **6.0x** speedup (eliminates intermediate tensor)
- AllReduce + Residual Add + RMSNorm (TokenWeave): compute-communication overlap for multi-GPU
- RMSNorm + Quantization: fused norm + FP8/INT8 conversion for quantized inference
- RMSNorm + Linear projection: norm output fed directly to GEMM without HBM round-trip

**Backward pass**: cache the RMS value (scalar per row) during forward; recompute normalized output from cached RMS

### 4.2 LayerNorm: Fused with Residual Add

```python
def layernorm(x, weight, bias, eps=1e-5):
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, keepdim=True, unbiased=False)
    return (x - mean) / sqrt(var + eps) * weight + bias
```

Two reductions (mean and variance) vs. one for RMSNorm. Still memory-bound.

Fusion patterns identical to RMSNorm but with additional mean subtraction. Used in:
- Original Transformers
- BERT/GPT-2 (historical)
- Diffusion model U-Nets (GroupNorm variant)
- Vision Transformers

### 4.3 RoPE (Rotary Position Embedding)

**Mathematical formulation**:
For each pair of dimensions `(2i, 2i+1)` at position `p`:
```
[q_{2i}', q_{2i+1}'] = [cos(p*theta_i), -sin(p*theta_i)] [q_{2i}  ]
                        [sin(p*theta_i),  cos(p*theta_i)] [q_{2i+1}]

where theta_i = base^(-2i/d), base typically 10000 or 500000
```

**GPU implementation strategies**:

1. **Pair-wise rotation** (most common):
```python
# Triton/CUDA kernel
cos_cache, sin_cache = precomputed_freqs[positions]  # lookup
q_even, q_odd = q[..., ::2], q[..., 1::2]
q_out_even = q_even * cos - q_odd * sin
q_out_odd = q_even * sin + q_odd * cos
```

2. **Complex multiplication** (mathematically equivalent):
```python
q_complex = view_as_complex(q.reshape(..., d//2, 2))
freqs_complex = polar(1.0, theta * positions)  # unit complex numbers
q_rotated = (q_complex * freqs_complex).view_as_real()
```

3. **Fused with QKV**: apply RoPE inside the projection kernel's epilogue
   - AWS Neuron: fuses matmul + RoPE in single operation
   - Only available for context encoding (full sequence), not token generation

4. **3D RoPE**: for multimodal models, extends rotation to spatial dimensions
   - Can be fused into linear projections for inference efficiency

**Frequency caching**: precompute `cos(theta)` and `sin(theta)` tables for all positions up to max_seq_len. These are small (e.g., 128K * 64 * 2 = 16MB for LLaMA 3) and cached in GPU memory.

### 4.4 SwiGLU / GeGLU: Fused Gate + Activation + Multiply

**SwiGLU** (LLaMA, Mistral, DeepSeek, Qwen):
```
SwiGLU(x) = SiLU(x @ W_gate) * (x @ W_up)
SiLU(x) = x * sigmoid(x)
```

**GeGLU** (Gemma):
```
GeGLU(x) = GELU(x @ W_gate) * (x @ W_up)
```

**Fusion approaches** (ordered by sophistication):

1. **Elementwise fusion only**: fuse SiLU + multiply after separate GEMMs
   - Triton: 80-88% peak bandwidth, 8x over PyTorch
   - Saves one HBM write of the intermediate gate tensor

2. **GEMM + elementwise fusion**: concatenated `[W_gate; W_up]` GEMM with fused epilogue
   - CUTLASS custom epilogue applies SiLU to even columns, multiplies with odd columns
   - Achieves 95-98% of cuBLAS throughput

3. **Full fusion (Bitdefender approach)**: custom CUTLASS MMA_Traits
   - Define fictional `SM80_16x4x16_F32BF16BF16F32_TN_GATED` instruction
   - Interleave gate/up weights in column layout
   - Apply gating within MMA accumulator using register-level operations
   - ~50% activation memory reduction
   - Max error vs. reference: 3.91e-06 (BF16)

### 4.5 Top-K Sampling: Efficient GPU Sampling

Sampling from LLM output distributions is becoming a bottleneck as vocabulary sizes grow (32K-200K tokens).

**FlashInfer Sorting-Free Sampling** (adopted by vLLM, SGLang, MLC-LLM):

Traditional approach: sort logits -> cumsum -> sample (requires O(V log V) sorting)

FlashInfer approach: **rejection sampling** without sorting:
1. Draw random value `u ~ Uniform(0,1)`
2. Compute prefix sums (CDF) using CUB parallel prefix sum
3. Find token where `CDF[k-1] <= u < CDF[k]`
4. For top-p/top-k filtering: use **dual pivot rejection sampling**
   - Binary search-like bounds narrow the valid probability range
   - Guaranteed O(log(1/epsilon)) rounds maximum
   - Each round shrinks search range by >= 50%

All implemented in a single CUDA kernel, avoiding the multi-kernel overhead of sort-based approaches. Result: **>50% reduction in sampling time** across all tested models.

**Numerical challenge**: floating-point prefix sums are non-associative, meaning parallel computation can produce non-monotonic CDF values. Special care required to prevent invalid token generation.

### 4.6 Beam Search: On-GPU Beam Management

Traditional beam search requires CPU coordination for beam selection. GPU-native approaches:
- Maintain beam hypotheses in GPU memory
- Top-k selection per beam on GPU
- Beam merging via parallel reduction
- Still limited by the sequential nature of autoregressive generation

### 4.7 Embedding Lookup: Fused with Norm, Quantized Embeddings

**Standard embedding**: simple gather from embedding table
```
output[i] = embedding_table[token_ids[i]]  # gather operation
```

**Quantized embeddings**:
- INT8/INT4 embedding weights reduce memory by 2-4x
- Dequantization fused with subsequent RMSNorm in single kernel
- Important for large vocabularies (200K+ tokens)

**Fused embedding + norm**: read quantized embedding, dequantize, apply RMSNorm, all without writing intermediate results to HBM.

### 4.8 Softmax: Online Softmax, Fused with Masking

**Three-pass naive softmax**:
```
1. m = max(x)           # Pass 1: reduction
2. e = exp(x - m)       # Pass 2: elementwise
3. output = e / sum(e)  # Pass 3: reduction + elementwise
```

**Two-pass online softmax** (Milakov & Giber):
```
1. Compute m and d simultaneously:
   for each x_i:
     m_new = max(m, x_i)
     d = d * exp(m_old - m_new) + exp(x_i - m_new)
2. Normalize: output_i = exp(x_i - m) / d
```

**FlashAttention integration**: online softmax is the key algorithmic insight enabling tiled attention:
- Running statistics `(m_i, l_i)` maintained per tile
- When processing new tile j: rescale previous output by `exp(m_{j-1} - m_j) * l_{j-1} / l_j`
- FlashAttention-4 adds **conditional rescaling**: only rescale when `m_j - m_{j-1} > tau` (threshold 8.0), reducing rescaling operations

**Fused causal masking**: adding `-inf` to future positions (upper triangle) before softmax ensures they become zero after exponentiation. FlashAttention skips computing tiles in the upper triangle entirely, yielding ~2x speedup for causal attention.

**Soft-capping** (Gemma 2): `cap * tanh(x / cap)` applied to attention logits before softmax. Must be integrated into the attention kernel's inner loop.

---

## 5. Vision Transformers and Multimodal

### 5.1 ViT Patch Embedding Convolution

Vision Transformer converts images to sequences via patch embedding:
```
Input: [batch, 3, H, W]
Conv2D: kernel_size=patch_size, stride=patch_size
  e.g., patch_size=16: [batch, 3, 224, 224] -> [batch, d_model, 14, 14]
Reshape + Transpose: -> [batch, 196, d_model]
```

**Kernel considerations**:
- The Conv2D with `kernel=stride=patch_size` is equivalent to a linear projection of flattened patches
- GPUs are highly optimized for convolution (cuDNN), making this efficient
- Hybrid approaches (FastViT): use Conv+Transformer stages for better latency
  - FastViT: higher accuracy than ViT-L/14 at ~4x lower latency

### 5.2 Vision-Language Model Kernel Patterns (LLaVA, etc.)

**LLaVA architecture**:
```
Image -> Vision Encoder (CLIP ViT) -> Projection -> LLM
Text -> Tokenizer -> Embedding -> LLM
```

Kernel-level considerations:
- Vision encoder runs once per image (not autoregressive)
- Projection layer: simple linear or MLP mapping vision features to LLM dimension
- LLM processes concatenated [vision_tokens, text_tokens]
- Vision tokens are typically 576-2048 per image
- For high-resolution: split image into patches, encode each, concatenate -- increases sequence length significantly

**FastVLM (CVPR 2025)**: hybrid convolution-transformer vision encoder
- Pareto-optimal latency vs. accuracy
- Trainable vision encoder during LLaVA Stage-2 for resolution scaling to 768
- Key insight: convolution layers in early stages are more efficient than attention for low-level feature extraction

### 5.3 Image Preprocessing on GPU

Traditional CPU preprocessing (resize, normalize, crop) becomes a bottleneck for high-throughput serving:
- NVIDIA DALI: GPU-accelerated data loading and augmentation pipeline
- torchvision transforms on GPU: basic operations (resize, normalize) run on GPU
- Fused preprocess + patch embedding: combine resize, normalize, patch extraction in single kernel

---

## 6. Diffusion Model Kernels

### 6.1 U-Net Architecture Kernel Profile

The U-Net in diffusion models (Stable Diffusion 1.x/2.x, SDXL) consists of:

```
Encoder path:
  Conv2D -> GroupNorm -> SiLU -> Conv2D -> (optional) Downsample
  + Self-Attention blocks at certain resolutions
  + Cross-Attention blocks for text conditioning

Middle block:
  Self-Attention + Cross-Attention at lowest resolution

Decoder path:
  Mirror of encoder with upsampling
  + Skip connections from encoder
```

**Dominant kernels by compute time**:
1. Self/Cross-Attention: FlashAttention applicable, dominates at low resolution
2. Conv2D: cuDNN optimized, dominates at high resolution
3. GroupNorm + SiLU: memory-bound, prime fusion target
4. Timestep embedding MLP: small, fused into norm/conv

### 6.2 DiT (Diffusion Transformer) Architecture

DiT replaces U-Net with a pure transformer:
```
Input: noisy latent patches -> linear patch embedding
Processing: N transformer blocks with:
  - Self-attention (FlashAttention applicable)
  - MLP (SwiGLU or standard FFN)
  - AdaLN (Adaptive Layer Normalization) conditioned on timestep + class
Output: linear projection -> predicted noise/velocity
```

**Kernel advantages over U-Net**:
- Uniform computation graph (no varying resolution stages)
- CUDA Graphs work perfectly (static graph)
- FlashAttention directly applicable
- No irregular convolution shapes

**AdaLN kernel**: `AdaLN(x, c) = (1 + gamma(c)) * LayerNorm(x) + beta(c)` where gamma, beta are predicted from conditioning. Fused into single kernel with norm.

### 6.3 MMDiT (Multimodal DiT) -- SD3 / Flux

**SD3 and FLUX use MMDiT**:
```
Two streams: text tokens and image tokens
Separate weights for each modality's QKV projections
Joint attention: bidirectional attention between text and image tokens
  - Text sees image, image sees text
  - Single attention operation over concatenated sequences
```

Kernel implications:
- Attention over combined sequence (text + image): standard FlashAttention works
- Separate QKV projections: two GEMMs instead of one (one for text, one for image)
- Modality-specific normalization: separate AdaLN parameters per stream

### 6.4 Rectified Flow

SD3 and FLUX use Rectified Flow instead of DDPM:
- Straight-line ODE paths instead of stochastic diffusion
- Velocity prediction instead of noise prediction
- Fewer sampling steps needed (typically 20-50 vs 50-1000)
- Kernel implication: same operations per step, but fewer total steps -> end-to-end faster

### 6.5 Cross-Attention in Diffusion Models

Text conditioning via cross-attention:
```
Q = image_features @ W_Q    # queries from image
K = text_features @ W_K      # keys from text encoder
V = text_features @ W_V      # values from text encoder
Attention = softmax(Q @ K^T / sqrt(d)) @ V
```
- Text features are computed once and reused across all denoising steps
- KV from text can be cached across steps
- FlashAttention with cross-attention mode (different Q and KV sequence lengths)

### 6.6 Group Norm in Convolutions

GroupNorm is the standard normalization in U-Net-based diffusion models:
```
GroupNorm: divide channels into G groups, normalize each group independently
Typical: G=32, channels per group = C/32
```
- Different from LayerNorm/RMSNorm: normalizes over spatial+channel subset
- Fused GroupNorm + SiLU: common pattern in ResNet blocks
- CUDA kernel: one threadblock per group per spatial position
- Performance: often the bottleneck in high-resolution U-Net blocks

### 6.7 Optimized Diffusion Kernels

**CUDA Graphs for diffusion inference**:
- Capture entire denoising step as a CUDA graph
- Replay with minimal CPU overhead per step
- Eliminates per-step kernel launch latency (hundreds of kernels per step)
- PyTorch `torch.compile` with `mode="reduce-overhead"` enables CUDA Graphs automatically

**torch.compile for diffusion models**:
- Operator fusion (GroupNorm + SiLU, etc.)
- Out-of-order execution
- Triton kernel generation for fused operations
- SDXL speedup with SDPA: inference time 4.63s -> 3.31s

**FastUSP framework (2025)**:
- Multi-level optimization for distributed diffusion inference
- CUDA Graphs + computation-communication reordering
- FP8 quantized collective communication
- Pipelined Ring attention with double buffering

---

## 7. Speech/Audio Model Kernels

### 7.1 Whisper Architecture and Kernel Opportunities

**Architecture**: Encoder-Decoder Transformer for speech recognition

#### Audio Preprocessing
```
Raw audio (16kHz) -> STFT -> Mel filterbank -> Log magnitude
Output: [batch, 128, 3000] mel spectrogram (128 mel bins, 30s window)
```
- STFT: can be computed on GPU using cuFFT
- Mel filterbank: sparse matrix multiply (fixed, non-learned)
- Log + normalization: elementwise, trivially fusible

#### Convolutional Stem (Encoder Input)
```
Conv1D(128, d_model, kernel=3, padding=1) + GELU
Conv1D(d_model, d_model, kernel=3, stride=2, padding=1) + GELU
```
- Stride-2 downsampling: 3000 -> 1500 frames (critical for efficiency)
- Small kernels (3x1): memory-bound on GPU, benefit from fusion with GELU
- Positional encoding added after conv stem

#### Encoder Transformer Blocks
- Standard pre-norm transformer with multi-head self-attention
- FlashAttention applicable for the 1500-length sequences
- Optimization: entire encoder runs once per audio segment (not autoregressive)

#### Decoder with Cross-Attention
- Causal self-attention on generated text tokens
- Cross-attention to encoder output (1500 frames)
- KV cache for encoder cross-attention can be computed once and reused
- Autoregressive: each token requires one decoder forward pass

#### Optimization Opportunities
- **1.58-bit quantization**: Whisper Small quantized to 1.58 bits with custom Optimium kernels for on-device deployment
- **LiteASR**: low-rank compression of encoder, reducing bottleneck
- **Streaming**: CarelessWhisper replaces self-attention with causal masks at chunk boundaries
- **Batched decoding**: process multiple audio segments simultaneously
- **KV cache optimization**: encoder cross-attention KV reused across all decoder steps

### 7.2 TTS Models (VITS, Bark, Chatterbox)

#### VITS Architecture
```
Text encoder -> Duration predictor -> Flow-based decoder -> HiFi-GAN vocoder
```
- Variational inference + adversarial learning
- Key kernels: 1D convolutions (dilated), flow transformations (affine coupling layers), HiFi-GAN upsampling convolutions
- Real-time synthesis: entire pipeline must run faster than audio playback speed

#### Bark (Suno)
```
Text -> Semantic tokens (GPT-style autoregressive)
Semantic tokens -> Coarse acoustic tokens (GPT-style)
Coarse tokens -> Fine acoustic tokens (non-autoregressive)
Fine tokens -> Waveform (EnCodec decoder)
```
- Three-stage transformer pipeline
- Each stage is a standard GPT-like model with FlashAttention applicable
- Bottleneck: three sequential autoregressive stages
- Can generate music, sound effects, not just speech

#### Chatterbox (Resemble AI, 2025)
- MIT-licensed, multilingual TTS + voice cloning
- Zero-shot cloning from seconds of audio
- Real-time synthesis
- Kernel opportunities: speaker embedding integration, prosody modeling, streaming vocoder

### 7.3 Spectrogram Processing on GPU

**GPU-accelerated audio pipeline**:
```
1. Resampling: polyphase filter (torchaudio on GPU)
2. STFT: cuFFT for batched FFTs
3. Mel filterbank: sparse matmul (cusparse or dense with zero-masking)
4. Log compression: elementwise kernel
5. Normalization: reduction + elementwise (similar to LayerNorm)
```

**Fusion opportunities**:
- Fuse mel filterbank + log + normalize into single kernel
- Pre-compute STFT on GPU to avoid CPU-GPU transfer
- torchaudio.transforms can run entirely on GPU since PyTorch 2.0

---

## 8. State Space Models

### 8.1 Mamba / Mamba-2 Selective Scan Kernel

#### Mamba-1 Selective Scan

The selective SSM makes parameters input-dependent, breaking the LTI (Linear Time-Invariant) property that enables fast convolution:
```
h_t = A(x_t) * h_{t-1} + B(x_t) * x_t    # state update
y_t = C(x_t) * h_t                         # output
```
Where A, B, C are functions of input x_t (data-dependent).

**Hardware-aware implementation** (three classical techniques):

1. **Kernel Fusion**: load SSM parameters (Delta, A, B, C) from HBM to SRAM, perform discretization and recurrence in SRAM, write only final outputs back to HBM. Never materialize the expanded state tensor `[B, L, D, N]` in HBM.

2. **Parallel Scan**: the recurrence, though sequential per element, is parallelizable via work-efficient parallel associative scan algorithm. The associative operator is:
   ```
   (a1, b1) * (a2, b2) = (a1*a2, a2*b1 + b2)
   ```
   This maps to GPU parallel prefix operations with O(L) work and O(log L) depth.

3. **Recomputation**: intermediate states not stored during forward pass. During backward pass, inputs reloaded from HBM to SRAM and states recomputed. Matches FlashAttention's memory strategy.

**Performance**: 20-40x faster than naive implementations. Surpasses FlashAttention for long sequences in throughput (linear vs. quadratic scaling).

**State size limitation**: Mamba-1 limited to small state expansion (N=16) because the scan operates on scalar or small-vector states without tensor core support.

#### Mamba-2: Structured State Space Duality (SSD)

Key insight: SSMs and attention are mathematically dual -- both can be expressed as operations on semiseparable matrices.

**SSD Algorithm** (chunk-based):
```
Divide sequence into chunks of size Q (typically 64-256):

Step 1: Intra-chunk computation (parallel matrix multiply)
  - Each chunk: compute attention-like matrix within chunk
  - Pure matmul -> tensor core friendly

Step 2: Chunk state computation (matmul)
  - Compute the state that each chunk produces

Step 3: Inter-chunk state passing (sequential on chunks, not tokens)
  - Pass states between chunks sequentially
  - 100x reduction in sequential operations (chunks vs. tokens)

Step 4: Output combination (matmul)
  - Combine intra-chunk results with inter-chunk contributions
```

**GPU advantages over Mamba-1**:
- Steps 1, 2, 4 are **pure matrix multiplications** -> tensor core utilization
- State expansion N=64 or 128 (vs N=16 in Mamba-1) now efficient
- **2-8x faster** than Mamba-1 selective scan
- Tensor parallelism: only 1 all-reduce per layer (vs 2 in Mamba-1)
- Supported in NVIDIA NeMo, Hugging Face Transformers 4.39+

**Remaining optimization opportunity**: current implementations "do not specifically take advantage of new features on H100 GPUs" (TMA, warpgroup MMA, etc.)

### 8.2 How Selective Scan Maps to GPU

```
GPU Memory Hierarchy Mapping:

HBM (80GB, 2TB/s):
  - Input sequence X: [batch, seq_len, d_model]
  - Output Y: [batch, seq_len, d_model]
  - Parameters: Delta, A, B, C projections

SRAM (20MB shared memory):
  - Chunk of input: [batch, chunk_size, d_inner]
  - SSM parameters for chunk: [batch, chunk_size, state_dim]
  - Running hidden state: [batch, d_inner, state_dim]

Registers:
  - Per-thread accumulation values
  - Current state elements
```

The kernel's memory access pattern:
1. Load chunk from HBM to SRAM (coalesced)
2. Compute discretization and scan entirely in SRAM
3. Write output chunk to HBM (coalesced)
4. Pass inter-chunk state via registers or shared memory

### 8.3 RWKV Kernel Patterns

RWKV reinvents RNNs with transformer-competitive performance:

#### WKV Operator (Core Mechanism)
```
wkv_t = sum_{i=1}^{t-1} exp(-(t-1-i)*w + k_i) * v_i + exp(u + k_t) * v_t
        / sum_{i=1}^{t-1} exp(-(t-1-i)*w + k_i) + exp(u + k_t)
```
Where w = time decay, u = current token bonus, k = key, v = value.

- Each channel processed independently: O(T) complexity per channel
- During training (parallel mode): entire sequence processed at once via custom CUDA kernel
- During inference (recurrent mode): O(1) per token, constant memory (no KV cache)

#### RWKV Architecture Evolution
| Version | Year | Key Innovation | Kernel Implication |
|---------|------|---------------|-------------------|
| RWKV-4 | 2023 | Foundational WKV | Custom CUDA WKV kernel |
| RWKV-5 "Eagle" | 2024 | Multi-headed WKV | Parallel multi-head WKV kernel |
| RWKV-6 "Finch" | 2024 | Data-dependent decay | More complex scan kernel |
| RWKV-7 "Goose" | March 2025 | Generalized Delta Rule | Same O(N) train, O(1) inference |

#### RWKV-7 Performance
- O(1) per-token inference cost regardless of sequence length
- At 1M tokens: Transformers OOM, RWKV-7 maintains constant 0.8x relative cost
- Training: parallelizable like transformers, O(T) complexity via CUDA kernel
- Triton and FLA (Flash Linear Attention) implementations available

### 8.4 Hybrid Attention + SSM Architectures

#### Jamba (AI21, ICLR 2025)
```
Architecture: interleaved Attention + Mamba layers
Ratio: 1 attention layer per 7 Mamba layers
MoE: added every 2 blocks with 16 experts
Context: 256K tokens
KV Cache: only 4GB even at long contexts (due to few attention layers)
```

Kernel requirements:
- FlashAttention for sparse attention layers
- Mamba selective scan for SSM layers
- MoE grouped GEMM for expert layers
- Three distinct kernel types in a single model

#### Jamba 1.5 Architecture
- 72 layers total
- Grouped-Query Attention for the attention layers
- Low-rank adaptation
- 16 MoE experts per MoE layer

#### Zamba
- Similar hybrid but with **shared self-attention layer** across Mamba blocks
- Single attention layer's parameters reused by all Mamba layers
- Reduces parameter count while maintaining performance
- Weight sharing kernel: same attention weights applied at multiple points in the network

#### Falcon-H1 (2025)
- Hybrid Transformer + Mamba architecture
- Combined with MoE (16 experts)
- GQA for attention layers
- 256K context
- Demonstrates strong RAG performance (Natural Questions, TriviaQA, HotpotQA)

---

## 9. Multi-GPU Patterns

### 9.1 Tensor Parallelism: Column Parallel + Row Parallel Linear Layers

Tensor Parallelism (TP) splits individual weight matrices across GPUs:

#### Column Parallel Linear
```
Weight W split along columns: W = [W_1 | W_2 | ... | W_tp]
Each GPU i computes: Y_i = X @ W_i  (partial output)
No communication needed for the forward pass of this layer alone
```

#### Row Parallel Linear
```
Weight W split along rows: W = [W_1; W_2; ...; W_tp]
Input X must be split: X_i = X[..., partition_i]
Each GPU i computes: Y_i = X_i @ W_i  (partial result)
AllReduce needed: Y = sum(Y_1, Y_2, ..., Y_tp)
```

#### Attention with TP
```
QKV projection: Column Parallel (split heads across GPUs)
  GPU 0: heads [0-3], GPU 1: heads [4-7], etc.
Output projection: Row Parallel (requires AllReduce)
  Each GPU has partial output, AllReduce combines
```

#### MLP with TP
```
Gate+Up projection: Column Parallel (split hidden dim)
Down projection: Row Parallel (requires AllReduce)
```

**Total communication per layer**: 2 AllReduce operations (one after attention, one after MLP).

**Constraint**: TP requires high-bandwidth interconnect (NVLink: 900 GB/s on H100). On PCIe (64 GB/s), TP overhead dominates for TP > 2.

### 9.2 Pipeline Parallelism: Micro-batching, Interleaved Scheduling

Pipeline Parallelism (PP) assigns consecutive layers to different GPUs:

```
GPU 0: layers [0-15]
GPU 1: layers [16-31]
GPU 2: layers [32-47]
GPU 3: layers [48-63]
```

#### Micro-batching (GPipe)
- Split batch into micro-batches
- Pipeline stages process different micro-batches simultaneously
- **Bubble**: GPUs idle during pipeline fill/drain
- Bubble fraction: `(pp_size - 1) / num_microbatches`

#### Interleaved Scheduling (1F1B)
- Alternate forward and backward passes
- Reduces peak memory compared to GPipe
- Virtual pipeline stages: assign non-contiguous layers to same GPU
  ```
  GPU 0: layers [0-7, 32-39]
  GPU 1: layers [8-15, 40-47]
  ...
  ```
- Reduces bubble fraction by factor of virtual stages

#### Communication
- Point-to-point send/recv between adjacent pipeline stages
- Only activation tensors transferred (not weights)
- Can use lower-bandwidth interconnect (InfiniBand sufficient)

### 9.3 Sequence Parallelism: Splitting Along Sequence Dimension

Sequence Parallelism (SP) extends TP by parallelizing operations that TP leaves sequential:

```
Standard TP: LayerNorm, Dropout, Residual Add are replicated on all GPUs
  -> wasteful for activation memory

SP: these operations are split along the sequence dimension
  -> each GPU holds [batch, seq/tp_size, d_model] for these operations
```

Communication pattern:
```
Before attention/MLP (column parallel): AllGather to reconstruct full sequence
After attention/MLP (row parallel): ReduceScatter to distribute results
```

Replaces the AllReduce in standard TP with AllGather + ReduceScatter (same total communication volume, but reduces activation memory by tp_size).

### 9.4 Context Parallelism: For Very Long Sequences

Context Parallelism (CP) partitions activations along the sequence dimension across **all** layers:

```
Sequence length: 1M tokens, CP degree: 8
Each GPU processes: 125K tokens of the sequence
```

Unlike SP (which only covers norm/dropout/residual):
- CP divides attention computation across GPUs
- Ring Attention: GPUs pass KV blocks around a ring, each computing partial attention with its local Q against visiting K,V
- Double buffering: compute attention on current KV block while receiving next block

Key differences from TP:
- TP splits model (weight) dimensions; CP splits data (sequence) dimension
- CP can use lower-bandwidth interconnect since KV blocks are streamed
- Essential for 1M+ token contexts (LLaMA 4 Scout: 10M tokens)

### 9.5 Ring AllReduce

```
GPUs: [0] -> [1] -> [2] -> [3] -> [0]  (ring topology)

Phase 1: ReduceScatter (k-1 steps for k GPUs)
  Step 1: GPU i sends chunk i to GPU (i+1), receives chunk (i-1)
  Step 2: GPU i sends reduced chunk to next GPU
  ...
  After k-1 steps: each GPU holds one fully-reduced chunk

Phase 2: AllGather (k-1 steps)
  Each GPU broadcasts its reduced chunk around the ring
  After k-1 steps: all GPUs have complete reduced data

Total steps: 2(k-1)
Bandwidth: 2 * (k-1)/k * data_size  (optimal for large messages)
```

**Performance**: near-optimal bandwidth utilization for large messages. Independent of number of GPUs (bandwidth term approaches 2*data_size).

### 9.6 Tree AllReduce

```
        [0,1,2,3]     <- root has full reduction
       /         \
   [0,1]         [2,3]     <- intermediate reductions
   /   \         /   \
 [0]   [1]     [2]   [3]   <- leaf GPUs

Phase 1: Reduce (upward)
  Leaves send to parents
  Parents reduce and forward up

Phase 2: Broadcast (downward)
  Root sends reduced result to children
  Children forward to leaves

Total steps: 2 * log2(k)
Latency: O(log k)  (better than Ring's O(k) for small messages)
```

**Trade-off**: Tree has better latency for small messages (log k vs k steps), Ring has better bandwidth for large messages.

### 9.7 NVLink vs PCIe Performance Implications

| Feature | NVLink (H100) | PCIe Gen5 |
|---------|---------------|-----------|
| Bandwidth (bidirectional) | 900 GB/s | 128 GB/s |
| Latency | ~1 us | ~5 us |
| Topology | Full mesh (DGX) | Switch-based |
| Best for | TP, AllReduce | PP, EP (cross-node) |
| GPU-to-GPU direct | Yes (P2P) | Via CPU (usually) |

**NVLink Switch (B200/GB200)**: enables NVLink across nodes, not just within a node. NVL72 (72 GPUs in a single NVLink domain) fundamentally changes parallelism strategies by making TP across 72 GPUs feasible.

### 9.8 NCCL Optimization

NCCL (NVIDIA Collective Communications Library) is the foundation for all multi-GPU communication.

#### Protocol Selection
| Protocol | Message Size | Bandwidth | Latency | Mechanism |
|----------|-------------|-----------|---------|-----------|
| Simple | Large (>64KB) | Near-peak | ~6us/hop | Bulk data transfer |
| LL (Low Latency) | Tiny (<1KB) | 25-50% peak | ~1us/hop | 8-byte atomic writes |
| LL128 | Medium | ~95% peak (NVLink) | ~2us/hop | 128-byte aligned atomics |

NCCL selects protocol dynamically based on message size, topology, and hardware capabilities.

#### Channel and SM Allocation
- Collectives subdivided into multiple channels (separate CUDA blocks on individual SMs)
- Aggressive channel multiplication risks NIC FIFO buffer inefficiency when per-channel chunks < 512KB
- NCCL 2.27: NVLink + IB SHARP reduces SM consumption from 16+ to 6 or fewer, freeing SMs for compute

#### Key NCCL 2.27 Optimizations (2025)
1. **SHARP support**: offloads collective operations to network devices
   - Eliminates redundant data transfer between endpoints
   - Scales to 1000+ GPUs for LLM training
2. **Symmetric memory**: identical virtual addresses across GPUs enable optimized kernels
   - Up to **9x latency reduction** for small messages
3. **GPUDirect P2P/RDMA**: eliminates CPU/host memory staging
4. **P2P_DIRECT mode**: bypasses IPC handles and FIFO buffering for same-process ranks
5. **QP separation**: forward queue pairs for bulk data, reverse for control traffic

#### Communication-Computation Overlap Patterns
```
Pattern 1: AllReduce during computation
  Start AllReduce for layer N's output
  While AllReduce is in flight, compute layer N+1's attention/MLP

Pattern 2: Fused AllReduce + RMSNorm (TokenWeave)
  Combine communication and normalization in single operation

Pattern 3: Pipelined send/recv for PP
  Send activation to next stage while computing backward for previous micro-batch
```

NVIDIA Transformer Engine kernels (RMSNorm, Linear, DotProductAttention) natively support overlap, achieving **250 TFLOPS/sec on H100s** in production MoE training.

---

## Appendix A: FlashAttention Evolution

| Version | Year | Target GPU | Key Innovation | Performance |
|---------|------|-----------|----------------|-------------|
| FlashAttention | 2022 | A100 | IO-aware tiling, online softmax | 2-4x over PyTorch |
| FlashAttention-2 | 2023 | A100/H100 | Better work partitioning, reduced non-matmul ops | 2x over FA1 |
| FlashAttention-3 | 2024 | H100 (Hopper) | Asynchronous execution, warp specialization, FP8 | 1.5-2x over FA2 |
| FlashAttention-4 | 2026 | B200 (Blackwell) | Algorithm-kernel pipelining co-design | 1.3x over cuDNN 9.13 |

### FlashAttention-4 Technical Deep Dive

**The Asymmetric Scaling Problem**: On Blackwell B200:
- Tensor cores: 8192 ops/clock/SM (2x Hopper)
- Exponential unit (MUFU): 16 ops/clock/SM (unchanged)
- Shared memory bandwidth: 128 bytes/clock/SM (unchanged)

The matmul is no longer the bottleneck. Softmax exponential and shared memory traffic now dominate.

**Forward Pass Solutions**:
1. **Software-emulated exponential**: polynomial approximation of 2^x using Cody-Waite range reduction
   - Degree-3 polynomial matches hardware within 1 BF16 ULP on 99% of inputs
   - 10-25% of evaluations use emulation; rest use hardware MUFU
   - Frees SFU for other operations

2. **Conditional softmax rescaling**: only rescale when `m_j - m_{j-1} > 8.0`
   - Reduces non-matmul operations in the critical path

3. **TMEM (Tensor Memory, 256KB/SM)**: accumulators live in TMEM instead of registers
   - Enables larger tile sizes (128x128)
   - Async MMA writes to TMEM while CUDA cores handle elementwise work

**Backward Pass Solutions**:
1. **2-CTA MMA mode**: two CTAs act as single larger tile
   - Each CTA stages half of operand B
   - Reduces shared memory traffic redundancy
   - dQ computed via distributed shared memory (DSMEM) between CTA pairs

2. **Ping-pong warpgroup schedule**: two warpgroups of 128 threads each
   - Output rescaling decoupled to separate "correction" warpgroup
   - Takes rescaling out of the critical path

**Implementation**: entirely in CuTe-DSL (Python-embedded), **20-30x faster compile** times vs C++ templates (2.5s vs 55s).

**Performance**: 1613 TFLOPs/s on B200 BF16 (71% utilization). Deterministic backward achieves 75% of non-deterministic speed.

---

## Appendix B: Liger Kernel -- Production Fused Kernels

[Liger Kernel](https://github.com/linkedin/Liger-Kernel) by LinkedIn provides the most comprehensive collection of production-ready fused Triton kernels for LLM training:

| Kernel | Speedup | Memory Reduction | Key Technique |
|--------|---------|-----------------|---------------|
| RMSNorm | 7x | 3x | Fused norm + scaling, cached RMS |
| RoPE | 8x | 3x | Fused rotation without intermediate |
| SwiGLU | 1.6x | 2x | Fused gate+activation+multiply |
| CrossEntropy | 3x | 5x | Online softmax with in-place replacement |
| FusedLinearCrossEntropy | - | >2x | Avoids materializing logits |
| GeGLU | 1.5x | 2x | GELU-gated variant |
| FusedLinearJSD | - | >2x | Jensen-Shannon divergence fusion |

**End-to-end impact**: 20% multi-GPU training throughput increase, 60% memory reduction.

Integrated with: Hugging Face Transformers, TorchTune, Axolotl, LLaMA-Factory.

---

## Appendix C: Kernel Optimization Decision Tree

```
Is the operation...

COMPUTE-BOUND (arithmetic intensity > machine's ops:byte ratio)?
  -> Optimize FLOPS: larger tiles, tensor core utilization, FP8/INT8
  -> Examples: GEMM, attention score computation (large batch)
  -> Tools: CUTLASS, cuBLAS, Triton with tl.dot

MEMORY-BOUND (arithmetic intensity < ops:byte ratio)?
  -> Optimize bandwidth: kernel fusion, minimize HBM accesses
  -> Examples: RMSNorm, SwiGLU activation, softmax, residual add
  -> Tools: Triton, custom CUDA, operator fusion via torch.compile
  -> Target: >80% of peak memory bandwidth (1365 GB/s on A100 for RMSNorm)

LATENCY-BOUND (small operations with kernel launch overhead)?
  -> Reduce kernel count: CUDA Graphs, aggressive fusion
  -> Examples: many small ops in diffusion model denoising steps
  -> Tools: CUDA Graphs, torch.compile(mode="reduce-overhead")

COMMUNICATION-BOUND (multi-GPU collective operations)?
  -> Overlap with computation, reduce data volume, use faster interconnect
  -> Examples: AllReduce in TP, All-to-All in EP
  -> Tools: NCCL tuning, DeepEP, TokenWeave, FP8 communication
```

---

## Sources

### Transformer Architecture
- [The Crystallization of Transformer Architectures (2017-2025)](https://jytan.net/blog/2025/transformer-architectures/)
- [What Changed in the Transformer Architecture](https://huggingface.co/blog/rishiraj/what-changed-in-the-transformer-architecture)
- [DeepSpeed Transformer Kernel](https://www.deepspeed.ai/tutorials/transformer_kernel/)
- [Kernel-Level GPU Optimization for Transformer Attention](https://oaqlabs.com/2025/10/12/kernel-level-gpu-optimization-for-transformer-attention-a-technical-deep-dive/)

### LLaMA Family
- [LLaMA Components: RMSNorm, SwiGLU, and RoPE](https://mbrenndoerfer.com/writing/llama-components-rmsnorm-swiglu-rope)
- [Llama 3 Architecture Overview](https://www.emergentmind.com/topics/llama-3-architecture)
- [The LLaMA Herd](https://syhya.github.io/posts/2025-04-06-llama/)
- [The Llama 4 Herd - Meta AI](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)
- [Welcome Llama 4 Maverick & Scout on Hugging Face](https://huggingface.co/blog/llama4-release)

### DeepSeek
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [DeepSeek-V3 Explained: Multi-head Latent Attention](https://towardsdatascience.com/deepseek-v3-explained-1-multi-head-latent-attention-ed6bee2a67c4/)
- [FlashMLA - DeepSeek](https://github.com/deepseek-ai/FlashMLA)
- [Understanding Multi-Head Latent Attention](https://planetbanatt.net/articles/mla.html)
- [MLA Explanation - Hugging Face](https://huggingface.co/blog/NormalUhr/mla-explanation)

### Mistral/Mixtral
- [Mistral Architecture: Sliding Window Attention](https://mbrenndoerfer.com/writing/mistral-architecture-sliding-window-attention)
- [NVIDIA-Accelerated Mistral 3](https://developer.nvidia.com/blog/nvidia-accelerated-mistral-3-open-models-deliver-efficiency-accuracy-at-any-scale/)
- [Mixtral - Hugging Face](https://huggingface.co/docs/transformers/model_doc/mixtral)

### Gemma
- [Gemma 2: Improving Open Language Models](https://arxiv.org/html/2408.00118v1)
- [Gemma 2 Architecture Deep Dive with PyTorch](https://amaarora.github.io/posts/2024-07-07%20Gemma.html)

### Qwen
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [Qwen 2.5 Technical Report](https://arxiv.org/pdf/2412.15115)
- [Qwen2.5-Max: Intelligence of Large-scale MoE](https://qwenlm.github.io/blog/qwen2.5-max/)

### GPT-4
- [Peering Inside GPT-4: MoE Architecture](https://medium.com/@seanbetts/peering-inside-gpt-4-understanding-its-mixture-of-experts-moe-architecture-2a42eb8bdcb3)
- [AI Expert Speculates on GPT-4 Architecture](https://wandb.ai/byyoung3/ml-news/reports/AI-Expert-Speculates-on-GPT-4-Architecture---Vmlldzo0NzA0Nzg4)

### Phi
- [Accelerate Microsoft Phi-4 SLMs](https://www.intel.com/content/www/us/en/developer/articles/technical/accelerate-microsoft-phi-4-small-language-models.html)
- [Phi-4-mini-instruct - Hugging Face](https://huggingface.co/microsoft/Phi-4-mini-instruct)

### Falcon
- [Falcon-H1: Hybrid-Head Language Models](https://falcon-lm.github.io/blog/falcon-h1/)
- [Falcon - Hugging Face](https://huggingface.co/docs/transformers/main/en/model_doc/falcon)

### FlashAttention
- [FlashAttention-4: Algorithm and Kernel Pipelining Co-Design](https://arxiv.org/abs/2603.05451)
- [FlashAttention-4 - Together AI](https://www.together.ai/blog/flashattention-4)
- [Evolution of FlashAttention - ICLR 2026](https://iclr-blogposts.github.io/2026/blog/2026/the-evolution-of-flashattention/)
- [Reverse-Engineering Flash Attention 4 - Modal](https://modal.com/blog/reverse-engineer-flash-attention-4)
- [FlashAttention-3 Paper](https://tridao.me/publications/flash3/flash3.pdf)
- [Tuning Flash Attention - NVIDIA](https://developer.nvidia.com/blog/tuning-flash-attention-for-peak-performance-in-nvidia-cuda-tile/)

### Kernel Optimization
- [From 11% to 88% Peak Bandwidth: Triton Kernels for LLM Inference](https://subhadipmitra.com/blog/2025/triton-kernels-llm-inference/)
- [Towards Fused Kernels for Gated MLP - Bitdefender](https://bit-ml.github.io/blog/post/fused-swiglu-kernel/)
- [Liger-Kernel: Efficient Triton Kernels for LLM Training](https://github.com/linkedin/Liger-Kernel)
- [Fused Triton Kernels in LLM Optimization](https://www.emergentmind.com/topics/fused-triton-kernels)
- [Tri-RMSNorm: Efficient Triton RMSNorm](https://github.com/dtunai/Tri-RMSNorm)

### MoE Systems
- [Accelerating MoE with Triton Persistent Grouped GEMM - PyTorch](https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/)
- [Democratizing MoE Training with NVIDIA PyTorch Parallelism](https://developer.nvidia.com/blog/accelerating-large-scale-mixture-of-experts-training-in-pytorch/)
- [Scaling DeepSeek MoEs with vLLM - Red Hat](https://developers.redhat.com/articles/2025/09/08/scaling-deepseek-style-moes-vllm-and-llm-d-using-wide-ep)
- [MoE in Megatron Core](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html)
- [Efficient MoE Communication - Perplexity](https://www.perplexity.ai/hub/blog/efficient-and-portable-mixture-of-experts-communication)

### Sampling
- [Sorting-Free GPU Kernels for LLM Sampling - FlashInfer](https://flashinfer.ai/2025/03/10/sampling.html)

### RoPE
- [Rotary Embeddings: A Relative Revolution - EleutherAI](https://blog.eleuther.ai/rotary-embeddings/)
- [A Deep Dive into RoPE CUDA Implementation](https://www.linkedin.com/pulse/accelerating-transformer-position-embeddings-deep-dive-kuchinad-ik9af)

### State Space Models
- [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/html/2312.00752v2)
- [Mamba-2: State Space Duality Part I-IV](https://goombalab.github.io/blog/2024/mamba2-part1-model/)
- [Jamba: Hybrid Transformer-Mamba](https://arxiv.org/abs/2403.19887)
- [RWKV: Reinventing RNNs](https://github.com/BlinkDL/RWKV-LM)
- [RWKV-7 Goose Architecture Analysis](https://www.youngju.dev/blog/ai-papers/2026-03-03-rwkv7-goose-architecture-analysis.en)

### Diffusion Models
- [Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748)
- [SD3 & FLUX: MMDiT Architecture](https://blog.sotaaz.com/post/sd3-flux-architecture-en)
- [FastUSP: Multi-Level Collaborative Acceleration](https://arxiv.org/html/2602.10940v1)
- [SDXL Optimizations - Hugging Face](https://huggingface.co/blog/simple_sdxl_optimizations)
- [CUDA Graph Best Practice - NVIDIA](https://docs.nvidia.com/dl-cuda-graph/examples/stable-diffusion-v2.html)

### Speech/Audio
- [Whisper Encoder-Decoder Architecture](https://www.emergentmind.com/topics/whisper-encoder-decoder-architecture)
- [Whisper MLPerf Inference Benchmark](https://mlcommons.org/2025/09/whisper-inferencev5-1/)
- [Best Open Source TTS Models - Resemble AI](https://www.resemble.ai/best-open-source-text-to-speech-models/)

### Multi-GPU Communication
- [Demystifying NCCL: GPU Communication Protocols](https://arxiv.org/html/2507.04786v1)
- [NCCL 2.27 - NVIDIA](https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27)
- [Understanding NCCL Tuning - NVIDIA](https://developer.nvidia.com/blog/understanding-nccl-tuning-to-accelerate-gpu-to-gpu-communication)
- [Flash Communication: Reducing TP Bottleneck](https://arxiv.org/html/2412.04964v1)
- [Parallelisms - NVIDIA NeMo](https://docs.nvidia.com/nemo-framework/user-guide/24.12/nemotoolkit/features/parallelisms.html)
- [Parallel Kittens: Simplifying Multi-GPU AI Kernels](https://hazyresearch.stanford.edu/static/posts/2025-11-17-pk/ParallelKittens.pdf)

### Vision/Multimodal
- [FastVLM: Efficient Vision Encoding for VLMs - CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Vasu_FastVLM_Efficient_Vision_Encoding_for_Vision_Language_Models_CVPR_2025_paper.pdf)

### Architecture Comparisons
- [LLM Architecture: DeepSeek V3 vs Llama 4](https://langcopilot.com/posts/2025-07-22-from-deepseek-v3-to-kimi-k2-eight-modern-llm-architectures)
- [The Big LLM Architecture Comparison - Sebastian Raschka](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)
