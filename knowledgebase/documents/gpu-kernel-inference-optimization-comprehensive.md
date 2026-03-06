# GPU Kernel Inference Optimization: Comprehensive Knowledge Base

> Last updated: March 2026. This document covers the complete landscape of LLM inference optimization, from individual kernel fusion patterns to datacenter-scale serving strategies.

---

## Table of Contents

1. [Prefill Optimization](#1-prefill-optimization)
2. [Decode Optimization](#2-decode-optimization)
3. [Speculative Decoding](#3-speculative-decoding)
4. [Kernel Fusion Patterns](#4-kernel-fusion-patterns-for-inference)
5. [Operator Scheduling and Graph Optimization](#5-operator-scheduling-and-graph-optimization)
6. [Batching Strategies](#6-batching-strategies)
7. [Context Length Optimization](#7-context-length-optimization)
8. [Multi-GPU Inference Optimization](#8-multi-gpu-inference-optimization)
9. [System-Level Optimization](#9-system-level-optimization)
10. [Benchmarking Inference Performance](#10-benchmarking-inference-performance)
11. [Cost Optimization](#11-cost-optimization)

---

## 1. Prefill Optimization

### 1.1 Understanding the Prefill Phase

The prefill phase processes all input tokens in parallel to build the KV cache. It is **compute-bound**: massive matrix multiplications across all input tokens simultaneously drive GPU tensor cores to maximum utilization. A single request with a long prompt can already saturate GPU compute capacity.

**Key characteristics:**
- All input tokens processed in one forward pass
- Generates the full KV cache for the input sequence
- Latency directly determines TTFT (Time to First Token)
- GPU utilization is high due to large matrix dimensions
- Arithmetic intensity is high (many FLOPs per byte of memory traffic)

**Bottleneck analysis:**
- For short prompts (<256 tokens): kernel launch overhead and small matrix sizes may underutilize tensor cores
- For medium prompts (256-4K tokens): well-balanced compute utilization
- For long prompts (>4K tokens): memory for KV cache becomes a concern; attention computation scales quadratically without FlashAttention

### 1.2 Chunked Prefill

Chunked prefill splits large prefill operations into smaller chunks that can be interleaved with decode steps. This is critical for production serving.

**How it works:**
1. A long prompt (e.g., 8192 tokens) is divided into chunks (e.g., 512 tokens each)
2. Each chunk is processed as a prefill step
3. Between chunks, pending decode requests can be scheduled
4. The KV cache is built incrementally across chunks

**Why it matters:**
- **Prevents decode starvation**: Without chunking, a single long prefill blocks all decode requests, causing ITL spikes
- **Improves GPU utilization**: Creates a better mix of compute-bound (prefill) and memory-bound (decode) operations
- **Enables SLO compliance**: Prevents tail latency blowups for concurrent requests

**Implementation in vLLM:**
```
# Enable chunked prefill
--enable-chunked-prefill
# Or in Python
LLM(enable_chunked_prefill=True)

# Tune the chunk size via max_num_batched_tokens
# Smaller chunks = better ITL, worse TTFT
# Larger chunks = better TTFT, worse ITL
```

**Practical guidance:**
- Start with `max_num_batched_tokens=2048` and tune based on your SLA requirements
- Monitor both TTFT and ITL percentiles (p50, p95, p99) while adjusting
- For latency-sensitive workloads: smaller chunks (512-1024 tokens)
- For throughput-sensitive workloads: larger chunks (2048-4096 tokens)

### 1.3 Tensor Parallelism for Prefill

Tensor parallelism splits individual model layers across multiple GPUs, enabling faster prefill for large models.

**When to use TP for prefill:**
- Model weights exceed single GPU memory
- TTFT SLO is tight and single-GPU prefill is too slow
- NVLink or equivalent high-bandwidth interconnect is available

**Key consideration:** AllReduce communication after each layer adds latency. On NVLink (900 GB/s bidirectional on H100), this overhead is ~10-30 microseconds per layer. On PCIe, it can be 10x higher, making TP less beneficial.

**Prefill-specific TP strategy:**
- TP=2 or TP=4 within a single node is typically optimal for prefill
- Higher TP degrees hit diminishing returns due to communication overhead
- For very long contexts, combine TP with context parallelism

### 1.4 Flash Attention for Long-Context Prefill

FlashAttention is essential for efficient prefill, especially with long contexts.

**Evolution:**
- **FlashAttention-1**: Tiled attention with online softmax, 2-4x speedup over PyTorch
- **FlashAttention-2**: Better parallelism and work partitioning, ~2x over FA-1
- **FlashAttention-3**: Hopper-specific (H100), warp specialization + TMA, FP8 support, 1.5-2x over FA-2, ~30% training time reduction on Llama-2 7B
- **FlashAttention-4 (2025)**: Advanced kernel fusion, sparsity-aware computation, memory-efficient data layouts

**Why FlashAttention matters for prefill:**
- Standard attention materializes the N x N attention matrix to HBM (O(N^2) memory)
- FlashAttention computes attention in tiles, never materializing the full matrix (O(N) memory)
- Enables processing 2x longer contexts on the same hardware
- Critical for 128K+ context windows

### 1.5 torch.compile with max-autotune for Prefill

`torch.compile(mode="max-autotune")` is the most aggressive optimization mode in PyTorch 2.x.

**What it does:**
- Leverages Triton-based or template-based matrix multiplications
- Benchmarks multiple kernel implementations and picks the fastest
- Applies aggressive operator fusion via TorchInductor
- Generates specialized kernels for exact tensor dimensions

**Practical guidance:**
```python
model = torch.compile(model, mode="max-autotune")
```

- **Compilation cost**: Takes significantly longer than `mode="reduce-overhead"` but produces faster kernels
- **Best use case**: Serving regime where compilation cost is amortized over many requests
- **Prefill-specific benefit**: Large matrix multiplications in prefill benefit most from autotuned GEMM kernels
- **Caveat**: Dynamic shapes (varying sequence lengths) may cause recompilation; use `torch._dynamo.config.dynamic_shapes = True` or pad to fixed sizes

### 1.6 Optimal Batch Size Tuning for Prefill

**The tradeoff:**
- Larger batch sizes increase arithmetic intensity and GPU utilization
- But also increase memory usage (KV cache grows linearly with batch size)
- And increase TTFT for individual requests (more work per forward pass)

**Guidelines:**
- For latency-optimized serving: batch size 1-4, minimize TTFT
- For throughput-optimized serving: batch size 8-64, maximize tokens/second
- Monitor GPU compute utilization; if <70%, increase batch size
- Monitor memory utilization; if >90%, decrease batch size or enable more aggressive quantization

---

## 2. Decode Optimization

### 2.1 Why Decode is Memory-Bandwidth Bound

The decode phase generates one token at a time. Each decode step requires:
1. Reading all model weights (~14 GB for a 7B FP16 model)
2. Reading the entire KV cache (grows with sequence length)
3. Performing a single matrix-vector multiplication (not matrix-matrix)
4. Writing back a single token's worth of activations

**The fundamental problem:**
- A single token produces a vector, not a matrix, so tensor cores are drastically underutilized
- The ratio of compute to memory access (arithmetic intensity) is extremely low
- On an H100 with 3.35 TB/s HBM bandwidth, reading 14 GB of weights takes ~4.2 ms
- The actual compute (matrix-vector multiply) takes <0.1 ms
- Therefore, **decode is >97% memory-bound**

**Bandwidth requirements scale with context:**
- KV cache for Llama-3.1-70B at 4096 tokens: ~1.34 GB per request
- At batch size 32 with 4K context: ~43 GB of KV cache reads per step
- This competes with weight reads for limited HBM bandwidth

### 2.2 CUDA Graphs for Decode

CUDA graphs capture a sequence of GPU operations and replay them with minimal CPU overhead, eliminating kernel launch latency.

**Why CUDA graphs matter for decode:**
- Each decode step launches dozens of small kernels (attention, FFN, norms, etc.)
- Kernel launch overhead: ~5-10 microseconds per kernel on modern GPUs
- For a 32-layer model with ~10 kernels per layer: ~1.6-3.2 ms of pure launch overhead
- CUDA graphs reduce this to a single graph launch: ~10-20 microseconds total
- **Typical speedup: 10-30% for decode at small batch sizes**

**Implementation approach:**
```python
# Capture phase (run once)
with torch.cuda.graph(graph):
    output = model(input_ids, kv_cache)

# Replay phase (run for each decode step)
input_ids.copy_(new_token_ids)
graph.replay()
```

**Limitations:**
- Requires fixed tensor shapes (padding needed for variable-length batches)
- Cannot use dynamic control flow during graph replay
- vLLM and SGLang handle this automatically with padded batch slots

**vLLM CUDA Graph integration:**
- FlashInfer backend enables `AttentionCGSupport.UNIFORM_BATCH` mode
- TRTLLM attention backend preferred for decode with CUDA graphs on Hopper+ GPUs
- Graphs are pre-captured for common batch sizes at startup

### 2.3 Batching Multiple Decode Requests

Batching is the primary way to improve decode efficiency by increasing arithmetic intensity.

**Why batching helps:**
- With batch size 1: read 14 GB weights, do 1 matrix-vector multiply (wasted bandwidth)
- With batch size 32: read 14 GB weights, do 32 matrix-vector multiplies (32x better compute/memory ratio)
- Batching amortizes the cost of reading weights across multiple requests

**Practical limits:**
- KV cache memory grows linearly with batch size
- At some point, KV cache + weights exceed GPU memory
- Typical sweet spots: batch 16-128 depending on model size and context length
- Continuous batching (Section 6) enables dynamic batch sizing

### 2.4 Persistent Kernels for Decode

Persistent kernels keep thread blocks alive across multiple operations, avoiding the overhead of kernel launches and thread block scheduling.

**How they work:**
- Instead of launching a new kernel for each operation, a single kernel persists
- Thread blocks synchronize via global memory barriers
- Operations are chained within the same kernel lifetime
- Particularly effective for the many small operations in decode

**Applications:**
- FlashFormer implements whole-model persistent kernels achieving 82% bandwidth utilization vs 68-75% for GPTFast
- Fused attention + FFN kernels that process an entire transformer block
- Amortize launch overhead for streaming quantization workloads

**Performance:**
- FlashFormer (whole-model kernel): 8-20% speedup over GPTFast for Llama 3.1 8B at batch size 1
- Up to 61% faster than vLLM at batch size 1 for unquantized models
- Benefits diminish at larger batch sizes where compute dominates

### 2.5 FP8 KV Cache to Reduce Memory Reads

Reducing KV cache precision directly reduces memory bandwidth consumption during decode.

**Precision hierarchy for KV cache:**
| Format | Bytes/element | Memory reduction vs FP16 | Accuracy impact |
|--------|---------------|--------------------------|-----------------|
| FP16/BF16 | 2 | Baseline | None |
| FP8 (E4M3) | 1 | 50% | <1% on most benchmarks |
| NVFP4 | 0.5 | 75% (50% vs FP8) | <1% on Qwen3-Coder-480B |
| INT4 | 0.5 | 75% | Variable, task-dependent |

**FP8 KV cache in practice:**
- Supported natively on Hopper (H100/H200) and Blackwell GPUs
- vLLM: `--kv-cache-dtype fp8` enables FP8 KV cache
- Values stored as FP8 E4M3, dequantized to FP8/BF16 before attention computation
- New tokens' KV vectors are quantized after computation, before cache insertion
- Fused `cat+quant` kernels for FP8 KV cache in MLA (Multi-Latent Attention)

**NVFP4 KV cache (Blackwell):**
- 4-bit KV cache with E4M3 FP8 scaling factors
- Doubles context length or batch size within same memory budget
- 3x better TTFT from improved cache hit rates (up to 20% improvement)
- Outperforms MXFP4 by ~5% on MMLU due to more granular block scaling
- Integrated with NVIDIA Dynamo's KV-aware routing and offload

### 2.6 Speculative Decoding to Increase Arithmetic Intensity

Speculative decoding transforms decode from a sequential, memory-bound operation into a more compute-intensive one. See Section 3 for full details.

**Core insight:** Instead of generating 1 token per forward pass, speculate K tokens and verify them in a single forward pass of the target model. This turns a batch-1 matrix-vector multiply into a batch-K matrix-matrix multiply, dramatically increasing arithmetic intensity.

---

## 3. Speculative Decoding

### 3.1 Fundamental Concept

Speculative decoding accelerates autoregressive generation by using a fast approximation to generate candidate tokens, then verifying them in parallel with the target model. The key guarantee: **the output distribution is identical to the target model** (lossless acceleration).

**Core algorithm:**
1. Draft model generates K candidate tokens autoregressively (fast, cheap)
2. Target model verifies all K tokens in a single forward pass (parallel)
3. Rejection sampling determines which tokens to accept
4. Accepted tokens are kept; first rejected token is resampled from adjusted distribution
5. Worst case: 1 token accepted (same as no speculation). Best case: K+1 tokens accepted.

**Speedup formula:**
```
Speedup = (accepted_tokens + 1) / (draft_cost + verify_cost)
         ≈ acceptance_length / (1 + draft_overhead_ratio)
```

Where `draft_overhead_ratio = draft_time / target_time`. Lower overhead and higher acceptance rate = better speedup.

### 3.2 Draft Model Speculation

The original approach: use a smaller model from the same family as the draft.

**Architecture:**
- Target: Llama-3.1-70B, Draft: Llama-3.1-8B
- Draft generates K tokens (typically K=3-8)
- Target verifies all K in one forward pass

**Acceptance rate factors:**
- Model family alignment (same tokenizer, similar training data)
- Task difficulty (easy tasks: 80-90% acceptance, hard tasks: 40-60%)
- Temperature (higher temperature = higher acceptance rates)
- Draft model quality vs. speed tradeoff

**Practical considerations:**
- Draft model must share the same tokenizer as target
- Draft model weights consume additional GPU memory
- Optimal K depends on acceptance rate: higher acceptance = larger K worthwhile
- At batch size >16, benefits diminish as target model becomes compute-bound

**Performance:** Typically 1.5-3x speedup at batch size 1, diminishing to near 1x at batch size 32+.

### 3.3 Self-Speculation (LayerSkip, Early Exit)

Self-speculative decoding uses the target model itself for drafting, eliminating the need for a separate draft model.

**LayerSkip (Meta, ACL 2024):**

Three components:
1. **Training recipe**: Layer dropout with increasing rates for later layers + early exit loss where all layers share the same LM head
2. **Draft phase**: Exit at an early layer (e.g., layer 8 of 32) using the shared LM head
3. **Verify phase**: Run remaining layers to verify and correct

**Advantages over external draft models:**
- No additional memory for draft model weights
- Shared KV cache between draft and verify phases
- Shared compute and activations
- No tokenizer mismatch issues

**Performance:**
- 2.16x speedup on CNN/DM summarization
- 1.82x speedup on coding tasks
- 2.0x speedup on TOPv2 semantic parsing

**Integration status (2025):**
- Hugging Face Transformers (Nov 2024)
- PyTorch torchtune (Dec 2024)
- Hugging Face TRL (Mar 2025)

**DEL (Dynamic Exit Layer, COLM 2025):**
- Context-aware selection of which layer to exit at
- Adapts exit layer based on token difficulty
- Improves acceptance rates over fixed exit layers

### 3.4 Medusa (Multiple Prediction Heads)

Medusa adds K extra language model heads to the base model, each trained to predict a different future position.

**Architecture:**
```
Base Model Hidden States
    |
    +-- LM Head (standard, next token)
    +-- Medusa Head 1 (token at position +1)
    +-- Medusa Head 2 (token at position +2)
    +-- Medusa Head 3 (token at position +3)
    ...
    +-- Medusa Head K (token at position +K)
```

**Key innovations:**
- **No separate draft model**: Heads are lightweight (single linear layer or small MLP)
- **Tree-based candidate construction**: With TopK=2 per head, creates a tree of 2^K candidate sequences
- **Tree attention**: Custom attention mask verifies entire tree in one forward pass
- **Medusa-2**: Full-model training (not just heads) + self-distillation for adding heads to any fine-tuned model without original training data

**Performance:**
- 2.2-3.6x speedup on various Vicuna models at batch size 1
- ~2x practical speedup in most configurations
- Heads add minimal parameters (~0.5-2% of model size per head)

**Practical guidance:**
- Train 3-5 Medusa heads for best speed/quality tradeoff
- Use TopK=2-3 per head to keep tree size manageable
- Tree verification adds compute overhead; too many heads hurt throughput

### 3.5 EAGLE / EAGLE-2 / EAGLE-3 (Autoregressive Draft)

The EAGLE family uses a lightweight autoregressive prediction head that operates on the target model's hidden states.

**EAGLE-1 (ICML 2024):**
- Attaches a small autoregressive draft head (~2-5% of target model parameters)
- Draft head ingests hidden states from the target model's last layer
- Generates token trees autoregressively
- 2-3x speedup

**EAGLE-2 (EMNLP 2024):**
- Dynamic tree retry mechanism
- Adjusts tree structure based on confidence
- 20-40% improvement over EAGLE-1

**EAGLE-3 (NeurIPS 2025):**
- **Token-level prediction** instead of feature prediction
- **Multi-layer feature fusion**: Combines low-, mid-, and high-level semantic features (not just top layer)
- **Training-time test**: Simulates actual inference conditions during training, fixing distribution mismatch
- 3.0-6.5x speedup vs vanilla autoregressive at batch size 1
- 20-40% improvement over EAGLE-2

**Batch size scaling behavior (critical insight):**
```
Batch Size 1:  ~4-6x speedup
Batch Size 4:  ~2.3x speedup
Batch Size 16: ~1.3x speedup
Batch Size 32: ~1.0x (break-even)
```
At large batch sizes, the target model becomes compute-bound, leaving no "free" bandwidth for speculation.

**Production deployment (Dec 2025):**
- EAGLE-3 on Google Cloud Vertex AI via LMSYS integration
- vLLM integration with full EAGLE-3 support
- Red Hat validated EAGLE-3 with vLLM for enterprise deployment

### 3.6 Lookahead Decoding

Lookahead decoding breaks sequential dependency using Jacobi iteration, requiring no draft model and no training.

**Algorithm:**
1. View autoregressive decoding as solving nonlinear equations: `x_t = f(x_1, ..., x_{t-1})`
2. Apply Jacobi iteration: initialize all future positions with guesses, iterate in parallel
3. Collect n-grams from Jacobi iteration trajectories
4. Verify promising n-gram candidates

**Two parallel branches per step:**
- **Lookahead branch**: Maintains a fixed 2D window, generates n-grams from Jacobi trajectories
- **Verification branch**: Selects and verifies promising n-gram candidates from the cache

**Key properties:**
- No additional model training or fine-tuning required
- Works with any autoregressive model
- Generates n-grams each step instead of single tokens
- Total steps to generate N tokens < N

**Performance:**
- 1.5-2.3x speedup with negligible computation overhead
- Lower speedup than EAGLE/Medusa but zero additional training cost
- Particularly useful for models without available draft models

**TensorRT-LLM integration:**
- NVIDIA optimized Qwen2.5-Coder with lookahead decoding in TensorRT-LLM
- Best for code generation tasks where n-gram patterns are predictable

### 3.7 Verification Algorithm: Rejection Sampling

The verification algorithm ensures that speculative decoding produces the **exact same distribution** as the target model.

**Algorithm (for single sequence):**
```
For each drafted token t_i at position i:
  p_draft = draft_model_probability(t_i)
  p_target = target_model_probability(t_i)

  acceptance_probability = min(1, p_target / p_draft)

  Draw u ~ Uniform(0, 1)
  If u < acceptance_probability:
    Accept t_i, move to t_{i+1}
  Else:
    Reject t_i
    Sample replacement from: norm(max(0, p_target - p_draft))
    Reject all subsequent tokens t_{i+1}, t_{i+2}, ...
    Break
```

**Properties:**
- Output distribution is provably identical to target model
- At worst, 1 token is generated per step (if first draft token rejected)
- At best, K+1 tokens generated (all K draft tokens accepted + 1 bonus)
- Higher temperature increases acceptance rates (distributions become more uniform)

### 3.8 Token Tree Construction and Verification

Modern speculative decoding uses **tree-structured** rather than linear candidate sequences.

**Why trees?**
- Linear: draft K tokens in sequence, verify K tokens. If token 3 is rejected, tokens 4-K are wasted.
- Tree: draft multiple alternatives at each position. If token 3a is rejected, token 3b might be accepted.

**Tree construction strategies:**
- **Static trees**: Fixed tree shape optimized offline (e.g., Sequoia)
- **Dynamic trees (DySpec)**: Explore sub-trees with higher draft probability, since they're more likely to be accepted
- **Medusa trees**: TopK candidates per head form a complete tree of width K

**Tree verification:**
- **Top-down (standard)**: Verify root first, then children of accepted nodes
- **Bottom-up (Traversal Verification, 2025)**: Verify leaf-to-root, enabling sequence-level acceptance and full utilization of drafted tokens
- **Parallel verification**: Custom attention masks enable verifying entire tree in single forward pass

**Acceptance rate optimization:**
- Match draft distribution as closely as possible to target distribution
- Use temperature-aware drafting
- Dynamically adjust tree depth based on observed acceptance rates
- Snowflake's Arctic Inference: optimized tree structures for production vLLM

### 3.9 SPEED and Other Advanced Variants

**Parallel Verification Approaches:**
- **AMUSD**: Speculates the next round during ongoing verification
- **PEARL (ICLR 2025)**: Parallel speculative decoding with adaptive draft length
- **SwiftSpec/SpecBranch**: Prepare larger cache consisting of token tree branching off the speculation being verified

**Mirror Speculative Decoding (2025):**
- Breaks the serial barrier between draft and verify phases
- Enables overlapping draft generation with verification of previous round

**Speculative Speculative Decoding (ICLR 2026):**
- Meta-level speculation on the speculation process itself
- Further amortizes overhead of the draft-verify cycle

---

## 4. Kernel Fusion Patterns for Inference

### 4.1 Why Kernel Fusion Matters

Every separate kernel launch incurs:
- **Launch overhead**: ~5-10 microseconds per kernel
- **Memory traffic**: Each kernel reads inputs from HBM and writes outputs to HBM
- **Synchronization**: Implicit barrier between consecutive kernels

For a transformer block with ~10 operations, naive execution involves 10 HBM round-trips. Fused kernels reduce this to 1-3 round-trips.

**Memory bandwidth savings example (RMSNorm + Residual):**
- Unfused: Read residual (2 bytes/elem) + Read input (2 bytes) + Write sum (2 bytes) + Read sum (2 bytes) + Write normalized (2 bytes) = 10 bytes/elem
- Fused: Read residual (2 bytes) + Read input (2 bytes) + Write normalized (2 bytes) = 6 bytes/elem
- **40% bandwidth reduction**

### 4.2 QKV Projection Fusion (3 GEMMs -> 1)

The standard transformer attention begins with three separate linear projections: Q = xW_Q, K = xW_K, V = xW_V.

**Fusion approach:**
```
# Unfused: 3 separate GEMMs
Q = x @ W_Q  # (B, S, D) x (D, D_head * num_heads)
K = x @ W_K
V = x @ W_V

# Fused: 1 GEMM with concatenated weights
W_QKV = concat(W_Q, W_K, W_V, dim=1)  # (D, 3 * D_head * num_heads)
QKV = x @ W_QKV  # Single GEMM, then split
Q, K, V = split(QKV, 3, dim=-1)
```

**Benefits:**
- 3x larger GEMM = better GPU utilization (especially at small batch sizes)
- Single kernel launch instead of 3
- Single weight read from HBM instead of 3 (if weights are fused in memory)
- Standard practice in all major inference frameworks

### 4.3 SwiGLU Fusion (up_proj + gate_proj + activation -> fused)

SwiGLU is the dominant FFN activation in modern LLMs (Llama, Mistral, etc.):
```
output = down_proj(SiLU(gate_proj(x)) * up_proj(x))
```

**Deep kernel fusion approach:**
```
# Unfused: 5 operations, multiple HBM round-trips
gate = x @ W_gate           # GEMM 1
up = x @ W_up               # GEMM 2
gate_activated = SiLU(gate)  # Elementwise
gated = gate_activated * up  # Elementwise
output = gated @ W_down      # GEMM 3

# Fused variants:
# Level 1: Fuse gate + up into single GEMM (like QKV fusion)
W_gate_up = concat(W_gate, W_up, dim=1)
gate_up = x @ W_gate_up  # Single GEMM

# Level 2: Fuse SiLU + elementwise multiply into GEMM epilogue
# gate_activated * up computed without writing intermediate to HBM

# Level 3 (DeepFusionKernel): Fuse both GEMMs + activation
# Single kernel: x -> gate_up GEMM -> SiLU -> multiply -> down GEMM -> output
```

**Performance:**
- Triton SwiGLU fusion: 1.6x speedup over unfused PyTorch
- Deep kernel fusion eliminates 2-3 intermediate HBM writes
- SwiGLU MLP blocks dominate parameter count (~67% of transformer parameters), making fusion here high-impact

### 4.4 RMSNorm + Residual Add Fusion

RMSNorm is used in all modern LLMs (replacing LayerNorm):
```
# RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight
```

**Fusion with residual:**
```
# Unfused: 2 kernels, 1 intermediate write
residual = x + residual      # Kernel 1: read x, read residual, write sum
output = RMSNorm(residual)   # Kernel 2: read sum, compute norm, write output

# Fused: 1 kernel, no intermediate write
output, residual = FusedRMSNormResidual(x, residual, weight)
# Read x + residual, compute norm in registers, write output + updated residual
```

**Measured performance:**
- PyTorch RMSNorm: 11% of peak memory bandwidth (168 GB/s on H100)
- Triton fused RMSNorm: 88% of peak bandwidth (1365 GB/s on H100)
- **8.1x speedup for standalone RMSNorm**
- **6.0x speedup for fused RMSNorm + Residual**

**Implementation details (Triton):**
- Pass 1: Accumulate squared values in GPU registers using FP32 precision
- Pass 2: Normalize inputs and apply learned weights, output in FP16/BF16
- Two-pass approach maintains numerical accuracy while avoiding temporary memory allocations

### 4.5 Attention + RoPE Fusion

Rotary Position Embeddings (RoPE) are applied to Q and K before attention:
```
# Unfused:
Q = x @ W_Q                  # GEMM
Q_rotated = apply_rope(Q)    # Elementwise rotation
K = x @ W_K                  # GEMM
K_rotated = apply_rope(K)    # Elementwise rotation
attn = attention(Q_rotated, K_rotated, V)

# Fused: RoPE applied within attention kernel
# RoPE is computed on-the-fly during Q*K^T computation
# No need to write rotated Q/K to HBM
```

**Implementations:**
- Liger Kernel: Fused RoPE implementation in Triton
- MAX (Modular): Fuses RoPE and RMSNorm in single GPU kernels with grid-level partitioning
- FlashAttention-3: RoPE can be applied within the attention kernel

### 4.6 Dequant + GEMM Fusion (for Quantized Models)

For quantized inference, weights are stored in low precision (INT4/INT8/FP8) and must be dequantized before matrix multiplication.

**Unfused approach:**
```
# Step 1: Dequantize weights (separate kernel)
W_fp16 = dequantize(W_int4, scale, zero_point)  # INT4 -> FP16
# Step 2: GEMM with dequantized weights
output = x @ W_fp16
```

**Fused approach:**
```
# Single kernel: dequantize + GEMM
# Dequantization happens in registers/shared memory during GEMM tile loading
output = fused_dequant_gemm(x, W_int4, scale, zero_point)
```

**Format-specific considerations:**

| Format | Dequant Location | Performance Impact |
|--------|-----------------|-------------------|
| W8A8 | Epilogue only (after MMA) | Fast: main loop is pure tensor core ops |
| W4A16 | Main loop (before MMA) | Moderate: dequant on CUDA cores during main loop |
| W4A8 | Main loop (CUDA cores) | Slower: dequant competes with tensor core MMA |
| W4A4 | Main loop (significant overhead) | Highest overhead in main loop |

**Performance results:**
- W4A16 SplitK fused kernel: 65% speedup on A100, 124% on H100 (peak 295%) vs blocked data parallelization
- W8A8 is the fastest quantized format because dequantization is deferred entirely to the epilogue

**Advanced: CodeGEMM (Dec 2025):**
- Replaces dequantization entirely with precomputed inner products
- Stores inner products between centroids and activations in a lightweight "Psumbook"
- Eliminates dequantization overhead completely

### 4.7 GEMM + Bias + Activation Epilogue Fusion

Epilogue fusion combines the GEMM output with subsequent operations (bias, activation) without writing the GEMM result to HBM.

**Standard epilogue pattern:**
```
# In CUTLASS/cuBLAS terminology:
# D = alpha * A @ B + beta * C    (standard GEMM)
# Extended epilogue:
# D = activation(A @ B + bias)    (fused)
```

**Common fused epilogues:**
- GEMM + bias + ReLU
- GEMM + bias + GELU
- GEMM + bias + SiLU (for SwiGLU)
- GEMM + residual add + RMSNorm
- GEMM + dequant (for quantized output)

**CUTLASS epilogue fusion:**
- CUTLASS 3.x supports custom epilogue visitors
- Can chain arbitrary elementwise operations after the GEMM
- Epilogue operations execute while GEMM tiles are still in registers/shared memory

### 4.8 Fused Softmax + Masking

In attention computation, softmax is applied after QK^T and before multiplication with V, often with a causal mask.

**Unfused:**
```
scores = Q @ K.T / sqrt(d_k)        # GEMM
masked_scores = scores + mask        # Elementwise (read/write full NxN matrix)
probs = softmax(masked_scores)       # Reduction + elementwise
output = probs @ V                   # GEMM
```

**Fused (FlashAttention approach):**
```
# All of the above in a single tiled kernel:
# 1. Compute QK^T tile
# 2. Apply mask in registers
# 3. Online softmax (no full materialization)
# 4. Multiply by V tile
# 5. Accumulate across tiles
# Never write the NxN attention matrix to HBM
```

**Memory savings:** O(N) instead of O(N^2) for sequence length N.

### 4.9 Fused Sampling (logits -> temperature -> top-p -> sample)

The final sampling step converts model logits into a selected token.

**Unfused (naive PyTorch):**
```python
logits = logits / temperature                    # Kernel 1
sorted_logits, indices = torch.sort(logits)     # Kernel 2 (expensive sort!)
cumulative_probs = torch.cumsum(softmax(sorted), -1)  # Kernel 3
mask = cumulative_probs < top_p                  # Kernel 4
filtered = sorted_logits * mask                  # Kernel 5
token = torch.multinomial(softmax(filtered), 1)  # Kernel 6
```

**Fused (FlashInfer approach):**
FlashInfer implements a **sorting-free** sampling kernel using rejection sampling:

1. **Inverse transform sampling**: Draw u ~ Uniform(0,1), compute prefix sums, find token where cumulative probability bracket contains u
2. **Rejection with pivot**: Maintain probability threshold, iteratively raise via sampling passes
3. **Dual pivot rejection**: Binary search with two pivots, guaranteeing O(log(1/epsilon)) convergence rounds

**Performance:**
- >50% reduction in sampling overhead on H100 GPUs
- Consistent latency scaling with batch size (no sort = predictable performance)
- Critical for large vocabularies (32K-128K tokens) where sorting is expensive
- Uses CUB primitives for parallel reduce and scan operations

**Practical note:** Floating-point non-associativity in parallel prefix-sum requires careful numerical stability handling to prevent invalid token generation.

---

## 5. Operator Scheduling and Graph Optimization

### 5.1 Operator Fusion at Graph Level

**TorchInductor (PyTorch 2.x):**
- Transforms models via FX graph capture
- Lowers to optimized Triton kernels
- Aggressive fusion of pointwise ops, reductions, and matrix multiplies
- Scheduling decisions: persistent vs non-persistent reductions
- Code generation with specialized kernels for exact dimensions

**TensorRT:**
- Layer and tensor fusion at the graph level
- Quantization-aware optimization
- Kernel auto-tuning for specific GPU architectures
- Supports both JIT (via torch.compile) and AOT compilation

**Torch-TensorRT:**
- Best of both worlds: PyTorch flexibility + TensorRT optimization
- `torch.compile` interface for JIT, AOT workflows for production
- Automatic fallback to PyTorch for unsupported operations

### 5.2 Constant Folding

Pre-computes operations on constant tensors at compile time.

**What gets folded:**
- Bias terms that don't change between inference calls
- RoPE frequency tensors (precomputed sin/cos tables)
- Attention masks for fixed sequence lengths
- Quantization scale/zero-point parameters
- Layer norm weights multiplication

**Benefits:**
- Eliminates runtime compute for static operations
- Enables pre-computed memory offsets
- Specialized kernels for exact dimensions (static shape optimization)
- Reduces graph complexity for further optimization passes

### 5.3 Layout Optimization

Memory layout significantly impacts kernel performance due to cache locality.

**Key layout decisions:**
- **Row-major vs column-major**: Determines which dimension is contiguous
- **NCHW vs NHWC**: NHWC (channels-last) is preferred for tensor cores on NVIDIA GPUs
- **Contiguous vs strided**: Contiguous tensors enable coalesced memory access
- **Interleaved quantization layouts**: INT4 weights may use special packed formats for efficient dequantization

**Practical impact:**
- Switching to channels-last layout can provide 10-30% speedup for convolution-heavy models
- Ensuring contiguous memory for attention QKV tensors enables efficient batched GEMM
- TensorRT automatically selects optimal layouts based on target GPU

### 5.4 Dead Code Elimination

Removes operations whose outputs are never used.

**Common in LLM inference:**
- Auxiliary training losses (not needed during inference)
- Dropout operations (identity during inference)
- Gradient computation nodes
- Unused attention head outputs (for structured pruning)

### 5.5 Common Subexpression Elimination (CSE)

Identifies and reuses duplicate computations.

**Examples in transformers:**
- RoPE sin/cos computed once and shared across Q and K
- Attention mask computed once for all heads
- Norm statistics shared across heads in multi-head attention
- Position embeddings reused across layers

---

## 6. Batching Strategies

### 6.1 Continuous Batching (Iteration-Level Scheduling)

The most important batching innovation for LLM serving. Instead of waiting for all requests in a batch to complete, new requests are inserted as slots free up.

**How it works:**
1. Maintain a pool of active batch slots
2. Each iteration: run one decode step for all active slots
3. When a request completes (EOS token), immediately replace with a waiting request
4. Prefill for new requests is interleaved with decode for existing requests

**Benefits:**
- Eliminates the "straggler problem" where short requests wait for long ones
- GPU utilization approaches 100% for steady-state workloads
- Throughput improvement of 2-10x over static batching

**Implementations:**
- vLLM: Core scheduling mechanism
- SGLang: RadixAttention + continuous batching
- TensorRT-LLM: In-flight batching
- NVIDIA Dynamo: Distributed continuous batching

### 6.2 Dynamic Batching (Request-Level)

Collects requests over a time window before processing them as a batch.

**Parameters:**
- `max_batch_size`: Maximum requests per batch
- `max_wait_time`: Maximum time to wait for batch to fill (e.g., 50ms)
- Batch is dispatched when either limit is reached

**Use cases:**
- Embedding models with fixed output length
- Classification tasks
- Any workload where all requests have similar processing time

**Comparison with continuous batching:**
- Dynamic batching: simpler, better for fixed-length outputs
- Continuous batching: better for variable-length generation

### 6.3 Micro-Batching for Pipeline Parallelism

When using pipeline parallelism, micro-batching splits the batch into smaller units that flow through pipeline stages.

**Purpose:**
- Reduces pipeline bubble (idle time between stages)
- Enables overlapping computation across stages
- Gradient accumulation across micro-batches (for training; less relevant for inference)

**For inference:**
- Micro-batch size = 1-4 requests typically
- Keeps all pipeline stages busy simultaneously
- Trade-off: more micro-batches = less bubble, but more communication overhead

### 6.4 Disaggregated Prefill/Decode Batching

The breakthrough architecture for 2025-2026 production LLM serving.

**Architecture:**
```
                    ┌──────────────┐
    Request ──────► │   Router     │
                    └──────┬───────┘
                           │
              ┌────────────┴────────────┐
              │                         │
    ┌─────────▼─────────┐   ┌──────────▼──────────┐
    │  Prefill Workers   │   │   Decode Workers     │
    │  (Compute-optimized)│──►│  (Bandwidth-optimized)│
    │  High-end GPUs     │   │  Many concurrent reqs │
    └────────────────────┘   └───────────────────────┘
              KV Cache Transfer (InfiniBand/NVLink)
```

**Why disaggregate?**
- Prefill is compute-bound: benefits from high FLOPS, small batch
- Decode is memory-bound: benefits from high bandwidth, large batch
- Mixed workloads create interference: prefill spikes cause decode latency

**KV cache transfer requirements (Llama-3.1-70B, 4K context):**
- Per-token cache: 327,680 bytes
- Total: 1.34 GB per request
- Required bandwidth for <500ms TTFT: 4.5 GB/s minimum
- 1 GbE: 10.7s (unusable) | 100 GbE: 0.43s (viable) | InfiniBand HDR: 54ms (optimal)

**Production results:**
| System | Improvement |
|--------|-------------|
| DistServe | 7.4x more requests, 12.6x better SLO |
| Splitwise | 2.35x throughput |
| Mooncake | 525% improvement with KV disaggregation |
| SGLang + DeepSeek-R1 | 52.3K input tok/s, 22.3K output tok/s on 96 H100s |

**Intra-GPU disaggregation (Nexus, 2025):**
- Extends vLLM with fine-grained SM partitioning
- Concurrent prefill-decode on the same GPU
- 2.2x lower TBT than vLLM/SGLang
- Outperforms disaggregated vLLM with half the GPU resources

**DuetServe (2025):**
- Scheduler prioritizes ongoing decode requests
- Admits prefill requests to fill remaining token budget
- Harmonizes prefill and decode within shared GPU resources

### 6.5 Priority-Based Scheduling

Not all requests are equal in production serving.

**Priority dimensions:**
- SLA tier (premium vs. free tier)
- Request age (prevent starvation)
- Expected completion time (shortest job first)
- Revenue impact (higher-paying customers first)

**Implementation approaches:**
- Multi-level feedback queue (MLFQ)
- Weighted fair queuing
- Preemptive scheduling with KV cache swapping
- vLLM supports priority-based scheduling via request metadata

### 6.6 Fairness in Multi-Tenant Serving

**Challenges:**
- Long-context requests consume disproportionate GPU memory (KV cache)
- High-throughput tenants can starve latency-sensitive tenants
- Different models on shared GPUs compete for resources

**Solutions:**
- **Token budget per tenant**: Limit total tokens in flight per tenant
- **GPU fractioning**: NVIDIA Run:ai delivers up to 2x greater user capacity via fractional GPU scheduling
- **SUN (Shared Use of Next-token prediction)**: Share a single frozen decode module across task-specific models, maintaining throughput with 50% fewer decode GPUs
- **Aegaeon**: Token-level auto-scaling with preemptive scale-down/scale-up for pending models

---

## 7. Context Length Optimization

### 7.1 Flash Attention for Long Context

FlashAttention is the foundational optimization enabling long context windows.

**Memory scaling:**
- Standard attention: O(N^2) memory for NxN attention matrix
- FlashAttention: O(N) memory via tiling and online softmax
- This difference is the reason models can process 128K+ tokens

**Practical limits:**
- FlashAttention-2 on H100: efficient up to ~128K tokens
- FlashAttention-3 on H100: optimized for Hopper, better FP8 support
- Beyond single-GPU memory: need distributed attention (Ring/Context Parallelism)

### 7.2 Ring Attention Across GPUs

Ring attention distributes long sequences across multiple GPUs in a ring topology.

**Algorithm:**
1. Partition sequence across N GPUs: each GPU holds S/N tokens
2. Each GPU computes local QK^T for its partition
3. KV blocks are passed around the ring (point-to-point communication)
4. Each GPU accumulates attention from all partitions using online softmax
5. Communication overlaps with computation

**Variants (2025-2026):**
- **Standard Ring Attention**: Simple ring topology, O(N) communication rounds
- **RingX**: Up to 3.4x speedup over conventional ring attention (SC'25, Frontier supercomputer)
- **USP (Unified Sequence Parallelism)**: Hybrid Ring + Ulysses, intra-node Ulysses for bandwidth, inter-node ring for scalability
- **LoongTrain**: 2D hybrid ring attention schemes
- **Cache-DiT v1.2.1 (Feb 2026)**: Ring Attention with batched P2P, USP, hybrid 2D/3D parallelism

**When to use ring attention:**
- Context length exceeds single GPU memory for KV cache
- 1M+ token contexts requiring 32+ GPUs
- Co-located GPUs with high-bandwidth interconnect

### 7.3 Sparse Attention Patterns for Long Context

For very long contexts, full attention is unnecessary -- most tokens attend primarily to nearby tokens and a few globally important ones.

**Patterns:**
- **Sliding window**: Each token attends to W nearest neighbors only
- **Global + local**: Selected global tokens attend to everything; rest use sliding window
- **Lambda-shaped (attention sinks + window)**: Keep first K tokens (sink tokens) + sliding window
- **Learned sparsity**: Model learns which tokens to attend to (training-time)
- **Native sparse attention**: Training-free approaches that identify important tokens at inference time

**KV cache savings from sparse attention:**
- Full attention at 128K context: all 128K KV entries per layer
- Sliding window (W=4096): only 4096 KV entries per layer = **32x reduction**
- Sink + window: ~4K-8K entries = **16-32x reduction**

### 7.4 KV Cache Compression for Long Context

**Approaches and compression ratios:**

| Method | Compression | Technique | Accuracy Impact |
|--------|-------------|-----------|-----------------|
| FP8 quantization | 2x | Reduce precision | <1% |
| NVFP4 quantization | 4x | 4-bit KV cache | <1% on Blackwell |
| Token pruning (BUZZ) | 2.5x | Evict low-attention tokens | Minimal |
| Latent space (SALS) | 6.4x | Compress to latent space | Moderate |
| RocketKV | 400x | Aggressive eviction | Task-dependent |

**SALS (Sparse Attention in Latent Space):**
- Compresses KV cache to latent space
- Selects critical tokens with much less computation
- Reconstructs only a small subset of important tokens
- 5.7x attention speedup vs FlashAttention-2 on 4K sequences
- 4.5x end-to-end throughput improvement on 32K sequences

**KVCompose (2025):**
- Uses composite tokens to represent compressed KV cache
- Structured compression maintaining attention quality

### 7.5 Sliding Window + Sink Tokens

The "StreamingLLM" pattern enables infinite-length generation with bounded memory.

**Architecture:**
```
KV Cache Layout:
[Sink Tokens (4-64)] [... gap ...] [Sliding Window (2048-8192)]
     ^                                     ^
     |                                     |
   Always retained                   Recent tokens only
   (attention sinks)                 (rolling buffer)
```

**Why sink tokens matter:**
- First few tokens accumulate disproportionately high attention scores
- Removing them causes catastrophic quality degradation
- Keeping just 4-64 sink tokens + sliding window maintains quality for most tasks

**Implementation:**
- Mistral models use native sliding window attention (W=4096)
- vLLM supports configurable sliding window sizes
- BUZZ: segments new token arrivals into cells, selects per-cell heavy-hitters, preserves sink and window regions

### 7.6 Context Parallelism

Context parallelism distributes the sequence dimension across GPUs (distinct from tensor parallelism which distributes model dimensions).

**Meta's implementation (2025):**
- **Pass-KV**: Input tokens split across CP ranks; KV tensors exchanged for full-context attention
- **Pass-Q**: Query tensors exchanged between ranks (alternative)

**Performance results (Meta):**
- 1M tokens in <1 minute on single H100 host (8 GPUs)
- 10M tokens in <1 minute across 32 H100 hosts
- Llama 3 405B: 128K prefill in 3.8 seconds (16 nodes), 1M prefill in 77 seconds
- Near-linear scaling demonstrated

**When to use:**
- Context parallelism for long sequences (>32K tokens)
- Tensor parallelism for large model layers
- Combine both for large models with long contexts

---

## 8. Multi-GPU Inference Optimization

### 8.1 Tensor Parallelism: When and How

**What it does:** Splits individual weight matrices across GPUs. Each GPU holds a slice of every layer.

**Communication pattern:** AllReduce after every layer (2x data volume per layer in ring-based AllReduce).

**When to use:**
- Model doesn't fit on single GPU
- Low-latency requirements (single request)
- High-bandwidth interconnect available (NVLink essential)

**Practical TP degrees:**
| Interconnect | Max Practical TP | Communication Overhead |
|-------------|------------------|----------------------|
| NVLink 4.0 (900 GB/s) | 8 | 10-30 us/layer |
| NVLink 3.0 (600 GB/s) | 8 | 15-50 us/layer |
| PCIe 5.0 (128 GB/s) | 2-4 | 100-500 us/layer |
| InfiniBand HDR (200 Gb/s) | 4-8 (cross-node) | 50-200 us/layer |

**Meta's DDA (Direct Data Access) optimization:**
- Flat DDA: Each rank directly loads from other ranks' memory, local reduce
- Tree DDA: Split AllReduce into reduce-scatter + all-gather with direct access
- **Results**: 10-50% speedup for decode (small messages), 10-30% for prefill
- ~10% reduction in TTIT (Time to Incremental Token)

### 8.2 Pipeline Parallelism: Micro-Batch Scheduling

**What it does:** Distributes consecutive groups of layers across GPUs. Data flows sequentially through stages.

**When to use:**
- Cross-node deployment with slow interconnect
- Model too large for single node
- Throughput more important than single-request latency

**Micro-batch scheduling for inference:**
- Split batch into micro-batches (e.g., 4 micro-batches of 8 requests)
- Pipeline fill time = (num_stages - 1) * micro-batch_time
- Steady-state: all stages active simultaneously
- Pipeline bubble = fill_time / total_time (minimize by increasing micro-batches)

**Key limitation:** Pipeline parallelism increases per-request latency because of inter-stage communication. Each pipeline stage adds one communication round.

### 8.3 Expert Parallelism for MoE

**What it does:** Distributes expert weights across GPUs. Each GPU holds a subset of experts.

**Communication pattern:** All-to-All exchange of tokens to/from experts based on routing.

**MoE inference challenges:**
- All-to-all communication: 10-30% of end-to-end latency for decode
- Token routing imbalance: some experts receive more tokens than others
- Expert activation sparsity: only top-K experts active per token

**Optimization approaches:**
- **Dynamic All-to-All**: Sub-chunks sent to remote neighbors for better overlap
- **Persistent All-to-All**: Addresses memory-handle exchange overhead and CPU overhead
- **Wide Expert Parallelism**: Spread experts across more GPUs for better parallelism

**KTransformers (SOSP 2025):**
- Hybrid CPU/GPU inference for MoE models
- Shared experts on GPU, routed experts on CPU
- Lock-free queue for CPU expert tasks
- Background worker threads execute tasks in parallel
- Up to 7.5x higher throughput on T4 GPUs vs GPU-only

**vLLM MoE Playbook (2025):**
- Wrong parallelism choice can cause 8x KV cache memory duplication
- Or communication overhead that cuts throughput in half
- Optimal: EP within nodes, DP across nodes for MoE models

### 8.4 Sequence Parallelism for Long Sequences

Distributes the sequence dimension across GPUs while keeping model parameters replicated.

**Differs from context parallelism:**
- Context parallelism: distributes attention computation
- Sequence parallelism: distributes non-attention operations (LayerNorm, Dropout, etc.) that are independent across the sequence dimension

**When to use:**
- Long sequences with large activation memory
- Combined with tensor parallelism for comprehensive memory savings
- Reduces activation memory from O(S * H) to O(S/P * H) where P = parallelism degree

### 8.5 Optimal Parallelism Strategy Selection

**Decision framework:**

```
Single GPU sufficient?
├─ Yes: No parallelism needed
└─ No:
    ├─ Single node (NVLink)?
    │   ├─ Dense model: TP = num_GPUs (up to 8)
    │   └─ MoE model: EP across GPUs, TP=1 or TP=2
    └─ Multi-node?
        ├─ Fast interconnect (IB/NVLink)?
        │   ├─ Dense: TP within node, PP across nodes
        │   └─ MoE: EP within node, DP across nodes
        └─ Slow interconnect?
            ├─ Dense: PP across nodes, TP=2-4 within node
            └─ MoE: DP across nodes, local EP
```

**Rule of thumb:**
- TP within nodes, PP across nodes when interconnect is slow
- If NVLink/InfiniBand: TP can extend across nodes
- MoE models: EP is almost always better than replicating experts
- Always benchmark: optimal strategy depends on model size, batch size, and hardware

### 8.6 Communication-Computation Overlap

The key to efficient multi-GPU inference is hiding communication latency behind computation.

**Techniques:**
- **Prefetch next layer's weights** while computing current layer (pipeline)
- **Overlap AllReduce with independent computation** in next layer's prep
- **Double buffering**: Compute on buffer A while communicating buffer B
- **NCCL streams**: Use separate CUDA streams for communication and computation

**Meta's approach:**
- All-to-all for MoE overlapped with expert computation
- DDA algorithms enable lower-latency communication that's easier to overlap
- Persistent kernels maintain GPU occupancy during communication

---

## 9. System-Level Optimization

### 9.1 CPU-GPU Pipeline Optimization

The CPU is often an overlooked bottleneck in LLM serving.

**CPU-bound operations:**
- Tokenization and detokenization
- Request scheduling and queue management
- KV cache management (block table updates)
- Sampling (if not GPU-fused)
- Network I/O and request parsing

**NEO (MLSys 2025):**
- Offloads part of attention compute and KV cache to host CPU
- Asymmetric GPU-CPU pipelining with load-aware scheduling
- Up to 7.5x higher throughput on T4 GPUs vs GPU-only
- Frees GPU memory for larger batch sizes

**Optimization strategies:**
- Pin CPU memory for faster DMA transfers
- Use multiple CPU threads for tokenization pipeline
- Async I/O for request handling (don't block GPU pipeline)
- Pre-allocate tensors to avoid allocation overhead

### 9.2 Tokenizer Optimization

Tokenization can become a bottleneck at high throughput.

**Optimizations:**
- **Rust-based tokenizers** (Hugging Face `tokenizers` library): 10-100x faster than Python
- **Batch tokenization**: Process multiple requests simultaneously
- **Pre-tokenization caching**: Cache tokenized common prefixes
- **Parallel tokenization**: Use multiple CPU cores via multiprocessing
- **Vocabulary size impact**: Larger vocabularies (128K+) increase lookup time

**Practical tip:** Always use the `tokenizers` library (Rust-based) instead of the Python `transformers` tokenizer. The difference is 100-1000 tokens/ms vs 1-10 tokens/ms.

### 9.3 Network I/O Optimization for Serving

**Considerations:**
- **gRPC vs HTTP**: gRPC with protobuf is 2-5x more efficient than JSON over HTTP for token streaming
- **Streaming responses**: Use server-sent events (SSE) or gRPC streaming for token-by-token output
- **Connection pooling**: Reuse connections to reduce TCP handshake overhead
- **Request batching at network level**: Aggregate small requests before dispatching to GPU

**NVIDIA Dynamo networking (2025):**
- Low Latency Communication Library for KV cache transfer
- Accelerates GPU-to-GPU transfer across heterogeneous memory/storage
- Smart routing to minimize KV cache recomputation

### 9.4 Request Scheduling Algorithms

**Algorithms in production:**
- **FCFS (First Come First Served)**: Simple, fair, but suboptimal for throughput
- **Shortest Job First (SJF)**: Minimize average latency, but requires output length prediction
- **Preemptive scheduling**: Pause long-running requests to serve short ones, swap KV cache to CPU
- **Token-budget scheduling**: Limit total tokens processed per iteration
- **SLA-aware scheduling**: Prioritize requests approaching SLA deadline

**vLLM scheduler:**
- Iteration-level scheduling with continuous batching
- Preemption via KV cache swapping (to CPU) or recomputation
- Priority queues for multi-tier serving
- Chunked prefill interleaving

### 9.5 Auto-Scaling Strategies

**Reactive scaling:**
- Scale based on real-time metrics (queue depth, GPU utilization, latency)
- Simple but prone to over/under-provisioning due to LLM loading delays (30s-5min)

**Predictive scaling (SageServe, 2025):**
- Forecast-aware auto-scaling using workload prediction
- Accounts for TPS variance and LLM loading delays
- Adapts to dynamic workloads while meeting diverse SLAs

**Token-level auto-scaling (Aegaeon, SOSP 2025):**
- Preemptively scales down active models and scales up pending models
- Token-level granularity for better SLO satisfaction
- Manages successive scale-downs and scale-ups with KV cache persistence

**GPU fractioning (NVIDIA Run:ai):**
- Fractional GPU scheduling for multi-model serving
- Up to 2x greater user capacity during peak periods on existing hardware
- Foundational capability for large-scale, multi-model inference

### 9.6 GPU Clock and Power Management

**Considerations:**
- **GPU boost clocks**: Modern GPUs have variable clock speeds based on thermal/power limits
- **Power capping**: Setting lower TDP reduces peak performance but improves efficiency (tokens/watt)
- **Persistence mode**: `nvidia-smi -pm 1` keeps GPU initialized, eliminating cold-start latency
- **Clock locking**: `nvidia-smi -lgc <min>,<max>` for consistent benchmarking
- **MIG (Multi-Instance GPU)**: Partition A100/H100 into isolated instances for multi-tenant serving

**Thermal considerations:**
- Sustained inference loads can cause thermal throttling
- Monitor `nvidia-smi -q -d TEMPERATURE` during load testing
- Ensure adequate cooling, especially in dense deployments (8x GPU servers)

---

## 10. Benchmarking Inference Performance

### 10.1 Core Metrics

**TTFT (Time to First Token):**
- Measures user-perceived responsiveness
- Includes: network latency + queue wait + prefill computation
- Dominated by prefill time for long prompts
- Target: <500ms for interactive use, <2s for batch processing
- MLPerf 2025 target: TTFT <= 500ms for interactive scenario

**TPOT (Time Per Output Token):**
- Average time between consecutive generated tokens (excluding TTFT)
- Measures streaming speed
- Lower TPOT = faster token stream
- Target: <30ms for interactive use (MLPerf 2025), <100ms for batch
- **Warning**: Different tools calculate TPOT differently (some include TTFT, some don't)

**ITL (Inter-Token Latency):**
- Per-token latency, not averaged -- captures variability
- Important for detecting latency spikes (e.g., from prefill interference)
- Report p50, p95, p99 percentiles
- p99 ITL is often the binding SLA constraint

**Throughput (tokens/sec):**
- Total tokens generated per second across all requests
- Distinguishes between input throughput and output throughput
- MLPerf uses tokens/s as standard unit for cross-platform comparison

**RPS (Requests Per Second):**
- Completed requests per second
- Less informative than tokens/s due to variable request lengths
- Useful for capacity planning

### 10.2 How to Measure Correctly

**Warm-up:**
- Always discard first 10-50 requests to allow for:
  - CUDA graph capture
  - JIT compilation (torch.compile)
  - KV cache pool allocation
  - NCCL initialization (multi-GPU)
- Run warm-up until metrics stabilize (typically 30-60 seconds)

**Percentile reporting:**
- Never report only mean/average -- it hides tail latency
- Report: p50, p90, p95, p99, p99.9
- p99 TTFT and p99 ITL are the most important SLA metrics
- Calculate percentiles over at least 1000 requests for statistical significance

**Steady-state measurement:**
- Measure under sustained load, not burst
- Run for at least 5 minutes after warm-up
- Monitor for thermal throttling (GPU temps >80C can cause slowdowns)
- Record GPU utilization, memory usage, and power draw alongside latency

**Common pitfalls:**
- Measuring cold start as representative latency
- Not accounting for request queuing time in TTFT
- Using mean instead of percentiles
- Comparing different input/output length distributions
- Not controlling for GPU clock speeds (boost vs sustained)

### 10.3 Load Testing Tools

**genAI-perf (NVIDIA):**
- Official NVIDIA benchmarking tool
- Supports Triton, vLLM, TensorRT-LLM backends
- Reports TTFT, TPOT, ITL with percentiles
- **Note**: Does not include TTFT in TPOT calculation

**LLMPerf:**
- Open-source, supports multiple providers
- **Note**: Includes TTFT in average token latency (different from genAI-perf)

**locust:**
- General-purpose load testing framework
- Python-based, easy to customize
- Good for simulating realistic traffic patterns
- Supports distributed load generation

**wrk / wrk2:**
- High-performance HTTP benchmarking
- wrk2 provides constant-rate load generation (better for SLA testing)
- Lightweight, minimal client-side overhead

**Custom harnesses:**
- For precise measurement, custom harnesses give most control
- Use async HTTP clients (aiohttp, httpx)
- Track per-token timestamps from SSE streams
- Calculate metrics from raw timestamps, not framework-reported values

**vLLM built-in benchmark:**
```bash
python -m vllm.entrypoints.openai.api_server --model <model>
# Then use the benchmark client:
python benchmarks/benchmark_serving.py \
  --model <model> \
  --dataset-name sharegpt \
  --request-rate 10
```

### 10.4 Comparison Methodology

**Fair A/B testing requirements:**
- Same hardware, same GPU model, same driver version
- Same input dataset (standardized: ShareGPT, LMSYS-Chat, or synthetic)
- Same input/output length distribution
- Same quantization level
- Same batch size or request rate
- Same warm-up procedure
- Control for GPU power state and thermal conditions

**Standardized benchmarks:**
- **MLPerf Inference**: Industry standard, rigorous methodology
  - Llama3.1-8B interactive scenario: TTFT <= 500ms, TPOT <= 30ms
  - Offline scenario: maximize throughput
- **LLM Arena**: Focus on quality, not performance
- **Custom**: Use ShareGPT traces for realistic serving benchmarks

### 10.5 SLA-Aware Benchmarking

**Define SLAs first, then benchmark:**
```
Example SLA:
- p99 TTFT < 500ms
- p99 ITL < 50ms
- p99 E2E latency < 10s for 500-token outputs
- Minimum throughput: 100 requests/minute
```

**Goodput measurement:**
- Goodput = requests that meet ALL SLA constraints / total time
- More meaningful than raw throughput for production systems
- DistServe optimizes for goodput, achieving 4.48x improvement

**Load curve testing:**
- Sweep request rates from low to saturation
- Plot latency percentiles vs. request rate
- Identify the "knee" where latency starts degrading
- Operating point should be left of the knee with margin

---

## 11. Cost Optimization

### 11.1 Tokens Per Dollar Analysis

**Current market (2025-2026):**

| Tier | Model Example | Input ($/1M tokens) | Output ($/1M tokens) |
|------|--------------|---------------------|---------------------|
| Budget | Gemini Flash-Lite | $0.075 | $0.30 |
| Budget | Llama 3.2 3B (self-hosted) | ~$0.01 | ~$0.04 |
| Mid | DeepSeek R1 | $0.55 | $2.19 |
| Mid | Claude Sonnet 4 | $3.00 | $15.00 |
| Premium | GPT-4 class | $2.50-$10.00 | $10.00-$30.00 |

**Price decline trajectory:** 10x annual decline. GPT-4 equivalent: $20/M tokens (2022) -> $0.40/M tokens (2025).

**Provider price variation:** For identical models, prices range 10x between cheapest and most expensive providers ($0.90 to $9.50 per million tokens for the same model).

### 11.2 Smaller Quantized Model vs Larger Model Tradeoff

**Decision framework:**
```
Quality requirement met by quantized smaller model?
├─ Yes: Use quantized smaller model (massive cost savings)
│   Example: Llama-3.1-8B-AWQ (4-bit) on single L4 GPU
│   Cost: ~$0.50/hour, ~$0.02/M tokens
└─ No: Use larger model with quantization
    Example: Llama-3.1-70B-GPTQ (4-bit) on 2x H100
    Cost: ~$7/hour, ~$0.30/M tokens
```

**Quantization cost impact:**
- 8-bit: 50% memory reduction, ~1% quality loss, 2x batch size
- 4-bit: 75% memory reduction, competitive quality, 4x batch size
- Combined with batching: 4-8x effective cost reduction per request
- Organizations report 60-70% operational cost reduction from quantization alone

**When to use larger models:**
- Complex reasoning tasks where quality difference is measurable
- Tasks with high error cost (medical, legal, financial)
- When the larger model enables substantially fewer retries/iterations

### 11.3 Spot Instance Utilization

**Current spot pricing (2025):**
- AWS H100: ~$3.90/hr on-demand, ~$1.50-2.50/hr spot (35-60% savings)
- Google A3-High (H100): ~$3.00/hr on-demand, ~$2.25/hr spot
- Google TPU v6e committed use: $0.39/chip-hour

**Strategies for spot instances:**
- **Stateless serving**: No persistent state, easy to restart on new instance
- **KV cache checkpointing**: Periodically save KV cache to persistent storage
- **Multi-region failover**: Spread across regions for spot availability
- **Hybrid spot + on-demand**: Base capacity on-demand, burst on spot
- **Preemption handling**: Gracefully drain requests on preemption notice (typically 2 min warning)

**Best practices:**
- Use spot for batch/offline inference (tolerant of interruptions)
- Use on-demand or reserved for latency-sensitive serving
- Mix: 60% spot + 40% on-demand for good cost/reliability balance

### 11.4 Multi-Model Serving on Single GPU

**Approaches:**
- **GPU fractioning (NVIDIA Run:ai)**: Up to 2x user capacity via fractional scheduling
- **Time-sharing**: Round-robin between models, swap weights in/out
- **LoRA serving**: Keep base model loaded, swap lightweight LoRA adapters per request
- **SUN architecture**: Shared frozen decode module across task-specific models, 50% fewer decode GPUs

**LoRA serving optimization:**
- Base model weights: loaded once, shared across all adapters
- LoRA weights: 0.1-1% of base model size, swap in ~1ms
- vLLM supports multi-LoRA serving with continuous batching
- Enable serving 10-100 "different models" on a single GPU

**MIG (Multi-Instance GPU):**
- A100/H100 can be partitioned into up to 7 instances
- Each instance is isolated (memory, compute, cache)
- Good for serving multiple small models
- Trade-off: each instance gets proportionally less memory and compute

### 11.5 Right-Sizing GPU Selection

**GPU selection guide by model size:**

| Model Size | Quantization | Minimum GPU | Recommended GPU | Approx. Cost/hr |
|-----------|-------------|-------------|-----------------|-----------------|
| 1-3B | FP16 | T4 (16GB) | L4 (24GB) | $0.50-$1.00 |
| 7-8B | FP16 | A10 (24GB) | L40S (48GB) | $1.00-$2.50 |
| 7-8B | INT4/AWQ | T4 (16GB) | L4 (24GB) | $0.50-$1.00 |
| 13B | FP16 | A100-40GB | A100-80GB | $2.00-$4.00 |
| 13B | INT4/AWQ | L4 (24GB) | L40S (48GB) | $1.00-$2.50 |
| 70B | FP16 | 2x A100-80GB | 2x H100 | $6.00-$8.00 |
| 70B | INT4/AWQ | A100-80GB | H100 (80GB) | $3.00-$4.00 |
| 405B | FP16 | 8x H100 | 8x H200 | $24.00-$32.00 |
| 405B | INT4/FP8 | 4x H100 | 4x H200 | $12.00-$16.00 |

**Key insight:** Many inference workloads perform well on L4 or A10 GPUs instead of expensive H100s. Always benchmark your specific workload on cheaper GPUs first.

**Self-hosting break-even analysis:**
- 7B model: ~50% utilization needed to beat GPT-3.5 Turbo pricing
- 13B model: ~10% utilization needed to beat GPT-4-turbo pricing
- Minimum scale: >8,000 conversations/day before self-hosted beats managed APIs
- Organization applying quantization (4x) + continuous batching (2x) + speculative decoding (2x) = **16x effective cost reduction** vs naive deployment

**Alternative accelerators:**
- Google TPU v6e: $0.39/chip-hour (committed use), significant savings for large-scale
- Organizations migrating NVIDIA -> TPU report 66% cost reduction ($2.1M -> $700K/month)
- AMD MI300X: Achieving performance parity with H100 at lower price points (as demonstrated by Meta's DDA optimizations)

---

## Appendix A: Reference Architecture for Production LLM Serving (2025-2026)

```
                         ┌─────────────┐
                         │  Load       │
                         │  Balancer   │
                         └──────┬──────┘
                                │
                    ┌───────────▼───────────┐
                    │  NVIDIA Dynamo /       │
                    │  Smart Router          │
                    │  (KV-cache aware)      │
                    └───────────┬───────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                   │
    ┌─────────▼──────┐ ┌───────▼────────┐ ┌───────▼────────┐
    │ Prefill Pool   │ │ Decode Pool    │ │ Decode Pool    │
    │ (H100/H200)    │ │ (H100/H200)    │ │ (H100/H200)    │
    │ TP=4, BS=1-4   │ │ TP=2, BS=32-128│ │ TP=2, BS=32-128│
    │ Chunked prefill│ │ CUDA graphs    │ │ CUDA graphs    │
    │ FlashAttn-3    │ │ FP8 KV cache   │ │ FP8 KV cache   │
    │ torch.compile  │ │ Continuous batch│ │ Continuous batch│
    └────────┬───────┘ └───────▲────────┘ └───────▲────────┘
             │                 │                   │
             └─────────────────┘                   │
              KV Cache Transfer                    │
              (InfiniBand/NVLink)                   │
                                                   │
                                    ┌──────────────┘
                                    │
                         ┌──────────▼──────────┐
                         │  Auto-Scaler        │
                         │  (Token-level,      │
                         │   forecast-aware)    │
                         └─────────────────────┘
```

**Stack:**
- Serving framework: vLLM or SGLang with NVIDIA Dynamo
- Attention: FlashAttention-3 (prefill) + FlashInfer/TRTLLM (decode)
- Quantization: FP8 weights + FP8/NVFP4 KV cache
- Batching: Continuous batching + chunked prefill
- Decode acceleration: CUDA graphs + speculative decoding (EAGLE-3 at low batch)
- Memory management: PagedAttention with <4% waste
- Sampling: FlashInfer fused sorting-free sampling kernels
- Multi-GPU: TP within node, PP across nodes (dense) or EP (MoE)
- Scaling: Token-level auto-scaling with KV-cache-aware routing

---

## Appendix B: Quick Reference - Optimization Impact Summary

| Optimization | Typical Speedup | Memory Impact | Implementation Complexity |
|-------------|----------------|---------------|--------------------------|
| FlashAttention-3 | 1.5-2x attention | 10-100x less attention memory | Use library (easy) |
| CUDA graphs (decode) | 10-30% | None | Framework support (easy) |
| FP8 KV cache | 1.2-1.5x decode | 50% KV reduction | Flag in vLLM (easy) |
| NVFP4 KV cache | 1.5-2x decode | 75% KV reduction | Blackwell only (easy) |
| Continuous batching | 2-10x throughput | Higher peak memory | Framework support (easy) |
| Chunked prefill | Better ITL p99 | Minimal | Flag in vLLM (easy) |
| Speculative decoding (EAGLE-3) | 3-6.5x at BS=1 | +2-5% for draft head | Moderate |
| Kernel fusion (RMSNorm) | 6-8x per op | None | Use Triton/Liger (moderate) |
| Kernel fusion (SwiGLU) | 1.6x per op | None | Use Triton/Liger (moderate) |
| Fused sampling | 50%+ reduction | None | Use FlashInfer (easy) |
| Disaggregated prefill/decode | 2-7x goodput | Needs 2 GPU pools | Architecture change (hard) |
| torch.compile max-autotune | 10-30% overall | None | One line of code (easy) |
| Quantization (W4) | 2-4x throughput | 75% weight reduction | Moderate |
| PagedAttention | 2-4x throughput | <4% waste | Use vLLM (easy) |
| Tensor parallelism | Linear scaling | Split across GPUs | Framework support (easy) |
| Whole-model kernel (FlashFormer) | 8-61% at BS=1 | None | Research stage (hard) |

---

## Appendix C: Sources and References

### Primary Sources
- [NVIDIA Mastering LLM Techniques](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)
- [NVIDIA Dynamo Framework](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)
- [NVIDIA NVFP4 KV Cache](https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache)
- [NVIDIA Speculative Decoding Introduction](https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/)
- [Meta Scaling LLM Inference](https://engineering.fb.com/2025/10/17/ai-research/scaling-llm-inference-innovations-tensor-parallelism-context-parallelism-expert-parallelism/)
- [vLLM Distributed Inference](https://blog.vllm.ai/2025/02/17/distributed-inference.html)
- [vLLM Large Scale Serving: DeepSeek](https://blog.vllm.ai/2025/12/17/large-scale-serving.html)
- [vLLM Speculative Decoding](https://blog.vllm.ai/2024/10/17/spec-decode.html)
- [vLLM MoE Playbook (AMD ROCm)](https://rocm.blogs.amd.com/software-tools-optimization/vllm-moe-guide/README.html)

### Speculative Decoding
- [EAGLE-3 (NeurIPS 2025)](https://arxiv.org/html/2503.01840v1)
- [EAGLE-3 on Vertex AI (LMSYS)](https://lmsys.org/blog/2025-12-01-eagle3-vertex/)
- [Medusa Paper](https://arxiv.org/abs/2401.10774)
- [LayerSkip (Meta)](https://ai.meta.com/research/publications/layerskip-enabling-early-exit-inference-and-self-speculative-decoding/)
- [Lookahead Decoding (LMSYS)](https://lmsys.org/blog/2023-11-21-lookahead-decoding/)
- [Snowflake Arctic Inference](https://www.snowflake.com/en/engineering-blog/fast-speculative-decoding-vllm-arctic/)
- [Traversal Verification](https://arxiv.org/abs/2505.12398)
- [Speculative Speculative Decoding (ICLR 2026)](https://openreview.net/pdf?id=aL1Wnml9Ef)

### Kernel Fusion and Optimization
- [FlashFormer: Whole-Model Kernels](https://arxiv.org/html/2505.22758v1)
- [FlashInfer Sorting-Free Sampling](https://flashinfer.ai/2025/03/10/sampling.html)
- [Triton Kernels for LLM Inference](https://subhadipmitra.com/blog/2025/triton-kernels-llm-inference/)
- [Liger Kernel (LinkedIn)](https://github.com/linkedin/Liger-Kernel)
- [W4A16 Fused Kernel with SplitK](https://arxiv.org/abs/2402.00025)
- [LiquidGEMM W4A8](https://arxiv.org/html/2509.01229v1)
- [Deep Kernel Fusion for Transformers](https://arxiv.org/html/2602.11808)

### Serving and Batching
- [DistServe Disaggregated Serving](https://haoailab.com/blogs/distserve/)
- [Disaggregated Prefill-Decode (JarvisLabs)](https://docs.jarvislabs.ai/blog/llm-optimization-disaggregated-prefill-decode)
- [NEO CPU Offloading](https://yangzhou1997.github.io/paper/neo_mlsys25.pdf)
- [SageServe Auto-Scaling](https://arxiv.org/html/2502.14617v3)
- [Aegaeon GPU Pooling (SOSP 2025)](https://ennanzhai.github.io/pub/sosp25-aegaeon.pdf)
- [KTransformers CPU/GPU Hybrid (SOSP 2025)](https://dl.acm.org/doi/pdf/10.1145/3731569.3764843)

### KV Cache and Context
- [SALS: Sparse Attention in Latent Space](https://arxiv.org/html/2510.24273)
- [KV Cache Compression Survey](https://github.com/October2001/Awesome-KV-Cache-Compression)
- [Long-Context Attention Benchmark](https://arxiv.org/html/2510.17896v1)
- [RingX Scalable Parallel Attention (SC'25)](https://dl.acm.org/doi/10.1145/3712285.3759859)

### Benchmarking and Metrics
- [BentoML Inference Metrics](https://bentoml.com/llm/inference-optimization/llm-inference-metrics)
- [NVIDIA Benchmarking Fundamentals](https://developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts/)
- [MLPerf Inference 5.1](https://mlcommons.org/2025/09/small-llm-inference-5-1/)
- [NVIDIA NIM Benchmarking Metrics](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html)

### Cost Analysis
- [Inference Unit Economics (Introl)](https://introl.com/blog/inference-unit-economics-true-cost-per-million-tokens-guide)
- [Cost Per Token Analysis (Introl)](https://introl.com/blog/cost-per-token-llm-inference-optimization)
- [NVIDIA Blackwell Cost Reduction](https://blogs.nvidia.com/blog/inference-open-source-models-blackwell-reduce-cost-per-token/)
- [GPU Cloud Cost Comparison](https://www.fluence.network/blog/best-gpu-for-llm/)
