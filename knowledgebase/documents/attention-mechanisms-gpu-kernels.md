# Attention Mechanisms and GPU Kernel Implementations

Comprehensive reference covering attention algorithms, their GPU kernel implementations, complexity analysis, and performance characteristics for LLM inference optimization.

---

## 1. FlashAttention (v1, v2)

### Algorithm Overview

FlashAttention is an IO-aware exact attention algorithm that computes mathematically identical results to standard scaled dot-product attention while minimizing memory reads/writes between GPU HBM (High Bandwidth Memory) and on-chip SRAM.

**Standard Attention** computes:
```
S = Q @ K^T          # [N, N] score matrix
P = softmax(S)       # [N, N] probability matrix
O = P @ V            # [N, d] output
```

This requires materializing the full N x N attention matrix in HBM, consuming O(N^2) memory and O(N^2) HBM accesses.

### The Online Softmax Trick

The key innovation enabling tiled attention is the online (incremental) softmax algorithm. Standard softmax requires a full pass to compute the denominator. The online approach maintains running statistics:

```
For each new block j of scores S_ij:
  m_new = max(m_old, rowmax(S_ij))           # running maximum
  l_new = e^(m_old - m_new) * l_old + rowsum(e^(S_ij - m_new))  # running sum
  O_new = (l_old * e^(m_old - m_new) * O_old + e^(S_ij - m_new) @ V_j) / l_new
```

This allows computing exact softmax incrementally across tiles without ever materializing the full N x N matrix.

### Tiling Strategy

The algorithm partitions Q, K, V into blocks that fit in SRAM:
- Block size B_r (rows/queries): typically 128
- Block size B_c (columns/keys): typically 64
- Constrained by SRAM size M: B_r * d + B_c * d + B_r * B_c + B_r * d <= M

### Forward Pass Pseudocode

```python
# FlashAttention-2 Forward Pass (simplified)
# Outer loop: iterate over blocks of Q (parallelized across SMs)
for i in range(0, N, B_r):
    Q_i = load_from_HBM(Q[i:i+B_r])     # Load query block to SRAM
    O_i = zeros(B_r, d)
    m_i = -inf(B_r)                       # Running max per row
    l_i = zeros(B_r)                      # Running sum per row

    # Inner loop: iterate over blocks of K, V
    for j in range(0, N, B_c):
        K_j = load_from_HBM(K[j:j+B_c])  # Load key block to SRAM
        V_j = load_from_HBM(V[j:j+B_c])  # Load value block to SRAM

        # All computation below happens in SRAM
        S_ij = Q_i @ K_j.T / sqrt(d)     # [B_r, B_c] scores
        m_new = max(m_i, rowmax(S_ij))
        P_ij = exp(S_ij - m_new)          # Safe softmax numerator
        l_new = exp(m_i - m_new) * l_i + rowsum(P_ij)

        # Rescale previous output and accumulate
        O_i = diag(exp(m_i - m_new)) @ O_i + P_ij @ V_j
        m_i = m_new
        l_i = l_new

    O_i = diag(1/l_i) @ O_i              # Final normalization
    store_to_HBM(O[i:i+B_r], O_i)
    store_to_HBM(LSE[i:i+B_r], m_i + log(l_i))  # For backward pass
```

### Backward Pass - Recomputation

Instead of storing the O(N^2) attention matrix for backpropagation, FlashAttention stores only:
- Output O (N x d)
- Log-sum-exp statistics LSE = m + log(l) (N x 1)

During the backward pass, S and P are recomputed from Q, K, V blocks in SRAM, trading ~25% extra FLOPs for massive memory savings.

### FlashAttention v1 vs v2 Differences

| Aspect | FlashAttention v1 | FlashAttention v2 |
|--------|-------------------|-------------------|
| **Outer loop** | Over K,V blocks (columns) | Over Q blocks (rows) |
| **Inner loop** | Over Q blocks | Over K,V blocks |
| **Parallelism** | batch * heads | batch * heads * N/B_r |
| **Non-matmul FLOPs** | Higher (rescaling in inner loop) | Reduced by keeping O,m,l in registers |
| **Occupancy** | Lower (less parallelism) | Higher (more blocks = more SMs utilized) |
| **Speedup** | Baseline | ~2x over v1 |

FlashAttention-2's key insight: swapping loop order enables parallelism over the sequence dimension, dramatically improving GPU occupancy. Each thread block processes one query block against all K,V blocks, keeping accumulators in fast registers.

### Complexity Analysis

| Metric | Standard Attention | FlashAttention |
|--------|-------------------|----------------|
| **HBM accesses** | Theta(Nd + N^2) | O(N^2 * d^2 / M) |
| **Memory** | O(N^2) | O(N) |
| **FLOPs** | O(N^2 * d) | O(N^2 * d) (same, exact) |

Where M = SRAM size (~100-200 KB per SM), d = head dimension (64-128). For typical values, d^2/M << 1, yielding 4-16x fewer HBM accesses.

**IO Complexity Proof Sketch**: Each block of Q (size B_r x d) is loaded once. For each Q block, all K,V blocks are loaded (N/B_c blocks, each B_c x d). Total HBM reads: (N/B_r) * (N/B_c) * B_c * d * 2 + (N/B_r) * B_r * d = O(N^2 * d / B_r + Nd). With B_r = Theta(M/d), this gives O(N^2 * d^2 / M).

### Performance Characteristics

- **A100 (FP16)**: ~124 TFLOPs/s (62% utilization) for head_dim=64
- **Wall-clock speedup**: 2-4x over standard PyTorch attention
- **Memory reduction**: 5-20x depending on sequence length
- **Exact computation**: No approximation, numerically equivalent to standard attention

Sources:
- [FlashAttention Paper (arXiv:2205.14135)](https://arxiv.org/pdf/2205.14135)
- [FlashAttention-2 Paper (arXiv:2307.08691)](https://arxiv.org/pdf/2307.08691)
- [Online Softmax to FlashAttention (UW CSE599m)](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf)
- [Understanding FlashAttention in Triton](https://towardsdatascience.com/understanding-flash-attention-writing-the-algorithm-from-scratch-in-triton-5609f0b143ea/)

---

## 2. FlashAttention-3 (Hopper Architecture)

### Architecture-Specific Optimizations

FlashAttention-3 exploits three hardware features unique to NVIDIA Hopper (H100/H200):

1. **Asynchronous TMA (Tensor Memory Accelerator)** for overlapping data movement with computation
2. **WGMMA (Warp Group Matrix Multiply-Accumulate)** for high-throughput tensor core operations
3. **FP8 Tensor Cores** for 2x throughput over FP16/BF16

### Warp Specialization

Warps within a CTA (Cooperative Thread Array) are divided into specialized roles:

```
CTA Layout:
  Producer Warpgroup (1 warp):
    - Issues TMA loads from HBM to shared memory
    - Minimal register usage (only 1 thread needed for TMA)
    - Uses setmaxnreg to donate registers to consumers

  Consumer Warpgroup 1:
    - Performs WGMMA operations (Q@K^T, P@V)
    - Claims extra registers from producer
    - Alternates with Consumer Warpgroup 2

  Consumer Warpgroup 2:
    - Same as Consumer 1 but phase-shifted
    - Executes softmax while CWG1 runs GEMMs
```

The `setmaxnreg` instruction dynamically reallocates registers between warpgroups, allowing compute-heavy consumers to use more registers than memory-bound producers.

### Ping-Pong Scheduling

Two consumer warpgroups alternate in a ping-pong pattern:

```
Timeline:
  CWG1: [GEMM1: S=QK^T] [------softmax------] [GEMM2: O=PV] [------softmax------]
  CWG2: [------softmax------] [GEMM1: S=QK^T] [------softmax------] [GEMM2: O=PV]
  TMA:  [load K_1,V_1] [load K_2,V_2] [load K_3,V_3] [load K_4,V_4] ...
```

This overlaps softmax (~3.9 TFLOPs/s throughput) with WGMMA (~989 TFLOPs/s), hiding the softmax bottleneck. Without ping-pong, softmax would stall the pipeline between GEMMs.

### Intra-Warpgroup GEMM-Softmax Pipelining

Within a single warpgroup, a two-stage software pipeline overlaps dependent operations using register double-buffering:

```python
# Two register buffers: S_cur, S_next
S_cur = WGMMA(Q, K_0)           # First GEMM
for j in range(1, num_blocks):
    # Stage 1: Start next GEMM while processing current softmax
    S_next = WGMMA_async(Q, K_j)  # Non-blocking GEMM start
    P_cur = softmax(S_cur)         # Process previous GEMM result

    # Stage 2: Start output GEMM while next score GEMM completes
    O += WGMMA_async(P_cur, V_{j-1})
    WGMMA_wait(S_next)            # Ensure S_next is ready
    S_cur = S_next
```

### FP8 Support with Accuracy Preservation

**Block Quantization**: Instead of per-tensor FP8 scaling, each B_r x d or B_c x d block has its own scale factor. This naturally fits FlashAttention's blocked computation and reduces outlier impact.

**Incoherent Processing**: Multiply Q and K by a random orthogonal matrix M before quantization:
```
Q_fp8 = quantize_fp8(Q @ M)
K_fp8 = quantize_fp8(K @ M)
# Since M @ M^T = I, QK^T is preserved exactly
# But each entry of Q@M is a random sum of Q entries, spreading outliers
```

M is constructed as: M = D1 @ H @ D2 where D1, D2 are random +/-1 diagonal matrices and H is the Hadamard matrix (computable in O(d log d)).

**Layout Considerations**: FP8 WGMMA requires k-major (reduction-dimension contiguous) layout. Solutions:
- In-kernel V transpose using LDSM/STSM instructions
- Register permutation via byte-permute to match accumulator layouts between consecutive GEMMs

### Persistent Kernels

FlashAttention-3 supports persistent kernel mode where CTAs remain resident on SMs and process multiple tiles from a work queue, reducing kernel launch overhead and improving SM utilization for variable-length sequences.

### Performance Numbers

| Configuration | Throughput | Utilization |
|--------------|-----------|-------------|
| H100 FP16 forward | 740 TFLOPs/s | 75% |
| H100 FP8 forward | ~1.2 PFLOPs/s | ~61% |
| vs FlashAttention-2 | 1.5-2.0x speedup | - |
| FP8 quantization error | 2.6x lower than naive FP8 | - |

**Ablation**: Warp specialization alone yields 570 TFLOPs/s; adding GEMM-softmax pipelining reaches 661 TFLOPs/s; full system achieves 740 TFLOPs/s.

Sources:
- [FlashAttention-3 Paper (arXiv:2407.08608)](https://arxiv.org/html/2407.08608v2)
- [FlashAttention-3 Blog (Tri Dao)](https://tridao.me/blog/2024/flash3/)
- [Together AI Blog](https://www.together.ai/blog/flashattention-3)
- [PyTorch Blog](https://pytorch.org/blog/flashattention-3/)

---

## 3. FlashInfer

### Architecture Overview

FlashInfer is a customizable attention engine for LLM serving that separates compile-time kernel specialization from runtime dynamic scheduling. It supports all common LLM serving patterns: prefill, decode, and append operations across paged, ragged, and padded KV cache formats.

### Composable Block-Sparse Format

FlashInfer unifies diverse KV cache layouts using block-sparse row (BSR) format:

```
KV Cache Layouts (all represented as BSR):
  - Paged Attention: Fixed-size blocks mapped via page tables
  - RadixAttention: Tree-structured prefix sharing
  - Ragged Tensors: Variable-length sequences packed contiguously
  - Padded Tensors: Fixed-length with padding (least efficient)
```

**Composable Formats**: Rather than forcing a single block size, FlashInfer decomposes KV cache into multiple sparse submatrices with different block dimensions:
- Shared-prefix regions: Large B_r blocks for shared memory bandwidth
- Unique per-request regions: Small blocks (potentially 1x1) to minimize padding
- Attention on larger blocks accesses shared KV cache via fast shared memory/registers

### Memory Access Pattern

```
Global Memory -> Shared Memory Pipeline:
  1. BSR indices compute scattered global memory addresses
  2. Asynchronous copies (LDGSTS, 128B width) gather into contiguous SRAM
  3. Tensor cores operate on contiguous shared memory data

For Hopper GPUs:
  - Contiguous KV layouts use TMA acceleration
  - Non-affine (scattered) patterns fall back to async copies
```

### JIT Compilation Pipeline

FlashInfer uses a JIT-based code generation approach:

```python
# User defines attention variant via functors:
class MyAttentionVariant:
    QueryTransform = RotaryEmbedding   # Fused RoPE
    KeyTransform = RotaryEmbedding
    LogitsTransform = SoftCapping(50)  # Gemma-style soft capping
    LogitsMask = CausalMask | SlidingWindowMask(4096)
    OutputTransform = Identity

# JIT compilation flow:
# 1. Functor substitution into CUDA template skeletons
# 2. Tile size selection based on workload analysis
# 3. PyTorch JIT compiler generates optimized CUDA
# 4. Compiled kernel cached and registered as custom op
```

Supports both softmax and non-softmax attention variants, with fusion of normalization, RoPE, and linear projections.

### Adaptive Tile Size Selection

FlashInfer implements multiple kernel variants with tile configurations:
- Query tiles: {1, 16, 32, 64, 128}
- KV tiles: {32, 64, 128}

Selection heuristic:
1. Calculate average query length per batch
2. Choose minimum query tile >= average length
3. Solve register/shared memory constraints to maximize SM occupancy

This contrasts with FlashAttention-2's fixed (128, 64) tiles, which waste resources for short decode sequences.

### Cascade Attention

For shared-prefix batch serving (e.g., system prompts):

```
Algorithm:
  Stage 1: Multi-Query Attention on shared prefix KV cache
    - Load shared KV into GPU shared memory (SMEM)
    - Multiple queries access same KV via fast SMEM
    - Uses prefill kernel for parallel query processing

  Stage 2: Batch Decode Attention on unique suffixes
    - Each request has distinct KV cache
    - Standard decode kernel (global memory / L2 cache)

  Stage 3: Attention State Merging
    - Merge partial results using associative operator:
    - [O_merged, LSE_merged] = [O_prefix, LSE_prefix] OP [O_suffix, LSE_suffix]
    - OP defined as: weighted combination using exp(LSE_a - LSE_max) scaling
```

The attention state composition operator is associative and commutative:
```
State = (O, LSE)  where LSE = log(sum(exp(scores)))
merge(S1, S2):
  LSE_new = log(exp(LSE_1) + exp(LSE_2))
  O_new = (exp(LSE_1 - LSE_new) * O_1 + exp(LSE_2 - LSE_new) * O_2)
```

**Performance**: Up to 31x speedup over baseline vLLM PageAttention for 32K shared prefix, batch size 256.

### Load-Balanced Dynamic Scheduling

Stream-K inspired work distribution:

```python
# Cost model per work unit:
cost(lq, lkv) = alpha * lq + beta * lkv

# KV chunk sizing:
Lkv_max = sum(ceil(lq[i]/Tq) * lkv[i]) / num_CTAs

# Greedy assignment via priority queue:
for chunk in sorted_by_kv_length(work_units):
    assign_to_least_loaded_CTA(chunk)
```

Generates deterministic aggregation order, crucial for numerical reproducibility. Compatible with CUDAGraph capture via persistent kernels with fixed grid sizes.

### Performance

- End-to-end: 29-69% inter-token-latency improvement over Triton backend
- Long-context: 28-30% reduction with fused RoPE kernels
- Parallel generation: 13-17% speedup with composable formats

Sources:
- [FlashInfer Paper (arXiv:2501.01005)](https://arxiv.org/html/2501.01005v1)
- [FlashInfer GitHub](https://github.com/flashinfer-ai/flashinfer)
- [Cascade Inference Blog](https://flashinfer.ai/2024/02/02/cascade-inference.html)
- [NVIDIA Technical Blog on FlashInfer](https://developer.nvidia.com/blog/run-high-performance-llm-inference-kernels-from-nvidia-using-flashinfer)
- [Dissecting FlashInfer](https://ydnyshhh.github.io/posts/flash_infer/)

---

## 4. PagedAttention (vLLM)

### Core Concept

PagedAttention applies operating system virtual memory concepts to KV cache management. KV cache is partitioned into fixed-size blocks (pages) that can be stored non-contiguously in GPU memory, eliminating fragmentation from contiguous allocation.

### Block Table Structure

```
Logical View (per sequence):
  [Block 0] [Block 1] [Block 2] ... [Block N]
  Each block stores keys/values for B consecutive tokens
  Typical block size: B = 16 tokens

Block Table (per sequence):
  logical_block_id -> (physical_block_id, num_filled_positions)

  Example for sequence with 35 tokens, block_size=16:
    Logical 0 -> Physical 7  (16/16 filled)
    Logical 1 -> Physical 3  (16/16 filled)
    Logical 2 -> Physical 12 (3/16 filled)   # last block partially filled
```

Physical blocks are allocated on-demand as tokens are generated, avoiding pre-allocation waste.

### Memory Layout

```
Physical Block Memory (for one layer, one head):
  Keys:   [block_size, head_dim] = [16, 128] = 4 KB (FP16)
  Values: [block_size, head_dim] = [16, 128] = 4 KB (FP16)
  Total per block per layer per head: 8 KB

  For LLaMA-13B (40 layers, 40 heads):
    Per block: 8 KB * 40 * 40 = 12.8 MB
    Per token: 12.8 MB / 16 = 800 KB
```

### Attention Kernel Access Pattern

```python
# PagedAttention kernel (simplified)
def paged_attention(query, key_cache, value_cache, block_table, context_len):
    output = zeros(head_dim)
    max_score = -inf
    exp_sum = 0

    num_blocks = ceil(context_len / block_size)
    for block_idx in range(num_blocks):
        # Look up physical block from block table
        physical_block = block_table[block_idx]

        # Load keys/values from non-contiguous physical location
        keys = key_cache[physical_block]    # [block_size, head_dim]
        values = value_cache[physical_block] # [block_size, head_dim]

        # Standard attention computation within block
        scores = query @ keys.T / sqrt(head_dim)
        # Online softmax update across blocks
        block_max = max(scores)
        new_max = max(max_score, block_max)
        exp_sum = exp_sum * exp(max_score - new_max) + sum(exp(scores - new_max))
        output = output * exp(max_score - new_max) + exp(scores - new_max) @ values
        max_score = new_max

    output = output / exp_sum
    return output
```

### Copy-on-Write (CoW)

For parallel sampling and beam search, multiple sequences share KV cache blocks:

```
Parallel Sampling (3 completions from same prompt):
  Sequence 1: [Shared Block 0] [Shared Block 1] [Unique Block A]
  Sequence 2: [Shared Block 0] [Shared Block 1] [Unique Block B]
  Sequence 3: [Shared Block 0] [Shared Block 1] [Unique Block C]

  Reference counts: Block 0 = 3, Block 1 = 3, Block A/B/C = 1

Copy-on-Write triggered when:
  - A sequence needs to modify a shared block (ref_count > 1)
  - System allocates new physical block
  - Copies data from original block
  - Updates block table for that sequence only
```

### Memory Efficiency

| Metric | Traditional | PagedAttention |
|--------|-------------|---------------|
| Internal fragmentation | 60-80% waste | < 4% waste |
| Parallel sampling memory | 3x for 3 samples | ~1x (shared blocks) |
| Beam search memory | Up to k*x | Shared early prefixes |
| Memory savings (beam search) | Baseline | Up to 55% reduction |
| Throughput improvement | Baseline | Up to 2.2x |

Sources:
- [vLLM Blog Post](https://blog.vllm.ai/2023/06/20/vllm.html)
- [PagedAttention Paper (arXiv:2309.06180)](https://arxiv.org/pdf/2309.06180)
- [vLLM Paged Attention Docs](https://docs.vllm.ai/en/stable/design/paged_attention/)

---

## 5. Ring Attention

### Algorithm

Ring Attention distributes attention computation across N GPUs arranged in a ring topology, enabling near-infinite context lengths by scaling memory linearly with GPU count.

### How It Works

```
Setup: N GPUs, sequence length L
  - Q, K, V split into N chunks of size L/N
  - GPU_i holds Q_i, K_i, V_i initially

Algorithm:
  for step in range(N):
    # Each GPU computes partial attention with current K,V
    O_i_partial = flash_attention(Q_i, K_current, V_current)

    # Merge partial result with running attention state
    O_i = merge_attention_states(O_i, O_i_partial)

    # Ring communication: send K,V to next GPU, receive from previous
    K_current = ring_send_recv(K_current)  # GPU_i sends to GPU_(i+1)
    V_current = ring_send_recv(V_current)  # GPU_i receives from GPU_(i-1)

  # After N steps, each GPU has seen all K,V blocks
```

### Key Properties

- **Memory per GPU**: O(L/N) for KV cache instead of O(L)
- **Communication**: Overlapped with computation for sufficiently long sequences
- **Correctness**: Uses the same online softmax merging as FlashAttention
- **Causal masking**: Handled per-block; some blocks are entirely masked

### Striped Attention Improvement

Standard Ring Attention has load imbalance with causal masking (later GPUs have more masked-out blocks). Striped Attention solves this:

```
Standard assignment:  GPU_0=[tok 0..L/N-1], GPU_1=[tok L/N..2L/N-1], ...
Striped assignment:   GPU_0=[tok 0,N,2N,...], GPU_1=[tok 1,N+1,2N+1,...], ...
```

Each GPU's tokens are distributed uniformly throughout the sequence, ensuring ~equal compute per step regardless of causal mask position.

### Context Parallel (PyTorch)

PyTorch provides `context_parallel()` context manager that automatically replaces SDPA with Ring Attention:

```python
with context_parallel(
    buffers=[Q, K, V],
    buffer_seq_dims=[2, 2, 2],  # sequence dimension
    process_group=cp_group,
):
    output = F.scaled_dot_product_attention(Q, K, V)
```

### Performance Considerations

- Communication cost: O(L * d / N) per step, N-1 steps = O(L * d * (N-1)/N)
- Compute per step: O((L/N)^2 * d) per GPU
- Communication hidden when: compute_time > communication_time
- Minimum sequence length for efficiency: L >> N * sqrt(bandwidth/compute_flops)

Sources:
- [Ring Attention Paper (arXiv:2310.01889)](https://arxiv.org/pdf/2310.01889)
- [Striped Attention (arXiv:2311.09431)](https://arxiv.org/pdf/2311.09431)
- [PyTorch Context Parallel Tutorial](https://docs.pytorch.org/tutorials/unstable/context_parallel.html)
- [Ring Attention Blog (Akasa)](https://akasa.com/blog/ring-attention)

---

## 6. Multi-Query Attention (MQA) and Grouped-Query Attention (GQA)

### Architecture Comparison

```
Multi-Head Attention (MHA):
  Q: [batch, seq, num_heads, head_dim]     # e.g., 32 heads
  K: [batch, seq, num_heads, head_dim]     # 32 KV heads
  V: [batch, seq, num_heads, head_dim]     # 32 KV heads
  KV cache per token per layer: 2 * 32 * 128 * 2 bytes = 16 KB

Multi-Query Attention (MQA):
  Q: [batch, seq, num_heads, head_dim]     # 32 query heads
  K: [batch, seq, 1, head_dim]            # 1 KV head (shared)
  V: [batch, seq, 1, head_dim]            # 1 KV head (shared)
  KV cache per token per layer: 2 * 1 * 128 * 2 bytes = 512 bytes
  Memory reduction: 32x

Grouped-Query Attention (GQA):
  Q: [batch, seq, num_heads, head_dim]     # 32 query heads
  K: [batch, seq, num_kv_heads, head_dim]  # e.g., 8 KV heads
  V: [batch, seq, num_kv_heads, head_dim]  # 8 KV heads
  Group size g = num_heads / num_kv_heads = 32/8 = 4
  KV cache per token per layer: 2 * 8 * 128 * 2 bytes = 4 KB
  Memory reduction: 4x
```

### Kernel Implementation Differences

**MHA Kernel**: Each thread block processes one query head against its corresponding KV head. Straightforward parallelism across heads.

**MQA Kernel**: Multiple query heads share the same KV head. The kernel must broadcast K,V to all query heads within a group:
```python
# MQA: K,V loaded once, reused across all query heads
kv = load_kv(kv_head_idx=0)  # Single KV head
for q_head in range(num_query_heads):
    output[q_head] = attention(Q[q_head], kv.K, kv.V)
```

**GQA Kernel**: Generalization where g query heads share each KV head group:
```python
for kv_group in range(num_kv_heads):
    kv = load_kv(kv_head_idx=kv_group)
    for q_head in range(kv_group * g, (kv_group + 1) * g):
        output[q_head] = attention(Q[q_head], kv.K, kv.V)
```

### XQA Optimization (TensorRT-LLM)

TensorRT-LLM implements XQA, a specialized kernel for MQA/GQA decode that:
- Fuses the KV broadcast with attention computation
- Uses heuristics to choose between XQA and masked MHA kernels
- Optimizes memory access patterns for the many-queries-to-few-KV-heads ratio

### Performance Impact

| Aspect | MHA | GQA (g=4) | MQA |
|--------|-----|-----------|-----|
| KV cache size | 1x | 1/g | 1/num_heads |
| Memory bandwidth (decode) | 1x | ~1/g | ~1/num_heads |
| Quality | Best | ~MHA | Slight degradation |
| Decode throughput | 1x | ~g*x | ~num_heads*x |

### Adoption

GQA is the default in: LLaMA 2/3, Mistral, Mixtral, Gemma, Qwen, PaLM. MQA used in: PaLM (original), Falcon, StarCoder.

Sources:
- [GQA Paper (arXiv:2305.13245)](https://arxiv.org/abs/2305.13245)
- [TensorRT-LLM GPT Attention](https://nvidia.github.io/TensorRT-LLM/advanced/gpt-attention.html)
- [IBM GQA Overview](https://www.ibm.com/think/topics/grouped-query-attention)

---

## 7. Sliding Window Attention (SWA)

### Algorithm

Each token attends only to the previous W tokens (window size), reducing per-layer complexity from O(N^2) to O(N * W).

```
Standard causal attention mask (N=8):
  1 0 0 0 0 0 0 0
  1 1 0 0 0 0 0 0
  1 1 1 0 0 0 0 0
  1 1 1 1 0 0 0 0
  1 1 1 1 1 0 0 0
  1 1 1 1 1 1 0 0
  1 1 1 1 1 1 1 0
  1 1 1 1 1 1 1 1

Sliding window attention mask (W=3):
  1 0 0 0 0 0 0 0
  1 1 0 0 0 0 0 0
  1 1 1 0 0 0 0 0
  0 1 1 1 0 0 0 0
  0 0 1 1 1 0 0 0
  0 0 0 1 1 1 0 0
  0 0 0 0 1 1 1 0
  0 0 0 0 0 1 1 1
```

### Effective Context Through Stacking

With L transformer layers and window W, the effective receptive field is L * W tokens:
- Token i at layer k can attend to tokens [i-W, i] at layer k
- Layer k-1 already incorporated information from [i-2W, i]
- After L layers: effective context = L * W

Mistral-7B: W=4096, L=32, effective context = 131,072 tokens.

### Rolling Buffer KV Cache

```python
# Rolling buffer: only store last W tokens
cache_size = window_size  # e.g., 4096
# Position wraps around using modulo:
cache_position = current_position % cache_size

# Write new KV:
kv_cache[cache_position] = new_kv

# Read: access positions [current - W, current]
# Naturally handled by modulo indexing
```

This saves 50% cache memory compared to full attention for sequences longer than 2*W.

### Kernel Implementation

FlashAttention and xFormers support SWA with modified masking:
```python
# In FlashAttention kernel, during score computation:
for j in range(0, N, B_c):
    # Skip blocks entirely outside the window
    if j > i + B_r or j < i - window_size:
        continue  # No computation needed
    S_ij = Q_i @ K_j.T
    # Apply window mask within block
    mask = (positions_i[:, None] - positions_j[None, :]) <= window_size
    S_ij = where(mask, S_ij, -inf)
```

**Performance**: 2x speed improvement for seq_len=16K with window=4K compared to full attention, due to skipping entire tile blocks outside the window.

Sources:
- [Mistral 7B Announcement](https://mistral.ai/news/announcing-mistral-7b/)
- [Sliding Window Attention Overview](https://www.abhik.xyz/concepts/attention/sliding-window-attention)

---

## 8. Cross Attention

### Architecture

Cross attention connects encoder and decoder in sequence-to-sequence models. Unlike self-attention where Q, K, V come from the same sequence:

```
Self-Attention:  Q, K, V all from decoder hidden states
Cross-Attention: Q from decoder, K and V from encoder output

Decoder Layer:
  1. Masked Self-Attention(decoder_states)
  2. Cross-Attention(Q=decoder_states, KV=encoder_output)
  3. Feed-Forward Network
```

### Kernel Considerations

Cross attention has distinct optimization characteristics:

```
Key differences from self-attention:
  - K, V are fixed once encoder runs (cached across all decode steps)
  - Q length varies (1 during autoregressive decode, full during prefill)
  - No causal mask needed (decoder can attend to all encoder positions)
  - K, V sequence length is typically different from Q length

Optimization opportunities:
  - Encoder KV cache is static: compute once, reuse across all decode steps
  - No causal mask simplifies kernel (no mask checking overhead)
  - Different Q and KV lengths require flexible tiling strategies
```

### FlashAttention Support

FlashAttention supports cross attention natively since Q and KV can have different sequence lengths. The same tiling strategy applies, with the inner loop iterating over encoder KV blocks.

### Multimodal Cross Attention

Used in vision-language models (e.g., Flamingo, LLaVA) where:
- K, V come from image encoder output (fixed-length, e.g., 256 patches)
- Q comes from language model (variable-length text)
- Additional considerations: image KV is much shorter than typical text KV

Sources:
- [Cross-Attention Mechanism (GeeksforGeeks)](https://www.geeksforgeeks.org/nlp/cross-attention-mechanism-in-transformers/)
- [BetterTransformer (PyTorch)](https://pytorch.org/blog/a-better-transformer-for-fast-transformer-encoder-inference/)

---

## 9. Linear Attention and Alternatives

### Linear Attention Framework

Standard softmax attention: O = softmax(QK^T / sqrt(d)) @ V, complexity O(N^2 d)

Linear attention replaces softmax with kernel feature maps phi:
```
Standard: A = softmax(Q @ K^T)
Linear:   A = phi(Q) @ phi(K)^T

# Rearranging using associativity:
O = phi(Q) @ (phi(K)^T @ V)   # O(N d^2) instead of O(N^2 d)
```

Since d << N for long sequences, this is dramatically faster.

### Lightning Attention-2

The first linear attention implementation achieving theoretical computational benefits in practice.

**Core Problem**: Causal linear attention requires cumulative summation (cumsum), which prevents parallelization and hardware efficiency.

**Solution**: Tiled computation separating intra-block and inter-block components:

```python
# Lightning Attention-2 Algorithm
# Intra-block: use standard attention (quadratic but small)
# Inter-block: use linear attention kernel trick (accumulated state)

state = zeros(d, d)  # Running state for inter-block
for block_i in range(num_blocks):
    Q_i, K_i, V_i = load_block(block_i)

    # Intra-block: standard attention within tile (on SRAM)
    intra_output = standard_attention(Q_i, K_i, V_i)  # O(B^2 * d)

    # Inter-block: linear attention with accumulated state
    inter_output = Q_i @ state  # O(B * d^2)

    # Combine outputs
    O_i = intra_output + inter_output

    # Update state for next block
    state += K_i^T @ V_i  # O(B * d^2)
```

**IO-Aware Implementation**: Built in Triton with explicit SRAM tiling, maximizing on-chip computation. Achieves constant speed regardless of sequence length.

### RWKV

RWKV (Receptance Weighted Key Value) uses a linear attention approximation called the WKV mechanism:

```
# RWKV attention (simplified)
# Time-mixing with exponential decay:
wkv_t = sum_{i=1}^{t-1} e^{-(t-1-i)*w + k_i} * v_i + e^{u+k_t} * v_t
        / (sum_{i=1}^{t-1} e^{-(t-1-i)*w + k_i} + e^{u+k_t})

# w: learned decay rate per channel
# u: learned bonus for current token
# Computable as RNN: O(d) per step, O(Nd) total
```

RWKV is conceptualized as the ratio of two state space models. Recurrent formulation enables O(1) memory per step during inference.

### Mamba / Selective State Space Models

Mamba addresses SSM limitations with input-dependent (selective) parameters:

```
# Standard SSM (Linear Time-Invariant):
h_t = A * h_{t-1} + B * x_t
y_t = C * h_t

# Mamba (Selective SSM):
B_t, C_t, Delta_t = f(x_t)    # Input-dependent parameters
A_bar_t = exp(Delta_t * A)     # Discretized transition
B_bar_t = Delta_t * B_t
h_t = A_bar_t * h_{t-1} + B_bar_t * x_t
y_t = C_t * h_t
```

**Hardware-Aware Algorithm**: Mamba uses a parallel scan for training (O(N) with O(log N) depth) and sequential scan for inference (O(1) per step). The kernel avoids materializing the full state in HBM by computing in SRAM.

**Mamba-2**: Simplifies to structured state space duality (SSD), connecting SSMs to structured matrix multiplication, enabling more efficient GPU kernels.

### Gated DeltaNet

Used in Qwen3-Next as a linear attention alternative:
- Combines delta rule (associative memory update) with gating
- Achieves competitive quality with softmax attention
- Linear complexity with constant memory during inference

Sources:
- [Lightning Attention-2 (arXiv:2401.04658)](https://arxiv.org/abs/2401.04658)
- [Mamba Paper (arXiv:2312.00752)](https://arxiv.org/pdf/2312.00752)
- [Linear Attention Fundamentals](https://haileyschoelkopf.github.io/blog/2024/linear-attn/)
- [Beyond Standard LLMs (Sebastian Raschka)](https://magazine.sebastianraschka.com/p/beyond-standard-llms)

---

## 10. Sparse Attention

### BigBird

BigBird combines three sparse attention patterns to achieve O(N) complexity while maintaining theoretical guarantees (universal approximation, Turing completeness):

```
Attention Pattern for token i:
  1. Global tokens (g tokens): Attend to/from all positions
     - First g tokens are global (e.g., [CLS], [SEP])

  2. Sliding window (w tokens): Local context
     - Attend to [i - w/2, i + w/2]

  3. Random tokens (r tokens): Long-range connections
     - r randomly selected positions per query

Combined mask per row: |active| = g + w + r << N
```

**Block-Sparse GPU Implementation**: BigBird uses b x b blocks as basic computation units (typically b=64), defining the attention as block-sparse matrix multiplication compatible with GPU tensor cores.

### Longformer

Similar to BigBird but with additional dilated sliding window patterns:
```
Longformer patterns:
  - Local sliding window (different sizes per layer)
  - Dilated sliding window (skip every k-th position)
  - Global attention on task-specific tokens
  - Block-sparse variant (BSLongformer): blocks of tokens instead of individual tokens
```

### DeepSpeed Sparse Attention

DeepSpeed provides a flexible sparse attention library supporting:
- Fixed patterns (local, strided)
- Variable patterns (configurable per head)
- BigBird and Longformer patterns
- Custom user-defined patterns

### GPU Implementation Strategy

```python
# Block-sparse attention using Triton
# Define sparse pattern as block mask
block_mask = compute_pattern(seq_len, block_size=64)  # [N/64, N/64] binary

# Only compute attention for non-zero blocks
for (block_row, block_col) in nonzero_blocks(block_mask):
    Q_block = Q[block_row * 64 : (block_row+1) * 64]
    K_block = K[block_col * 64 : (block_col+1) * 64]
    V_block = V[block_col * 64 : (block_col+1) * 64]
    partial_output = attention(Q_block, K_block, V_block)
    accumulate(output, block_row, partial_output)
```

FlashAttention supports block-sparse patterns efficiently by simply skipping tile computations for masked-out blocks.

Sources:
- [BigBird Paper (arXiv:2007.14062)](https://arxiv.org/abs/2007.14062)
- [Understanding BigBird (HuggingFace)](https://huggingface.co/blog/big-bird)
- [DeepSpeed Sparse Attention](https://www.deepspeed.ai/tutorials/sparse-attention/)
- [Sparse Flash Attention (arXiv:2306.01160)](https://ar5iv.labs.arxiv.org/html/2306.01160)

---

## 11. KV Cache Optimization

### Quantized KV Cache

#### FP8 KV Cache
```python
# FP8 quantization for KV cache
# 2x memory reduction vs FP16, minimal quality impact
# Supported by vLLM, TensorRT-LLM, FlashInfer

# Per-tensor quantization:
scale = max(abs(tensor)) / max_fp8_value
tensor_fp8 = round(tensor / scale).to(fp8_e4m3)

# Dequantization during attention:
tensor_fp16 = tensor_fp8.to(fp16) * scale
```

#### INT4/INT8 KV Cache (KIVI)

KIVI: Tuning-free asymmetric 2-bit KV cache quantization:
```
Key insight: Keys and Values have different distribution patterns
  - Keys: Outliers appear in specific channels (per-channel quantization)
  - Values: Outliers appear in specific tokens (per-token quantization)

Algorithm:
  1. Quantize Key cache per-channel (along head_dim axis)
  2. Quantize Value cache per-token (along sequence axis)
  3. Use asymmetric quantization (different zero-point per group)
  4. Group size: 128 for keys, per-token for values

Memory: 2-bit KV cache = 8x reduction vs FP16
Quality: Maintains >99% of FP16 accuracy on most benchmarks
```

### KV Cache Compression Methods

#### H2O (Heavy-Hitter Oracle)
```
Strategy: Retain only tokens with highest cumulative attention scores
Algorithm:
  1. Track attention score history for each token
  2. Identify "heavy hitter" tokens (high cumulative attention)
  3. Evict tokens with lowest accumulated attention scores
  4. Always retain recent tokens (sliding window)

Cache composition: [heavy_hitter_tokens] + [recent_window_tokens]
Compression: Retains 20% of tokens while maintaining quality
```

#### SnapKV
```
Strategy: Select important KV based on attention patterns in observation window
Algorithm:
  1. Use last few tokens as "observation window"
  2. Compute attention scores from observation tokens to all previous tokens
  3. Select top-k tokens per head based on pooled attention scores
  4. Compress KV cache to selected tokens + recent window

Key insight: Attention patterns in recent tokens predict future importance
```

#### XKV
```
Strategy: Cross-layer KV cache sharing and compression
  - Identify redundant KV pairs across layers
  - Share KV cache between similar layers
  - Compress per-layer KV based on layer importance scores
```

### Streaming / Sink Tokens (StreamingLLM)

```
Problem: LLMs fail when KV cache exceeds training context length
Observation: "Attention sinks" - initial tokens receive disproportionate
  attention regardless of semantic content (artifact of softmax normalization)

StreamingLLM Cache Layout:
  [Sink tokens (4)] + [... evicted ...] + [Recent window (last W tokens)]

  sink_size = 4         # First 4 tokens always retained
  window_size = W       # Recent tokens retained
  total_cache = 4 + W   # Much smaller than full context

Implementation:
  - Maintain circular buffer of size W for recent tokens
  - Pin first 4 token KVs permanently
  - Evict intermediate tokens as new tokens arrive

Result: Stable generation up to 4M+ tokens with fixed cache size
Limitation: Enables infinite generation, NOT infinite context understanding
```

### KITTY (Recent, 2025)

Combines sink tokens with sliding window for value cache quantization:
- Sink tokens (initial) and Local tokens (recent) retained in FP16
- Middle tokens quantized to per-token INT4
- Achieves near-lossless 2-bit value cache quantization

### MiniKV (2024-2025)

Achieves 86% KV cache compression while retaining accuracy:
- Enables 44K token prompts with 48% higher throughput on A100
- Combines importance-based selection with extreme quantization (2-bit)

Sources:
- [KIVI Paper (arXiv:2402.02750)](https://arxiv.org/html/2402.02750v2)
- [vLLM Quantized KV Cache Docs](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/)
- [HuggingFace KV Cache Quantization Blog](https://huggingface.co/blog/kv-cache-quantization)
- [StreamingLLM (arXiv:2309.17453)](https://arxiv.org/abs/2309.17453)
- [MiniKV (arXiv:2411.18077)](https://arxiv.org/html/2411.18077v3)

---

## 12. Attention Variants for Inference

### Prefill Attention (Compute-Bound)

```
Characteristics:
  - Process entire prompt at once: Q=[seq_len, d], K=[seq_len, d], V=[seq_len, d]
  - Large matrix multiplications: O(N^2 * d) FLOPs
  - Compute-bound: arithmetic intensity = N * d / (memory_bytes)
  - Parallelizable across sequence dimension
  - Uses FlashAttention with large tile sizes (128x64 or larger)

Optimization strategies:
  - Maximize tensor core utilization
  - Use largest possible tile sizes
  - Enable torch.compile with max-autotune for operator fusion
  - FP8 compute (2x throughput on Hopper)
```

### Decode Attention (Memory-Bound)

```
Characteristics:
  - Single new token: Q=[1, d], K=[context_len, d], V=[context_len, d]
  - Dominated by KV cache loading from HBM
  - Memory-bound: must load entire KV cache for one output vector
  - Arithmetic intensity: 2 * d / (2 * context_len * d * sizeof(dtype)) ~ 1/context_len
  - Very low compute-to-memory ratio

Optimization strategies:
  - Minimize KV cache size (GQA, quantization, compression)
  - Use smaller tile sizes adapted for single-query (FlashInfer adaptive tiles)
  - CUDAGraph to eliminate kernel launch overhead
  - Batched decode: amortize KV cache reads across batch
  - PagedAttention for memory efficiency
```

### Chunked Prefill

```
Problem: Long prefills block decode requests, causing latency spikes
Solution: Split prefill into smaller chunks, interleave with decode

Algorithm:
  chunk_size = 512  # Configurable
  for chunk_start in range(0, prompt_len, chunk_size):
    chunk_end = min(chunk_start + chunk_size, prompt_len)
    # Process prefill chunk
    prefill_attention(Q[chunk_start:chunk_end], K[:chunk_end], V[:chunk_end])
    # Interleave with pending decode requests
    for decode_request in pending_decodes:
        decode_attention(decode_request)

Benefits:
  - Reduces time-to-first-token (TTFT) for concurrent requests
  - Enables prefill-decode batching for better GPU utilization
  - Configurable chunk size trades latency vs throughput
```

### Disaggregated Prefill and Decode

```
Architecture: Separate GPU pools for prefill vs decode
  Prefill GPUs: High-compute, process prompts
  Decode GPUs: High-bandwidth, generate tokens

KV Cache Transfer:
  - Layer-by-layer transfer: begin sending KV cache as soon as each
    layer finishes, overlapping with subsequent layer computation
  - Reduces total transfer latency vs waiting for full prefill

Benefits:
  - Each GPU pool optimized for its workload characteristics
  - No interference between compute-bound and memory-bound phases
  - Better hardware utilization for both phases
```

### Speculative Decoding Attention

```
Standard speculative decoding:
  1. Draft model generates k candidate tokens autoregressively
  2. Target model verifies all k tokens in single forward pass
  3. Accept longest matching prefix, reject rest
  4. Statistically equivalent to target model's distribution

Attention during verification:
  - Q = [k+1 candidate tokens, d]  (draft tokens + next)
  - K, V = [context + k candidates, d]
  - Uses causal mask ensuring each candidate sees only predecessors
  - Single forward pass verifies all candidates simultaneously
```

### Medusa (Multi-Head Speculative)

```
Architecture:
  - Add k extra prediction heads to the base LLM
  - Each head i predicts the (i+1)-th future token
  - Heads are lightweight MLPs on top of the last hidden state

Tree Attention:
  - TopK candidates per head form a tree of possible continuations
  - Tree structured as: root -> [head1_topk] -> [head2_topk] -> ...
  - Consolidated into a single tree attention mask:

  Example tree (k=2, topk=3):
    Position 0 (known):     token_A
    Position 1 (head 1):    [B1, B2, B3]
    Position 2 (head 2):    [C1, C2, C3] for each B

  Tree attention mask (sparse):
    Processes all paths simultaneously with custom causal mask
    Each candidate attends only to its ancestors in the tree

Verification:
  - Typical acceptance selects longest valid prefix
  - Accepts 2-3 tokens per step on average (vs 1 for autoregressive)
```

### EAGLE (Enhanced Speculative)

```
Key differences from Medusa:
  - Predicts feature vectors, not token probabilities
  - Uses sampling results as input (handles token-level uncertainty)
  - Autoregressive draft at feature level, parallel at token level

EAGLE-2: Dynamic draft tree sizing based on confidence scores
EAGLE-3: 3.0-6.5x speedup, 20-40% improvement over EAGLE-2
  - Training-time test approach for better draft accuracy
```

Sources:
- [NVIDIA Chunked Prefill Blog](https://developer.nvidia.com/blog/streamlining-ai-inference-performance-and-deployment-with-nvidia-tensorrt-llm-chunked-prefill/)
- [Medusa Paper (arXiv:2401.10774)](https://arxiv.org/pdf/2401.10774)
- [EAGLE Paper (arXiv:2401.15077)](https://arxiv.org/pdf/2401.15077)
- [EAGLE-2 (arXiv:2406.16858)](https://arxiv.org/html/2406.16858v1)
- [Disaggregated Prefill/Decode (Perplexity)](https://www.perplexity.ai/hub/blog/disaggregated-prefill-and-decode)

---

## 13. cuDNN Flash Attention

### Overview

NVIDIA's cuDNN implements scaled dot-product attention (SDPA) using FlashAttention-2 algorithm with deep hardware-specific optimizations. It is the default fused attention backend in NVIDIA Transformer Engine for Hopper GPUs.

### Key Features

```
Supported configurations:
  - Data types: FP16, BF16, FP8 (E4M3)
  - Masks: Causal, sliding window, padding, arbitrary
  - Attention variants: MHA, MQA, GQA
  - Sequence formats: Fixed-length, ragged (variable-length)
  - Operations: Forward, backward (training + inference)

Architecture-specific optimizations:
  - Automatic tile size selection based on GPU architecture
  - Heuristic performance knob tuning per problem size
  - Stream-K parallelization for decode (single-query) attention
```

### Graph API

```python
# cuDNN Frontend API (Python)
import cudnn

# Build attention operation graph
graph = cudnn.pygraph()
Q = graph.tensor(shape=[B, H, S_q, D])
K = graph.tensor(shape=[B, H, S_kv, D])
V = graph.tensor(shape=[B, H, S_kv, D])

# SDPA operation with options
O, stats = graph.sdpa(
    q=Q, k=K, v=V,
    is_inference=True,
    attn_scale=1.0/sqrt(D),
    causal_mask=True,
    sliding_window_length=4096,  # Optional
)

# Build and execute
graph.build()
graph.execute({Q: q_data, K: k_data, V: v_data, O: o_data})
```

### Performance

| Configuration | Performance |
|--------------|-------------|
| cuDNN BF16 vs PyTorch eager | Up to 2x faster |
| cuDNN FP8 vs PyTorch eager | Up to 3x faster |
| FP8 peak (H200, head_dim=256) | 1.2 PFLOPs |
| Stream-K decode attention | 200% avg speedup on Hopper+Ampere |
| Llama2-70B LoRA finetuning | 1.15x with FP8 SDPA |

### cuDNN vs FlashAttention

- cuDNN auto-selects between flash and non-flash algorithms based on problem size
- Integrates seamlessly with Transformer Engine for mixed-precision training
- Forward compatibility: works on future GPU architectures via PTX JIT
- Available through PyTorch's `torch.nn.functional.scaled_dot_product_attention` backend selection

Sources:
- [cuDNN Attention Docs](https://docs.nvidia.com/deeplearning/cudnn/frontend/v1.9.0/operations/Attention.html)
- [Accelerating Transformers with cuDNN 9 (NVIDIA Blog)](https://developer.nvidia.com/blog/accelerating-transformers-with-nvidia-cudnn-9)
- [CUDA Tile Flash Attention Tuning (NVIDIA Blog)](https://developer.nvidia.com/blog/tuning-flash-attention-for-peak-performance-in-nvidia-cuda-tile/)

---

## 14. xFormers Memory Efficient Attention

### Implementation

Meta's xFormers provides a memory-efficient attention implementation built on NVIDIA CUTLASS, offering an alternative to FlashAttention.

### Architecture

```
Dispatch hierarchy:
  memory_efficient_attention(Q, K, V)
    -> Flash backend (preferred for supported configs)
    -> CUTLASS backend (broader GPU/dtype support)

CUTLASS Backend:
  - Supports FP16, BF16, FP32
  - Works on: A100, V100, P100 (broader than Flash)
  - Both forward and backward passes
```

### Key Differences from FlashAttention

| Aspect | FlashAttention | xFormers CUTLASS |
|--------|---------------|-----------------|
| **Intermediate storage** | Registers | Shared memory |
| **Head dim support** | d <= 256 | Arbitrary d |
| **FW parallelism** | O(B * H) blocks | O(B * M * H) blocks |
| **FW performance (d>64)** | Baseline | Up to 1.8x faster |
| **Overall FW perf** | 5% faster avg | Baseline |

The CUTLASS backend stores intermediate results in shared memory (vs Flash's registers), enabling support for arbitrary embedding dimensions but potentially using more SRAM.

### Threading Model

```
xFormers CUTLASS kernel:
  - Extends CUTLASS threadblock::Mma classes
  - Chains multiple matmuls within same kernel (Q@K^T then P@V)
  - Does "prologue early": starts loading next tile during computation
  - One thread block per query position (more parallelism for short sequences)

FlashAttention kernel:
  - One thread block per query block (larger blocks, fewer launches)
  - Better for long sequences with large tile sizes
  - Less launch overhead
```

### Performance

- CUTLASS kernels: average 31% faster than vanilla PyTorch, 10% median
- Compared to FlashAttention: 5% slower on average in forward pass
- For head_dim > 64: xFormers can outperform Flash by up to 1.8x
- Backward pass: competitive with FlashAttention

Sources:
- [xFormers Optimized Operators](https://facebookresearch.github.io/xformers/components/ops.html)
- [CUTLASS Discussion](https://github.com/NVIDIA/cutlass/discussions/657)
- [xFormers CUTLASS PR](https://github.com/facebookresearch/xformers/pull/362)

---

## 15. SageAttention

### Overview

SageAttention achieves 2-5x speedup over FlashAttention through quantized attention computation, without losing end-to-end quality across language, image, and video models.

### SageAttention v1: INT8 Quantization

**Key Insight**: INT8 matmul on RTX 4090/3090 is 4x faster than FP16 and 2x faster than FP8.

```
Algorithm:
  1. Smooth K matrix: K_smooth = K - mean(K, dim=token)
     - Subtracts per-channel mean to reduce outlier magnitude
     - Preserves attention scores: softmax((Q @ K^T) = softmax(Q @ K_smooth^T + bias)
     - Bias is constant across softmax row, doesn't affect output
     - Overhead: < 0.2%

  2. Quantize Q and K_smooth to INT8:
     - Per-token quantization for Q (each row scaled independently)
     - Per-token quantization for K_smooth
     - scale_q = max(abs(Q[i])) / 127 per row
     - scale_k = max(abs(K_smooth[j])) / 127 per row

  3. Compute S = Q_int8 @ K_smooth_int8^T using INT8 tensor cores
     - Result in INT32, dequantized to FP16

  4. Compute P @ V in FP16 with FP16 accumulator
     - FP16 accum is 2x faster than FP32 accum
     - Sufficient accuracy since P is already normalized
```

### SageAttention v2: INT4 + FP8

```
Improvements over v1:
  1. Q, K quantized to INT4 (per-thread granularity)
     - Finer granularity than per-tensor while remaining hardware-efficient
     - Each thread handles its own quantization parameters

  2. P, V quantized to FP8
     - Better dynamic range than INT8 for probability-weighted values

  3. Thorough outlier smoothing:
     - Multi-step smoothing pipeline for extreme outliers
     - Adaptive per-layer selection between precision variants

Performance:
  - 3x faster than FlashAttention-2
  - 4.5x faster than xFormers
  - Matches FlashAttention-3 FP8 speed on Hopper with better accuracy
```

### Kernel Implementation

```python
# Triton kernel structure (simplified)
@triton.jit
def sage_attention_kernel(Q, K, V, O, ...):
    # Fused with RoPE to eliminate I/O overhead
    q_block = load_and_rope(Q, block_id)

    # Smooth K (subtract channel mean)
    k_block = load_and_smooth(K, block_id)

    # Quantize to INT8 on-the-fly in registers
    q_int8, q_scale = quantize_per_token_int8(q_block)
    k_int8, k_scale = quantize_per_token_int8(k_block)

    # INT8 GEMM using tensor core mma(u8.u8.s32)
    scores_int32 = tl.dot(q_int8, k_int8.T)
    scores = dequantize(scores_int32, q_scale, k_scale)

    # Standard softmax
    p = softmax(scores)

    # FP16 GEMM with FP16 accumulator: mma(f16.f16.f16)
    output = tl.dot(p.to(fp16), v_block.to(fp16))
```

### Adaptive Quantization

Four kernel variants per layer, selected based on cosine similarity threshold:
1. INT8 Q,K + FP16 P,V (fastest, slight accuracy loss)
2. INT8 Q,K + FP16 P,V with FP32 accumulator
3. FP16 Q,K + FP16 P,V with FP16 accumulator
4. Full FP16 (fallback)

### Performance Benchmarks

| GPU | Speed vs FlashAttention-2 | Peak TOPS |
|-----|--------------------------|-----------|
| RTX 4090 | 2.1x | 340 TOPS (headdim=64,128) |
| RTX 3090 | 2.7x | - |
| H100 (SageAttn2) | Matches FA3-FP8 | - |

Accepted at ICLR 2025, ICML 2025, NeurIPS 2025 Spotlight.

Sources:
- [SageAttention Paper (arXiv:2410.02367)](https://arxiv.org/html/2410.02367v9)
- [SageAttention2 Paper (arXiv:2411.10958)](https://arxiv.org/abs/2411.10958)
- [SageAttention GitHub](https://github.com/thu-ml/SageAttention)

---

## 16. Native Sparse Attention (NSA) - DeepSeek

### Overview

NSA is a hardware-aligned, natively trainable sparse attention mechanism from DeepSeek that combines three parallel attention branches. Won Best Paper at ACL 2025.

### Three-Branch Architecture

```
For each query q_t:

Branch 1 - Compression (coarse-grained global context):
  - Group sequential keys into blocks of length l with stride d (d < l)
  - Compress each block via learnable MLP: c_i = phi(k_{id+1:id+l})
  - MLP includes intra-block position encoding
  - Compressed sequence length: ~(t-l)/d (much shorter than t)
  - Compute attention: O_compress = softmax(q @ C^T) @ V_compressed

Branch 2 - Selection (fine-grained important tokens):
  - Derive importance scores from compression branch attention
  - Score aggregation across GQA group heads for consistency
  - Select top-n blocks with highest importance scores
  - Retain ALL tokens within selected blocks (contiguous access)
  - n=16 blocks, block_size=64 tokens in experiments
  - Compute attention: O_select = softmax(q @ K_selected^T) @ V_selected

Branch 3 - Sliding Window (local context):
  - Standard sliding window attention, w=512 tokens
  - Captures local patterns and recent context
  - O_window = sliding_window_attention(q, K_recent, V_recent)

Output Combination:
  - Learned gating: g_c, g_s, g_w = sigmoid(W_gate @ q)
  - O = g_c * O_compress + g_s * O_select + g_w * O_window
```

### Hardware-Aligned Kernel Design (Triton)

```python
# NSA Triton kernel structure
@triton.jit
def nsa_kernel(Q, K, V, block_indices, ...):
    # Outer loop: over GQA groups (placed in Triton grid scheduler)
    gqa_group = tl.program_id(0)

    # Load all heads' queries within GQA group simultaneously
    queries = load_gqa_group_queries(Q, gqa_group, position_t)

    # Load shared sparse KV block indices (same for all heads in group)
    sparse_indices = load_block_indices(block_indices, gqa_group)

    # Inner loop: sequentially load contiguous KV blocks
    for block_idx in sparse_indices:
        # Load contiguous key/value block into SRAM
        k_block = load_contiguous_kv(K, block_idx)  # [block_size, head_dim]
        v_block = load_contiguous_kv(V, block_idx)

        # Compute attention for all heads in group (shared KV)
        for head in range(gqa_group_size):
            scores = queries[head] @ k_block.T
            # Online softmax accumulation
            update_attention_state(head, scores, v_block)
```

Key optimizations:
- **Group-centric loading**: All heads in a GQA group share KV fetches
- **Contiguous KV blocks**: Selected blocks are loaded sequentially (no scattered access)
- **Balanced arithmetic intensity**: Reduced memory bandwidth via block-level sparsity

### Training Methodology

```
Key design decisions:
  1. Separate KV projections per branch (prevents shortcut learning
     where local patterns dominate gradient signals)
  2. End-to-end differentiable: compression MLP and selection mechanism
     both learnable, enabling gradient flow
  3. Native pretraining: trained from initialization (not post-hoc pruning)
  4. Synchronized adaptation between sparse attention and other components
```

### Active Token Budget

```
At 32K sequence length:
  Compression: ~(32K - 64) / 32 = ~1000 compressed representations
  Selection:   16 blocks * 64 tokens = 1024 tokens
  Sliding:     512 tokens
  Fixed:       1 initial block + 2 local blocks = 3 * 64 = 192 tokens
  Total:       ~2,560 active tokens (vs 32,768 for full attention)
  Sparsity:    ~92%
```

### Performance

| Metric | NSA vs Full Attention |
|--------|----------------------|
| Forward (64K) | 9.0x speedup |
| Backward (64K) | 6.0x speedup |
| Decode (64K) | 11.6x speedup |
| MMLU | 56.5% vs 56.7% (within margin) |
| GSM8K | 52.0% vs 48.6% (+3.4%) |
| Needle-in-haystack (64K) | Perfect accuracy |
| LongBench avg | 0.469 vs 0.437 (+0.032) |

Sources:
- [NSA Paper (arXiv:2502.11089)](https://arxiv.org/abs/2502.11089)
- [NSA ACL Anthology](https://aclanthology.org/2025.acl-long.1126/)
- [DeepSeek NSA Announcement](https://x.com/deepseek_ai/status/1891745487071609327)
- [MarkTechPost NSA Overview](https://www.marktechpost.com/2025/02/18/deepseek-ai-introduces-nsa-a-hardware-aligned-and-natively-trainable-sparse-attention-mechanism-for-ultra-fast-long-context-training-and-inference/)

---

## Summary: Choosing the Right Attention Mechanism

### Decision Matrix

| Scenario | Recommended Approach |
|----------|---------------------|
| Standard inference (< 8K ctx) | FlashAttention-2 / cuDNN SDPA |
| Hopper GPU inference | FlashAttention-3 with FP8 |
| High-throughput serving | PagedAttention + FlashInfer + GQA |
| Long context (> 32K) | NSA, Sliding Window, or Ring Attention |
| Shared prefix batching | FlashInfer Cascade Attention |
| Consumer GPUs (RTX 3090/4090) | SageAttention (INT8) |
| Memory-constrained | KV cache quantization (KIVI/FP8) + H2O/SnapKV |
| Infinite streaming | StreamingLLM with attention sinks |
| Multi-GPU long context | Ring Attention / Context Parallel |
| Speculative decoding | EAGLE-3 with tree attention |
| Linear complexity needed | Lightning Attention-2 or Mamba |

### Complexity Comparison

| Method | Time Complexity | Memory | HBM Accesses |
|--------|----------------|--------|--------------|
| Standard Attention | O(N^2 d) | O(N^2) | O(Nd + N^2) |
| FlashAttention | O(N^2 d) | O(N) | O(N^2 d^2 / M) |
| Sliding Window | O(N W d) | O(N) | O(N W d^2 / M) |
| Sparse (BigBird) | O(N(g+w+r)d) | O(N) | O(N(g+w+r)d) |
| Linear Attention | O(N d^2) | O(d^2) | O(N d^2) |
| NSA | O(N * s * d) | O(N) | Proportional to sparsity |
| Ring Attention | O(N^2 d / P) per GPU | O(N/P) | Distributed |

Where N = sequence length, d = head dimension, M = SRAM size, W = window size, P = number of GPUs, s = active tokens in sparse pattern.
