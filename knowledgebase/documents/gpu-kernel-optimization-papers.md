# GPU Kernel Optimization & ML Inference: Comprehensive Paper Reference

> A curated knowledge base of the most important research papers on GPU kernel optimization, ML inference acceleration, attention mechanisms, quantization, pruning, model architectures, and serving systems. Each entry includes full citation, key insights, algorithm summaries, performance results, and implications for kernel design.

---

## Table of Contents

1. [Attention Mechanisms & Kernels](#1-attention-mechanisms--kernels)
2. [Decoding Optimization](#2-decoding-optimization)
3. [KV Cache Management](#3-kv-cache-management)
4. [Serving Systems & Scheduling](#4-serving-systems--scheduling)
5. [Speculative Decoding](#5-speculative-decoding)
6. [Quantization](#6-quantization)
7. [Pruning & Sparsity](#7-pruning--sparsity)
8. [Model Architectures](#8-model-architectures)
9. [Positional Encoding & Activation Functions](#9-positional-encoding--activation-functions)
10. [Mixture of Experts](#10-mixture-of-experts)
11. [Distributed & Long-Context](#11-distributed--long-context)
12. [Parameter-Efficient Fine-Tuning](#12-parameter-efficient-fine-tuning)
13. [Scaling Laws](#13-scaling-laws)
14. [Kernel Frameworks & DSLs](#14-kernel-frameworks--dsls)

---

## 1. Attention Mechanisms & Kernels

### 1.1 FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

- **Authors:** Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Re
- **Date:** May 27, 2022 (NeurIPS 2022)
- **arXiv:** [2205.14135](https://arxiv.org/abs/2205.14135)

**Key Insight:** Standard attention is bottlenecked by memory I/O between HBM and SRAM, not by arithmetic. By making the attention algorithm IO-aware --- accounting for reads and writes at each level of the GPU memory hierarchy --- one can achieve wall-clock speedups without approximation.

**Algorithm / Technique:**
- **Tiling:** The Q, K, V matrices are partitioned into blocks that fit into GPU SRAM (~20MB on A100). The outer loop iterates over K/V blocks, loading them into SRAM. The inner loop iterates over Q blocks, computing partial attention scores incrementally.
- **Online Softmax:** Uses the online softmax trick (Milakov & Gimelshein 2018) to compute exact softmax in a single pass without materializing the full N x N attention matrix in HBM. Maintains running max and running sum of exponentials, rescaling partial results as new blocks arrive.
- **Kernel Fusion:** The entire attention computation (Q*K^T, scaling, masking, softmax, dropout, V multiplication) is fused into a single GPU kernel, eliminating intermediate reads/writes to HBM.
- **Recomputation in Backward Pass:** Rather than storing O(N^2) attention weights for the backward pass, FlashAttention recomputes them from Q, K, V blocks in SRAM, trading FLOPs for memory. This reduces memory from O(N^2) to O(N).
- **IO Complexity:** O(N^2 * d^2 / M) HBM accesses, where M is SRAM size. This is provably optimal for certain SRAM sizes, compared to O(N^2 * d + N^2) for standard attention.
- **Block-Sparse Extension:** Supports block-sparse attention patterns for even longer sequences (up to 64K).

**Performance Results:**
- 15% end-to-end wall-clock speedup on BERT-large (seq length 512) vs. MLPerf 1.1 record
- 3x speedup on GPT-2 (seq length 1K)
- 2.4x speedup on long-range arena (seq length 1K-4K)
- Memory: O(N) instead of O(N^2), enabling 4x longer context on same hardware
- Block-sparse FlashAttention: up to 64K sequence length with improved quality

**Implications for Kernel Design:**
- The memory hierarchy is the primary constraint for attention, not compute. Kernel designers must optimize for IO, not just FLOPs.
- Tiling to SRAM is the fundamental technique for IO-aware kernel design.
- Fusing multiple operations into a single kernel avoids costly HBM round-trips.
- Recomputation can be cheaper than materialization when memory bandwidth is the bottleneck.

---

### 1.2 FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning

- **Authors:** Tri Dao
- **Date:** July 17, 2023 (ICLR 2024)
- **arXiv:** [2307.08691](https://arxiv.org/abs/2307.08691)

**Key Insight:** FlashAttention-1 reached only 25-40% of theoretical max FLOPs/s on A100 because of suboptimal work partitioning between warps and thread blocks. By reducing non-matmul FLOPs, improving parallelism across the sequence length dimension, and better distributing work within thread blocks, one can nearly close the gap with optimized GEMM.

**Algorithm / Technique:**
- **Reduced Non-matmul FLOPs:** Restructures the online softmax computation to minimize rescaling operations. The algorithm applies the log-sum-exp correction lazily at the end rather than at every block iteration.
- **Parallelism over Sequence Length:** FlashAttention-1 parallelized over batch and heads. FlashAttention-2 additionally parallelizes over the sequence length dimension (Q blocks), improving occupancy especially for long sequences with fewer heads.
- **Better Warp Partitioning:** Eliminates warp-level synchronization for shared-memory reads. Each warp works on its own subset of Q blocks and accumulates results independently, avoiding cross-warp communication through shared memory.
- **Loop Structure Swap:** The outer loop now iterates over Q blocks (rows), and the inner loop over K/V blocks (columns). This reversal reduces the amount of shared memory communication and rescaling.
- **Causal Masking Optimization:** For causal attention, entire K/V blocks that are fully masked are skipped, reducing computation by ~50% for autoregressive models.

**Performance Results:**
- 2x speedup over FlashAttention-1
- Reaches 50-73% of theoretical max FLOPs/s on A100 (vs. 25-40% for FA1)
- GPT-style training: 225 TFLOPs/s per A100 GPU (72% model FLOPs utilization)
- Forward pass: up to 73% of A100 peak
- Combined forward + backward: ~70% of A100 peak

**Implications for Kernel Design:**
- Non-matmul instructions (register shuffles, shared memory reads, control flow) are first-class concerns. Even a small percentage of non-matmul FLOPs can dominate wall-clock time when the matmul pipeline is saturated.
- Parallelization strategy must consider all dimensions (batch, heads, sequence) to maximize GPU occupancy.
- Within-threadblock work distribution (warp partitioning) matters as much as between-threadblock parallelism.

---

### 1.3 FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision

- **Authors:** Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao
- **Date:** July 11, 2024 (NeurIPS 2024)
- **arXiv:** [2407.08608](https://arxiv.org/abs/2407.08608)

**Key Insight:** The NVIDIA Hopper GPU (H100) introduces new hardware capabilities --- asynchronous data movement (TMA), warp specialization, and FP8 Tensor Cores --- that require fundamentally different kernel designs than Ampere (A100). FlashAttention-3 exploits these features to achieve near-peak utilization.

**Algorithm / Technique:**
- **Warp Specialization:** Divides warps within a CTA (cooperative thread array) into producer and consumer roles. Producers handle data movement (using TMA for asynchronous HBM-to-SRAM copies), consumers handle Tensor Core computation. This overlaps memory latency with compute.
- **Asynchronous Softmax / GEMM Interleaving:** On Hopper, the WGMMA (warp group matrix-multiply-accumulate) instruction is asynchronous. FlashAttention-3 pipelines the softmax computation with the next GEMM, overlapping non-matmul operations with matmul operations in the instruction pipeline.
- **FP8 with Block Quantization:** Leverages H100's native FP8 Tensor Cores for 2x theoretical throughput over FP16. Uses per-block quantization (Q, K, V quantized independently per tile) to maintain accuracy. Tiles are small enough that per-tile scale factors capture the dynamic range.
- **Incoherent Processing for FP8:** Applies random orthogonal transformations to spread outlier magnitudes across dimensions before quantization, reducing quantization error. Combined with block quantization, achieves 2.6x lower numerical error than naive FP8 attention.
- **Pingpong Scheduling:** For small batch sizes, alternates WGMMA instructions between two warp groups in a single SM to avoid register bank conflicts, increasing Tensor Core utilization.

**Performance Results:**
- FP16: up to 740 TFLOPs/s on H100 (75% utilization)
- FP8: close to 1.2 PFLOPs/s on H100
- 1.5-2.0x speedup over FlashAttention-2 on H100
- FP8 FlashAttention-3: 2.6x lower numerical error than naive FP8 attention

**Implications for Kernel Design:**
- Each GPU generation requires rethinking kernel design from scratch. Hopper's asynchronous TMA, warp specialization, and WGMMA are qualitatively different from Ampere's programming model.
- Overlapping data movement with computation is essential on Hopper. Producer-consumer warp specialization is the key pattern.
- FP8 for attention is viable with block quantization + incoherent processing. The 2x throughput boost is real but requires careful numerical treatment.
- Pingpong scheduling between warp groups is necessary for small-batch scenarios to avoid under-utilization.

---

### 1.4 SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration

- **Authors:** Jintao Zhang, Jia Wei, Haofeng Huang, Pengle Zhang, Jun Zhu, Jianfei Chen
- **Date:** October 3, 2024 (ICLR 2025)
- **arXiv:** [2410.02367](https://arxiv.org/abs/2410.02367)

**Key Insight:** While quantization research has focused on linear layers, attention itself (with O(N^2) complexity) becomes the dominant cost at long sequence lengths. Quantizing Q and K to INT8 and computing P = Q*K^T on INT8 Tensor Cores, while keeping PV in FP16, yields a plug-and-play 2x speedup with negligible accuracy loss.

**Algorithm / Technique:**
- Quantizes Q and K matrices to INT8 per-head, per-channel
- Computes QK^T using INT8 Tensor Cores (2x throughput vs. FP16)
- Applies smoothing to handle activation outliers
- Keeps softmax and PV multiplication in FP16/FP32 for numerical stability
- SageAttention2 extends to 4-bit with per-thread quantization
- SageAttention3 explores microscaling FP4 attention

**Performance Results:**
- 2.1x faster than FlashAttention-2, 2.7x faster than xformers
- Superior accuracy over FlashAttention-3's FP8
- Almost no end-to-end metrics loss across language, image, and video models
- Plug-and-play: no retraining needed

**Implications for Kernel Design:**
- INT8 Tensor Cores for attention QK^T are underexploited. The key is asymmetric precision: INT8 for Q*K^T, FP16 for softmax and PV.
- Per-head, per-channel quantization granularity is sufficient for attention matrices.
- Quantized attention kernels can be drop-in replacements for FP16 attention.

---

### 1.5 Native Sparse Attention (NSA)

- **Authors:** Jingyang Yuan, Huazuo Gao, Damai Dai, Junyu Luo, Liang Zhao, et al. (DeepSeek-AI & Peking University)
- **Date:** February 16, 2025
- **arXiv:** [2502.11089](https://arxiv.org/abs/2502.11089)

**Key Insight:** Sparse attention methods typically sacrifice training efficiency for inference speed. NSA is a hardware-aligned sparse attention that is natively trainable end-to-end, using a dynamic hierarchical strategy that combines coarse-grained compression, fine-grained selection, and sliding windows.

**Algorithm / Technique:**
- **Three Attention Paths:** Each query attends through (1) compressed coarse-grained tokens (temporal blocks aggregated via learned compression), (2) selectively retained fine-grained tokens (top-k blocks chosen by importance scoring), and (3) sliding window for local context.
- **Hardware Alignment:** Block sizes and computation patterns are designed to align with GPU memory access patterns and Tensor Core tile sizes.
- **End-to-End Training:** Unlike post-hoc sparsity methods, NSA is trained from scratch with the sparse pattern, enabling the model to learn to utilize the hierarchical structure.
- **Arithmetic Intensity Balancing:** Computation is balanced across the three paths to maximize hardware utilization.

**Performance Results:**
- Maintains or exceeds Full Attention quality on general benchmarks, long-context tasks, and reasoning
- Substantial speedups on 64k-length sequences for decoding, forward, and backward passes
- Enables pre-training compute reduction without performance loss

**Implications for Kernel Design:**
- Sparse attention kernels must be hardware-aligned: block sizes should match Tensor Core tiles and memory transaction sizes.
- Multi-path attention (coarse + fine + local) is more effective than single-granularity sparsity.
- Training-time integration of sparsity patterns produces better quality than post-hoc pruning.

---

### 1.6 Differential Transformer

- **Authors:** Tianzhu Ye, Li Dong, Yuqing Xia, Yutao Sun, Yi Zhu, Gao Huang, Furu Wei (Microsoft Research & Tsinghua University)
- **Date:** October 7, 2024 (ICLR 2025)
- **arXiv:** [2410.05258](https://arxiv.org/abs/2410.05258)

**Key Insight:** Standard softmax attention assigns non-negligible probability mass to irrelevant tokens (noise). Differential attention, inspired by differential amplifiers / noise-cancelling headphones, computes two separate attention maps and takes their difference, cancelling out common-mode noise.

**Algorithm / Technique:**
- Splits Q and K into two groups: (Q1, K1) and (Q2, K2)
- Computes two softmax attention maps: A1 = softmax(Q1 * K1^T / sqrt(d)), A2 = softmax(Q2 * K2^T / sqrt(d))
- The effective attention is: A_diff = A1 - A2
- Applies the differential attention to V: O = A_diff * V
- Learnable scalar lambda parameters control the balance

**Performance Results:**
- Outperforms standard Transformer across scales (from 830M to 13B parameters) on language modeling
- Superior key information retrieval (in-context learning, multi-needle-in-haystack)
- Reduces hallucination in summarization and question answering
- Fewer activation outliers (beneficial for quantization)
- 6.8x fewer context length tokens needed for in-context learning

**Implications for Kernel Design:**
- Differential attention requires computing two softmax maps and subtracting, doubling some computation but potentially simplifying downstream quantization.
- The reduction in activation outliers means Diff Transformer models are more amenable to aggressive quantization.
- Kernel implementation requires careful handling of the subtraction to maintain numerical stability.

---

### 1.7 Grouped-Query Attention (GQA)

- **Authors:** Joshua Ainslie, James Lee-Thorp, Michal de Jong, Yinfei Yang, Siddhartha Reddy Jonnalagadda, Santiago Ontanon
- **Date:** May 22, 2023 (EMNLP 2023)
- **arXiv:** [2305.13245](https://arxiv.org/abs/2305.13245)

**Key Insight:** Multi-Head Attention (MHA) has separate K, V heads per query head, creating large KV caches. Multi-Query Attention (MQA) shares a single K, V head across all query heads, drastically reducing KV cache but sometimes losing quality. GQA is the middle ground: group query heads and share K, V within groups.

**Algorithm / Technique:**
- Instead of H key-value heads (MHA) or 1 (MQA), uses G key-value head groups (1 < G < H)
- Each group of H/G query heads shares a single key-value head pair
- Existing MHA models can be "uptrained" to GQA using only 5% of original pre-training compute by mean-pooling adjacent KV heads
- The number of groups G is a tunable hyperparameter trading off quality vs. KV cache size

**Performance Results:**
- GQA with G=8 achieves quality close to MHA while being nearly as fast as MQA
- KV cache reduction: H/G factor (e.g., 8x reduction with G=8 for a 64-head model)
- Uptraining recipe needs only 5% of original pre-training compute
- Adopted by LLaMA-2 70B, Mistral, and most modern LLMs

**Implications for Kernel Design:**
- GQA changes the shape of attention kernels: multiple query heads share K, V. Kernels must broadcast K, V across grouped query heads efficiently.
- KV cache layout should be optimized for GQA: storing KV per group rather than per head.
- The ratio H/G affects the compute-to-memory ratio of attention, influencing tiling strategies.

---

## 2. Decoding Optimization

### 2.1 Flash-Decoding

- **Authors:** Tri Dao, Daniel Haziza, Francisco Massa, Grigory Sizov (with contributions from Erich Elsen, Ashish Vaswani, Michaël Benesty)
- **Date:** October 12, 2023 (Blog post)
- **Link:** [princeton-nlp.github.io/flash-decoding](https://princeton-nlp.github.io/flash-decoding/)

**Key Insight:** During autoregressive decoding, the query length is 1 (a single new token), but K/V can be very long (entire context). FlashAttention parallelizes over batch and heads, but a single query doesn't provide enough work to utilize all SMs. Flash-Decoding adds a new parallelism dimension: splitting the KV sequence length.

**Algorithm / Technique:**
- **Split-K for Attention:** Partitions the K, V cache along the sequence length into S chunks. Each chunk is assigned to a separate thread block.
- **Parallel Partial Attention:** Each thread block independently computes attention over its KV chunk, producing a partial output vector and the corresponding log-sum-exp (for online softmax merging).
- **Reduction:** A lightweight final reduction kernel merges the S partial outputs using the online softmax trick: re-weight each partial output by exp(local_lse - global_lse) and sum.
- **Adaptive Splitting:** The number of splits S is chosen heuristically based on sequence length and available SMs.

**Performance Results:**
- Up to 8x speedup for long sequences (64K tokens) during decoding
- Scales well from 512 to 64K sequence length
- Negligible overhead for short sequences (gracefully degrades to standard FlashAttention)

**Implications for Kernel Design:**
- The decode phase (Nq=1) is fundamentally different from the prefill phase (Nq>>1). Kernels must handle both regimes.
- Split-K parallelism is essential when the primary work dimension (batch * heads) doesn't fill the GPU.
- The two-kernel approach (parallel partial attention + reduction) is a general pattern for parallelizing reductions.

---

### 2.2 FlashDecoding++: Faster Large Language Model Inference on GPUs

- **Authors:** Ke Hong, Guohao Dai, Jiaming Xu, Qiuli Mao, Xiuhong Li, Jun Liu, Kangdi Chen, Yuhan Dong, Yu Wang
- **Date:** November 2, 2023 (MLSys 2024)
- **arXiv:** [2311.01282](https://arxiv.org/abs/2311.01282)

**Key Insight:** Flash-Decoding's partial softmax requires a synchronization step to merge results. FlashDecoding++ eliminates this synchronization by using a unified maximum value across partial softmax computations, pre-computed from a lightweight analysis of the data distribution.

**Algorithm / Technique:**
- **Unified Max Value:** Pre-computes a conservative upper bound on the attention logit maximum, so all partial softmax computations use the same max, eliminating the need for a correction/merge step. This removes ~20% overhead from the synchronization.
- **Flat GEMM Optimization:** During decoding, GEMMs are "flat" (one dimension is very small). FlashDecoding++ applies double-buffering and padding-free tiling to optimize these shapes.
- **Heuristic Dataflow:** Dynamically selects dataflow patterns based on input shapes and hardware capabilities, including handling of both NVIDIA and AMD GPUs.

**Performance Results:**
- Up to 4.86x speedup on NVIDIA GPUs and 2.18x on AMD GPUs over HuggingFace
- 1.37x average speedup over state-of-the-art LLM inference engines
- Works across NVIDIA and AMD hardware

**Implications for Kernel Design:**
- Synchronization in parallel reductions is expensive. Trading precision (conservative max estimate) for eliminated sync is a powerful optimization.
- Flat GEMMs (M=1 or very small M) require specialized kernels, not standard cuBLAS.
- Cross-platform kernel design (NVIDIA + AMD) requires architecture-aware heuristics.

---

## 3. KV Cache Management

### 3.1 PagedAttention / vLLM: Efficient Memory Management for Large Language Model Serving

- **Authors:** Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica
- **Date:** September 12, 2023 (SOSP 2023)
- **arXiv:** [2309.06180](https://arxiv.org/abs/2309.06180)

**Key Insight:** Existing LLM serving systems pre-allocate contiguous GPU memory for the KV cache of each request based on the maximum sequence length, wasting 60-80% of memory due to fragmentation and over-reservation. PagedAttention applies the OS concept of virtual memory paging to KV cache management.

**Algorithm / Technique:**
- **Paged KV Cache:** Divides KV cache into fixed-size blocks (pages), each holding K and V for a fixed number of tokens (e.g., 16 tokens per block).
- **Non-contiguous Storage:** Blocks for a single sequence need not be contiguous in physical GPU memory. A page table maps logical block indices to physical memory locations.
- **On-demand Allocation:** Blocks are allocated only as tokens are generated, eliminating over-reservation.
- **Copy-on-Write Sharing:** Multiple sequences (e.g., beam search candidates, shared prefixes) can reference the same physical KV blocks. Blocks are copied only when modified.
- **Efficient Swap & Preemption:** Blocks can be swapped between GPU and CPU memory for preemption/scheduling.
- **Modified Attention Kernel:** The attention kernel is modified to read K, V from non-contiguous block locations via the page table, adding a level of indirection.

**Performance Results:**
- Near-zero memory waste (vs. 60-80% in prior systems)
- 2-4x throughput improvement over FasterTransformer and Orca at same latency
- Adopted as de facto standard: TensorRT-LLM, HuggingFace TGI, LightLLM all use PagedAttention

**Implications for Kernel Design:**
- Attention kernels must support non-contiguous KV memory access via block-based indirection (page tables).
- The block size is a critical parameter: too small increases page table overhead, too large increases internal fragmentation.
- Memory allocator design is as important as compute kernel design for serving throughput.

---

### 3.2 KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache

- **Authors:** Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, Xia Hu
- **Date:** February 5, 2024 (ICML 2024)
- **arXiv:** [2402.02750](https://arxiv.org/abs/2402.02750)

**Key Insight:** Key and Value caches have fundamentally different statistical properties. Keys have large per-channel outliers (consistent across tokens), while Values have per-token outliers. Therefore, Keys should be quantized per-channel and Values per-token, in an asymmetric 2-bit scheme.

**Algorithm / Technique:**
- **Per-channel Key quantization:** Groups key vectors across the token dimension and quantizes each channel to 2-bit with per-channel scale/zero-point.
- **Per-token Value quantization:** Quantizes each token's value vector to 2-bit with per-token scale/zero-point.
- **Residual Length:** The most recent few tokens' KV cache is kept in full precision (FP16) to maintain quality, since the model relies most on recent tokens.
- **Tuning-free:** No calibration data or retraining needed. Works out-of-the-box.
- **Hardware-efficient:** The 2-bit format uses standard INT operations with dequantization during attention.

**Performance Results:**
- 2.6x less peak memory usage (including model weights)
- Enables up to 4x larger batch sizes
- 2.35x-3.47x throughput on real LLM inference workloads
- Almost no quality degradation on LLaMA, Falcon, Mistral

**Implications for Kernel Design:**
- KV cache quantization requires asymmetric strategies: per-channel for K, per-token for V. Kernels must handle both patterns.
- Attention kernels must support mixed-precision: dequantize 2-bit K, V on-the-fly during attention computation.
- The "residual length" pattern (recent tokens in FP16, older tokens in INT2) requires branching or segmented kernels.

---

### 3.3 StreamingLLM: Efficient Streaming Language Models with Attention Sinks

- **Authors:** Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis
- **Date:** September 29, 2023 (ICLR 2024)
- **arXiv:** [2309.17453](https://arxiv.org/abs/2309.17453)

**Key Insight:** LLMs disproportionately attend to the very first few tokens ("attention sinks") regardless of their semantic content. This happens because softmax needs somewhere to "dump" unused attention probability. Keeping these sink tokens' KV cache enables stable infinite-length streaming.

**Algorithm / Technique:**
- **Attention Sink Discovery:** Empirically demonstrates that initial tokens (positions 0-3) receive disproportionately high attention scores across all layers and heads, even when semantically irrelevant.
- **StreamingLLM Window:** Maintains KV cache for: (1) the first few "sink" tokens (typically 4), and (2) a sliding window of the most recent tokens (e.g., last 1024).
- **Eviction Policy:** When the window is full, evict the oldest non-sink tokens. No recomputation needed.
- **Position Re-encoding:** Tokens in the window are re-assigned contiguous position IDs to avoid out-of-distribution positional encoding issues.
- **Training-time Sink Token:** Proposes adding a dedicated learnable "sink token" during pre-training for even more stable streaming.

**Performance Results:**
- Enables stable language modeling up to 4M+ tokens without fine-tuning (LLaMA-2, MPT, Falcon, Pythia)
- 22.2x speedup over sliding window recomputation baseline in streaming settings
- Perplexity remains stable over millions of tokens (window attention alone fails catastrophically)

**Implications for Kernel Design:**
- KV cache eviction policies are not just software engineering --- they affect attention kernel correctness. Kernels must handle non-contiguous position IDs.
- The "sink + window" pattern requires efficient concatenation of two non-adjacent KV segments.
- Position re-encoding in the kernel must be decoupled from physical KV cache positions.

---

### 3.4 H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models

- **Authors:** Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Zhangyang Wang, Beidi Chen
- **Date:** June 24, 2023 (NeurIPS 2023)
- **arXiv:** [2306.14048](https://arxiv.org/abs/2306.14048)

**Key Insight:** A small fraction of tokens ("Heavy Hitters" or H2) contribute the vast majority of attention score mass. An eviction policy that retains H2 tokens plus recent tokens preserves quality while drastically reducing KV cache size.

**Algorithm / Technique:**
- **Heavy Hitter Identification:** Tracks cumulative attention scores across generation steps. Tokens with consistently high attention scores are marked as H2.
- **Dynamic Eviction:** Maintains a budget of K total KV entries: some reserved for H2 tokens, the rest for a sliding window of recent tokens. When budget is exceeded, evict the lowest-scoring non-recent, non-H2 token.
- **Theoretical Foundation:** Formulates KV cache eviction as a dynamic submodular problem, providing theoretical guarantees on the quality of the greedy eviction strategy.

**Performance Results:**
- H2O with 20% heavy hitters improves throughput by up to 29x over DeepSpeed Zero-Inference and HF Accelerate
- 1.9x latency reduction at same batch size
- Works on OPT, LLaMA, GPT-NeoX across diverse tasks

**Implications for Kernel Design:**
- Attention kernels can benefit from importance scoring metadata to skip low-attention KV entries.
- Dynamic eviction requires auxiliary data structures (score accumulators) maintained alongside the KV cache.
- The H2 identification can be computed cheaply as a byproduct of the attention computation.

---

### 3.5 SnapKV: LLM Knows What You Are Looking for Before Generation

- **Authors:** Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, Deming Chen
- **Date:** April 22, 2024 (NeurIPS 2024)
- **arXiv:** [2404.14469](https://arxiv.org/abs/2404.14469)

**Key Insight:** Each attention head consistently focuses on specific prompt features, and these important positions can be identified from an "observation window" at the end of the prompt before generation begins. Compression decisions can be made once at prefill time rather than dynamically during decoding.

**Algorithm / Technique:**
- **Observation Window:** Uses the last few tokens of the prompt as a window to observe which KV positions receive high attention from each head.
- **Per-head Selection:** For each attention head independently, selects the top-k KV positions with highest attention scores from the observation window.
- **Clustering:** Groups selected positions into clusters to ensure spatial locality in the KV cache.
- **One-shot Compression:** The selection happens once after prefill, with no dynamic updates during generation.

**Performance Results:**
- 3.6x generation speed increase and 8.2x memory efficiency improvement on 16K token inputs
- Handles up to 380K context on a single A100-80GB GPU
- Negligible accuracy drop on Needle-in-a-Haystack evaluation
- Average 92% compression rate at 1024 budget, 68% at 4096 budget

**Implications for Kernel Design:**
- One-shot KV cache compression at prefill time is simpler than dynamic eviction and can be integrated into the prefill kernel.
- Per-head selection requires tracking attention scores per head during a "probe" pass.
- The compressed KV cache has different sizes per head, requiring either padding or ragged-tensor support.

---

### 3.6 xKV: Cross-Layer SVD for KV-Cache Compression

- **Authors:** Chi-Chih Chang et al.
- **Date:** March 24, 2025
- **arXiv:** [2503.18893](https://arxiv.org/abs/2503.18893)

**Key Insight:** The dominant singular vectors of KV caches are remarkably well-aligned across multiple adjacent layers. By applying SVD across grouped layers, the KV cache can be consolidated into a shared low-rank subspace, achieving much higher compression than per-layer methods.

**Algorithm / Technique:**
- **Cross-Layer Grouping:** Groups adjacent transformer layers whose KV caches share similar singular vector structure.
- **Joint SVD:** Applies SVD across the grouped KV caches to find the shared low-rank subspace.
- **Shared Projection:** Stores only the shared basis vectors and per-layer coefficients, dramatically reducing storage.
- **Post-training:** No retraining needed; works as a post-processing step.

**Performance Results:**
- Up to 6.8x higher compression than state-of-the-art inter-layer methods
- Up to 8.5x compression with maintained accuracy on long-context benchmarks
- 2.7% accuracy improvement over baselines at same compression rate
- Compatible with MLA (DeepSeek-Coder-V2): 3x compression on coding tasks

**Implications for Kernel Design:**
- Cross-layer KV sharing requires modified attention kernels that project from a shared basis per layer.
- The SVD-based compression can be applied offline and the kernel only needs to handle the low-rank representation at inference time.
- Compatible with other compression methods (quantization, eviction) for multiplicative gains.

---

## 4. Serving Systems & Scheduling

### 4.1 Orca: A Distributed Serving System for Transformer-Based Generative Models

- **Authors:** Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, Byung-Gon Chun
- **Date:** July 2022 (OSDI 2022)
- **Link:** [USENIX OSDI 2022](https://www.usenix.org/conference/osdi22/presentation/yu)

**Key Insight:** Traditional serving systems batch requests at the request level: a batch starts together and ends together. For autoregressive models, this is wasteful because requests have different lengths and finish at different times. Iteration-level scheduling allows the scheduler to add/remove requests from the batch at every iteration (token generation step).

**Algorithm / Technique:**
- **Iteration-Level Scheduling:** The scheduler invokes the engine for a single iteration at a time. After each iteration, finished requests are removed and new requests can be added immediately.
- **Continuous Batching:** The batch composition changes dynamically at every step, maximizing GPU utilization.
- **Selective Batching:** Different operations within a single iteration may use different batch compositions (e.g., prefill for new requests, decode for continuing ones).

**Performance Results:**
- Pioneered continuous batching, now the standard in all modern serving systems
- Significantly better GPU utilization than static batching
- Foundation for vLLM, TGI, TensorRT-LLM, and SGLang

**Implications for Kernel Design:**
- Kernels must support variable-length sequences within a single batch (ragged batching).
- The kernel interface should be iteration-granular, not request-granular.
- Memory management must handle dynamic batch composition without reallocation.

---

### 4.2 SGLang: Efficient Execution of Structured Language Model Programs

- **Authors:** Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Jeff Huang, Chuyue Sun, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark Barrett, Ying Sheng
- **Date:** December 12, 2023 (NeurIPS 2024)
- **arXiv:** [2312.07104](https://arxiv.org/abs/2312.07104)

**Key Insight:** LLM programs often involve multiple generation calls with shared prefixes (e.g., few-shot prompting, tree-of-thought, agentic workflows). RadixAttention automatically reuses KV cache across calls by maintaining a radix tree index over cached prefixes.

**Algorithm / Technique:**
- **RadixAttention:** Maintains an LRU cache of KV caches indexed by a radix tree (trie) on token sequences. When a new request arrives, the system finds the longest matching prefix in the radix tree and reuses its KV cache.
- **Automatic Prefix Sharing:** No user annotation needed. The system automatically detects shared prefixes across requests and within multi-turn conversations.
- **Compressed Finite State Machines:** Accelerates constrained decoding (JSON, regex) by pre-computing valid token sets.
- **Frontend Language:** Provides Python primitives for generation, forking, and parallelism control.

**Performance Results:**
- Up to 6.4x higher throughput than state-of-the-art systems
- Reduces KV cache memory through sharing, enabling larger batch sizes
- Reduces prefill latency by avoiding recomputation of shared prefixes

**Implications for Kernel Design:**
- Attention kernels should support prefix-aware KV cache: reading from shared prefix blocks + request-specific blocks.
- The radix tree cache requires efficient cache lookup and eviction, impacting memory layout.
- Constrained decoding benefits from fused sampling kernels that enforce token constraints.

---

### 4.3 DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving

- **Authors:** Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, Hao Zhang
- **Date:** January 17, 2024 (OSDI 2024)
- **arXiv:** [2401.09670](https://arxiv.org/abs/2401.09670)

**Key Insight:** Prefill (compute-bound, batch-friendly) and decode (memory-bound, latency-sensitive) have fundamentally different hardware requirements. Colocating them causes mutual interference. Disaggregating them onto separate GPU pools with independent parallelism strategies optimizes both phases.

**Algorithm / Technique:**
- **Disaggregated GPU Pools:** Separate GPU clusters for prefill and decode, connected by high-bandwidth interconnect.
- **Phase-specific Parallelism:** Prefill pool uses tensor parallelism (TP) for low latency; decode pool uses pipeline parallelism (PP) for throughput.
- **KV Cache Transfer:** After prefill, the KV cache is transferred from the prefill GPU to the decode GPU via network.
- **Goodput Optimization:** Co-optimizes resource allocation and parallelism for both phases to maximize goodput (requests served within SLO).
- **Placement-aware Scheduling:** Places prefill and decode pools to minimize communication overhead given cluster bandwidth.

**Performance Results:**
- 7.4x more requests served or 12.6x tighter SLO compliance vs. state-of-the-art
- Eliminates prefill-decode interference
- Enables independently tuned parallelism for each phase

**Implications for Kernel Design:**
- Prefill and decode kernels can be independently optimized for their respective regimes (compute-bound vs. memory-bound).
- KV cache serialization/deserialization over network must be efficient (potential for compression during transfer).
- The KV cache format must be standardized across prefill and decode GPUs.

---

### 4.4 Splitwise: Efficient Generative LLM Inference Using Phase Splitting

- **Authors:** Pratyush Patel, Esha Choukse, Chaojie Zhang, Aashaka Shah, Iñigo Goiri, Saeed Maleki, Ricardo Bianchini (Microsoft Research)
- **Date:** November 30, 2023 (ISCA 2024)
- **arXiv:** [2311.18677](https://arxiv.org/abs/2311.18677)

**Key Insight:** Prefill is compute-intensive (limited by FLOPS), while decode is memory-bandwidth-intensive (limited by HBM bandwidth). Using the same GPU type for both is suboptimal. Splitwise proposes a three-tier pool (prefill, decode, hybrid) to match hardware to workload characteristics.

**Algorithm / Technique:**
- **Three-tier Pool Design:** Dedicated prefill pool (compute-optimized GPUs), dedicated decode pool (bandwidth-optimized GPUs), and a hybrid pool for flexible overflow handling.
- **KV Cache Network Transfer:** Transfers KV cache from prefill to decode GPU after prompt processing.
- **Heterogeneous Hardware:** Can use different GPU types for each pool (e.g., H100 for prefill, A100 for decode).

**Performance Results:**
- 1.4x higher throughput at 20% lower cost (or 2.35x throughput at same cost/power)
- 15% lower power consumption at 1.76x higher throughput
- Demonstrated benefits with network KV cache transfer

**Implications for Kernel Design:**
- Kernels should be optimized for their phase: prefill kernels maximize FLOPS utilization, decode kernels minimize memory access.
- KV cache transfer compression (quantization during transfer) can reduce network bottleneck.
- Heterogeneous GPU deployment means kernels must be optimized per-architecture.

---

### 4.5 Sarathi-Serve: Taming Throughput-Latency Tradeoff in LLM Inference

- **Authors:** Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S. Gulavani, Alexey Tumanov, Ramachandran Ramjee
- **Date:** March 4, 2024 (OSDI 2024)
- **arXiv:** [2403.02310](https://arxiv.org/abs/2403.02310)

**Key Insight:** Prefill requests create "stalls" in decoding --- when a new prefill is computed, ongoing decode requests must wait, increasing tail latency. Chunked prefills break the prefill into smaller chunks and interleave them with decode iterations, eliminating stalls.

**Algorithm / Technique:**
- **Chunked Prefills:** Instead of processing the entire prompt in one iteration, breaks it into fixed-size chunks (e.g., 512 tokens). Each chunk is processed in a separate iteration.
- **Decode-maximal Batching:** Each iteration processes one prefill chunk alongside as many ongoing decode requests as possible ("piggybacking").
- **Stall-free Scheduling:** New requests join a running batch without pausing ongoing decodes, because their prefill is chunked.
- **Uniform Batch Shape:** Since each iteration has roughly similar compute (one chunk + many decodes), GPU utilization is more consistent.

**Performance Results:**
- 2.6x throughput improvement for Mistral-7B on single A100
- 6.9x improvement for Falcon-180B on 8 A100s vs. Orca and vLLM
- Eliminates tail latency spikes caused by large prefills

**Implications for Kernel Design:**
- Attention kernels must efficiently handle mixed batches: some sequences doing prefill (multi-token Q), others doing decode (single-token Q).
- Chunk size is a tuning parameter that trades prefill latency for decode consistency.
- Ragged-batch attention (variable Q lengths within a batch) must be well-optimized.

---

### 4.6 Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving

- **Authors:** Ruoyu Qin, Zheming Li, Weiran He, Mingxing Zhang, Yongwei Wu, Weimin Zheng, Xinran Xu (Moonshot AI)
- **Date:** June 28, 2024
- **arXiv:** [2407.00079](https://arxiv.org/abs/2407.00079)

**Key Insight:** GPU HBM is the scarcest resource in LLM serving, but CPU DRAM and SSD are underutilized. Mooncake builds a disaggregated KV cache pool that uses CPU DRAM and SSD as extensions of GPU HBM, with a KV-cache-centric scheduler that orchestrates transfers.

**Algorithm / Technique:**
- **Disaggregated KV Cache:** Separates KV cache storage from computation. KV blocks can reside on GPU HBM, CPU DRAM, or SSD, and are transferred on-demand.
- **Conductor Scheduler:** Global scheduler that dispatches requests based on KV cache location, workload, and SLO requirements. Can replicate or migrate KV blocks proactively.
- **Prefill-Decode Separation:** Dedicated pools for prefill and decode, with the KV cache acting as the intermediary.
- **Prediction-based Early Rejection:** Under overload, predicts whether a request can meet SLO before accepting it.

**Performance Results:**
- Up to 525% throughput increase in simulated scenarios while meeting SLOs
- 75% more requests handled in real Kimi workloads
- Powers the production Kimi chatbot (one of China's largest LLM services)

**Implications for Kernel Design:**
- Attention kernels must support KV cache that may be partially on different memory tiers (GPU, CPU, SSD).
- Asynchronous KV cache prefetching into SRAM/HBM should be overlapped with computation.
- KV cache block format must support efficient serialization for cross-device transfer.

---

### 4.7 LServe: Efficient Long-sequence LLM Serving with Unified Sparse Attention

- **Authors:** Shang Yang, Junxian Guo, Haotian Tang, Qinghao Hu, Guangxuan Xiao, Jiaming Tang, Yujun Lin, Song Han (MIT, SJTU)
- **Date:** February 20, 2025 (MLSys 2025)
- **arXiv:** [2502.14866](https://arxiv.org/abs/2502.14866)

**Key Insight:** Long-context serving suffers from both quadratic prefill cost and large KV cache. LServe unifies structured sparse attention patterns for both prefill and decode, converting half of attention heads to "streaming heads" that only attend to sinks + recent tokens.

**Algorithm / Technique:**
- **Streaming Head Conversion:** Identifies attention heads that can be converted to streaming heads (sink + sliding window only) with minimal quality loss.
- **Hierarchical KV Page Selection:** For remaining "full" heads, dynamically selects only the most important KV pages using query-centric similarity scoring.
- **Block-wise Skipping:** Skips computation on unimportant KV blocks entirely, with block sizes aligned to hardware.
- **Unified Framework:** Same sparse pattern applies to both prefill and decode phases.

**Performance Results:**
- 2.9x prefilling speedup and 1.3-2.1x decoding speedup over vLLM
- Maintains long-context accuracy on standard benchmarks

**Implications for Kernel Design:**
- Sparse attention kernels must support heterogeneous head patterns (some heads full, some streaming).
- Block-level skipping requires efficient index-based KV access.
- The selection policy kernel must be fast enough to not negate the savings from skipping.

---

## 5. Speculative Decoding

### 5.1 Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads

- **Authors:** Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen, Tri Dao
- **Date:** January 19, 2024 (ICML 2024)
- **arXiv:** [2401.10774](https://arxiv.org/abs/2401.10774)

**Key Insight:** Speculative decoding typically requires a separate draft model, which is hard to obtain and maintain. Medusa adds lightweight "Medusa heads" (extra linear layers) on top of the base model's hidden states to predict multiple future tokens simultaneously, eliminating the need for a separate draft model.

**Algorithm / Technique:**
- **Multiple Decoding Heads:** Adds K extra linear heads on top of the last hidden state, each predicting the token at position t+1, t+2, ..., t+K.
- **Tree-based Attention:** Constructs a tree of candidate continuations from the Medusa heads' predictions. Each path through the tree is a candidate multi-token sequence.
- **Parallel Verification:** All candidate sequences are verified simultaneously in a single forward pass using a tree attention mask.
- **Medusa-1:** Freezes the base model and trains only the Medusa heads (parameter-efficient, no quality loss).
- **Medusa-2:** Fine-tunes both the base model and Medusa heads together with a combined training objective (higher speedup, requires full fine-tuning).
- **Typical Acceptance:** A modified acceptance criterion that relaxes rejection for tokens within the "typical set" of the distribution.

**Performance Results:**
- Medusa-1: 2.2x speedup without quality degradation
- Medusa-2: 2.3-3.6x speedup
- No separate draft model needed
- Works with existing model weights (Medusa-1)

**Implications for Kernel Design:**
- Tree attention requires specialized attention masks (not just causal triangular). Kernels must support arbitrary tree-structured masks efficiently.
- Verification of multiple candidates in one forward pass requires batched attention with shared prefix.
- The Medusa heads are small linear layers that should be fused with the main model's last layer.

---

### 5.2 EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty

- **Authors:** Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang
- **Date:** January 26, 2024 (ICML 2024)
- **arXiv:** [2401.15077](https://arxiv.org/abs/2401.15077)

**Key Insight:** Autoregression at the feature level (second-to-top-layer hidden states) is much more predictable than at the token level. By using features advanced by one timestep, EAGLE resolves the uncertainty in feature prediction, enabling a lightweight draft model that operates on features rather than tokens.

**Algorithm / Technique:**
- **Feature-level Drafting:** The draft model predicts the next token's second-to-top-layer features, not the next token directly. This is simpler because features are smoother and more predictable than token distributions.
- **Time-shifted Features:** Concatenates the current feature with the next token's embedding (from the previous step), providing temporal context that reduces prediction uncertainty.
- **Lightweight Draft Head:** A single transformer layer that takes the concatenated features and predicts the next feature.
- **Tree-structured Verification:** Builds a tree of candidate token sequences from the draft features and verifies them in a single forward pass through the target model.

**Performance Results:**
- 2.7x-3.5x latency speedup on LLaMA2-Chat 70B
- Doubles throughput while maintaining output distribution
- Lossless: generated text distribution is identical to the base model

### 5.2.1 EAGLE-2: Dynamic Draft Trees

- **Date:** June 24, 2024 (EMNLP 2024)
- **arXiv:** [2406.16858](https://arxiv.org/abs/2406.16858)

**Key Innovation:** EAGLE-1 uses static draft trees. EAGLE-2 observes that the draft model's confidence scores are well-calibrated (approximate acceptance rates), enabling context-aware dynamic draft tree construction.

**Performance Results:**
- Up to 5x speedup (1.3x improvement over EAGLE-1)
- Lossless acceleration

**Implications for Kernel Design:**
- Feature-level speculation enables very lightweight draft models (a single transformer layer) that can run efficiently on the same GPU.
- Tree-structured attention verification requires custom attention masks (same requirement as Medusa).
- Dynamic tree construction requires efficient batch-dynamic attention kernel support.

---

### 5.3 LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding

- **Authors:** Mostafa Elhoushi, Akshat Shrivastava, Diana Liskovich, Basil Hosmer, Bram Wasti, Liangzhen Lai, Anas Mahmoud, Bilge Soran, Anil Kag, Ahmed Aly, et al. (Meta)
- **Date:** April 25, 2024 (ACL 2024)
- **arXiv:** [2404.16710](https://arxiv.org/abs/2404.16710)

**Key Insight:** Instead of using a separate draft model for speculative decoding, use early layers of the same model as the "draft" and later layers for verification. This is "self-speculative decoding" --- no additional model, no additional memory, shared KV cache.

**Algorithm / Technique:**
- **Layer Dropout Training:** Trains with increasing dropout rates for later layers (e.g., layer 1 has 0% dropout, layer 32 has 50%). This makes early layers more capable of standalone prediction.
- **Shared Exit:** All layers share the same LM head (no auxiliary classifiers needed).
- **Self-Speculative Decoding:** During inference, exit early (e.g., after layer 8 of 32) to generate draft tokens. Then verify with the full model, reusing the already-computed early-layer activations.
- **Shared Compute:** The draft and verification phases share all early-layer KV cache and activations.

**Performance Results:**
- Up to 2.16x speedup on summarization (CNN/DM)
- 1.82x on coding tasks
- 2.0x on semantic parsing (TOPv2)
- No additional memory footprint

**Implications for Kernel Design:**
- Early exit requires the ability to branch out of the transformer stack at arbitrary layer depths.
- The verification pass can skip already-computed early layers, requiring layer-selective forward pass support.
- Shared KV cache between draft and verification simplifies memory management.

---

## 6. Quantization

### 6.1 GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers

- **Authors:** Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh
- **Date:** October 31, 2022 (ICLR 2023)
- **arXiv:** [2210.17323](https://arxiv.org/abs/2210.17323)

**Key Insight:** Large language models can be accurately quantized to 3-4 bits post-training using approximate second-order (Hessian) information to determine optimal rounding directions and compensate for quantization error in remaining weights.

**Algorithm / Technique:**
- **Layer-wise Quantization:** Processes one weight matrix at a time, quantizing columns sequentially.
- **Optimal Brain Quantization (OBQ) Extension:** Based on the OBS/OBQ framework, uses the inverse Hessian to determine how to round each weight and how to adjust remaining weights to compensate for quantization error.
- **Lazy Batch Updates:** Instead of updating remaining weights after each column, batches updates in groups of 128 columns for better GPU utilization.
- **Cholesky-based Hessian:** Efficiently computes the needed Hessian rows using Cholesky decomposition.
- **Arbitrary Bit-widths:** Supports any target bit-width (2, 3, 4, 8) with the same algorithm.

**Performance Results:**
- First to run OPT-175B and BLOOM-176B on a single GPU (via 3-4 bit quantization)
- ~4 GPU hours to quantize 175B parameter models
- 3.25x inference speedup on A100, 4.5x on A6000
- Reasonable accuracy even at 2-bit / ternary quantization
- Minimal perplexity increase at 4-bit

**Implications for Kernel Design:**
- INT4/INT3 dequantization must be fused into GEMM kernels for real speedups. Separate dequantize-then-compute is too slow.
- The group quantization pattern (different scale/zero per group of channels) requires kernels that handle per-group parameters.
- Weight-only quantization means the GEMM is W_int * X_fp16 --- requires mixed-precision GEMM kernels.

---

### 6.2 AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration

- **Authors:** Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han
- **Date:** June 1, 2023 (MLSys 2024, Best Paper Award)
- **arXiv:** [2306.00978](https://arxiv.org/abs/2306.00978)

**Key Insight:** Not all weights are equally important. Only ~1% of weights are "salient" (corresponding to channels with large activation magnitudes), and protecting these via per-channel scaling dramatically reduces quantization error. The key is to look at activations, not weights, to determine importance.

**Algorithm / Technique:**
- **Salient Channel Detection:** Identifies channels with large activation magnitudes from a small calibration set. These channels' weights must be preserved more precisely.
- **Per-channel Scaling:** Instead of mixed-precision (which is hardware-unfriendly), scales salient channels up before quantization, effectively allocating more precision to important weights. The inverse scaling is absorbed into the next layer's weight or applied during computation.
- **Grid Search for Scales:** Searches for the optimal per-channel scaling factors that minimize quantization error, using activation statistics.
- **No Backpropagation:** The method is purely analytical --- no gradient computation, no fine-tuning, no reconstruction loss.
- **Hardware-friendly:** Results in standard uniform INT4 weights (same format as GPTQ).

**Performance Results:**
- Preserves generalization across domains/modalities without calibration set overfitting
- Comparable or better quality than GPTQ at INT4
- Works on instruction-tuned models, multimodal models (not just base LMs)
- MLSys 2024 Best Paper Award
- Supported in TensorRT-LLM, llama.cpp, and many other frameworks

**Implications for Kernel Design:**
- AWQ produces standard INT4 weights, usable with the same GEMM kernels as GPTQ.
- The per-channel scaling is folded into weights at quantization time, requiring no kernel changes.
- The insight that 1% of channels dominate quality suggests adaptive precision kernels could be even more efficient.

---

### 6.3 SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models

- **Authors:** Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, Song Han
- **Date:** November 18, 2022 (ICML 2023)
- **arXiv:** [2211.10438](https://arxiv.org/abs/2211.10438)

**Key Insight:** Weight-only quantization (W4A16) doesn't fully exploit INT8 Tensor Cores. Quantizing both weights AND activations (W8A8) would enable INT8 GEMM, but activation outliers make naive activation quantization fail. SmoothQuant migrates the quantization difficulty from activations to weights via a mathematically equivalent per-channel scaling.

**Algorithm / Technique:**
- **Activation Outlier Analysis:** Identifies that activation outliers are concentrated in specific channels (same channels across all tokens), making per-tensor quantization difficult.
- **Smoothing Transformation:** Applies a diagonal scaling matrix that divides activations by per-channel scales and multiplies weights by the same scales. Mathematically equivalent: Y = (X * diag(s)^-1) * (diag(s) * W) = X * W. But now activations are smoother and weights absorb the difficulty.
- **Migration Strength (alpha):** A hyperparameter controlling how much difficulty to shift from activations to weights. alpha=0.5 works well in practice.
- **W8A8 INT8 GEMM:** After smoothing, both weights and activations can be quantized to INT8, enabling use of INT8 Tensor Cores for the entire linear layer.

**Performance Results:**
- 1.56x speedup and 2x memory reduction for LLMs
- Negligible accuracy loss on OPT, BLOOM, GLM, MT-NLG, LLaMA, Falcon, Mistral, Mixtral
- Enables INT8 GEMM for all matrix multiplications in the model

**Implications for Kernel Design:**
- W8A8 enables INT8 Tensor Cores, which have 2x throughput of FP16 on A100.
- The smoothing transformation can be fused into the preceding LayerNorm kernel.
- Per-tensor quantization (vs. per-channel) simplifies the GEMM kernel since a single scale factor applies to the entire output.
- Online activation quantization requires a quantize kernel before each GEMM.

---

### 6.4 QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks

- **Authors:** Albert Tseng, Jerry Chee, Qingyao Sun, Volodymyr Kuleshov, Christopher De Sa
- **Date:** February 6, 2024 (ICML 2024)
- **arXiv:** [2402.04396](https://arxiv.org/abs/2402.04396)

**Key Insight:** At extreme compression (2-3 bits per weight), scalar quantization (like GPTQ) breaks down because the discrete grid is too coarse. Vector quantization using lattice codebooks, combined with incoherence processing to spread outliers, achieves much better accuracy at extreme bit-widths.

**Algorithm / Technique:**
- **Randomized Hadamard Transform (RHT):** Applies a random Hadamard rotation to weight matrices to make them "incoherent" --- spreading outlier magnitudes across all dimensions. This makes the weights more amenable to quantization.
- **E8 Lattice Codebook:** Uses the E8 lattice (optimal 8-dimensional sphere packing) as a vector quantization codebook. Each group of 8 weights is quantized to the nearest E8 lattice point.
- **Hardware-efficient Lookup:** The E8 lattice has highly symmetric structure enabling efficient encoding/decoding via lookup tables.
- **Fine-tuning:** After quantization, performs lightweight fine-tuning to recover lost accuracy.

**Performance Results:**
- State-of-the-art at 2-3 bit quantization (far better than GPTQ at 2-bit)
- QuIP# 3-bit scales better than "lossless" 4-bit methods
- Enables 2-bit models that are actually usable
- Supports fast inference via GPU kernels

**Implications for Kernel Design:**
- Vector quantization requires lookup-based dequantization (table lookup for E8 lattice points), which is fundamentally different from scalar dequant.
- 8-dimensional VQ operates on groups of 8 weights, requiring the GEMM kernel to process weights in groups.
- The Hadamard transform can be fused into the model or applied once at load time.

---

### 6.5 AQLM: Extreme Compression of Large Language Models via Additive Quantization

- **Authors:** Vage Egiazarian, Andrei Panferov, Denis Kuznedelev, Elias Frantar, Artem Babenko, Dan Alistarh
- **Date:** January 11, 2024 (ICML 2024)
- **arXiv:** [2401.06118](https://arxiv.org/abs/2401.06118)

**Key Insight:** Additive quantization (representing each weight vector as a sum of codewords from multiple codebooks) can be adapted for LLMs and achieves state-of-the-art compression at 2 bits per parameter.

**Algorithm / Technique:**
- **Multi-Codebook Quantization:** Represents each weight vector as the sum of M codeword vectors, one from each of M codebooks (e.g., M=2 codebooks with 2^8=256 entries each, using 2 bytes = 2 bits/weight for a group of 8 weights).
- **Input-adaptive Fine-tuning:** Jointly optimizes codebook entries using calibration data, adapting to the specific weight distribution.
- **Cross-block Joint Optimization:** Optimizes codebook parameters across all layers within each transformer block simultaneously.
- **Beam Search Encoding:** Uses beam search (not greedy assignment) to find optimal codeword combinations.

**Performance Results:**
- First Pareto-optimal scheme at <3 bits per parameter
- LLaMA-2 7B at 2-bit: 6.93 perplexity (1.29 better than best prior, 1.81 from FP16)
- LLaMA-2 70B at 2-bit: 3.94 perplexity
- Practical: GPU and CPU implementations match/outperform FP16 speed in smaller memory

**Implications for Kernel Design:**
- Additive quantization requires a sum-of-codebook-lookups operation in the GEMM inner loop, different from both scalar and single-codebook VQ.
- The codebook lookup is memory-bound and benefits from codebook caching in shared memory/registers.
- Multi-codebook VQ can be parallelized across codebooks within a warp.

---

### 6.6 BitNet: The Era of 1-bit LLMs

- **Authors:** Shuming Ma, Hongyu Wang, Lingxiao Ma, Lei Wang, Wenhui Wang, Shaohan Huang, Li Dong, Ruiping Wang, Jilong Xue, Furu Wei (Microsoft Research)
- **Date:** February 27, 2024
- **arXiv:** [2402.17764](https://arxiv.org/abs/2402.17764)

**Key Insight:** LLMs can be trained from scratch with ternary weights {-1, 0, 1} (1.58 bits) and match FP16 model quality at the same scale. This eliminates multiply operations entirely, replacing them with additions and subtractions.

**Algorithm / Technique:**
- **Ternary Weight Training:** Each weight is constrained to {-1, 0, 1} during training using an absmean quantization function.
- **1.58 Bits:** The ternary representation requires log2(3) = 1.58 bits per parameter.
- **Straight-through Estimator:** Gradients pass through the quantization function during backpropagation.
- **Activation Quantization:** Activations are quantized to 8-bit during forward pass.
- **No Multiplication:** The matrix multiplication W*x becomes entirely additions and subtractions (since W is ternary), fundamentally changing the compute requirements.

**Performance Results:**
- Matches FP16 Transformer quality in perplexity and end-task performance at same model size and training tokens
- Significantly lower latency, memory, throughput, and energy consumption
- BitNet b1.58 2B4T: first open-source native 1-bit LLM at 2B parameters

**Implications for Kernel Design:**
- Ternary weights eliminate multiplications entirely: GEMM becomes GEAA (general addition-accumulation).
- New kernel primitives needed: instead of FMA (fused multiply-add), need conditional add/subtract based on ternary value.
- Memory layout can be extremely compressed: 1.58 bits/weight vs. 16 bits for FP16 (10x compression).
- Custom hardware (addition-only accelerators) could further exploit this.

---

## 7. Pruning & Sparsity

### 7.1 SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot

- **Authors:** Elias Frantar, Dan Alistarh
- **Date:** January 2, 2023 (ICML 2023)
- **arXiv:** [2301.00774](https://arxiv.org/abs/2301.00774)

**Key Insight:** Large language models can be pruned to 50-60% sparsity in one-shot (no retraining) by using the same approximate second-order framework as GPTQ, but for pruning instead of quantization.

**Algorithm / Technique:**
- **Layer-wise Pruning:** Processes weight matrices one at a time.
- **Hessian-based Importance:** Uses the inverse Hessian (from calibration data) to determine which weights to prune and how to adjust remaining weights to compensate.
- **Adaptive Mask Selection:** Iteratively selects weights to prune, updating the compensation for remaining weights after each pruning decision.
- **Semi-structured Support:** Extends to N:M sparsity patterns (e.g., 2:4, 4:8) that are accelerated by NVIDIA Tensor Cores.
- **Compatible with Quantization:** Can be combined with weight quantization for compound compression.

**Performance Results:**
- 50% unstructured sparsity with negligible perplexity increase on OPT-175B and BLOOM-176B
- 60% unstructured sparsity with small perplexity increase
- <4.5 hours for 175B+ parameter models
- Removes 100B+ weights while maintaining accuracy

**Implications for Kernel Design:**
- 2:4 sparsity is natively accelerated by A100/H100 Tensor Cores (2x speedup for supported GEMM shapes).
- Unstructured sparsity at 50-60% doesn't provide speedups on current hardware without sparse format support.
- Combining SparseGPT's 2:4 pruning with 4-bit quantization enables both sparse Tensor Core acceleration and reduced memory bandwidth.

---

### 7.2 Wanda: A Simple and Effective Pruning Approach for Large Language Models

- **Authors:** Mingjie Sun, Zhuang Liu, Anna Bair, J. Zico Kolter
- **Date:** June 20, 2023 (ICLR 2024)
- **arXiv:** [2306.11695](https://arxiv.org/abs/2306.11695)

**Key Insight:** The product of weight magnitude and input activation norm (|W| * ||X||) is a better pruning criterion than weight magnitude alone, and it requires no weight update or reconstruction. This is dramatically simpler than SparseGPT while achieving competitive results.

**Algorithm / Technique:**
- **Importance Score:** For each weight w_ij, the importance score is |w_ij| * ||x_j||_2, where x_j is the j-th input feature's activation norm computed from a small calibration set.
- **Per-output Pruning:** Prunes weights independently for each output neuron (row of the weight matrix), selecting the bottom-k by importance score.
- **No Weight Update:** Unlike SparseGPT, does not adjust remaining weights after pruning. The pruned model is used as-is.
- **No Retraining:** No backward passes, no gradient computation, no fine-tuning.

**Performance Results:**
- Significantly outperforms magnitude pruning
- Competitive with SparseGPT (which requires Hessian computation and weight updates)
- Orders of magnitude faster: no Hessian, no reconstruction
- Evaluated on LLaMA and LLaMA-2 across diverse benchmarks

**Implications for Kernel Design:**
- The simplicity of Wanda enables real-time pruning decisions: could potentially prune dynamically per-batch based on activation patterns.
- Same kernel implications as SparseGPT for 2:4 and unstructured sparsity.
- The activation-aware criterion suggests that dynamic sparsity (different mask per input) could be beneficial.

---

## 8. Model Architectures

### 8.1 DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model

- **Authors:** DeepSeek-AI
- **Date:** May 7, 2024
- **arXiv:** [2405.04434](https://arxiv.org/abs/2405.04434)

**Key Insight:** Multi-Head Latent Attention (MLA) compresses the KV cache into a low-dimensional latent vector, drastically reducing KV cache size while maintaining quality. Combined with fine-grained MoE (DeepSeekMoE), this enables strong, economical training and efficient inference.

**Algorithm / Technique:**
- **Multi-Head Latent Attention (MLA):**
  - Compresses K and V into a joint latent vector c_KV = W_DKV * h (down-projection)
  - At attention time, K = W_UK * c_KV, V = W_UV * c_KV (up-projection)
  - Only the latent c_KV is cached (much smaller than full K, V)
  - Decoupled RoPE: applies RoPE to a separate small set of key components, keeping the latent compression compatible with positional encoding
- **DeepSeekMoE:**
  - Fine-grained experts: many small experts rather than few large ones
  - Shared experts: a subset of experts always activated (non-routed), ensuring baseline quality
  - Top-k routing with auxiliary-loss-free load balancing in V3

**Performance Results (V2):**
- 236B total params, 21B activated per token
- 93.3% KV cache reduction vs. standard MHA
- 42.5% training cost savings vs. DeepSeek-67B
- 5.76x max generation throughput improvement
- Top-tier open-source performance with only 21B active params

### 8.1.1 DeepSeek-V3 Technical Report

- **Date:** December 26, 2024
- **arXiv:** [2412.19437](https://arxiv.org/abs/2412.19437)

**Key Additions in V3:**
- 671B total params, 37B activated per token
- Auxiliary-loss-free load balancing for MoE
- Multi-token prediction (MTP) training objective
- FP8 mixed-precision training (first validated at this scale)
- 14.8T tokens pre-training
- Only 2.788M H800 GPU hours for full training (remarkably efficient)

**Implications for Kernel Design:**
- MLA requires custom attention kernels that work with compressed KV (latent projection + up-projection during attention).
- The latent vector is much smaller than full KV, changing the compute-to-memory ratio of attention.
- Fine-grained MoE requires efficient routing and expert dispatch kernels that handle many small experts.
- FP8 training kernels for GEMM with per-tile quantization (1x128 for activations, 128x128 for weights).

---

### 8.2 Mamba: Linear-Time Sequence Modeling with Selective State Spaces

- **Authors:** Albert Gu, Tri Dao
- **Date:** December 1, 2023
- **arXiv:** [2312.00752](https://arxiv.org/abs/2312.00752)

**Key Insight:** Structured State Space Models (SSMs) have linear-time complexity but fail on language because they cannot perform content-based reasoning (parameters are input-independent). Making SSM parameters input-dependent ("selective") solves this, but breaks the efficient convolution-based computation. A hardware-aware parallel scan algorithm recovers efficiency.

**Algorithm / Technique:**
- **Selective State Spaces:** The A, B, C matrices of the SSM are functions of the input: B = Linear(x), C = Linear(x), Delta = softplus(Linear(x)). This allows the model to selectively propagate or forget information.
- **Hardware-aware Parallel Scan:** Since input-dependent parameters break FFT-based computation, Mamba uses a parallel scan (prefix sum) algorithm. The scan is implemented with a custom CUDA kernel that:
  - Loads SSM parameters into SRAM
  - Performs the scan in SRAM, avoiding HBM round-trips
  - Fuses discretization, scan, and output computation into a single kernel
- **Simplified Architecture:** No attention, no MLP blocks. The entire block is: Linear projection -> Conv1d -> Selective SSM -> Linear projection.
- **Parallel Training, Recurrent Inference:** During training, processes the entire sequence in parallel (scan). During inference, processes one token at a time recurrently (constant memory).

**Performance Results:**
- 5x higher throughput than Transformers during inference
- Linear scaling with sequence length (vs. quadratic for attention)
- Mamba-3B matches/outperforms Transformers at same size, matches 2x larger Transformers
- State-of-the-art on language, audio, and genomics

### 8.2.1 Mamba-2: Transformers are SSMs (Structured State Space Duality)

- **Authors:** Tri Dao, Albert Gu
- **Date:** May 31, 2024 (ICML 2024)
- **arXiv:** [2405.21060](https://arxiv.org/abs/2405.21060)

**Key Innovation:** Establishes a theoretical duality between SSMs and attention: a scalar-diagonal SSM is equivalent to causal attention with a 1-semiseparable mask. This unification enables:
- 2-8x faster core layer than Mamba-1
- Architecture uses matrix-valued state (SSD layer) that maps to efficient matrix operations
- Can use Tensor Cores for the SSM computation

**Implications for Kernel Design:**
- Selective SSMs require custom CUDA kernels for the parallel scan, with careful SRAM management.
- The scan kernel must be IO-aware (same philosophy as FlashAttention): minimize HBM accesses.
- Mamba-2's SSD layer can leverage Tensor Cores via structured matrix operations.
- Inference is O(1) memory per step (constant state), eliminating the KV cache entirely.

---

### 8.3 RWKV: Reinventing RNNs for the Transformer Era

- **Authors:** Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, et al.
- **Date:** May 22, 2023 (EMNLP 2023)
- **arXiv:** [2305.13048](https://arxiv.org/abs/2305.13048)

**Key Insight:** Linear attention mechanisms can be formulated as either a Transformer (for parallel training) or an RNN (for efficient inference), achieving the best of both worlds: parallelizable training with constant-time inference.

**Algorithm / Technique:**
- **Receptance Weighted Key Value (RWKV):** A linear attention variant where attention weights decay exponentially with distance.
- **WKV Operator:** The core computation: wkv_t = sum_{i=1}^{t-1} e^{-(t-1-i)*w + k_i} * v_i, where w is a learned channel-wise decay. This can be computed as a recurrence (O(1) per step) or unrolled for parallel training.
- **Time-mixing and Channel-mixing:** Replaces self-attention and FFN with linear-complexity alternatives.
- **Token Shift:** Uses shifted token representations for temporal context.
- **Dual Formulation:** Mathematically equivalent parallel (training) and recurrent (inference) forms.

**Performance Results:**
- Scaled up to 14B parameters (largest dense RNN ever trained at time of publication)
- On-par with similarly-sized Transformers
- Constant inference memory and compute per token (no KV cache growth)
- Efficient training (parallelizable)

**Implications for Kernel Design:**
- The WKV operator requires specialized CUDA kernels for the exponential-decay weighted sum.
- During inference, the recurrent form eliminates KV cache, requiring only constant-size state.
- During training, the parallel form benefits from similar IO-aware optimizations as attention.
- The channel-wise decay (w) means different channels have different effective context lengths.

---

## 9. Positional Encoding & Activation Functions

### 9.1 RoPE: Rotary Position Embedding (RoFormer)

- **Authors:** Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu
- **Date:** April 20, 2021
- **arXiv:** [2104.09864](https://arxiv.org/abs/2104.09864)

**Key Insight:** Position information can be encoded by rotating the query and key vectors in 2D subspaces. This naturally incorporates relative position through the angle between rotated vectors, while using absolute position encoding (the rotation angle).

**Algorithm / Technique:**
- **Rotation Matrix:** For each position m, applies a rotation to pairs of dimensions: (q_{2i}, q_{2i+1}) -> (q_{2i}*cos(m*theta_i) - q_{2i+1}*sin(m*theta_i), q_{2i}*sin(m*theta_i) + q_{2i+1}*cos(m*theta_i))
- **Relative Position via Inner Product:** The dot product q_m^T * k_n depends on (m-n), the relative position, because rotations compose: R(m)^T * R(n) = R(m-n).
- **Base Frequency Scaling:** theta_i = 10000^(-2i/d). Different dimensions have different frequencies, providing multi-scale position information.
- **Long-context Extension:** By modifying the base frequency (NTK-aware scaling, YaRN), RoPE can extrapolate beyond training sequence length.

**Performance Results:**
- Adopted by virtually all modern LLMs: LLaMA, Mistral, Qwen, DeepSeek, etc.
- Natural relative position encoding without explicit relative position bias
- Compatible with KV caching (position is baked into Q, K at computation time)
- Extensible to longer contexts via frequency scaling

**Implications for Kernel Design:**
- RoPE is applied to Q and K before attention, requiring a fused kernel: (load Q/K -> apply rotation -> proceed to attention).
- The rotation can be fused into the QKV projection GEMM's epilogue.
- For KV caching, RoPE must be applied at cache insertion time (position-dependent), not at attention time.
- Dimension-pair structure maps well to GPU warp lane organization (pairs of elements processed together).

---

### 9.2 GLU Variants / SwiGLU: GLU Variants Improve Transformer

- **Authors:** Noam Shazeer
- **Date:** February 12, 2020
- **arXiv:** [2002.05202](https://arxiv.org/abs/2002.05202)

**Key Insight:** Replacing the standard ReLU/GELU FFN with a Gated Linear Unit variant produces better perplexity. SwiGLU (Swish-gated Linear Unit) emerged as the best variant and is now standard in modern LLMs.

**Algorithm / Technique:**
- **Standard FFN:** FFN(x) = W2 * GELU(W1 * x)
- **SwiGLU FFN:** FFN(x) = W2 * (Swish(W1 * x) * (W3 * x))
  - W1 and W3 are two separate "gate" and "up" projections
  - Swish(x) = x * sigmoid(x) is the gating function
  - Element-wise multiply creates the gating mechanism
  - W2 is the down projection
- **Parameter Count:** SwiGLU has 3 weight matrices vs. 2, so the hidden dimension is typically reduced by 2/3 to maintain similar parameter count (e.g., 2/3 * 4d = 8d/3 instead of 4d).

**Performance Results:**
- Consistent perplexity improvements over ReLU and GELU across tasks
- Adopted by LLaMA, Mistral, Qwen, DeepSeek, PaLM, and most modern LLMs
- No computational drawback (same FLOPs with adjusted hidden size)

**Implications for Kernel Design:**
- SwiGLU requires two projections (W1, W3) that can be computed as a single fused GEMM with double output width, then split.
- The element-wise Swish + multiply can be fused into the GEMM's epilogue.
- The three-matrix structure changes the optimal tiling for the FFN kernel.

---

## 10. Mixture of Experts

### 10.1 Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity

- **Authors:** William Fedus, Barret Zoph, Noam Shazeer
- **Date:** January 11, 2021 (JMLR 2022)
- **arXiv:** [2101.03961](https://arxiv.org/abs/2101.03961)

**Key Insight:** Simplifying Mixture-of-Experts routing to select just one expert per token (top-1 routing) --- instead of top-2 or more --- dramatically reduces complexity, communication, and instability while maintaining quality gains from conditional computation.

**Algorithm / Technique:**
- **Top-1 Routing:** Each token is routed to exactly one expert (vs. top-2 in prior work). The router is a simple linear layer + softmax.
- **Capacity Factor:** Each expert has a fixed capacity (max tokens it can process). Overflow tokens are passed through the residual connection unchanged.
- **Load Balancing Loss:** An auxiliary loss encourages uniform distribution of tokens across experts, preventing expert collapse.
- **Selective Precision:** Training instabilities in MoE are mitigated by casting the router to FP32 while keeping experts in BF16.
- **Expert Parallelism:** Different experts reside on different devices. All-to-all communication routes tokens to their assigned expert.

**Performance Results:**
- 7x pre-training speedup over T5-Base/Large at same compute budget
- 4x speedup over T5-XXL
- Successfully trained up to 1.6 trillion parameters
- First demonstration of training MoE with lower precision (BF16)

### 10.1.1 GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding

- **Authors:** Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, Zhifeng Chen
- **Date:** June 30, 2020 (ICLR 2021)
- **arXiv:** [2006.16668](https://arxiv.org/abs/2006.16668)

**Key Innovation:** Automatic sharding system for MoE that annotates a few critical tensors with partitioning policies and lets the compiler automatically propagate sharding decisions. Scaled MoE to 600B parameters on 2048 TPU v3 in 4 days.

**Implications for Kernel Design:**
- MoE routing + dispatch is a critical kernel: must efficiently (1) compute router scores, (2) sort/group tokens by expert, (3) dispatch to expert GEMMs, and (4) gather results.
- Expert GEMMs are "grouped GEMMs" (many small GEMMs with different sizes), requiring specialized batched GEMM kernels.
- All-to-all communication must be overlapped with expert computation.
- Load imbalance across experts causes stragglers; kernels should handle variable-size expert batches.

---

## 11. Distributed & Long-Context

### 11.1 Ring Attention with Blockwise Transformers for Near-Infinite Context

- **Authors:** Hao Liu, Matei Zaharia, Pieter Abbeel
- **Date:** October 3, 2023 (ICLR 2024)
- **arXiv:** [2310.01889](https://arxiv.org/abs/2310.01889)

**Key Insight:** Long sequences can be distributed across devices in a ring topology, where each device computes attention on its local Q against sequentially-arriving K, V blocks from neighboring devices. By overlapping the KV block communication with attention computation, the communication is completely hidden.

**Algorithm / Technique:**
- **Ring Topology:** N devices form a ring. Each device holds a block of the sequence (Q_i, K_i, V_i).
- **Blockwise Attention:** Each device computes attention of its Q block against incoming K, V blocks using FlashAttention-style tiling.
- **Overlapped Communication:** While computing attention with the current K, V block, the device simultaneously sends its K, V to the next device and receives new K, V from the previous device.
- **Online Softmax Merge:** As K, V blocks arrive from around the ring, the device incrementally updates its partial attention output using the online softmax technique.
- **Device-count Scaling:** Maximum sequence length scales linearly with the number of devices: L_max = N * L_per_device.

**Performance Results:**
- Near-infinite context: sequence length scales linearly with device count
- Zero communication overhead (fully overlapped with computation, assuming compute >= transfer time)
- Effective for both training and inference on million-token sequences

**Implications for Kernel Design:**
- Each device runs a modified FlashAttention kernel that processes streaming K, V blocks and incrementally updates outputs.
- The kernel must support the online softmax merge across externally-arriving K, V blocks.
- Double-buffering of K, V blocks is needed: one buffer being computed on, one being received.
- The kernel's block size must be matched to the communication chunk size for full overlap.

---

## 12. Parameter-Efficient Fine-Tuning

### 12.1 LoRA: Low-Rank Adaptation of Large Language Models

- **Authors:** Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (Microsoft)
- **Date:** June 17, 2021 (ICLR 2022)
- **arXiv:** [2106.09685](https://arxiv.org/abs/2106.09685)

**Key Insight:** The weight updates during fine-tuning have low intrinsic rank. Instead of updating the full weight matrix W, inject trainable low-rank decomposition: W + Delta_W = W + B*A, where B is d x r and A is r x d (r << d). Only A and B are trained.

**Algorithm / Technique:**
- **Low-Rank Decomposition:** For a pre-trained weight matrix W (d x d), add Delta_W = B * A where rank r is typically 4-64.
- **Freeze Base Model:** Original W is frozen. Only A (initialized with random Gaussian) and B (initialized to zero, so Delta_W starts at zero) are trained.
- **No Inference Overhead:** At inference, merge W' = W + B*A once, then use W' as a standard weight matrix.
- **Selective Application:** Typically applied to attention projection matrices (Q, K, V, O), sometimes to FFN.
- **Serving Multiple Adapters:** Different LoRA adapters can be hot-swapped by changing only the small B*A matrices.

**Performance Results:**
- 10,000x fewer trainable parameters than full fine-tuning
- 3x less GPU memory
- Quality on-par or better than full fine-tuning (RoBERTa, DeBERTa, GPT-2, GPT-3)
- Adopted universally for LLM adaptation

**Implications for Kernel Design:**
- At inference with merged weights, no kernel changes needed.
- For multi-adapter serving, the kernel must handle: y = (W + B_i * A_i) * x, where the adapter index i varies per request in a batch.
- This can be decomposed: y = W*x + B_i*(A_i*x). The adapter term A_i*x is a small GEMV (r dimensions), and B_i*(A_i*x) is another small GEMV. These can be fused or batched.
- For batches mixing different adapters, grouped GEMM over the adapter terms is needed.

---

### 12.2 QLoRA: Efficient Finetuning of Quantized LLMs

- **Authors:** Tim Dettmers, Artidoro Pagnoni, Aman Srivastava, Ari Holtzman
- **Date:** May 23, 2023 (NeurIPS 2023)
- **arXiv:** [2305.14314](https://arxiv.org/abs/2305.14314)

**Key Insight:** Backpropagation through a 4-bit quantized base model into LoRA adapters preserves full 16-bit fine-tuning quality, enabling fine-tuning of a 65B model on a single 48GB GPU.

**Algorithm / Technique:**
- **4-bit NormalFloat (NF4):** A new data type that is information-theoretically optimal for normally-distributed weights. The 16 quantization levels are chosen to divide the standard normal CDF into 16 equal-probability bins.
- **Double Quantization:** Quantization constants (scales) themselves are quantized to 8-bit, reducing their memory overhead from 0.5 bits/param to 0.127 bits/param.
- **Paged Optimizers:** Uses NVIDIA unified memory to page optimizer states between GPU and CPU, handling memory spikes during gradient checkpointing.
- **Frozen 4-bit Base + FP16 LoRA:** The base model is in NF4, LoRA adapters are in FP16/BF16. During forward: dequantize NF4 to FP16 on-the-fly, apply LoRA. During backward: gradients flow through the dequantization to update LoRA.

**Performance Results:**
- Fine-tune 65B model on a single 48GB GPU (vs. >780GB for full fine-tuning)
- Guanaco 65B: 99.3% of ChatGPT performance on Vicuna benchmark
- 1,000+ models fine-tuned for analysis
- No quality loss vs. full 16-bit fine-tuning

**Implications for Kernel Design:**
- Training kernels must support backward pass through NF4 dequantization (straight-through gradient).
- NF4 dequantization must be fused into the GEMM for both forward and backward passes.
- Double quantization adds another level of indirection in the dequant kernel.
- Memory management (paged optimizers) requires cooperation between CUDA and unified memory subsystems.

---

## 13. Scaling Laws

### 13.1 Scaling Laws for Neural Language Models (Kaplan et al.)

- **Authors:** Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei (OpenAI)
- **Date:** January 23, 2020
- **arXiv:** [2001.08361](https://arxiv.org/abs/2001.08361)

**Key Insight:** Language model loss follows smooth power-law relationships with model size (N), dataset size (D), and compute budget (C), spanning 7+ orders of magnitude. Architectural details (width, depth, head count) have minimal impact within a wide range.

**Key Findings:**
- Loss scales as power law: L(N) ~ N^{-0.076}, L(D) ~ D^{-0.095}, L(C) ~ C^{-0.050}
- Larger models are significantly more sample-efficient
- Optimal strategy: train very large models on modest data (later revised by Chinchilla)
- Architecture shape doesn't matter much (within reasonable ranges)

---

### 13.2 Training Compute-Optimal Large Language Models (Chinchilla / Hoffmann et al.)

- **Authors:** Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Elias Rutherford, et al. (DeepMind)
- **Date:** March 29, 2022 (NeurIPS 2022)
- **arXiv:** [2203.15556](https://arxiv.org/abs/2203.15556)

**Key Insight:** Kaplan et al.'s recommendation to train very large models on modest data was wrong. For compute-optimal training, model size and training tokens should scale equally: double the model, double the data.

**Key Findings:**
- Current LLMs (at time of publication) are significantly undertrained
- Optimal ratio: ~20 tokens per parameter (e.g., 70B model needs ~1.4T tokens)
- Chinchilla (70B, 1.4T tokens) uniformly outperforms Gopher (280B, 300B tokens) at same compute budget
- Also outperforms GPT-3 (175B), Jurassic-1 (178B), and Megatron-Turing NLG (530B)

**Implications for Kernel Design:**
- Chinchilla-optimal models are smaller but train on more data, shifting the bottleneck from model-size to data throughput.
- Training kernel efficiency (GEMM, attention, communication) directly translates to $ savings because optimal training consumes the full compute budget.
- Inference-time optimization is more important for Chinchilla-optimal models (smaller models are deployed more widely).
- These scaling laws inform hardware procurement: the ratio of compute to memory bandwidth matters.

---

## 14. Kernel Frameworks & DSLs

### 14.1 Liger Kernel: Efficient Triton Kernels for LLM Training

- **Authors:** Pin-Lun Hsu, Yun Dai, Vignesh Kothapalli, Qingquan Song, Shao Tang, Siyu Zhu, Steven Shimizu, Shivam Sahni, Haowen Ning, Yanning Chen
- **Date:** October 14, 2024 (ICML25 Workshop)
- **arXiv:** [2410.10989](https://arxiv.org/abs/2410.10989)

**Key Insight:** Many transformer operations (RMSNorm, RoPE, SwiGLU, CrossEntropy) are memory-bound and leave significant performance on the table in standard PyTorch/HuggingFace implementations. Fused Triton kernels for these operations yield 20% training throughput increase and 60% GPU memory reduction.

**Algorithm / Technique:**
- **Kernel Fusion:** Fuses multi-step operations into single GPU kernels:
  - FusedLinearCrossEntropy: combines the final linear projection + cross-entropy loss computation, avoiding materialization of the full logit tensor
  - Fused RMSNorm + residual addition
  - Fused SwiGLU (gate, up-projection, and activation in one kernel)
  - Fused RoPE application
- **Input Chunking:** For operations with large intermediate tensors (like cross-entropy with large vocabulary), processes in chunks to bound peak memory.
- **Triton Implementation:** Written in OpenAI Triton for accessibility and portability (vs. raw CUDA).
- **Drop-in Replacement:** Compatible with HuggingFace, FlashAttention, FSDP, DeepSpeed.

**Performance Results:**
- 20% average training throughput increase
- 60% GPU memory reduction
- Up to 80% memory savings for post-training (DPO, ORPO, etc.)
- Works out-of-the-box with popular frameworks

**Implications for Kernel Design:**
- The "long tail" of non-attention operations (norms, activations, loss computation) collectively accounts for significant overhead.
- FusedLinearCrossEntropy is particularly impactful: the logit tensor (batch * seq * vocab) is often the largest tensor in training.
- Triton provides an accessible path for writing fused kernels without CUDA expertise.
- Chunking large intermediate computations is a general technique for bounding peak memory.

---

### 14.2 ThunderKittens: Simple, Fast, and Adorable AI Kernels

- **Authors:** Benjamin F. Spector, Simran Arora, Aaryan Singhal, Daniel Y. Fu, Christopher Re (Stanford / Hazy Research)
- **Date:** October 27, 2024
- **arXiv:** [2410.20399](https://arxiv.org/abs/2410.20399)

**Key Insight:** GPU kernel development is needlessly complex. By centering abstractions around the 16x16 matrix tile (the fundamental unit of Tensor Core computation), a simple embedded DSL can express high-performance AI kernels with PyTorch-like syntax while matching or exceeding hand-tuned CUDA.

**Algorithm / Technique:**
- **Tile-centric Abstraction:** The basic data structure is a 16x16 tile (register-level or shared-memory-level). All operations are defined over tiles.
- **Three-level Hierarchy:**
  - **Warp-level:** 16x16 tiles, Tensor Core operations (WGMMA on Hopper), warp-level primitives
  - **Thread-block level:** Shared memory tiles, warp coordination, synchronization
  - **Grid-level:** HBM tiles, kernel launch configuration
- **Automatic Layout Management:** TK picks optimal memory layouts to minimize bank conflicts and maximize Tensor Core compatibility.
- **Producer-Consumer Template:** Built-in support for warp-specialized asynchronous patterns (producer warps load data, consumer warps compute).
- **Embedded in C++:** TK is a header-only C++ library, compiled with nvcc. No new language or compiler.

**Performance Results:**
- Matches CuBLAS on GEMM
- Matches FlashAttention-3 on attention inference
- 10-40% faster than best baselines on attention backward pass
- 8x faster on state space models (Mamba)
- 14x faster on linear attention
- In production at ML inference providers and HFT firms

**Implications for Kernel Design:**
- The 16x16 tile is the fundamental unit of modern GPU kernel design. All abstractions should map to this primitive.
- Warp specialization (producer/consumer) is the key pattern for Hopper and should be a first-class abstraction.
- Bank conflict avoidance and layout optimization can be automated, reducing kernel development effort.
- A well-designed DSL can achieve expert-level performance with dramatically less code and expertise.

---

## Cross-Cutting Summary: Key Themes in GPU Kernel Optimization

### Memory Hierarchy is King
Every breakthrough paper in this collection (FlashAttention, Mamba, KIVI, StreamingLLM) optimizes for the GPU memory hierarchy. The pattern is consistent: identify which level of memory (HBM, L2, SRAM, registers) is the bottleneck, and restructure the algorithm to minimize data movement at that level.

### The Tiling Paradigm
From FlashAttention's SRAM tiling to ThunderKittens' 16x16 tiles to Ring Attention's device-level blocking, tiling is the universal technique. The tile size must match hardware: 16x16 for Tensor Cores, SRAM capacity for FlashAttention blocks, network bandwidth for Ring Attention blocks.

### Asymmetric Phases Demand Asymmetric Optimization
Prefill (compute-bound, parallel) and decode (memory-bound, sequential) require fundamentally different kernels, scheduling, and even hardware. This is recognized by Flash-Decoding, DistServe, Splitwise, Sarathi-Serve, and Mooncake.

### Compression Everywhere
Modern inference stacks compress at every level:
- **Weights:** GPTQ, AWQ, SmoothQuant, QuIP#, AQLM, BitNet (2-4 bit)
- **KV Cache:** KIVI, SnapKV, H2O, StreamingLLM, xKV (2-bit, eviction, SVD)
- **Attention:** SageAttention, FlashAttention-3 FP8 (INT8/FP8)
- **Architecture:** GQA (shared KV heads), MLA (latent compression), MoE (sparse activation)

### Hardware-Software Co-design
Each GPU generation demands new kernel strategies: FlashAttention for Ampere, FlashAttention-3 for Hopper, ThunderKittens for both. The trend toward warp specialization, async TMA, and FP8 Tensor Cores means kernels must be rewritten per-generation to capture new capabilities.

### The Serving Stack is the Product
Raw kernel speed is necessary but insufficient. The serving system (scheduling, memory management, batching, caching) determines end-to-end throughput. Orca's continuous batching, vLLM's PagedAttention, SGLang's RadixAttention, and Mooncake's disaggregated KV cache are as important as the underlying attention kernel.

---

*Last updated: March 2026. This document covers papers through early 2025.*
