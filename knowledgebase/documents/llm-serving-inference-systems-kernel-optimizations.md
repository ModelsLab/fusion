# LLM Serving & Inference Systems: Comprehensive Kernel-Level Optimization Reference

## Table of Contents

1. [vLLM](#1-vllm)
2. [SGLang](#2-sglang)
3. [TensorRT-LLM](#3-tensorrt-llm)
4. [llama.cpp / GGML](#4-llamacpp--ggml)
5. [ExLlamaV2 / ExLlamaV3](#5-exllamav2--exllamav3)
6. [DeepSpeed-Inference / DeepSpeed-FastGen](#6-deepspeed-inference--deepspeed-fastgen)
7. [MLC-LLM / Apache TVM](#7-mlc-llm--apache-tvm)
8. [NVIDIA Triton Inference Server](#8-nvidia-triton-inference-server)
9. [Hugging Face TGI](#9-hugging-face-tgi-text-generation-inference)
10. [Ollama](#10-ollama)
11. [ONNX Runtime](#11-onnx-runtime)
12. [FlashAttention (v1/v2/v3)](#12-flashattention-v1v2v3)
13. [FlashInfer](#13-flashinfer)
14. [Key Serving Concepts](#14-key-serving-concepts)

---

## 1. vLLM

### Architecture Overview

vLLM is a high-throughput, memory-efficient inference and serving engine for LLMs. Its core innovation is PagedAttention, which treats GPU memory like an operating system's virtual memory, dividing KV cache into fixed-size blocks that can be allocated on demand.

**Core Components:**
- **API Server**: FastAPI-based, mounts OpenAI-compatible endpoints (`/v1/completions`, `/v1/chat/completions`) served via Uvicorn
- **Scheduler**: Maintains `waiting` and `running` queues; supports FCFS and priority-based selection
- **KV-Cache Manager (Block Manager)**: Heart of paged attention; maintains a `free_block_queue` pool with hundreds of thousands of available blocks
- **Model Executor**: `UniProcExecutor` wraps a single `Worker` per GPU; supports eager mode (standard PyTorch) and captured mode (CUDA graph replay)
- **Model Runner**: Handles forward pass execution, input preparation, and attention metadata construction

### Scheduler Implementation

The V1 scheduler treats both prefill and decode in a unified manner:

- **Prioritizes decode requests** (those already in the `running` queue) to minimize inter-token latency
- Uses a simple dictionary to dynamically allocate a **fixed token budget** per request
- Request lifecycle: `WAITING` -> `RUNNING` -> completion with KV-cache cleanup
- **Preemption**: Can temporarily pause lower-priority requests or reorder tokens to maintain consistent throughput
- **Dynamic Load Balancing**: Distributes work based on GPU memory load and active sequence length
- **Scoring for request routing** (distributed): `len(waiting) * 4 + len(running)` -- picks engine with minimal score

### Block Manager & KV-Cache Management

**Block Structure:**
- Block size calculation: `2 x block_size (default=16) x num_kv_heads x head_size x dtype_bytes`
- `allocate_slots` function: determines required blocks via `ceil(new_tokens / block_size)`, checks availability, fetches from pool
- `req_to_blocks` dictionary maps request IDs to allocated block lists

**Memory Utilization:**
- GPU utilization frequently exceeds 90% (vs <70% in traditional systems)
- Controlled via `gpu_memory_utilization` parameter (default 0.8)
- KV cache reduces fragmentation to near-zero through paged allocation

### PagedAttention v1 and v2

PagedAttention divides KV cache into non-contiguous fixed-size blocks. The custom attention kernel (`csrc/attention/attention_kernels.cu`) accesses KV memory through block tables via `slot_mapping` constructed during forward pass input preparation.

**Key Operations:**
1. Copy buffers from CPU to GPU
2. Compute position indices
3. Build `slot_mapping` for non-contiguous KV block access
4. Execute paged attention kernels

The system flattens sequences into a single concatenated "super sequence" with position indices and attention masks ensuring each sequence attends only to its own tokens.

### Continuous Batching Algorithm

vLLM implements **token-level batching** where each GPU kernel invocation processes multiple tokens from multiple sequences, even if those sequences are at different stages of generation. New requests can be inserted into an ongoing batch dynamically; completed requests are immediately removed without blocking others.

### Speculative Decoding

vLLM V1 implements three proposal schemes (no separate draft LLM required):

**N-gram Drafting:**
- Takes the last `prompt_lookup_max` tokens; finds a prior match in the sequence
- If found, proposes the `k` tokens that followed that match
- Decrements search window to `prompt_lookup_min` on failure

**EAGLE:**
- Performs "model surgery" on the target LM: keeps embeddings and LM head, replaces transformer stack with lightweight MLP
- Fine-tunes the MLP as a cheap draft model
- EAGLE-2 uses confidence scores to dynamically adjust draft tree structure
- EAGLE-3 attaches a lightweight autoregressive prediction head to target model's internal layers

**Medusa:**
- Trains auxiliary linear heads on top of embeddings (before LM head) to predict next `k` tokens in parallel
- Multiple heads produce parallel predictions, verified via tree attention

**Verification Algorithm (all methods):**
- Process draft tokens left-to-right
- Accept token if `p_large(token) >= p_draft(token)`
- Otherwise accept with probability `p_large / p_draft`
- Guarantees statistically equivalent distribution to standard autoregressive decoding
- Implemented via Triton-based rejection sampling kernel

### Chunked Prefill

- Caps new tokens per step at `long_prefill_token_threshold`
- A prompt with 25 tokens split into 8-token chunks requires 3+ engine steps
- Token sampling occurs only in the final chunk
- Can be combined with speculative decoding's dynamic speculation length

### Prefix Caching

- Hashes request token sequences into `BlockHash` objects (combining previous hash, current tokens, optional metadata)
- `find_longest_cache_hit` performs linear search against `cached_block_hash_to_block` dictionary
- Reference counting prevents premature deallocation
- Blocks become invalid only when about to be reallocated while containing stale hash associations
- Enabled by default via `enable_prefix_caching` parameter

### Tensor Parallelism & Pipeline Parallelism

**Tensor Parallelism (TP):**
- Shards model across multiple GPUs on the same node
- `MultiProcExecutor` spawns daemon processes per rank; rank 0 is the driver
- Workers execute lockstep: if any DP replica has work, all replicas execute; replicas without requests perform dummy steps

**Pipeline Parallelism (PP):**
- Partitions model layers across nodes when TP alone is insufficient
- Used for multi-node serving scenarios

**Distributed Architecture:**
- Headless nodes run `CoreEngineProcManager` spawning `DPEngineCoreProc` processes
- Input/output/main threads block on queues or ZMQ DEALER sockets
- API server creates `DPLBAsyncMPClient` with asyncio tasks for load balancing

### Custom CUDA Kernels

| Kernel | Description |
|--------|------------|
| `attention_kernels.cu` | Multi-head query attention for paged KV caches |
| PagedAttention kernel | Non-contiguous KV memory access via block tables |
| Rotary Embedding (RoPE) | Triton kernel for position encoding (optimized with huge performance gain) |
| Fused Add RMSNorm | `fused_add_rmsnorm` for combined residual + normalization |
| FlashInfer `concat_mla_k` | MLA key projection concatenation |
| FP8 GEMM kernels | Optimized attention projections for FP8 |
| TRTLLM-Gen kernels | On-the-fly dequantization within tensor cores for FP4 |
| Triton rejection sampling | Speculative decoding accept/reject logic |
| xgrammar bitmask expansion | Grammar-constrained decoding logit masking |
| CUDA Graph capture | Pre-baked GPU work DAGs for warmup batch sizes |

### Quantization Backends

**FP8 (W8A8):**
- All Linear modules (except `lm_head`) quantized to FP8_E4M3 with per-tensor scale
- Hopper/Ada: W8A8 with hardware support
- Turing/Ampere: W8A16 weight-only via Marlin kernels

**AWQ:**
- Uses official AWQ kernel as default
- Marlin variant selected automatically at runtime when hardware supports it
- Better for small batch sizes (memory-bound scenarios)

**GPTQ:**
- Uses ExLlamaV2 kernel by default
- Marlin/Machete kernels available for larger batch sizes
- Per-channel weight quantization

**Additional Methods:**
- AQLM, SqueezeLLM, bitsandbytes (NF4, FP4), EETQ, EXL2
- LLM Compressor integration for GPTQ, SmoothQuant, SparseGPT, RTN

### Multi-LoRA Serving

- Enabled via `option.enable_lora = true`
- Multiple LoRA adapters served concurrently through the adapters API
- LoRA weights stored efficiently alongside base model weights

### Performance Tuning Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gpu_memory_utilization` | 0.8 | Target VRAM fraction |
| `block_size` | 16 | KV-cache block token count |
| `max_num_seqs` | varies | Maximum concurrent sequences |
| `enforce_eager` | false | Disable CUDA graphs for debugging |
| `long_prefill_token_threshold` | varies | Enables chunked prefill |
| `enable_prefix_caching` | true | Toggle prefix caching |
| `temperature`, `top_p`, `top_k` | varies | Sampling configuration |

**Benchmarking Metrics:**
- **TTFT** (time-to-first-token): Request submission to first output token
- **ITL** (inter-token latency): Time between consecutive tokens
- **TPOT** (time-per-output-token): Average ITL across all outputs
- **Goodput**: Throughput meeting SLO constraints (e.g., p99 e2e latency < 500ms)

### Guided Decoding (Structured Output)

- `StructuredOutputManager` maintains `_grammar_bitmask` tensors (32-bit integers per `vocab_size/32`)
- Grammars compile to FSM backends (e.g., xgrammar, LLGuidance)
- Post-forward pass: expands bitmask to vocab size (32x expansion ratio) and masks disallowed logits to negative infinity
- Request lifecycle: `WAITING_FOR_FSM` -> `WAITING` after compilation
- `accept_tokens` advances FSM state left-to-right through sampled tokens

---

## 2. SGLang

### Architecture Overview

SGLang is a high-performance serving framework for LLMs and multimodal models, deployed on 400,000+ GPUs worldwide. It provides low-latency, high-throughput inference from single-GPU to distributed clusters.

**Core Components:**
- **Frontend API Server**: Handles HTTP requests and tokenization
- **Tokenizer Server**: Separate tokenization service
- **Backend Scheduler**: Per-GPU scheduler for batch management
- **RadixCache**: Radix tree-based KV cache manager
- **sgl-router**: Intelligent request routing component

### RadixAttention: Prefix Tree for KV Cache Reuse

RadixAttention automates KV cache reuse through a radix tree data structure:

**Implementation Details:**
- `TreeNode` class with `children = defaultdict(TreeNode)` for dynamic prefix matching
- **Longest matching prefix** detection by sequential tree traversal until no matching child exists
- Variable-length prefixes (unlike vLLM's fixed 16-token blocks)
- O(prefix_length) lookup complexity vs O(1) per block in vLLM
- Cache nodes store KV tensors along with `last_access_time` and `hit_count` for LRU eviction

**Example:** Request "Hello, how are you?" creates initial cache node. Subsequent "Hello, what's your name?" branches from shared "Hello, " prefix, creating independent child nodes for diverging sequences.

**Performance Impact:**
- Few-shot learning: 85-95% cache hit rates (vs vLLM's 15-25%)
- Multi-turn chat: 4.7x faster throughput
- Overall: 3.0x-5.0x speedup over vLLM across workloads

### KV Cache Management

- `req_to_token_pool` and `token_to_kv_pool` components manage memory allocation
- `ApproximateRadixTree` structures at router level for cache-aware load balancing predictions
- Page reuse via hashing: identical prompt prefixes share cache pages
- Reference counting for deallocation; defragmentation compacts fragmented memory

### Scheduler Algorithm

**OverlapScheduler (v0.4+):**
- Achieves near-zero CPU overhead through overlapped execution
- Maintains both `current_batch` and `next_batch_future` references
- While GPU executes current batch and records `torch.cuda.Event()`, CPU asynchronously prepares the next batch
- Reduces CPU overhead from 15-25% to less than 2%
- GPU utilization: 95-98% (vs 78-85% with traditional schedulers)

**Scheduling Loop:**
1. Request parsing and IR conversion
2. Batch preparation with zero-overhead CPU processing
3. GPU forward pass with constraint application
4. Cache updates

### Constrained Decoding (Grammar-Guided)

- **XGrammarBackend** creates FSMs from grammar specifications
- JSON schema compilation: `compile_json_schema(grammar_str)` generates FSM states
- Compressed FSM: multi-token paths compressed into single-step paths for faster decoding
- Generates bitmask for allowed tokens at each step, restricting vocabulary to valid continuations
- 99.8% validity rate (vs 78% with post-processing approaches)
- Backends: XGrammar and LLGuidance

### Tensor Parallelism

- Integrated with FSDP and TP for large-scale training strategies
- Supports tensor, pipeline, expert, and data parallelism
- SpecForge enables efficient scaling across GPU clusters

### FlashInfer Integration

- Uses **FlashAttention-3 for prefill** and **FlashInfer for decode** on Hopper architecture
- JIT-compiled kernels with TVM FFI for Python binding
- Integration achieved median inter-token latency drops of 29-69%
- Up to 21% improvement in time-to-first-token on Llama-3 models

### Speculative Decoding

- **Multiple Token Prediction (MTP)**: Predicts multiple draft tokens with a lightweight draft model, verifies in parallel
- Up to 60% higher output throughput for DeepSeek V3 without quality loss
- Llama 4 Maverick draft model: 2.18x speedup on MT-Bench
- Scout variant: 2.0x acceleration
- **SpecForge**: Accelerated speculative decoding training

### Chunked Prefill

- Manages memory usage during prefill phase
- Splits long prompts into chunks processed across multiple forward passes
- Essential for serving long-context requests without OOM

### Disaggregated Serving

- Encode-Prefill-Decode (EPD) Disaggregation with Mooncake as transfer backend
- Decouples compute-intensive multimodal encoders from language model nodes
- Prefill-decode separation across different GPU pools

### Multi-Node Serving (sgl-router)

- Intelligent request routing based on RadixAttention cache hit predictions
- Maintains approximate radix trees per worker node
- Routing score: `cache_hit_rate * 2.0 - load_factor`
- Cache-aware routing yields 1.9x throughput improvement with 3.8x higher cache hit rates vs round-robin
- Workers report cache updates to `DistributedCacheTracker`
- Balances cache hit rate (70% weight) against current load (30% weight)

---

## 3. TensorRT-LLM

### Architecture Overview

TensorRT-LLM is NVIDIA's open-source library for optimized LLM inference, architected on PyTorch with a high-level Python API supporting single-GPU to distributed multi-node deployments.

**Core Components:**
- **Builder**: Compiles model definitions into optimized TensorRT engines
- **Executor**: C++ runtime orchestrating inference execution
- **Model Definition**: PyTorch-native code for customizable model architectures
- **Plugin System**: Extensible kernel integration framework

### Plugin System for Custom Kernels

TensorRT-LLM uses a plugin architecture to integrate custom CUDA kernels:
- Plugins wrap hand-optimized CUDA kernels for attention, GEMM, and normalization
- `gpt_attention` plugin implements MHA, MQA, and GQA via `tensorrt_llm.functional.gpt_attention`
- Kernel fusion combines LayerNorm, matrix multiplications, bias additions, and activation functions into single CUDA kernels
- Eliminates intermediate tensor materialization and reduces kernel launch overhead

### In-Flight Batching

- Adds new requests to running batches at each generation step
- Processes context and generation phases concurrently
- Requires input tensors to be packed (no padding) for efficiency
- Constraint: sequences in context phase must precede generation phase sequences in the input tensor
- Maximizes GPU utilization by keeping compute units busy as individual requests complete

### KV Cache Management

**Contiguous KV Cache:**
- Single monolithic tensor: `[max_batch_size * max_beam_width, 2, num_heads, max_seqlen, hidden_dim_per_head]`
- Inefficient for variable-length sequences

**Paged KV Cache:**
- Decomposes KV cache into blocks distributed by cache manager during processing
- Block sizes: 8, 16, 32, 64, 128 tokens
- Allocates and recycles blocks from a pool dynamically
- Enables prefix sharing among requests with common prefixes
- Near-zero memory waste from internal fragmentation
- Requires ~60% more VRAM than model weights alone

**INT8/FP8 KV Cache:**
- Per-tensor scaling factors stored in `kv_cache_scaling_factor` tensor
- Dequantization occurs on-the-fly in MHA kernels
- FP8 KV cache enables 2-3x larger batch sizes on H100
- 1.5x performance gain vs FP16

### Custom Attention Kernels

**Context Phase (FMHA):**
- Disabled mode: stores intermediate Q*K^T tensor (slowest, highest memory)
- Enabled mode: single-kernel FMHA using vanilla MHA for short sequences or Flash Attention for longer
- FP8 Context FMHA: `use_fp8_context_fmha` for Hopper/Ada GPUs only
- Paged Context FMHA: when both `use_fp8_context_fmha` and `use_paged_context_fmha` enabled

**Generation Phase (Masked MHA):**
- Processes one token per sequence
- Fuses QKV bias addition, RoPE computation, dequantization/quantization
- **Multi-Block Mode**: distributes work across multiple CUDA thread-blocks when `batch_size * num_heads` is small relative to GPU multiprocessors; enabled by default since v0.13

**XQA Kernel (for MQA/GQA):**
- Optimized for generation phase of MQA and GQA
- Uses tensor cores for acceleration
- Reduces data loading and conversion overhead
- Supports beam search optimization
- Performance: 2.4x throughput increase on H200 single GPU (1,227 to 2,941 tokens/sec for Llama-70B)
- 8-GPU: 1.9x throughput increase (13,232 to 25,300 tokens/sec)
- Supported: FP16, BF16 compute; FP16, BF16, FP8, INT8 KV cache types
- Hopper qgmma kernel added to XQA JIT codepath
- Disable with `--disable_xqa`; force with `TRTLLM_FORCE_XQA=1`

### Quantization Methods

**FP8 (W8A8) - Primary Choice:**
- Optimal performance/accuracy balance on Hopper/Blackwell
- Per-tensor scaling factors via calibration across representative samples
- Ada supports with reduced efficiency
- LLaMA-v2-7B batch 1: 1.51x speedup vs FP16

**INT8 SmoothQuant (W8A8):**
- Migrates quantization difficulty from activations to weights
- Recommended as Ada fallback or when FP8 insufficient
- Medium performance improvement, slightly higher accuracy loss than FP8

**INT4-AWQ (W4A16):**
- Activation-aware weight quantization compressing to 4 bits
- Excels in small-batch scenarios (batch size <= 4) where inference is memory-bound
- Performance advantage diminishes at larger batches

**INT4-GPTQ:**
- Alternative 4-bit compression using per-channel weight optimization
- For memory-constrained scenarios

**INT4-FP8 AWQ (W4A8):**
- Combines INT4 weight quantization with FP8 activation quantization

**FP4 (Blackwell B200):**
- Next-generation quantization for Blackwell architecture

### Positional Embeddings

- **RoPE**: Fused into attention when `rotary_embedding_dim > 0`; supports GPT-NeoX and GPT-J forms
- **ALiBi**: Computes bias on-the-fly from ALiBi slopes within the optimized kernel
- **Relative Attention Bias**: Pre-computed or computed on-the-fly via `max_distance`

### Chunked Context

- Splits context into chunks to batch with more generation tokens
- Requires FMHA paged KV-cache enabled
- Context chunk sizes (except last) must be integer multiples of KV-cache block size

### Sliding Window / Cyclic KV Cache

- Circular buffer storing only last N tokens (`max_attention_window_size`)
- New tokens overwrite least recently used entries when cache fills
- Supports per-layer configuration via tensor/list/vector

### StreamingLLM

- Maintains N recent tokens plus S sink tokens
- Context phase uses dense self-attention on all tokens but saves only N to KV cache
- Adjusts relative position embedding to cache positions
- Enabled via `streamingllm` flag

### Multi-GPU: Tensor, Pipeline, Expert Parallelism

- **TP**: Splits each layer across GPUs; requires all-reduce at each layer boundary
- **PP**: Partitions layers across pipeline stages
- **EP**: Expert parallelism for MoE models across GPUs/nodes

### Speculative Decoding

- Supports EAGLE, MTP, and N-Gram algorithms
- Integrated with XQA kernel (numerical stability fixes applied)

### Performance Benchmarks

- **Peak**: 10,000+ output tokens/sec on H100 with FP8
- **TTFT**: Sub-100ms
- **vs vLLM**: 1.34x higher throughput (short sequences), 2.72x better TPOT (long sequences)
- **vs PyTorch**: 4x throughput improvement

### Engine Build Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `max_batch_size` | 256 | Production: increase to 2048 |
| `max_num_tokens` | 8192 | Tokens per batch iteration |
| `use_paged_context_fmha` | varies | Enable paged attention |
| `--disable_xqa` | false | Disable XQA kernel |
| `context_fmha_type` | enabled | Context phase FMHA mode |
| `remove_input_padding` | true | Packed (non-padded) mode |

### Memory Estimation

70B model in FP8: ~220GB total = 70GB weights + 120GB KV cache (batch 256, seq 8192) + 30GB overhead

---

## 4. llama.cpp / GGML

### GGML Tensor Library Internals

GGML is a general-purpose tensor library providing the foundational compute abstraction for llama.cpp.

**Core Structures:**
- `ggml_tensor`: Multi-dimensional array with shape and type metadata
- `ggml_context`: Memory arena for efficient allocation
- `ggml_cgraph`: Directed acyclic graph of operations
- `ggml_backend`: Hardware-agnostic execution through pluggable implementations

**Computation Model:**
- Context-based memory arenas for efficient allocation
- Computation graph builder constructs operation sequences minimizing data movement
- Backend scheduler (`ggml_backend_sched`) dispatches batched operations across available hardware
- Manages memory transfers and kernel launches

### CUDA Backend

**Kernel Options:**
- MMQ (matrix multiplication) kernels vs cuBLAS-based paths (selectable via build flags)
- FlashAttention support for accelerated attention computations
- Architecture coverage: compute capability 6.0+ (Pascal through Hopper)
- Multi-GPU: peer-to-peer memory copies between devices; configurable batch size thresholds

### Metal Backend (Apple Silicon)

- GPU acceleration on Apple Silicon (M1-M4)
- BF16 for Metal operations on M2+ chips
- Embedded shader compilation via `GGML_METAL_EMBED_LIBRARY`
- Dynamic dispatch to compute kernels based on device capability
- Enabled by default on macOS (`GGML_METAL=ON`)

### Vulkan Backend

- Vendor-agnostic GPU access across Linux, Windows, and macOS (via MoltenVK)
- Abstracts device selection and memory management
- Cross-platform alternative to CUDA/Metal

### SYCL Backend

- Intel oneAPI support for Arc and Flex GPUs
- Optional FP16 computation for memory efficiency

### CPU Backend

**SIMD Optimizations:**
- x86: SSE4.2, AVX, AVX2, AVX512, AVX512-VNNI, AMX
- ARM: NEON, SVE, DOTPROD, I8MM, SME (M1-M4 specialized paths)
- RISC-V: RVV, ZVFH, ZFH
- PowerPC: VSX
- KleidiAI integration for ARM CPU acceleration
- Dynamic multi-level optimization loading based on runtime CPU detection

### Memory Mapping (mmap)

- Memory-mapped file I/O for efficient model loading
- Avoids copying entire model into RAM
- OS page cache handles memory management transparently

### GGUF Format

GGUF (General GGML Universal File) is a binary format storing tensors and metadata in a single file:
- Contains model architecture, tokenizer configuration, hyperparameters
- Eliminates need for secondary config.json
- Superseded GGML format (August 2023)
- Designed for fast saving/loading of model data

### Quantization Formats (Complete Reference)

#### IQ Series (Importance-Weighted Quantization)

| Format | Bits/Weight | Size (GiB, 8B model) | Description |
|--------|------------|----------------------|-------------|
| IQ1_S | 2.00 | 1.87 | Ultra-low 1.5-bit importance quantization |
| IQ1_M | 2.15 | 2.01 | Medium 1.5-bit importance quantization |
| IQ2_XXS | 2.38 | 2.23 | Extra-extra-small 2-bit importance |
| IQ2_XS | 2.59 | 2.42 | Extra-small 2-bit importance |
| IQ2_S | 2.74 | 2.56 | Small 2-bit importance |
| IQ2_M | 2.93 | 2.74 | Medium 2-bit importance |
| IQ3_XXS | 3.25 | 3.04 | Extra-extra-small 3-bit importance |
| IQ3_XS | 3.50 | 3.27 | Extra-small 3-bit importance |
| IQ3_S | 3.66 | 3.42 | Small 3-bit importance |
| IQ3_M | 3.76 | 3.52 | Medium 3-bit importance mix |
| IQ4_XS | 4.46 | 4.17 | Extra-small 4-bit importance |
| IQ4_NL | 4.68 | 4.38 | Non-linear 4-bit importance |

#### K-Quant Series (Super-Block Quantization)

| Format | Bits/Weight | Size (GiB, 8B model) | Description |
|--------|------------|----------------------|-------------|
| Q2_K_S | 2.97 | 2.78 | Small 2-bit K-quant |
| Q2_K | 3.16 | 2.95 | Standard 2-bit K-quant |
| Q3_K_S | 3.64 | 3.41 | Small 3-bit K-quant |
| Q3_K_M | 4.00 | 3.74 | Medium 3-bit K-quant |
| Q3_K_L | 4.30 | 4.02 | Large 3-bit K-quant |
| Q4_K_S | 4.67 | 4.36 | Small 4-bit K-quant |
| Q4_K_M | 4.89 | 4.58 | Medium 4-bit K-quant (recommended) |
| Q5_K_S | 5.57 | 5.21 | Small 5-bit K-quant |
| Q5_K_M | 5.70 | 5.33 | Medium 5-bit K-quant |
| Q6_K | 6.56 | 6.14 | Standard 6-bit K-quant |
| Q8_0 | 8.50 | 7.95 | 8-bit symmetric quantization |

#### Traditional Quantization

| Format | Bits/Weight | Size (GiB, 8B model) |
|--------|------------|----------------------|
| Q4_0 | 4.34 | 4.34 |
| Q4_1 | 4.78 | 4.78 |
| Q5_0 | 5.21 | 5.21 |
| Q5_1 | 5.65 | 5.65 |
| F16 | 16.00 | 14.96 |

**K-Quant Architecture:**
- Super-blocks of 256 values with multiple scales and mixed precision
- Adapts to local tensor characteristics for better quality at similar compression
- Q4_K_M is the most popular variant, balancing quality and size at ~4.5 bpw

**IQ Architecture:**
- Importance-weighted quantization using importance matrices
- Preserves critical weights with higher precision based on activation patterns
- Superior quality at ultra-low bit rates (1.5-3 bits)

**TQ Series:**
- Ternary quantization schemes (values limited to -1, 0, 1 with scaling)

### KV Cache Management

- `llama_kv_cache` structure stores key-value pairs from attention layers
- Pre-allocated based on context size
- Enables efficient autoregressive generation without recomputation

### Batch Processing

- `llama_batch` structure groups tokens with position and sequence IDs
- Parallel processing across available hardware
- Backend scheduler handles memory transfers and kernel launches

### Speculative Decoding

- Smaller draft model proposes token sequences
- Target model verifies proposals in single forward pass
- Successful predictions skip expensive inference steps
- Configurable via `--draft` flag with separate model path

### Grammar-Constrained Decoding

- GBNF (GGML BNF) grammars restrict output to valid formats (JSON, SQL, etc.)
- Sampler filters logits to enforce grammar compliance at each step
- Ensures well-formed structured output without post-processing
- Performance impact is minimal due to efficient FSM-based token filtering

---

## 5. ExLlamaV2 / ExLlamaV3

### Architecture Overview

ExLlamaV2/V3 is a Python library designed for efficient inference of large language models on consumer GPUs, prioritizing performance through custom CUDA kernels, advanced quantization, and distributed execution.

**Core Components:**
- ~80+ custom C++/CUDA functions in the `exllamav3_ext` module
- Compiled through PyTorch's C++ extension infrastructure
- Supports 30+ model architectures (Llama, Mistral, Qwen, Gemma, GLM4, Mixtral, Phi, etc.)

### Custom CUDA Kernels

**GEMM Kernels:**
- Quantized matrix multiplication on 16x16 tiles matching GPU architecture
- Optimized for memory bandwidth (quantized operations are memory-bound)
- Efficient tensor core utilization

**Attention Kernels:**
- Flash Attention integration for accelerated computation
- Q/K/V projections with KV cache mechanisms
- Optional 2-8 bit KV cache quantization
- Sliding window attention for Mistral/Gemma2 architectures
- GQA (Grouped Query Attention) support
- Automatic head splitting across devices during tensor parallelism

**RoPE Kernels:**
- Apply rotation matrices to query/key vectors
- Fused with attention computation

**Activation Kernels:**
- SiLU gating with fused operations
- Combined activation and projection in single kernel launch

### EXL2 Quantization Format

Based on GPTQ optimization method:
- Supports 2, 3, 4, 5, 6, and 8-bit quantization
- **Mixed precision**: Different levels within a model and within each layer
- Preserves most important weights and layers with more bits
- Tries different quantization parameters and measures error
- Achieves target average bits per weight given as argument

### EXL3 Quantization Format

Based on the QTIP algorithm from Cornell RelaxML:
- **Bitrate**: 1.6-8+ bits per weight (variable per layer)
- **Tile-based processing**: 16x16 tensor core tiles for optimal GPU utilization
- **Hadamard transform**: Decorrelates weights before quantization
- **Trellis-encoded indices**: Compact representation for compressed weights

**Five-Stage Conversion Pipeline:**
1. **Calibration**: Captures Hessian matrices via forward pass
2. **Hadamard Transform**: Decorrelates weight matrices
3. **LDL Decomposition**: Optimizes quantization parameters
4. **Tile-Based Quantization**: Processes in 16x16 blocks
5. **Trellis Encoding**: Compresses indices

### Dynamic Batching & Streaming

- Generator orchestrates job management through continuous batching
- Queue of generation requests with per-job `stop_conditions`, `banned_strings`, `token_healing`
- Results stream asynchronously with `time_prefill` and `time_generate` metrics

### Cache Management

- PageTable allocates KV cache in fixed-size pages with intelligent reuse
- **Page reuse via hashing**: Identical prompt prefixes share cache pages (prefix caching)
- Reference counting for deallocation
- Defragmentation compacts fragmented memory to maximize contiguous free space
- 2-8 bit KV cache quantization to reduce memory footprint

### Tensor Parallelism

- Weights split along output dimension for column-parallel or input dimension for row-parallel
- Zero-copy tensor transfer between processes via shared memory
- NCCL-based all-reduce with AVX2 fallback for CPU operations
- Expert parallelism for MoE model distribution across GPUs

### Speculative Decoding

- Integrated into the dynamic generator API
- Draft model proposes tokens; target model verifies
- Combined with all inference and sampling features

---

## 6. DeepSpeed-Inference / DeepSpeed-FastGen

### SplitFuse Approach

Dynamic SplitFuse is a novel token composition strategy for prompt processing and token generation:

**Two Key Behaviors:**
1. **Long Prompt Decomposition**: Split into smaller chunks, scheduled across multiple forward passes; only the final pass performs generation
2. **Short Prompt Composition**: Composed to exactly fill a target token budget; even short prompts may be decomposed to ensure precise budget alignment

### Three Performance Insights

1. **Token-Centric Scheduling**: Token count (not batch composition) predominantly determines latency, enabling single-signal scheduling optimization
2. **Throughput Saturation**: Two operating regions:
   - Memory-bound (small token counts, scaling throughput)
   - Compute-bound (large counts, near-constant throughput)
   - Optimal: consistently operate in the throughput-saturating region
3. **Concave Function Optimization**: Token-throughput curve is concave; for a given token pool, evenly splitting between batches maximizes throughput

### Dynamic SplitFuse Batching

Unlike competing systems:
- vLLM: performs either token generation OR prompt processing (preemption-based)
- Orca: runs prompts at full length alongside generation
- **SplitFuse**: dynamic composition of fixed-sized batches with BOTH generation and prompt tokens

**Benefits:**
- Consistent forward sizes
- No preemption patterns
- Distributed long-prompt processing for lower latency

### Kernel Optimizations

- Blocked KV-caching with non-contiguous blocks
- Continuous batching
- High-performance CUDA kernels
- Tensor parallelism support
- Partial RoPE support for Phi-2 model family
- Reduced scheduling overhead and increased token sampling efficiency

### Performance Results

- **Throughput**: Up to 2.3x higher effective throughput vs vLLM
- **Latency**: 2x lower average latency
- **Tail Latency**: Up to 3.7x lower P95 token-level tail latency
- **Scalability**: Linear across 16 replicas

---

## 7. MLC-LLM / Apache TVM

### Architecture Overview

MLC LLM is a machine learning compiler and high-performance deployment engine with the mission to enable native AI model deployment on every platform.

**Key Components:**
- **MLCEngine**: Unified high-performance LLM inference engine across platforms
- **Apache TVM**: Open ML compilation framework
- **TensorIR**: Tensor-level representation
- **Relax**: Graph-level representation with Python-first transformations

### How TVM Compiles Models

**TVM Unity Architecture:**
- Cross-level design using TensorIR (tensor-level) and Relax (graph-level)
- Python-first transformations for vertical compilers
- High-performance CPU/GPU code generation instantly without tuning
- Dynamic shape and symbolic shape tracking by design

**Compilation Pipeline:**
1. Model definition in Python (PyTorch-compatible)
2. Conversion to Relax IR (graph-level representation)
3. TensorIR lowering (tensor-level operations)
4. Backend-specific code generation (CUDA, Metal, Vulkan, OpenCL)
5. JIT compilation for target platform

### Relax IR for LLM Optimization

- Graph-level intermediate representation
- Supports dynamic shapes (variable sequence lengths)
- Python-first transformation passes
- Enables operator fusion, memory planning, and scheduling optimization

### Universal Deployment

MLC LLM compiles and deploys to:
- **GPU backends**: CUDA, Metal, Vulkan, OpenCL, WebGPU
- **Platforms**: REST server, Python, JavaScript, iOS, Android, browsers
- **APIs**: OpenAI-compatible endpoints

**Just-in-time compilation** generates GPU code for each platform, enabling cross-platform execution without manual porting.

---

## 8. NVIDIA Triton Inference Server

### Architecture Overview

Triton is a production inference serving platform with a request-processing pipeline:

```
Client Request -> HTTP/REST, gRPC, or C API -> Per-Model Scheduler -> Backend -> Response
```

**Core Components:**
- Model Repository (file-system based)
- Per-model schedulers with optional batching
- Backend execution engines
- Model management API

### Backend System

| Backend | Description |
|---------|-------------|
| TensorRT | Optimized GPU inference via compiled TRT engines |
| TensorRT-LLM | LLM-specific with in-flight batching, paged KV cache |
| vLLM | PagedAttention-based LLM serving (with Multi-LoRA) |
| ONNX Runtime | Cross-platform with multiple execution providers |
| PyTorch | Direct PyTorch model serving |
| Python | Custom pre/post-processing via Python scripts |
| Custom | Extend via Backend C API |

### Dynamic Batching

Enabled and configured per-model via `ModelDynamicBatching` in model configuration:

**Configuration Parameters:**
- `preferred_batch_size`: Target batch sizes for dynamic batching
- `max_queue_delay_microseconds`: Maximum wait time for batch formation
- `queue_policy`: Queue size, priorities, and timeouts
- `preserve_ordering`: Maintain request order in responses

**Behavior:**
- Combines one or more inference requests into a single batch
- Reduces latency and increases throughput via higher resource utilization
- Operates independently per model

### Scheduling Algorithms

**Stateless Models:**
- Default scheduler: single request per inference
- Dynamic batcher: combines multiple requests

**Stateful Models:**
- Sequence batcher: maintains state across related requests
- Supports correlation-based sequence routing

### Ensemble Models

- Chain multiple models in inference pipelines
- Business logic scripting for complex workflows
- Input/output tensor routing between pipeline stages

### Model Analyzer & Navigator

- **Model Analyzer**: Profiles model performance across configurations; recommends optimal batch sizes, instance counts, and concurrency
- **Model Navigator**: Automates model format conversion and optimization
- Health endpoints, utilization metrics, throughput/latency tracking for Kubernetes

---

## 9. Hugging Face TGI (Text Generation Inference)

### Three-Tier Architecture

**Note:** TGI entered maintenance mode as of December 2025, with vLLM being integrated as a TGI backend.

**Component 1 - Router (Rust):**
- High-performance Rust HTTP/gRPC server
- Handles all client-facing requests and validation
- Manages request queuing with continuous batching
- Forwards inference requests to model server via gRPC
- Bypasses Python GIL through Rust's type system and memory safety

**Component 2 - Model Server (Python):**
- Python gRPC server performing actual model inference
- Loads models, manages KV cache, executes forward passes
- Leverages PyTorch, custom CUDA/ROCm kernels, hardware-specific optimizations
- Supports tensor parallelism via NCCL

**Component 3 - Launcher:**
- Launches one or more model server shards
- Configures router with compatible arguments

### Continuous Batching Call Flow

1. Client sends `generate_stream` request
2. Router sends `prefill(batch)` to model server
3. Model server returns generations, cached batch, timings
4. Router sends `decode(cached_batch)` for subsequent tokens
5. New requests trigger `prefill(new_batch)`, pausing current batch
6. `decode(batch1, batch2)` processes multiple batches concurrently
7. `filter_batch` removes completed requests
8. `clear_cache` cleans up abandoned requests

### Custom CUDA Kernels

- Flash Attention integration for accelerated attention computation
- CUDA GRAPHS recorded for LLM forward passes on a set of batch sizes
- Paged Attention support (gRPC v3 protocol)

### Quantization Support

| Method | Description |
|--------|-------------|
| bitsandbytes (NF4, FP4) | 4-bit quantization via bitsandbytes library |
| GPTQ | Post-training quantization with `quantize` CLI |
| AWQ | Activation-aware weight quantization |
| EETQ | INT8 quantization |
| EXL2 | ExLlamaV2 quantization format |
| FP8 | 8-bit floating point |

### Router Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max-concurrent-requests` | 128 | Max concurrent requests |
| `max-input-tokens` | 1024 | Maximum input length |
| `max-total-tokens` | 2048 | Max input + output |
| `waiting-served-ratio` | 1.2 | Ratio controlling batch formation |
| `max-batch-prefill-tokens` | 4096 | Max tokens in prefill batch |
| `max-waiting-tokens` | 20 | Max tokens before forced batch |
| `max-batch-size` | unset | Hard limit on batch size |

### Model Server Variants

- **NVIDIA CUDA**: Default, optimized for NVIDIA GPUs
- **AMD ROCm**: Maintained in main repository
- **Intel GPU**: Maintained in main repository
- **Intel Gaudi**: Forked repository (`tgi-gaudi`)
- **AWS Inferentia2**: Neuron support in main repository
- **Google TPU**: Via Optimum TPU

### Multi-Backend Support

TGI supports multiple inference backends:
- Native TGI (Python/CUDA)
- TensorRT-LLM backend
- vLLM backend (integrated in Q1 2025)

---

## 10. Ollama

### Architecture Overview

Ollama is a user-friendly tool for running LLMs locally, built on top of llama.cpp with additional management features.

**Core Components:**
- **Go-based server**: HTTP API for model management and inference
- **llama.cpp integration**: Subprocess model for inference execution
- **Model registry**: Pull/push models from Ollama's model library
- **Modelfile system**: Configuration files defining model behavior

### llama.cpp Integration

Three integration mechanisms:
1. **CMake-based build system**: Compiles llama.cpp with multiple GPU backend support
2. **Runner subprocess**: Ollama spawns llama.cpp as a subprocess serving HTTP requests
3. **CGO bindings**: Direct C API bindings within the runner process

### Model Management & Modelfile

Modelfiles define model configuration:
- Model base layer (FROM instruction)
- Runtime parameters controlling memory and performance
- System prompts defining default behavior
- Template definitions for prompt formatting
- Adapter layers (LoRA)

### Concurrency Handling

- **Parallel request auto-selection**: 4 or 1 based on available memory
- **Maximum queue**: 512 requests
- **Memory management**: Models loaded as needed; insufficient memory triggers queuing
- **Model eviction**: Inactive models unloaded to make room for new ones
- **Sequence management**: Both runners manage concurrent inference through sequences with different implementations

### Supported Backends

- CUDA (NVIDIA GPUs)
- Metal (Apple Silicon)
- ROCm (AMD GPUs)
- CPU fallback (SIMD-optimized)

---

## 11. ONNX Runtime

### Architecture Overview

ONNX Runtime is a cross-platform inference engine that executes ONNX models with hardware-specific optimizations through its extensible Execution Provider (EP) framework.

### Execution Providers

| Provider | Target Hardware | Key Features |
|----------|----------------|--------------|
| CUDA | NVIDIA GPUs | Direct GPU kernel execution |
| TensorRT | NVIDIA GPUs | TRT engine optimization; FP16/INT8 |
| ROCm | AMD GPUs | HIP-based execution |
| MIGraphX | AMD GPUs | Graph-level optimization |
| DirectML | Windows GPUs | Hardware-agnostic on Windows |
| OpenVINO | Intel CPUs/GPUs | Intel optimization toolkit |
| oneDNN | Intel CPUs | GEMM and convolution optimization |
| CoreML | Apple | Neural Engine, GPU, CPU |
| NNAPI | Android | Neural network acceleration |
| QNN | Qualcomm | Hexagon DSP, GPU, HTP |

**Priority strategy**: TensorRT EP first, then CUDA EP for NVIDIA; ROCm/MIGraphX for AMD; oneDNN/OpenVINO for Intel.

### Graph Optimizations

**Three Optimization Levels:**

**1. Basic (all EPs):**
- Constant folding: statically computes parts relying only on constant initializers
- Redundant node elimination: removes identity, slice, unsqueeze, dropout
- Node fusions: Conv+Add, Conv+Mul, Conv+BatchNorm, Relu+Clip, Reshape

**2. Extended (CPU, CUDA, ROCm):**
- GELU fusion
- Layer Normalization fusion
- BERT Embedding Layer fusion
- Attention fusion with optional approximation
- Skip Layer Normalization fusion
- Bias GELU fusion

**3. Layout (CPU only):**
- Data layout transformations for optimal memory access patterns

**Online vs Offline Mode:**
- Online: optimizations applied during session initialization
- Offline: serialized optimized model to disk for faster subsequent loading
- Constraint: offline models must run on compatible hardware

### Olive Optimization Toolkit

Olive is a hardware-aware model optimization tool:
- 40+ built-in optimization components
- Quantization: INT4 block-wise weight-only, INT8 QLinearMatMul
- Compression, graph optimization, fine-tuning
- Techniques: AWQ, GPTQ, RTN quantization
- Supports NPUs (v0.9.0+)
- Automated model format conversion
- CLI-based workflow

### Quantization Support

**INT4:**
- Block-wise weight-only quantization
- Applied to specific operators
- MatMulNBits for 4-bit inference

**INT8:**
- QLinearMatMul, QuantizeLinear, DequantizeLinear operators
- Static and dynamic quantization modes
- TensorRT EP: native INT8 calibration tables

**Limitations:**
- Quantized models cannot use CUDAExecutionProvider for dynamic quantization
- MatMulInteger and DynamicQuantizeLinear nodes unsupported by CUDA EP

---

## 12. FlashAttention (v1/v2/v3)

### Core Algorithm

FlashAttention reorders the attention computation and leverages **tiling** and **recomputation** to reduce memory usage from O(N^2) to O(N) in sequence length.

**Key Insight:** Fuse online-softmax with back-to-back GEMM kernels, avoiding materializing the full N x N attention matrix in HBM.

### Tiling Strategy

- Tile size (Bc) is critical: larger tiles mean more reuse per global memory load but require more shared memory
- Maximum head dimension of 128 constrains tile size before exceeding SM's shared memory budget
- Blocks of Q, K, V loaded into SRAM, attention computed tile-by-tile
- Online softmax maintains running statistics (max and sum) across tiles

### FlashAttention-1

- First IO-aware exact attention algorithm
- 2-4x speedup over standard attention
- Linear memory usage vs quadratic
- Fuses softmax normalization across tiles using online algorithm

### FlashAttention-2

**Key Improvements:**
- Parallelizes over sequence length dimension (not just batch/head)
- Inner loop over blocks of K and V (reversed from v1)
- Improved GPU occupancy and work distribution

**CUDA Implementation (Hopper):**
- Fused online-softmax with back-to-back GEMM kernels
- Tensor Memory Accelerator (TMA) for efficient data movement
- Warpgroup Matrix-Multiply-Accumulate (WGMMA) instructions
- CUTLASS Layouts and Tensors for memory management
- Overlapped copy and GEMM operations
- Optimal tile sizes for Q, K, V balancing register pressure and shared memory
- Asynchronous copy (`cp.async`) with carefully placed barriers

**Performance:** ~350 TFLOPS on H100 (FP16 forward pass)

### FlashAttention-3

**Key Innovations:**
- **Producer-consumer asynchrony**: Warp-specialized software pipelining
- Separate producer (data movement) and consumer (tensor core) warps
- Exploits asynchronous execution of data movement and tensor cores

**Hopper-Specific Features:**
- Warpgroup Matrix-Multiply-Accumulate (WGMMA)
- Tensor Memory Accelerator (TMA)
- FP8 support with improved numerical accuracy

**Performance:**
- 540-570 TFLOPS (FP16 forward, H100) -- 1.5-2.0x vs FlashAttention-2
- Reduces FP8 numerical error by 2.6x vs standard per-tensor quantization

---

## 13. FlashInfer

### Architecture Overview

FlashInfer is an efficient and customizable attention engine for LLM inference serving (Best Paper, MLSys 2025). Integrated into SGLang, vLLM, and MLC-Engine.

### Unified Sparse Representation

Every KV cache layout represented as a **block/vector sparse matrix**:
- Handles diverse block sizes, masking rules, and positional encoding schemes
- Unified storage absorbs request dynamism

### JIT Compilation & Template Specialization

- Customizable attention template adapting to various settings through JIT compilation
- CUDA/CUTLASS codebase with specialization knobs
- Eliminates need for massive pre-compiled kernel libraries
- Template parameters: head dimension, data types, block sizes, masking, position encoding

### Plan/Run API (Inspector-Executor Interface)

Two-phase execution:
1. **Plan Stage**: Gathers metadata required for kernel launch; inspects request shapes and prefix sharing patterns
2. **Run Stage**: Executes tuned kernels through lightweight scheduler
- Plans can be reused across all layers in a generation step (same metadata)
- PyTorch-friendly interface

### Core Operator Families

| Family | Description |
|--------|-------------|
| **Attention** | Variable sequence lengths, diverse cache layouts |
| **GEMM** | Matrix multiplication with FP4/FP8 tensor core support |
| **Communication** | All-reduce and all-to-all operations |
| **Sampling** | Rejection-based token selection (no global sorting) |

### Prefill and Decode Kernels

- Separate optimized kernels for prefill (compute-bound) and decode (memory-bound) phases
- Prefill: leverages FlashAttention-2/3 algorithms
- Decode: optimized for single-token per-sequence processing with paged KV cache

### Backend Selection

- Multiple backend implementations: CUTLASS, cuDNN, FlashAttention-2/3, TensorRT-LLM
- User-selectable via: `attention = BatchAttention(backend="cutlass")`

### Logits Processing Pipeline

Modular composition of post-processing:
- Temperature -> Softmax -> TopP -> Sample
- Emits efficient fused rejection sampling-based implementation

### Performance

- Median inter-token latency drops of 29-69% when integrated into SGLang
- Up to 21% TTFT improvement on Llama-3 models

---

## 14. Key Serving Concepts

### Continuous Batching vs Static Batching

**Static Batching:**
- Fills batch with waiting items up to batch size
- Processes until ALL batched items complete
- Short-completion requests must wait for longest request
- GPU utilization drops as requests finish at different times
- Simple but highly wasteful

**Continuous Batching:**
- Completed requests immediately removed from batch
- Freed slots filled with next queued request
- Dramatically higher GPU utilization
- Industry standard for all production LLM serving

### Prefill vs Decode Phase Optimization

**Prefill Phase:**
- Processes all input tokens together in parallel
- **Compute-bound**: GEMM kernels achieve high GPU utilization
- Single forward pass for entire prompt
- Dominates TTFT (time-to-first-token)

**Decode Phase:**
- Generates one token per forward pass (autoregressive)
- **Memory-bandwidth-bound**: Low batch sizes under-utilize compute
- KV cache read dominates; each step reads all previous KV pairs
- Dominates ITL (inter-token latency) and throughput

**Optimization Strategies:**
- Prefill: maximize parallelism, use Flash Attention, chunked prefill for long contexts
- Decode: maximize batch size, use CUDA graphs, KV cache compression, paged attention

### Memory-Bound vs Compute-Bound Analysis

**Arithmetic Intensity** = FLOPs / Bytes Transferred

| Phase | Arithmetic Intensity | Bottleneck | Strategy |
|-------|---------------------|------------|----------|
| Prefill (large batch) | High | Compute | Tensor core optimization, GEMM tuning |
| Prefill (small batch) | Medium | Mixed | Kernel fusion, Flash Attention |
| Decode (small batch) | Very Low | Memory BW | Batching, KV cache compression, quantization |
| Decode (large batch) | Medium | Mixed | CUDA graphs, operator fusion |

**Roofline Model Application:**
- Below the roofline ridge point: memory-bound -> reduce data movement
- Above ridge point: compute-bound -> maximize FLOPS utilization

### Token Throughput vs Latency Tradeoffs

**Throughput Optimization:**
- Increase batch size until compute-bound
- Continuous batching to maintain high occupancy
- Prefix caching to avoid redundant computation
- Quantization to reduce memory footprint and enable larger batches

**Latency Optimization:**
- Speculative decoding for lower ITL
- CUDA graphs to eliminate kernel launch overhead
- Smaller batch sizes for predictable latency
- Disaggregated serving to isolate prefill latency from decode

**Batch Size Regions:**
- Below saturation `B_sat`: step latency dominated by HBM bandwidth; increasing batch improves throughput linearly
- Beyond `B_sat`: compute-bound with increasing ITL; throughput plateaus

### Scheduling Algorithms

**First-Come-First-Served (FCFS):**
- Process requests in arrival order
- Simple, fair, but suboptimal for mixed workloads

**Priority-Based:**
- Weight requests by importance or deadline
- Better for SLO-constrained deployments

**Token-Budget Scheduling (vLLM V1):**
- Fixed token budget per step
- Mix prefill and decode tokens within budget
- Dynamic allocation based on request state

**Dynamic SplitFuse (DeepSpeed-FastGen):**
- Decompose long prompts, compose short prompts to fill exact budget
- Consistent forward sizes for predictable performance

**Overlap Scheduling (SGLang):**
- Prepare next batch on CPU while GPU executes current batch
- Near-zero CPU overhead

### Disaggregated Serving (Prefill/Decode Separation)

**Motivation:**
- Prefill is compute-bound; decode is memory-bound
- Mixed batches cause interference: prefill requests increase forward pass latency, degrading decode throughput
- Different hardware configurations optimal for each phase

**Key Systems:**

**DistServe:**
- Introduced the concept of splitting prefill/decode across separate compute pools
- Independent scaling of each phase
- Became default architecture across major serving stacks in 2025

**Mooncake (Best Paper, FAST 2025):**
- KVCache-centric disaggregated architecture (powers Moonshot AI's Kimi)
- Leverages underutilized CPU, DRAM, and SSD for disaggregated KV cache storage
- Separates prefill and decoding clusters
- Powers Kimi K2 on 128 H200 GPUs: 224k tokens/sec prefill, 288k tokens/sec decode

**Splitwise:**
- Alternative approach to prefill/decode separation
- Optimizes resource allocation per phase

**SGLang EPD Disaggregation:**
- Encode-Prefill-Decode separation with Mooncake as transfer backend
- Decouples multimodal encoders from language model nodes

**Production Adoption (2025):**
- NVIDIA Dynamo, llm-d, Ray Serve LLM, SGLang, vLLM, LMCache, MoonCake all support disaggregation
- Became the default playbook across nearly every major LLM serving stack

### Speculative Decoding Theory and Variants

**Core Theory:**
- Pair target model with lightweight draft mechanism
- Draft proposes k candidate tokens cheaply
- Target model verifies all k proposals in single forward pass (parallel verification)
- Accept longest valid prefix matching target model's distribution
- Guarantees lossless output quality (statistically equivalent to autoregressive sampling)

**Verification Algorithm:**
- For each draft token (left-to-right):
  - If `p_target(token) >= p_draft(token)`: accept deterministically
  - Else: accept with probability `p_target(token) / p_draft(token)`
  - On rejection: sample from adjusted distribution `max(0, p_target - p_draft)` normalized
- Expected speedup: `1 / (1 - acceptance_rate)` times the draft model's speed advantage

**Variant Taxonomy:**

| Variant | Draft Source | Training Required | Key Innovation |
|---------|-------------|-------------------|----------------|
| Standard (Leviathan et al.) | Separate small LM | No (uses existing model) | Original formulation |
| EAGLE | Target model surgery (MLP) | Fine-tune MLP head | Feature-level prediction |
| EAGLE-2 | Target model surgery | Fine-tune MLP head | Dynamic draft tree via confidence scores |
| EAGLE-3 | Prediction head on internal layers | Fine-tune head | No separate draft model needed |
| Medusa | Parallel linear heads | Train heads | k parallel predictions |
| Lookahead | N-gram from past generations | No training | No additional model |
| N-gram (vLLM) | Pattern matching in prompt | No training | Simplest implementation |
| MTP (SGLang) | Lightweight draft model | Fine-tune | Multiple token prediction |
| Variational (VSD) | Optimized draft | Train | 9.6% better than EAGLE-3 |

**Tree Attention for Verification:**
- Draft models can propose tree-structured candidates (not just linear sequences)
- Tree attention verifies all branches in parallel
- Dynamic tree construction based on confidence scores (EAGLE-2)
- Prunes invalid branches efficiently

**Challenges with Continuous Batching:**
- Speculative decoding at high concurrency creates variable-length outputs per step
- Batch management complexity increases with tree verification
- Fixed speculation length helps (SGLang approach)

**Performance Results (2025):**
- EAGLE-3 on Llama 4 Maverick: 2.18x speedup (MT-Bench)
- MTP on DeepSeek V3: up to 60% higher output throughput
- VSD: outperforms EAGLE-3 by 9.6% on MT-Bench/HumanEval/GSM8K

---

## Cross-System Comparison Matrix

| Feature | vLLM | SGLang | TensorRT-LLM | llama.cpp | ExLlamaV2/V3 | TGI |
|---------|------|--------|--------------|-----------|--------------|-----|
| Language | Python/C++ | Python/C++ | Python/C++ | C/C++ | Python/CUDA | Rust/Python |
| Attention | PagedAttention | RadixAttention | FMHA/XQA | FlashAttention | FlashAttention | FlashAttention |
| KV Cache | Paged blocks | Radix tree | Paged/contiguous | Pre-allocated | Paged with hash | Paged |
| Batching | Continuous | Continuous | In-flight | Static batch | Dynamic | Continuous |
| Quantization | FP8/AWQ/GPTQ/INT8 | FP8/AWQ/GPTQ | FP8/FP4/INT8/INT4 | GGUF (1.5-8 bit) | EXL2/EXL3 (1.6-8 bit) | FP8/AWQ/GPTQ |
| Spec. Decode | EAGLE/Medusa/N-gram | MTP/EAGLE | EAGLE/MTP/N-gram | Draft model | Dynamic generator | Draft model |
| Multi-GPU | TP/PP/DP | TP/PP/EP/DP | TP/PP/EP | PP (RPC) | TP/EP | TP |
| Prefix Cache | Hash-based | RadixAttention | Paged blocks | Manual | Hash-based pages | Limited |
| Structured Output | xgrammar/LLGuidance | XGrammar/LLGuidance | Guided decoding | GBNF grammar | Token filtering | Grammar |
| Primary Target | Data center | Data center | Data center | Edge/consumer | Consumer GPU | Data center |

---

## Sources

- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [vLLM Paged Attention Design](https://docs.vllm.ai/en/stable/design/paged_attention/)
- [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html)
- [vLLM Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode/)
- [vLLM Quantization](https://docs.vllm.ai/en/latest/features/quantization/)
- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [Mini-SGLang: Efficient Inference Engine in a Nutshell](https://lmsys.org/blog/2025-12-17-minisgl/)
- [Inside SGLang: Anatomy](https://blog.sugiv.fyi/inside-sglang-anatomy-high-performance-structured-llm-inference-system)
- [SGLang Speculative Decoding Tutorial](https://company.hpc-ai.com/blog/sglang-speculative-decoding-tutorial)
- [Accelerating SGLang with MTP](https://lmsys.org/blog/2025-07-17-mtp/)
- [TensorRT-LLM Overview](https://nvidia.github.io/TensorRT-LLM/overview.html)
- [TensorRT-LLM GPT Attention](https://nvidia.github.io/TensorRT-LLM/advanced/gpt-attention.html)
- [TensorRT-LLM XQA Kernel](https://nvidia.github.io/TensorRT-LLM/blogs/XQA-kernel.html)
- [TensorRT-LLM Quantization](https://nvidia.github.io/TensorRT-LLM/blogs/quantization-in-TRT-LLM.html)
- [TensorRT-LLM Optimization Guide](https://introl.com/blog/tensorrt-llm-optimization-nvidia-inference-stack-guide)
- [llama.cpp GitHub](https://github.com/ggml-org/llama.cpp)
- [llama.cpp Quantize README](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)
- [llama.cpp DeepWiki](https://deepwiki.com/ggml-org/llama.cpp)
- [GGUF Quantization Guide 2026](https://www.decodesfuture.com/articles/llama-cpp-gguf-quantization-guide-2026)
- [ExLlamaV2 GitHub](https://github.com/turboderp-org/exllamav2)
- [ExLlamaV3 GitHub](https://github.com/turboderp-org/exllamav3)
- [ExLlamaV3 DeepWiki](https://deepwiki.com/turboderp-org/exllamav3)
- [DeepSpeed-FastGen Paper](https://arxiv.org/html/2401.08671v1)
- [DeepSpeed-FastGen Blog](https://github.com/microsoft/DeepSpeed/blob/master/blogs/deepspeed-fastgen/README.md)
- [MLC-LLM GitHub](https://github.com/mlc-ai/mlc-llm)
- [MLC-LLM Introduction](https://llm.mlc.ai/docs/get_started/introduction)
- [Apache TVM LLM Optimization](https://tvm.apache.org/docs/how_to/tutorials/optimize_llm.html)
- [Triton Inference Server Architecture](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/architecture.html)
- [Triton Dynamic Batching](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html)
- [TGI Architecture](https://huggingface.co/docs/text-generation-inference/en/architecture)
- [TGI GitHub](https://github.com/huggingface/text-generation-inference)
- [Ollama GitHub](https://github.com/ollama/ollama)
- [Ollama llama.cpp Integration DeepWiki](https://deepwiki.com/ollama/ollama/5.2-llama.cpp-integration)
- [ONNX Runtime](https://onnxruntime.ai/)
- [ONNX Runtime Graph Optimizations](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html)
- [Olive Toolkit](https://github.com/microsoft/Olive)
- [FlashAttention GitHub](https://github.com/Dao-AILab/flash-attention)
- [FlashAttention-3 Paper](https://tridao.me/publications/flash3/flash3.pdf)
- [FlashAttention-2 on Hopper Case Study](https://arxiv.org/abs/2312.11918)
- [FlashInfer GitHub](https://github.com/flashinfer-ai/flashinfer)
- [FlashInfer NVIDIA Blog](https://developer.nvidia.com/blog/run-high-performance-llm-inference-kernels-from-nvidia-using-flashinfer)
- [Disaggregated Inference Retrospective](https://haoailab.com/blogs/distserve-retro/)
- [Mooncake Paper](https://arxiv.org/abs/2407.00079)
- [EAGLE-2 Paper](https://arxiv.org/html/2406.16858v1)
- [Speculative Decoding Survey](https://github.com/hemingkx/SpeculativeDecodingPapers)
- [Continuous Batching from First Principles](https://huggingface.co/blog/continuous_batching)
- [Prefill and Decode for Concurrent Requests](https://huggingface.co/blog/tngtech/llm-performance-prefill-decode-concurrent-requests)
