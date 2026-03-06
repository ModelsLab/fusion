---
id: llm_serving_systems_deep_dive
kind: document
title: LLM Serving Systems - Architecture and Kernel-Level Deep Dive
category: serving
summary: Comprehensive technical guide to all major LLM serving systems - vLLM, SGLang, TensorRT-LLM, llama.cpp - covering their architectures, scheduling algorithms, custom kernels, and optimization strategies.
tags:
  - vllm
  - sglang
  - tensorrt-llm
  - llama-cpp
  - serving
  - continuous-batching
  - speculative-decoding
  - paged-attention
source_ids:
  - vllm-docs
  - sglang-docs
  - tensorrt-llm-docs
  - vllm-paged-attention
gpu_families:
  - Ampere
  - Ada
  - Hopper
  - Blackwell
workloads:
  - serving
  - decode
  - prefill
---

# LLM Serving Systems - Deep Dive

## Serving System Architecture Overview

### The Two Phases of LLM Inference

**Prefill (Prompt Processing)**:
- Process entire input prompt at once
- Compute-bound: large matrix multiplies
- Produces KV cache for all input tokens
- Latency: proportional to prompt length
- Metric: Time to First Token (TTFT)

**Decode (Token Generation)**:
- Generate one token at a time (autoregressive)
- Memory-bound: read entire KV cache + model weights for single token
- Each step appends to KV cache
- Latency: per-token latency
- Metrics: Time Per Output Token (TPOT), Inter-Token Latency (ITL)

### Key Metrics

| Metric | Definition | Typical Target |
|--------|-----------|---------------|
| TTFT | Time from request to first generated token | <500ms for chat |
| TPOT | Average time per output token | <50ms for chat |
| ITL | Time between consecutive output tokens | <100ms for streaming |
| Throughput | Total tokens/second across all requests | Maximize |
| P99 Latency | 99th percentile latency | <2x median |

## vLLM

### Architecture

```
Request Queue → Scheduler → Model Executor → Output Processor
                    ↕              ↕
              Block Manager   GPU Workers
                    ↕
              KV Cache Pool
```

**Scheduler**: Decides which requests to process each iteration
- Maintains waiting, running, and swapped queues
- Implements preemption (swap to CPU when GPU memory full)
- Supports priority-based scheduling

**Block Manager**: Manages paged KV cache
- Allocates/frees physical blocks
- Maintains block tables (logical → physical mapping)
- Implements copy-on-write for beam search / parallel sampling

**Model Executor**: Runs the model
- Manages GPU workers (one per GPU for tensor parallelism)
- Handles model loading and weight distribution
- Coordinates multi-GPU execution

### PagedAttention in vLLM

```python
# Block allocation:
BLOCK_SIZE = 16  # tokens per block
num_blocks = total_gpu_memory_for_kv / (2 * num_layers * num_heads * head_dim * block_size * dtype_size)

# Block table for a sequence with 50 tokens:
# Needs ceil(50/16) = 4 blocks
block_table = [physical_block_3, physical_block_7, physical_block_12, physical_block_1]
# Blocks don't need to be contiguous!

# Memory efficiency:
# - No internal fragmentation (blocks are small)
# - Only last block can have waste (<16 tokens)
# - Waste < block_size per sequence on average
```

### vLLM Custom CUDA Kernels
- `paged_attention_v1`: Single-block attention kernel for decode
- `paged_attention_v2`: Multi-block attention with KV-parallel decomposition
- `reshape_and_cache`: Insert new KV entries into paged cache
- `copy_blocks`: For beam search fork/copy operations
- `rotary_embedding`: Fused RoPE kernel
- `silu_and_mul`: Fused SwiGLU activation
- `rms_norm`: Fused RMSNorm
- `awq_dequant`: AWQ dequantization kernels
- `gptq_marlin`: GPTQ with Marlin backend
- `machete`: Latest mixed-precision GEMM kernel
- `moe_align_block_size`: MoE token routing
- `topk_softmax`: Fused top-k with softmax for MoE gating

### vLLM Configuration Tuning

```python
# Key parameters:
engine_args = EngineArgs(
    model="meta-llama/Llama-3-70B",
    tensor_parallel_size=4,           # Number of GPUs for TP
    gpu_memory_utilization=0.90,      # Fraction of GPU memory for KV cache
    max_num_seqs=256,                 # Max concurrent sequences
    max_model_len=32768,              # Max sequence length
    enforce_eager=False,              # False = use CUDA graphs
    enable_chunked_prefill=True,      # Chunk long prefills
    max_num_batched_tokens=4096,      # Max tokens per iteration
    quantization="awq",              # Quantization method
    kv_cache_dtype="fp8",            # KV cache precision
    enable_prefix_caching=True,       # Cache shared prefixes
    speculative_model="draft-model",  # Speculative decoding
    num_speculative_tokens=5,         # Tokens to speculate
)
```

### vLLM Speculative Decoding
```
# Draft model generates k tokens quickly
# Target model verifies all k+1 tokens in one forward pass (parallel)
# Accept matching tokens, reject at first mismatch

# Token acceptance algorithm (modified rejection sampling):
for i in range(k):
    p = target_model_prob[i][draft_token[i]]
    q = draft_model_prob[i][draft_token[i]]
    if random() < min(1, p/q):
        accept(draft_token[i])
    else:
        resample from adjusted distribution
        break
```

## SGLang

### Architecture

```
HTTP Server → Router → TokenizerManager → Scheduler → TpModelWorker
                                              ↕
                                      RadixCache
```

### RadixAttention (Key Innovation)
```
# Radix tree stores KV cache indexed by token sequences
# Shared prefixes are stored once, reused across requests

# Example:
# Request 1: "You are a helpful assistant. What is 2+2?"
# Request 2: "You are a helpful assistant. Write a poem."
# Shared prefix: "You are a helpful assistant. "
# KV cache for shared prefix computed once, shared by both requests

# Radix tree structure:
root
 └── "You are a helpful" → KV cache node
      ├── "assistant. What is 2+2?" → KV cache node
      └── "assistant. Write a poem." → KV cache node

# Automatic prefix matching with LRU eviction of cold prefixes
```

### SGLang Constrained Decoding
```python
# Grammar-guided generation using FSM (Finite State Machine)
# Jump-forward decoding: when grammar constrains next tokens to a
# known sequence, skip token-by-token generation

# Example: JSON mode
# When generating '": ' after a key, the FSM knows the exact tokens
# Skip calling the model for these deterministic tokens
```

### SGLang Performance Features
- **FlashInfer Integration**: Uses FlashInfer for attention kernels
- **Efficient KV Cache**: RadixAttention + FlashInfer paged cache
- **CUDA Graph**: Captures decode step as CUDA graph
- **Tensor Parallelism**: Multi-GPU support
- **Chunked Prefill**: Split long prefills across iterations
- **Torch.compile**: Integrates with PyTorch compilation

## TensorRT-LLM

### Architecture

```
Model Definition (Python) → Build Engine (TensorRT) → Runtime (C++ Executor)
                                    ↕
                              Plugin System
```

### Build Phase
```python
import tensorrt_llm

# Define model (LLaMA example):
model = tensorrt_llm.models.LLaMAForCausalLM.from_hugging_face(
    "meta-llama/Llama-3-70B",
    dtype="float16",
    quantization=QuantConfig(quant_algo=QuantAlgo.FP8),
    tp_size=4,
    pp_size=1,
)

# Build TensorRT engine:
engine = tensorrt_llm.build(model, build_config=BuildConfig(
    max_batch_size=128,
    max_input_len=4096,
    max_seq_len=8192,
    max_num_tokens=4096,
    strongly_typed=True,
    # Enable plugins
    plugin_config=PluginConfig(
        gemm_plugin="auto",
        gpt_attention_plugin="auto",
        moe_plugin="auto",
    ),
))
```

### TensorRT-LLM Key Optimizations
1. **Custom Attention (XQA)**: Optimized GQA/MQA decode attention
2. **In-Flight Batching**: Continuous batching at the C++ level
3. **Paged KV Cache**: Similar to vLLM but with TensorRT kernels
4. **FP8 GEMM**: Native FP8 on Hopper via cuBLASLt and custom plugins
5. **Quantization Plugins**: AWQ, GPTQ, SmoothQuant, FP8 - all as TRT plugins
6. **GEMM Plugin**: Custom GEMM selection with workspace management
7. **Multi-GPU**: TP + PP with NCCL

### TensorRT-LLM Performance Tuning
```
# Key build-time flags:
--max_batch_size 128       # Larger = more throughput
--max_input_len 4096       # Max prompt length
--max_num_tokens 8192      # Max tokens per batch
--paged_kv_cache           # Enable paged KV
--remove_input_padding     # Pack variable-length inputs
--use_fused_mlp            # Fuse MLP layers
--multi_block_mode         # Split attention across multiple blocks
--use_fp8_context_fmha     # FP8 attention for prefill
--tokens_per_block 128     # KV cache block size
```

## llama.cpp / GGML

### Architecture
```
Model Loading (GGUF) → Computation Graph (GGML) → Backend Execution
                                                     ├── CPU (AVX2/AVX-512/ARM NEON)
                                                     ├── CUDA
                                                     ├── Metal
                                                     ├── Vulkan
                                                     └── SYCL
```

### GGML Tensor Library
```c
// GGML builds a computation graph, then executes it
struct ggml_context * ctx = ggml_init(params);

// Define tensors and operations
struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_embd, n_tokens);
struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, n_embd, n_ff);
struct ggml_tensor * c = ggml_mul_mat(ctx, a, b);  // matmul with on-the-fly dequant

// Build and execute graph
struct ggml_cgraph * gf = ggml_new_graph(ctx);
ggml_build_forward_expand(gf, c);
ggml_graph_compute_with_ctx(ctx, gf, n_threads);
```

### GGML CUDA Backend Kernels
- `mul_mat_q4_K`: INT4 K-quant matmul
- `mul_mat_q8_0`: INT8 matmul
- `flash_attn_ext`: Flash attention implementation
- `rope`: RoPE kernel
- `rms_norm`: RMSNorm kernel
- `soft_max`: Softmax kernel
- `dequantize_*`: Dequantization kernels for each quant type

### llama.cpp Key Features
- **Memory mapping (mmap)**: Map model file directly, OS handles paging
- **Batch inference**: Process multiple tokens in parallel
- **KV cache**: Ring buffer with configurable size
- **Grammar sampling**: BNF grammar for constrained output
- **Speculative decoding**: Draft model support
- **Server mode**: OpenAI-compatible HTTP server
- **GPU offloading**: Offload N layers to GPU, rest on CPU

## Continuous Batching

### How It Works
```
# Static batching (bad):
# Wait for all sequences in batch to finish → wasted compute on padding

# Continuous batching (good):
# Each iteration, independently decide which sequences to process
# Finished sequences are immediately replaced with new requests

Iteration 1: [Seq_A(prefill), Seq_B(decode), Seq_C(decode)]
Iteration 2: [Seq_A(decode), Seq_B(decode), Seq_C(done→Seq_D(prefill))]
Iteration 3: [Seq_A(decode), Seq_B(done→Seq_E(prefill)), Seq_D(decode)]
```

### Scheduling Algorithms

**FCFS (First Come First Served)**:
- Simple, fair
- May lead to head-of-line blocking (long request blocks short ones)

**Shortest Job First**:
- Minimize average latency
- Hard to predict job length for LLM

**Preemptive scheduling**:
- Swap out long-running requests to make room
- vLLM: swap KV cache to CPU memory
- Resume later when GPU memory is available

### Chunked Prefill
```
# Problem: long prefill blocks all decode requests
# Solution: split prefill into chunks, interleave with decode

# Without chunked prefill:
# Iter 1: [Seq_A prefill 4096 tokens] → 200ms, all decode requests stalled
# Iter 2: [Seq_A decode, Seq_B decode, Seq_C decode]

# With chunked prefill (chunk=512):
# Iter 1: [Seq_A prefill chunk 1 (512 tok), Seq_B decode, Seq_C decode]
# Iter 2: [Seq_A prefill chunk 2 (512 tok), Seq_B decode, Seq_C decode]
# ...
# Iter 8: [Seq_A prefill chunk 8 (512 tok), Seq_B decode, Seq_C decode]
# Iter 9: [Seq_A decode, Seq_B decode, Seq_C decode]

# Decode latency stays bounded while prefill progresses
```

## Disaggregated Serving

### Concept
Separate prefill and decode onto different GPU pools:

```
Prefill Pool (compute-optimized):     Decode Pool (memory-optimized):
├── GPU 0 (H100)                      ├── GPU 0 (H100 / L40S)
├── GPU 1 (H100)                      ├── GPU 1
└── GPU 2 (H100)                      └── GPU 2

Request flow:
1. Request arrives → Prefill Pool
2. Prefill runs, generates KV cache
3. KV cache transferred to Decode Pool (via NVLink/network)
4. Decode runs token-by-token
5. Tokens streamed back to client
```

### Why Disaggregate?
- Prefill is compute-bound → benefits from high FLOPS GPUs
- Decode is memory-bound → benefits from high bandwidth, doesn't need all FLOPS
- Mixed batching (prefill + decode) leads to interference
- Separate pools can be independently scaled

### Key Systems
- **Splitwise** (Microsoft): first disaggregated serving paper
- **DistServe** (Peking University): KV cache migration protocol
- **Mooncake** (Moonshot AI): KVCache-centric disaggregated architecture with CPU-memory KV cache pool

## Performance Comparison Guide

### Throughput Benchmark (LLaMA-70B, H100x4, FP8)

| System | Tokens/sec (decode) | TTFT (ms) | Config |
|--------|-------------------|-----------|--------|
| vLLM | ~4000-6000 | ~200 | TP=4, PagedAttn, FP8 |
| SGLang | ~4000-6000 | ~180 | TP=4, FlashInfer, FP8 |
| TensorRT-LLM | ~5000-7000 | ~150 | TP=4, in-flight batching |
| llama.cpp | ~500-1000 | ~500 | CUDA, Q4_K_M |

Note: Numbers are approximate and depend heavily on workload, batch size, and sequence lengths.

### When to Use Which

| Scenario | Recommendation | Why |
|----------|---------------|-----|
| Production serving, flexibility | vLLM or SGLang | Easy setup, good defaults |
| Maximum throughput, NVIDIA GPUs | TensorRT-LLM | Best optimized for NVIDIA |
| Prefix-heavy workload | SGLang | RadixAttention for prefix caching |
| Consumer GPU / edge | llama.cpp | Best quantization, CPU+GPU hybrid |
| Multi-LoRA serving | vLLM | Native multi-LoRA support |
| AMD GPUs | vLLM or SGLang | ROCm support |
| Apple Silicon | llama.cpp (MLX) | Metal backend, unified memory |
