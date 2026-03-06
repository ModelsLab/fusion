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

## Practical Deployment Recipes

### Recipe 1: vLLM Quick Start (5 Minutes)

**Install:**
```bash
pip install vllm
```

**Minimal Python script to serve a model (5 lines):**
```python
from vllm import LLM, SamplingParams
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
params = SamplingParams(temperature=0.7, max_tokens=256)
outputs = llm.generate(["Explain quantum computing in one paragraph."], params)
print(outputs[0].outputs[0].text)
```

**Start an OpenAI-compatible API server:**
```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto
```

**Test the endpoint:**
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 128
  }'
```

**Common startup errors and fixes:**

| Error | Cause | Fix |
|-------|-------|-----|
| `torch.cuda.OutOfMemoryError` | Model too large for GPU | Lower `--gpu-memory-utilization 0.85` or use `--quantization awq` |
| `ValueError: ... not enough memory for KV cache` | KV cache allocation failed | Reduce `--max-model-len` (e.g., `--max-model-len 4096`) |
| `RuntimeError: Cannot find the config.json` | Model name wrong or no HF token | Set `HF_TOKEN` env var or use `--tokenizer` explicitly |
| `ImportError: No module named 'vllm'` | Wrong Python env or CUDA mismatch | Reinstall with `pip install vllm --force-reinstall` |
| Server starts but hangs on first request | CUDA graph compilation | Wait 30-60s on first request; add `--enforce-eager` to skip CUDA graphs for debugging |

---

### Recipe 2: SGLang Quick Start (5 Minutes)

**Install:**
```bash
pip install "sglang[all]"
```

**Start the server:**
```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --port 30000
```

**Test the endpoint (OpenAI-compatible):**
```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 128
  }'
```

**Enable RadixAttention prefix caching:**
```bash
# Prefix caching is enabled by default in SGLang.
# To explicitly disable it (for benchmarking comparisons):
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disable-radix-cache

# To verify prefix caching is active, check server logs for:
#   "RadixCache is enabled"
# Or hit the /get_server_info endpoint:
curl http://localhost:30000/get_server_info | python -m json.tool
```

---

### Recipe 3: vLLM Production Deployment

**Docker run command with all important flags:**
```bash
docker run --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  --ipc=host \
  --env "HF_TOKEN=hf_your_token_here" \
  vllm/vllm-openai:latest \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 16384 \
  --max-num-seqs 128 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --kv-cache-dtype fp8 \
  --dtype auto \
  --disable-log-requests
```

**Flag explanations:**

| Flag | What It Does | Tuning Guidance |
|------|-------------|-----------------|
| `--gpu-memory-utilization 0.92` | Fraction of GPU memory reserved for KV cache (after model weights) | 0.90-0.95 for production; lower if OOM during bursts |
| `--max-model-len 16384` | Maximum sequence length (prompt + generation) | Set to your actual max; lower = more concurrent sequences |
| `--max-num-seqs 128` | Maximum concurrent requests in a batch | Higher = more throughput but more memory; start at 64, increase |
| `--tensor-parallel-size 4` | Split model across N GPUs | Must evenly divide num_attention_heads; use NVLink GPUs |
| `--enable-chunked-prefill` | Prevents long prompts from blocking decode | Always enable in production |
| `--enable-prefix-caching` | Cache KV for shared prefixes (system prompts) | Enable if requests share common prefixes |
| `--kv-cache-dtype fp8` | Store KV cache in FP8 instead of FP16 | 2x more concurrent sequences; negligible quality loss |
| `--disable-log-requests` | Suppress per-request logging | Enable for production to reduce I/O |

**Tensor parallelism setup (multi-GPU):**
```bash
# 2x GPU (models up to ~30B params in FP16)
--tensor-parallel-size 2

# 4x GPU (models up to ~70B params in FP16)
--tensor-parallel-size 4

# 8x GPU (models up to ~140B params in FP16)
--tensor-parallel-size 8

# Verify GPUs are visible:
nvidia-smi -L

# Check NVLink topology (important for TP performance):
nvidia-smi topo -m
```

**Monitoring throughput and latency:**
```bash
# vLLM exposes Prometheus metrics at /metrics
curl http://localhost:8000/metrics

# Key metrics to watch:
# vllm:num_requests_running        - current batch size
# vllm:num_requests_waiting        - queue depth (should be low)
# vllm:gpu_cache_usage_perc        - KV cache utilization (alert if >95%)
# vllm:avg_generation_throughput_toks_per_s - tokens/sec
# vllm:e2e_request_latency_seconds - end-to-end latency histogram

# Quick throughput check with watch:
watch -n 1 'curl -s http://localhost:8000/metrics | grep throughput'
```

---

### Recipe 4: Quantized Model Deployment

**vLLM with AWQ model:**
```bash
# AWQ models are pre-quantized on HuggingFace (look for "-AWQ" suffix)
vllm serve TheBloke/Llama-2-70B-Chat-AWQ \
  --quantization awq \
  --dtype auto \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90
```

**vLLM with FP8 model:**
```bash
# FP8 quantized models (Hopper/Ada GPUs required)
vllm serve neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8 \
  --dtype auto \
  --max-model-len 8192 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90

# Or quantize on-the-fly from FP16 (slower startup):
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --quantization fp8 \
  --dtype auto \
  --tensor-parallel-size 4
```

**vLLM with GPTQ model:**
```bash
# GPTQ models (look for "-GPTQ" suffix on HuggingFace)
vllm serve TheBloke/Llama-2-70B-Chat-GPTQ \
  --quantization gptq \
  --dtype float16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90
```

**SGLang with quantized models:**
```bash
# AWQ
python -m sglang.launch_server \
  --model-path TheBloke/Llama-2-70B-Chat-AWQ \
  --quantization awq \
  --port 30000

# FP8
python -m sglang.launch_server \
  --model-path neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8 \
  --port 30000

# GPTQ
python -m sglang.launch_server \
  --model-path TheBloke/Llama-2-70B-Chat-GPTQ \
  --quantization gptq \
  --port 30000
```

---

### Recipe 5: Benchmarking Your Deployment

**Python script to benchmark TTFT, TPOT, and throughput:**
```python
import time
import requests
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

API_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"

def benchmark_single_request(prompt, max_tokens=128):
    """Measure TTFT and TPOT for a single request."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }
    start = time.perf_counter()
    first_token_time = None
    token_count = 0

    with requests.post(API_URL, json=payload, stream=True) as resp:
        for line in resp.iter_lines():
            if line and line.startswith(b"data: "):
                data = line[6:]
                if data == b"[DONE]":
                    break
                chunk = json.loads(data)
                if chunk["choices"][0]["delta"].get("content"):
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    token_count += 1

    end = time.perf_counter()
    ttft = (first_token_time - start) if first_token_time else None
    total_time = end - (first_token_time or start)
    tpot = total_time / max(token_count - 1, 1) if token_count > 1 else None

    return {
        "ttft_ms": ttft * 1000 if ttft else None,
        "tpot_ms": tpot * 1000 if tpot else None,
        "tokens": token_count,
        "total_s": end - start,
    }

def load_test(num_requests=50, concurrency=10, prompt="Write a short story about a robot.", max_tokens=128):
    """Run concurrent requests and report aggregate metrics."""
    results = []
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(benchmark_single_request, prompt, max_tokens) for _ in range(num_requests)]
        for f in as_completed(futures):
            results.append(f.result())

    wall_time = time.perf_counter() - start
    ttfts = [r["ttft_ms"] for r in results if r["ttft_ms"] is not None]
    tpots = [r["tpot_ms"] for r in results if r["tpot_ms"] is not None]
    total_tokens = sum(r["tokens"] for r in results)

    print(f"=== Load Test Results ({num_requests} reqs, concurrency={concurrency}) ===")
    print(f"Total wall time:   {wall_time:.1f}s")
    print(f"Throughput:        {total_tokens / wall_time:.1f} tok/s")
    print(f"TTFT  p50/p90/p99: {np.percentile(ttfts, 50):.0f} / {np.percentile(ttfts, 90):.0f} / {np.percentile(ttfts, 99):.0f} ms")
    print(f"TPOT  p50/p90/p99: {np.percentile(tpots, 50):.1f} / {np.percentile(tpots, 90):.1f} / {np.percentile(tpots, 99):.1f} ms")
    print(f"Total tokens:      {total_tokens}")

if __name__ == "__main__":
    # Single request benchmark
    print("=== Single Request ===")
    result = benchmark_single_request("Explain general relativity in 3 sentences.")
    print(f"TTFT: {result['ttft_ms']:.0f}ms | TPOT: {result['tpot_ms']:.1f}ms | Tokens: {result['tokens']}")

    # Load test
    print()
    load_test(num_requests=50, concurrency=10)
```

**How to interpret the results:**

| Metric | Good | Acceptable | Investigate |
|--------|------|------------|-------------|
| TTFT p50 | <200ms | <500ms | >1000ms |
| TTFT p99 | <500ms | <2000ms | >5000ms |
| TPOT p50 | <30ms | <50ms | >80ms |
| Throughput (8B model, 1 GPU) | >1000 tok/s | >500 tok/s | <200 tok/s |
| Throughput (70B model, 4 GPU) | >2000 tok/s | >1000 tok/s | <500 tok/s |
| P99/P50 ratio | <2x | <3x | >5x (scheduling issue) |

---

## Real-World Performance Numbers

Measured on representative workloads (128 token prompt, 128 token output, batch size 16). Numbers are approximate and vary with software version, driver, and workload.

### LLaMA 3.1 8B on RTX 4090 (24GB VRAM)

| Precision | TTFT (ms) | Decode (tok/s) | Peak VRAM (GB) | Serving Framework |
|-----------|-----------|----------------|-----------------|-------------------|
| FP16 | ~35 | ~95 | ~16.5 | vLLM |
| AWQ INT4 | ~25 | ~130 | ~6.2 | vLLM |
| FP8 | ~30 | ~115 | ~9.0 | vLLM |

### LLaMA 3.1 70B on H100 80GB (TP=4)

| Precision | TTFT (ms) | Decode (tok/s) | Peak VRAM/GPU (GB) | Serving Framework |
|-----------|-----------|----------------|---------------------|-------------------|
| FP16 | ~120 | ~55 | ~38 | vLLM |
| AWQ INT4 | ~85 | ~80 | ~20 | vLLM |
| FP8 | ~95 | ~70 | ~22 | vLLM |

### Mistral 7B on RTX 3090 (24GB VRAM)

| Precision | TTFT (ms) | Decode (tok/s) | Peak VRAM (GB) | Serving Framework |
|-----------|-----------|----------------|-----------------|-------------------|
| FP16 | ~40 | ~75 | ~14.5 | vLLM |
| AWQ INT4 | ~28 | ~105 | ~5.0 | vLLM |
| FP8 | N/A (Ampere lacks native FP8) | N/A | N/A | N/A |

Notes:
- Decode tok/s is per-request in a batched setting (batch=16). Total throughput = tok/s x batch.
- RTX 3090 (Ampere) does not have native FP8 compute; use AWQ or GPTQ for quantization.
- TTFT scales roughly linearly with prompt length.
- Numbers measured with vLLM v0.6.x; SGLang produces comparable results for these workloads.

---

## Troubleshooting Guide

### 10 Most Common vLLM/SGLang Errors

**1. CUDA Out of Memory during model loading**
```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 512.00 MiB
```
Fix: Reduce `--gpu-memory-utilization` (e.g., 0.85), reduce `--max-model-len`, use quantization (`--quantization awq`), or increase `--tensor-parallel-size`.

**2. Not enough KV cache blocks**
```
ValueError: No available memory for the cache blocks. Try increasing `gpu_memory_utilization` when initializing the engine.
```
Fix: Increase `--gpu-memory-utilization` to 0.95, reduce `--max-model-len`, or use `--kv-cache-dtype fp8` to halve KV cache memory.

**3. Model not found / authentication error**
```
OSError: meta-llama/Llama-3.1-8B-Instruct is not accessible. You need to accept the license.
```
Fix: Accept the model license on HuggingFace, then `export HF_TOKEN=hf_your_token` or `huggingface-cli login`.

**4. CUDA graph capture failure**
```
RuntimeError: CUDA error: operation not permitted when stream is capturing
```
Fix: Start with `--enforce-eager` to disable CUDA graphs. If that works, the issue is a dynamic shape operation incompatible with graph capture. File a bug with the serving framework.

**5. Tensor parallel size mismatch**
```
ValueError: Total number of attention heads (32) must be divisible by tensor parallel size (3).
```
Fix: `--tensor-parallel-size` must evenly divide the model's `num_attention_heads`. Use 1, 2, 4, or 8 for most models.

**6. Tokenizer errors with chat template**
```
jinja2.exceptions.TemplateError: Conversation roles must alternate user/assistant/user/assistant
```
Fix: Ensure your request alternates roles correctly. If using a custom model, set `--chat-template` to a valid Jinja2 template file.

**7. SGLang FlashInfer compilation error**
```
ImportError: cannot import name 'BatchPrefillWithPagedKVCacheWrapper' from 'flashinfer'
```
Fix: Reinstall with matching CUDA version: `pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/`. Ensure your PyTorch CUDA version matches.

**8. Requests timing out under load**
```
requests.exceptions.ReadTimeout / 504 Gateway Timeout
```
Fix: Queue is full. Increase `--max-num-seqs`, add more GPUs, reduce `--max-model-len`, or add a load balancer with multiple vLLM instances.

**9. Slow first request (30-120 seconds)**
```
INFO: Capturing CUDA graphs... (this log appears, and then a long pause)
```
This is normal. vLLM/SGLang compile CUDA graphs on the first request for each batch size. Subsequent requests are fast. To warm up: send a dummy request after startup. To skip: use `--enforce-eager` (at the cost of ~5-10% throughput).

**10. NaN or garbage output**
```
Output contains repeated tokens, NaN logits, or gibberish
```
Fix: Usually a dtype/quantization issue. Try `--dtype float16` explicitly. If using a quantized model, ensure the quantization format matches the flag (`--quantization awq` for AWQ models). Check that the model files are not corrupted by redownloading (`huggingface-cli download --force`).
