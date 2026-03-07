---
id: runtime-compatibility-matrix
kind: document
title: "Runtime Compatibility Matrix: What Works Where"
category: runtime
summary: Compatibility matrix mapping optimization techniques to serving frameworks, with version requirements and known issues
tags:
  - runtime
  - vllm
  - sglang
  - tensorrt-llm
  - llama-cpp
  - exllamav2
  - transformers
  - compatibility
runtimes:
  - vllm
  - sglang
  - tensorrt-llm
  - llama-cpp
  - exllamav2
  - transformers
---

# Runtime Compatibility Matrix: What Works Where

This document is the definitive reference for which optimization techniques work with which
serving frameworks. Before recommending or attempting any optimization, consult this matrix
to avoid wasting time on unsupported combinations.

**How to read cell values:**
- Supported = works in production, with the minimum version noted
- Partial = works with caveats, limitations, or only on certain model architectures
- Not supported = does not work; do not attempt
- Planned = on the roadmap but not yet merged

---

## 1. Master Compatibility Table

### 1A. Quantization Techniques

| Technique | vLLM | SGLang | TensorRT-LLM | llama.cpp | ExLlamaV2 | HF Transformers |
|-----------|------|--------|---------------|-----------|-----------|-----------------|
| **AWQ INT4** | Supported (v0.3.0+) | Supported (v0.1.0+) | Supported (v0.7+) | Not supported | Not supported | Supported (AutoAWQ) |
| **GPTQ INT4** | Supported (v0.3.0+) | Supported (v0.1.0+) | Supported (v0.7+) | Not supported | Partial (via convert) | Supported (AutoGPTQ) |
| **FP8 W8A8** | Supported (v0.5.0+, Hopper/Ada) | Supported (v0.2.0+, Hopper/Ada) | Supported (v0.9+) | Not supported | Not supported | Partial (torch 2.4+) |
| **FP8 Weights-Only** | Supported (v0.5.0+) | Supported (v0.2.0+) | Supported (v0.9+) | Not supported | Not supported | Partial |
| **GGUF Q4_K_M** | Supported (v0.5.3+) | Not supported | Not supported | Supported (native) | Not supported | Not supported |
| **GGUF Q5_K_M** | Supported (v0.5.3+) | Not supported | Not supported | Supported (native) | Not supported | Not supported |
| **GGUF Q8_0** | Supported (v0.5.3+) | Not supported | Not supported | Supported (native) | Not supported | Not supported |
| **EXL2** | Not supported | Not supported | Not supported | Not supported | Supported (native) | Not supported |
| **BitsAndBytes INT8** | Supported (v0.4.0+) | Not supported | Not supported | Not supported | Not supported | Supported (native) |
| **BitsAndBytes NF4** | Supported (v0.4.0+) | Not supported | Not supported | Not supported | Not supported | Supported (native) |
| **Marlin Kernels (INT4)** | Supported (v0.4.2+) | Supported (v0.1.5+) | Not supported | Not supported | Not supported | Not supported |
| **Marlin Kernels (FP8)** | Supported (v0.5.1+) | Supported (v0.2.0+) | Not supported | Not supported | Not supported | Not supported |
| **INT8 W8A8 (SmoothQuant)** | Supported (v0.5.0+) | Partial | Supported (v0.7+) | Not supported | Not supported | Not supported |
| **AQLM** | Supported (v0.4.3+) | Not supported | Not supported | Not supported | Not supported | Supported |
| **QuIP#** | Supported (v0.4.3+) | Not supported | Not supported | Not supported | Not supported | Supported |

### 1B. Attention & Memory Optimizations

| Technique | vLLM | SGLang | TensorRT-LLM | llama.cpp | ExLlamaV2 | HF Transformers |
|-----------|------|--------|---------------|-----------|-----------|-----------------|
| **FlashAttention-2** | Supported (v0.2.0+) | Supported (v0.1.0+) | Supported (plugin) | Not supported (own impl) | Not supported (own impl) | Supported (torch 2.2+) |
| **FlashAttention-3** | Supported (v0.6.0+, Hopper) | Supported (v0.3.0+, Hopper) | Planned | Not supported | Not supported | Planned |
| **FlashInfer** | Supported (v0.5.0+) | Supported (native default) | Not supported | Not supported | Not supported | Not supported |
| **PagedAttention** | Supported (native, v0.1.0+) | Supported (v0.1.0+) | Supported (v0.7+) | Not supported | Partial (own paging) | Not supported |
| **KV Cache FP8** | Supported (v0.5.0+, Hopper/Ada) | Supported (v0.2.0+) | Supported (v0.9+) | Not supported | Not supported | Not supported |
| **KV Cache INT8** | Supported (v0.5.0+) | Partial | Supported (v0.9+) | Supported (native Q8) | Not supported | Not supported |
| **Prefix Caching** | Supported (v0.4.0+) | Supported (RadixAttention, native) | Supported (v0.9+) | Partial (prompt cache) | Not supported | Not supported |
| **Chunked Prefill** | Supported (v0.4.1+) | Supported (v0.1.5+) | Supported (v0.8+) | Not supported | Not supported | Not supported |
| **Sliding Window Attention** | Supported (v0.3.0+) | Supported (v0.1.0+) | Supported (v0.8+) | Supported (native) | Supported | Supported |
| **Continuous Batching** | Supported (native) | Supported (native) | Supported (native) | Not supported | Not supported | Not supported |

### 1C. Compilation & Kernel Optimizations

| Technique | vLLM | SGLang | TensorRT-LLM | llama.cpp | ExLlamaV2 | HF Transformers |
|-----------|------|--------|---------------|-----------|-----------|-----------------|
| **CUDA Graphs** | Supported (v0.4.0+) | Supported (v0.1.0+) | Supported (native) | Not supported | Supported | Not supported |
| **torch.compile** | Supported (v0.5.0+, experimental) | Partial (v0.2.0+) | Not supported (uses TRT) | Not supported | Not supported | Supported (torch 2.0+) |
| **Custom Triton Ops** | Supported (native) | Supported (native) | Not supported | Not supported | Not supported | Partial (via extensions) |
| **TensorRT Engine** | Not supported | Not supported | Supported (native) | Not supported | Not supported | Not supported |

### 1D. Parallelism & Scaling

| Technique | vLLM | SGLang | TensorRT-LLM | llama.cpp | ExLlamaV2 | HF Transformers |
|-----------|------|--------|---------------|-----------|-----------|-----------------|
| **Tensor Parallelism** | Supported (v0.2.0+) | Supported (v0.1.0+) | Supported (native) | Partial (split layers) | Not supported | Supported (device_map) |
| **Pipeline Parallelism** | Supported (v0.5.0+) | Partial | Supported (native) | Not supported | Not supported | Supported (device_map) |
| **Speculative Decoding (EAGLE)** | Supported (v0.5.0+) | Supported (v0.2.0+) | Planned | Not supported | Not supported | Not supported |
| **Speculative Decoding (Medusa)** | Supported (v0.4.3+) | Supported (v0.1.5+) | Supported (v0.9+) | Partial (experimental) | Not supported | Not supported |
| **Speculative Decoding (Draft Model)** | Supported (v0.4.0+) | Supported (v0.1.5+) | Supported (v0.8+) | Supported (v0.3+) | Not supported | Partial (assisted generation) |

### 1E. Quick Lookup: Common Scenarios

**"Can I use AWQ with SGLang on Ada GPU?"** -- Yes. SGLang supports AWQ INT4 since v0.1.0. Ada GPUs (RTX 4090, L40S) are fully supported. Use `--quantization awq` flag.

**"Can I use FP8 on Ampere?"** -- No. FP8 requires Hopper (H100, H200) or Ada (RTX 4090, L40S). On Ampere, use AWQ INT4 or GPTQ INT4 instead.

**"Can I use GGUF models with vLLM?"** -- Yes, since v0.5.3. But GGUF in vLLM is slower than native AWQ/GPTQ. Use GGUF with llama.cpp for best single-user performance.

**"Can I do speculative decoding in llama.cpp?"** -- Yes, draft-model speculative decoding is supported since llama.cpp b2000+. EAGLE/Medusa are not supported.

**"Can I use ExLlamaV2 for multi-user serving?"** -- Not recommended. ExLlamaV2 has no continuous batching. Use vLLM or SGLang for multi-user serving scenarios.

---

## 2. Per-Runtime Quick Reference

### 2A. vLLM

**Overview**: The most widely adopted open-source LLM serving engine. Invented PagedAttention. Strong quantization support, continuous batching, and multi-GPU serving.

**Installation**:
```bash
# Standard install (CUDA 12.1+)
pip install vllm

# Specific version
pip install vllm==0.6.4

# From source (for latest features)
pip install git+https://github.com/vllm-project/vllm.git
```

**Supported Quantizations**:
```bash
# AWQ INT4
python -m vllm.entrypoints.openai.api_server \
    --model TheBloke/Llama-2-70B-Chat-AWQ \
    --quantization awq \
    --dtype half

# GPTQ INT4
python -m vllm.entrypoints.openai.api_server \
    --model TheBloke/Llama-2-70B-Chat-GPTQ \
    --quantization gptq

# FP8 W8A8 (Hopper/Ada only)
python -m vllm.entrypoints.openai.api_server \
    --model neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8 \
    --quantization fp8

# Marlin (fast INT4 kernels, Ampere+)
python -m vllm.entrypoints.openai.api_server \
    --model TheBloke/Llama-2-70B-Chat-AWQ \
    --quantization marlin

# GGUF (v0.5.3+)
python -m vllm.entrypoints.openai.api_server \
    --model ./model.Q4_K_M.gguf \
    --tokenizer meta-llama/Llama-2-7b-chat-hf

# BitsAndBytes NF4
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-70b-chat-hf \
    --quantization bitsandbytes \
    --load-format bitsandbytes
```

**Attention Backends**:
```bash
# FlashAttention-2 (default on Ampere/Ada)
--attention-backend flash-attn

# FlashInfer (better for decode-heavy workloads)
--attention-backend flashinfer

# FlashAttention-3 (Hopper only, v0.6.0+)
--attention-backend flash-attn  # auto-selects FA3 on Hopper when available
```

**Speculative Decoding**:
```bash
# Draft model
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-70b-chat-hf \
    --speculative-model meta-llama/Llama-2-7b-chat-hf \
    --num-speculative-tokens 5

# EAGLE
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-70b-chat-hf \
    --speculative-model eagle-model-path \
    --speculative-draft-tensor-parallel-size 1

# Medusa
python -m vllm.entrypoints.openai.api_server \
    --model medusa-model-path \
    --speculative-model "[ngram]" \
    --num-speculative-tokens 5
```

**CUDA Graphs**:
```bash
# Enabled by default. Disable for debugging:
--enforce-eager

# Control max batch size for CUDA graph capture:
--max-num-seqs 256
```

**Key Flags**:
```bash
--tensor-parallel-size N          # Tensor parallelism across N GPUs
--pipeline-parallel-size N        # Pipeline parallelism (v0.5.0+)
--enable-prefix-caching           # Enable automatic prefix caching
--enable-chunked-prefill          # Enable chunked prefill
--max-model-len 32768             # Maximum context length
--gpu-memory-utilization 0.9      # Fraction of GPU memory to use
--kv-cache-dtype fp8              # FP8 KV cache (Hopper/Ada)
--dtype auto                      # auto, half, bfloat16, float
--max-num-batched-tokens 8192     # Max tokens in a batch
--swap-space 4                    # CPU swap space in GB
```

**Version Milestones**:
| Version | Key Additions |
|---------|--------------|
| v0.2.x | PagedAttention, FlashAttention-2, basic TP |
| v0.3.x | AWQ, GPTQ, sliding window |
| v0.4.x | CUDA graphs, prefix caching, chunked prefill, BnB, Medusa, Marlin |
| v0.5.x | FP8, FlashInfer, pipeline parallelism, EAGLE, GGUF, torch.compile |
| v0.6.x | FlashAttention-3, multi-modal support, improved scheduling |

**Known Issues and Gotchas**:
- GGUF support is slower than native AWQ/GPTQ because vLLM dequantizes GGUF to FP16 at load time
- Marlin kernels require models in Marlin format or AWQ/GPTQ models that can be auto-converted
- FP8 KV cache on Ada GPUs may show slightly lower accuracy than on Hopper due to different FP8 formats (E4M3 vs E5M2)
- CUDA graphs are incompatible with `--enforce-eager` and some debugging flags
- BitsAndBytes integration is slower than AWQ/GPTQ for serving; use only when pre-quantized model is unavailable
- torch.compile support is experimental; may increase startup time by 5-15 minutes
- Pipeline parallelism requires homogeneous GPUs

---

### 2B. SGLang

**Overview**: High-performance serving framework built around RadixAttention for automatic prefix caching. Uses FlashInfer as its default attention backend. Often matches or exceeds vLLM throughput.

**Installation**:
```bash
# Full install with all backends
pip install "sglang[all]"

# Minimal install
pip install sglang

# From source
pip install git+https://github.com/sgl-project/sglang.git
```

**Supported Quantizations**:
```bash
# AWQ INT4
python -m sglang.launch_server \
    --model-path TheBloke/Llama-2-70B-Chat-AWQ \
    --quantization awq

# GPTQ INT4
python -m sglang.launch_server \
    --model-path TheBloke/Llama-2-70B-Chat-GPTQ \
    --quantization gptq

# FP8 (Hopper/Ada)
python -m sglang.launch_server \
    --model-path neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8 \
    --quantization fp8

# Marlin
python -m sglang.launch_server \
    --model-path TheBloke/Llama-2-70B-Chat-AWQ \
    --quantization marlin
```

**RadixAttention (Automatic Prefix Caching)**:
- Enabled by default; no flag needed
- Uses a radix tree to automatically detect and cache shared prefixes
- Particularly effective for multi-turn chat, few-shot prompting, and tool-use patterns
- Superior to vLLM's prefix caching for workloads with diverse prefix patterns

**Attention Backend**:
```bash
# FlashInfer (default, recommended)
--attention-backend flashinfer

# FlashAttention-2
--attention-backend flash-attn
```

**Speculative Decoding**:
```bash
# EAGLE
python -m sglang.launch_server \
    --model-path meta-llama/Llama-2-70b-chat-hf \
    --speculative-algorithm EAGLE \
    --speculative-eagle-path eagle-model-path \
    --speculative-num-draft-tokens 5

# Draft model
python -m sglang.launch_server \
    --model-path meta-llama/Llama-2-70b-chat-hf \
    --speculative-algorithm DRAFT \
    --speculative-draft-model-path meta-llama/Llama-2-7b-chat-hf
```

**Key Flags**:
```bash
--tp N                            # Tensor parallelism
--chunked-prefill-size 8192       # Chunked prefill token budget
--mem-fraction-static 0.88        # GPU memory fraction for KV cache
--max-running-requests 128        # Max concurrent decoding requests
--schedule-policy lpm             # Scheduling: lpm (longest prefix match) or fcfs
--enable-torch-compile            # Enable torch.compile (experimental)
--cuda-graph-max-bs 160           # Max batch size for CUDA graph capture
--kv-cache-dtype fp8_e5m2         # FP8 KV cache
--context-length 32768            # Override default context length
```

**Known Issues and Gotchas**:
- Does not support GGUF models; convert to HF format first
- BitsAndBytes not supported; use AWQ or GPTQ instead
- FlashInfer must be installed separately on some platforms: `pip install flashinfer`
- Pipeline parallelism is experimental and less mature than vLLM's
- RadixAttention memory overhead is minimal (~2-5%) but can affect very memory-constrained setups
- torch.compile support lags behind vLLM's implementation

---

### 2C. TensorRT-LLM

**Overview**: NVIDIA's official inference optimization library. Converts models to TensorRT engines for maximum performance on NVIDIA GPUs. Highest throughput on Hopper GPUs but requires explicit build step.

**Installation**:
```bash
# PyPI install (latest)
pip install tensorrt_llm -U --pre --extra-index-url https://pypi.nvidia.com

# Verify installation
python -c "import tensorrt_llm; print(tensorrt_llm.__version__)"

# Docker (recommended for production)
docker pull nvcr.io/nvidia/tritonserver:24.07-trtllm-python-py3
```

**Build Process** (required before serving):
```bash
# Step 1: Convert HF checkpoint to TRT-LLM checkpoint
python convert_checkpoint.py \
    --model_dir ./llama-2-70b-chat-hf \
    --output_dir ./tllm_checkpoint \
    --dtype float16 \
    --tp_size 4

# Step 2: Build TensorRT engine
trtllm-build \
    --checkpoint_dir ./tllm_checkpoint \
    --output_dir ./tllm_engines \
    --gemm_plugin float16 \
    --max_batch_size 64 \
    --max_input_len 4096 \
    --max_seq_len 8192 \
    --paged_kv_cache enable \
    --use_custom_all_reduce enable
```

**Quantization at Build Time**:
```bash
# FP8 (Hopper)
trtllm-build \
    --checkpoint_dir ./tllm_checkpoint \
    --output_dir ./tllm_engines \
    --use_fp8_context_fmha enable \
    --strongly_typed

# INT8 SmoothQuant
python convert_checkpoint.py \
    --model_dir ./llama-2-70b-chat-hf \
    --output_dir ./tllm_checkpoint \
    --dtype float16 \
    --smoothquant 0.5 \
    --per_token --per_channel

# AWQ INT4
python ../quantization/quantize.py \
    --model_dir ./llama-2-70b-chat-hf \
    --output_dir ./tllm_checkpoint \
    --dtype float16 \
    --qformat int4_awq \
    --awq_block_size 128 \
    --calib_size 512

# GPTQ INT4
python convert_checkpoint.py \
    --model_dir ./llama-2-70b-chat-gptq \
    --output_dir ./tllm_checkpoint \
    --dtype float16 \
    --use_weight_only \
    --weight_only_precision int4_gptq
```

**Plugin System**:
```bash
# Key plugins (enabled via trtllm-build flags)
--gemm_plugin float16          # Custom GEMM kernels (strongly recommended)
--gpt_attention_plugin float16 # Custom attention (required for most features)
--paged_kv_cache enable        # PagedAttention
--remove_input_padding enable  # Ragged batching
--context_fmha enable          # Flash attention for context phase
--use_custom_all_reduce enable # Optimized allreduce for TP
--multi_block_mode enable      # Multi-block attention for long sequences
```

**Speculative Decoding**:
```bash
# Draft model approach
# Build both target and draft engines, then:
python run.py \
    --engine_dir ./target_engines \
    --draft_engine_dir ./draft_engines \
    --use_draft_logits \
    --max_draft_len 5

# Medusa
# Build with medusa heads:
trtllm-build \
    --checkpoint_dir ./medusa_checkpoint \
    --speculative_decoding_mode medusa \
    --max_draft_len 63
```

**Key Flags for Serving** (via Triton Inference Server):
```bash
# Launch with Triton
python launch_triton_server.py \
    --world_size 4 \
    --model_repo ./triton_model_repo

# Inflight batching config (in config.pbtxt)
# max_batch_size: 128
# batching_strategy: "inflight_fused_batching"
# kv_cache_free_gpu_mem_fraction: 0.85
```

**Known Issues and Gotchas**:
- Must rebuild engine when changing max batch size, sequence length, or TP degree
- Engine is GPU-architecture-specific: an engine built for H100 will not run on A100
- Build process can take 10-60 minutes depending on model size
- Quantization calibration requires representative dataset and can be finicky
- Not all HuggingFace model architectures are supported; check supported model list
- Plugin system interactions can be complex; missing plugins cause cryptic errors
- No GGUF or EXL2 support; must start from HF checkpoints
- Docker deployment strongly recommended over bare-metal for dependency management

---

### 2D. llama.cpp

**Overview**: CPU/GPU inference engine using GGUF format. Best for single-user, local inference. Excellent Apple Silicon support via Metal. Efficient GPU offloading with partial layer loading.

**Installation**:
```bash
# Build from source (CUDA)
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc)

# Build from source (Metal / Apple Silicon)
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release -j$(sysctl -n hw.ncpu)

# Build from source (ROCm / AMD)
cmake -B build -DGGML_HIPBLAS=ON
cmake --build build --config Release -j$(nproc)

# Build from source (Vulkan)
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release -j$(nproc)

# Python bindings
pip install llama-cpp-python

# Python bindings with CUDA
CMAKE_ARGS="-DGGML_CUDA=ON" pip install llama-cpp-python

# Python bindings with Metal
CMAKE_ARGS="-DGGML_METAL=ON" pip install llama-cpp-python
```

**GGUF Quantization Formats** (from highest to lowest quality):
| Format | Bits/Weight | Quality | Speed | Use Case |
|--------|-------------|---------|-------|----------|
| F16 | 16.0 | Baseline | Slowest | Reference only |
| Q8_0 | 8.5 | Near-lossless | Fast | When VRAM allows |
| Q6_K | 6.6 | Excellent | Fast | High quality + good speed |
| Q5_K_M | 5.7 | Very good | Faster | Balanced |
| Q4_K_M | 4.8 | Good | Fast | **Recommended default** |
| Q4_K_S | 4.6 | Good | Fast | Slightly smaller |
| Q3_K_M | 3.9 | Acceptable | Fastest | Memory constrained |
| Q2_K | 3.4 | Poor | Fastest | Extreme memory constraints |
| IQ4_XS | 4.3 | Good | Fast | Importance-matrix quant, better than Q4_K_M at same size |
| IQ3_XXS | 3.1 | Fair | Fast | Ultra-low bit, needs calibration |

**GPU Offloading**:
```bash
# Offload all layers to GPU
./llama-cli -m model.gguf -ngl 999

# Offload specific number of layers
./llama-cli -m model.gguf -ngl 35

# Split across multiple GPUs (rough layer split)
./llama-cli -m model.gguf -ngl 999 --tensor-split 0.5,0.5

# Check how many layers fit
# Rule of thumb: each layer of a 7B model ~ 200MB in Q4_K_M
```

**Server Mode** (OpenAI-compatible API):
```bash
./llama-server \
    -m model.gguf \
    -ngl 999 \
    --host 0.0.0.0 \
    --port 8080 \
    -c 8192 \
    --n-predict 2048 \
    -np 4               # parallel slots (concurrent requests)
```

**Speculative Decoding**:
```bash
# Draft model speculative decoding
./llama-cli -m large-model.gguf \
    --draft-model small-model.gguf \
    --draft-max 8 \
    --draft-min 1 \
    -ngl 999
```

**Prompt Caching**:
```bash
# Save prompt cache
./llama-cli -m model.gguf --prompt-cache cache.bin -f prompt.txt

# Reuse prompt cache
./llama-cli -m model.gguf --prompt-cache cache.bin --prompt-cache-ro -f prompt.txt
```

**Key Flags**:
```bash
-ngl N              # Number of layers to offload to GPU (999 = all)
-c N                # Context size (default: 512)
-b N                # Batch size for prompt processing (default: 2048)
-t N                # Number of CPU threads
--mmap              # Use mmap for model loading (default: on)
--mlock             # Lock model in RAM (prevent swapping)
-np N               # Number of parallel decoding slots (server mode)
--flash-attn        # Enable Flash Attention (CUDA only, experimental)
--cont-batching     # Enable continuous batching in server mode
--no-mmap           # Disable mmap (forces full load into RAM)
-fa                 # Short form for --flash-attn
```

**Known Issues and Gotchas**:
- No AWQ or GPTQ support; must convert to GGUF format first
- No continuous batching in the traditional sense; server mode uses slot-based concurrency
- Flash Attention support is experimental and CUDA-only
- Metal (Apple Silicon) performance is excellent but lacks some features available in CUDA builds
- Multi-GPU support is basic layer-splitting, not true tensor parallelism
- Prompt cache is per-model and not shared across server slots
- Very large context (>32K) requires explicit `-c` flag and sufficient memory
- IQ (importance-matrix) quants require calibration data for best results

---

### 2E. ExLlamaV2

**Overview**: Highly optimized inference library for GPTQ/EXL2 quantized models. Best single-GPU decode performance for quantized models. No continuous batching; designed for single-user or small-batch use.

**Installation**:
```bash
# PyPI
pip install exllamav2

# From source (for latest features)
pip install git+https://github.com/turboderp/exllamav2.git
```

**EXL2 Format** (native, recommended):
```bash
# Convert HF model to EXL2
python convert.py \
    -i ./llama-2-70b-chat-hf \
    -o ./llama-2-70b-exl2-4.0bpw \
    -cf ./llama-2-70b-exl2-4.0bpw \
    -b 4.0 \                     # target bits per weight
    -cal ./calibration_data.parquet
```

EXL2 bits-per-weight options:
| BPW | Quality | Comparable GGUF |
|-----|---------|-----------------|
| 8.0 | Near-lossless | Q8_0 |
| 6.0 | Excellent | Q6_K |
| 5.0 | Very good | Q5_K_M |
| 4.0 | Good | Q4_K_M |
| 3.0 | Acceptable | Q3_K_M |
| 2.5 | Poor | Q2_K |

**Running Inference**:
```python
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2StreamingGenerator, ExLlamaV2Sampler

config = ExLlamaV2Config(model_dir)
model = ExLlamaV2(config)
cache = ExLlamaV2Cache(model, max_seq_len=8192, lazy=True)
model.load_autosplit(cache)  # Auto-split across GPUs
tokenizer = ExLlamaV2Tokenizer(config)
generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)
```

**TabbyAPI** (OpenAI-compatible server for ExLlamaV2):
```bash
pip install tabbyapi
python -m tabbyapi --model-dir ./llama-2-70b-exl2-4.0bpw
```

**Key Features**:
- CUDA graphs enabled by default for decode
- Automatic multi-GPU layer splitting (not tensor parallelism)
- Dynamic batching for small batch sizes (1-8)
- Custom CUDA kernels for EXL2/GPTQ dequantization + matmul fused ops

**Known Issues and Gotchas**:
- No continuous batching; not suitable for high-concurrency serving
- No tensor parallelism; multi-GPU is layer-split only
- EXL2 conversion requires calibration dataset and can take hours for 70B+ models
- Only supports decoder-only transformer models (no encoder-decoder, no vision)
- CUDA only; no Metal, ROCm, or Vulkan support
- Limited model architecture support compared to vLLM or llama.cpp
- No FP8, no FlashAttention integration (uses own optimized attention kernels)

---

### 2F. HF Transformers

**Overview**: The reference implementation. Broadest model support but slowest for production serving. Use for prototyping, fine-tuning, or when no other runtime supports your model architecture.

**Installation**:
```bash
pip install transformers accelerate
pip install bitsandbytes                    # For INT8/NF4 quantization
pip install auto-gptq                       # For GPTQ models
pip install autoawq                         # For AWQ models
pip install optimum                         # For optimization utilities
pip install flash-attn --no-build-isolation # For FlashAttention-2
```

**Quantization Loaders**:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# BitsAndBytes INT8
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-chat-hf",
    load_in_8bit=True,
    device_map="auto"
)

# BitsAndBytes NF4 (QLoRA style)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-chat-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# AWQ
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-70B-Chat-AWQ",
    device_map="auto"
)

# GPTQ
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-70B-Chat-GPTQ",
    device_map="auto"
)
```

**torch.compile** (PyTorch 2.0+):
```python
import torch

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Full model compile
model = torch.compile(model, mode="reduce-overhead")

# Or just compile the forward pass
model.forward = torch.compile(model.forward, mode="reduce-overhead")
```

**FlashAttention-2**:
```python
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-chat-hf",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)
```

**SDPA (Scaled Dot-Product Attention)** -- built into PyTorch 2.0+:
```python
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-chat-hf",
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    device_map="auto"
)
```

**Multi-GPU with device_map**:
```python
# Auto split across available GPUs
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-chat-hf",
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# Custom device map
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-chat-hf",
    device_map={
        "model.embed_tokens": 0,
        "model.layers.0-19": 0,
        "model.layers.20-39": 1,
        "model.layers.40-59": 2,
        "model.layers.60-79": 3,
        "model.norm": 3,
        "lm_head": 3,
    }
)
```

**Assisted Generation (Speculative Decoding)**:
```python
from transformers import AutoModelForCausalLM

target = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-70b-chat-hf", device_map="auto")
draft = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map="auto")

outputs = target.generate(
    input_ids,
    assistant_model=draft,
    max_new_tokens=256,
    do_sample=False
)
```

**Known Issues and Gotchas**:
- No continuous batching, no PagedAttention; not suitable for production serving
- BitsAndBytes quantization is slower than AWQ/GPTQ Marlin kernels by 2-3x for inference
- device_map="auto" does naive layer splitting, not tensor parallelism
- torch.compile has long warmup time (first call can take minutes)
- torch.compile with dynamic shapes (varying sequence lengths) can cause recompilation
- FlashAttention-2 requires separate pip install and CUDA toolkit
- SDPA is a good default if FlashAttention-2 is hard to install
- FP8 support is limited; use vLLM or TensorRT-LLM for FP8 serving

---

## 3. Decision Flowchart

Use this flowchart to select the right runtime for your use case.

```
Is your model a standard HuggingFace model?
|
+-- Yes
|   |
|   +-- Do you need serving (multi-user, high throughput)?
|       |
|       +-- Yes
|       |   |
|       |   +-- Is your GPU Hopper (H100/H200) or Ada (RTX 4090/L40S)?
|       |   |   |
|       |   |   +-- Yes (Hopper/Ada)
|       |   |   |   |
|       |   |   |   +-- Need maximum throughput, willing to build engines?
|       |   |   |   |   +-- Yes --> TensorRT-LLM with FP8
|       |   |   |   |   +-- No  --> vLLM or SGLang with FP8 or AWQ
|       |   |   |   |
|       |   |   |   +-- Lots of shared prefixes (chat, few-shot)?
|       |   |   |       +-- Yes --> SGLang (RadixAttention)
|       |   |   |       +-- No  --> vLLM (more mature ecosystem)
|       |   |   |
|       |   |   +-- No (Ampere: A100/A10G/RTX 3090)
|       |   |       |
|       |   |       +-- Use vLLM with AWQ INT4 or GPTQ INT4
|       |   |       +-- Use Marlin kernels for best INT4 perf
|       |   |       +-- SGLang also works well here
|       |   |
|       |   +-- Consumer GPU (RTX 3060-4070)?
|       |       +-- Model fits in VRAM with AWQ? --> vLLM with AWQ
|       |       +-- Model doesn't fit? --> llama.cpp with GGUF (partial offload)
|       |
|       +-- No (single-user, local inference)
|           |
|           +-- NVIDIA GPU?
|           |   |
|           |   +-- Want best quality per VRAM? --> ExLlamaV2 with EXL2
|           |   +-- Want simplicity? --> llama.cpp with GGUF Q4_K_M
|           |   +-- Need very long context? --> llama.cpp (efficient KV cache)
|           |
|           +-- Apple Silicon?
|           |   +-- Use llama.cpp with Metal (-DGGML_METAL=ON)
|           |   +-- GGUF Q4_K_M for most models
|           |   +-- Q6_K or Q8_0 if unified memory allows
|           |
|           +-- AMD GPU?
|           |   +-- llama.cpp with ROCm (-DGGML_HIPBLAS=ON)
|           |   +-- vLLM ROCm build (less mature)
|           |
|           +-- CPU only?
|               +-- llama.cpp with Q4_K_M
|               +-- Use -t N to set thread count
|
+-- No (custom/non-standard model)
    |
    +-- Is there a vLLM model loader for it?
    |   +-- Yes --> Use vLLM (check: python -c "from vllm import LLM; LLM('model')")
    |   +-- No  --> Check SGLang support
    |
    +-- Is it a standard decoder-only transformer (just custom layers)?
    |   +-- Yes --> HF Transformers + torch.compile
    |   +-- No  --> HF Transformers (may need custom code)
    |
    +-- Is it an encoder-decoder model?
        +-- HF Transformers (only option with full support)
```

---

## 4. Environment Setup Recipes

### 4A. vLLM (CUDA 12.1+)

```bash
# Create environment
conda create -n vllm python=3.11 -y
conda activate vllm

# Install vLLM
pip install vllm

# Verify
python -c "import vllm; print(vllm.__version__)"

# Quick test
python -m vllm.entrypoints.openai.api_server \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --dtype half \
    --port 8000

# Test with curl
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "prompt": "Hello", "max_tokens": 50}'
```

### 4B. SGLang

```bash
# Create environment
conda create -n sglang python=3.11 -y
conda activate sglang

# Install SGLang with all dependencies
pip install "sglang[all]"

# Install FlashInfer separately if needed
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/

# Quick test
python -m sglang.launch_server \
    --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --port 8000
```

### 4C. TensorRT-LLM

```bash
# Option 1: pip install (requires CUDA 12.4+)
conda create -n trtllm python=3.11 -y
conda activate trtllm

pip install tensorrt_llm -U --pre --extra-index-url https://pypi.nvidia.com

# Verify
python -c "import tensorrt_llm; print(tensorrt_llm.__version__)"

# Option 2: Docker (recommended)
docker pull nvcr.io/nvidia/tritonserver:24.07-trtllm-python-py3
docker run --gpus all -it --rm \
    -v /path/to/models:/models \
    nvcr.io/nvidia/tritonserver:24.07-trtllm-python-py3

# Option 3: Build from source
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
pip install -e .
```

### 4D. llama.cpp (CUDA)

```bash
# Clone and build
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# CUDA build
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc)

# Test
./build/bin/llama-cli -m model.Q4_K_M.gguf -ngl 999 -p "Hello" -n 50

# Convert HF model to GGUF
pip install -r requirements/requirements-convert_hf_to_gguf.txt
python convert_hf_to_gguf.py ./model-directory --outtype q8_0

# Quantize further
./build/bin/llama-quantize model-f16.gguf model-Q4_K_M.gguf Q4_K_M
```

### 4E. llama.cpp (Apple Silicon / Metal)

```bash
# Clone and build
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Metal build (Apple Silicon)
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release -j$(sysctl -n hw.ncpu)

# Test (Metal uses GPU by default with -ngl)
./build/bin/llama-cli -m model.Q4_K_M.gguf -ngl 999 -p "Hello" -n 50

# Python bindings with Metal
CMAKE_ARGS="-DGGML_METAL=ON" pip install llama-cpp-python
```

### 4F. ExLlamaV2

```bash
# Create environment
conda create -n exl2 python=3.11 -y
conda activate exl2

# Install
pip install exllamav2

# Or from source for latest features
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/turboderp/exllamav2.git

# Quick test
python -c "import exllamav2; print('ExLlamaV2 installed')"

# TabbyAPI for serving
pip install tabbyapi
python -m tabbyapi --model-dir ./model-exl2-4.0bpw
```

### 4G. HF Transformers (Full Stack)

```bash
# Create environment
conda create -n hf python=3.11 -y
conda activate hf

# Core packages
pip install transformers accelerate torch

# Quantization backends
pip install bitsandbytes       # BnB INT8/NF4
pip install auto-gptq          # GPTQ
pip install autoawq            # AWQ
pip install optimum            # Optimization utilities

# Flash Attention (optional, requires CUDA toolkit)
pip install flash-attn --no-build-isolation

# Quick test
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
model = AutoModelForCausalLM.from_pretrained(
    'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    torch_dtype=torch.float16,
    device_map='auto'
)
print('HF Transformers ready')
"
```

---

## 5. Dependency Requirements Table

| Runtime | Python | CUDA Toolkit | PyTorch | GPU Arch Min | VRAM Min | Notes |
|---------|--------|-------------|---------|-------------|----------|-------|
| **vLLM** | 3.9-3.12 | 12.1+ | 2.3+ | sm_70 (Volta) | 4GB | sm_80+ recommended; Volta is best-effort |
| **SGLang** | 3.9-3.12 | 12.1+ | 2.3+ | sm_80 (Ampere) | 4GB | FlashInfer requires sm_80+; no Volta |
| **TensorRT-LLM** | 3.10-3.12 | 12.4+ | 2.3+ | sm_80 (Ampere) | 8GB | Hopper recommended; Docker strongly preferred |
| **llama.cpp** | N/A (C++) | 11.7+ (opt) | N/A | sm_60 (Pascal) | 2GB | Also supports Metal, ROCm, Vulkan, CPU-only |
| **llama-cpp-python** | 3.8-3.12 | 11.7+ (opt) | N/A | sm_60 (Pascal) | 2GB | Python bindings; same backend as llama.cpp |
| **ExLlamaV2** | 3.9-3.12 | 11.8+ | 2.1+ | sm_75 (Turing) | 4GB | CUDA only; no Metal/ROCm/Vulkan |
| **HF Transformers** | 3.9-3.12 | 11.8+ | 2.0+ | sm_37 (Kepler) | 2GB | Broadest compatibility; CPU also works |

### Extended Dependency Notes

**vLLM**:
- Requires NCCL for multi-GPU (installed automatically with PyTorch)
- Ray is optional (for multi-node distributed serving)
- FlashInfer backend requires separate install: `pip install flashinfer`
- CUDA 12.1 is the minimum; 12.4+ recommended for best FP8 support

**SGLang**:
- FlashInfer is a hard dependency (installed with `sglang[all]`)
- Requires Triton (installed automatically with PyTorch)
- zmq used for inter-process communication
- Requires Ampere+ (sm_80); will not work on Turing or older

**TensorRT-LLM**:
- Requires TensorRT 10.x (installed automatically)
- cuBLAS, cuDNN installed as part of CUDA toolkit
- Triton Inference Server recommended for production deployment
- MPI required for multi-node (openmpi or mpich)
- polygraphy, onnx optional for debugging

**llama.cpp**:
- CMake 3.21+ required for building
- CUDA toolkit optional (CPU works, just slower)
- Metal support requires macOS 13+ and Xcode command line tools
- ROCm 5.6+ for AMD GPU support
- No Python dependency for C++ build

**ExLlamaV2**:
- CUDA toolkit must match PyTorch CUDA version
- Ninja build system recommended for faster compilation
- safetensors required for model loading
- sentencepiece or tiktoken for tokenization (model-dependent)

**HF Transformers**:
- accelerate required for `device_map` functionality
- safetensors recommended (faster loading than pickle)
- tokenizers (Rust-based, installed automatically)
- BitsAndBytes requires CUDA and Linux (no Windows native, WSL works)
- flash-attn requires CUDA toolkit matching PyTorch + Ninja + packaging

---

## 6. Cross-Reference: Quantization Format Compatibility

This table maps quantization file formats to which runtimes can load them directly,
to avoid unnecessary format conversions.

| Source Format | vLLM | SGLang | TensorRT-LLM | llama.cpp | ExLlamaV2 | HF Transformers |
|--------------|------|--------|---------------|-----------|-----------|-----------------|
| **HF safetensors (FP16/BF16)** | Direct | Direct | Convert required | Convert to GGUF | Convert to EXL2 | Direct |
| **AWQ safetensors** | Direct | Direct | Convert required | Not supported | Not supported | Direct (AutoAWQ) |
| **GPTQ safetensors** | Direct | Direct | Convert required | Not supported | Direct | Direct (AutoGPTQ) |
| **GGUF** | Direct (v0.5.3+) | Not supported | Not supported | Direct | Not supported | Not supported |
| **EXL2** | Not supported | Not supported | Not supported | Not supported | Direct | Not supported |
| **TRT Engine** | Not supported | Not supported | Direct | Not supported | Not supported | Not supported |
| **FP8 safetensors** | Direct | Direct | Convert required | Not supported | Not supported | Partial |
| **BitsAndBytes** | Direct | Not supported | Not supported | Not supported | Not supported | Direct |
| **Marlin** | Direct | Direct | Not supported | Not supported | Not supported | Not supported |

**Conversion Paths** (when you have format A but need format B):
- HF --> GGUF: `python convert_hf_to_gguf.py` (in llama.cpp repo)
- HF --> EXL2: `python convert.py` (in ExLlamaV2 repo, requires calibration data)
- HF --> AWQ: Use AutoAWQ library with calibration data
- HF --> GPTQ: Use AutoGPTQ library with calibration data
- HF --> TRT Engine: `convert_checkpoint.py` + `trtllm-build` (in TRT-LLM repo)
- HF --> FP8: Use llm-compressor, AutoFP8, or TRT-LLM quantization scripts
- AWQ/GPTQ --> Marlin: Automatic conversion in vLLM/SGLang with `--quantization marlin`
- GPTQ --> EXL2: Direct load supported in ExLlamaV2, or re-quantize for better quality

---

## 7. Performance Tier Reference

Rough performance expectations for common configurations. All numbers are relative
to FP16 baseline on the same hardware.

### Single GPU Throughput (tokens/sec, decode, batch=1)

| Configuration | A100 80GB | H100 80GB | RTX 4090 24GB | RTX 3090 24GB | M2 Ultra 192GB |
|--------------|-----------|-----------|---------------|---------------|----------------|
| **vLLM FP16** | 1.0x baseline | 1.5x | 0.7x | 0.5x | N/A |
| **vLLM AWQ INT4** | 1.6x | 2.2x | 1.3x | 0.9x | N/A |
| **vLLM FP8** | N/A (Ampere) | 2.0x | 1.1x | N/A | N/A |
| **SGLang AWQ INT4** | 1.7x | 2.3x | 1.4x | N/A (sm_80+) | N/A |
| **TRT-LLM FP8** | N/A (Ampere) | 2.5x | N/A | N/A | N/A |
| **TRT-LLM AWQ INT4** | 1.8x | 2.4x | N/A | N/A | N/A |
| **llama.cpp Q4_K_M** | 1.4x | 1.8x | 1.2x | 0.8x | 0.4x |
| **ExLlamaV2 EXL2 4bpw** | 1.7x | 2.0x | 1.5x | 1.0x | N/A |

*Values are approximate and model-dependent. Larger models benefit more from quantization.
"N/A" means the combination is not supported or not applicable.*

### Multi-User Throughput Ranking (high concurrency)

For serving workloads with many concurrent users, frameworks rank approximately:

1. **TensorRT-LLM** -- Highest throughput on Hopper, but requires engine builds
2. **SGLang** -- Best for workloads with prefix sharing (chat, few-shot)
3. **vLLM** -- Most versatile, strong continuous batching
4. **llama.cpp server** -- Acceptable for low concurrency (1-8 users)
5. **ExLlamaV2 / TabbyAPI** -- Single-user or very low concurrency only
6. **HF Transformers** -- Not suitable for production serving

---

## 8. Troubleshooting Common Compatibility Issues

### "Model loads but inference is garbage / wrong outputs"

- **Cause**: Quantization method mismatch. Loading an AWQ model with `--quantization gptq` or vice versa.
- **Fix**: Check model card for quantization method. Use matching `--quantization` flag.

### "CUDA out of memory during model load"

- **Cause**: Model too large for GPU VRAM even with quantization.
- **Fix**: Use a smaller quantization (Q4 instead of Q8), reduce `--max-model-len`, reduce `--gpu-memory-utilization`, or use tensor parallelism across multiple GPUs.

### "FlashAttention not available"

- **Cause**: flash-attn package not installed, or GPU architecture too old (pre-Ampere).
- **Fix**: `pip install flash-attn --no-build-isolation`. Requires sm_80+. On older GPUs, use `--attention-backend xformers` (vLLM) or SDPA (Transformers).

### "Marlin kernel not found"

- **Cause**: Marlin requires specific model format. Not all AWQ/GPTQ models auto-convert.
- **Fix**: Use `--quantization awq` or `--quantization gptq` instead. Marlin auto-conversion works for AWQ models with group_size=128.

### "FP8 not supported on this GPU"

- **Cause**: FP8 requires Hopper (sm_90) or Ada (sm_89). Ampere and older do not have FP8 hardware.
- **Fix**: Use AWQ INT4 or GPTQ INT4 on Ampere. Use INT8 W8A8 (SmoothQuant) as a middle ground.

### "GGUF model extremely slow in vLLM"

- **Cause**: vLLM dequantizes GGUF to FP16 at load time; you get no quantization speedup.
- **Fix**: Use llama.cpp for GGUF models, or convert to AWQ/GPTQ for vLLM.

### "SGLang fails on Turing GPU (RTX 2080)"

- **Cause**: SGLang requires Ampere (sm_80) or newer due to FlashInfer dependency.
- **Fix**: Use vLLM (supports Volta sm_70+) or llama.cpp (supports Pascal sm_60+).

### "TensorRT-LLM engine won't load on different GPU"

- **Cause**: TRT engines are architecture-specific. An H100 engine will not run on A100.
- **Fix**: Rebuild the engine on the target GPU architecture. Always build on the same GPU type you deploy to.

### "BitsAndBytes fails on Windows"

- **Cause**: BitsAndBytes has limited Windows support.
- **Fix**: Use WSL2 on Windows, or switch to AWQ/GPTQ which work natively.
