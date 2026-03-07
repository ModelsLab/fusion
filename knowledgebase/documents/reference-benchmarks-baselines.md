---
id: reference-benchmarks-baselines
kind: document
title: "Reference Benchmarks: Expected Performance Baselines"
category: benchmarks
summary: Expected performance numbers for common model×GPU×precision combinations so the agent can evaluate optimization quality
tags:
  - benchmarks
  - baselines
  - performance
  - reference
gpu_families:
  - Ampere
  - Ada
  - Hopper
workloads:
  - prefill
  - decode
  - serving
---

# Reference Benchmarks: Expected Performance Baselines

This document provides approximate performance baselines for common model, GPU,
precision, and runtime combinations. These numbers are drawn from community
benchmarks, published results, and empirical testing as of late 2024 / early 2025.

**Important caveats:**

- All numbers are approximate ranges, not exact guarantees.
- Community benchmarks vary +/-20% due to driver versions, cooling, power limits,
  BIOS settings, batch size details, sequence length, KV cache configuration, etc.
- Use these as sanity checks, not as ground truth.
- "tok/s" for decode means output tokens per second at batch size 1 unless noted.
- Tensor parallelism (TP) is noted where applicable.

---

## 1. LLM Decode Performance (tokens/sec, batch=1)

Single-stream autoregressive decode. This workload is **memory-bandwidth bound**
because each token generation reads the full model weights once.

| Model | GPU | Precision | Runtime | tok/s | Source/Date |
|-------|-----|-----------|---------|-------|-------------|
| LLaMA 3.1 8B | RTX 4090 | FP16 | vLLM | ~95-110 | community benchmarks 2024 |
| LLaMA 3.1 8B | RTX 4090 | AWQ INT4 | vLLM | ~140-170 | community benchmarks 2024 |
| LLaMA 3.1 8B | RTX 4090 | GPTQ INT4 | vLLM | ~130-160 | community benchmarks 2024 |
| LLaMA 3.1 8B | RTX 4090 | GGUF Q4_K_M | llama.cpp | ~80-100 | community benchmarks 2024 |
| LLaMA 3.1 8B | RTX 4090 | GGUF Q5_K_M | llama.cpp | ~70-90 | community benchmarks 2024 |
| LLaMA 3.1 8B | RTX 4090 | GGUF Q8_0 | llama.cpp | ~50-65 | community benchmarks 2024 |
| LLaMA 3.1 8B | RTX 3090 | FP16 | vLLM | ~55-70 | community benchmarks 2024 |
| LLaMA 3.1 8B | RTX 3090 | AWQ INT4 | vLLM | ~85-110 | community benchmarks 2024 |
| LLaMA 3.1 8B | RTX 3090 | GGUF Q4_K_M | llama.cpp | ~55-70 | community benchmarks 2024 |
| LLaMA 3.1 8B | A100 80GB | BF16 | vLLM | ~110-135 | community benchmarks 2024 |
| LLaMA 3.1 8B | A100 80GB | AWQ INT4 | vLLM | ~160-200 | community benchmarks 2024 |
| LLaMA 3.1 8B | H100 SXM | BF16 | vLLM | ~130-160 | community benchmarks 2024 |
| LLaMA 3.1 8B | H100 SXM | FP8 | vLLM | ~180-220 | community benchmarks 2024 |
| LLaMA 3.1 8B | H100 SXM | AWQ INT4 | vLLM | ~210-260 | community benchmarks 2024 |
| LLaMA 3.1 70B | RTX 4090 | GGUF Q4_K_M | llama.cpp | ~12-18 | community benchmarks 2024 |
| LLaMA 3.1 70B | A100 80GB | BF16 | vLLM | ~18-25 | community benchmarks 2024 |
| LLaMA 3.1 70B | A100 80GB (TP=2) | BF16 | vLLM | ~30-42 | community benchmarks 2024 |
| LLaMA 3.1 70B | H100 SXM | BF16 | vLLM | ~25-35 | community benchmarks 2024 |
| LLaMA 3.1 70B | H100 SXM (TP=2) | FP8 | vLLM | ~40-55 | community benchmarks 2024 |
| LLaMA 3.1 70B | H100 SXM (TP=4) | BF16 | vLLM | ~35-50 | community benchmarks 2024 |
| LLaMA 3.1 70B | H100 SXM (TP=4) | FP8 | vLLM | ~55-75 | community benchmarks 2024 |
| LLaMA 3.1 405B | H100 SXM (TP=8) | FP8 | vLLM | ~15-22 | community benchmarks 2024 |
| Mistral 7B v0.3 | RTX 4090 | FP16 | vLLM | ~100-120 | community benchmarks 2024 |
| Mistral 7B v0.3 | RTX 4090 | AWQ INT4 | vLLM | ~150-180 | community benchmarks 2024 |
| Mistral 7B v0.3 | RTX 4090 | GPTQ INT4 | vLLM | ~140-170 | community benchmarks 2024 |
| Mistral 7B v0.3 | RTX 4090 | GGUF Q4_K_M | llama.cpp | ~85-105 | community benchmarks 2024 |
| Mistral 7B v0.3 | RTX 3090 | FP16 | vLLM | ~60-75 | community benchmarks 2024 |
| Mistral 7B v0.3 | RTX 3090 | AWQ INT4 | vLLM | ~90-115 | community benchmarks 2024 |
| Mistral 7B v0.3 | H100 SXM | BF16 | vLLM | ~140-170 | community benchmarks 2024 |
| Mistral 7B v0.3 | H100 SXM | FP8 | vLLM | ~190-230 | community benchmarks 2024 |
| Mixtral 8x7B | RTX 4090 | AWQ INT4 | vLLM | ~70-90 | community benchmarks 2024 |
| Mixtral 8x7B | RTX 4090 | GPTQ INT4 | vLLM | ~60-80 | community benchmarks 2024 |
| Mixtral 8x7B | A100 80GB | BF16 | vLLM | ~50-70 | community benchmarks 2024 |
| Mixtral 8x7B | H100 SXM | BF16 | vLLM | ~70-95 | community benchmarks 2024 |
| Mixtral 8x7B | H100 SXM | FP8 | vLLM | ~90-120 | community benchmarks 2024 |
| Qwen2.5 7B | RTX 4090 | FP16 | vLLM | ~100-115 | community benchmarks 2024 |
| Qwen2.5 7B | RTX 4090 | AWQ INT4 | vLLM | ~145-175 | community benchmarks 2024 |
| Qwen2.5 7B | RTX 3090 | FP16 | vLLM | ~58-72 | community benchmarks 2024 |
| Qwen2.5 7B | H100 SXM | BF16 | vLLM | ~135-165 | community benchmarks 2024 |
| Qwen2.5 72B | H100 SXM (TP=2) | FP8 | vLLM | ~38-52 | community benchmarks 2024 |
| Qwen2.5 72B | H100 SXM (TP=4) | BF16 | vLLM | ~32-48 | community benchmarks 2024 |
| Phi-3 mini 3.8B | RTX 4090 | FP16 | vLLM | ~170-200 | community benchmarks 2024 |
| Phi-3 mini 3.8B | RTX 4090 | AWQ INT4 | vLLM | ~230-280 | community benchmarks 2024 |
| Phi-3 mini 3.8B | RTX 3090 | FP16 | vLLM | ~100-130 | community benchmarks 2024 |
| Phi-3 mini 3.8B | H100 SXM | BF16 | vLLM | ~220-270 | community benchmarks 2024 |
| Gemma 2 9B | RTX 4090 | FP16 | vLLM | ~80-100 | community benchmarks 2024 |
| Gemma 2 9B | RTX 4090 | AWQ INT4 | vLLM | ~120-150 | community benchmarks 2024 |
| Gemma 2 9B | H100 SXM | BF16 | vLLM | ~110-140 | community benchmarks 2024 |
| Gemma 2 27B | H100 SXM | BF16 | vLLM | ~45-60 | community benchmarks 2024 |
| Gemma 2 27B | H100 SXM | FP8 | vLLM | ~60-80 | community benchmarks 2024 |
| CodeLlama 34B | A100 80GB | BF16 | vLLM | ~28-38 | community benchmarks 2024 |
| CodeLlama 34B | H100 SXM | BF16 | vLLM | ~40-55 | community benchmarks 2024 |
| DeepSeek-V2-Lite 16B | RTX 4090 | FP16 | vLLM | ~65-85 | community benchmarks 2024 |

---

## 2. LLM Prefill Performance (tokens/sec)

Prefill (prompt processing) is **compute-bound** for long prompts. Performance
scales with TFLOPS, not memory bandwidth. Short prompts (< ~128 tokens) can still
be memory-bound.

| Model | GPU | Precision | Prompt Length | tok/s | Notes |
|-------|-----|-----------|--------------|-------|-------|
| LLaMA 3.1 8B | RTX 4090 | FP16 | 128 | ~6,000-8,000 | partially memory-bound |
| LLaMA 3.1 8B | RTX 4090 | FP16 | 512 | ~4,500-6,000 | compute-bound |
| LLaMA 3.1 8B | RTX 4090 | FP16 | 2048 | ~3,000-4,200 | compute-bound |
| LLaMA 3.1 8B | RTX 4090 | FP16 | 4096 | ~2,200-3,200 | compute-bound |
| LLaMA 3.1 8B | RTX 4090 | AWQ INT4 | 512 | ~5,500-7,500 | less compute per token |
| LLaMA 3.1 8B | RTX 4090 | AWQ INT4 | 2048 | ~4,000-5,500 | |
| LLaMA 3.1 8B | RTX 3090 | FP16 | 512 | ~2,500-3,500 | |
| LLaMA 3.1 8B | RTX 3090 | FP16 | 2048 | ~1,800-2,600 | |
| LLaMA 3.1 8B | A100 80GB | BF16 | 512 | ~8,000-11,000 | |
| LLaMA 3.1 8B | A100 80GB | BF16 | 2048 | ~5,500-7,500 | |
| LLaMA 3.1 8B | A100 80GB | BF16 | 4096 | ~3,800-5,500 | |
| LLaMA 3.1 8B | H100 SXM | BF16 | 512 | ~14,000-18,000 | |
| LLaMA 3.1 8B | H100 SXM | BF16 | 2048 | ~9,000-13,000 | |
| LLaMA 3.1 8B | H100 SXM | BF16 | 4096 | ~6,500-9,500 | |
| LLaMA 3.1 8B | H100 SXM | FP8 | 2048 | ~13,000-18,000 | |
| LLaMA 3.1 70B | H100 SXM (TP=2) | BF16 | 512 | ~2,500-3,500 | |
| LLaMA 3.1 70B | H100 SXM (TP=2) | BF16 | 2048 | ~1,800-2,600 | |
| LLaMA 3.1 70B | H100 SXM (TP=4) | BF16 | 2048 | ~3,200-4,500 | |
| LLaMA 3.1 70B | H100 SXM (TP=4) | FP8 | 2048 | ~5,000-7,000 | |
| Mistral 7B v0.3 | RTX 4090 | FP16 | 512 | ~5,000-6,500 | |
| Mistral 7B v0.3 | RTX 4090 | FP16 | 2048 | ~3,500-4,800 | |
| Mistral 7B v0.3 | H100 SXM | BF16 | 2048 | ~10,000-14,000 | |
| Qwen2.5 7B | RTX 4090 | FP16 | 512 | ~5,000-6,500 | |
| Qwen2.5 7B | H100 SXM | BF16 | 2048 | ~9,500-13,000 | |
| Phi-3 mini 3.8B | RTX 4090 | FP16 | 512 | ~8,000-11,000 | smaller model, faster |
| Phi-3 mini 3.8B | RTX 4090 | FP16 | 2048 | ~5,500-7,500 | |

---

## 3. LLM Serving Throughput (requests/sec at target latency)

Batched serving with continuous batching. Numbers assume typical chat workloads
(~256 input tokens, ~128 output tokens) unless noted otherwise.

| Model | GPU | Precision | Concurrency | Target P99 Latency | req/s | Runtime |
|-------|-----|-----------|-------------|---------------------|-------|---------|
| LLaMA 3.1 8B | RTX 4090 | FP16 | 8 | <2s TTFT | ~3-5 | vLLM |
| LLaMA 3.1 8B | RTX 4090 | AWQ INT4 | 16 | <2s TTFT | ~6-10 | vLLM |
| LLaMA 3.1 8B | A100 80GB | BF16 | 32 | <2s TTFT | ~8-14 | vLLM |
| LLaMA 3.1 8B | H100 SXM | BF16 | 32 | <2s TTFT | ~12-20 | vLLM |
| LLaMA 3.1 8B | H100 SXM | FP8 | 64 | <2s TTFT | ~18-28 | vLLM |
| LLaMA 3.1 8B | H100 SXM | FP8 | 128 | <5s TTFT | ~25-40 | vLLM |
| LLaMA 3.1 70B | H100 SXM (TP=4) | BF16 | 16 | <3s TTFT | ~2-4 | vLLM |
| LLaMA 3.1 70B | H100 SXM (TP=4) | FP8 | 32 | <3s TTFT | ~4-7 | vLLM |
| LLaMA 3.1 70B | H100 SXM (TP=8) | FP8 | 64 | <3s TTFT | ~6-10 | vLLM |
| Mistral 7B v0.3 | RTX 4090 | FP16 | 8 | <2s TTFT | ~4-6 | vLLM |
| Mistral 7B v0.3 | RTX 4090 | AWQ INT4 | 16 | <2s TTFT | ~7-11 | vLLM |
| Mistral 7B v0.3 | H100 SXM | BF16 | 32 | <2s TTFT | ~14-22 | vLLM |
| Mixtral 8x7B | H100 SXM | BF16 | 16 | <3s TTFT | ~5-9 | vLLM |
| Qwen2.5 7B | RTX 4090 | FP16 | 8 | <2s TTFT | ~3-5 | vLLM |
| Qwen2.5 7B | H100 SXM | BF16 | 32 | <2s TTFT | ~12-18 | vLLM |
| Phi-3 mini 3.8B | RTX 4090 | FP16 | 16 | <1s TTFT | ~8-14 | vLLM |
| Phi-3 mini 3.8B | H100 SXM | BF16 | 64 | <1s TTFT | ~25-40 | vLLM |

Notes on serving throughput:
- TTFT = Time To First Token (prefill latency).
- req/s depends heavily on input/output length distribution.
- Continuous batching (vLLM, TGI, TensorRT-LLM) is essential for good throughput.
- These numbers assume PagedAttention or equivalent KV cache management.
- Higher concurrency trades latency for throughput.

---

## 4. Diffusion Model Performance

Image generation latency for common diffusion models. Times are wall-clock
end-to-end (includes VAE decode).

| Model | GPU | Precision | Steps | Resolution | Time (sec) | Notes |
|-------|-----|-----------|-------|------------|------------|-------|
| SDXL | RTX 4090 | FP16 | 50 | 1024x1024 | ~3.5-4.5 | default scheduler |
| SDXL | RTX 4090 | FP16 | 25 | 1024x1024 | ~1.8-2.5 | DPM++ 2M |
| SDXL | RTX 4090 | FP16+LCM | 4 | 1024x1024 | ~0.4-0.6 | LCM-LoRA distilled |
| SDXL | RTX 4090 | FP16+torch.compile | 50 | 1024x1024 | ~2.5-3.2 | compiled UNet |
| SDXL | RTX 3090 | FP16 | 50 | 1024x1024 | ~5.5-7.0 | |
| SDXL | A100 80GB | FP16 | 50 | 1024x1024 | ~4.0-5.0 | |
| SDXL | H100 SXM | BF16 | 50 | 1024x1024 | ~2.0-2.8 | |
| SDXL | H100 SXM | BF16+torch.compile | 50 | 1024x1024 | ~1.5-2.0 | |
| SD 1.5 | RTX 4090 | FP16 | 50 | 512x512 | ~1.5-2.0 | |
| SD 1.5 | RTX 4090 | FP16 | 25 | 512x512 | ~0.8-1.1 | |
| SD 1.5 | RTX 3090 | FP16 | 50 | 512x512 | ~2.5-3.5 | |
| Flux.1-dev | RTX 4090 | FP16 | 50 | 1024x1024 | ~11-15 | large transformer model |
| Flux.1-dev | RTX 4090 | FP16 | 28 | 1024x1024 | ~6-9 | recommended steps |
| Flux.1-dev | RTX 4090 | FP16+torch.compile | 28 | 1024x1024 | ~4.5-6.5 | |
| Flux.1-dev | H100 SXM | BF16 | 28 | 1024x1024 | ~3-5 | |
| Flux.1-schnell | RTX 4090 | FP16 | 4 | 1024x1024 | ~1.0-1.5 | distilled, few steps |
| Flux.1-schnell | RTX 4090 | FP16+torch.compile | 4 | 1024x1024 | ~0.7-1.0 | |
| Flux.1-schnell | H100 SXM | BF16 | 4 | 1024x1024 | ~0.5-0.8 | |
| PixArt-Sigma | RTX 4090 | FP16 | 20 | 1024x1024 | ~2.0-3.0 | DiT-based |
| PixArt-Sigma | H100 SXM | BF16 | 20 | 1024x1024 | ~1.0-1.5 | |
| SDXL Turbo | RTX 4090 | FP16 | 1 | 512x512 | ~0.15-0.25 | single-step |
| SDXL Turbo | RTX 4090 | FP16 | 4 | 512x512 | ~0.4-0.6 | |

Notes on diffusion performance:
- `torch.compile` typically gives 20-40% speedup on diffusion models.
- LCM/Turbo/Lightning distillation reduces steps dramatically (50 -> 4-8).
- FP8 quantization of UNet/DiT can give 15-30% speedup on Ada/Hopper/Blackwell GPUs.
- INT8 weight-only quantization gives ~10-20% speedup with minimal quality loss.
- Batch size > 1 improves throughput but not per-image latency much.

---

## 5. Audio Model Performance

| Model | GPU | Precision | Audio Length | RTF | Processing Time | Notes |
|-------|-----|-----------|-------------|-----|-----------------|-------|
| Whisper large-v3 | RTX 4090 | FP16 | 30s | ~0.02 | ~0.6s | transcription |
| Whisper large-v3 | RTX 4090 | FP16 | 5min | ~0.02-0.03 | ~6-9s | |
| Whisper large-v3 | RTX 4090 | INT8 (CTranslate2) | 30s | ~0.01 | ~0.3s | faster-whisper |
| Whisper large-v3 | RTX 4090 | INT8 (CTranslate2) | 5min | ~0.01-0.02 | ~3-6s | faster-whisper |
| Whisper large-v3 | RTX 3090 | FP16 | 30s | ~0.03-0.04 | ~1.0-1.2s | |
| Whisper large-v3 | A100 80GB | FP16 | 30s | ~0.02 | ~0.6s | |
| Whisper large-v3 | H100 SXM | BF16 | 30s | ~0.01 | ~0.3s | |
| Whisper medium | RTX 4090 | FP16 | 30s | ~0.01 | ~0.3s | |
| Whisper medium | RTX 3090 | FP16 | 30s | ~0.02 | ~0.5s | |
| Whisper small | RTX 4090 | FP16 | 30s | ~0.005 | ~0.15s | |
| Chatterbox TTS | RTX 3090 | FP32 | 10s gen | N/A | ~8-12s | baseline unoptimized |
| Chatterbox TTS | RTX 3090 | FP16 | 10s gen | N/A | ~5-8s | AMP enabled |
| Chatterbox TTS | RTX 4090 | FP16 | 10s gen | N/A | ~3-5s | AMP enabled |
| XTTS v2 | RTX 4090 | FP16 | 10s gen | N/A | ~2-4s | |
| XTTS v2 | RTX 3090 | FP16 | 10s gen | N/A | ~3-6s | |
| Bark | RTX 4090 | FP16 | 10s gen | N/A | ~6-10s | |
| Encodec | RTX 4090 | FP32 | 30s | ~0.003 | ~0.1s | audio codec |
| Wav2Vec2 large | RTX 4090 | FP16 | 30s | ~0.005 | ~0.15s | feature extraction |

Notes on audio:
- RTF = Real-Time Factor. RTF < 1.0 means faster than real-time.
- RTF < 0.05 is considered excellent for transcription.
- TTS models vary wildly; autoregressive TTS is much slower than non-autoregressive.
- faster-whisper (CTranslate2 INT8) is typically 2-4x faster than vanilla Whisper.
- Whisper batched inference can process multiple segments in parallel for higher throughput.

---

## 6. Kernel-Level Benchmarks (for the agent writing custom kernels)

These are useful when evaluating whether a custom Triton or CUDA kernel is
performing well. Numbers represent best-known achieved performance.

### 6.1 GEMM (Matrix Multiply)

| Shape (M x N x K) | GPU | Precision | Best Known TFLOPS | Library | % of Peak |
|--------------------|-----|-----------|-------------------|---------|-----------|
| 4096 x 4096 x 4096 | H100 SXM | BF16 | ~760 | cuBLAS | ~77% |
| 4096 x 4096 x 4096 | H100 SXM | FP8 | ~1400 | cuBLAS | ~71% |
| 4096 x 4096 x 4096 | A100 80GB | BF16 | ~240 | cuBLAS | ~77% |
| 4096 x 4096 x 4096 | RTX 4090 | FP16 | ~120 | cuBLAS | ~73% |
| 8192 x 8192 x 8192 | H100 SXM | BF16 | ~850 | cuBLAS | ~86% |
| 8192 x 8192 x 8192 | H100 SXM | FP8 | ~1600 | cuBLAS | ~81% |
| 8192 x 8192 x 8192 | A100 80GB | BF16 | ~270 | cuBLAS | ~87% |
| 1 x 4096 x 4096 | H100 SXM | BF16 | ~12 (bandwidth-bound) | cuBLAS | memory-bound |
| 1 x 4096 x 4096 | H100 SXM | BF16 | N/A | Triton | can beat cuBLAS |
| 1 x 4096 x 11008 | H100 SXM | BF16 | N/A | Triton GEMV | memory-bound |
| 16 x 4096 x 4096 | H100 SXM | BF16 | ~45 | cuBLAS | partially memory-bound |
| 16 x 4096 x 4096 | H100 SXM | BF16 | ~50 | Triton | can beat cuBLAS here |

### 6.2 Attention

| Config | GPU | Precision | Best Known TFLOPS | Library | Notes |
|--------|-----|-----------|-------------------|---------|-------|
| B=1, H=32, S=4096, D=128 | H100 SXM | BF16 | ~280 | FlashAttention-2 | ~28% peak (memory-bound at B=1) |
| B=1, H=32, S=4096, D=128 | A100 80GB | BF16 | ~100 | FlashAttention-2 | ~32% peak (memory-bound) |
| B=8, H=32, S=4096, D=128 | H100 SXM | BF16 | ~650 | FlashAttention-2 | ~66% peak |
| B=16, H=32, S=2048, D=128 | H100 SXM | BF16 | ~750 | FlashAttention-2 | ~76% peak |
| B=32, H=32, S=2048, D=128 | H100 SXM | BF16 | ~800 | FlashAttention-2 | ~81% peak |
| B=1, H=32, S=4096, D=128 | RTX 4090 | FP16 | ~45 | FlashAttention-2 | memory-bound |
| B=8, H=32, S=4096, D=128 | RTX 4090 | FP16 | ~100 | FlashAttention-2 | ~61% peak |
| B=1, H=8, S=4096, D=128 (GQA) | H100 SXM | BF16 | ~70 | FlashAttention-2 | GQA is more memory-bound |
| FlashAttention-3 | H100 SXM | FP8 | ~1200 | FA3 | with FP8, large batch |

### 6.3 Element-wise / Memory-Bound Kernels

| Operator | Shape | GPU | Precision | Achieved BW | Library | % of Peak HBM BW |
|----------|-------|-----|-----------|-------------|---------|-------------------|
| RMSNorm | B=1, S=1, D=4096 | H100 SXM | BF16 | ~2.8 TB/s | Triton | ~84% |
| RMSNorm | B=1, S=2048, D=4096 | H100 SXM | BF16 | ~3.0 TB/s | Triton | ~90% |
| RMSNorm | B=1, S=1, D=4096 | A100 80GB | BF16 | ~1.7 TB/s | Triton | ~83% |
| RMSNorm | B=1, S=1, D=4096 | RTX 4090 | FP16 | ~0.85 TB/s | Triton | ~84% |
| LayerNorm | B=1, S=2048, D=4096 | H100 SXM | BF16 | ~2.9 TB/s | Triton | ~87% |
| Softmax | B=32, S=4096 | H100 SXM | BF16 | ~2.5 TB/s | Triton | ~75% |
| Softmax | B=32, S=4096 | A100 80GB | BF16 | ~1.6 TB/s | Triton | ~78% |
| RoPE | B=1, H=32, S=4096, D=128 | H100 SXM | BF16 | ~2.4 TB/s | Triton | ~72% |
| RoPE | B=1, H=32, S=4096, D=128 | A100 80GB | BF16 | ~1.5 TB/s | Triton | ~74% |
| SiLU+Mul (gated) | B=1, S=1, D=11008 | H100 SXM | BF16 | ~2.7 TB/s | Triton (fused) | ~81% |
| Residual Add | B=1, S=2048, D=4096 | H100 SXM | BF16 | ~3.1 TB/s | Triton | ~93% |
| Quantize (BF16->INT8) | B=1, S=2048, D=4096 | H100 SXM | mixed | ~2.2 TB/s | Triton | ~66% |

### 6.4 Fused Kernels

| Fused Op | Shape | GPU | Precision | Speedup vs Unfused | Notes |
|----------|-------|-----|-----------|-------------------|-------|
| RMSNorm + Residual Add | B=1, S=2048, D=4096 | H100 SXM | BF16 | ~1.5-1.8x | saves one global memory round-trip |
| SiLU + Mul + Down Proj quantize | B=1, S=1, D=11008 | H100 SXM | BF16 | ~1.3-1.5x | fuse activation with quantization |
| Fused QKV projection | B=1, S=1, D=4096 -> 3x4096 | H100 SXM | BF16 | ~1.1-1.2x | minor due to GEMM dominance |

---

## 7. GPU Theoretical Peaks (Quick Reference)

### Compute

| GPU | Arch | FP32 TFLOPS | FP16/BF16 TFLOPS | FP16/BF16 (sparse) | FP8 TFLOPS | FP8 (sparse) | INT8 TOPS | INT8 (sparse) |
|-----|------|-------------|-------------------|---------------------|------------|--------------|-----------|---------------|
| RTX 3090 | Ampere | 35.6 | 35.6 | 71 | N/A | N/A | 71 | 142 |
| RTX 3090 Ti | Ampere | 40.0 | 40.0 | 80 | N/A | N/A | 80 | 160 |
| RTX 4090 | Ada | 82.6 | 82.6 | 165 | 165 | 330 | 330 | 660 |
| RTX 4080 | Ada | 48.7 | 48.7 | 97 | 97 | 194 | 194 | 388 |
| A100 40GB | Ampere | 19.5 | 156 | 312 | N/A | N/A | 312 | 624 |
| A100 80GB | Ampere | 19.5 | 156 | 312 | N/A | N/A | 312 | 624 |
| A10G | Ampere | 31.2 | 31.2 | 62.5 | N/A | N/A | 62.5 | 125 |
| L4 | Ada | 30.3 | 30.3 | 60.6 | 60.6 | 121 | 121 | 242 |
| L40S | Ada | 91.6 | 91.6 | 183 | 183 | 366 | 366 | 733 |
| H100 SXM | Hopper | 66.9 | 494.5 | 989 | 989 | 1979 | 989 | 1979 |
| H100 PCIe | Hopper | 51.2 | 378 | 756 | 756 | 1513 | 756 | 1513 |
| H200 SXM | Hopper | 66.9 | 494.5 | 989 | 989 | 1979 | 989 | 1979 |

### Memory Bandwidth and Capacity

| GPU | VRAM | HBM/GDDR Type | Memory BW (TB/s) | L2 Cache | TDP (W) |
|-----|------|---------------|-------------------|----------|---------|
| RTX 3090 | 24 GB | GDDR6X | 0.936 | 6 MB | 350 |
| RTX 3090 Ti | 24 GB | GDDR6X | 1.008 | 6 MB | 450 |
| RTX 4090 | 24 GB | GDDR6X | 1.008 | 72 MB | 450 |
| RTX 4080 | 16 GB | GDDR6X | 0.717 | 64 MB | 320 |
| A100 40GB | 40 GB | HBM2e | 1.555 | 40 MB | 250 |
| A100 80GB | 80 GB | HBM2e | 2.039 | 40 MB | 300 |
| A10G | 24 GB | GDDR6 | 0.600 | 6 MB | 150 |
| L4 | 24 GB | GDDR6 | 0.300 | 48 MB | 72 |
| L40S | 48 GB | GDDR6 | 0.864 | 96 MB | 350 |
| H100 SXM | 80 GB | HBM3 | 3.35 | 50 MB | 700 |
| H100 PCIe | 80 GB | HBM3 | 2.00 | 50 MB | 350 |
| H200 SXM | 141 GB | HBM3e | 4.80 | 50 MB | 700 |

### Key Ratios for Performance Analysis

| GPU | BF16 TFLOPS / HBM BW (ops/byte) | Implication |
|-----|----------------------------------|-------------|
| RTX 3090 | 35.6 / 0.936 = 38 | moderate arithmetic intensity threshold |
| RTX 4090 | 82.6 / 1.008 = 82 | high arithmetic intensity threshold |
| A100 80GB | 156 / 2.039 = 76 | high arithmetic intensity threshold |
| H100 SXM | 494.5 / 3.35 = 148 | very high arithmetic intensity threshold |
| H200 SXM | 494.5 / 4.80 = 103 | lower threshold than H100 (more BW) |

These ops/byte ratios determine the crossover point between memory-bound and
compute-bound regimes. Operations with arithmetic intensity below this threshold
are memory-bound; above it they are compute-bound.

---

## 8. How to Interpret These Numbers

### Memory-Bound vs Compute-Bound

- **Decode (autoregressive generation)** is memory-bound: each output token reads
  the full model weights once. Performance scales with memory bandwidth, not FLOPS.
  - This is why RTX 4090 (1.0 TB/s) and A100 80GB (2.0 TB/s) have similar decode
    speed despite A100 having 2x the BF16 TFLOPS.
  - This is why quantization (INT4, INT8) helps decode so much: fewer bytes to read.

- **Prefill (prompt processing)** is compute-bound for sequences longer than ~128
  tokens. Performance scales with TFLOPS.
  - This is why H100 (494 BF16 TFLOPS) is much faster at prefill than RTX 4090
    (82 BF16 TFLOPS).
  - Quantization helps prefill less because compute is the bottleneck, not memory.

### Quantization Impact Rules of Thumb

- FP16/BF16 -> INT4 (AWQ/GPTQ): ~1.4-1.7x faster decode, ~1.2-1.4x faster prefill
- FP16/BF16 -> INT8: ~1.3-1.5x faster decode, ~1.1-1.3x faster prefill
- BF16 -> FP8 (Ada/Hopper/Blackwell): ~1.4-1.6x faster decode, ~1.3-1.5x faster prefill
- GGUF Q4_K_M vs AWQ INT4: GGUF is typically 10-20% slower due to dequantization
  overhead, but has better quality-preserving mixed quantization

### What "Good" Looks Like for Custom Kernels

- **Memory-bound kernels** (RMSNorm, LayerNorm, Softmax, RoPE, residual add):
  - 70-85% of theoretical HBM bandwidth = excellent
  - 50-70% = acceptable, room for improvement
  - <50% = something is wrong (bad memory access pattern, unnecessary recomputation)

- **Compute-bound kernels** (GEMM, attention with large batch):
  - 70-85% of peak TFLOPS = excellent (this is what cuBLAS typically achieves)
  - 50-70% = good for custom Triton kernels
  - <50% = investigate tile sizes, memory layout, occupancy

- **Fused kernels**: The goal is to reduce global memory round-trips. A fused
  RMSNorm+residual should be ~1.5-2x faster than the unfused version. If the
  speedup is <1.2x, the fusion is not effective.

### Community Benchmark Variance

- Numbers from different sources can vary +/-20% due to:
  - Different driver versions and CUDA versions
  - Power limit and thermal throttling differences
  - Different vLLM / llama.cpp versions and configurations
  - Different prompt lengths and generation lengths
  - KV cache quantization settings
  - Chunked prefill settings
  - Speculative decoding configurations

---

## 9. Performance Sanity Checks

Rules of thumb the agent should use to validate optimization results:

### LLM Decode Sanity Checks

- LLaMA 8B BF16 decode on RTX 4090: expect ~100 tok/s
  - AWQ INT4 should reach ~150+ tok/s
  - If you see < 80 tok/s on FP16, something is wrong (check GPU utilization, batch size, KV cache)
  - If AWQ is slower than FP16, the quantization or dequantization path is broken

- LLaMA 8B BF16 decode on H100: expect ~150 tok/s
  - FP8 should reach ~200+ tok/s
  - If you see < 100 tok/s, check tensor parallelism config or memory layout

- LLaMA 70B on single H100: expect ~25-35 tok/s at BF16
  - If you see > 50 tok/s at BF16 on a single GPU, the number is suspicious
  - TP=2 should roughly 1.5-1.8x the single-GPU number (not 2x due to communication)
  - TP=4 should roughly 2.5-3.5x the single-GPU number

- Mistral 7B should be ~5-10% faster than LLaMA 8B at same precision due to
  fewer parameters and GQA efficiency

- Phi-3 mini 3.8B should be ~1.7-2x faster than 7B models due to ~half the parameters

### Prefill Sanity Checks

- LLaMA 8B BF16 prefill on H100 at seq_len=2048: expect ~10,000-13,000 tok/s
  - If you see < 5,000 tok/s, FlashAttention may not be enabled
  - If you see > 20,000 tok/s, verify the measurement methodology

- Prefill tok/s should decrease with sequence length (quadratic attention cost)
  - 4096 tokens should be roughly 0.6-0.7x the speed of 2048 tokens (with FlashAttention)
  - Without FlashAttention, the drop-off is much steeper

### Custom Kernel Sanity Checks

- Custom Triton RMSNorm should achieve > 80% HBM bandwidth
  - If < 60%, check for unnecessary global memory reads/writes
  - Common mistake: reading the input twice (once for mean, once for normalize)

- Custom Triton GEMM beating cuBLAS on large square matrices (4096x4096x4096) is
  VERY hard and usually not worth attempting
  - cuBLAS has years of hand-tuned assembly kernels for these shapes
  - Triton GEMM at 60-70% of cuBLAS on large squares is normal and acceptable

- Custom Triton GEMM beating cuBLAS on thin shapes (batch=1 decode, M=1) IS
  possible and worthwhile
  - cuBLAS is optimized for large matrices, not GEMV-like shapes
  - Triton can win by 10-30% on M=1-16 shapes with careful tuning
  - This is the sweet spot for custom decode kernels

- Custom Triton Softmax should get within 5-10% of the PyTorch built-in for
  standard shapes. If it is > 20% slower, something is wrong.

### Diffusion Sanity Checks

- SDXL on RTX 4090 at 50 steps: expect ~3.5-4.5s
  - With torch.compile: ~2.5-3.2s
  - If > 6s, something is wrong (check if xformers/SDPA is enabled)
  - If < 2s at 50 steps without distillation, the number is suspicious

- Flux.1-dev is ~3-4x slower than SDXL (larger transformer backbone)
  - If Flux is faster than SDXL, something is likely misconfigured

- LCM/Turbo/Lightning should achieve near-equivalent quality at 4-8 steps as
  the base model at 25-50 steps. If quality is significantly worse, check the
  LoRA weights and scheduler configuration.

### Audio Sanity Checks

- Whisper large-v3 on RTX 4090: RTF ~0.02 (50x real-time)
  - faster-whisper (CTranslate2 INT8) should achieve RTF ~0.01 (100x real-time)
  - If RTF > 0.1, the model may be running on CPU or without batched decoding

- TTS models generating 10s of audio:
  - Autoregressive TTS (XTTS, Bark): 2-10s is normal
  - Non-autoregressive TTS: < 1s is possible
  - If an autoregressive TTS claims < 0.5s for 10s audio, verify it is not
    measuring only the vocoder step

### General Red Flags

- Any result claiming > 95% of theoretical peak FLOPS on a real workload is
  almost certainly a measurement error
- Any result claiming INT4 is slower than FP16 for decode likely has a broken
  quantization path or is measuring prefill-dominated workload
- If TP=N gives exactly Nx speedup, the communication overhead is not being
  measured correctly
- If two very different GPUs give identical performance, something is CPU-bound
  or otherwise bottlenecked outside the GPU
