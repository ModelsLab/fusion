---
id: model_optimization_ladders
kind: document
title: Model-Specific Optimization Ladders
category: optimization
summary: Step-by-step optimization sequences for popular LLM models, ordered from easiest/most impactful to hardest/least impactful, with expected speedup at each step.
tags:
  - optimization-ladder
  - llama
  - mistral
  - deepseek
  - qwen
  - phi
  - gemma
source_ids: []
operators:
  - attention
  - matmul
  - general
gpu_families:
  - Ampere
  - Ada
  - Hopper
  - Blackwell
---

# Model-Specific Optimization Ladders

## Optimization Ladder Methodology
Apply optimizations in order. Measure after each step. Stop when target is met.

```
Level 0: Baseline (naive PyTorch)
Level 1: Framework selection (vLLM/SGLang/TRT-LLM)
Level 2: Quantization (AWQ/FP8/GGUF)
Level 3: Kernel optimization (FlashAttention, fused ops)
Level 4: System optimization (CUDA graphs, continuous batching)
Level 5: Architecture-specific (speculative decoding, MLA optimization)
Level 6: Custom kernels (Triton/CUDA for model-specific bottlenecks)
```

## LLaMA 3 Family (8B / 70B / 405B)

### LLaMA 3 8B on RTX 4090 (24GB)
| Step | Optimization | Tokens/s | Memory | Speedup |
|------|-------------|----------|--------|---------|
| 0 | BF16 transformers | ~35 | 16GB | 1.0x |
| 1 | vLLM BF16 | ~55 | 17GB | 1.6x |
| 2 | AWQ INT4 (g128) | ~95 | 6GB | 2.7x |
| 3 | FlashAttention-2 | ~105 | 6GB | 3.0x |
| 4 | CUDA graphs | ~115 | 6.5GB | 3.3x |
| 5 | Speculative (draft=0.5B) | ~150 | 7.5GB | 4.3x |

```bash
# Step 2: Quantize with AutoAWQ
python -m awq.entry --model_path meta-llama/Llama-3-8B \
  --w_bit 4 --q_group_size 128 --version gemm

# Step 1+3+4: vLLM with all optimizations
vllm serve meta-llama/Llama-3-8B-AWQ \
  --quantization awq \
  --dtype float16 \
  --max-model-len 8192 \
  --enable-cuda-graph \
  --gpu-memory-utilization 0.9
```

### LLaMA 3 70B on H100 (80GB)
| Step | Optimization | Tokens/s | Memory | Speedup |
|------|-------------|----------|--------|---------|
| 0 | BF16 transformers | ~15 | 140GB (2xH100) | 1.0x |
| 1 | vLLM BF16 TP=2 | ~25 | 75GB/GPU | 1.7x |
| 2 | FP8 (single H100) | ~40 | 72GB | 2.7x |
| 3 | FP8 + chunked prefill | ~45 | 72GB | 3.0x |
| 4 | CUDA graphs + continuous batch | ~55 | 73GB | 3.7x |
| 5 | Speculative (8B draft) | ~70 | 80GB | 4.7x |

### LLaMA 3 405B (Multi-Node)
```
TP=8 on single DGX H100: ~20 tok/s per request
FP8 quantization: ~30 tok/s (1.5x)
Expert parallelism (if MoE variant): N/A for dense
Pipeline parallelism across 2 nodes: better throughput at high batch
```

## Mistral 7B / Mixtral 8x7B

### Mistral 7B - Sliding Window Attention Optimization
```
Key difference from LLaMA: sliding window attention (W=4096)
- KV cache memory: O(W) instead of O(seq_len)
- For seq_len=32K: 8x less KV memory than full attention
- Enables longer contexts on smaller GPUs

Optimization order:
1. vLLM (handles sliding window natively)
2. AWQ INT4 → fits on 8GB GPU
3. FlashAttention with sliding window mask
4. CUDA graphs (fixed KV window = perfect for graphs)
```

### Mixtral 8x7B - MoE-Specific Optimization
| Step | Optimization | Tokens/s | Memory | Notes |
|------|-------------|----------|--------|-------|
| 0 | BF16 (needs ~90GB) | ~20 | 90GB | 2xA100-40GB minimum |
| 1 | AWQ INT4 | ~45 | 24GB | Fits single RTX 4090 |
| 2 | Expert offloading | ~30 | 12GB | Trade speed for memory |
| 3 | Fused MoE kernel | ~55 | 24GB | vLLM fused_moe |
| 4 | Expert parallelism (2 GPU) | ~70 | 14GB/GPU | Split experts |

```python
# Mixtral critical kernel: fused MoE gate + dispatch + GEMM
# vLLM implementation: vllm/model_executor/layers/fused_moe/
# Key: avoid materializing full (batch, num_experts, hidden) tensor
# Instead: permute tokens, grouped GEMM, unpermute
```

## DeepSeek-V3 (MLA + MoE)

### MLA (Multi-Head Latent Attention) Optimization
```
Standard MHA KV cache: 2 * n_layers * n_heads * d_head * seq_len
DeepSeek MLA KV cache: n_layers * d_compress * seq_len

Example (DeepSeek-V3):
  Standard: 2 * 60 * 128 * 128 * 4096 = 8.05 GB per sequence
  MLA: 60 * 512 * 4096 = 125 MB per sequence (64x reduction!)

MLA kernel requirements:
1. Compressed KV projection: c_kv = W_dkv @ h  (small GEMM)
2. KV decompression: K,V = W_uk @ c_kv, W_uv @ c_kv (fused)
3. Attention on decompressed K,V (standard FlashAttention)
4. OR: absorb decompression into Q projection (faster)
```

### DeepSeek-V3 Optimization Ladder
```
1. MLA-aware serving (SGLang has best support)
2. FP8 quantization (model was trained in FP8)
3. Expert parallelism for 256 experts
4. Fused MoE with auxiliary-loss-free routing
5. Overlapped expert computation + all-to-all communication
6. Multi-token prediction head optimization
```

## Qwen 2.5 Family

### Qwen 2.5 72B
```
Architecture: GQA (8 KV heads for 64 Q heads), SwiGLU, RMSNorm
Similar to LLaMA but with:
- Larger vocabulary (152K vs 128K) → larger embedding/LM head
- Different intermediate size ratios

Optimization priorities:
1. LM head is 72B * 152K * 2 bytes = 21.9 GB in BF16
   → INT8 LM head quantization saves 11GB
2. GQA reduces KV cache: only 8 KV heads
3. Standard AWQ/FP8 quantization works well
```

## Phi-3 / Phi-4 (Small Models)

### Phi-3 Mini (3.8B) - Edge Optimization
| Step | Optimization | Tokens/s (RTX 3060 12GB) | Memory |
|------|-------------|--------------------------|--------|
| 0 | BF16 | ~45 | 7.6GB |
| 1 | INT4 AWQ | ~90 | 2.5GB |
| 2 | GGUF Q4_K_M (llama.cpp) | ~100 | 2.8GB |
| 3 | Flash Attention | +15% | same |
| 4 | Speculative (Phi-1.5 draft) | ~140 | 4GB |

```
Small model optimization differs from large:
- Compute-bound even at batch=1 (small enough to saturate GPU)
- Quantization helps memory more than speed
- Speculative decoding very effective (small draft overhead)
- llama.cpp often faster than vLLM for single-user
```

## Gemma 2 (9B / 27B)

### Gemma 2 Specific: Interleaved Attention
```
Gemma 2 alternates:
- Local sliding window attention (W=4096)
- Global full attention
Every other layer uses local attention

Optimization implications:
1. Local layers: small KV cache, fast attention
2. Global layers: full KV cache, standard optimization
3. Mixed KV cache sizes per layer → custom memory management
4. CUDA graphs: need separate graphs for local vs global layers
```

## Optimization Decision Tree

```
Is model < 3B params?
  YES → llama.cpp GGUF Q4_K_M, speculative decoding
  NO → continue

Does model fit in FP16 on target GPU?
  YES → Start with FP16, add CUDA graphs + FlashAttention
  NO → continue

Does model fit in FP8 on target GPU?
  YES → FP8 quantization (best quality/speed tradeoff)
  NO → continue

Does model fit in INT4 on target GPU?
  YES → AWQ INT4, Marlin kernel for speed
  NO → continue

Multi-GPU available?
  YES → Tensor parallelism (2-8 GPU), then FP8
  NO → Need smaller model or offloading

Is model MoE?
  YES → Expert parallelism + fused MoE kernels
  NO → Standard TP/PP

Is latency critical (real-time)?
  YES → Speculative decoding + CUDA graphs + disaggregated serving
  NO → Maximize throughput with continuous batching

Target: throughput or latency?
  Throughput → Large batch, continuous batching, FP8
  Latency → Small batch, CUDA graphs, speculative decoding
```

## Universal Optimization Checklist
```
[ ] 1. Profile baseline (tokens/s, memory, TTFT, TPOT)
[ ] 2. Select serving framework (vLLM for throughput, SGLang for latency)
[ ] 3. Apply quantization (FP8 > AWQ INT4 > GPTQ INT4)
[ ] 4. Enable FlashAttention (usually default in modern frameworks)
[ ] 5. Enable CUDA graphs (--enforce-eager=false in vLLM)
[ ] 6. Tune batch size and max_num_seqs
[ ] 7. Enable chunked prefill for long contexts
[ ] 8. Consider speculative decoding for latency
[ ] 9. Profile again, identify remaining bottlenecks
[ ] 10. Custom kernels for model-specific operations
```
