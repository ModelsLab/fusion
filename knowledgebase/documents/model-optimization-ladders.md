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

## Practical Optimization Scripts

### Script 1: Measure Baseline Performance
```python
# baseline_benchmark.py — run this FIRST before any optimization
import torch, time, json
from vllm import LLM, SamplingParams

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
PROMPTS = [
    "Explain the theory of relativity in simple terms.",
    "Write a Python function to find prime numbers up to N.",
    "What are the main causes of climate change?",
] * 10  # 30 prompts total

# Load model
llm = LLM(model=MODEL, dtype="float16", gpu_memory_utilization=0.9)
params = SamplingParams(max_tokens=256, temperature=0.7)

# Warmup
_ = llm.generate(PROMPTS[:3], params)

# Benchmark
start = time.perf_counter()
outputs = llm.generate(PROMPTS, params)
elapsed = time.perf_counter() - start

total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
results = {
    "model": MODEL,
    "dtype": "float16",
    "num_prompts": len(PROMPTS),
    "total_output_tokens": total_tokens,
    "total_time_sec": round(elapsed, 2),
    "tokens_per_sec": round(total_tokens / elapsed, 1),
    "avg_latency_per_request_ms": round(elapsed / len(PROMPTS) * 1000, 1),
    "gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
}
print(json.dumps(results, indent=2))
# Save for comparison
with open("baseline_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

### Script 2: Compare Before vs After Optimization
```python
# compare_results.py
import json

with open("baseline_results.json") as f:
    baseline = json.load(f)
with open("optimized_results.json") as f:
    optimized = json.load(f)

print(f"{'Metric':<30} {'Baseline':>12} {'Optimized':>12} {'Speedup':>10}")
print("-" * 66)
for key in ["tokens_per_sec", "avg_latency_per_request_ms", "gpu_memory_gb"]:
    b, o = baseline[key], optimized[key]
    if "memory" in key:
        ratio = f"{b/o:.2f}x less"
    elif "latency" in key:
        ratio = f"{b/o:.2f}x faster"
    else:
        ratio = f"{o/b:.2f}x faster"
    print(f"{key:<30} {b:>12} {o:>12} {ratio:>10}")
```

### Script 3: Quick AWQ Optimization (Most Common Path)
```bash
# Step 1: Measure baseline (FP16)
python baseline_benchmark.py  # save results

# Step 2: Quantize to AWQ INT4 (one-time, ~15 min)
pip install autoawq
python -c "
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
model = AutoAWQForCausalLM.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')
model.quantize(tokenizer, quant_config={'zero_point': True, 'q_group_size': 128, 'w_bit': 4, 'version': 'gemm'})
model.save_quantized('./Llama-3.1-8B-AWQ')
tokenizer.save_pretrained('./Llama-3.1-8B-AWQ')
"

# Step 3: Serve optimized model
vllm serve ./Llama-3.1-8B-AWQ \
  --quantization awq \
  --dtype float16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9

# Step 4: Re-run benchmark with AWQ model, save as optimized_results.json
# Step 5: Compare with compare_results.py

# Expected results (RTX 4090, batch=1 decode):
#   FP16: ~65 tokens/sec, 16 GB VRAM
#   AWQ:  ~260 tokens/sec, 4.5 GB VRAM (4x faster, 3.5x less memory)
```

### Script 4: FP8 Optimization (Hopper GPUs)
```bash
# No quantization step needed! vLLM handles FP8 on-the-fly
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --quantization fp8 \
  --kv-cache-dtype fp8_e4m3 \
  --dtype float16 \
  --max-model-len 8192

# Expected results (H100 SXM, batch=1 decode):
#   FP16: ~208 tokens/sec, 16 GB model
#   FP8:  ~416 tokens/sec, 8 GB model (2x faster, 2x less memory)
#   FP8 + FP8 KV: same speed, allows 2x longer context or larger batch
```

### Script 5: Enable Speculative Decoding
```bash
# vLLM with draft model (for latency-critical applications)
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --speculative-model meta-llama/Llama-3.2-1B-Instruct \
  --num-speculative-tokens 5 \
  --quantization awq \
  --dtype float16

# Expected: 1.5-2.5x latency reduction for decode
# Trade-off: uses more GPU memory (draft model + target model)
# Best when: single-user, latency matters more than throughput
```

### vLLM Flag Quick Reference
```bash
# Performance flags (most impactful first)
--quantization awq          # AWQ INT4 (3-4x decode speedup)
--quantization fp8          # FP8 (2x speedup, Hopper+ only)
--kv-cache-dtype fp8_e4m3   # FP8 KV cache (2x KV memory savings)
--enable-chunked-prefill    # Better long-context handling
--max-num-seqs 256          # Max concurrent sequences (tune for throughput)
--gpu-memory-utilization 0.95  # Use more GPU memory for KV cache
--enforce-eager false       # Enable CUDA graphs (default: enabled)

# Debugging flags
--enforce-eager true        # Disable CUDA graphs (for debugging)
--disable-log-requests      # Reduce log noise in production

# Multi-GPU
--tensor-parallel-size 2    # Split model across 2 GPUs
--pipeline-parallel-size 2  # Pipeline across 2 GPUs (less common)
```
