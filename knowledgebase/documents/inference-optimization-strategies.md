---
id: inference_optimization_strategies
kind: document
title: Inference Optimization Strategies - Complete Playbook
category: optimization
summary: Complete playbook for LLM inference optimization covering prefill, decode, speculative decoding, kernel fusion, operator scheduling, multi-GPU strategies, and cost optimization.
tags:
  - inference
  - optimization
  - speculative-decoding
  - kernel-fusion
  - cuda-graphs
  - continuous-batching
  - disaggregated-serving
  - prefill
  - decode
source_ids:
  - vllm-docs
  - sglang-docs
gpu_families:
  - Ampere
  - Ada
  - Hopper
  - Blackwell
workloads:
  - decode
  - prefill
  - serving
---

# Inference Optimization Strategies - Complete Playbook

## The Inference Optimization Stack

```
Layer 5: System (scheduling, batching, disaggregation)
Layer 4: Graph (operator fusion, CUDA graphs, torch.compile)
Layer 3: Kernel (attention, GEMM, norm, activation kernels)
Layer 2: Data format (quantization, sparsity, KV cache compression)
Layer 1: Hardware (GPU selection, multi-GPU topology, precision support)
```

Optimize from bottom up: hardware selection → data format → kernels → graph → system.

## Prefill Optimization (Compute-Bound)

### Why It's Compute-Bound
For prompt length S, each layer does:
- QKV projection: 2*S*H*3D FLOPs (GEMM, M=S is large)
- Attention: 4*S*S*D FLOPs (quadratic in S for self-attention)
- MLP: 2*S*H*I + 2*S*I*H FLOPs
All GEMMs have large M dimension → compute-bound on GPU.

### Optimization Strategies

1. **Use FP8/FP4 for 2-4x compute throughput**
   - FP8 tensor cores: 2x FLOPS vs FP16 on Ada/Hopper/Blackwell
   - FP4 tensor cores: 4x FLOPS vs FP16 on Blackwell
   - Marginal accuracy loss for prefill computation

2. **torch.compile with max-autotune**
   ```python
   model = torch.compile(model, mode="max-autotune")
   # Generates optimized Triton kernels for all ops
   # Fuses pointwise ops, selects best GEMM config
   # Best for prefill where shapes are known
   ```

3. **Flash Attention for long context**
   - O(S) memory instead of O(S^2)
   - IO-optimal: minimizes HBM reads
   - FlashAttention-3 on Hopper: warp specialization + TMA

4. **Tensor Parallelism**
   - Split large GEMMs across GPUs
   - Linear scaling for prefill (compute-bound)
   - Communication cost: 2 AllReduce per layer

5. **Chunked Prefill**
   - Don't process entire long prompt at once
   - Chunk into 512-2048 token pieces
   - Interleave with decode requests → better latency for decode

### Prefill Latency Estimation
```
Approximate TTFT for LLaMA-70B, FP8, H100:
TTFT ≈ (prompt_tokens * 2 * num_params) / (peak_fp8_flops * utilization)
     ≈ (2048 * 2 * 70e9) / (1979e12 * 0.7)
     ≈ 207 ms

For TP=4: ≈ 207/4 * 1.2 (overhead) ≈ 62 ms
```

## Decode Optimization (Memory-Bound)

### Why It's Memory-Bound
Each decode step generates 1 token:
- Must read ALL model weights: ~70 GB for FP8 70B model
- Must read ALL KV cache: grows with context length
- Only performs 1 row of computation per GEMM (M=1)
- Arithmetic intensity ≈ 1 FLOP/byte → far below ridge point

### Decode Latency Breakdown
```
For LLaMA-70B, FP8, H100, batch=1, ctx=2048:
Weight read time: 70 GB / 3350 GB/s = 20.9 ms (THIS DOMINATES)
KV cache read: ~0.32 GB / 3350 GB/s = 0.1 ms
Compute: negligible (only ~140 GFLOP)
Overhead (launch, sync): ~1-2 ms

Total: ~22-24 ms per token → ~42-45 tokens/sec
```

### Optimization Strategies

1. **Weight Quantization (most impactful for decode)**
   ```
   FP16 (140 GB read per step): ~42 ms/token on H100
   FP8  (70 GB read per step):  ~21 ms/token
   INT4 (35 GB read per step):  ~10.5 ms/token
   ```
   INT4 AWQ gives ~4x decode speedup over FP16 because you read 4x less data.

2. **Batching (second most impactful)**
   ```
   With batch=1:  read 70 GB weights for 1 token  → 1 token/weight-read
   With batch=32: read 70 GB weights for 32 tokens → 32 tokens/weight-read
   With batch=128: read 70 GB for 128 tokens → 128 tokens/weight-read

   Throughput scales nearly linearly with batch size until compute-bound!
   Break-even batch (FP8, H100): ~300 tokens to saturate compute
   ```

3. **CUDA Graphs**
   ```python
   # Problem: kernel launch overhead (~5-10 us per launch * ~200 launches per step = ~1-2 ms)
   # At 20ms per token, that's 5-10% overhead!

   # Solution: capture entire decode step as CUDA graph
   # Launch overhead: single graph launch ~10 us total

   # Requirement: static shapes, no dynamic control flow
   # Works well for decode (fixed batch size, single token)
   ```

4. **KV Cache Quantization**
   - FP8 KV: 2x less memory bandwidth for attention
   - INT4 KV: 4x less (with some accuracy cost)
   - Critical for long context (128K+ tokens)

5. **Speculative Decoding**
   ```
   Standard decode: 1 token per forward pass
   Speculative (k=5): potentially 5+ tokens per forward pass

   Draft model (1.5B) generates 5 candidate tokens: ~3 ms
   Target model (70B) verifies all 5 in one pass: ~22 ms (same as 1 token!)
   If acceptance rate = 80%: ~4 accepted tokens per 25 ms = 160 tok/s (vs 45 tok/s)

   Speedup = avg_accepted_tokens / (1 + draft_overhead/target_time)
   ```

6. **Persistent Kernels**
   - Kernel stays resident on SMs, processes work from a queue
   - Eliminates per-launch overhead
   - Particularly good for small-batch decode

## Speculative Decoding - Complete Guide

### Variants

| Variant | Draft | Speed | Quality |
|---------|-------|-------|---------|
| Draft model | Small LLM (~1B) | Good | Exact (rejection sampling) |
| Self-speculative | Skip layers | Good | Approximate |
| Medusa | Extra heads on target | Fast (no draft) | Near-exact |
| Eagle | Autoregressive head | Very fast | Near-exact |
| Lookahead | Jacobi iteration | Moderate | Exact |

### Draft Model Speculation
```python
# Algorithm:
draft_tokens = []
draft_probs = []
for i in range(k):
    logits = draft_model.forward(context + draft_tokens)
    token = sample(logits)
    draft_tokens.append(token)
    draft_probs.append(softmax(logits))

# Verify all k tokens in single target forward:
target_logits = target_model.forward(context + draft_tokens)  # processes k+1 positions

# Modified rejection sampling:
for i in range(k):
    p = target_logits[i][draft_tokens[i]]  # target prob for draft token
    q = draft_probs[i][draft_tokens[i]]    # draft prob for draft token
    if random() < min(1, p/q):
        accept(draft_tokens[i])
    else:
        # Resample from adjusted distribution: max(0, p - q)
        resample_token = sample(max(0, target_logits[i] - draft_probs[i]))
        accept(resample_token)
        break  # reject rest

# If all k accepted, sample one more from target_logits[k]
```

### Medusa (Multi-Head Speculation)
```
# Add k extra "Medusa heads" to the target model
# Each head predicts token at position +1, +2, ..., +k
# No separate draft model needed!

# Architecture:
# main_model → hidden_state
#   ├── lm_head → token at position t+1 (standard)
#   ├── medusa_head_1 → token at position t+2
#   ├── medusa_head_2 → token at position t+3
#   └── medusa_head_k → token at position t+k+1

# Each Medusa head: 1-2 layer MLP (tiny overhead)
# Generate candidate tree, verify with single forward pass
```

### Eagle (Autoregressive Draft on Target Features)
```
# Key insight: use the target model's hidden states as features
# for an autoregressive draft head (lighter than full draft model)

# Draft head: takes target model hidden state + previous token embedding
# Generates draft tokens autoregressively but cheaply
# ~2x better acceptance rate than Medusa
```

## Kernel Fusion Patterns

### Critical Fusions for LLM Inference

```
1. QKV Projection Fusion:
   Instead of: Q = X @ W_q, K = X @ W_k, V = X @ W_v  (3 GEMM launches)
   Do: QKV = X @ [W_q; W_k; W_v]  (1 GEMM, larger N dimension)
   Speedup: ~1.5-2x (less launch overhead, better utilization)

2. Gate + Up Projection Fusion (SwiGLU):
   Instead of: gate = X @ W_gate, up = X @ W_up  (2 GEMMs)
   Do: gate_up = X @ [W_gate; W_up]  (1 GEMM)
   Then: output = silu(gate) * up  (fused pointwise)

3. RMSNorm + Residual:
   Instead of: x = x + residual; y = rmsnorm(x)  (2 kernels, 2 memory passes)
   Do: y, new_residual = fused_rmsnorm_residual(x, residual, weight)  (1 kernel, 1 pass)
   Speedup: ~2x (memory-bound, saves one full read+write)

4. Attention + RoPE:
   Apply rotary embedding inside the attention kernel
   Saves separate RoPE kernel launch and memory round-trip

5. GEMM + Bias + Activation (Epilogue Fusion):
   output = gelu(X @ W + bias)  →  single GEMM with fused epilogue
   Uses cuBLASLt epilogue or CUTLASS EVT

6. Dequant + GEMM (Quantized Models):
   For INT4/FP4 weights: dequantize in registers during GEMM
   No separate dequantization pass needed
   See: Marlin kernel, AWQ kernels

7. Fused Softmax + Masking:
   Apply causal mask and softmax in one kernel
   FlashAttention does this implicitly

8. Fused Sampling:
   logits → temperature_scale → top_k_filter → softmax → multinomial_sample
   All in one kernel (avoid multiple passes over vocabulary)
```

### Memory Savings from Fusion

For hidden_dim=8192, batch=32 (FP16):
```
Unfused RMSNorm + Residual:
  Read residual: 32 * 8192 * 2 = 512 KB
  Write sum: 512 KB
  Read sum: 512 KB
  Write normalized: 512 KB
  Total: 2 MB of memory traffic

Fused RMSNorm + Residual:
  Read (x + residual): 1024 KB
  Write (normalized + new_residual): 1024 KB
  Total: 2 MB but in ONE pass (1 kernel launch vs 2)
  Actual saving: ~40% faster due to cache effects and 1 fewer launch
```

## CUDA Graphs for Inference

### When to Use
- Decode step with fixed batch size (static shapes)
- Reduces kernel launch overhead from ~1-2ms to ~10us
- Especially valuable when per-token latency is 10-30ms

### Implementation Pattern
```python
# Capture phase
static_input = torch.zeros(batch_size, 1, hidden_dim, device='cuda')
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())

with torch.cuda.stream(s):
    for _ in range(3):  # warmup
        static_output = model.decode_step(static_input)

torch.cuda.current_stream().wait_stream(s)

# Capture the graph
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_output = model.decode_step(static_input)

# Replay phase (actual inference)
static_input.copy_(real_input)  # copy new data into static buffer
g.replay()                       # run entire decode step with one launch
result = static_output.clone()   # copy result out
```

### Limitations
- Static shapes only (can pre-capture graphs for common batch sizes)
- No dynamic control flow (if/else based on data)
- Memory is fixed at capture time
- Need to re-capture if model changes

### torch.compile with reduce-overhead mode
```python
# Automatically uses CUDA graphs + Triton kernel generation
model = torch.compile(model, mode="reduce-overhead")
# Best for decode: fixed shapes, repeated execution
# Automatically handles graph capture and replay
```

## Multi-GPU Inference Strategies

### Tensor Parallelism (TP)
```
Best for: latency-sensitive, few GPUs (2-8)
How: Split weight matrices across GPUs

For Linear(H, N) with TP=4:
  GPU 0: Linear(H, N/4) → partial output
  GPU 1: Linear(H, N/4) → partial output
  GPU 2: Linear(H, N/4) → partial output
  GPU 3: Linear(H, N/4) → partial output
  AllReduce(partial_outputs) → full output

Communication per layer: 2 * hidden_dim * batch * dtype_size * (TP-1)/TP
For LLaMA-70B, FP16, batch=1: 2 * 8192 * 2 * 0.75 ≈ 25 KB per AllReduce
With NVLink (900 GB/s): <1 us per AllReduce
```

### Pipeline Parallelism (PP)
```
Best for: throughput, many GPUs, when TP communication is bottleneck

How: Split layers across GPUs
PP=4 for 80-layer model:
  GPU 0: layers 0-19
  GPU 1: layers 20-39
  GPU 2: layers 40-59
  GPU 3: layers 60-79

Decode: each token passes through all GPUs sequentially
  Latency: same as TP=1 (sequential pipeline)
  Throughput: up to 4x with micro-batching

Prefill: pipeline multiple micro-batches
  GPU 0: [micro-batch 1]
  GPU 1: [micro-batch 0]    GPU 0: [micro-batch 2]
  GPU 2: [micro-batch -1]   GPU 1: [micro-batch 1]    GPU 0: [micro-batch 3]
  (Pipeline fills up, then all GPUs stay busy)
```

### Expert Parallelism (EP) for MoE
```
Best for: MoE models (Mixtral, DeepSeek-V3)

How: Different experts on different GPUs
Mixtral 8x7B with EP=8:
  GPU 0: Expert 0
  GPU 1: Expert 1
  ...
  GPU 7: Expert 7

Routing: AllToAll to send tokens to correct expert GPU
Each GPU processes its expert's tokens
AllToAll to return results

Challenge: load imbalance (popular experts get more tokens)
```

### Choosing Parallelism Strategy

| Model | GPUs | Recommended | Reason |
|-------|------|------------|--------|
| 7B | 1 | None needed | Fits on single GPU |
| 70B | 2 | TP=2 | NVLink, low latency |
| 70B | 4 | TP=4 | NVLink, minimum latency |
| 70B | 8 | TP=4, PP=2 | Beyond NVLink domain |
| 405B | 8 | TP=8 | Maximum TP for 1 node |
| 405B | 16 | TP=8, PP=2 | 2 nodes |
| Mixtral-8x7B | 8 | TP=2, EP=4 | Exploit MoE structure |
| DeepSeek-V3 | 8 | TP=4, EP=2 | Hybrid parallelism |

## Optimization Decision Tree

```
Starting point: measure baseline performance (tokens/sec, latency)

1. Is TTFT too high?
   → Profile prefill: is it compute-bound?
     YES → Use FP8/FP4, TP for more FLOPS, torch.compile max-autotune
     NO (memory-bound somehow) → Check for graph breaks, unnecessary syncs

2. Is per-token latency too high?
   → Profile decode: confirm memory-bound
     → Quantize weights: FP8 → INT4 for 2-4x speedup
     → Enable CUDA graphs: save ~1-2ms per step
     → Increase batch size: better amortization of weight reads
     → Speculative decoding: 2-4x if acceptance rate > 70%

3. Is throughput too low?
   → Increase batch size (most impactful)
   → Enable continuous batching
   → Quantize to fit more concurrent requests
   → Add more GPUs with TP/PP

4. Is latency variance too high (P99 >> P50)?
   → Check for: GC pauses, memory allocation, compilation
   → Enable CUDA graphs (deterministic timing)
   → Disable torch.compile JIT (pre-compile)
   → Chunked prefill to prevent decode starvation

5. Memory issues?
   → See memory optimization guide
   → Quantize weights + KV cache
   → Enable PagedAttention
```

## Practical Optimization Recipes

### Recipe 1: Baseline Measurement Script

Complete script to measure tokens/sec, TTFT, TPOT, peak memory, and GPU utilization against a vLLM server.

```python
#!/usr/bin/env python3
"""
Baseline LLM inference benchmark for vLLM.
Measures: tokens/sec, TTFT, TPOT, peak GPU memory, GPU utilization.
Usage:
    python benchmark_baseline.py --url http://localhost:8000 --model meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import time
import json
import statistics
import subprocess
import requests
from typing import List, Dict

def get_gpu_stats() -> Dict:
    """Get current GPU memory and utilization via nvidia-smi."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
         "--format=csv,nounits,noheader"],
        capture_output=True, text=True
    )
    lines = result.stdout.strip().split("\n")
    parts = lines[0].split(",")
    return {
        "memory_used_mb": int(parts[0].strip()),
        "memory_total_mb": int(parts[1].strip()),
        "gpu_utilization_pct": int(parts[2].strip()),
    }

def measure_single_request(url: str, model: str, prompt: str, max_tokens: int) -> Dict:
    """Send one completion request and measure timing."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    first_token_time = None
    token_times = []
    start = time.perf_counter()

    with requests.post(f"{url}/v1/completions", json=payload, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if line.startswith("data: ") and line.strip() != "data: [DONE]":
                now = time.perf_counter()
                if first_token_time is None:
                    first_token_time = now
                token_times.append(now)

    end = time.perf_counter()
    num_tokens = len(token_times)
    ttft = first_token_time - start if first_token_time else None
    inter_token_latencies = [
        token_times[i] - token_times[i - 1] for i in range(1, len(token_times))
    ]
    tpot = statistics.mean(inter_token_latencies) if inter_token_latencies else None
    total_time = end - start
    tokens_per_sec = num_tokens / total_time if total_time > 0 else 0

    return {
        "num_tokens": num_tokens,
        "total_time_s": total_time,
        "ttft_ms": ttft * 1000 if ttft else None,
        "tpot_ms": tpot * 1000 if tpot else None,
        "tokens_per_sec": tokens_per_sec,
    }

def run_benchmark(url: str, model: str, prompt: str, max_tokens: int,
                  warmup: int = 3, iterations: int = 10):
    """Run full benchmark with warmup and statistical analysis."""
    print(f"=== Baseline Benchmark ===")
    print(f"Model: {model}")
    print(f"Max tokens: {max_tokens}")
    print(f"Warmup: {warmup}, Iterations: {iterations}")
    print()

    # Warmup
    print("Running warmup...")
    for i in range(warmup):
        measure_single_request(url, model, prompt, max_tokens)
        print(f"  warmup {i+1}/{warmup} done")

    # Measure
    print("Running measurements...")
    results = []
    for i in range(iterations):
        gpu_before = get_gpu_stats()
        r = measure_single_request(url, model, prompt, max_tokens)
        gpu_after = get_gpu_stats()
        r["peak_gpu_memory_mb"] = max(gpu_before["memory_used_mb"],
                                       gpu_after["memory_used_mb"])
        r["gpu_utilization_pct"] = gpu_after["gpu_utilization_pct"]
        results.append(r)
        print(f"  iter {i+1}/{iterations}: {r['tokens_per_sec']:.1f} tok/s, "
              f"TTFT={r['ttft_ms']:.1f}ms, TPOT={r['tpot_ms']:.2f}ms")

    # Statistical analysis
    tps_vals = [r["tokens_per_sec"] for r in results]
    ttft_vals = [r["ttft_ms"] for r in results if r["ttft_ms"]]
    tpot_vals = [r["tpot_ms"] for r in results if r["tpot_ms"]]
    mem_vals = [r["peak_gpu_memory_mb"] for r in results]
    util_vals = [r["gpu_utilization_pct"] for r in results]

    print()
    print("=== Results ===")
    print(f"Tokens/sec:      mean={statistics.mean(tps_vals):.1f}, "
          f"std={statistics.stdev(tps_vals):.1f}, "
          f"p50={statistics.median(tps_vals):.1f}")
    print(f"TTFT (ms):       mean={statistics.mean(ttft_vals):.1f}, "
          f"std={statistics.stdev(ttft_vals):.1f}, "
          f"p50={statistics.median(ttft_vals):.1f}")
    print(f"TPOT (ms):       mean={statistics.mean(tpot_vals):.2f}, "
          f"std={statistics.stdev(tpot_vals):.2f}, "
          f"p50={statistics.median(tpot_vals):.2f}")
    print(f"Peak GPU Mem:    max={max(mem_vals)} MB")
    print(f"GPU Utilization: mean={statistics.mean(util_vals):.0f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Explain the theory of general relativity in detail.")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    run_benchmark(args.url, args.model, args.prompt, args.max_tokens,
                  args.warmup, args.iterations)
```

### Recipe 2: AWQ INT4 Optimization (Step-by-Step)

**Step 1: Measure FP16 baseline**

```bash
# Start vLLM with FP16
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dtype float16 \
    --port 8000

# Run benchmark (from Recipe 1)
python benchmark_baseline.py --url http://localhost:8000 \
    --model meta-llama/Llama-3.1-8B-Instruct --max-tokens 256
# Record: baseline_tps, baseline_ttft, baseline_memory
```

**Step 2: Quantize to AWQ INT4 (2 commands)**

```bash
# Install AutoAWQ
pip install autoawq

# Quantize (runs offline, takes 10-30 min depending on model size)
python -c "
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = 'meta-llama/Llama-3.1-8B-Instruct'
quant_path = 'Llama-3.1-8B-Instruct-AWQ-INT4'

model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

quant_config = {'zero_point': True, 'q_group_size': 128, 'w_bit': 4, 'version': 'GEMM'}
model.quantize(tokenizer, quant_config=quant_config)
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
print('Quantization complete.')
"
```

Or use a pre-quantized model from Hugging Face (faster):

```bash
# Many models already have AWQ variants available:
# e.g., hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4
```

**Step 3: Measure AWQ INT4 performance**

```bash
# Start vLLM with the AWQ model
python -m vllm.entrypoints.openai.api_server \
    --model ./Llama-3.1-8B-Instruct-AWQ-INT4 \
    --quantization awq \
    --dtype float16 \
    --port 8000

# Run same benchmark
python benchmark_baseline.py --url http://localhost:8000 \
    --model ./Llama-3.1-8B-Instruct-AWQ-INT4 --max-tokens 256
```

**Expected improvements:**
- Decode tokens/sec: ~2.5-3.5x improvement (memory bandwidth is the bottleneck; 4x fewer bytes read)
- TTFT: ~1.2-1.5x improvement (prefill is compute-bound, less benefit)
- GPU memory: ~3.5x reduction (16 GB FP16 -> ~4.5 GB INT4 for 8B model)
- Quality: negligible degradation for most tasks (AWQ preserves salient weights)

**When to stop vs continue optimizing:**
- Stop if you hit your latency/throughput target
- Stop if quality metrics (eval scores) start dropping
- Continue to Recipe 3 (CUDA graphs) if you need another 10-30% on top
- Continue to Recipe 5 (speculative decoding) if you need 2-3x more on decode

### Recipe 3: Enable CUDA Graphs

**vLLM: CUDA graphs are enabled by default.** To explicitly control them:

```bash
# Enabled (default) - captures decode step as a CUDA graph
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --enforce-eager=false \
    --port 8000

# Disabled - for debugging or when CUDA graphs cause issues
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --enforce-eager \
    --port 8000
```

**Expected improvement: 10-30%** reduction in per-token latency (decode). The benefit comes from eliminating kernel launch overhead (~1-2ms per step from ~200 individual kernel launches).

**When CUDA graphs DON'T help:**
- Large batch sizes where kernel launch overhead is a tiny fraction of total time
- Prefill-dominant workloads (long prompts, short outputs) -- prefill uses dynamic shapes
- Models with dynamic control flow (e.g., Mixture-of-Experts routing can complicate graph capture)
- When running with `torch.compile(mode="reduce-overhead")` which already uses CUDA graphs internally

**How to verify they are active (check logs):**

```bash
# Look for these log messages when vLLM starts:
# "Using CUDA graph for decoding..."
# "Graph capturing finished in X secs, took Y GiB"

# If you see:
# "Eager mode is enforced"
# then CUDA graphs are NOT active.

# You can also check by comparing latency:
# Run with --enforce-eager, note TPOT
# Run without --enforce-eager, note TPOT
# The difference is the CUDA graph benefit
```

### Recipe 4: Speculative Decoding Setup

**vLLM command with draft model:**

```bash
# Use a smaller model from the same family as the draft
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model meta-llama/Llama-3.1-8B-Instruct \
    --num-speculative-tokens 5 \
    --port 8000

# Alternative: use [ngram] for n-gram based speculation (no draft model needed)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model [ngram] \
    --num-speculative-tokens 5 \
    --ngram-prompt-lookup-max 4 \
    --port 8000
```

**How to measure acceptance rate:**

```python
# vLLM logs the acceptance rate. Look for log lines like:
# "Speculative metrics: acceptance_rate=0.82, ..."

# Or query the /metrics endpoint:
import requests
metrics = requests.get("http://localhost:8000/metrics").text
# Look for: vllm:spec_decode_draft_acceptance_rate
for line in metrics.split("\n"):
    if "spec_decode" in line:
        print(line)
```

**How to choose k (num_speculative_tokens):**

| k value | Best when | Trade-off |
|---------|-----------|-----------|
| 3 | Low acceptance rate (<60%) | Conservative, less wasted compute |
| 5 | Moderate acceptance rate (60-80%) | Good default starting point |
| 7-10 | High acceptance rate (>80%), e.g., code completion, translation | More aggressive, higher ceiling |

Rule of thumb: **expected speedup = mean_accepted / (1 + draft_cost/target_cost)**. If acceptance rate drops below ~50%, try lower k or a better-aligned draft model.

```
# Quick test: try k=3, 5, 7 and measure tokens/sec
# Pick the k that gives highest throughput
for k in 3 5 7; do
    echo "Testing k=$k"
    # restart server with --num-speculative-tokens $k
    # run benchmark, record tokens/sec
done
```

**When speculative decoding hurts:**
- Low acceptance rate (<40%): draft tokens are mostly rejected, wasting compute
- High-batch serving: draft model competes for GPU resources and memory with the target model
- Very small target models: draft overhead is proportionally large
- Creative/high-temperature sampling: low predictability means low acceptance
- The draft model itself is too large (>15% of target model cost)

### Recipe 5: Memory-Bound Decode Optimization Checklist

```
[ ] Step 1: Measure baseline decode tokens/sec
    → Use Recipe 1 script with max-tokens=256, short prompt
    → Record: tokens/sec, TPOT, GPU memory used

[ ] Step 2: Check if decode is memory-bound (should be at batch=1)
    → GPU compute utilization should be LOW (<30%)
    → Memory bandwidth utilization should be HIGH (>60%)
    → Verify: nvidia-smi shows low SM activity during decode
    → Rule of thumb: batch=1 decode is almost always memory-bound

[ ] Step 3: Quantize weights (FP16 -> INT4 = ~3x speedup expected)
    → Follow Recipe 2 for AWQ INT4
    → Expected: 2.5-3.5x decode tokens/sec improvement
    → Check quality on your eval set before proceeding

[ ] Step 4: Enable CUDA graphs (10-30% more)
    → Follow Recipe 3 (usually on by default in vLLM)
    → Verify active via logs
    → Expected: 10-30% additional TPOT reduction

[ ] Step 5: Try speculative decoding (1.5-3x more)
    → Follow Recipe 4
    → Tune k based on acceptance rate
    → Expected: 1.5-3x more if acceptance rate > 70%
    → Skip if serving high-concurrency (batch > 8)

[ ] Step 6: Quantize KV cache to FP8 (allows larger batch)
    → vLLM: --kv-cache-dtype fp8
    → Halves KV cache memory → fit 2x more concurrent requests
    → Minimal quality impact for most models
    → Most impactful for long-context workloads (>8K tokens)
```

### Recipe 6: Compute-Bound Prefill Optimization Checklist

```
[ ] Step 1: Measure TTFT baseline
    → Use Recipe 1 with long prompt (2048+ tokens), max-tokens=1
    → Record: TTFT in ms
    → This isolates prefill performance

[ ] Step 2: Enable chunked prefill
    → vLLM: --enable-chunked-prefill --max-num-batched-tokens 2048
    → Improves decode latency when prefill and decode run concurrently
    → May slightly increase TTFT but improves overall system latency

[ ] Step 3: Use FlashAttention (verify it is active)
    → pip install flash-attn
    → vLLM uses FlashAttention by default when installed
    → Check logs for: "Using FlashAttention-2" or similar
    → If not active, check: GPU support (Ampere+), head dim compatibility
    → FlashAttention-3 on Hopper: even faster with warp specialization

[ ] Step 4: Increase tensor parallelism if multi-GPU
    → vLLM: --tensor-parallel-size 2 (or 4, 8)
    → Prefill scales nearly linearly with TP (compute-bound)
    → Requires NVLink for best results (PCIe adds latency)
    → Expected: ~1.8x speedup per 2x TP increase

[ ] Step 5: Consider FP8 compute (Ada/Hopper/Blackwell: ~1.5x faster matmul)
    → vLLM: --dtype float8 (or use a pre-quantized FP8 model)
    → FP8 tensor cores (Ada/Hopper/Blackwell): 2x FLOPS vs FP16
    → Realized improvement: ~1.5-1.8x for prefill (not full 2x due to overhead)
    → Blackwell FP4: up to ~3x realized improvement
```

### How to Identify Your Bottleneck

**Python code to measure time per phase (prefill vs decode):**

```python
#!/usr/bin/env python3
"""Measure prefill vs decode time separately to identify bottleneck."""

import time
import requests
import json

def measure_phases(url: str, model: str, prompt: str, max_tokens: int = 128):
    """Measure TTFT (prefill indicator) and TPOT (decode indicator) separately."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }

    token_count = 0
    first_token_time = None
    last_token_time = None
    start = time.perf_counter()

    with requests.post(f"{url}/v1/completions", json=payload, stream=True, timeout=120) as resp:
        for line in resp.iter_lines():
            if not line:
                continue
            decoded = line.decode("utf-8")
            if decoded.startswith("data: ") and decoded.strip() != "data: [DONE]":
                now = time.perf_counter()
                if first_token_time is None:
                    first_token_time = now
                last_token_time = now
                token_count += 1

    ttft_ms = (first_token_time - start) * 1000
    decode_time_ms = (last_token_time - first_token_time) * 1000 if token_count > 1 else 0
    tpot_ms = decode_time_ms / (token_count - 1) if token_count > 1 else 0

    return {
        "ttft_ms": ttft_ms,
        "decode_time_ms": decode_time_ms,
        "tpot_ms": tpot_ms,
        "total_tokens": token_count,
        "decode_tokens_per_sec": (token_count - 1) / (decode_time_ms / 1000) if decode_time_ms > 0 else 0,
    }

# Run with short prompt (decode-dominated)
short = measure_phases("http://localhost:8000", "your-model", "Hello", max_tokens=256)

# Run with long prompt (prefill-dominated)
long_prompt = "Explain quantum computing. " * 200  # ~1000 tokens
long = measure_phases("http://localhost:8000", "your-model", long_prompt, max_tokens=16)

print("=== Short prompt (decode-dominated) ===")
print(f"  TTFT:       {short['ttft_ms']:.1f} ms")
print(f"  TPOT:       {short['tpot_ms']:.2f} ms")
print(f"  Decode tok/s: {short['decode_tokens_per_sec']:.1f}")
print()
print("=== Long prompt (prefill-dominated) ===")
print(f"  TTFT:       {long['ttft_ms']:.1f} ms")
print(f"  TPOT:       {long['tpot_ms']:.2f} ms")
print()
print("=== Diagnosis ===")
if short['tpot_ms'] > 30:
    print("DECODE IS SLOW -> Follow Recipe 5 (Memory-Bound Decode Checklist)")
elif long['ttft_ms'] > 500:
    print("PREFILL IS SLOW -> Follow Recipe 6 (Compute-Bound Prefill Checklist)")
else:
    print("Both phases look reasonable. Profile further if needed.")
```

**Decision tree based on measurements:**

```
START: Run both short-prompt and long-prompt benchmarks above
  │
  ├─ TTFT > target? (e.g., > 200ms for 8B, > 500ms for 70B)
  │   ├─ YES → Prefill bottleneck
  │   │   ├─ Single GPU?
  │   │   │   ├─ Use FP8 compute (Recipe 6, Step 5)
  │   │   │   ├─ Enable FlashAttention (Recipe 6, Step 3)
  │   │   │   └─ Consider torch.compile max-autotune
  │   │   └─ Multi GPU?
  │   │       └─ Increase TP (Recipe 6, Step 4)
  │   └─ NO → Prefill is fine
  │
  ├─ TPOT > target? (e.g., > 25ms for 8B, > 50ms for 70B)
  │   ├─ YES → Decode bottleneck
  │   │   ├─ Using FP16? → Quantize to INT4 AWQ (Recipe 2)
  │   │   ├─ Already quantized? → Enable CUDA graphs (Recipe 3)
  │   │   ├─ Already have CUDA graphs? → Try speculative decoding (Recipe 4)
  │   │   └─ Batch=1? → Increase batch size if throughput matters
  │   └─ NO → Decode is fine
  │
  └─ Both fine but throughput too low?
      ├─ Increase batch size (continuous batching)
      ├─ Quantize to free memory for more concurrent requests
      └─ Add GPUs with TP or PP
```

### Real Performance Numbers

Approximate single-stream (batch=1) decode performance. Numbers are representative ranges based on community benchmarks and published results. Actual performance varies with prompt length, generation length, software version, and system configuration.

**LLaMA 3.1 8B Instruct**

| GPU | FP16 | AWQ INT4 | FP8 | INT4 + CUDA Graphs | INT4 + Spec. Decoding (k=5) |
|-----|------|----------|-----|---------------------|------------------------------|
| RTX 4090 (24GB) | 55-65 tok/s | 140-170 tok/s | 100-130 tok/s | 160-190 tok/s | 250-350 tok/s |
| A100 80GB | 70-85 tok/s | 160-200 tok/s | N/A (no FP8, Ampere) | 180-220 tok/s | 300-450 tok/s |
| H100 80GB | 100-130 tok/s | 250-310 tok/s | 200-250 tok/s | 280-340 tok/s | 450-650 tok/s |

**LLaMA 3.1 70B Instruct**

| GPU | FP16 | AWQ INT4 | FP8 | INT4 + CUDA Graphs | INT4 + Spec. Decoding (k=5) |
|-----|------|----------|-----|---------------------|------------------------------|
| RTX 4090 (24GB) | OOM | ~20 tok/s (offload) | N/A | ~22 tok/s (offload) | N/A (insufficient VRAM) |
| A100 80GB (TP=2) | 18-24 tok/s | 45-60 tok/s | 35-45 tok/s | 55-70 tok/s | 80-120 tok/s |
| A100 80GB (TP=4) | 30-40 tok/s | 75-95 tok/s | 55-70 tok/s | 85-105 tok/s | 130-190 tok/s |
| H100 80GB (TP=2) | 30-40 tok/s | 75-95 tok/s | 60-80 tok/s | 85-110 tok/s | 140-200 tok/s |
| H100 80GB (TP=4) | 50-65 tok/s | 120-155 tok/s | 95-125 tok/s | 140-175 tok/s | 220-320 tok/s |

**Key takeaways from the numbers:**
- INT4 quantization is the single highest-impact optimization for decode (2.5-3.5x)
- CUDA graphs add 10-20% on top of quantization
- Speculative decoding can add another 1.5-2.5x but requires extra memory for the draft model
- FP8 gives ~1.7x over FP16, but INT4 beats FP8 for decode (fewer bytes to read)
- H100 advantage over A100 is ~1.5x for memory-bound decode (3.35 TB/s vs 2.0 TB/s HBM bandwidth)
- RTX 4090 is competitive with A100 for small models due to high memory bandwidth per dollar
