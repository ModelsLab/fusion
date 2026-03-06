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
   - FP8 tensor cores: 2x FLOPS vs FP16 on Hopper
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
