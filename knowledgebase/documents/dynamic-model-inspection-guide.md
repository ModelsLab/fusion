---
id: dynamic-model-inspection-guide
kind: document
title: "Dynamic Model Inspection: Extract Architecture from Any Checkpoint"
category: inspection
summary: Complete guide for dynamically extracting model architecture, parameter counts, memory requirements, and operator inventory from any model checkpoint using PyTorch tools — never hardcode model specs
tags:
  - model-inspection
  - architecture
  - memory-estimation
  - operator-inventory
  - dynamic
gpu_families:
  - Ampere
  - Ada
  - Hopper
  - Blackwell
workloads:
  - prefill
  - decode
  - serving
operators:
  - attention
  - matmul
  - rmsnorm
  - layernorm
  - embedding
  - lm-head
backends:
  - python
runtimes:
  - transformers
  - vllm
  - sglang
---

## Why Dynamic Inspection

Never hardcode model specs. New models appear weekly. The agent MUST extract architecture details from the actual checkpoint at optimization time. This document provides every tool and pattern needed.

## Quick Inspection Script (copy-paste ready)

```python
#!/usr/bin/env python3
"""Fusion Model Inspector — extract architecture from any HF model or local checkpoint."""
import json
import sys
import os

def inspect_model(model_id_or_path, output_path="model_inspection_report.json"):
    from transformers import AutoConfig

    # --- Step 1: Load config (no weights, no GPU) ---
    try:
        config = AutoConfig.from_pretrained(model_id_or_path, trust_remote_code=True)
    except Exception as e:
        # Fallback: try loading config.json directly
        config_path = os.path.join(model_id_or_path, "config.json") if os.path.isdir(model_id_or_path) else None
        if config_path and os.path.exists(config_path):
            with open(config_path) as f:
                raw = json.load(f)
            print(f"WARNING: AutoConfig failed ({e}), using raw config.json")
            print(json.dumps(raw, indent=2))
            return raw
        raise

    # --- Step 2: Extract architecture fields ---
    def g(attr, default=None):
        return getattr(config, attr, default)

    model_type = g("model_type", "unknown")
    hidden = g("hidden_size") or g("d_model") or g("n_embd")
    n_layers = g("num_hidden_layers") or g("n_layer") or g("num_layers")
    n_heads = g("num_attention_heads") or g("n_head")
    n_kv_heads = g("num_key_value_heads", n_heads)  # defaults to MHA if not set
    intermediate = g("intermediate_size") or g("n_inner") or (hidden * 4)
    vocab = g("vocab_size")
    max_ctx = g("max_position_embeddings") or g("n_positions") or g("max_seq_len", 2048)
    head_dim = hidden // n_heads if (hidden and n_heads) else None
    gqa_ratio = n_heads // n_kv_heads if (n_heads and n_kv_heads) else 1

    # Activation
    act = g("hidden_act") or g("activation_function", "unknown")

    # RoPE
    rope_theta = g("rope_theta")
    rope_scaling = g("rope_scaling")

    # Sliding window
    sliding_window = g("sliding_window")

    # MoE
    n_experts = g("num_local_experts") or g("num_experts", 1)
    experts_per_tok = g("num_experts_per_tok") or g("top_k", 1)

    # Tied embeddings
    tie_embeddings = g("tie_word_embeddings", False)

    # Native dtype
    native_dtype = str(g("torch_dtype", "float32"))

    # --- Step 3: Calculate parameters ---
    if hidden and n_heads and n_kv_heads and intermediate and n_layers and vocab:
        qkv_per_layer = hidden * (n_heads + 2 * n_kv_heads) * head_dim
        o_per_layer = n_heads * head_dim * hidden

        if act in ("silu", "swiglu", "gelu_new") or "glu" in str(act).lower():
            # GLU-style: gate_proj + up_proj + down_proj = 3 * hidden * intermediate
            ffn_per_layer = 3 * hidden * intermediate
        else:
            # Standard: fc1 + fc2 = 2 * hidden * intermediate
            ffn_per_layer = 2 * hidden * intermediate

        norm_per_layer = 2 * hidden  # 2 norms (pre-attn, pre-ffn)

        if n_experts > 1:
            # MoE: each expert has its own FFN, plus router weights
            attn_per_layer = qkv_per_layer + o_per_layer + hidden  # +hidden for norm
            expert_per_layer = n_experts * ffn_per_layer
            router_per_layer = hidden * n_experts
            per_layer = attn_per_layer + expert_per_layer + router_per_layer + norm_per_layer
            active_per_layer = attn_per_layer + experts_per_tok * ffn_per_layer + norm_per_layer
        else:
            per_layer = qkv_per_layer + o_per_layer + ffn_per_layer + norm_per_layer
            active_per_layer = per_layer

        embed_params = vocab * hidden
        lm_head_params = 0 if tie_embeddings else vocab * hidden
        final_norm = hidden

        total_params = n_layers * per_layer + embed_params + lm_head_params + final_norm
        active_params = n_layers * active_per_layer + embed_params + lm_head_params + final_norm
    else:
        total_params = active_params = per_layer = active_per_layer = embed_params = lm_head_params = None

    # --- Step 4: Calculate memory ---
    def mem_gb(params, bytes_per_param):
        return round(params * bytes_per_param / 1e9, 2) if params else None

    memory = {
        "weights_fp32_gb": mem_gb(total_params, 4),
        "weights_bf16_gb": mem_gb(total_params, 2),
        "weights_fp8_gb": mem_gb(total_params, 1),
        "weights_int4_gb": mem_gb(total_params, 0.5),
    }

    # --- Step 5: KV cache per token ---
    if n_layers and n_kv_heads and head_dim:
        kv_bytes_bf16 = 2 * n_layers * n_kv_heads * head_dim * 2  # 2 for K,V; 2 bytes for bf16
        kv_bytes_fp8 = 2 * n_layers * n_kv_heads * head_dim * 1
        kv_bytes_int4 = 2 * n_layers * n_kv_heads * head_dim * 0.5
    else:
        kv_bytes_bf16 = kv_bytes_fp8 = kv_bytes_int4 = None

    kv_cache = {
        "bytes_per_token_bf16": kv_bytes_bf16,
        "bytes_per_token_fp8": kv_bytes_fp8,
        "bytes_per_token_int4": kv_bytes_int4,
    }
    if kv_bytes_bf16 and max_ctx:
        kv_cache["full_context_bf16_gb"] = round(kv_bytes_bf16 * max_ctx / 1e9, 2)
        kv_cache["full_context_fp8_gb"] = round(kv_bytes_fp8 * max_ctx / 1e9, 2)

    # --- Step 6: "Will it fit?" estimates ---
    gpu_vram = {
        "RTX 3090": 24, "RTX 4090": 24, "RTX 5090": 32,
        "A100 40GB": 40, "A100 80GB": 80, "H100 80GB": 80, "H200 141GB": 141,
    }
    fit_table = {}
    if total_params:
        for gpu_name, vram in gpu_vram.items():
            fit_table[gpu_name] = {}
            for prec, bpp in [("bf16", 2), ("fp8", 1), ("int4", 0.5)]:
                weight_gb = total_params * bpp / 1e9
                overhead_gb = 1.5  # CUDA context, activations, etc.
                remaining = vram - weight_gb - overhead_gb
                kv_per_tok = kv_bytes_bf16 if prec == "bf16" else kv_bytes_fp8 if prec == "fp8" else kv_bytes_int4
                max_tokens = int(remaining * 1e9 / kv_per_tok) if (remaining > 0 and kv_per_tok) else 0
                fit_table[gpu_name][prec] = {
                    "fits": weight_gb + overhead_gb < vram,
                    "weight_gb": round(weight_gb, 2),
                    "free_for_kv_gb": round(max(0, remaining), 2),
                    "max_context_tokens": max(0, max_tokens),
                }

    # --- Step 7: Build report ---
    report = {
        "model_id": model_id_or_path,
        "model_type": model_type,
        "architecture": {
            "hidden_size": hidden,
            "num_hidden_layers": n_layers,
            "num_attention_heads": n_heads,
            "num_key_value_heads": n_kv_heads,
            "head_dim": head_dim,
            "gqa_ratio": gqa_ratio,
            "intermediate_size": intermediate,
            "vocab_size": vocab,
            "max_position_embeddings": max_ctx,
            "hidden_act": act,
            "rope_theta": rope_theta,
            "rope_scaling": rope_scaling,
            "sliding_window": sliding_window,
            "num_experts": n_experts if n_experts > 1 else None,
            "num_experts_per_tok": experts_per_tok if n_experts > 1 else None,
            "tie_word_embeddings": tie_embeddings,
            "native_dtype": native_dtype,
        },
        "parameters": {
            "total": total_params,
            "active": active_params if n_experts > 1 else total_params,
            "per_layer": per_layer,
            "active_per_layer": active_per_layer if n_experts > 1 else per_layer,
            "embedding": embed_params,
            "lm_head": lm_head_params,
            "total_billions": round(total_params / 1e9, 2) if total_params else None,
        },
        "memory": memory,
        "kv_cache": kv_cache,
        "gpu_fit": fit_table,
        "features": {
            "grouped_query_attention": gqa_ratio > 1,
            "multi_query_attention": n_kv_heads == 1 if n_kv_heads else False,
            "mixture_of_experts": n_experts > 1,
            "sliding_window_attention": sliding_window is not None,
            "rope_scaling": rope_scaling is not None,
            "tied_embeddings": tie_embeddings,
        },
        "optimization_hints": [],
    }

    # --- Step 8: Generate optimization hints ---
    hints = report["optimization_hints"]

    if total_params and total_params > 30e9:
        hints.append("Model >30B params — consider tensor parallelism or quantization to INT4/FP8")
    if total_params and total_params <= 10e9:
        hints.append("Model ≤10B params — single GPU likely sufficient, focus on quantization + kernel fusion")

    if gqa_ratio > 1:
        hints.append(f"Uses GQA (ratio {gqa_ratio}:1) — KV cache is {gqa_ratio}x smaller than MHA, good for long context")

    if n_experts and n_experts > 1:
        hints.append(f"MoE model ({n_experts} experts, top-{experts_per_tok}) — active params much less than total, expert parallelism possible")
        hints.append("MoE routing kernel and grouped GEMM are key optimization targets")

    if vocab and vocab > 64000:
        hints.append(f"Large vocabulary ({vocab}) — embedding and lm_head are significant, consider chunked LM head or FP8 LM head")

    if sliding_window:
        hints.append(f"Sliding window attention (window={sliding_window}) — can limit KV cache memory")

    if kv_bytes_bf16 and max_ctx and max_ctx > 32768:
        ctx_gb = kv_bytes_bf16 * max_ctx / 1e9
        hints.append(f"Long context model ({max_ctx} tokens) — full KV cache at BF16 = {ctx_gb:.1f} GB, consider FP8/INT4 KV cache quantization")

    if act in ("silu",) or "glu" in str(act).lower():
        hints.append("Uses SiLU/GLU activation — fuse gate_proj * up_proj * SiLU into single kernel for decode speedup")

    # --- Output ---
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(json.dumps(report, indent=2, default=str))
    return report

if __name__ == "__main__":
    model_id = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Llama-3.1-8B"
    output = sys.argv[2] if len(sys.argv) > 2 else "model_inspection_report.json"
    inspect_model(model_id, output)
```

## Usage

```bash
# From HuggingFace model ID
python inspect_model.py meta-llama/Llama-3.1-8B

# From local checkpoint path
python inspect_model.py /path/to/model/checkpoint/

# With custom output
python inspect_model.py deepseek-ai/DeepSeek-V3 report.json
```

## Operator Inventory (Optional Deep Inspection)

When the agent needs to know exactly which PyTorch modules are in the model:

```python
import torch
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

# Load on meta device — no memory, no weights, just structure
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

# Classify all operators
operator_types = {}
for name, module in model.named_modules():
    cls = type(module).__name__
    if cls not in operator_types:
        operator_types[cls] = {"count": 0, "examples": []}
    operator_types[cls]["count"] += 1
    if len(operator_types[cls]["examples"]) < 3:
        operator_types[cls]["examples"].append(name)

# Identify hot operators for optimization
hot_operators = []
for cls, info in sorted(operator_types.items(), key=lambda x: -x[1]["count"]):
    if any(k in cls.lower() for k in ["linear", "attention", "norm", "embed", "conv"]):
        hot_operators.append({"class": cls, "count": info["count"], "example": info["examples"][0]})

print("Hot operators to optimize:")
for op in hot_operators:
    print(f"  {op['class']}: {op['count']} instances (e.g., {op['example']})")
```

## Shape Extraction from Safetensors (No Model Loading)

For very large models where even meta-device loading is slow:

```python
import json
import os
from safetensors import safe_open

def get_shapes_from_safetensors(model_path):
    """Extract all parameter shapes without loading any weights."""
    index_path = os.path.join(model_path, "model.safetensors.index.json")

    if os.path.exists(index_path):
        # Multi-shard model: read index
        with open(index_path) as f:
            index = json.load(f)

        shapes = {}
        seen_files = set()
        for param_name, shard_file in index["weight_map"].items():
            if shard_file not in seen_files:
                seen_files.add(shard_file)
                shard_path = os.path.join(model_path, shard_file)
                with safe_open(shard_path, framework="pt") as f:
                    for key in f.keys():
                        shapes[key] = list(f.get_slice(key).get_shape())
        return shapes

    # Single shard
    shard_path = os.path.join(model_path, "model.safetensors")
    if os.path.exists(shard_path):
        with safe_open(shard_path, framework="pt") as f:
            return {key: list(f.get_slice(key).get_shape()) for key in f.keys()}

    return {}

# Usage
shapes = get_shapes_from_safetensors("/path/to/model")
for name, shape in sorted(shapes.items()):
    print(f"{name}: {shape}")
```

## GGUF Model Inspection

```python
def inspect_gguf(gguf_path):
    """Extract architecture from GGUF file metadata."""
    try:
        from gguf import GGUFReader
        reader = GGUFReader(gguf_path)

        metadata = {}
        for field in reader.fields:
            metadata[field] = reader.fields[field].parts[-1].tolist()

        # Common GGUF metadata keys
        arch_keys = [
            "llama.block_count",        # num_hidden_layers
            "llama.embedding_length",   # hidden_size
            "llama.feed_forward_length", # intermediate_size
            "llama.attention.head_count",     # num_attention_heads
            "llama.attention.head_count_kv",  # num_key_value_heads
            "llama.context_length",     # max_position_embeddings
            "llama.rope.freq_base",     # rope_theta
            "general.architecture",     # model type
            "general.name",
            "general.quantization_version",
        ]

        report = {}
        for key in arch_keys:
            if key in metadata:
                report[key] = metadata[key]

        return report
    except ImportError:
        print("pip install gguf")
        return None
```

## Profiling Operator Time Distribution

After inspection, profile to find where time actually goes:

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

def profile_operator_distribution(model, input_ids, num_warmup=3, num_runs=5):
    """Profile operator-level time distribution."""
    model.eval()

    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            model(input_ids)

    # Profile
    with profile(
        activities=[ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        for _ in range(num_runs):
            with torch.no_grad():
                model(input_ids)

    # Aggregate by operator type
    events = prof.key_averages()
    total_cuda_time = sum(e.cuda_time_total for e in events if e.cuda_time_total > 0)

    print(f"\nOperator Time Distribution (total CUDA time: {total_cuda_time/1000:.1f} ms)")
    print("-" * 80)
    for event in sorted(events, key=lambda e: -e.cuda_time_total)[:20]:
        pct = event.cuda_time_total / total_cuda_time * 100 if total_cuda_time > 0 else 0
        print(f"{event.key:50s} {event.cuda_time_total/1000:8.2f} ms  {pct:5.1f}%")

    return events
```

## Memory Estimation: "Will It Fit?"

```python
def will_it_fit(report, gpu_vram_gb, batch_size=1, context_length=2048, precision="bf16"):
    """Check if model fits on GPU with given config."""
    prec_map = {"fp32": 4, "bf16": 2, "fp16": 2, "fp8": 1, "int8": 1, "int4": 0.5}
    bpp = prec_map.get(precision, 2)

    weight_gb = report["parameters"]["total"] * bpp / 1e9

    kv_bpp = bpp  # KV cache same precision by default
    kv_per_token = report["kv_cache"]["bytes_per_token_bf16"]
    if precision in ("fp8", "int8"):
        kv_per_token = report["kv_cache"]["bytes_per_token_fp8"]

    kv_gb = batch_size * context_length * kv_per_token / 1e9

    # Activation memory (rough estimate: ~2x one layer's hidden state per batch)
    hidden = report["architecture"]["hidden_size"]
    act_gb = batch_size * context_length * hidden * 4 / 1e9  # fp32 activations

    overhead_gb = 1.0  # CUDA context

    total_gb = weight_gb + kv_gb + act_gb + overhead_gb

    fits = total_gb < gpu_vram_gb

    return {
        "fits": fits,
        "total_gb": round(total_gb, 2),
        "weight_gb": round(weight_gb, 2),
        "kv_cache_gb": round(kv_gb, 2),
        "activation_gb": round(act_gb, 2),
        "overhead_gb": overhead_gb,
        "free_gb": round(gpu_vram_gb - total_gb, 2),
        "gpu_vram_gb": gpu_vram_gb,
    }
```

## Decision Making from Inspection

After running the inspector, the agent should use the report to decide:

| Report Field | Decision |
|---|---|
| `parameters.total > 30B` | Need tensor parallelism or aggressive quantization |
| `parameters.total <= 10B` | Single GPU, focus on quant + kernel fusion |
| `features.gqa = true` | KV cache is compact, long context feasible |
| `features.moe = true` | Expert parallelism, grouped GEMM optimization |
| `gpu_fit[gpu][prec].fits = false` | Must use lower precision or tensor parallelism |
| `gpu_fit[gpu][prec].max_context_tokens < needed` | Need KV cache quantization or offloading |
| `architecture.vocab_size > 64K` | LM head is a bottleneck, consider chunked/FP8 |
| `architecture.sliding_window != null` | Can cap KV cache size, saves memory |
| `kv_cache.full_context_bf16_gb > 10` | Need FP8 or INT4 KV cache |
