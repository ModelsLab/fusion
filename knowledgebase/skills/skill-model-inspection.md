---
id: skill_model_inspection
kind: skill
title: Dynamic Model Architecture Inspection
category: inspection
summary: Extract architecture details, layer structure, parameter counts, memory requirements, and operator inventory from any model checkpoint or HuggingFace model ID using PyTorch tooling.
support_level: stable
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
  - moe-routing
gpu_families:
  - Ampere
  - Ada
  - Hopper
  - Blackwell
gpu_ids: []
precision:
  - fp32
  - bf16
  - fp16
  - fp8
  - int8
  - int4
bottlenecks:
  - memory
  - compute
  - mixed
goals:
  - throughput
  - latency
  - memory
preferred_backends:
  - python
required_tools:
  - run_command
  - write_file
  - read_file
  - search_knowledge_base
steps:
  - write a Python script that loads model config and architecture from the checkpoint or HuggingFace ID
  - extract all architecture parameters (layers, dims, heads, vocab, context length, activation, pos encoding)
  - enumerate every module and classify operators (attention, linear, norm, activation, embedding)
  - calculate parameter counts per layer and total
  - calculate memory requirements at each precision (FP32, BF16, FP8, INT4)
  - calculate KV cache size per token at each precision
  - identify special architecture features (GQA, MQA, MoE, sliding window, MLA)
  - estimate memory-bound vs compute-bound classification for decode vs prefill
  - output a structured JSON report the agent can use for all subsequent optimization decisions
verification:
  - the inspection script runs without errors on the target model
  - parameter count matches known values (cross-check with model card if available)
  - memory estimates are within 10% of actual GPU memory usage
benchmark_rubric:
  - inspection should complete in under 60 seconds even for large models (config-only load)
failure_recovery:
  - if model is too large to load, use config-only inspection (AutoConfig)
  - if model format is unknown, try safetensors header parsing for parameter shapes
  - if HuggingFace is unreachable, inspect local checkpoint files directly
artifacts_to_save:
  - model_inspection_report.json
  - model_architecture_summary.md
runtime_adapters:
  - transformers
  - vllm
  - sglang
  - llama-cpp
reference_source_ids: []
---

## Steps

1. Load model config without loading weights (fast, no GPU needed):
   ```python
   from transformers import AutoConfig
   config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
   ```

2. Extract architecture parameters from config object:
   ```python
   # These field names vary by architecture - check what exists
   arch_info = {
       "model_type": getattr(config, "model_type", "unknown"),
       "num_hidden_layers": getattr(config, "num_hidden_layers", None),
       "hidden_size": getattr(config, "hidden_size", None),
       "intermediate_size": getattr(config, "intermediate_size", None),
       "num_attention_heads": getattr(config, "num_attention_heads", None),
       "num_key_value_heads": getattr(config, "num_key_value_heads", None),
       "vocab_size": getattr(config, "vocab_size", None),
       "max_position_embeddings": getattr(config, "max_position_embeddings", None),
       "hidden_act": getattr(config, "hidden_act", None),
       "rope_theta": getattr(config, "rope_theta", None),
       "rope_scaling": getattr(config, "rope_scaling", None),
       "sliding_window": getattr(config, "sliding_window", None),
       "num_experts": getattr(config, "num_local_experts", None),
       "num_experts_per_tok": getattr(config, "num_experts_per_tok", None),
       "tie_word_embeddings": getattr(config, "tie_word_embeddings", False),
       "torch_dtype": str(getattr(config, "torch_dtype", "float32")),
   }
   ```

3. Calculate GQA ratio and head dimensions:
   ```python
   if arch_info["num_key_value_heads"]:
       arch_info["gqa_ratio"] = arch_info["num_attention_heads"] // arch_info["num_key_value_heads"]
       arch_info["head_dim"] = arch_info["hidden_size"] // arch_info["num_attention_heads"]
   ```

4. Calculate parameter count and memory at each precision:
   ```python
   # Quick estimate from config (no weight loading needed)
   def estimate_params(config):
       h = config.hidden_size
       i = config.intermediate_size
       n_layers = config.num_hidden_layers
       v = config.vocab_size
       n_kv = getattr(config, "num_key_value_heads", config.num_attention_heads)
       n_heads = config.num_attention_heads
       head_dim = h // n_heads

       # Per-layer params
       qkv = h * (n_heads + 2 * n_kv) * head_dim  # Q, K, V projections
       o_proj = n_heads * head_dim * h  # output projection
       gate_up_down = 3 * h * i  # gate_proj + up_proj + down_proj (for SwiGLU)
       norm = 2 * h  # 2 norms per layer
       per_layer = qkv + o_proj + gate_up_down + norm

       # MoE multiplier
       n_experts = getattr(config, "num_local_experts", 1)
       if n_experts > 1:
           per_layer = qkv + o_proj + norm + n_experts * (gate_up_down) + h * n_experts  # router

       # Global params
       embed = v * h
       lm_head = 0 if getattr(config, "tie_word_embeddings", False) else v * h
       final_norm = h

       total = n_layers * per_layer + embed + lm_head + final_norm
       return total, per_layer, embed, lm_head
   ```

5. Calculate KV cache size per token:
   ```python
   def kv_cache_per_token(config, precision_bytes=2):
       """Returns bytes per token for KV cache"""
       n_layers = config.num_hidden_layers
       n_kv = getattr(config, "num_key_value_heads", config.num_attention_heads)
       head_dim = config.hidden_size // config.num_attention_heads
       # 2 for K and V, per layer
       return 2 * n_layers * n_kv * head_dim * precision_bytes
   ```

6. Full inspection with weight shapes (if needed):
   ```python
   from safetensors import safe_open
   # Parse safetensors index for shapes without loading weights
   import json, os
   index_path = os.path.join(model_path, "model.safetensors.index.json")
   if os.path.exists(index_path):
       with open(index_path) as f:
           index = json.load(f)
       # index["weight_map"] gives param_name -> shard_file
       # Can derive shapes from metadata
   ```

7. Enumerate operators by loading model skeleton:
   ```python
   from transformers import AutoModelForCausalLM
   import torch
   # Load with empty weights (no memory used)
   with torch.device("meta"):
       model = AutoModelForCausalLM.from_config(config)

   # Walk all modules
   operator_inventory = {}
   for name, module in model.named_modules():
       mod_type = type(module).__name__
       if mod_type not in operator_inventory:
           operator_inventory[mod_type] = []
       operator_inventory[mod_type].append(name)
   ```

8. Output structured report:
   ```python
   report = {
       "model_id": model_id,
       "architecture": arch_info,
       "parameters": {
           "total": total_params,
           "per_layer": per_layer_params,
           "embedding": embed_params,
           "lm_head": lm_head_params,
       },
       "memory": {
           "fp32_gb": total_params * 4 / 1e9,
           "bf16_gb": total_params * 2 / 1e9,
           "fp8_gb": total_params * 1 / 1e9,
           "int4_gb": total_params * 0.5 / 1e9,
       },
       "kv_cache": {
           "bytes_per_token_bf16": kv_bytes_bf16,
           "bytes_per_token_fp8": kv_bytes_fp8,
           "max_context_gb_bf16": kv_bytes_bf16 * max_ctx / 1e9,
       },
       "features": {
           "gqa": gqa_ratio > 1,
           "gqa_ratio": gqa_ratio,
           "moe": n_experts > 1,
           "sliding_window": sliding_window is not None,
           "rope_scaling": rope_scaling is not None,
       },
       "operator_inventory": operator_inventory,
       "optimization_hints": [],
   }
   ```

## Verification

- Parameter count matches model card or community-known values
- Memory estimate is within 10% of actual measured GPU memory after loading
- All architecture fields are populated (no None for critical fields)
- Operator inventory correctly identifies attention, linear, norm, and activation modules

## Benchmark Rubric

- Config-only inspection completes in <5 seconds
- Full skeleton inspection completes in <30 seconds
- Weight-shape-only inspection (safetensors index) completes in <2 seconds

## Failure Recovery

- If `trust_remote_code=True` fails due to missing dependencies, try without and fall back to generic config fields
- If model uses a custom architecture class, parse `config.json` directly as JSON
- If safetensors index doesn't exist (single file model), use `safe_open` to read tensor names and shapes
- For GGUF models, use `gguf` Python package to read metadata header
