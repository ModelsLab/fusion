---
id: skill_quantization_eval
kind: skill
title: Quantization Strategy Evaluation
category: quantization
summary: Systematically evaluate quantization strategies for a model on target GPU, comparing quality, speed, and memory across AWQ, GPTQ, FP8, and GGUF formats.
support_level: stable
source_ids:
  - awq-activation-aware-weight-quantization
workloads:
  - decode
  - serving
operators:
  - matmul
  - gemm
gpu_families:
  - Ampere
  - Ada
  - Hopper
  - Blackwell
precision:
  - fp8
  - int8
  - int4
  - fp4
bottlenecks:
  - memory
goals:
  - throughput
  - latency
  - memory-efficiency
preferred_backends:
  - vllm
  - sglang
  - llama-cpp
required_tools:
  - read_file
  - write_file
  - run_command
  - run_benchmark
  - search_knowledge_base
steps:
  - identify target GPU and its supported precisions
  - determine if model fits in FP16 (if yes, benchmark as baseline)
  - quantize with AWQ INT4 (g128) and benchmark
  - if GPU supports FP8, quantize with FP8 and benchmark
  - compare perplexity, tokens/sec, and memory usage
  - select best strategy based on quality/speed/memory tradeoff
verification:
  - perplexity degradation is within acceptable range (<0.5 PPL for INT4, <0.1 for FP8)
  - speed improvement matches expected theoretical gain
  - model produces coherent outputs on representative prompts
benchmark_rubric:
  - decode tokens/sec at batch=1, batch=16, batch=64
  - perplexity on standard eval set (wikitext, c4)
  - peak GPU memory usage
  - time to first token (TTFT)
failure_recovery:
  - if INT4 quality is poor, try larger group_size (64 or 32)
  - if FP8 is not available, fall back to INT8 SmoothQuant
  - if model doesn't fit even in INT4, consider layer pruning or smaller model
artifacts_to_save:
  - benchmark_json
  - perplexity_comparison
  - memory_analysis
runtime_adapters:
  - vllm
  - sglang
  - llama-cpp
  - transformers
reference_source_ids:
  - awq-activation-aware-weight-quantization
  - marlin-kernel
---

## Steps
- identify GPU capabilities and model size
- benchmark FP16 baseline if it fits
- evaluate AWQ INT4, FP8, and GGUF variants
- select based on quality/speed/memory tradeoff

## Decision Guide
| GPU Memory | Model Size | Recommended |
|-----------|-----------|-------------|
| 24 GB | 7B | FP16 or FP8 |
| 24 GB | 13B | INT4 AWQ |
| 24 GB | 34B+ | Won't fit single GPU |
| 80 GB | 70B | FP8 or INT4 |
| 80 GB | 7B-34B | FP16 (enough room) |
