---
id: skill_awq_eval
kind: skill
title: AWQ And INT4 Evaluation Loop
type: ""
category: quantization
summary: Evaluate AWQ or similar INT4 paths as a first-class optimization branch for memory-bound consumer and mixed serving workloads.
support_level: stable
reliability: ""
review_status: ""
url: ""
tags: []
aliases: []
family: ""
market: ""
compute_capability: ""
memory_gb: 0
memory_bandwidth_gbps: 0
preferred_precisions: []
experimental_precisions: []
strengths: []
constraints: []
source_ids: []
workloads:
  - prefill
  - decode
  - serving
operators:
  - quantization
  - matmul
  - attention
gpu_families:
  - Ada
  - Blackwell
gpu_ids:
  - rtx4090
  - rtx5090
  - rtx6000ada
  - rtxpro6000blackwellworkstation
precision:
  - int4
  - awq-int4
  - bf16
  - fp16
bottlenecks:
  - memory
  - mixed
goals:
  - memory
  - cost
  - throughput
priority: 0
preconditions: []
actions: []
metrics: []
tradeoffs: []
preferred_backends:
  - runtime
  - triton
  - cuda_cutlass_cpp
required_tools:
  - plan_optimization
  - run_benchmark
  - run_profile
  - search_knowledge_base
steps:
  - measure the bf16 or fp16 control path first
  - run an AWQ or equivalent weight-only INT4 branch on the same prompt mix
  - compare throughput, memory footprint, and quality before writing new kernels
  - only move into fused dequantization kernel work if INT4 helps but still leaves a hot matmul path
verification:
  - quality drift stays within the deployment tolerance
  - the quantized path is stable across representative prompt lengths
  - memory savings are measured together with throughput
benchmark_rubric:
  - record TTFT, inter-token latency, steady-state tokens per second, and VRAM footprint
  - track prefill and decode separately because the best quant path can differ
failure_recovery:
  - fall back to bf16 or fp16 if quality or runtime coverage is insufficient
  - switch from weight-only quantization to KV-cache compression if decode remains memory-bound
artifacts_to_save:
  - benchmark_json
  - quality_report
  - runtime_config
  - notes_md
runtime_adapters:
  - vllm
  - tensorrt-llm
  - transformers
  - sglang
reference_source_ids:
  - awq-activation-aware-weight-quantization
  - vllm-docs
  - tensorrt-llm-docs
backend: ""
runtimes:
  - vllm
  - tensorrt-llm
  - transformers
  - sglang
use_cases: []
notes: []
reference_paths: []
backends: []
path: ""
---

## Steps

- measure the bf16 or fp16 control path first
- run an AWQ or equivalent weight-only INT4 branch on the same prompt mix
- compare throughput, memory footprint, and quality before writing new kernels
- only move into fused dequantization kernel work if INT4 helps but still leaves a hot matmul path

## Verification

- quality drift stays within the deployment tolerance
- the quantized path is stable across representative prompt lengths
- memory savings are measured together with throughput

## Benchmark Rubric

- record TTFT, inter-token latency, steady-state tokens per second, and VRAM footprint
- track prefill and decode separately because the best quant path can differ

## Failure Recovery

- fall back to bf16 or fp16 if quality or runtime coverage is insufficient
- switch from weight-only quantization to KV-cache compression if decode remains memory-bound
