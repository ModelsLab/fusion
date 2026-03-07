---
id: skill_fp8_serving_eval
kind: skill
title: FP8 Serving Evaluation
type: ""
category: precision
summary: Validate packaged or synthesized FP8 deployment branches before investing in custom tensor-core kernel work on Ada, Hopper, or Blackwell GPUs (all have FP8 tensor cores).
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
  - matmul
  - attention
gpu_families:
  - Ada
  - Hopper
  - Blackwell
gpu_ids:
  - rtx4090
  - l40s
  - h100
  - h200
  - b200
precision:
  - fp8
  - bf16
  - fp16
bottlenecks:
  - compute
  - mixed
goals:
  - throughput
  - cost
  - memory
priority: 0
preconditions: []
actions: []
metrics: []
tradeoffs: []
preferred_backends:
  - runtime
  - cute_dsl
  - cuda_cutlass_cpp
required_tools:
  - plan_optimization
  - run_benchmark
  - search_knowledge_base
steps:
  - measure the higher-precision control path first
  - if the model already has a packaged FP8 checkpoint or runtime flavor, test that first
  - if no packaged FP8 checkpoint exists, try synthesizing one with TensorRT Model Optimizer, Transformer Engine, torchao float8 flows, or llm-compressor using the same benchmark suite
  - record calibration data, quantization configuration, and any higher-precision fallback modules
  - track quality drift and throughput together before any custom kernel work
  - only move beyond the FP8 path if the hot operator remains dominant after the runtime baseline
verification:
  - quality stays within the deployment budget
  - the FP8 branch is stable across representative prompts and batch sizes
  - fallback higher-precision blocks are documented when required
benchmark_rubric:
  - measure throughput, latency, tensor utilization, and memory footprint
  - store the exact runtime and calibration configuration with the result
failure_recovery:
  - fallback to bf16 or fp16 for unstable layers
  - fallback to weight-only or mixed-precision quantization if full FP8 activation paths regress quality or fail in the runtime
  - switch to AWQ or KV-cache optimization if the remaining bottleneck is mostly memory traffic
artifacts_to_save:
  - benchmark_json
  - quality_report
  - runtime_config
  - calibration_notes_md
runtime_adapters:
  - tensorrt-llm
  - vllm
  - sglang
reference_source_ids:
  - nvidia-transformer-engine
  - tensorrt-llm-docs
  - nvidia-blackwell-cutlass
  - nvidia-tensorrt-model-optimizer
  - llm-compressor
  - torchao
backend: ""
runtimes:
  - tensorrt-llm
  - vllm
  - sglang
use_cases: []
notes: []
reference_paths: []
backends: []
path: ""
---

## Steps

- measure the higher-precision control path first
- if the model already has a packaged FP8 checkpoint or runtime flavor, test that first
- if no packaged FP8 checkpoint exists, try synthesizing one with TensorRT Model Optimizer, Transformer Engine, torchao float8 flows, or llm-compressor using the same benchmark suite
- record calibration data, quantization configuration, and any higher-precision fallback modules
- track quality drift and throughput together before any custom kernel work
- only move beyond the FP8 path if the hot operator remains dominant after the runtime baseline

## Verification

- quality stays within the deployment budget
- the FP8 branch is stable across representative prompts and batch sizes
- fallback higher-precision blocks are documented when required

## Benchmark Rubric

- measure throughput, latency, tensor utilization, and memory footprint
- store the exact runtime and calibration configuration with the result

## Failure Recovery

- fallback to bf16 or fp16 for unstable layers
- fallback to weight-only or mixed-precision quantization if full FP8 activation paths regress quality or fail in the runtime
- switch to AWQ or KV-cache optimization if the remaining bottleneck is mostly memory traffic
