---
id: weight_only_int4_multimodal
kind: strategy
title: Use Weight-Only INT4 As A Fit-First Branch For DiT And Multimodal Transformers
type: ""
category: precision
summary: For diffusion, DiT, video, and multimodal transformer models, weight-only INT4 should be treated as a generic quantization branch that primarily targets memory fit and bandwidth reduction, not assumed as the fastest path by default.
support_level: stable
reliability: ""
review_status: ""
url: ""
tags: []
aliases:
  - dit-int4
  - multimodal-int4
  - weight-only-int4
family: ""
market: ""
compute_capability: ""
memory_gb: 0
memory_bandwidth_gbps: 0
preferred_precisions: []
experimental_precisions: []
strengths: []
constraints: []
source_ids:
  - huggingface-optimum-quanto
  - huggingface-diffusers-quanto
  - huggingface-diffusers-quantization-blog
  - torchao
  - llm-compressor
workloads:
  - image-generation
  - image-editing
  - video-generation
  - audio-generation
  - multimodal
operators:
  - quantization
  - matmul
  - attention
  - feedforward
gpu_families:
  - Ampere
  - Ada
  - Hopper
  - Blackwell
gpu_ids:
  - rtx3090
  - rtx4090
  - l40s
  - h100
  - b200
precision:
  - int4
  - qint4
bottlenecks:
  - memory
  - mixed
goals:
  - memory
  - fit
  - throughput
priority: 91
preconditions:
  - the model is transformer-heavy enough for weight-only quantization to matter
  - quality gates exist for the target modality
  - the runtime has fused or efficient dequant plus GEMM kernels for the chosen path
actions:
  - treat weight-only INT4 as a first-class branch for DiT, diffusion, and multimodal transformer models
  - use Quanto, torchao, TensorRT Model Optimizer, or an equivalent modality-compatible tool before assuming AWQ flows apply
  - benchmark fit, load time, and steady-state throughput separately because lower-bit weight formats can reduce memory while still losing throughput
  - if INT4 makes a resident path fit on a limited-VRAM card, compare that resident INT4 path against an offloaded INT8 path before doing kernel work
metrics:
  - generation latency
  - VRAM footprint
  - load time
  - quality drift
tradeoffs:
  - load-time conversion can be expensive unless the quantized artifact is cached
  - some INT4 paths are slower than INT8 or FP8 when dequant overhead dominates
  - output heads, embeddings, or fragile layers may need higher precision
preferred_backends: []
required_tools: []
steps: []
verification: []
benchmark_rubric: []
failure_recovery: []
artifacts_to_save: []
runtime_adapters: []
reference_source_ids: []
backend: ""
runtimes: []
use_cases: []
notes: []
reference_paths: []
backends: []
path: ""
---

## Actions

- treat weight-only INT4 as a first-class branch for DiT, diffusion, and multimodal transformer models
- use Quanto, torchao, TensorRT Model Optimizer, or an equivalent modality-compatible tool before assuming AWQ flows apply
- benchmark fit, load time, and steady-state throughput separately because lower-bit weight formats can reduce memory while still losing throughput
- if INT4 makes a resident path fit on a limited-VRAM card, compare that resident INT4 path against an offloaded INT8 path before doing kernel work
- on 24 GB consumer cards, also compare INT4 against an explicit model-budget / staged-prefetch branch because the budgeted higher-precision path may fit the target shape without winning on smaller shapes
- when adapter weights or LoRAs are involved, prefer caching the base INT4 artifact first and attaching the adapters afterward instead of baking every adapter variant into a separate quantized cache

## Tradeoffs

- load-time conversion can be expensive unless the quantized artifact is cached
- some INT4 paths are slower than INT8 or FP8 when dequant overhead dominates
- output heads, embeddings, or fragile layers may need higher precision

## Metrics

- generation latency
- VRAM footprint
- load time
- quality drift
