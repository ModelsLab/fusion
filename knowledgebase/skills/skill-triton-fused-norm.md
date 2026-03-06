---
id: skill_triton_fused_norm
kind: skill
title: Triton Fusion Loop
type: ""
category: kernel
summary: Generate, verify, and benchmark Triton kernels for fused bandwidth-heavy operators such as norms, softmax, dequant, and KV transforms.
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
  - layernorm
  - rmsnorm
  - softmax
  - dequantization
  - kv-cache
  - attention
gpu_families:
  - Ampere
  - Ada
  - Hopper
  - Blackwell
gpu_ids:
  - rtx4090
  - rtx5090
  - a100
  - h100
  - h200
  - b200
precision:
  - bf16
  - fp16
  - fp8
  - int8
  - int4
bottlenecks:
  - memory
  - mixed
goals:
  - throughput
  - latency
priority: 0
preconditions: []
actions: []
metrics: []
tradeoffs: []
preferred_backends:
  - triton
required_tools:
  - read_file
  - write_file
  - run_command
  - run_benchmark
  - search_knowledge_base
steps:
  - start from a known-good Triton tutorial or internal template
  - generate the fused kernel around the actual hot operator rather than a synthetic guess
  - run a reference comparison after every candidate edit
  - benchmark the passing kernels on the representative decode or prefill shape set
verification:
  - the Triton kernel matches the reference implementation within tolerance
  - shape guards and data-layout assumptions are explicit
  - the kernel shows a real benchmark gain before runtime integration
benchmark_rubric:
  - track kernel latency, HBM throughput, end-to-end tokens per second, and launch stability
  - test both small and production-like shapes because Triton winners can be shape-sensitive
failure_recovery:
  - fall back to CuTe or CUTLASS when the kernel becomes tensor-core dominated
  - fall back to the runtime-native kernel if the maintenance cost outweighs the gain
artifacts_to_save:
  - kernel_source
  - correctness_report
  - benchmark_json
  - shape_notes_md
runtime_adapters:
  - transformers
  - vllm
  - sglang
reference_source_ids:
  - triton-tutorials
  - triton-block-scaled-matmul
  - cuda-agent
backend: ""
runtimes:
  - transformers
  - vllm
  - sglang
use_cases: []
notes: []
reference_paths: []
backends: []
path: ""
---

## Steps

- start from a known-good Triton tutorial or internal template
- generate the fused kernel around the actual hot operator rather than a synthetic guess
- run a reference comparison after every candidate edit
- benchmark the passing kernels on the representative decode or prefill shape set

## Verification

- the Triton kernel matches the reference implementation within tolerance
- shape guards and data-layout assumptions are explicit
- the kernel shows a real benchmark gain before runtime integration

## Benchmark Rubric

- track kernel latency, HBM throughput, end-to-end tokens per second, and launch stability
- test both small and production-like shapes because Triton winners can be shape-sensitive

## Failure Recovery

- fall back to CuTe or CUTLASS when the kernel becomes tensor-core dominated
- fall back to the runtime-native kernel if the maintenance cost outweighs the gain
