---
id: skill_torch_compile_sweep
kind: skill
title: torch.compile Mode Sweep
type: ""
category: runtime
summary: Systematically sweep torch.compile modes before hand-writing kernels when the model still runs in a PyTorch-native stack.
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
  - attention
  - matmul
  - layernorm
  - rmsnorm
  - moe
gpu_families:
  - Ampere
  - Ada
  - Hopper
  - Blackwell
gpu_ids: []
precision:
  - bf16
  - fp16
  - fp8
bottlenecks:
  - latency
  - compute
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
  - torch_compile
required_tools:
  - run_benchmark
  - search_knowledge_base
steps:
  - benchmark eager mode first
  - sweep default, reduce-overhead, max-autotune, and max-autotune-no-cudagraphs on the same shapes
  - separate compile-time cost from steady-state performance in the report
  - treat the best compile mode as the runtime control path before kernel generation work
verification:
  - the compiled graph is valid for the representative shape set
  - compile warmup and steady-state metrics are both recorded
  - fallback eager behavior is documented for unsupported graph regions
benchmark_rubric:
  - track TTFT, inter-token latency, prefill throughput, and compile duration
  - repeat with stable and dynamic shapes if the deployment workload mixes both
failure_recovery:
  - fall back to eager mode when compile stability is poor
  - move to Triton or CuTe only after the best compile mode has been measured
artifacts_to_save:
  - benchmark_json
  - compile_notes_md
  - runtime_config
  - diff_report
runtime_adapters:
  - transformers
reference_source_ids:
  - pytorch-torch-compile
  - pytorch-cuda-graphs
backend: ""
runtimes:
  - transformers
use_cases: []
notes: []
reference_paths: []
backends: []
path: ""
---

## Steps

- benchmark eager mode first
- sweep default, reduce-overhead, max-autotune, and max-autotune-no-cudagraphs on the same shapes
- separate compile-time cost from steady-state performance in the report
- treat the best compile mode as the runtime control path before kernel generation work

## Verification

- the compiled graph is valid for the representative shape set
- compile warmup and steady-state metrics are both recorded
- fallback eager behavior is documented for unsupported graph regions

## Benchmark Rubric

- track TTFT, inter-token latency, prefill throughput, and compile duration
- repeat with stable and dynamic shapes if the deployment workload mixes both

## Failure Recovery

- fall back to eager mode when compile stability is poor
- move to Triton or CuTe only after the best compile mode has been measured
