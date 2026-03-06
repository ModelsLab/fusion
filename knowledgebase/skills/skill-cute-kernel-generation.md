---
id: skill_cute_kernel_generation
kind: skill
title: CuTe DSL Kernel Generation
type: ""
category: kernel
summary: Author, compile, verify, and benchmark CuTe DSL kernels for tensor-core-heavy hot paths on NVIDIA GPUs.
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
  - gemm
  - attention
  - moe
gpu_families:
  - Ampere
  - Hopper
  - Blackwell
gpu_ids:
  - a100
  - h100
  - h200
  - b200
  - rtx5090
precision:
  - bf16
  - fp16
  - fp8
  - int8
  - int4
  - nvfp4
bottlenecks:
  - compute
goals:
  - throughput
  - latency
priority: 0
preconditions: []
actions: []
metrics: []
tradeoffs: []
preferred_backends:
  - cute_dsl
required_tools:
  - init_cute_workspace
  - build_cute_workspace
  - verify_cute_workspace
  - run_benchmark
  - search_knowledge_base
steps:
  - start from a CuTe workspace and pin the target GPU family
  - generate the kernel from a known-good GEMM or attention template instead of freehand code
  - compile immediately after each candidate edit
  - verify against a stable reference with explicit tolerances before benchmarking
  - benchmark the passing variants on representative shapes and workloads
verification:
  - the CuTe workspace builds cleanly
  - reference outputs match within tolerance across representative problem sizes
  - the winning variant improves end-to-end metrics on the target runtime path
benchmark_rubric:
  - record end-to-end throughput, latency, tensor utilization, and problem size coverage
  - keep separate variant families for prefill and decode if shape behavior diverges
failure_recovery:
  - drop back to Triton for simple elementwise or reduction-heavy fusions
  - drop to CUTLASS or CUDA C++ if the CuTe abstraction cannot express the final kernel shape
artifacts_to_save:
  - workspace_dir
  - kernel_source
  - build_log
  - correctness_report
  - benchmark_json
runtime_adapters:
  - vllm
  - tensorrt-llm
  - transformers
  - sglang
reference_source_ids:
  - nvidia-cute-dsl
  - nvidia-cutlass-overview
  - nvidia-blackwell-cutlass
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

- start from a CuTe workspace and pin the target GPU family
- generate the kernel from a known-good GEMM or attention template instead of freehand code
- compile immediately after each candidate edit
- verify against a stable reference with explicit tolerances before benchmarking
- benchmark the passing variants on representative shapes and workloads

## Verification

- the CuTe workspace builds cleanly
- reference outputs match within tolerance across representative problem sizes
- the winning variant improves end-to-end metrics on the target runtime path

## Benchmark Rubric

- record end-to-end throughput, latency, tensor utilization, and problem size coverage
- keep separate variant families for prefill and decode if shape behavior diverges

## Failure Recovery

- drop back to Triton for simple elementwise or reduction-heavy fusions
- drop to CUTLASS or CUDA C++ if the CuTe abstraction cannot express the final kernel shape
