---
id: skill_cuda_cutlass_kernel_generation
kind: skill
title: CUDA C++ Or CUTLASS Kernel Generation
type: ""
category: kernel
summary: Generate or repair CUDA C++ or CUTLASS kernels when the hot path needs lower-level control than Triton can provide.
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
precision:
  - bf16
  - fp16
  - fp8
  - int8
  - int4
  - nvfp4
bottlenecks:
  - compute
  - latency
goals:
  - throughput
  - latency
priority: 0
preconditions: []
actions: []
metrics: []
tradeoffs: []
preferred_backends:
  - cuda_cutlass_cpp
  - cute_dsl
required_tools:
  - read_file
  - write_file
  - replace_in_file
  - run_command
  - run_benchmark
  - search_knowledge_base
steps:
  - start from an official CUTLASS or CuTe template instead of writing raw CUDA from zero
  - specialize tile shapes and epilogues for the actual GPU family
  - compile and run a small correctness harness before any benchmark sweep
  - only integrate into the runtime after the kernel beats the current control path
verification:
  - the kernel compiles with the target CUDA toolchain
  - reference outputs pass tolerance checks on representative shapes
  - the kernel win survives an end-to-end benchmark instead of only a microkernel benchmark
benchmark_rubric:
  - track kernel latency, tensor core utilization, and end-to-end tokens per second
  - compare against Triton and runtime-native baselines when possible
failure_recovery:
  - fallback to CuTe DSL if the C++ template surface becomes too slow to iterate on
  - fallback to Triton for memory-bound fusions that do not justify CUTLASS complexity
artifacts_to_save:
  - kernel_source
  - build_log
  - correctness_report
  - benchmark_json
  - patch_diff
runtime_adapters:
  - vllm
  - tensorrt-llm
  - transformers
  - sglang
reference_source_ids:
  - nvidia-cutlass-overview
  - nvidia-blackwell-cutlass
  - cuda-agent
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

- start from an official CUTLASS or CuTe template instead of writing raw CUDA from zero
- specialize tile shapes and epilogues for the actual GPU family
- compile and run a small correctness harness before any benchmark sweep
- only integrate into the runtime after the kernel beats the current control path

## Verification

- the kernel compiles with the target CUDA toolchain
- reference outputs pass tolerance checks on representative shapes
- the kernel win survives an end-to-end benchmark instead of only a microkernel benchmark

## Benchmark Rubric

- track kernel latency, tensor core utilization, and end-to-end tokens per second
- compare against Triton and runtime-native baselines when possible

## Failure Recovery

- fallback to CuTe DSL if the C++ template surface becomes too slow to iterate on
- fallback to Triton for memory-bound fusions that do not justify CUTLASS complexity
