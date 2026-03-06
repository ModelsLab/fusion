---
id: skill_nsight_diagnosis
kind: skill
title: Nsight-Based Bottleneck Diagnosis
type: ""
category: profiling
summary: Use Nsight-derived counters to distinguish memory-bound from compute-bound hot paths before generating kernels.
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
  - gemm
  - layernorm
  - rmsnorm
  - kv-cache
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
  - int8
  - int4
bottlenecks:
  - memory
  - compute
  - latency
  - mixed
goals:
  - throughput
  - latency
  - automation
priority: 0
preconditions: []
actions: []
metrics: []
tradeoffs: []
preferred_backends:
  - profiling
required_tools:
  - run_profile
  - search_knowledge_base
  - detect_environment
steps:
  - capture a profile on a representative workload instead of an empty synthetic run
  - classify the hot path as memory-bound, compute-bound, or launch-bound from counters and traces
  - map the hot kernel back to the runtime operator or model layer if possible
  - feed the diagnosis into strategy and skill selection before writing new kernels
verification:
  - the profiled workload matches the real production shape mix
  - the diagnosis is backed by counters such as HBM throughput, occupancy, or tensor utilization
  - the resulting optimization branch explains the measured bottleneck
benchmark_rubric:
  - store both raw profiler outputs and normalized notes
  - record operator attribution, not just kernel names
failure_recovery:
  - fall back to wall-clock benchmark comparisons if detailed counters are unavailable
  - re-profile after every major kernel or precision change because the bottleneck can move
artifacts_to_save:
  - profile_json
  - profile_raw
  - bottleneck_notes_md
  - operator_map_json
runtime_adapters:
  - vllm
  - tensorrt-llm
  - transformers
  - sglang
reference_source_ids:
  - nsight-python
  - nvidia-cuda-programming-guide
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

- capture a profile on a representative workload instead of an empty synthetic run
- classify the hot path as memory-bound, compute-bound, or launch-bound from counters and traces
- map the hot kernel back to the runtime operator or model layer if possible
- feed the diagnosis into strategy and skill selection before writing new kernels

## Verification

- the profiled workload matches the real production shape mix
- the diagnosis is backed by counters such as HBM throughput, occupancy, or tensor utilization
- the resulting optimization branch explains the measured bottleneck

## Benchmark Rubric

- store both raw profiler outputs and normalized notes
- record operator attribution, not just kernel names

## Failure Recovery

- fall back to wall-clock benchmark comparisons if detailed counters are unavailable
- re-profile after every major kernel or precision change because the bottleneck can move
