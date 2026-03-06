---
id: skill_blackwell_attention_tuning
kind: skill
title: Blackwell Attention Tuning
type: ""
category: attention
summary: Tune Blackwell attention kernels with CUDA Tile, cuTile, or CuTe-backed paths using correctness, profiling, and decode or prefill benchmarks.
support_level: experimental
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
  - paged-attention
gpu_families:
  - Blackwell
gpu_ids:
  - b200
  - rtx5090
  - rtxpro6000blackwellserver
  - rtxpro6000blackwellworkstation
precision:
  - bf16
  - fp16
  - fp8
  - fp4
  - nvfp4
bottlenecks:
  - compute
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
  - cuTile
  - cute_dsl
  - cuda_cutlass_cpp
required_tools:
  - plan_optimization
  - run_profile
  - run_benchmark
  - verify_cute_workspace
  - search_knowledge_base
steps:
  - establish a bf16 or fp8 baseline on the exact Blackwell target
  - capture Nsight or equivalent profiling to confirm the attention core is still hot
  - prototype CUDA Tile or cuTile attention variants before freehand CUDA
  - run correctness checks with fixed seeds and tolerance gates after every candidate
  - benchmark decode and prefill separately because the winning tile shapes can differ
verification:
  - attention outputs match the reference implementation within agreed tolerances
  - the winning kernel improves real decode or prefill metrics instead of only microbenchmarks
  - Blackwell-specific launch or precision assumptions are recorded with the artifact
benchmark_rubric:
  - track tokens per second, TTFT, inter-token latency, occupancy, and tensor utilization
  - store benchmark deltas per GPU because Blackwell workstation and datacenter parts can diverge
failure_recovery:
  - fall back to CuTe or CUTLASS templates if the CUDA Tile variant becomes too brittle
  - revert to the fp8 control path if low-precision attention introduces quality drift
artifacts_to_save:
  - kernel_source
  - benchmark_json
  - profile_json
  - correctness_report
  - winner_notes_md
runtime_adapters:
  - vllm
  - tensorrt-llm
  - sglang
reference_source_ids:
  - nvidia-cuda-tile-flash-attention
  - nvidia-blackwell-cutlass
  - rightnow-tile
backend: ""
runtimes:
  - vllm
  - tensorrt-llm
  - sglang
use_cases: []
notes: []
reference_paths: []
backends: []
path: ""
---

## Steps

- establish a bf16 or fp8 baseline on the exact Blackwell target
- capture Nsight or equivalent profiling to confirm the attention core is still hot
- prototype CUDA Tile or cuTile attention variants before freehand CUDA
- run correctness checks with fixed seeds and tolerance gates after every candidate
- benchmark decode and prefill separately because the winning tile shapes can differ

## Verification

- attention outputs match the reference implementation within agreed tolerances
- the winning kernel improves real decode or prefill metrics instead of only microbenchmarks
- Blackwell-specific launch or precision assumptions are recorded with the artifact

## Benchmark Rubric

- track tokens per second, TTFT, inter-token latency, occupancy, and tensor utilization
- store benchmark deltas per GPU because Blackwell workstation and datacenter parts can diverge

## Failure Recovery

- fall back to CuTe or CUTLASS templates if the CUDA Tile variant becomes too brittle
- revert to the fp8 control path if low-precision attention introduces quality drift
