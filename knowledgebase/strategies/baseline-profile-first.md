---
id: baseline_profile_first
kind: strategy
title: Profile Before Rewriting Kernels
type: ""
category: workflow
summary: Capture a clean baseline so later kernel work is driven by measured bottlenecks instead of intuition.
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
source_ids:
  - nsight-python
  - pytorch-cuda-graphs
workloads:
  - all
operators:
  - all
gpu_families: []
gpu_ids: []
precision:
  - all
bottlenecks:
  - all
goals:
  - throughput
  - latency
  - memory
priority: 100
preconditions:
  - have a reproducible workload with stable batch size and context length
  - record both tokens/sec and per-token latency
actions:
  - measure a no-change baseline with Nsight Compute or Nsight Systems before changing code
  - track achieved occupancy, DRAM throughput, tensor utilization, and kernel launch overhead
  - keep the exact same workload for before and after comparisons
metrics:
  - tokens/sec
  - TTFT
  - inter-token latency
  - dram__throughput
  - sm__throughput
tradeoffs:
  - profiling adds upfront time but saves wasted kernel iterations later
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

- measure a no-change baseline with Nsight Compute or Nsight Systems before changing code
- track achieved occupancy, DRAM throughput, tensor utilization, and kernel launch overhead
- keep the exact same workload for before and after comparisons

## Tradeoffs

- profiling adds upfront time but saves wasted kernel iterations later

## Metrics

- tokens/sec
- TTFT
- inter-token latency
- dram__throughput
- sm__throughput
