---
id: search_loop_agentic
kind: strategy
title: Use An Iterative Agent Search Loop For Kernel Generation
type: ""
category: workflow
summary: Kernel generation should be iterative with correctness, profiling, and retained world knowledge instead of one-shot prompting.
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
source_ids:
  - k-search
  - cuda-agent
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
  - automation
priority: 78
preconditions:
  - the system can compile, verify, benchmark, and store candidate kernels
actions:
  - generate multiple candidates instead of trusting a single model output
  - persist failed and winning patterns so later searches start from a stronger prior
  - keep correctness and profiling in the loop after every generation step
metrics:
  - search iterations to first win
  - win rate over baseline
  - correctness pass rate
tradeoffs:
  - agent loops can burn a lot of compute without strong pruning and caching
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

- generate multiple candidates instead of trusting a single model output
- persist failed and winning patterns so later searches start from a stronger prior
- keep correctness and profiling in the loop after every generation step

## Tradeoffs

- agent loops can burn a lot of compute without strong pruning and caching

## Metrics

- search iterations to first win
- win rate over baseline
- correctness pass rate
