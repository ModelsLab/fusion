---
id: paged_kv_cache
kind: strategy
title: Use Paged KV Cache And Paged Attention Layouts
type: ""
category: memory
summary: Paged KV layouts are a first-class serving optimization for decode-heavy transformer inference.
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
  - vllm-paged-attention
  - pagedattention-paper
  - flashinfer-docs
workloads:
  - decode
  - serving
operators:
  - attention
  - kv-cache
  - paged-attention
gpu_families: []
gpu_ids: []
precision:
  - bf16
  - fp16
  - fp8
  - int8
  - int4
bottlenecks:
  - memory
  - latency
  - mixed
goals:
  - throughput
  - latency
  - memory
priority: 95
preconditions:
  - serving stack can manage block tables or page metadata
  - decode is a meaningful part of the latency budget
actions:
  - move KV allocation to paged blocks instead of contiguous per-request slabs
  - match the attention kernel to the chosen page and block layout
  - re-measure memory fragmentation and end-to-end decode throughput
metrics:
  - KV-cache footprint
  - tokens/sec
  - memory fragmentation
tradeoffs:
  - more runtime metadata and allocator complexity
  - kernel layout must align with the page format
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

- move KV allocation to paged blocks instead of contiguous per-request slabs
- match the attention kernel to the chosen page and block layout
- re-measure memory fragmentation and end-to-end decode throughput

## Tradeoffs

- more runtime metadata and allocator complexity
- kernel layout must align with the page format

## Metrics

- KV-cache footprint
- tokens/sec
- memory fragmentation
