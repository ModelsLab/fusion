---
id: disaggregated_prefill_decode
kind: strategy
title: Disaggregated Prefill And Decode Serving
category: serving
summary: Separate prefill and decode onto different GPU pools to optimize each independently and prevent interference.
support_level: experimental
source_ids: []
workloads:
  - serving
  - prefill
  - decode
gpu_families:
  - Hopper
  - Blackwell
precision:
  - fp8
  - fp16
bottlenecks:
  - mixed
goals:
  - throughput
  - latency
priority: 65
preconditions:
  - high-throughput serving with mixed workload
  - sufficient GPU resources for separate pools
  - KV cache transfer bandwidth is not the bottleneck
actions:
  - profile interference between prefill and decode in mixed serving
  - evaluate if TTFT or decode latency is suffering from contention
  - deploy prefill pool with compute-optimized settings (max-autotune)
  - deploy decode pool with memory-optimized settings (CUDA graphs, quantized KV)
  - implement KV cache transfer protocol between pools
metrics:
  - TTFT improvement
  - decode latency stability (P99)
  - throughput per dollar
tradeoffs:
  - more complex infrastructure
  - KV cache transfer adds latency
  - need 2x GPU management
---

## When To Use
- Production serving at scale (100+ req/sec)
- When decode P99 latency is suffering due to prefill interference
- When prefill and decode have very different optimal configurations

## Key Systems
- Splitwise (Microsoft), DistServe, Mooncake (Moonshot AI)
