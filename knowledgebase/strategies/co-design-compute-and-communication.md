---
id: co_design_compute_and_communication
kind: strategy
title: Co-Design Compute And Communication In Multi-GPU Runs
category: distributed-optimization
summary: In distributed inference or training, profile and optimize NCCL or communication overlap together with kernel execution instead of treating communication as a fixed background cost.
support_level: experimental
reliability: curated
review_status: reviewed
tags:
  - distributed
  - nccl
  - communication
  - profiling
  - multi-gpu
workloads:
  - distributed-inference
  - distributed-training
  - moe
operators:
  - collectives
  - allreduce
  - alltoall
  - matmul
  - attention
gpu_families:
  - Ampere
  - Ada
  - Hopper
  - Blackwell
bottlenecks:
  - communication
  - mixed
goals:
  - latency
  - throughput
priority: 78
preconditions:
  - the run spans multiple GPUs or includes communication-heavy stages
actions:
  - collect profile evidence for communication and compute on the same timeline before proposing kernels
  - attribute stalls to compute, communication, or overlap failure instead of only ranking kernels by isolated microbenchmarks
  - treat fused compute plus communication schedules as valid candidates when the distributed runtime allows them
metrics:
  - end-to-end latency
  - tokens/sec
  - overlap ratio
  - communication time share
tradeoffs:
  - distributed tuning can increase search space dramatically
  - a kernel that wins in isolation can lose once communication overlap changes
source_ids:
  - cuco-compute-communication-codesign
path: strategies/co-design-compute-and-communication.md
---

## Summary

For multi-GPU optimization, profile first and optimize the compute plus communication schedule together. This is the distributed analogue of Fusion's profiler-first single-GPU rule.
