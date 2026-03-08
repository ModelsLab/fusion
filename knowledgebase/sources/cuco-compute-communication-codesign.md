---
id: cuco-compute-communication-codesign
kind: source
title: CUCo Compute And Communication Co-Design
type: paper
category: distributed-optimization
summary: CUCo proposes agentic co-design of compute and communication kernels, arguing that optimizing them independently misses end-to-end latency wins in distributed LLM systems.
reliability: research
review_status: reviewed
url: https://arxiv.org/abs/2603.02376
tags:
  - cuco
  - communication
  - compute
  - distributed
  - nccl
---

## Key Takeaways

- Jointly optimizing computation and communication can unlock wins that isolated kernel tuning misses.
- This matters most for distributed inference and training regimes with overlapping compute and collective communication.
- Fusion should keep this as a retrieval path for multi-GPU sessions instead of assuming the best single-kernel plan is the best end-to-end plan.
