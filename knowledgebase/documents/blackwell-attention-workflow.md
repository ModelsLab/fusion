---
id: blackwell-attention-workflow
kind: document
title: Blackwell Attention Workflow
category: optimization-playbook
summary: End-to-end notes for choosing cuTile, CuTe DSL, CUTLASS, and FP8 paths when optimizing Blackwell attention kernels.
support_level: recommended
reliability: curated
review_status: reviewed
tags:
  - blackwell
  - attention
  - cutile
  - cute
  - fp8
gpu_families:
  - Blackwell
workloads:
  - decode
  - prefill
operators:
  - attention
  - kv-cache
precision:
  - bf16
  - fp8
  - nvfp4
backends:
  - cutile
  - cute
  - cutlass
source_ids:
  - nvidia-cuda-tile-flash-attention
  - nvidia-blackwell-cutlass
  - nvidia-cute-dsl
path: documents/blackwell-attention-workflow.md
---

## Summary

On Blackwell, the first attention decision should not be "Triton or CUDA?" It should be "is this attention path a good fit for cuTile first?" When the workload is decode- or prefill-heavy and the GPU is Blackwell, cuTile and Blackwell-native CUTLASS capabilities should be considered before more generic kernels.

## Recommended Order

- Establish a bf16 or fp8 baseline in the real runtime first.
- Profile the existing attention path to determine whether the limiting factor is memory traffic, tensor-core utilization, or launch overhead.
- Try cuTile-oriented attention tuning first on Blackwell.
- If the problem needs a more explicit custom kernel path, move to CuTe DSL or CUTLASS/CUDA.
- Re-run correctness and end-to-end model benchmarks after each promoted candidate.

## What To Save

- Baseline benchmark and profile artifacts.
- Candidate build logs for each backend attempt.
- Correctness verification outputs with tolerances.
- End-to-end TTFT, ITL, and tokens/sec results after runtime patching.
