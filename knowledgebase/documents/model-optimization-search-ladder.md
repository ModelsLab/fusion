---
id: model-optimization-search-ladder
kind: document
title: Model Optimization Search Ladder
category: optimization-playbook
summary: Ordered workflow for exhausting low-hanging runtime, precision, and kernel strategies before locking a winning model optimization candidate.
support_level: recommended
reliability: curated
review_status: reviewed
tags:
  - optimization
  - search
  - fallback
  - quantization
  - kernels
workloads:
  - decode
  - prefill
  - serving
operators:
  - attention
  - matmul
  - layernorm
  - kv-cache
precision:
  - fp16
  - bf16
  - int8
  - int4
  - fp8
  - nvfp4
backends:
  - runtime
  - torch_compile
  - cuda_graphs
  - triton
  - cute
  - cutlass
source_ids:
  - pytorch-torch-compile
  - pytorch-cuda-graphs
  - nvidia-cute-dsl
  - nvidia-blackwell-cutlass
  - awq-activation-aware-weight-quantization
path: documents/model-optimization-search-ladder.md
---

## Goal

Do not stop at the first positive delta. Build a candidate ladder, test every applicable family, keep the current best candidate, and fall back to it whenever a later experiment regresses or fails.

## Required Order

1. Establish a seeded correctness and performance baseline.
2. Build an applicability matrix for the current GPU, model, runtime, and workload.
3. Test runtime-only low-hanging fruit first.
4. Test precision, checkpoint, and quantization variants that are actually supported by the target GPU.
5. Test compile and graph-capture paths if the environment supports them.
6. Only then spend time on custom kernels and backend rewrites.
7. Rank all passing candidates and promote the current best.

## Low-Hanging Runtime Pass

- remove unnecessary debug or analysis paths from hot decode loops
- choose the fastest valid attention implementation for the active runtime
- stabilize allocator and cache behavior before deeper kernel work
- verify that logging, progress bars, and optional hooks are not left on in production paths

## Precision And Quantization Pass

- On Ampere and Ada, test fp16, bf16, AWQ or INT4, and KV-cache changes before chasing native FP8 or NVFP4.
- On Hopper, test FP8 before more speculative low-precision branches.
- On Blackwell, FP8 comes before NVFP4 or block-scaled FP4.
- Skip unsupported precision branches explicitly and record why they were not tested.

## Compile And Graph Pass

- Test `torch.compile` only when the environment can support it cleanly.
- If compile or inductor dependencies are missing, record the blocker and move on.
- Benchmark CUDA Graphs after shapes and buffers are stable.

## Kernel Pass

- Triton first for memory-bound fused paths and rapid iteration.
- CuTe DSL or CUTLASS first for NVIDIA-specific tensor-core-heavy kernels.
- CUDA C++ when DSLs stop expressing the winning shape or integration requires it.

## Promotion Rule

- Keep one current best candidate at all times.
- If a new candidate is slower, less stable, or numerically worse, fall back immediately.
- End the search only after every applicable family has been tested, rejected with evidence, or blocked by the environment.
