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
3. Test packaged model-family, checkpoint, and runtime-flavor variants early.
4. Test runtime-only low-hanging fruit next.
5. Test precision, checkpoint, and quantization variants that are actually supported by the target GPU.
6. Test compile and graph-capture paths if the environment supports them.
7. Only then spend time on custom kernels and backend rewrites.
8. Rank all passing candidates and promote the current best.

## Model Family And Checkpoint Pass

- Check whether the package or upstream repo already ships faster variants such as turbo, distilled, or smaller decode models.
- Test hardware-targeted checkpoints like FP8 or INT4 only when they are truly compatible with the target GPU and runtime.
- If no packaged FP8 checkpoint exists, treat FP8 conversion as its own branch instead of skipping it outright. Try runtime-supported library flows such as TensorRT Model Optimizer, Transformer Engine, torchao float8, or llm-compressor before custom kernel work.
- Treat alternate packaged runtimes as first-class candidates, not as footnotes after kernel work.

## Low-Hanging Runtime Pass

- remove unnecessary debug or analysis paths from hot decode loops
- choose the fastest valid attention implementation for the active runtime
- stabilize allocator and cache behavior before deeper kernel work
- verify that logging, progress bars, and optional hooks are not left on in production paths

## Precision And Quantization Pass

- On Ampere and Ada, test fp16, bf16, AWQ or INT4, and KV-cache changes before chasing native FP8 or NVFP4.
- On Hopper, test FP8 before more speculative low-precision branches. If no packaged FP8 artifact exists, attempt calibrated FP8 conversion with Model Optimizer, Transformer Engine, torchao, or llm-compressor when the runtime supports it.
- On Blackwell, FP8 comes before NVFP4 or block-scaled FP4, and synthesized FP8 should be evaluated before more custom kernels when the software stack supports it.
- Treat FP8 conversion as an explicit candidate family with calibration, quality validation, and fallback rules, not as a note that depends on pre-quantized checkpoints.
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
- Compare steady-state normalized metrics like `rtf`, `x_real_time`, or tokens per second when model families or output lengths differ.
- Keep compile, download, and warmup overhead separate from steady-state generation speed.
- If a new candidate is slower, less stable, or numerically worse, fall back immediately.
- End the search only after every applicable family has been tested, rejected with evidence, or blocked by the environment.
