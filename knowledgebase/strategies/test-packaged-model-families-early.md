---
id: test_packaged_model_families_early
kind: strategy
title: Test Packaged Model Families And Checkpoints Early
category: model-variant
summary: If a package or upstream repo already ships turbo, distilled, quantized, or hardware-targeted checkpoints, benchmark them before investing in lower-level kernel work.
support_level: recommended
reliability: curated
review_status: reviewed
tags:
  - checkpoint
  - turbo
  - distilled
  - fp8
  - search-ladder
workloads:
  - decode
  - prefill
  - serving
operators:
  - all
gpu_families:
  - Ampere
  - Ada
  - Hopper
  - Blackwell
precision:
  - fp32
  - fp16
  - bf16
  - int4
  - fp8
  - nvfp4
bottlenecks:
  - latency
  - throughput
  - mixed
goals:
  - latency
  - throughput
  - cost
priority: 93
preconditions:
  - The package, model card, or upstream repo already exposes alternate model families, checkpoints, or runtime flavors.
  - A reproducible benchmark harness exists so variants can be compared under the same text, prompt, and seed policy.
actions:
  - Inspect the package and model card for turbo, distilled, quantized, or hardware-targeted checkpoints before patching kernels.
  - Benchmark each packaged family as its own candidate with the same workload and metrics.
  - Treat the best packaged variant as the new control before testing compile, quantization, or custom kernel paths.
metrics:
  - gen_s
  - rtf
  - x_real_time
  - tokens/sec
tradeoffs:
  - Faster packaged variants can change output length, quality, or model behavior and must be benchmarked as separate candidates.
  - A packaged variant can make lower-level optimization work unnecessary for some workloads.
source_ids:
  - resembleai-chatterbox
  - resembleai-chatterbox-turbo
  - pytorch-torch-compile
path: strategies/test-packaged-model-families-early.md
---

## Summary

Do not treat the default checkpoint as the only model worth optimizing. If the package already ships a turbo, distilled, or hardware-targeted path, benchmark it early and compare it against the current best candidate using normalized metrics like `rtf`, `x_real_time`, or tokens per second.

## Recommended Flow

- Establish the standard-model baseline first.
- Discover packaged variants from the package source, model cards, or upstream repo docs.
- Register each variant as a separate candidate.
- Keep the best packaged variant as the control before deeper runtime patching or kernel generation.
