---
id: chatterbox_blackwell_turbo_compile
kind: example
title: Chatterbox Blackwell Turbo Plus torch.compile
category: runtime-patch
summary: Real RTX PRO 6000 Blackwell Server Edition benchmark showing that the packaged Turbo path plus torch.compile(max-autotune-no-cudagraphs) beat both the standard model and the uncompiled Turbo path.
support_level: experimental
reliability: validated
review_status: reviewed
tags:
  - chatterbox
  - turbo
  - blackwell
  - torch-compile
  - tts
source_ids:
  - resembleai-chatterbox
  - resembleai-chatterbox-turbo
  - pytorch-torch-compile
workloads:
  - decode
operators:
  - attention
  - matmul
gpu_families:
  - Blackwell
gpu_ids:
  - rtxpro6000blackwellserver
precision:
  - fp32
backend: torch_compile
runtimes:
  - custom-hf
use_cases:
  - tts
  - latency
  - runtime-integration
notes:
  - The host needed a Blackwell-capable PyTorch build first. `torch 2.6.0+cu124` warned that `sm_120` was unsupported, while `torch 2.7.1+cu128` ran correctly.
  - The winning branch used the packaged Turbo model family and `torch.compile(mode="max-autotune-no-cudagraphs")` with `torch.set_float32_matmul_precision("high")`.
  - Baseline standard model on the corrected stack measured `gen_s_mean 1.0528` and `x_real_time_mean 4.0398`.
  - Standard no-attn plus compile measured `gen_s_mean 0.6826` and `x_real_time_mean 6.2310`.
  - Turbo baseline measured `gen_s_mean 0.6375` and `x_real_time_mean 8.3025`.
  - Turbo plus compile measured `gen_s_mean 0.5402` and `x_real_time_mean 9.6739`.
reference_source_ids:
  - resembleai-chatterbox-turbo
  - pytorch-torch-compile
reference_paths:
  - scripts/chatterbox_bench.py
path: examples/chatterbox-blackwell-turbo-compile.md
---

## Winning Flow

- Upgrade the environment to a Blackwell-capable PyTorch build before benchmarking anything on `sm_120`.
- Benchmark the standard packaged model family first.
- Benchmark the packaged Turbo family next.
- Only after that apply `torch.compile(mode="max-autotune-no-cudagraphs")` to the Turbo transformer path.

## Why It Won

- Turbo was a stronger control path than the standard model family on this workload.
- The no-cudagraph compile mode avoided the graph-overwrite failure that hit `reduce-overhead` on the standard decode loop.
- `torch.set_float32_matmul_precision("high")` let the compiled path use a better steady-state matmul configuration without forcing a lower-precision model conversion.

## What To Save

- The exact PyTorch and CUDA wheel versions used on Blackwell.
- Separate artifacts for standard baseline, standard compile, turbo baseline, and turbo compile.
- Compile failures for rejected modes such as `reduce-overhead`, because they explain why the promoted mode was chosen.
