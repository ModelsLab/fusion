---
id: chatterbox_rtx3090_turbo
kind: example
title: Chatterbox RTX 3090 Turbo Plus torch.compile
category: model-variant
summary: Real RTX 3090 benchmark showing that the packaged Turbo model family plus torch.compile(max-autotune-no-cudagraphs) delivered the best decode speed on the host.
support_level: experimental
reliability: validated
review_status: reviewed
tags:
  - chatterbox
  - turbo
  - rtx3090
  - tts
  - checkpoint
source_ids:
  - resembleai-chatterbox
  - resembleai-chatterbox-turbo
workloads:
  - decode
operators:
  - attention
  - matmul
gpu_families:
  - Ampere
gpu_ids:
  - rtx3090
precision:
  - fp32
backend: packaged-runtime
runtimes:
  - custom-hf
use_cases:
  - tts
  - latency
  - model-selection
notes:
  - The earlier standard-model runtime patch on RTX 3090 improved the standard path to about `x_real_time_mean 1.8874`.
  - The packaged Turbo family reached `gen_s_mean 1.3457` and `x_real_time_mean 3.9337` on the same benchmark text, roughly a `2.08x` improvement in `x_real_time_mean` over the prior standard-model winner.
  - Rebuilding the environment with uv-managed Python 3.10 and `torch 2.7.1+cu128` enabled `torch.compile(max-autotune-no-cudagraphs)` on the Turbo path.
  - The final promoted winner reached `gen_s_mean 0.9021` and `x_real_time_mean 5.7932`, roughly a `3.07x` improvement in `x_real_time_mean` over the prior standard-model winner.
reference_source_ids:
  - resembleai-chatterbox
  - resembleai-chatterbox-turbo
  - pytorch-torch-compile
reference_paths:
  - scripts/chatterbox_bench.py
path: examples/chatterbox-rtx3090-turbo.md
---

## Winning Flow

- Benchmark the reference standard Chatterbox path.
- Apply low-risk runtime fixes like disabling unused attention extraction.
- Then benchmark the packaged Turbo family before attempting deeper kernel work.
- If Turbo is the new control, rebuild onto a modern PyTorch stack and test `torch.compile(mode="max-autotune-no-cudagraphs")`.

## Why It Won

- The package already shipped a different model family tuned for faster generation.
- That packaged variant produced a much larger normalized speedup than the first standard-path runtime patch alone.
- A modern compile stack on top of the winning packaged family unlocked another large speedup.
- It proved that model-family selection can dominate micro-optimizations for some TTS workloads, and compile should then be tested against that new control.

## What To Save

- Comparable artifacts for the standard baseline, standard no-attn patch, Turbo baseline, and Turbo compile winner.
- The exact prompt text, seeds, and environment versions used in the comparison.
- A note that output length and waveform content can change across model families, so normalized metrics like `rtf` and `x_real_time` matter more than raw wall clock alone.
