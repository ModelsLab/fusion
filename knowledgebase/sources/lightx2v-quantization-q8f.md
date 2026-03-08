---
id: lightx2v-quantization-q8f
kind: source
title: LightX2V Quantization Guide With Q8F Support
type: official-doc
category: quantization
summary: LightX2V documents `fp8-q8f` and `int8-q8f` as first-class quantization modes for Ada-class GPUs such as RTX 40 series and L40S, alongside Triton, SGL, and other backends.
reliability: official
review_status: reviewed
url: https://github.com/ModelTC/LightX2V/blob/main/docs/EN/source/method_tutorials/quantization.md
tags:
  - lightx2v
  - wan
  - q8f
  - ada
  - fp8
  - int8
---

## Key Facts

- LightX2V lists `fp8-q8f` and `int8-q8f` as supported DiT quantization modes.
- The documented target hardware includes RTX 40 series and L40S-class GPUs.
- The same guide also exposes Triton and SGL quantization modes, which makes it a good decision surface for backend comparisons rather than a single-backend tutorial.

## Fusion Implication

- On Ada-class GPUs, LightX2V-style model paths should not default to Triton just because FP8 is available.
- If `q8f` is supported by the runtime and the model artifacts exist, Fusion should benchmark it as an early branch before deeper kernel work.
