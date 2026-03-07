---
id: huggingface-optimum-quanto
kind: source
title: Optimum Quanto
type: official-repo
category: quantization
summary: Hugging Face Optimum Quanto provides generic tensor and module quantization utilities, including qint4, qint8, and float8 paths used by transformer and diffusion stacks.
reliability: official
review_status: reviewed
url: https://github.com/huggingface/optimum-quanto
tags:
  - optimum-quanto
  - quanto
  - qint4
  - qint8
  - transformer
  - diffusion
---

## Key Features

- Generic quantization toolkit rather than an LLM-only quantizer
- Supports `qint4`, `qint8`, and float8-style quantization types
- Relevant when a model already uses Quanto-backed `QLinear` modules and needs a lower-bit repack branch

## Notes

- Useful for multimodal and DiT models where the right language is generic weight-only quantization instead of AWQ by default
- Offline conversion and cache artifacts matter because repacking can be expensive on large models
