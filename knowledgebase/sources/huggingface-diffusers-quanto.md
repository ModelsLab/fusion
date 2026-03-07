---
id: huggingface-diffusers-quanto
kind: source
title: Diffusers Quanto Quantization
type: official-doc
category: quantization
summary: Hugging Face Diffusers documents Quanto-based quantization flows for transformer components, including INT4, INT8, and FP8 weight quantization for diffusion and multimodal pipelines.
reliability: official
review_status: reviewed
url: https://huggingface.co/docs/diffusers/en/quantization/quanto
tags:
  - diffusers
  - quanto
  - quantization
  - int4
  - fp8
  - multimodal
---

## Key Features

- Official Diffusers path for `optimum-quanto` quantization of transformer modules
- Supports weight quantization to `int4`, `int8`, and `float8`
- Relevant for diffusion, DiT, and multimodal transformer pipelines where AWQ-specific tooling may not apply
- Useful when the goal is lower VRAM footprint or making a larger workload fit on a single GPU

## Notes

- Diffusers documents this as a transformer quantization path, not an LLM-only optimization
- The practical win can be memory fit, throughput, or both, depending on kernel support and dequant overhead
