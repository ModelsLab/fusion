---
id: huggingface-diffusers-quantization-blog
kind: source
title: Faster Diffusion Models With Quantization
type: official-blog
category: quantization
summary: Hugging Face's diffusion quantization guide compares FP8, INT8, and INT4 tradeoffs, showing that lower-bit paths can reduce memory dramatically but do not automatically become the fastest option.
reliability: official
review_status: reviewed
url: https://huggingface.co/blog/diffusers-quantization
tags:
  - diffusers
  - quantization
  - int4
  - int8
  - fp8
  - throughput
---

## Key Features

- Practical comparison of FP8, INT8, and INT4 on diffusion workloads
- Useful reminder that INT4 is often a fit-first choice rather than an automatic throughput-first choice
- Supports the policy that multimodal quantization branches must be benchmarked, not assumed

## Notes

- Use this source to justify a benchmark branch, not as a universal winner declaration
- Particularly relevant on Ampere-class consumer GPUs where memory pressure is high but native FP8 is absent
