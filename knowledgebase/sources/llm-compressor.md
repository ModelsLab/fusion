---
id: llm-compressor
kind: source
title: LLM Compressor
type: official-doc
category: quantization
summary: vLLM's LLM Compressor supports FP8, FP4, INT8, and INT4 quantization, calibration, and model export flows for deployment on vLLM-compatible runtimes.
reliability: official
review_status: reviewed
url: https://docs.vllm.ai/usage/quantization/llm_compressor/
tags:
  - vllm
  - llm-compressor
  - quantization
  - fp8
  - fp4
  - int4
---

## Key Features

- One-shot quantization and calibration for vLLM-ready model artifacts
- FP8, FP4, INT8, and INT4 support with runtime-oriented compression schemes
- Useful when a model does not ship a packaged FP8 checkpoint but the target runtime can consume a converted artifact
