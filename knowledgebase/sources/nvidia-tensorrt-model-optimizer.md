---
id: nvidia-tensorrt-model-optimizer
kind: source
title: NVIDIA TensorRT Model Optimizer
type: official-doc
category: quantization
summary: NVIDIA Model Optimizer supports post-training quantization and checkpoint conversion flows, including FP8 export paths for TensorRT-LLM and related serving stacks.
reliability: official
review_status: reviewed
url: https://github.com/NVIDIA/TensorRT-Model-Optimizer
tags:
  - nvidia
  - modelopt
  - tensorrt-llm
  - quantization
  - fp8
  - nvfp4
---

## Key Features

- Offline quantization from bf16 or fp16 checkpoints when no packaged low-precision checkpoint exists
- Calibration and export flows for TensorRT-LLM deployment
- FP8, FP4, and other post-training quantization recipes on NVIDIA GPU stacks
- Useful as a search branch before custom kernel work on Hopper or Blackwell
