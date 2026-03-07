---
id: deepspeed-zero-inference
kind: source
title: DeepSpeed ZeRO-Inference
type: official-project
category: inference-systems
summary: ZeRO-Inference is an official DeepSpeed inference system that focuses on serving very large models by combining quantization, KV/offload techniques, and explicit control over data movement.
reliability: official
review_status: reviewed
url: https://www.deepspeed.ai/2022/09/09/zero-inference.html
tags:
  - deepspeed
  - zero-inference
  - offload
  - quantization
  - inference
---

## Fusion Notes

- most directly relevant when the model cannot stay resident on GPU and transfer scheduling becomes the dominant cost
- the main transferable lesson is that compression and movement policy should be optimized together instead of independently
