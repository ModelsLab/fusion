---
id: lcm-latent-consistency
kind: source
title: "Latent Consistency Models (LCM): Synthesizing High-Resolution Images with Few-Step Inference"
category: paper
url: https://arxiv.org/abs/2310.04378
summary: Distill diffusion models into latent consistency models that generate high-quality images in 1-4 steps instead of 25-50, via consistency distillation in latent space. LCM-LoRA enables training-free adaptation.
tags:
  - lcm
  - consistency-model
  - distillation
  - few-step
  - diffusion
  - lora
source_ids: []
operators:
  - attention
  - convolution
gpu_families:
  - Ampere
  - Ada
  - Hopper
key_contributions:
  - Consistency distillation in latent space (not pixel space) for efficiency
  - 2-4 step generation with quality close to 50-step DDPM
  - LCM-LoRA enables plug-and-play acceleration without full retraining
  - Works with SDXL, SD 1.5, and other latent diffusion models
  - 5-10x speedup over standard diffusion sampling
github: https://github.com/luosiallen/latent-consistency-model
---

## Usage (LCM-LoRA, training-free)
```python
from diffusers import DiffusionPipeline, LCMScheduler
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
image = pipe("a photo of a cat", num_inference_steps=4, guidance_scale=1.0).images[0]
# 4 steps instead of 50 → ~10x faster
```
