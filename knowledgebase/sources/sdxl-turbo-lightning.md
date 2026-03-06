---
id: sdxl-turbo-lightning
kind: source
title: SDXL-Turbo and SDXL-Lightning - Few-Step Diffusion Models
category: library
url: https://huggingface.co/stabilityai/sdxl-turbo
summary: Distilled versions of SDXL that generate images in 1-4 steps. SDXL-Turbo uses adversarial distillation (ADD), SDXL-Lightning uses progressive distillation. Both are training-free to use (pre-distilled weights).
tags:
  - sdxl-turbo
  - sdxl-lightning
  - distillation
  - few-step
  - diffusion
  - adversarial-distillation
source_ids: []
operators:
  - attention
  - convolution
gpu_families:
  - Ampere
  - Ada
  - Hopper
key_contributions:
  - SDXL-Turbo (Stability AI) - adversarial distillation, 1-step generation
  - SDXL-Lightning (ByteDance) - progressive distillation, 1-4 step LoRA adapters
  - Both produce high quality images with minimal steps
  - Lightning provides LoRA weights that can be applied to any SDXL fine-tune
  - Effectively 10-50x faster than standard SDXL sampling
lightning_url: https://huggingface.co/ByteDance/SDXL-Lightning
---

## Usage (SDXL-Lightning LoRA, training-free)
```python
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
).to("cuda")
pipe.load_lora_weights(hf_hub_download("ByteDance/SDXL-Lightning", "sdxl_lightning_4step_lora.safetensors"))
pipe.fuse_lora()
pipe.scheduler = EulerDiscreteScheduler.from_config(
    pipe.scheduler.config, timestep_spacing="trailing"
)
image = pipe("a cat astronaut", num_inference_steps=4, guidance_scale=0).images[0]
# 4 steps, no CFG needed → extremely fast
```
