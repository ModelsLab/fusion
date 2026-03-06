---
id: deepcache-diffusion
kind: source
title: "DeepCache: Accelerating Diffusion Models via Feature Caching"
category: paper
url: https://arxiv.org/abs/2312.00858
summary: Training-free acceleration for diffusion models by caching and reusing high-level features from the U-Net decoder across adjacent denoising steps, achieving 2.3x speedup on Stable Diffusion with minimal quality loss.
tags:
  - training-free
  - diffusion
  - feature-caching
  - stable-diffusion
  - u-net
source_ids: []
operators:
  - attention
  - convolution
gpu_families:
  - Ampere
  - Ada
  - Hopper
key_contributions:
  - Caches U-Net decoder features at high levels (low resolution) where features change slowly
  - Reuses cached features for N-1 out of N steps, only computing full U-Net every Nth step
  - 2.3x speedup on SDXL, 10.6x with aggressive caching
  - No retraining, works as drop-in acceleration
github: https://github.com/horseee/DeepCache
---

## Usage
```python
pip install DeepCache
from DeepCache import DeepCacheSDHelper
helper = DeepCacheSDHelper(pipe=pipe)
helper.set_params(cache_interval=3, cache_branch_id=0)
helper.enable()
image = pipe(prompt).images[0]
```
