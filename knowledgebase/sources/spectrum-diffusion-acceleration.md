---
id: spectrum-diffusion-acceleration
kind: source
title: "Spectrum: Training-Free Diffusion Acceleration via Chebyshev Polynomial Feature Forecasting"
category: paper
url: https://arxiv.org/abs/2603.01623
summary: Training-free method to accelerate diffusion model inference by forecasting latent features using Chebyshev polynomials, achieving up to 4.79x speedup on FLUX.1 and 4.67x on Wan2.1-14B without quality loss.
tags:
  - training-free
  - diffusion
  - acceleration
  - chebyshev
  - feature-caching
  - flux
  - video-generation
source_ids: []
operators:
  - attention
  - general
gpu_families:
  - Ampere
  - Ada
  - Hopper
key_contributions:
  - Global feature forecasting using Chebyshev polynomial approximation (vs local step-skipping)
  - Error bound that does NOT compound with step size (unlike prior feature caching methods)
  - Online ridge regression to fit coefficients during sampling
  - Works on FLUX.1, HunyuanVideo, Wan2.1-14B without retraining
  - 3.5-4.79x speedup with maintained or improved sample quality
  - CVPR 2026
github: https://github.com/hanjq17/Spectrum
---

## How It Works
1. During early denoising steps: run full model, collect feature snapshots
2. Fit Chebyshev polynomial coefficients to feature trajectory via ridge regression
3. For subsequent steps: forecast features from polynomial → skip expensive model forward pass
4. Unlike DeepCache/TGATE (local approximation), Spectrum models the GLOBAL trajectory
