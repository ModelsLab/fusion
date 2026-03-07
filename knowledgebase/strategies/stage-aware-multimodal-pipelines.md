---
id: stage_aware_multimodal_pipelines
kind: strategy
title: Treat Multimodal Pipelines As Stage-Aware Systems
category: workflow
summary: Diffusion, DiT, video, and audio-video systems should be optimized stage by stage instead of as one monolithic benchmark target.
support_level: recommended
reliability: curated
review_status: reviewed
workloads:
  - sampling
  - refinement
  - serving
operators:
  - attention
  - matmul
  - vae
  - scheduler
goals:
  - throughput
  - latency
  - memory
  - quality
priority: 92
actions:
  - split the pipeline into meaningful stages such as text encoder, core transformer or unet, vae, scheduler, and upscaler
  - benchmark and attribute hotspots per stage before writing custom kernels
  - allow one stage to stay higher precision if it protects output quality
  - keep a current best pipeline configuration and revert per-stage regressions immediately
metrics:
  - normalized primary metric per task
  - stage latency
  - peak memory
  - quality checks
tradeoffs:
  - stage-aware optimization adds orchestration overhead
  - the fastest per-stage settings may not compose into the best full pipeline result
source_ids:
  - spectrum-diffusion-acceleration
  - sdxl-turbo-lightning
---

## Actions

- split the pipeline into meaningful stages such as text encoder, core transformer or unet, vae, scheduler, and upscaler
- benchmark and attribute hotspots per stage before writing custom kernels
- allow one stage to stay higher precision if it protects output quality
- keep a current best pipeline configuration and revert per-stage regressions immediately
