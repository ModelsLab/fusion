---
id: task-aware-benchmark-quality-harnesses
kind: document
title: Task-Aware Benchmark And Quality Harnesses
category: optimization-playbook
summary: Build workload-specific benchmark and quality harnesses instead of forcing every model into a token-per-second evaluation loop.
support_level: recommended
reliability: curated
review_status: reviewed
tags:
  - benchmarking
  - quality
  - multimodal
  - diffusion
  - video
  - audio
workloads:
  - decode
  - serving
  - denoise
  - sampling
  - refinement
precision:
  - bf16
  - fp16
  - fp8
source_ids:
  - pytorch-torch-compile
  - sdxl-turbo-lightning
  - spectrum-diffusion-acceleration
path: documents/task-aware-benchmark-quality-harnesses.md
---

## Principle

Do not force every optimization task into `tokens_per_sec`.

- Text generation: `tokens_per_sec`, `ttft_ms`, `itl_ms`, memory, quality drift
- Image generation or editing: `images_per_sec`, latency, steps per second, memory, edit fidelity
- Video generation: frames per second, clips per hour, latency, temporal consistency, memory
- Audio generation: `x_real_time`, `rtf`, latency, WER or MOS-style quality checks

## Required Harness Structure

1. A benchmark command with repeated warmup and measured runs.
2. A quality command or metric bundle for the output modality.
3. A normalized primary metric for ranking candidates within that task.
4. Explicit failure thresholds so low-quality fast paths do not get promoted.

## Anti-Pattern

- Do not compare model families using only raw wall time when they emit different numbers of tokens, images, frames, or audio seconds.
- Do not treat missing quality metrics as success for multimodal workloads.
