---
id: transfer-bottleneck-playbook
kind: document
title: Transfer Bottleneck Playbook
category: optimization-playbook
summary: Practical ladder for traces dominated by host-device movement, offload churn, and synchronization instead of pure GPU math throughput.
support_level: recommended
reliability: curated
review_status: reviewed
tags:
  - memory
  - offload
  - transfer
  - profiling
  - mmap
workloads:
  - text-generation
  - image-generation
  - image-editing
  - video-generation
  - audio-generation
  - multimodal
operators:
  - general
  - attention
  - transformer
  - unet
  - vae
precision:
  - fp16
  - bf16
  - int8
  - int4
  - fp8
backends:
  - runtime
  - torch_compile
  - cuda_graphs
  - triton
  - cute
  - cutlass
source_ids:
  - huggingface-diffusers-group-offloading
  - flexgen-offload-compression
  - pytorch-torch-load-mmap
  - fal-ai-flashpack
path: documents/transfer-bottleneck-playbook.md
---

## Diagnosis Order

1. Separate load and warmup from the steady-state phase.
2. Measure whether `cudaMemcpyAsync`, host-to-device transfer, or synchronization dominates the trace.
3. Identify whether the dominant movement is weights, activations, cached conditioning, or repeated CPU/GPU swaps.

## First Fixes

- keep the dominant modules resident when the working set fits
- reduce offload churn between phases or timesteps
- reuse pinned host buffers instead of reallocating them repeatedly
- prefetch the next block or stage if the runtime can overlap movement and compute
- use fit-first quantization when residency is blocked by VRAM

## mmap Guidance

- `torch.load(..., mmap=True)` is a real optimization for `.pt` and `.pth` load paths
- it is not the main answer for safetensors-native checkpoint loaders
- use it when startup CPU memory spikes matter or when a custom cache artifact is stored as `.pt`

## Cold-Start Guidance

- FlashPack is a real load-time optimization branch for PyTorch module initialization and checkpoint reloads
- it can reduce cold-start latency without changing steady-state kernel throughput
- if the profile still shows heavy host-to-device movement after warmup is removed, fix residency and transfer churn before spending time on checkpoint-packing formats

## Escalation Rule

- only move to compile and kernel work after the transfer bill is reduced enough that GPU kernels dominate the steady-state profile
