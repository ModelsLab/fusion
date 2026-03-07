---
id: pytorch-torch-load-mmap
kind: source
title: "PyTorch torch.load mmap Support"
type: official-doc
category: checkpoint-loading
summary: PyTorch exposes `mmap=True` on `torch.load` so tensor storages can be memory-mapped from disk instead of copied eagerly into CPU memory. This mainly matters for `.pt` or pickle-style checkpoints, not safetensors-first loaders.
reliability: official
review_status: reviewed
url: https://pytorch.org/docs/stable/generated/torch.load.html
tags:
  - pytorch
  - torch-load
  - mmap
  - checkpoint-loading
  - cpu-memory
---

## Applicability

- useful when the project loads `.pt` or `.pth` checkpoints through `torch.load`
- can reduce eager CPU copies and shorten peak host-memory spikes during load
- does not directly optimize safetensors-native load paths

## Fusion Notes

- treat `mmap=True` as a load-path experiment, not a steady-state kernel optimization
- validate whether the project actually uses `torch.load` on the hot checkpoint path before testing it
