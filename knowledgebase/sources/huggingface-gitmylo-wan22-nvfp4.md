---
id: huggingface-gitmylo-wan22-nvfp4
kind: source
title: GitMylo WAN 2.2 NVFP4 Experiment
type: community-model-card
category: quantization
summary: Community NVFP4 WAN 2.2 checkpoints exist on Hugging Face, including mixed NVFP4 and FP8 layer choices intended to reduce quality loss, but they are experimental and runtime-dependent.
reliability: community
review_status: reviewed
url: https://huggingface.co/GitMylo/Wan_2.2_nvfp4
tags:
  - wan
  - nvfp4
  - fp8
  - blackwell
  - quantization
---

## Key Features

- Shows that WAN-family community experiments are already exploring NVFP4-style paths
- Suggests Blackwell-era search branches should include packaged FP4 or mixed FP4/FP8 artifacts where available

## Notes

- Treat as an experimental lead, not an authoritative production recipe
- Useful for Fusion retrieval because it distinguishes "native low-precision branch exists in the ecosystem" from "the current runtime can use it today"
