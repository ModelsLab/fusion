---
id: chatterbox_rtx3090_no_attn_decode
kind: example
title: Chatterbox RTX 3090 No-Attention Decode Patch
category: runtime-patch
summary: Real RTX 3090 benchmark showing that gating unused attention extraction off in English Chatterbox decode preserves identical seeded output while improving decode speed.
support_level: experimental
reliability: validated
review_status: reviewed
tags:
  - chatterbox
  - tts
  - rtx3090
  - attention
  - sdpa
source_ids:
  - resembleai-chatterbox
workloads:
  - decode
operators:
  - attention
gpu_families:
  - Ampere
gpu_ids:
  - rtx3090
precision:
  - fp32
backend: transformers
runtimes:
  - custom-hf
use_cases:
  - tts
  - latency
  - runtime-integration
notes:
  - Final resolved environment used torch 2.6.0+cu124 on an RTX 3090 host whose driver exposed CUDA 13.1.
  - The winning patch preserved the seeded output hash `53625f7387173de0edf89f3f164945aa3a225d4f2ee3481a9a50b690cfc989eb`.
  - Deterministic single-run generation improved from 2.9289s to 2.7418s for the same 3.4s audio output.
  - Warm steady-state generation improved to 1.8014s mean for 3.4s of audio, or about 1.89x real time.
reference_source_ids:
  - resembleai-chatterbox
reference_paths:
  - package:chatterbox/models/t3/t3.py
  - package:chatterbox/models/t3/inference/t3_hf_backend.py
path: examples/chatterbox-rtx3090-no-attn-decode.md
---

## Patch Shape

- Keep the existing decode loop and cache behavior.
- Change the backend forward path so `output_attentions` is only enabled when an alignment analyzer is actually attached.
- Leave multilingual or analyzer-enabled paths untouched.

## Why It Won

- The English Chatterbox path did not consume attention maps during normal decode.
- The existing implementation still requested them, which triggered a slower attention path.
- Removing that unnecessary request produced an identical seeded waveform hash and a measurable decode speedup.

## What To Save

- Before and after benchmark artifacts with fixed RNG seeds.
- Output hashes or tolerance checks proving that the optimization did not change the seeded result.
- Host details: GPU, driver, resolved PyTorch build, and model package version.
