---
id: resembleai-chatterbox-turbo
kind: source
title: Chatterbox Turbo Hugging Face Model Card
type: official
category: model-docs
summary: Official Chatterbox Turbo model card describing the faster packaged TTS path and intended usage flow.
support_level: stable
reliability: official
review_status: reviewed
url: https://huggingface.co/ResembleAI/chatterbox-turbo
tags:
  - tts
  - speech
  - turbo
  - checkpoint
  - runtime-variant
path: sources/resembleai-chatterbox-turbo.md
---

## Why It Matters

- It exposes a packaged faster Chatterbox model family that should be benchmarked before custom kernel work.
- It gives Fusion a first-class checkpoint and runtime-flavor branch to test in the optimization ladder.
- It is a concrete example of why model-family variants belong in the same candidate set as runtime flags, compile modes, and kernels.

## Notes

- The packaged code downloads from the `ResembleAI/chatterbox-turbo` repo id and uses a separate `ChatterboxTurboTTS` path.
- Turbo is not just a small flag flip on the standard checkpoint; it is a distinct model family and should be benchmarked independently.
