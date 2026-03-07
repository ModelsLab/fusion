---
id: resembleai-chatterbox
kind: source
title: Chatterbox TTS Hugging Face README
type: official
category: model-docs
summary: Official Chatterbox TTS model card and README covering install, CUDA usage, and reference generation flow.
support_level: stable
reliability: official
review_status: reviewed
url: https://huggingface.co/ModelsLab/chatterbox
tags:
  - tts
  - speech
  - llama
  - voice-cloning
  - cuda
path: sources/resembleai-chatterbox.md
---

## Why It Matters

- It defines the reference install and inference path for Chatterbox TTS.
- It is the baseline source for validating any runtime or kernel optimization against the expected usage flow.
- It is useful for reproducing a real CUDA-backed speech workload instead of optimizing an isolated toy kernel.

## Notes

- The packaged code downloads checkpoints from the `ResembleAI/chatterbox` repo id internally even when the model card is mirrored elsewhere.
- The default example loads the model on CUDA and runs `ChatterboxTTS.from_pretrained(device="cuda")`.
