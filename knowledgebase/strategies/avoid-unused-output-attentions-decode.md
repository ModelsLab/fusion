---
id: avoid-unused-output-attentions-decode
kind: strategy
title: Avoid Unused output_attentions in Decode Loops
category: runtime-patch
summary: When a decode path only needs attention maps for an optional analyzer or debug hook, gate output_attentions off by default to preserve optimized attention kernels.
support_level: recommended
reliability: curated
review_status: reviewed
tags:
  - attention
  - sdpa
  - decode
  - transformers
  - runtime-patch
workloads:
  - decode
  - serving
operators:
  - attention
gpu_families:
  - Ampere
  - Ada
  - Hopper
  - Blackwell
precision:
  - fp32
  - fp16
  - bf16
bottlenecks:
  - latency
  - mixed
goals:
  - latency
  - throughput
priority: 85
preconditions:
  - The model only consumes attention maps when a debug, alignment, or attribution path is enabled.
  - A reproducible correctness harness exists so seeded outputs can be compared before and after the change.
actions:
  - Inspect the decode loop for output_attentions=True on every step.
  - Gate output_attentions on whether an analyzer or hook is actually attached.
  - Re-run deterministic decode benchmarks and compare output hashes or numeric tolerances.
metrics:
  - gen_s
  - rtf
  - x_real_time
tradeoffs:
  - If an analyzer depends on attention maps, disabling them removes those signals.
  - This should not be applied blindly if downstream logic reads tfmr_out.attentions.
source_ids:
  - pytorch-attention-docs
  - resembleai-chatterbox
path: strategies/avoid-unused-output-attentions-decode.md
---

## Summary

Some custom decode loops keep `output_attentions=True` turned on even when attention maps are not consumed in the active runtime path. On Hugging Face-backed transformer models, that can force a slower attention implementation and block the fastest SDPA path.

## Recommended Flow

- First verify whether attention maps are only needed for an optional runtime analyzer.
- If the analyzer is disabled, request logits and cache only.
- Compare deterministic seeded outputs before and after the change.
- Keep the optimization guarded so multilingual or debug paths can still enable attention collection when needed.
