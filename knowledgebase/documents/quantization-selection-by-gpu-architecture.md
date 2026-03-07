---
id: quantization_selection_by_gpu_architecture
kind: document
title: Quantization Selection By GPU Architecture
type: ""
category: quantization
summary: A practical policy for choosing FP8, NVFP4, INT4, and weight-only quantization branches across Ampere, Ada, Hopper, and Blackwell, including multimodal and DiT workloads.
support_level: stable
reliability: ""
review_status: ""
url: ""
tags: []
aliases:
  - quantization-by-arch
  - arch-quant-policy
source_ids:
  - huggingface-optimum-quanto
  - huggingface-diffusers-quanto
  - huggingface-diffusers-quantization-blog
  - huggingface-quantstack-wan22-gguf
  - huggingface-gitmylo-wan22-nvfp4
  - nvidia-transformer-engine
  - nvidia-tensorrt-model-optimizer
  - llm-compressor
---

## Core Rule

Choose quantization by combining:

- GPU-native low-precision support
- model family
- runtime support
- whether the current bottleneck is fit, bandwidth, or compute

Do not use one naming scheme everywhere. AWQ is an LLM-oriented weight-only INT4 family. Diffusion, DiT, and multimodal transformer models often need a more generic weight-only INT4 branch such as Quanto or torchao.

## Ampere

Examples:

- RTX 3090
- A100
- A10

Recommended default order:

1. packaged qint8 / int8 / weight-compressed checkpoints
2. weight-only INT4 if memory fit or bandwidth is the wall
3. runtime/kernel work after the best low-byte resident path is known

Notes:

- No native FP8 path to treat as a first-class default
- Weight-only INT4 can unlock resident execution that is impossible with larger weights
- For multimodal and DiT models, INT4 should be benchmarked as a fit-first branch, not assumed to beat INT8 on throughput
- Packaged Q4 artifacts can exist in alternate runtimes like GGUF, but do not treat them as apples-to-apples evidence for a native PyTorch server path

## Ada

Examples:

- RTX 4090
- L40
- L40S
- RTX 6000 Ada

Recommended default order:

1. packaged FP8 if the runtime and model support it
2. weight-only INT4 for fit or decode-heavy memory pressure
3. kernel and compile work

Notes:

- Ada supports FP8 and often benefits from mature FP8 inference stacks
- INT4 remains strong for consumer-VRAM fit and fused dequant GEMM paths
- For diffusion and DiT, compare FP8 against INT4 instead of assuming the LLM winner transfers directly

## Hopper

Examples:

- H100
- H200

Recommended default order:

1. FP8 first
2. INT4 for fit or bandwidth-limited regimes
3. deeper kernel work

Notes:

- Hopper has strong native FP8 support and mature serving/runtime stacks around it
- INT4 is still valuable for fit, but FP8 is often the cleaner throughput-first branch when supported

## Blackwell

Examples:

- B200
- RTX 5090
- RTX PRO 6000 Blackwell

Recommended default order:

1. FP8 or NVFP4 / FP4-style native paths when available
2. INT4 when compatibility or artifact availability favors it
3. backend-specific kernel work

Notes:

- Blackwell changes the low-precision frontier because FP4 / NVFP4 becomes a native first-class consideration
- Do not automatically carry Ampere/Ada INT4 assumptions forward when a native narrower floating-point path exists
- Community FP4 or NVFP4 artifacts can be useful search leads, but they still need runtime and quality validation

## Model-Family Rule

Use these naming conventions correctly:

- LLM serving:
  AWQ, GPTQ, Marlin, EXL2, FP8, KV quantization
- Diffusion / DiT / multimodal transformers:
  weight-only INT4, qint4, fp8, Model Optimizer, Quanto, torchao

This distinction matters because the conversion tools, kernels, and runtime compatibility are different even when the high-level goal is the same.

## Benchmark Policy

For any quantization branch, record all of:

- load time
- peak VRAM
- steady-state latency / throughput
- quality drift
- whether the quantized path allows a faster resident configuration that was previously impossible

That last point is often the real win on cards like the RTX 3090.
