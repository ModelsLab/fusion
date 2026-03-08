---
id: lightx2v_wan22_rtx4090_q8f_beats_triton
kind: example
title: LightX2V Wan 2.2 On RTX 4090 Favored Q8F Over Triton
type: ""
category: video-generation
summary: On an RTX 4090, LightX2V Wan 2.2 I2V benchmarks showed that `int8-q8f` beat both Triton variants and also beat `fp8-q8f` at `480p` and `720p`.
support_level: measured
reliability: measured
review_status: reviewed
url: ""
tags:
  - lightx2v
  - wan
  - rtx4090
  - q8f
  - fp8
  - int8
source_ids:
  - lightx2v-quantization-q8f
gpu_ids:
  - rtx4090
workloads:
  - video-generation
operators:
  - attention
  - matmul
  - quantization
precision:
  - int8
  - fp8
---

## Measured Results

Target workload:

- task: LightX2V Wan 2.2 I2V
- frames: `21`
- steps: `4`
- attention mode: `sage_attn2`
- offload: `cpu_offload=true`, `offload_granularity=block`

`480p`:

- `int8-triton`: about `25.57s`
- `fp8-triton`: about `27.96s`
- `int8-q8f`: about `14.45s`
- `fp8-q8f`: about `15.05s`

`720p`:

- `int8-triton`: about `28.05s`
- `fp8-triton`: about `30.62s`
- `int8-q8f`: about `21.97s`
- `fp8-q8f`: about `26.29s`

## Interpretation

- The kernel family choice mattered more than the INT8 vs FP8 choice.
- For this Ada-targeted runtime, `q8f` was the right backend family and Triton was not the winning path.
- Even on a GPU with native FP8 support, the best measured path was still `int8-q8f`, not FP8.

## Retrieval Guidance

- If Fusion detects LightX2V-style Wan quantized inference on RTX 40-class hardware, retrieve `q8f` as an early branch.
- Do not assume FP8 is the winner just because the card supports FP8 tensor cores.
