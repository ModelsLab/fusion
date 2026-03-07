---
id: wan_rtx3090_live_qint4_repack
kind: example
title: WAN RTX 3090 Live qint4 Repack Findings
type: ""
category: video-generation
summary: For WAN-style dual-transformer video pipelines already loaded as Quanto qint8 models, a live repack to qint4 on an RTX 3090 was not a practical request-time load path and should be converted into an offline cached artifact flow.
support_level: experimental
reliability: measured
review_status: reviewed
url: ""
tags: []
source_ids:
  - huggingface-optimum-quanto
  - huggingface-diffusers-quanto
  - huggingface-diffusers-quantization-blog
gpu_ids:
  - rtx3090
workloads:
  - video-generation
operators:
  - quantization
  - matmul
precision:
  - qint8
  - qint4
---

## Observed Behavior

- The active WAN qint8 checkpoints already load as `optimum.quanto.nn.qlinear.QLinear` modules.
- Repacking those modules to qint4 is technically possible.
- A live qint8-to-qint4 repack for the full dual-transformer WAN path consumed tens of gigabytes of host RAM and several minutes of CPU-heavy conversion time before reaching inference.
- A first attempt also exposed a device-placement issue until the repacked modules were re-profiled by the memory manager.
- A redesigned offline converter that builds base-transformer qint4 cache artifacts before reattaching the runtime LoRA reduced the operational footprint substantially, but the first 3090 run was still a multi-minute conversion job and did not finish quickly enough to promote qint4 as the serving winner.

## Practical Conclusion

- Treat this as an offline artifact-conversion branch, not a normal request-time load branch.
- Prefer caching the base quantized transformers first and then applying runtime adapters such as few-step LoRAs afterward, so the cached INT4 artifact is reusable and cheaper to build.
- Benchmark qint4 only after caching the converted weights or shipping a packaged qint4 checkpoint.
- Compare the cached resident qint4 path against the already-strong resident qint8 path on Ampere consumer GPUs.

## Why This Matters

- On cards like the RTX 3090, qint4 may still be the right answer for memory fit.
- But if the conversion itself is done live, the load path can become more expensive than the serving win.
