---
id: novelty_aware_island_search
kind: strategy
title: Use Novelty-Aware Search Lanes For Inner Loop Optimization
category: search
summary: When the outer loop is exhausted and multiple plausible implementation families remain, run inner-loop search across distinct lanes and preserve diverse survivors instead of only following the current top score.
support_level: stable
workloads:
  - text-generation
  - image-generation
  - image-editing
  - video-generation
  - audio-generation
  - multimodal
operators:
  - attention
  - gemm
  - communication
  - general
gpu_families:
  - all
precision:
  - any
bottlenecks:
  - compute
  - memory
  - mixed
goals:
  - throughput
  - latency
  - search-efficiency
priority: 86
preconditions:
  - outer-loop packaged/runtime/quantization branches are exhausted or blocked
  - there are multiple realistic implementation families or backend choices
actions:
  - split early search into distinct lanes by backend, schedule, precision path, or architecture family
  - preserve diversity among beam survivors when scores are near-tied
  - use prior failures and archive inspirations to avoid redundant mutations
  - switch from exploration to exploitation once a small set of strong lanes emerges
tradeoffs:
  - more bookkeeping is required than plain greedy ranking
  - diversity pressure can temporarily keep a slightly weaker candidate alive
metrics:
  - candidate uniqueness
  - survivor diversity
  - best score by round
  - duplicate rejection rate
source_ids:
  - ut-infraai-cuco
---

## Actions

- split early search into distinct lanes by backend, schedule, precision path, or architecture family
- preserve diversity among beam survivors when scores are near-tied
- use prior failures and archive inspirations to avoid redundant mutations
- switch from exploration to exploitation once a small set of strong lanes emerges

## Why

- near-duplicate candidates waste compile and benchmark budget
- diversity early in the run helps avoid local optima
