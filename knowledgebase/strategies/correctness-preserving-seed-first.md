---
id: correctness_preserving_seed_first
kind: strategy
title: Build A Correctness-Preserving Seed Before Aggressive Search
category: search
summary: Before broad inner-loop mutation, create a verified seed that preserves interfaces and correctness scaffolding so later search rounds optimize a stable baseline instead of chasing basic breakage.
support_level: stable
workloads:
  - text-generation
  - image-generation
  - image-editing
  - video-generation
  - audio-generation
  - multimodal
operators:
  - general
gpu_families:
  - all
precision:
  - any
bottlenecks:
  - mixed
goals:
  - correctness
  - search-efficiency
priority: 88
preconditions:
  - the project already has a working baseline or reference path
actions:
  - create a conservative verified seed before broad mutation
  - keep initialization, verification, and stable interfaces fixed while searching hot regions
  - only escalate to architectural rewrites after the seed is proven correct
tradeoffs:
  - can add an extra upfront step before performance work
  - may feel slower early, but reduces wasted rounds on trivial breakage
metrics:
  - build pass rate
  - verify pass rate
  - number of wasted search rounds
source_ids:
  - ut-infraai-cuco
---

## Actions

- create a conservative verified seed before broad mutation
- keep initialization, verification, and stable interfaces fixed while searching hot regions
- only escalate to architectural rewrites after the seed is proven correct

## Why

- correctness-first seeds reduce search churn
- the agent can focus later rounds on performance instead of repeatedly rebuilding basic functionality
