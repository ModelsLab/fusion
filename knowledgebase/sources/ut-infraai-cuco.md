---
id: ut-infraai-cuco
kind: source
title: "CUCo: Compute And Communication Co-design"
type: official-project
category: optimization-agents
summary: CUCo is an agentic framework that splits correctness-preserving transformation from performance search and uses novelty-aware island-based evolution, archive inspirations, cascading evaluation, and meta-summarization.
reliability: official
review_status: reviewed
url: https://ut-infraai.github.io/cuco/
tags:
  - cuco
  - evolutionary-search
  - optimization-agent
  - cuda
  - distributed
  - novelty
  - meta-learning
---

## Key Features

- Separates a correctness-first fast path from a performance-oriented slow path
- Uses archive inspirations, novelty filtering, and optional island search to preserve diversity
- Stores failures as first-class search memory instead of discarding them
- Periodically summarizes cross-generation lessons into reusable recommendations

## Notes

- CUCo is compute-and-communication focused, so some workload-specific details are NCCL and distributed-kernel specific
- The reusable patterns for Fusion are the search discipline, evaluation cascade, and memory architecture rather than the exact communication APIs
