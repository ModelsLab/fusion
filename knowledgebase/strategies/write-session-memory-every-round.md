---
id: write_session_memory_every_round
kind: strategy
title: Write Session Memory Every Round
category: workflow
summary: Persist short markdown memory after important session events so future optimization turns can resume from evidence instead of re-deriving history.
support_level: recommended
reliability: curated
review_status: reviewed
workloads:
  - decode
  - serving
  - sampling
  - refinement
goals:
  - reliability
  - throughput
  - latency
priority: 88
actions:
  - write a short markdown memory entry after wins, failures, blockers, or environment changes
  - include the candidate, outcome, relevant metrics, lessons, and next steps
  - keep a session memory index current so the next agent turn can resume quickly
metrics:
  - candidate outcome
  - benchmark metrics
  - quality status
tradeoffs:
  - memory writing adds small overhead per round
  - low-value noisy notes should be avoided
---

## Actions

- write a short markdown memory entry after wins, failures, blockers, or environment changes
- include the candidate, outcome, relevant metrics, lessons, and next steps
- keep a session memory index current so the next agent turn can resume quickly
