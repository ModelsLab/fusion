---
id: cuco_patterns_for_fusion_inner_loop
kind: document
title: CuCo Patterns For Fusion Inner Loop
category: search
summary: "Practical CuCo-inspired rules for Fusion's optimization loop: verified seed first, phased explore/exploit search, novelty-aware survivor selection, archive inspirations, failure retention, and periodic meta-memory updates."
support_level: stable
reliability: curated
review_status: reviewed
source_ids:
  - ut-infraai-cuco
---

## Core Transfer

What transfers from CuCo into Fusion is not the NCCL-specific machinery. It is the optimization discipline:

- build a correctness-preserving seed first
- mutate only bounded hot regions when possible
- search in phases instead of using one flat strategy from start to finish
- preserve diversity explicitly with novelty checks and archive sampling
- store failures and negative examples as reusable memory
- periodically compress multi-round evidence into a smaller recommendation set

## Fast Path Before Slow Path

When a project already has a working implementation, the agent should first create a conservative verified seed for the optimized branch:

- keep interfaces, initialization, and verification scaffolding stable
- isolate mutable hot regions
- confirm build and correctness before deeper performance search

Fusion should treat this as a distinct step before launching a broad inner-loop search.

## Explore Then Exploit

CuCo's two-phase schedule generalizes well:

- early search should maximize structural diversity
- later search should refine the best architectures already discovered

For Fusion this means:

- use multiple search lanes early across backend, schedule, precision, or architecture family
- avoid burning the entire budget on local tuning of one lane before alternatives are tested
- tighten around the strongest survivors only after the search has seen distinct candidate families

## Novelty And Archive Retrieval

Near-duplicate candidates consume benchmark budget without improving search quality.

Fusion should:

- reject or down-rank near-duplicates when ranking survivors
- include structurally different inspirations instead of only the current best lineage
- preserve at least some survivor diversity in beam-style search when scores are close

Useful candidate metadata includes:

- backend
- search lane or island
- structural signature
- hypothesis
- parent candidate

## Failures As Search Memory

Compilation failures, correctness regressions, and runtime crashes should not disappear.

They should be kept as:

- negative examples for later prompts
- reflexion artifacts for the same lane
- session memory entries

This prevents the agent from rediscovering the same dead ends.

## Meta Memory

Periodic summarization should not re-read the entire run each time. It should update a compact scratchpad:

- what was tried
- what worked
- what consistently failed
- which direction should get the next budget

This is especially valuable when a session spans many turns or resumes on a different machine.

## Fusion-Specific Recommendation

Use CuCo's pattern most aggressively when:

- the outer loop is already exhausted
- the hotspot is well localized
- there are multiple plausible implementation families
- evaluation is expensive enough that wasted duplicates matter

Do not over-apply it when the real win is still sitting in an obvious packaged runtime or quantization branch.
