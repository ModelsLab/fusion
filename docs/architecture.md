# Fusion Architecture

Fusion is split into two layers:

1. knowledge and planning
2. execution and verification

The repository now implements the first layer in a way that runs on macOS without CUDA.

## Layer 1: Knowledge And Planning

The CLI embeds a curated corpus of:

- GPU archetypes and target-specific constraints
- optimization strategies grouped by workload, operator family, and bottleneck
- references to official docs, papers, public repos, and queued community links

This keeps runtime prompts small. An agent can ask Fusion for the subset of strategies that fit a GPU, workload, and precision instead of receiving a giant free-form prompt.

## Layer 2: Execution And Verification

The next Linux-focused phase should add:

- kernel candidate generation in Triton and CUDA
- correctness harnesses against reference implementations
- benchmark runners for throughput, latency, and memory
- Nsight Compute / Nsight Systems integration
- before/after reports with ranked wins and residual bottlenecks

## Why The Knowledge Base Is Embedded

The user goal is not a thin wrapper around a hosted model. Fusion needs a durable, inspectable optimization corpus inside the repo so:

- recommendations stay source-backed
- model context stays small
- the CLI can function with limited or no network access
- future agent loops can query structure instead of re-learning the same facts

## Immediate Linux Roadmap

1. add `profile run` to capture baseline Nsight metrics
2. add `benchmark run` to time representative decode/prefill cases
3. add `kernel synthesize` to generate Triton/CUDA candidates with a provider-backed model
4. add `kernel verify` to check numerical correctness
5. add `kernel rank` to compare variants and persist winners by GPU family
