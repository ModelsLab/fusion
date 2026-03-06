---
id: skill_speculative_decoding_eval
kind: skill
title: Speculative Decoding Evaluation
category: inference
summary: Evaluate and benchmark speculative decoding strategies for a given model, measuring acceptance rates, speedup, and quality impact.
support_level: stable
source_ids: []
workloads:
  - decode
  - serving
operators:
  - attention
  - matmul
gpu_families:
  - Ampere
  - Ada
  - Hopper
  - Blackwell
precision:
  - fp16
  - bf16
  - fp8
  - int4
bottlenecks:
  - memory
goals:
  - latency
preferred_backends:
  - vllm
  - sglang
required_tools:
  - read_file
  - write_file
  - run_command
  - run_benchmark
  - search_knowledge_base
steps:
  - identify the target model and available draft model candidates
  - evaluate acceptance rate on representative prompts (need >60% for meaningful speedup)
  - benchmark end-to-end tokens/sec with and without speculation
  - measure draft model overhead (should be <20% of target model time)
  - test with different k values (3, 5, 7) to find optimal
  - compare with self-speculation (LayerSkip) if training access available
verification:
  - acceptance rate exceeds 60% on representative data
  - end-to-end speedup is at least 1.5x
  - output quality matches non-speculative baseline
benchmark_rubric:
  - tokens per second with and without speculation
  - acceptance rate by position (1st speculated, 2nd, etc.)
  - memory overhead from draft model
  - latency distribution (P50, P95, P99)
failure_recovery:
  - if acceptance rate is low, try a different draft model or reduce k
  - if memory is insufficient, consider self-speculation or Medusa heads
  - if quality differs, verify rejection sampling is correctly implemented
artifacts_to_save:
  - benchmark_json
  - acceptance_rate_analysis
  - quality_comparison
runtime_adapters:
  - vllm
  - sglang
reference_source_ids: []
---

## Steps
- identify target model and draft model candidates
- evaluate acceptance rate on representative prompts
- benchmark end-to-end with and without speculation
- find optimal number of speculative tokens (k)

## Verification
- acceptance rate >60%, speedup >1.5x, quality matches baseline
