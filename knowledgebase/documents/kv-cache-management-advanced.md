---
id: kv_cache_management_advanced
kind: document
title: Advanced KV Cache Management Strategies
category: memory
summary: Comprehensive guide to KV cache optimization including PagedAttention internals, cache-aware scheduling, eviction policies, compression, semantic caching, and multi-GPU cache coordination.
tags:
  - kv-cache
  - paged-attention
  - memory-management
  - cache-eviction
  - streaming
  - compression
source_ids: []
operators:
  - attention
  - memory
gpu_families:
  - Ampere
  - Ada
  - Hopper
  - Blackwell
---

## 1. KV Cache Memory Analysis

### Per-Token Memory Formula

Each transformer layer stores a key tensor and a value tensor for every token in the sequence.
The memory consumed per token across the full model is:

```
KV_bytes_per_token = 2 * n_layers * n_kv_heads * d_head * bytes_per_element
```

Where:
- `2` accounts for both K and V projections
- `n_kv_heads` equals `n_heads` for MHA, or the GQA/MQA group count
- `d_head` is typically `hidden_size / n_heads`
- `bytes_per_element` is 2 for FP16/BF16, 1 for FP8, 0.5 for INT4

### Per-Token KV Memory Table (FP16)

| Model             | Layers | KV Heads | d_head | KV bytes/token | KV for 4K ctx  | KV for 128K ctx |
|--------------------|--------|----------|--------|----------------|----------------|-----------------|
| LLaMA-2 7B         | 32     | 32       | 128    | 524,288 B (512 KB) | 2.0 GB       | 64.0 GB         |
| LLaMA-2 13B        | 40     | 40       | 128    | 819,200 B (800 KB) | 3.1 GB       | 100.0 GB        |
| LLaMA-2 70B        | 80     | 8 (GQA)  | 128    | 327,680 B (320 KB) | 1.25 GB      | 40.0 GB         |
| LLaMA-3 8B         | 32     | 8 (GQA)  | 128    | 131,072 B (128 KB) | 0.5 GB       | 16.0 GB         |
| LLaMA-3 70B        | 80     | 8 (GQA)  | 128    | 327,680 B (320 KB) | 1.25 GB      | 40.0 GB         |
| Mistral 7B         | 32     | 8 (GQA)  | 128    | 131,072 B (128 KB) | 0.5 GB       | 16.0 GB         |
| Mixtral 8x7B       | 32     | 8 (GQA)  | 128    | 131,072 B (128 KB) | 0.5 GB       | 16.0 GB         |
| DeepSeek-V3 (MLA)  | 61     | -        | -      | ~60 KB (compressed) | 0.23 GB     | 7.5 GB          |
| Qwen-2 72B         | 80     | 8 (GQA)  | 128    | 327,680 B (320 KB) | 1.25 GB      | 40.0 GB         |

DeepSeek-V3 uses Multi-head Latent Attention (MLA) which compresses KV into a low-rank latent of dimension 512 per layer, drastically reducing the per-token footprint compared to standard GQA at similar model scale.

### Batch-Level Memory

For a serving system running `B` concurrent sequences of average length `S`:

```
Total_KV_memory = B * S * KV_bytes_per_token
```

On an 80 GB A100, serving LLaMA-2 7B at FP16 with 4096-token contexts, the KV cache alone limits concurrency to roughly `80 GB / 2.0 GB = 40` simultaneous sequences (ignoring model weight memory).

## 2. PagedAttention Internals

### Core Idea

PagedAttention (Kwon et al., 2023) applies operating-system-style virtual memory to KV cache management. Instead of pre-allocating contiguous memory for the maximum sequence length, it divides KV storage into fixed-size **blocks** and maps them through a **block table** per sequence.

### Block Structure

```
Block = [block_size tokens] x [n_kv_heads] x [d_head] x [2 for K,V]
```

Typical block sizes: 16 or 32 tokens. Each block occupies a fixed physical memory slot in a pre-allocated GPU memory pool.

### Virtual-to-Physical Mapping

```
Sequence "A" block table:
  Virtual Block 0 -> Physical Block 7
  Virtual Block 1 -> Physical Block 2
  Virtual Block 2 -> Physical Block 15

Sequence "B" block table:
  Virtual Block 0 -> Physical Block 7   (shared prefix, CoW)
  Virtual Block 1 -> Physical Block 11
```

The block table is a simple integer array per sequence: `block_table[seq_id][virtual_block_idx] = physical_block_idx`.

### PagedAttention v1 vs v2

**v1:** Each thread block processes one attention head for one sequence. The kernel partitions across `(batch, head)` pairs. For long sequences the single thread block must iterate over all KV blocks sequentially, limiting parallelism.

**v2:** Introduces an additional partitioning dimension along the sequence (KV block) axis. Multiple thread blocks cooperate on one `(batch, head)` pair by computing partial softmax results across disjoint KV block ranges, then a final reduction kernel combines them. This improves GPU utilization for long-context scenarios.

### Copy-on-Write for Beam Search

When beam search forks a sequence:

1. The child sequence copies the parent block table (integer copy, not data copy).
2. Physical blocks gain a reference count. Shared blocks have `ref_count > 1`.
3. When a sequence needs to append tokens to a shared block, the runtime allocates a new physical block, copies the data, and updates only that sequence's block table entry.
4. When `ref_count` drops to zero, the physical block returns to the free pool.

This reduces memory from `O(beam_width * seq_len)` to near `O(seq_len)` for shared prefixes.

### Block Size Selection

| Block Size | Pros | Cons |
|------------|------|------|
| 8          | Low internal fragmentation | Higher block table overhead, more kernel launches |
| 16         | Good balance for most workloads | Default in vLLM |
| 32         | Better memory throughput per block | Wastes memory for short sequences |
| 64+        | Maximizes coalesced access | Significant fragmentation for variable-length batches |

Average internal fragmentation = `block_size / 2` tokens per sequence on the last block.

## 3. Cache-Aware Scheduling

### vLLM Scheduler

The vLLM scheduler maintains three queues: **waiting** (new requests), **running** (actively generating), and **swapped** (preempted to CPU).

Scheduling loop per iteration:

```python
# Simplified vLLM scheduling logic
def schedule_step(self):
    free_blocks = self.block_manager.get_num_free_gpu_blocks()

    # 1. Try to resume swapped sequences
    for seq in self.swapped_queue:
        needed = seq.get_num_required_blocks()
        if needed <= free_blocks:
            self.running_queue.append(seq)
            self.swap_in(seq)  # CPU -> GPU block copy
            free_blocks -= needed

    # 2. Admit new requests from waiting queue
    for seq in self.waiting_queue:
        needed = seq.get_num_initial_blocks()
        if needed <= free_blocks:
            self.running_queue.append(seq)
            free_blocks -= needed

    # 3. If running sequences need blocks but none free, preempt
    while cannot_allocate_new_blocks():
        victim = self.running_queue.pop()  # FCFS or priority
        self.swap_out(victim)              # GPU -> CPU block copy
        self.swapped_queue.append(victim)
```

### Preemption Policies

- **Swap:** Copy KV blocks to CPU RAM, reclaim GPU blocks. Resume later by swapping back. High latency but preserves work.
- **Recompute:** Discard KV blocks entirely. When the request resumes, rerun prefill from the prompt. Lower memory overhead but burns FLOPs.
- **Priority-based:** Assign priorities to requests (e.g., by arrival time, SLO deadline). Preempt lowest-priority first.

### SGLang Scheduling

SGLang extends cache-aware scheduling with **RadixAttention** (see Section 6). The scheduler considers prefix cache hits when ordering requests: a request whose prompt shares a cached prefix is cheaper to schedule because it requires fewer new KV blocks.

## 4. Token Eviction Strategies

When the KV cache is full and the sequence must continue, eviction policies decide which tokens' KV entries to drop.

### StreamingLLM (Attention Sinks)

Key observation: the first few tokens (attention sinks) receive disproportionately high attention mass regardless of content. StreamingLLM retains:

```
Kept tokens = [first S sink tokens] + [last W window tokens]
Evicted     = everything in between
```

Typical configuration: `S = 4` sink tokens, `W = 1024-4096` window tokens. This enables theoretically infinite-length generation with fixed KV memory.

Limitation: information in evicted middle tokens is permanently lost. Suitable for streaming/chat but not for tasks requiring long-range retrieval.

### H2O (Heavy-Hitter Oracle)

H2O tracks cumulative attention scores across all past forward passes. Tokens that consistently receive high attention (heavy hitters) are retained; low-attention tokens are evicted.

```
Policy: keep top-k tokens by cumulative_attention[layer][token]
Evict:  tokens with lowest cumulative attention
Budget: configurable per-layer or global
```

Advantage over StreamingLLM: retains semantically important tokens anywhere in the sequence, not just at the boundaries.

### SnapKV

SnapKV observes attention patterns during a single prefill pass to identify important tokens, then compresses the KV cache before decode begins:

1. Run full prefill to compute attention weights.
2. For each layer and head, select the top-k tokens by attention from the observation window.
3. Cluster selected positions and keep contiguous spans for memory efficiency.
4. Discard KV entries for unselected tokens.

This achieves 3-6x KV compression with minimal quality loss on long-context benchmarks.

### ScissorHands

ScissorHands identifies "pivotal tokens" by tracking which tokens are persistently attended to across multiple inference steps. Unlike H2O which uses cumulative scores, ScissorHands uses a persistence-based criterion:

```
A token is pivotal if it appears in the top-k attended tokens
for at least P% of recent inference steps.
```

### XKV (Cross-Layer KV Sharing)

XKV identifies layers with highly similar KV representations and shares one copy across those layers. This is an architectural-level compression that reduces unique KV entries by 30-60% in some models. The selection of which layers share is determined by analyzing representational similarity during a calibration pass.

## 5. KV Cache Compression

### Quantized KV Cache

Storing KV entries in reduced precision shrinks cache memory proportionally:

| Precision | Bytes/element | Relative size | Quality impact |
|-----------|---------------|---------------|----------------|
| FP16/BF16 | 2             | 1.0x (baseline) | None           |
| FP8 (E4M3)| 1             | 0.5x          | Minimal for most models |
| INT8       | 1             | 0.5x          | Low with proper calibration |
| INT4       | 0.5           | 0.25x         | Moderate, needs per-channel scales |

vLLM supports FP8 KV cache natively with `--kv-cache-dtype fp8_e4m3`. The quantization is applied per-block with per-tensor or per-channel scaling factors stored alongside the blocks.

```python
# vLLM FP8 KV cache configuration
engine_args = EngineArgs(
    model="meta-llama/Llama-3-70b",
    kv_cache_dtype="fp8_e4m3",      # 2x KV memory savings
    quantization_param_path="./kv_scales.json",  # optional calibrated scales
)
```

### Grouped-Query Attention as Implicit Compression

GQA (Ainslie et al., 2023) reduces KV heads from `n_heads` to `n_kv_heads`, providing a compression ratio of `n_heads / n_kv_heads`:

| Model       | Attn Heads | KV Heads | KV Compression Ratio |
|-------------|-----------|----------|----------------------|
| LLaMA-2 7B  | 32        | 32 (MHA) | 1x                   |
| LLaMA-2 70B | 64        | 8 (GQA)  | 8x                   |
| LLaMA-3 8B  | 32        | 8 (GQA)  | 4x                   |
| Mistral 7B  | 32        | 8 (GQA)  | 4x                   |

### Multi-head Latent Attention (DeepSeek MLA)

MLA compresses the full KV projection into a low-rank latent vector per token:

```
Standard: K, V each have shape [n_heads, d_head]  ->  stored as-is
MLA:      joint KV compressed to latent c of dimension d_c (e.g., 512)
          K = W_uk @ c,  V = W_uv @ c   (reconstructed on-the-fly during attention)
```

KV storage per token per layer drops from `2 * n_heads * d_head` to just `d_c` elements. For DeepSeek-V3 with `d_c = 512` and FP16, that is 1024 bytes per layer versus tens of thousands for standard MHA at similar model scale.

The trade-off: attention computation must reconstruct K and V from the latent, adding FLOPs. This is worthwhile when the workload is memory-bound (decode phase) but less beneficial during compute-bound prefill.

## 6. Prefix Caching

### RadixAttention (SGLang)

SGLang organizes the KV cache as a **radix tree** keyed by token sequences. When a new request arrives, the scheduler walks the tree to find the longest matching prefix and reuses those cached KV blocks.

```
Radix Tree Example:
root
 |-- [SYS_PROMPT tokens 0..127]  -> blocks [P0, P1, P2, P3]
      |-- [USER_A tokens 128..255] -> blocks [P4, P5]
      |    |-- [TURN_1 tokens 256..383] -> blocks [P6, P7]
      |-- [USER_B tokens 128..191] -> blocks [P8]
```

All requests sharing the same system prompt reuse blocks P0-P3. Multi-turn conversations reuse previous turns.

**Eviction:** LRU on leaf nodes. When memory pressure rises, the least recently used leaf branches are freed first, preserving widely shared prefixes.

### Automatic Prefix Caching (vLLM)

vLLM implements hash-based prefix caching. Each block is keyed by the hash of its token content plus preceding block hashes:

```python
block_hash = hash(parent_block_hash, tuple(token_ids_in_block))
```

When a new request's prompt hashes match existing cached blocks, they are reused without recomputation. This is transparent to the user and works across unrelated requests that happen to share prompt prefixes.

### Cache Hit Rate Analysis

| Workload Pattern | Expected Hit Rate | Notes |
|------------------|-------------------|-------|
| Shared system prompt | 70-95% | Most chat deployments |
| Multi-turn conversation | 85-99% | Previous turns cached |
| Few-shot prompting | 60-80% | Shared examples |
| Independent prompts | 0-5% | Minimal sharing |
| Code completion (repo-level) | 40-70% | Shared file context |

## 7. Multi-GPU KV Cache

### Tensor Parallel KV Distribution

Under tensor parallelism (TP), attention heads are sharded across GPUs. Each GPU stores only its shard of the KV cache:

```
TP=4, model has 32 KV heads:
  GPU 0: KV heads 0-7    (local blocks)
  GPU 1: KV heads 8-15   (local blocks)
  GPU 2: KV heads 16-23  (local blocks)
  GPU 3: KV heads 24-31  (local blocks)
```

Each GPU runs PagedAttention on its local KV subset. No cross-GPU KV communication is needed during attention (only the all-reduce after the attention output projection).

### Disaggregated Serving and KV Transfer

In disaggregated (prefill-decode split) architectures, a prefill node computes KV values and transfers them to a decode node:

```
Prefill Node                         Decode Node
  [prompt] -> compute KV  --RDMA-->  receive KV blocks
                                     [decode token by token]
```

Key considerations:
- **Transfer bandwidth:** For LLaMA-70B at 4096 tokens, KV transfer is ~1.25 GB. Over NVLink (900 GB/s) this takes ~1.4 ms. Over InfiniBand (400 Gb/s) this takes ~25 ms.
- **Block format compatibility:** Both nodes must agree on block size, dtype, and head layout.
- **Pipeline overlap:** Transfer can overlap with the prefill of the next request.

### KV Cache Migration

For load balancing or fault tolerance, KV blocks can be migrated between GPUs:

1. Serialize block table and physical block data on source GPU.
2. Transfer via NVLink, PCIe, or network.
3. Allocate physical blocks on destination, deserialize, update block table.
4. Resume generation on the destination GPU.

Frameworks like Mooncake and DistServe implement KV migration for heterogeneous cluster scheduling.

## 8. Speculative Decoding KV Cache

### Draft Model KV Management

Speculative decoding runs a small draft model to generate `K` candidate tokens, then verifies them in parallel with the target model. This creates two separate KV caches:

```
Draft model KV:  small, fast to compute, discarded after verification
Target model KV: large, updated only with accepted tokens
```

Memory overhead: the draft model KV is typically 5-15% of the target model KV (since the draft model is much smaller). It can be allocated from the same block pool or a separate smaller pool.

### Accepted/Rejected Token Handling

```python
# After speculative verification
draft_tokens  = [t0, t1, t2, t3, t4]     # 5 draft tokens
accepted_mask = [1,  1,  1,  0,  0]       # first 3 accepted

# Target model KV: append KV for t0, t1, t2 (already computed during verify)
# Target model KV: discard KV for t3, t4
# Draft model KV:  reset to position after t2, regenerate from there
```

### Tree Attention Cache

Advanced speculative methods (Medusa, EAGLE, SequoIA) generate a tree of candidates rather than a single chain. The KV cache must handle tree-structured attention masks:

```
Token tree:
      [root]
     /      \
   [a]      [b]
   / \        \
 [c] [d]     [e]

KV cache stores entries for all 5 candidates.
Attention mask ensures each candidate only attends to its ancestors.
After verification, prune non-accepted branches and compact KV.
```

## 9. Continuous Batching and KV Memory

### Dynamic Allocation

In continuous batching (Orca-style iteration-level scheduling), sequences enter and leave the batch at every decode step. KV memory is allocated and freed dynamically:

```
Step N:   Batch = [A(pos=100), B(pos=50), C(pos=200)]
          Allocate 1 new block for A (crossed block boundary)

Step N+1: Batch = [A(pos=101), B(pos=51), D(pos=0)]
          C finished, freed 13 blocks
          D arrived, allocated 0 blocks (prefill pending)

Step N+2: Batch = [A(pos=102), B(pos=52), D(pos=512)]
          D prefill complete, allocated 32 blocks
```

### Fragmentation

Over time, physical block allocation becomes fragmented as sequences of varying lengths allocate and free blocks in arbitrary order. Unlike contiguous allocation, PagedAttention is inherently fragmentation-resistant because blocks are uniformly sized and non-contiguous allocation is the norm.

However, **external fragmentation** can still occur in the block pool when the free block count is sufficient but the allocator's bookkeeping structures (e.g., CUDA memory pools) become inefficient.

### Defragmentation

vLLM performs defragmentation by:
1. Identifying blocks that can be consolidated (e.g., blocks from completed sequences).
2. Returning them to the free list immediately (no compaction needed because blocks are uniform size).
3. Periodically resetting the CUDA memory pool if the allocator's overhead grows.

## 10. Implementation Details

### Estimating KV Cache Capacity

```python
def estimate_kv_capacity(
    gpu_memory_gb: float,
    model_memory_gb: float,
    overhead_gb: float,
    n_layers: int,
    n_kv_heads: int,
    d_head: int,
    block_size: int,
    dtype_bytes: int = 2,  # FP16
) -> dict:
    """Estimate how many tokens/sequences the KV cache can hold."""
    available_gb = gpu_memory_gb - model_memory_gb - overhead_gb
    available_bytes = available_gb * (1024 ** 3)

    # Memory per block: block_size tokens * 2(K,V) * layers * kv_heads * d_head * dtype
    bytes_per_block = block_size * 2 * n_layers * n_kv_heads * d_head * dtype_bytes
    total_blocks = int(available_bytes // bytes_per_block)
    total_tokens = total_blocks * block_size

    return {
        "available_memory_gb": available_gb,
        "bytes_per_block": bytes_per_block,
        "total_blocks": total_blocks,
        "total_tokens": total_tokens,
        "max_sequences_at_2048": total_tokens // 2048,
        "max_sequences_at_4096": total_tokens // 4096,
    }

# Example: LLaMA-3 8B on A100-80GB
result = estimate_kv_capacity(
    gpu_memory_gb=80,
    model_memory_gb=16,    # FP16 weights
    overhead_gb=2,         # activations, framework
    n_layers=32,
    n_kv_heads=8,          # GQA
    d_head=128,
    block_size=16,
)
# result: ~760K tokens, ~185 sequences at 4096 context
```

### Custom KV Cache with FP8 Quantization

```python
import torch

class FP8KVCache:
    """Simplified FP8 KV cache with per-block scaling."""

    def __init__(self, num_blocks, block_size, n_layers, n_kv_heads, d_head, device):
        self.block_size = block_size
        # Store KV in FP8 with per-block scale factors
        self.k_cache = torch.zeros(
            (num_blocks, n_layers, block_size, n_kv_heads, d_head),
            dtype=torch.float8_e4m3fn, device=device
        )
        self.v_cache = torch.zeros_like(self.k_cache)
        self.k_scales = torch.ones(num_blocks, n_layers, device=device, dtype=torch.float32)
        self.v_scales = torch.ones_like(self.k_scales)
        self.free_blocks = list(range(num_blocks))

    def allocate_block(self) -> int:
        if not self.free_blocks:
            raise RuntimeError("KV cache out of memory")
        return self.free_blocks.pop()

    def free_block(self, block_idx: int):
        self.free_blocks.append(block_idx)

    def store(self, block_idx: int, layer: int, positions: slice, k: torch.Tensor, v: torch.Tensor):
        """Quantize and store KV entries into a block."""
        k_scale = k.abs().max() / 448.0  # FP8 E4M3 max value
        v_scale = v.abs().max() / 448.0
        self.k_cache[block_idx, layer, positions] = (k / k_scale).to(torch.float8_e4m3fn)
        self.v_cache[block_idx, layer, positions] = (v / v_scale).to(torch.float8_e4m3fn)
        self.k_scales[block_idx, layer] = k_scale
        self.v_scales[block_idx, layer] = v_scale

    def fetch(self, block_idx: int, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Dequantize and return KV entries from a block."""
        k = self.k_cache[block_idx, layer].to(torch.float16) * self.k_scales[block_idx, layer]
        v = self.v_cache[block_idx, layer].to(torch.float16) * self.v_scales[block_idx, layer]
        return k, v
```

### StreamingLLM Eviction Implementation

```python
class StreamingLLMCache:
    """KV cache with attention-sink-aware eviction."""

    def __init__(self, sink_tokens: int = 4, window_size: int = 2044):
        self.sink_tokens = sink_tokens
        self.window_size = window_size
        self.max_tokens = sink_tokens + window_size

    def evict(self, kv_cache: torch.Tensor, current_len: int) -> torch.Tensor:
        """
        kv_cache shape: [layers, 2, seq_len, heads, d_head]
        Returns evicted cache with sink + window tokens.
        """
        if current_len <= self.max_tokens:
            return kv_cache

        sink = kv_cache[:, :, :self.sink_tokens]
        window = kv_cache[:, :, current_len - self.window_size:]
        # Concatenate sinks and recent window
        return torch.cat([sink, window], dim=2)
```

### Monitoring KV Cache Utilization

```python
def log_kv_metrics(block_manager, step: int):
    """Log KV cache utilization metrics for monitoring."""
    total = block_manager.total_num_gpu_blocks
    free = block_manager.get_num_free_gpu_blocks()
    used = total - free

    metrics = {
        "step": step,
        "kv_blocks_total": total,
        "kv_blocks_used": used,
        "kv_blocks_free": free,
        "kv_utilization_pct": 100.0 * used / total,
        "kv_cache_hit_rate": block_manager.get_prefix_cache_hit_rate(),
    }

    # Alert if utilization exceeds threshold
    if metrics["kv_utilization_pct"] > 90.0:
        logger.warning(
            f"KV cache utilization at {metrics['kv_utilization_pct']:.1f}%, "
            f"preemption likely. Consider reducing max_num_seqs or max_model_len."
        )
    return metrics
```

## Quick Reference: Choosing a KV Strategy

| Constraint | Recommended Strategy |
|-----------|---------------------|
| Memory-limited, short contexts | FP8 KV cache quantization |
| Memory-limited, long contexts | StreamingLLM or SnapKV eviction |
| High-throughput chat serving | PagedAttention + prefix caching |
| Multi-turn conversations | RadixAttention (SGLang) |
| Shared system prompts | Automatic prefix caching (vLLM) |
| Beam search workloads | PagedAttention with CoW |
| Ultra-long documents (>128K) | MLA models (DeepSeek) or KV eviction |
| Multi-GPU serving | TP with disaggregated prefill |
| Latency-sensitive | Speculative decoding with tree attention |
