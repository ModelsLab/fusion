---
id: multi_gpu_distributed_kernels
kind: document
title: Multi-GPU and Distributed Kernel Optimization
category: distributed
summary: Complete guide to multi-GPU kernel patterns including tensor parallelism kernels, pipeline parallelism, NCCL optimization, NVLink/NVSwitch utilization, communication-computation overlap, and distributed attention.
tags:
  - multi-gpu
  - tensor-parallelism
  - pipeline-parallelism
  - nccl
  - nvlink
  - all-reduce
  - distributed
  - ring-attention
source_ids: []
operators:
  - allreduce
  - allgather
  - matmul
  - attention
gpu_families:
  - Ampere
  - Ada
  - Hopper
  - Blackwell
---

## 1. Tensor Parallelism Kernels

Tensor parallelism (TP) splits weight matrices across GPUs so each device computes a shard of a linear layer. The two fundamental patterns are column-parallel and row-parallel.

### Column-Parallel Linear (AllGather Output)

The weight matrix W is split column-wise: W = [W_0 | W_1 | ... | W_{tp-1}]. Each GPU holds W_i and computes Y_i = X @ W_i. Outputs are gathered to reconstruct the full result.

```python
import torch
import torch.distributed as dist

class ColumnParallelLinear(torch.nn.Module):
    """Column-parallel linear: each rank holds W[:, rank_slice]."""

    def __init__(self, in_features: int, out_features: int, tp_group: dist.ProcessGroup):
        super().__init__()
        self.tp_size = dist.get_world_size(tp_group)
        self.tp_group = tp_group
        assert out_features % self.tp_size == 0
        self.local_out = out_features // self.tp_size
        self.weight = torch.nn.Parameter(torch.empty(self.local_out, in_features))
        self.bias = torch.nn.Parameter(torch.empty(self.local_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Local GEMM: (batch, seq, in) @ (in, local_out) -> (batch, seq, local_out)
        y_local = torch.nn.functional.linear(x, self.weight, self.bias)
        # AllGather across TP group to get (batch, seq, out_features)
        y_gathered = [torch.empty_like(y_local) for _ in range(self.tp_size)]
        dist.all_gather(y_gathered, y_local, group=self.tp_group)
        return torch.cat(y_gathered, dim=-1)
```

### Row-Parallel Linear (AllReduce Output)

The weight matrix is split row-wise: W = [W_0; W_1; ...]. Input must already be partitioned (typically the output of a preceding column-parallel layer). Each GPU computes Y_i = X_i @ W_i, then results are summed via AllReduce.

```python
class RowParallelLinear(torch.nn.Module):
    """Row-parallel linear: each rank holds W[rank_slice, :]."""

    def __init__(self, in_features: int, out_features: int, tp_group: dist.ProcessGroup):
        super().__init__()
        self.tp_size = dist.get_world_size(tp_group)
        self.tp_group = tp_group
        assert in_features % self.tp_size == 0
        self.local_in = in_features // self.tp_size
        self.weight = torch.nn.Parameter(torch.empty(out_features, self.local_in))
        self.bias = torch.nn.Parameter(torch.empty(out_features))

    def forward(self, x_shard: torch.Tensor) -> torch.Tensor:
        # x_shard: (batch, seq, local_in) -- already partitioned from column-parallel
        y_local = torch.nn.functional.linear(x_shard, self.weight)
        # AllReduce sums partial products across ranks
        dist.all_reduce(y_local, op=dist.ReduceOp.SUM, group=self.tp_group)
        return y_local + self.bias
```

### Attention Head Parallelism

Multi-head attention naturally parallelizes across heads. With TP=N, each GPU handles H/N heads. QKV projection is column-parallel (splitting heads), output projection is row-parallel (AllReduce to merge).

```
GPU 0: heads 0..7    -> Q0,K0,V0 -> Attn0 -> O0
GPU 1: heads 8..15   -> Q1,K1,V1 -> Attn1 -> O1
                                              |
                        RowParallel OutProj + AllReduce -> final output
```

### Bandwidth Requirements for TP

For a model with hidden dimension H and TP degree T on a transformer block:

```
AllReduce volume per layer (forward) = 2 * batch * seq * H bytes  (one for MLP, one for Attn)
AllReduce volume per layer (backward) = 4 * batch * seq * H bytes
Total per-layer comm (fwd+bwd) = 6 * batch * seq * H bytes

Example: H=8192 (70B model), batch*seq=4096, bf16:
  Per-layer = 6 * 4096 * 8192 * 2 bytes = 384 MB
  80 layers => 30.7 GB total per step
  NVLink 4.0 bidirectional BW = 900 GB/s => ~34 ms ideal
```

## 2. Pipeline Parallelism

Pipeline parallelism (PP) assigns contiguous groups of layers to different GPUs. The key challenge is reducing the pipeline bubble.

### Micro-Batching and 1F1B Schedule

The 1F1B (one-forward-one-backward) schedule minimizes memory by interleaving forward and backward micro-batches.

```
Timeline for PP=4, micro_batches=8, 1F1B schedule:

GPU 0: F0 F1 F2 F3 | B0 F4 | B1 F5 | B2 F6 | B3 F7 | B4 B5 B6 B7
GPU 1:    F0 F1 F2 | B0 F3 | B1 F4 | B2 F5 | B3 F6 | B4 F7 B5 B6 B7
GPU 2:       F0 F1 | B0 F2 | B1 F3 | B2 F4 | B3 F5 | B4 F6 B5 F7 B6 B7
GPU 3:          F0 | B0 F1 | B1 F2 | B2 F3 | B3 F4 | B4 F5 B5 F6 B6 F7 B7

Bubble fraction = (PP - 1) / (num_microbatches + PP - 1)
With PP=4, M=8: bubble = 3/11 = 27%
With PP=4, M=32: bubble = 3/35 = 8.6%
```

### Interleaved Stages (Virtual Pipeline Parallelism)

Each GPU holds multiple non-contiguous stage chunks. For a 32-layer model with PP=4 and virtual_stages=2:

```
GPU 0: layers 0-3,   16-19
GPU 1: layers 4-7,   20-23
GPU 2: layers 8-11,  24-27
GPU 3: layers 12-15, 28-31

Bubble fraction with interleaving = (PP - 1) / (num_microbatches * virtual_stages + PP - 1)
With PP=4, M=8, V=2: bubble = 3/19 = 15.8%  (down from 27%)
```

### Point-to-Point Communication for PP

```python
def pipeline_send_recv(tensor, src_rank, dst_rank, group):
    """Bidirectional point-to-point for pipeline stage boundary."""
    ops = []
    if dist.get_rank() == src_rank:
        ops.append(dist.P2POp(dist.isend, tensor, dst_rank, group))
    if dist.get_rank() == dst_rank:
        recv_buf = torch.empty_like(tensor)
        ops.append(dist.P2POp(dist.irecv, recv_buf, src_rank, group))
    reqs = dist.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()
```

## 3. NCCL Optimization

### Ring vs Tree AllReduce

```
Ring AllReduce:
  - Latency: 2 * (N-1) * alpha + 2 * ((N-1)/N) * S * beta
  - Optimal for large messages (>256 KB per GPU)
  - Bandwidth-optimal: uses (N-1)/N of link bandwidth

Tree AllReduce:
  - Latency: 2 * log2(N) * alpha + 2 * log2(N) * S * beta
  - Better for small messages (latency-bound)
  - NCCL auto-selects based on message size
```

### SHARP In-Network Reduction

Mellanox SHARP offloads AllReduce to the network switch ASIC, halving inter-node traffic.

```
Without SHARP:  reduce-scatter + all-gather = 2 * S * (N-1)/N  across network
With SHARP:     each node sends S once to switch, switch reduces, broadcasts back
                = 2 * S  total network traffic (independent of N)
```

### Key NCCL Environment Variables

```bash
# Algorithm selection
export NCCL_ALGO=Ring            # Ring, Tree, CollnetDirect, CollnetChain
export NCCL_PROTO=Simple         # Simple, LL, LL128
export NCCL_MIN_NCHANNELS=8      # Minimum number of channels
export NCCL_MAX_NCHANNELS=32     # Maximum number of channels

# Tuning for large clusters
export NCCL_BUFFSIZE=8388608     # 8MB per-channel buffer (default 4MB)
export NCCL_NTHREADS=512         # Threads per NCCL block
export NCCL_CROSS_NIC=1          # Allow cross-NIC communication

# Debugging
export NCCL_DEBUG=INFO           # WARN, INFO, TRACE
export NCCL_DEBUG_SUBSYS=INIT,NET  # Subsystem-specific debug

# IB / network
export NCCL_IB_HCA=mlx5_0,mlx5_1  # Select specific HCAs
export NCCL_IB_GID_INDEX=3         # RoCE GID index
export NCCL_SOCKET_IFNAME=eth0     # Control plane interface

# Performance
export NCCL_P2P_LEVEL=NVL          # NVL, PIX, PXB, PHB, SYS
export NCCL_SHM_DISABLE=0          # Enable shared memory for intra-node
export NCCL_NET_GDR_LEVEL=5        # GPUDirect RDMA aggressiveness
```

### Custom NCCL Plugin Pattern

```c
// Skeleton for a custom NCCL net plugin
#include "nccl_net.h"

ncclResult_t pluginInit(ncclDebugLogger_t logger) {
    // Initialize custom transport (e.g., custom RDMA, DPDK)
    return ncclSuccess;
}

ncclResult_t pluginSend(void* sendComm, void* data, int size,
                        int tag, void** request) {
    // Post non-blocking send on custom fabric
    return ncclSuccess;
}

ncclNet_t NCCL_PLUGIN_SYMBOL = {
    .name = "CustomFabric",
    .init = pluginInit,
    .send = pluginSend,
    .recv = pluginRecv,
    // ... remaining ops
};
```

## 4. NVLink and NVSwitch

### Topology Overview

```
DGX A100 (Ampere):
  - 8x A100 GPUs, NVLink 3.0, 12 links/GPU
  - 600 GB/s bidirectional per GPU
  - NVSwitch v2: full all-to-all bisection bandwidth
  - Total NVLink fabric: 4.8 TB/s

DGX H100 (Hopper):
  - 8x H100 GPUs, NVLink 4.0, 18 links/GPU
  - 900 GB/s bidirectional per GPU
  - NVSwitch v3: 3.6 TB/s per switch, 2 switches
  - Total NVLink fabric: 7.2 TB/s

DGX B200 (Blackwell):
  - 8x B200 GPUs, NVLink 5.0, 18 links/GPU
  - 1800 GB/s bidirectional per GPU
  - NVSwitch v4: 14.4 TB/s total fabric
  - 5th-gen NVLink doubles per-lane bandwidth
```

### P2P vs Collective Communication

```python
# P2P: explicit send/recv between two GPUs
# Best for pipeline parallelism stage boundaries
# Uses direct NVLink paths, latency ~1-5 us intra-node

# Collective: AllReduce, AllGather, ReduceScatter
# Best for tensor parallelism
# NVSwitch enables single-hop all-to-all

# Bandwidth utilization check
def nvlink_utilization(message_bytes, elapsed_seconds, nvlink_bw_gbs):
    """Calculate NVLink utilization percentage."""
    achieved_gbs = message_bytes / elapsed_seconds / 1e9
    utilization = achieved_gbs / nvlink_bw_gbs * 100
    return utilization

# Example: 128 MB AllReduce on H100 NVLink
# Ideal time = 128 MB / 900 GB/s = 0.142 ms
# Achieved time with NCCL ~ 0.16 ms => 89% utilization
```

### Topology-Aware Placement

```
DGX H100 NVSwitch topology (all-to-all):

  GPU0 ---NVLink--- GPU1
   |  \           /  |
   |   NVSwitch(s)   |
   |  /           \  |
  GPU2 ---NVLink--- GPU3
   ...same pattern for GPU4-7...

Key rules:
  - TP group should be within one NVSwitch domain (intra-node)
  - PP stages on adjacent GPUs or across nodes via InfiniBand
  - DP replicas across nodes
```

## 5. Communication-Computation Overlap

### Async AllReduce During Backward Pass

The gradient AllReduce for layer N can overlap with the backward compute of layer N-1.

```python
class OverlappedAllReduceLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, tp_group):
        ctx.save_for_backward(input, weight)
        ctx.tp_group = tp_group
        return torch.nn.functional.linear(input, weight)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        # Compute grad_weight locally
        grad_weight = grad_output.reshape(-1, grad_output.shape[-1]).T @ \
                      input.reshape(-1, input.shape[-1])
        # Launch async AllReduce for grad_weight -- overlaps with grad_input compute
        handle = dist.all_reduce(grad_weight, op=dist.ReduceOp.SUM,
                                 group=ctx.tp_group, async_op=True)
        # Compute grad_input while AllReduce is in flight
        grad_input = grad_output @ weight
        # Synchronize AllReduce before returning grad_weight
        handle.wait()
        return grad_input, grad_weight, None
```

### Split-K GEMM with Partial Reduce

For large K dimensions, split the K-reduction across GPUs and perform a partial AllReduce.

```python
def split_k_matmul(A_shard, B_shard, tp_group):
    """
    A is split along K: A_shard is (M, K/tp).
    B is split along K: B_shard is (K/tp, N).
    Each GPU computes partial C = A_shard @ B_shard, then AllReduce sums.
    """
    C_partial = torch.matmul(A_shard, B_shard)  # (M, N) partial sum
    dist.all_reduce(C_partial, op=dist.ReduceOp.SUM, group=tp_group)
    return C_partial
```

### Pipelining Communication and Compute via Chunking

```python
def chunked_allreduce_overlap(tensor, compute_fn, num_chunks, group):
    """
    Split tensor into chunks, overlap AllReduce of chunk i
    with compute on chunk i+1.
    """
    chunks = tensor.chunk(num_chunks, dim=0)
    results = []
    prev_handle = None
    for i, chunk in enumerate(chunks):
        if prev_handle is not None:
            prev_handle.wait()
        handle = dist.all_reduce(chunk, group=group, async_op=True)
        if i + 1 < len(chunks):
            # Overlap: do compute on next chunk while current AllReduce runs
            compute_fn(chunks[i + 1])
        prev_handle = handle
    if prev_handle is not None:
        prev_handle.wait()
    return torch.cat(chunks, dim=0)
```

## 6. Sequence Parallelism

### Megatron Sequence Parallelism (SP)

In Megatron-SP, LayerNorm and Dropout operate on sequence-sharded activations. An AllGather reconstructs the full sequence before attention/MLP, and a ReduceScatter re-shards afterward.

```
Forward pass through one transformer block with TP+SP:

  Input: (batch, seq/tp, hidden)  -- sequence sharded
         |
    [AllGather]  -> (batch, seq, hidden)
         |
    [ColumnParallel QKV]  -> (batch, seq, hidden/tp) per head group
         |
    [Attention]  -> (batch, seq, hidden/tp)
         |
    [RowParallel OutProj]  -- includes ReduceScatter fused
         |
  Output: (batch, seq/tp, hidden)  -- sequence sharded again
         |
    [LayerNorm]  -- operates on local shard only, no comm
         |
    [AllGather]  -> (batch, seq, hidden)
         |
    [ColumnParallel MLP Up]
    [RowParallel MLP Down + ReduceScatter]
         |
  Output: (batch, seq/tp, hidden)

Memory savings: activation memory reduced by factor of TP for LayerNorm/Dropout.
Communication: replaces AllReduce with AllGather + ReduceScatter (same total volume).
```

### DeepSpeed Ulysses (All-to-All Attention)

Ulysses partitions the sequence across GPUs and uses All-to-All to redistribute Q, K, V for attention computation across heads.

```python
def ulysses_attention(q_local, k_local, v_local, sp_group):
    """
    q_local: (batch, seq/sp, num_heads, head_dim) -- sequence sharded
    After all-to-all: (batch, seq, num_heads/sp, head_dim) -- head sharded
    """
    sp_size = dist.get_world_size(sp_group)

    # All-to-All: redistribute from seq-sharded to head-sharded
    q_head_shard = all_to_all_reshape(q_local, sp_group,
                                       scatter_dim=2, gather_dim=1)
    k_head_shard = all_to_all_reshape(k_local, sp_group,
                                       scatter_dim=2, gather_dim=1)
    v_head_shard = all_to_all_reshape(v_local, sp_group,
                                       scatter_dim=2, gather_dim=1)

    # Standard attention on full sequence, subset of heads
    attn_out = flash_attention(q_head_shard, k_head_shard, v_head_shard)

    # All-to-All back: head-sharded -> seq-sharded
    out_seq_shard = all_to_all_reshape(attn_out, sp_group,
                                        scatter_dim=1, gather_dim=2)
    return out_seq_shard

    # Communication volume: 2 * All-to-All = 2 * batch * seq * hidden * (sp-1)/sp
```

### Ring Attention (Sequence Sharding with Streaming KV)

Ring Attention shards the sequence and passes KV blocks around a ring. Each GPU computes partial attention with its local Q against rotating KV chunks.

```python
def ring_attention(q_local, k_local, v_local, sp_group):
    """
    Each GPU holds (batch, seq/sp, heads, dim) for Q, K, V.
    K,V blocks rotate around the ring while Q stays local.
    """
    sp_size = dist.get_world_size(sp_group)
    rank = dist.get_rank(sp_group)

    # Running online softmax accumulators
    acc_out = torch.zeros_like(q_local)
    acc_lse = torch.full((*q_local.shape[:-1], 1), float('-inf'),
                         device=q_local.device)

    k_buf, v_buf = k_local.clone(), v_local.clone()

    for step in range(sp_size):
        # Compute local attention block: Q_local @ K_buf^T -> scores -> V_buf
        block_out, block_lse = flash_attention_with_lse(q_local, k_buf, v_buf)

        # Online softmax merge
        acc_out, acc_lse = online_softmax_merge(acc_out, acc_lse,
                                                 block_out, block_lse)

        # Ring-shift K,V to next rank (async)
        if step < sp_size - 1:
            next_rank = (rank + 1) % sp_size
            prev_rank = (rank - 1) % sp_size
            k_recv = torch.empty_like(k_buf)
            v_recv = torch.empty_like(v_buf)
            send_ops = [dist.P2POp(dist.isend, k_buf, next_rank, sp_group),
                        dist.P2POp(dist.isend, v_buf, next_rank, sp_group)]
            recv_ops = [dist.P2POp(dist.irecv, k_recv, prev_rank, sp_group),
                        dist.P2POp(dist.irecv, v_recv, prev_rank, sp_group)]
            reqs = dist.batch_isend_irecv(send_ops + recv_ops)
            for r in reqs:
                r.wait()
            k_buf, v_buf = k_recv, v_recv

    return acc_out
```

## 7. Expert Parallelism for MoE

### All-to-All Dispatch and Combine

Mixture-of-Experts routes tokens to expert-holding GPUs via All-to-All.

```python
def moe_dispatch_combine(hidden_states, router_logits, expert_group, num_experts, top_k):
    """
    1. Route: compute top-k expert assignments per token
    2. Dispatch: All-to-All sends tokens to expert-holding GPUs
    3. Compute: each GPU runs its local experts on received tokens
    4. Combine: All-to-All returns results, weighted sum by router probs
    """
    ep_size = dist.get_world_size(expert_group)
    experts_per_rank = num_experts // ep_size

    # Router: (batch*seq, num_experts) -> top_k experts per token
    scores = torch.softmax(router_logits, dim=-1)
    topk_weights, topk_indices = torch.topk(scores, top_k, dim=-1)

    # Build dispatch buffers: sort tokens by target expert rank
    send_counts, recv_counts = compute_send_recv_counts(topk_indices, ep_size)
    dispatched = all_to_all_variable(hidden_states, send_counts, recv_counts,
                                      expert_group)

    # Local expert computation
    expert_outputs = run_local_experts(dispatched, experts_per_rank)

    # Combine: All-to-All returns processed tokens
    combined = all_to_all_variable(expert_outputs, recv_counts, send_counts,
                                    expert_group)
    return weighted_sum(combined, topk_weights)
```

### EP + TP Combinations

```
Common MoE parallelism layouts for a 16-expert model on 8 GPUs:

EP=8, TP=1:  Each GPU holds 2 experts, no tensor parallelism within experts
EP=4, TP=2:  4 expert groups (4 experts each), each expert tensor-parallel across 2 GPUs
EP=2, TP=4:  2 expert groups, heavy TP within each expert

Tradeoff:
  - Higher EP = less expert computation per GPU but more All-to-All traffic
  - Higher TP = AllReduce within experts but less dispatch traffic
  - Rule of thumb: use EP when expert FFN is small, TP when expert FFN is large
```

### Capacity Factor Impact

```
capacity_factor = (tokens_per_expert * num_experts) / (total_tokens * top_k)

CF = 1.0: exactly balanced, no dropped tokens, no padding
CF = 1.25: 25% padding buffer, handles moderate imbalance
CF > 1.5: wasteful padding, consider auxiliary load-balancing loss

Communication volume for dispatch All-to-All:
  = 2 * total_tokens * top_k * hidden_dim * sizeof(dtype) * (ep_size - 1) / ep_size

Example: 4096 tokens, top_k=2, hidden=4096, bf16, EP=8:
  = 2 * 4096 * 2 * 4096 * 2 * 7/8 = 115 MB per All-to-All
```

## 8. Distributed Attention (Context Parallelism)

### Splitting KV Across GPUs

Context parallelism (CP) splits the KV cache across GPUs for long-context inference, distinct from TP which splits heads.

```
Context parallelism for 128K sequence on CP=4:

GPU 0: KV for tokens     0 - 32767    (processes queries for all tokens)
GPU 1: KV for tokens 32768 - 65535
GPU 2: KV for tokens 65536 - 98303
GPU 3: KV for tokens 98304 - 131071

Each GPU computes partial attention with its KV shard, then
AllReduce the weighted outputs using online softmax correction.
```

### Distributed FlashAttention

```python
def distributed_flash_attention(q, k_local, v_local, cp_group):
    """
    q: (batch, seq, heads, dim) -- full query on every GPU
    k_local, v_local: (batch, seq/cp, heads, dim) -- KV sharded across cp_group

    Each GPU computes FlashAttention with full Q and local KV,
    then merges partial outputs using log-sum-exp correction.
    """
    cp_size = dist.get_world_size(cp_group)

    # Local partial attention with FlashAttention kernel
    partial_out, partial_lse = flash_attn_func(
        q, k_local, v_local, return_lse=True
    )
    # partial_out: (batch, seq, heads, dim)
    # partial_lse: (batch, heads, seq)  -- log-sum-exp for softmax denominator

    # Gather all partial outputs and LSEs
    all_outs = [torch.empty_like(partial_out) for _ in range(cp_size)]
    all_lses = [torch.empty_like(partial_lse) for _ in range(cp_size)]
    dist.all_gather(all_outs, partial_out, group=cp_group)
    dist.all_gather(all_lses, partial_lse, group=cp_group)

    # Online softmax merge across partitions
    merged_out = all_outs[0]
    merged_lse = all_lses[0]
    for i in range(1, cp_size):
        new_lse = torch.logaddexp(merged_lse, all_lses[i])
        w_old = torch.exp(merged_lse - new_lse).unsqueeze(-1)
        w_new = torch.exp(all_lses[i] - new_lse).unsqueeze(-1)
        merged_out = w_old * merged_out + w_new * all_outs[i]
        merged_lse = new_lse

    return merged_out
```

## 9. Quantized Communication

### FP8 AllReduce

Reducing communication precision from BF16 to FP8 halves bandwidth requirements.

```python
def fp8_allreduce(tensor_bf16, group):
    """
    Quantize to FP8 before AllReduce, dequantize after.
    Saves 50% bandwidth at cost of ~0.1% training quality.
    """
    # Compute per-tensor scale
    amax = tensor_bf16.abs().max()
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scale = fp8_max / amax.clamp(min=1e-12)

    # Quantize
    tensor_fp8 = (tensor_bf16 * scale).to(torch.float8_e4m3fn)

    # AllReduce in FP8 (requires NCCL 2.19+ with FP8 support)
    # Fallback: cast to int8 for transport, reduce in fp32 on receive
    tensor_i8 = tensor_fp8.view(torch.int8)
    dist.all_reduce(tensor_i8, group=group)  # bitwise transport

    # Dequantize
    result = tensor_i8.view(torch.float8_e4m3fn).to(torch.bfloat16) / scale
    return result
```

### Gradient Compression with PowerSGD

```python
class PowerSGDCompressor:
    """
    Low-rank gradient compression. Reduces AllReduce volume from
    M*N to (M+N)*rank, typically 10-100x compression.
    """
    def __init__(self, rank=4):
        self.rank = rank
        self.Q = None  # Maintained across steps for warm-start

    def compress_and_allreduce(self, grad, group):
        M, N = grad.shape
        if self.Q is None:
            self.Q = torch.randn(N, self.rank, device=grad.device)

        # Compress: project gradient onto low-rank basis
        P = grad @ self.Q                    # (M, rank)
        dist.all_reduce(P, group=group)      # AllReduce small matrix

        # Orthogonalize
        P, _ = torch.linalg.qr(P)

        # Second projection
        Q = grad.T @ P                       # (N, rank)
        dist.all_reduce(Q, group=group)      # AllReduce small matrix

        self.Q = Q  # warm-start for next iteration

        # Decompress
        return P @ Q.T  # (M, N) approximation of sum of gradients

        # Communication: 2 * (M + N) * rank * sizeof(float)
        # vs baseline:       M * N * sizeof(float)
        # Example: M=N=8192, rank=4: compression ratio = 8192*8192 / (2*8192*4*2) = 512x
```

## 10. Topology-Aware Placement

### DGX Configurations and Mapping Strategy

```
DGX H100 (8 GPUs per node):
  Intra-node: NVLink 4.0 (900 GB/s bidirectional per GPU)
  Inter-node: 8x 400G InfiniBand NDR (3.2 Tb/s = 400 GB/s per node)

Bandwidth ratio: NVLink / IB = 900 / 50 = 18x per GPU
=> Minimize inter-node communication, keep TP intra-node

Recommended parallelism mapping for 128-GPU training (16 nodes):

  TP=8  (within one DGX node, uses NVLink)
  PP=4  (across 4 nodes, uses InfiniBand for small activations)
  DP=4  (across 4 node groups, gradient AllReduce)

  Node 0:  GPU 0-7  = TP group 0,  PP stage 0,  DP group 0
  Node 1:  GPU 8-15 = TP group 1,  PP stage 1,  DP group 0
  Node 2:  GPU 16-23= TP group 2,  PP stage 2,  DP group 0
  Node 3:  GPU 24-31= TP group 3,  PP stage 3,  DP group 0
  Node 4:  GPU 32-39= TP group 4,  PP stage 0,  DP group 1
  ...
```

### Mapping Model Dimensions to Topology

```python
def compute_parallelism_config(
    model_params_B: float,
    num_layers: int,
    hidden_dim: int,
    num_heads: int,
    total_gpus: int,
    gpus_per_node: int,
    gpu_memory_gb: float,
    nvlink_bw_gbs: float,
    ib_bw_gbs: float,
):
    """
    Heuristic for parallelism assignment.

    Rules:
    1. TP <= gpus_per_node (never cross node boundary)
    2. TP must divide num_heads evenly
    3. PP * TP <= total_gpus
    4. DP = total_gpus / (TP * PP)
    5. Minimize PP to reduce bubble overhead
    6. Increase TP before PP if model fits with given TP
    """
    # Memory per GPU estimate (bf16, with activations)
    param_memory_gb = model_params_B * 2  # bf16 parameters
    optimizer_memory_gb = model_params_B * 12  # Adam states in fp32

    for tp in [8, 4, 2, 1]:
        if tp > gpus_per_node:
            continue
        if num_heads % tp != 0:
            continue
        per_gpu_params = param_memory_gb / tp
        for pp in range(1, total_gpus // tp + 1):
            per_gpu_layers = num_layers / pp
            per_gpu_mem = per_gpu_params / pp + optimizer_memory_gb / (tp * pp)
            if per_gpu_mem < gpu_memory_gb * 0.85:  # 85% memory threshold
                dp = total_gpus // (tp * pp)
                if dp >= 1:
                    return {"TP": tp, "PP": pp, "DP": dp}
    raise ValueError("Cannot fit model with available GPUs")
```

### Bandwidth Planning Table

```
Operation          | Volume (per step)           | Preferred Link    | Overlap?
-------------------|-----------------------------|-------------------|----------
TP AllReduce       | 2*B*S*H per layer           | NVLink (intra)    | Partial
TP AllGather (SP)  | B*S*H per layer             | NVLink (intra)    | Yes
PP Send/Recv       | B*uB*H per micro-batch      | IB or NVLink      | Yes (1F1B)
DP AllReduce       | 2 * total_params            | IB (inter-node)   | Yes (backward)
MoE All-to-All     | 2*tokens*topk*H             | NVLink + IB       | Partial
CP AllGather       | B*S*H*heads (KV)            | NVLink            | With compute
Ring Attn P2P      | B*(S/CP)*H per step         | NVLink ring       | Yes

BF16 bandwidth requirements example (70B model, B=1, S=4096, H=8192, 80 layers):
  TP AllReduce:  80 * 2 * 4096 * 8192 * 2 = 10.2 GB/step
  At NVLink 900 GB/s: 11.4 ms ideal, ~13 ms with overhead
  DP AllReduce:  70B * 2 = 140 GB (once per step, overlapped with backward)
  At 400 GB/s IB: 350 ms, but overlapped across ~60% of backward
```

## Summary of Best Practices

1. Always keep TP within a single NVSwitch domain (one node). Crossing nodes for TP is bandwidth-prohibitive.
2. Use sequence parallelism (Megatron-SP) alongside TP to save activation memory at zero extra communication cost.
3. Prefer 1F1B with interleaved stages for PP. Target micro-batch count >= 4x PP degree.
4. Overlap DP gradient AllReduce with backward computation. Bucket gradients (128 MB default in PyTorch DDP).
5. For MoE, tune capacity factor to 1.0-1.25 and use auxiliary load-balancing loss to minimize All-to-All waste.
6. Use FP8 communication (NCCL 2.19+) when training allows; monitor for convergence impact.
7. Profile with NCCL_DEBUG=INFO and nsys to verify overlap and identify communication bottlenecks.
8. On Hopper and Blackwell, enable CUDA_DEVICE_MAX_CONNECTIONS=1 for best TP overlap with computation streams.
