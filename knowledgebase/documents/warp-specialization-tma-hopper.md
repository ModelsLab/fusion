---
id: warp_specialization_tma_hopper
kind: document
title: Warp Specialization, TMA, and Hopper/Blackwell Advanced Patterns
category: kernel
summary: Deep guide to Hopper and Blackwell advanced kernel patterns including warp specialization, TMA (Tensor Memory Accelerator), thread block clusters, distributed shared memory, WGMMA, and named barriers.
tags:
  - hopper
  - blackwell
  - warp-specialization
  - tma
  - wgmma
  - distributed-shared-memory
  - thread-block-clusters
  - named-barriers
  - persistent-kernels
source_ids:
  - nvidia-cuda-programming-guide
gpu_families:
  - Hopper
  - Blackwell
operators:
  - matmul
  - attention
  - gemm
precision:
  - fp16
  - bf16
  - fp8
  - fp4
---

# Warp Specialization, TMA, and Hopper/Blackwell Advanced Patterns

## Tensor Memory Accelerator (TMA)

### What TMA Is
TMA is a dedicated hardware unit on Hopper+ that handles bulk async memory copies independently of warp execution. Warps don't stall waiting for data - TMA handles it in the background.

### TMA Operations
```cpp
// 1D bulk copy: global → shared memory
// No warp threads are consumed!
cute::copy(tma_load_A, tA_gA(_, _, k), tA_sA(_, _, smem_pipe_write.index()));

// 2D tiled copy (most common for GEMM):
// Copies a rectangular tile from global to shared memory
// Handles out-of-bounds automatically (fills with zeros)

// TMA descriptor setup:
CUtensorMap tensor_map;
cuTensorMapEncodeTiled(
    &tensor_map,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
    2,                          // 2D tensor
    global_ptr,                 // base pointer
    {N, K},                     // global dimensions
    {N * sizeof(half)},         // global strides (in bytes)
    {TILE_N, TILE_K},          // box dimensions (tile size)
    {1, 1},                     // element strides
    CU_TENSOR_MAP_INTERLEAVE_NONE,
    CU_TENSOR_MAP_SWIZZLE_128B,  // swizzle for bank-conflict-free access
    CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
);
```

### TMA vs cp.async (Ampere)
| Feature | cp.async (Ampere) | TMA (Hopper+) |
|---------|------------------|---------------|
| Initiator | Warp threads | Dedicated hardware unit |
| Warp stall | Threads issue copy, then can do other work | No warp involvement at all |
| Shapes | 1D vectorized (4/8/16 bytes per thread) | 2D/3D tiles (arbitrary rectangle) |
| Bounds checking | Manual (masks) | Automatic (fills OOB with zeros) |
| Swizzle | Manual | Hardware-accelerated |
| Throughput | Limited by warp issue rate | Dedicated DMA engine |

### TMA Pipeline Pattern
```cpp
// Producer thread (or warp):
for (int k = 0; k < num_k_tiles; k++) {
    // TMA initiates bulk copy - producer thread is FREE after this
    cute::copy(tma_load_A, gA_tile(k), sA(pipe_write));
    cute::copy(tma_load_B, gB_tile(k), sB(pipe_write));

    // Signal that data will arrive
    pipeline.producer_arrive(pipe_write);
    pipe_write = next_stage(pipe_write);
}

// Consumer warp group (separate from producer):
for (int k = 0; k < num_k_tiles; k++) {
    // Wait for TMA to complete
    pipeline.consumer_wait(pipe_read);

    // Compute using WGMMA on shared memory data
    cute::gemm(tiled_mma, sA(pipe_read), sB(pipe_read), accum);

    // Release stage for reuse
    pipeline.consumer_release(pipe_read);
    pipe_read = next_stage(pipe_read);
}
```

## Warp Specialization

### Concept
Instead of all warps doing the same thing (cooperative), specialize warps for different roles:

```
Traditional (Cooperative):
  All warps: [load A][load B][sync][compute][sync][load A][load B][sync][compute]
  → Memory and compute are serialized

Warp-Specialized:
  Producer warps: [load A+B stage 0][load A+B stage 1][load A+B stage 2]...
  Consumer warps: [wait stage 0][compute stage 0][wait stage 1][compute stage 1]...
  → Memory and compute overlap perfectly!
```

### Implementation with Named Barriers
```cpp
// Named barriers (Hopper): up to 16 barriers per CTA
// Used to synchronize producer and consumer warp groups

// Producer warp group:
__syncthreads();  // initial sync
for (int stage = 0; stage < num_stages; stage++) {
    // Issue TMA loads for this stage
    if (warp_group_role == PRODUCER) {
        tma_load(smem_A[stage], gmem_A[k_tile + stage]);
        tma_load(smem_B[stage], gmem_B[k_tile + stage]);
        // Arrive on named barrier (signal data ready)
        named_barrier_arrive(barrier_id[stage], producer_count);
    }
}

// Consumer warp group:
for (int stage = 0; stage < num_stages; stage++) {
    if (warp_group_role == CONSUMER) {
        // Wait for data
        named_barrier_wait(barrier_id[stage], expected_count);
        // Compute WGMMA
        wgmma(smem_A[stage], smem_B[stage], accum);
        // Signal that we're done with this stage
        named_barrier_arrive(consumer_done_barrier[stage], consumer_count);
    }
}
```

### Warp Group Allocation
```
Typical H100 GEMM kernel (8 warps = 256 threads):
  Warp Group 0 (warps 0-3): PRODUCER - issues TMA loads
  Warp Group 1 (warps 4-7): CONSUMER - executes WGMMA

Alternative (ping-pong with 2 consumer groups):
  Warp Group 0 (warps 0-1): PRODUCER
  Warp Group 1 (warps 2-5): CONSUMER A - computes even stages
  Warp Group 2 (warps 6-7): CONSUMER B - computes odd stages
```

## WGMMA (Warp Group Matrix Multiply-Accumulate)

### What It Is
WGMMA operates at the warp group level (128 threads = 4 warps), executing much larger matrix operations per instruction than warp-level MMA.

```
Ampere mma.sync: 16x8x16 per warp (32 threads)
Hopper wgmma:    64x256x16 per warp group (128 threads)

→ 32x larger tile per instruction!
→ Better amortization of instruction overhead
```

### WGMMA Operand Sources
```
_SS suffix: Both A and B from Shared Memory
  wgmma.mma_async.sync.aligned.m64n256k16.f16.f16.f16 {D}, desc_A, desc_B
  → Most flexible, used when both operands need reloading

_RS suffix: A from Registers, B from Shared Memory
  wgmma.mma_async.sync.aligned.m64n256k16.f16.f16.f16 {D}, {A_regs}, desc_B
  → Used when A is already in registers (e.g., accumulated from previous WGMMA)
```

### Asynchronous Execution
```
// WGMMA is asynchronous - doesn't block the warp group
wgmma_async(desc_A, desc_B, accumulator);  // returns immediately
// ... can do other work here (e.g., compute softmax for attention)
wgmma_wait();  // wait for result when needed
```

## Thread Block Clusters

### What They Are
A cluster is a group of thread blocks that can access each other's shared memory via Distributed Shared Memory (DSMEM).

```
Traditional: each CTA has its own shared memory, isolated
Cluster: multiple CTAs share a "super shared memory" space

Cluster of 4 CTAs:
  CTA 0: smem_0 (228 KB) ──┐
  CTA 1: smem_1 (228 KB) ──┤── Can all access each other's SMEM
  CTA 2: smem_2 (228 KB) ──┤   via DSMEM load/store
  CTA 3: smem_3 (228 KB) ──┘
  Total accessible: 912 KB!
```

### Use Cases
1. **Larger tiles**: Access more shared memory for bigger GEMM tiles
2. **Multicast loads**: TMA can broadcast to multiple CTAs in a cluster
3. **Cross-CTA reduction**: Reduce across CTAs without global memory
4. **Split-K within cluster**: Different CTAs compute partial results, combine via DSMEM

### Cluster Launch
```cpp
// Set cluster size in kernel attributes
cudaFuncSetAttribute(kernel, cudaFuncAttributeClusterDimsMustBeSet, 1);
cudaFuncSetAttribute(kernel, cudaFuncAttributeRequiredClusterWidth, 2);
cudaFuncSetAttribute(kernel, cudaFuncAttributeRequiredClusterHeight, 2);

// Launch with cluster
dim3 grid(num_blocks_x, num_blocks_y);
dim3 block(256);
dim3 cluster(2, 2);  // 2x2 cluster = 4 CTAs per cluster

cudaLaunchKernelEx(&config, kernel, args...);
```

## Persistent Kernels

### Concept
Launch exactly as many CTAs as there are SMs. Each CTA loops over multiple work items, avoiding repeated kernel launch overhead.

```cpp
__global__ void persistent_gemm(
    ProblemDesc* problems,
    int num_problems,
    int* global_tile_counter  // atomic counter for work distribution
) {
    // Each CTA loops until all work is done
    while (true) {
        // Atomically claim next work tile
        int tile_id = atomicAdd(global_tile_counter, 1);
        if (tile_id >= total_tiles) break;

        // Compute which problem and which tile within that problem
        auto [problem_id, local_tile] = decompose_tile(tile_id, problems);

        // Execute the tile
        compute_gemm_tile(problems[problem_id], local_tile);
    }
}

// Launch with exactly num_SMs blocks:
persistent_gemm<<<num_SMs, 256>>>(problems, num_problems, &counter);
```

### Benefits
1. **Zero launch overhead for subsequent tiles** (same CTA keeps running)
2. **Better load balancing** (fast CTAs automatically grab more work)
3. **Resource persistence** (shared memory, registers stay allocated)
4. **Reduced scheduler pressure** (fewer CTA launches)

### When to Use
- Decode step: many small GEMMs per iteration
- Grouped GEMM: variable-size problems
- Stream-K: balanced work distribution across SMs

## Blackwell-Specific Features

### Second-Gen Transformer Engine
- Micro-tensor scaling: finer granularity scaling for FP8/FP4
- Block-level scale factors instead of per-tensor
- Better accuracy with same throughput

### FP4 (E2M1) Tensor Cores
```
FP4 values: {0, 0.5, 1, 1.5, 2, 3, 4, 6} and negatives
Must use block scaling (MX format) for useful dynamic range

Blackwell FP4 GEMM:
- 2x throughput vs FP8
- 4x throughput vs FP16
- Block scale factor per 32 elements
```

### Decompression Engine
Hardware unit that decompresses data during memory loads:
- LZ4 decompression in hardware
- Useful for compressed model weights
- Reduces memory bandwidth requirements

### Enhanced TMA
- Larger transfer sizes
- Better 3D tile support
- Multicast improvements for clusters

### NVLink 5.0
- 1800 GB/s bidirectional per GPU (2x NVLink 4.0)
- Better for tensor parallelism communication
- Enables larger effective memory pool across GPUs
