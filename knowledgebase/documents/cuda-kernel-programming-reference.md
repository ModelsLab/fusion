---
id: cuda-kernel-programming-reference
kind: document
title: CUDA Kernel Programming In-Depth Reference
category: kernel-programming
summary: Authoritative reference covering the CUDA memory model, warp execution, occupancy, streams, graphs, PTX/SASS, async operations, thread block clusters, dynamic parallelism, and profiling with Nsight tools.
support_level: recommended
reliability: curated
review_status: reviewed
tags:
  - cuda
  - kernel
  - memory-model
  - warp
  - occupancy
  - streams
  - cuda-graphs
  - ptx
  - sass
  - async
  - tma
  - hopper
  - profiling
  - nsight
gpu_families:
  - Ampere
  - Ada Lovelace
  - Hopper
  - Blackwell
workloads:
  - prefill
  - decode
  - training
  - inference
operators:
  - gemm
  - attention
  - elementwise
  - reduction
precision:
  - fp32
  - fp16
  - bf16
  - fp8
backends:
  - cuda
  - cutlass
  - triton
reference_source_ids:
  - nvidia-cuda-programming-guide
---

# CUDA Kernel Programming In-Depth Reference

This document is an authoritative reference for CUDA kernel programming, covering hardware execution models, memory hierarchies, concurrency primitives, and profiling methodology. All information targets CUDA 12.x+ unless otherwise noted, with architecture-specific features called out by compute capability.

---

## 1. CUDA Memory Model

### 1.1 Memory Hierarchy Overview

| Memory Type | Scope | Lifetime | Latency (cycles) | Bandwidth | Cached | Size (typical) |
|---|---|---|---|---|---|---|
| Registers | Thread | Thread | 0-1 | Highest | N/A | 255 per thread (CC 7.x+) |
| Local Memory | Thread | Thread | ~200-800 | Same as global | L1+L2 | Up to stack limit |
| Shared Memory | Block | Block | ~20-30 | ~1.5 TB/s per SM | N/A | Up to 228 KB/SM (Hopper) |
| L1 Cache | SM | Automatic | ~30 | ~1.5 TB/s per SM | N/A | Configurable carveout |
| L2 Cache | Device | Automatic | ~200 | ~6 TB/s (H100) | N/A | 50 MB (H100) |
| Global Memory (HBM) | Device | Application | ~400-800 | 3 TB/s (H100) | L1+L2 | 80 GB (H100 SXM) |
| Constant Memory | Device | Application | ~4 (cached hit) | Broadcast | Dedicated 10 KB cache | 64 KB total |
| Texture Memory | Device | Application | ~4 (cached hit) | Via texture cache | Dedicated cache | Bound to global |

### 1.2 Global Memory

Global memory resides in HBM (or GDDR on consumer GPUs). All threads across all blocks can read and write global memory. Access latency is 400-800 cycles uncached.

**Coalescing Rules:**

Coalescing is the single most important optimization for global memory. The hardware combines memory requests from threads in a warp into as few memory transactions as possible.

- The memory system services requests in **32-byte**, **64-byte**, or **128-byte** transactions aligned to those boundaries.
- **L1 cache line size: 128 bytes.** L2 cache line (sector) size: **32 bytes.**
- A fully coalesced 32-thread warp accessing consecutive 4-byte floats results in one 128-byte transaction.
- Misaligned or strided access patterns cause multiple transactions per warp, reducing effective bandwidth proportionally.

**Coalescing patterns by compute capability (CC 7.0+):**

```
// OPTIMAL: Consecutive threads access consecutive addresses (stride-1)
// One 128-byte L1 transaction for 32 x 4-byte elements
data[threadIdx.x]

// BAD: Stride-2 access, 50% bandwidth utilization
data[2 * threadIdx.x]

// WORST: Random access, potentially 32 separate transactions
data[random_index[threadIdx.x]]
```

**Alignment:**
- Structures should be aligned to their size or 16 bytes (use `__align__(16)`)
- `cudaMalloc` returns 256-byte-aligned pointers
- Use `__ldg()` for read-only global loads through the texture/read-only cache path

### 1.3 Shared Memory

On-chip memory shared by all threads in a block. Partitioned into **32 banks** (CC 2.0+), each 4 bytes wide, with consecutive 4-byte words mapped to consecutive banks.

**Bank Conflict Rules:**

```
Bank index = (byte_address / 4) % 32
```

- **No conflict**: Each thread accesses a different bank, or all threads access the same address (broadcast).
- **N-way conflict**: N threads access different addresses in the same bank. The access is serialized into N sequential accesses.
- **Multicast** (CC 2.0+): If multiple threads access the same address within a bank, the value is broadcast (no conflict).

**Bank conflict examples:**

```c
__shared__ float s[256];

// NO conflict: stride-1, each thread hits a different bank
s[threadIdx.x]          // thread 0 -> bank 0, thread 1 -> bank 1, ...

// NO conflict: stride-1 with offset
s[threadIdx.x + 3]      // all different banks

// 2-WAY conflict: stride-2, threads 0 and 16 hit bank 0
s[2 * threadIdx.x]

// 32-WAY conflict: all threads hit bank 0
s[32 * threadIdx.x]

// Padding trick to avoid conflicts in column-major access:
__shared__ float tile[32][33];  // 33 instead of 32 avoids conflicts on column access
```

**Shared memory configuration (Ampere+):**
- Volta/Turing: Unified 128 KB L1/shared per SM; shared configurable up to 96 KB
- Ampere (A100): Up to 164 KB shared per SM
- Hopper (H100): Up to 228 KB shared per SM
- Use `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)` for >48 KB

**Static vs Dynamic allocation:**

```c
// Static: size known at compile time
__shared__ float buffer[256];

// Dynamic: size specified at kernel launch
extern __shared__ float dyn_buffer[];
kernel<<<grid, block, shared_bytes>>>(args);
```

### 1.4 Local Memory

Per-thread private memory that physically resides in global memory (HBM) but is cached through L1/L2. The compiler spills registers to local memory when register pressure exceeds limits. Local memory accesses are coalesced across threads in a warp because consecutive thread IDs access consecutive addresses.

### 1.5 Constant Memory

64 KB total, cached in a dedicated 10 KB per-SM constant cache. Optimized for the case where all threads in a warp read the same address (broadcast). If threads read different addresses, accesses are serialized.

```c
__constant__ float params[1024];
cudaMemcpyToSymbol(params, host_data, sizeof(float) * 1024);
```

### 1.6 Texture Memory

Provides hardware-accelerated interpolation, boundary clamping, and a separate cache hierarchy. Primarily useful for:
- 2D spatial locality patterns (image processing)
- Read-only data with non-coalesced access patterns
- Free hardware bilinear interpolation

Since Kepler, the read-only data cache (`__ldg()`) provides many of the same caching benefits without the texture API overhead.

### 1.7 Memory Fence Operations

Memory fences enforce ordering of memory operations but do NOT synchronize threads.

```c
__threadfence_block();   // Ensures writes visible to all threads in same block
__threadfence();         // Ensures writes visible to all threads in device
__threadfence_system();  // Ensures writes visible to all threads + host (unified memory)
```

**When to use:** After an `atomicCAS` lock acquisition, use `__threadfence()` before accessing the protected data to ensure all prior writes are visible. Use `__threadfence_block()` when the producer/consumer threads are in the same block.

### 1.8 Atomic Operations

Atomic operations guarantee read-modify-write without data races. Performance characteristics:

| Operation | Location | CC 7.x+ Throughput |
|---|---|---|
| `atomicAdd` (int32) | Global | Full speed on dedicated units |
| `atomicAdd` (float) | Global | Native since CC 2.0 |
| `atomicAdd` (double) | Global | Native since CC 6.0 |
| `atomicAdd` (half2) | Global | Native since CC 7.0 |
| `atomicCAS` | Global | 1 per clock per partition |
| `atomicAdd` | Shared | 1 per clock per bank (CC 7.x+) |

**Optimization: warp-aggregated atomics** reduce contention by having one thread per warp perform the atomic on behalf of all warp threads (see `atomicAggInc` pattern in Section 2).

---

## 2. Warp Execution Model

### 2.1 SIMT Architecture

Threads are organized into warps of 32 threads. All threads in a warp execute the same instruction at the same time (SIMT -- Single Instruction, Multiple Thread). The warp is the fundamental scheduling unit.

**Key invariants:**
- Warp size is 32 on all NVIDIA GPUs to date
- Threads within a warp have lane IDs 0-31
- `threadIdx.x / 32` gives the warp index within a block

### 2.2 Warp Divergence

When threads in a warp encounter a branch where different threads take different paths, the warp must execute both paths sequentially, masking off inactive threads.

```c
// BAD: 50% of threads diverge, both branches execute sequentially
if (threadIdx.x % 2 == 0) {
    path_A();   // Threads 0,2,4,... active; 1,3,5,... masked
} else {
    path_B();   // Threads 1,3,5,... active; 0,2,4,... masked
}

// BETTER: Divergence at warp boundaries -- no divergence within a warp
if (threadIdx.x / 32 % 2 == 0) {
    path_A();   // Entire warps take one path
} else {
    path_B();   // Other entire warps take the other
}
```

**Volta+ Independent Thread Scheduling:** Starting with Volta (CC 7.0), threads have independent program counters and call stacks. Diverged threads can reconverge at any point, not just at the branch join point. This enables new patterns but means code can no longer assume lock-step execution without explicit synchronization.

### 2.3 Warp Shuffle Operations

Warp shuffles allow direct register-to-register data exchange between threads in a warp without shared memory. They are faster than shared memory for intra-warp communication.

```c
// All shuffle variants require a mask and synchronize participating threads
T __shfl_sync(unsigned mask, T var, int srcLane, int width=32);
T __shfl_up_sync(unsigned mask, T var, unsigned delta, int width=32);
T __shfl_down_sync(unsigned mask, T var, unsigned delta, int width=32);
T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width=32);
```

**Mask parameter:** A 32-bit value where bit `i` indicates lane `i` participates. Use `0xFFFFFFFF` when all 32 lanes participate. The mask serves two purposes: (1) specifies which threads must reconverge before the shuffle executes, and (2) defines which threads are valid sources.

**Warp reduction with shuffles:**

```c
// Sum reduction across full warp -- result in lane 0
__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

// Broadcast from lane 0 to all lanes
float bcast = __shfl_sync(0xFFFFFFFF, val, 0);

// Butterfly (XOR) reduction -- all lanes get final result
__device__ float warpReduceSumAll(float val) {
    for (int mask = 16; mask > 0; mask /= 2)
        val += __shfl_xor_sync(0xFFFFFFFF, val, mask);
    return val;
}
```

### 2.4 Warp Vote Operations

```c
// Returns bitmask: bit i set if predicate is true for lane i
unsigned __ballot_sync(unsigned mask, int predicate);

// Returns non-zero if predicate is true for ALL participating lanes
int __all_sync(unsigned mask, int predicate);

// Returns non-zero if predicate is true for ANY participating lane
int __any_sync(unsigned mask, int predicate);

// Returns bitmask of currently active lanes (detective, not collective)
unsigned __activemask();
```

**Warp-aggregated atomic using match+ballot:**

```c
__device__ int atomicAggInc(int *ptr) {
    // Find all lanes targeting the same pointer
    int mask = __match_any_sync(__activemask(), (unsigned long long)ptr);
    // Elect leader (lowest set bit)
    int leader = __ffs(mask) - 1;
    int lane = threadIdx.x % 32;
    int res;
    if (lane == leader)
        res = atomicAdd(ptr, __popc(mask));  // One atomic for the whole group
    // Broadcast result and compute per-lane offset
    res = __shfl_sync(mask, res, leader);
    return res + __popc(mask & ((1 << lane) - 1));
}
```

### 2.5 Cooperative Groups

Cooperative Groups (CUDA 9+) provides a flexible API for defining and synchronizing thread groups at various granularities.

**Group types:**

```c
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Implicit groups
cg::thread_block  block = cg::this_thread_block();
cg::grid_group    grid  = cg::this_grid();            // Requires cooperative launch
cg::cluster_group cluster = cg::this_cluster();       // Hopper+

// Tiled partitions (compile-time size, must be power of 2, <= 32)
cg::thread_block_tile<16> tile16 = cg::tiled_partition<16>(block);
cg::thread_block_tile<32> warp   = cg::tiled_partition<32>(block);

// Runtime-sized partitions
cg::coalesced_group active = cg::coalesced_threads();  // Active threads only
```

**Tile operations (equivalent to warp primitives but scoped to tile):**

```c
auto tile = cg::tiled_partition<32>(cg::this_thread_block());
tile.sync();                          // Synchronize tile
float sum = tile.shfl_down(val, 1);   // Shuffle within tile
unsigned ballot = tile.ballot(pred);  // Vote within tile
bool all = tile.all(pred);            // All-reduce predicate
```

**Grid-wide synchronization (cooperative launch):**

```c
__global__ void kernel() {
    cg::grid_group grid = cg::this_grid();
    // ... phase 1 work ...
    grid.sync();  // Barrier across ALL thread blocks in the grid
    // ... phase 2 work ...
}

// Must launch with cooperative API:
void *args[] = { &arg1, &arg2 };
cudaLaunchCooperativeKernel((void*)kernel, grid_dim, block_dim, args);
```

---

## 3. Occupancy Optimization

### 3.1 What Is Occupancy

Occupancy = (active warps per SM) / (maximum warps per SM). Higher occupancy helps hide memory latency by having more warps available to schedule when one stalls. However, maximum occupancy does not always yield maximum performance.

**Maximum warps per SM by architecture:**

| Architecture | CC | Max Threads/SM | Max Warps/SM | Max Blocks/SM |
|---|---|---|---|---|
| Volta | 7.0 | 2048 | 64 | 32 |
| Turing | 7.5 | 1024 | 32 | 16 |
| Ampere | 8.0 | 2048 | 64 | 32 |
| Ampere (GA10x) | 8.6 | 1536 | 48 | 16 |
| Hopper | 9.0 | 2048 | 64 | 32 |

### 3.2 Resource Limiters

Three resources limit occupancy:

**1. Registers per thread:**
- Each SM has a register file (e.g., 65,536 32-bit registers on CC 7.0+)
- If kernel uses 32 registers/thread and block is 256 threads: 32 * 256 = 8,192 registers/block
- Max concurrent blocks = 65,536 / 8,192 = 8 blocks = 2048 threads = 100% occupancy (CC 8.0)
- If kernel uses 64 registers/thread: 64 * 256 = 16,384 regs/block, max 4 blocks = 1024 threads = 50%
- Use `--ptxas-options=-v` to see register count at compile time

**2. Shared memory per block:**
- Ampere: 164 KB max shared per SM, configurable carveout
- Hopper: 228 KB max shared per SM
- If kernel uses 48 KB shared and SM has 164 KB: max 3 blocks
- Sweet spot is typically 24-48 KB per block

**3. Thread block size:**
- Must be a multiple of warp size (32)
- Typical choices: 128, 256, 512
- Smaller blocks (128) allow more blocks per SM, better for memory-bound kernels
- Larger blocks (256-512) reduce per-block overhead, better for compute-bound

### 3.3 Launch Bounds

Compiler hints to control register allocation:

```c
__global__ void __launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)
myKernel(...) { ... }

// Example: 256 threads/block, at least 2 blocks/SM
__global__ void __launch_bounds__(256, 2) myKernel(...) {
    // Compiler will try to limit registers to allow 2 blocks of 256 threads
    // On CC 8.0: 65536 / (2 * 256) = 128 registers max per thread
}
```

If the compiler cannot achieve the requested `minBlocksPerMultiprocessor`, it still compiles but may issue a warning. The compiler may spill registers to local memory to meet the constraint.

### 3.4 Occupancy Calculator API

```c
int blockSize;
int minGridSize;

// Automatically find block size that maximizes occupancy
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, myKernel, 0, 0);

// Query occupancy for a specific configuration
int numBlocks;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, myKernel, blockSize, sharedMemBytes);

float occupancy = (float)(numBlocks * blockSize) / maxThreadsPerSM;
```

### 3.5 When Less Occupancy Is Better

- **Compute-bound kernels** with high ILP (instruction-level parallelism) may prefer fewer threads with more registers per thread.
- **Kernels with large working sets** in registers benefit from lower occupancy if it avoids spilling.
- Profile with Nsight Compute to find the occupancy sweet spot empirically.

---

## 4. CUDA Streams and Events

### 4.1 Stream Fundamentals

A stream is a sequence of operations (kernels, memcpy, memset) that execute in order. Operations in different streams may execute concurrently.

```c
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// These may overlap if GPU resources allow
kernelA<<<grid, block, 0, stream1>>>(args);
kernelB<<<grid, block, 0, stream2>>>(args);

cudaStreamSynchronize(stream1);
cudaStreamDestroy(stream1);
```

### 4.2 Default Stream Semantics

Two modes with critically different behavior:

**Legacy default stream (default before CUDA 7):**
- Implicitly synchronizes with ALL other streams in the same context
- A kernel on the default stream waits for all prior work on any stream to complete
- Shared across all host threads in the same context

**Per-thread default stream (recommended):**
- Each host thread gets its own default stream
- No implicit synchronization with other streams
- Enable with: `--default-stream per-thread` compile flag, or `#define CUDA_API_PER_THREAD_DEFAULT_STREAM` before any CUDA headers

**Non-blocking streams:**

```c
cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
// This stream never synchronizes with the legacy default stream
```

### 4.3 Stream Priorities

```c
int leastPriority, greatestPriority;
cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
// leastPriority is numerically largest (lowest priority)
// greatestPriority is numerically smallest (highest priority)

cudaStream_t highPriorityStream;
cudaStreamCreateWithPriority(&highPriorityStream, cudaStreamNonBlocking, greatestPriority);
```

Priorities affect kernel scheduling when GPU resources are contended. Higher-priority streams preempt at warp-granularity. Priorities are hints and do NOT affect memory transfer ordering.

### 4.4 Events

Events are lightweight markers inserted into streams for timing and synchronization.

```c
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, stream);
kernel<<<grid, block, 0, stream>>>(args);
cudaEventRecord(stop, stream);

cudaEventSynchronize(stop);
float ms;
cudaEventElapsedTime(&ms, start, stop);

// Cross-stream synchronization: stream2 waits for event in stream1
cudaEventRecord(event, stream1);
cudaStreamWaitEvent(stream2, event, 0);  // stream2 waits until event completes
```

### 4.5 Multi-Stream Patterns

**Pattern: Overlapping compute and H2D/D2H transfers:**

```c
for (int i = 0; i < NUM_CHUNKS; i++) {
    int stream_idx = i % NUM_STREAMS;
    cudaMemcpyAsync(d_in + i*chunk, h_in + i*chunk, chunk_bytes,
                    cudaMemcpyHostToDevice, streams[stream_idx]);
    kernel<<<grid, block, 0, streams[stream_idx]>>>(d_in + i*chunk, d_out + i*chunk);
    cudaMemcpyAsync(h_out + i*chunk, d_out + i*chunk, chunk_bytes,
                    cudaMemcpyDeviceToHost, streams[stream_idx]);
}
```

**Pattern: Breadth-first launch for maximum concurrency:**

```c
// WRONG: depth-first prevents overlap (all ops in stream[0] before stream[1])
for (int s = 0; s < N; s++) {
    cudaMemcpyAsync(..., streams[s]);
    kernel<<<..., streams[s]>>>();
    cudaMemcpyAsync(..., streams[s]);
}

// RIGHT: breadth-first enables overlap
for (int s = 0; s < N; s++) cudaMemcpyAsync(H2D, streams[s]);
for (int s = 0; s < N; s++) kernel<<<..., streams[s]>>>();
for (int s = 0; s < N; s++) cudaMemcpyAsync(D2H, streams[s]);
```

Note: Modern GPUs (CC 7.0+) with multiple copy engines are less sensitive to launch order, but breadth-first remains the safer default.

### 4.6 Callback and Host Functions

```c
cudaLaunchHostFunc(stream, myHostCallback, userData);
// Callback executes on CPU when all prior work in stream completes
// WARNING: callback must not call CUDA API functions
```

---

## 5. CUDA Graphs

### 5.1 Motivation

For workloads with many small kernels (e.g., LLM decode), per-kernel launch overhead (5-10 us) dominates. CUDA Graphs capture an entire workflow and replay it with a single launch, reducing CPU overhead to near zero.

### 5.2 Graph Creation: Stream Capture

```c
cudaGraph_t graph;
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

// All operations are captured, not executed
kernelA<<<grid, block, 0, stream>>>(args);
kernelB<<<grid, block, 0, stream>>>(args);
cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream);

cudaStreamEndCapture(stream, &graph);
```

**Capture modes:**
- `cudaStreamCaptureModeGlobal`: Any attempt to use a non-capturing stream from a captured stream's work is an error
- `cudaStreamCaptureModeThreadLocal`: Only checks within the capturing thread
- `cudaStreamCaptureModeRelaxed`: No cross-stream checks

**Cross-stream capture (fork-join):**

```c
cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal);
kernelA<<<..., stream1>>>(...);

cudaEventRecord(fork_event, stream1);
cudaStreamWaitEvent(stream2, fork_event);    // Fork

kernelB<<<..., stream1>>>(...);               // Parallel branch 1
kernelC<<<..., stream2>>>(...);               // Parallel branch 2

cudaEventRecord(join_event, stream2);
cudaStreamWaitEvent(stream1, join_event);    // Join

kernelD<<<..., stream1>>>(...);
cudaStreamEndCapture(stream1, &graph);       // Must end in origin stream
```

### 5.3 Graph Creation: Explicit API

```c
cudaGraph_t graph;
cudaGraphCreate(&graph, 0);

cudaKernelNodeParams params = {};
params.func = (void*)myKernel;
params.gridDim = grid;
params.blockDim = block;
params.kernelParams = args;

cudaGraphNode_t nodeA, nodeB;
cudaGraphAddKernelNode(&nodeA, graph, NULL, 0, &params);  // No dependencies
cudaGraphAddKernelNode(&nodeB, graph, &nodeA, 1, &params); // Depends on nodeA
```

### 5.4 Instantiation and Launch

```c
cudaGraphExec_t graphExec;
cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
// Instantiation validates topology and pre-computes scheduling

// Launch (can be called repeatedly)
cudaGraphLaunch(graphExec, stream);
cudaStreamSynchronize(stream);

// A graphExec cannot run concurrently with itself
// Subsequent launches serialize automatically
```

### 5.5 Graph Update

**Whole graph update (topology must match):**

```c
cudaGraphExecUpdateResultInfo resultInfo;
cudaGraphExecUpdate(graphExec, newGraph, &resultInfo);
if (resultInfo.result != cudaGraphExecUpdateSuccess) {
    cudaGraphExecDestroy(graphExec);
    cudaGraphInstantiate(&graphExec, newGraph, NULL, NULL, 0);
}
```

**Individual node update (more efficient for few changes):**

```c
cudaGraphExecKernelNodeSetParams(graphExec, kernelNode, &newParams);
cudaGraphExecMemcpyNodeSetParams(graphExec, memcpyNode, &newCopyParams);
```

**Node enable/disable:**

```c
cudaGraphNodeSetEnabled(graphExec, node, 0);  // Skip this node
cudaGraphNodeSetEnabled(graphExec, node, 1);  // Re-enable
```

### 5.6 Conditional Nodes (CUDA 12.4+)

Enable data-dependent control flow within graphs without host round-trips.

**IF conditional:**

```c
cudaGraphConditionalHandle handle;
cudaGraphConditionalHandleCreate(&handle, graph, 0, 0);

cudaGraphNodeParams cParams = {};
cParams.type = cudaGraphNodeTypeConditional;
cParams.conditional.handle = handle;
cParams.conditional.type = cudaGraphCondTypeIf;
cParams.conditional.size = 1;  // 1 body = if, 2 bodies = if/else

cudaGraphNode_t condNode;
cudaGraphAddNode(&condNode, graph, deps, numDeps, &cParams);

// Populate the body graph
cudaGraph_t bodyGraph;
cudaGraphConditionalNodeGetBody(condNode, &bodyGraph, 0);
// Add nodes to bodyGraph...

// Set condition from a kernel:
__global__ void setCondition(cudaGraphConditionalHandle h) {
    cudaGraphSetConditional(h, should_execute ? 1 : 0);
}
```

**WHILE conditional:** Body executes repeatedly while condition is non-zero.

**SWITCH conditional:** Selects which body graph to execute based on condition value.

### 5.7 Memory Nodes

Graphs can manage GPU allocations with GPU-ordered lifetimes:

```c
cudaGraphNodeParams allocParams = {};
allocParams.type = cudaGraphNodeTypeMemAlloc;
allocParams.alloc.poolProps.allocType = cudaMemAllocationTypePinned;
allocParams.alloc.bytesize = size;
cudaGraphAddNode(&allocNode, graph, NULL, 0, &allocParams);
void *dptr = allocParams.alloc.dptr;  // Fixed virtual address
```

### 5.8 When to Use Graphs

**Good candidates:**
- LLM decode phase (many small kernels, fixed topology)
- Inference pipelines with fixed structure
- Iterative solvers with repeated computation pattern
- Any workload where CPU launch overhead > kernel execution time

**Poor candidates:**
- Highly dynamic workloads where topology changes every iteration
- Single large kernel launches (overhead already amortized)
- Workloads requiring host-side decisions between steps (unless using conditional nodes)

---

## 6. PTX and SASS

### 6.1 Compilation Pipeline

```
CUDA C++ --> (nvcc frontend) --> PTX --> (ptxas) --> SASS/cubin
                                    |
                                    +--> Embedded in executable for JIT
```

### 6.2 PTX (Parallel Thread Execution)

PTX is a **virtual** ISA -- an intermediate representation that is architecture-independent. It provides forward compatibility: PTX compiled for `compute_70` can run on any GPU with CC >= 7.0 via JIT compilation.

**Key PTX characteristics:**
- Infinite virtual register set (physical allocation deferred to ptxas)
- Explicit memory space annotations: `.global`, `.shared`, `.local`, `.const`, `.param`
- Typed registers: `.pred` (predicate), `.b16/.b32/.b64` (bits), `.f32/.f64` (float), `.s32/.u32` (signed/unsigned int)
- Predicated execution (avoids branches for simple conditionals)

**Key PTX instructions:**

```
// Arithmetic
add.f32     %f3, %f1, %f2;          // f3 = f1 + f2
mul.f32     %f3, %f1, %f2;          // f3 = f1 * f2
fma.rn.f32  %f3, %f1, %f2, %f4;    // f3 = f1*f2 + f4 (fused multiply-add)
mad.lo.s32  %r3, %r1, %r2, %r4;    // r3 = r1*r2 + r4 (integer)

// Memory
ld.global.f32   %f1, [%rd1];        // Load from global memory
st.global.f32   [%rd1], %f1;        // Store to global memory
ld.shared.f32   %f1, [%rd1];        // Load from shared memory
ld.param.u64    %rd1, [param0];     // Load kernel parameter
atom.global.add.f32 %f1, [%rd1], %f2;  // Atomic add

// Control flow
setp.lt.f32 %p1, %f1, %f2;         // Set predicate: p1 = (f1 < f2)
@%p1 bra   LABEL;                   // Conditional branch
bar.sync    0;                       // __syncthreads()

// Warp-level
shfl.sync.down.b32 %r1, %r2, %r3, 0x1f, 0xffffffff;
vote.sync.ballot.b32 %r1, %p1, 0xffffffff;

// Async copy (CC 8.0+)
cp.async.ca.shared.global [%rd_shared], [%rd_global], 16;
cp.async.commit_group;
cp.async.wait_group 0;

// TMA (CC 9.0+)
cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
    [%rd_smem], [%rd_tensor_map, {%r_x, %r_y}], [%rd_mbar];
```

### 6.3 SASS (Streaming ASSembler)

SASS is the **actual machine code** executed on the GPU. It is architecture-specific and NOT forward-compatible.

**Key differences from PTX:**
- Physical register allocation (limited register file)
- Architecture-specific instruction encoding and scheduling
- Dual-issue and instruction pairing rules
- Bank-specific load/store instructions
- Scoreboard and barrier management exposed

**Viewing SASS:**
```bash
cuobjdump -sass executable.cubin
nvdisasm executable.cubin
# Or in Nsight Compute: Source page shows SASS correlation
```

### 6.4 Cubin Format

A cubin is a binary containing SASS for a specific compute capability. Fat binaries can embed multiple cubins plus PTX:

```bash
# Compile for multiple targets
nvcc -gencode arch=compute_80,code=sm_80 \
     -gencode arch=compute_90,code=sm_90 \
     -gencode arch=compute_90,code=compute_90 \  # Embeds PTX for JIT
     kernel.cu -o kernel
```

### 6.5 Compatibility Rules

- **PTX forward compatibility:** PTX compiled for `compute_XX` runs on any SM >= XX via JIT
- **SASS backward compatibility (within major version):** `sm_86` cubin runs on CC 8.6 and 8.9, but NOT on 8.0
- **No SASS forward compatibility across major versions:** `sm_80` cubin does NOT run on CC 9.0
- **Best practice:** Always embed PTX for the lowest CC you support, plus SASS for specific targets

### 6.6 JIT Compilation

When the driver loads a kernel and finds no compatible cubin, it JIT-compiles embedded PTX. First compilation is cached in `~/.nv/ComputeCache/`. JIT overhead is typically 100ms-seconds per kernel, amortized after first run.

---

## 7. Async Operations

### 7.1 cp.async (CC 8.0+, Ampere)

Asynchronous copy from global to shared memory that bypasses registers. The data path is: Global Memory -> L2 -> L1 -> Shared Memory, without consuming register file bandwidth.

```c
// Low-level intrinsic
__pipeline_memcpy_async(&shared[tid], &global[tid], sizeof(float));
__pipeline_commit();
__pipeline_wait_prior(0);  // Wait for all committed groups

// C++ API (preferred)
#include <cuda/pipeline>
cuda::memcpy_async(shared + tid, global + tid,
                   cuda::aligned_size_t<4>(sizeof(float)), barrier);
barrier.arrive_and_wait();
```

**Supported transfer sizes:** 4, 8, or 16 bytes per thread. Best performance with 16-byte (128-bit) copies.

### 7.2 Pipeline Patterns

Multi-stage software pipelining overlaps compute on one data tile with async loads of the next tile.

```c
// Double-buffering with 2 pipeline stages
__shared__ float buf[2][TILE_SIZE];
cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

// Prologue: fill stage 0
pipe.producer_acquire();
cuda::memcpy_async(&buf[0][tid], &input[0 * TILE + tid], sizeof(float), pipe);
pipe.producer_commit();

for (int i = 0; i < num_tiles; i++) {
    int cur = i % 2;
    int nxt = (i + 1) % 2;

    // Start loading next tile
    if (i + 1 < num_tiles) {
        pipe.producer_acquire();
        cuda::memcpy_async(&buf[nxt][tid], &input[(i+1)*TILE + tid], sizeof(float), pipe);
        pipe.producer_commit();
    }

    // Wait for current tile
    cuda::pipeline_consumer_wait_prior<1>(pipe);  // Wait until at most 1 group pending

    // Compute on current tile
    compute(output + i * TILE, buf[cur]);

    pipe.consumer_release();
}
```

### 7.3 Async Barriers (CC 8.0+)

```c
#include <cuda/barrier>

__shared__ cuda::barrier<cuda::thread_scope_block> bar;

if (threadIdx.x == 0)
    init(&bar, blockDim.x);
__syncthreads();

// Barrier tracks both thread arrivals AND pending byte transactions
cuda::memcpy_async(shared_dest, global_src, num_bytes, bar);

// Phase flips only when ALL threads arrive AND all bytes are transferred
auto token = bar.arrive();
bar.wait(std::move(token));
```

### 7.4 TMA -- Tensor Memory Accelerator (CC 9.0+, Hopper)

TMA is a dedicated hardware unit that handles bulk tensor data movement between global and shared memory. A single thread issues TMA operations; the entire block can continue computing.

**Advantages over cp.async:**
- Single-threaded programming model (one elected thread issues the copy)
- Handles multi-dimensional tensor addressing in hardware
- Supports 1D through 5D tensors
- Hardware swizzling to avoid shared memory bank conflicts
- Supports reductions (add, min, max, bitwise AND/OR) during transfer

**1D bulk copy:**

```c
#include <cuda/barrier>
using barrier = cuda::barrier<cuda::thread_scope_block>;

__shared__ alignas(128) float smem_data[SIZE];
__shared__ barrier bar;

if (threadIdx.x == 0) {
    init(&bar, blockDim.x);
    // Single thread issues TMA -- requires 16-byte alignment
    cuda::memcpy_async(smem_data, global_data,
                       cuda::aligned_size_t<16>(sizeof(smem_data)), bar);
}
__syncthreads();
auto token = bar.arrive();
bar.wait(std::move(token));
// smem_data is now ready
```

**Multi-dimensional tensor copy (requires tensor map):**

```c
// Host: create tensor map descriptor
CUtensorMap tensor_map;
cuTensorMapEncodeTiled(
    &tensor_map,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
    2,                    // rank (2D tensor)
    global_ptr,           // base address
    globalDims,           // {width, height}
    globalStrides,        // {width * sizeof(float)}  (must be multiple of 16)
    boxDims,              // tile dims to copy {TILE_W, TILE_H}
    elementStrides,       // {1, 1}
    CU_TENSOR_MAP_INTERLEAVE_NONE,
    CU_TENSOR_MAP_SWIZZLE_128B,   // Avoid bank conflicts
    CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
);

// Device kernel
__global__ void kernel(const __grid_constant__ CUtensorMap tensor_map) {
    __shared__ alignas(128) float tile[TILE_H][TILE_W];
    __shared__ barrier bar;

    if (threadIdx.x == 0) {
        init(&bar, blockDim.x);
        int32_t coords[2] = {blockIdx.x * TILE_W, blockIdx.y * TILE_H};
        // Issue 2D TMA copy
        cuda::ptx::cp_async_bulk_tensor(
            cuda::ptx::space_shared, cuda::ptx::space_global,
            &tile, &tensor_map, coords, bar);
    }
    __syncthreads();
    bar.arrive_and_wait();
}
```

**Alignment requirements:**
- Global memory base address: 16-byte aligned
- Shared memory destination: 128-byte aligned
- Global strides: multiples of 16 bytes

### 7.5 Warp-Specialized Pipelines (Hopper)

TMA enables warp specialization where dedicated "producer" warps handle data movement while "consumer" warps perform computation:

```c
__global__ void warp_specialized_kernel(...) {
    int warp_id = threadIdx.x / 32;

    if (warp_id == 0) {
        // Producer warp: issue TMA loads
        for (int stage = 0; stage < num_stages; stage++) {
            producer_acquire(pipe, stage);
            if (threadIdx.x == 0) {
                tma_load(smem_buf[stage], global_data + stage * tile_size, barrier[stage]);
            }
            producer_commit(pipe, stage);
        }
    } else {
        // Consumer warps: compute on loaded tiles
        for (int stage = 0; stage < num_stages; stage++) {
            consumer_wait(pipe, stage);
            compute(smem_buf[stage], output);
            consumer_release(pipe, stage);
        }
    }
}
```

---

## 8. Thread Block Clusters (CC 9.0+, Hopper)

### 8.1 Overview

Thread Block Clusters are a new hierarchy level: **Thread < Warp < Block < Cluster < Grid**. A cluster is a group of thread blocks guaranteed to be concurrently scheduled on SMs within the same GPC (GPU Processing Cluster).

### 8.2 Cluster Launch

**Compile-time specification:**

```c
__global__ void __cluster_dims__(2, 1, 1) my_kernel(...) {
    // This kernel always launches with clusters of 2 blocks
}
```

**Runtime specification:**

```c
cudaLaunchConfig_t config = {};
config.gridDim = total_blocks;
config.blockDim = threads_per_block;

cudaLaunchAttribute attr;
attr.id = cudaLaunchAttributeClusterDimension;
attr.val.clusterDim = {4, 1, 1};  // 4 blocks per cluster
config.attrs = &attr;
config.numAttrs = 1;

cudaLaunchKernelEx(&config, my_kernel, arg1, arg2);
```

**Size limits:**
- Maximum portable cluster size: **8 thread blocks**
- H100 non-portable max: **16** (requires `cudaFuncAttributeNonPortableClusterSizeAllowed`)
- Cluster dimensions are in block units, not thread units

### 8.3 Distributed Shared Memory (DSMEM)

Within a cluster, any thread can access the shared memory of any block in the cluster. This enables inter-SM communication without going through global memory.

```c
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void __cluster_dims__(4, 1, 1) dsmem_kernel(...) {
    __shared__ float my_smem[256];

    auto cluster = cg::this_cluster();
    int peer_rank = (cluster.block_rank() + 1) % cluster.num_blocks();

    // Get pointer to peer block's shared memory
    float *peer_smem = cluster.map_shared_rank(my_smem, peer_rank);

    cluster.sync();  // Ensure all blocks have written their shared memory

    // Direct load from another SM's shared memory
    float val = peer_smem[threadIdx.x];
}
```

**Performance:**
- DSMEM is ~7x faster than communicating through global memory
- Uses a dedicated SM-to-SM network within the GPC
- Can operate simultaneously with L2 cache accesses (combined bandwidth)
- Access should be coalesced and aligned to 32-byte segments for best performance

### 8.4 Cross-SM Synchronization

```c
auto cluster = cg::this_cluster();
cluster.sync();  // Barrier across all blocks in the cluster
```

Async transaction barriers also work across blocks within a cluster, enabling producer-consumer patterns where one block loads data and another block consumes it.

### 8.5 STAS: Register to DSMEM (CC 9.0+)

Store from registers directly into another SM's shared memory:

```c
#include <cuda/ptx>

float *remote_smem = cluster.map_shared_rank(my_smem, target_rank);
cuda::ptx::st_async(remote_smem + offset, my_register_value, remote_barrier);
```

Supports 4, 8, or 16-byte transfers. Useful for scatter patterns across cluster blocks.

---

## 9. Dynamic Parallelism

### 9.1 Overview

Dynamic parallelism (CDP -- CUDA Dynamic Parallelism) allows kernels to launch child kernels from device code. Available since CC 3.5; CDP2 is default since CUDA 12.0.

```c
__global__ void parentKernel(float *data, int n) {
    // Standard kernel launch syntax on device
    if (threadIdx.x == 0 && n > THRESHOLD) {
        childKernel<<<gridDim_child, blockDim_child>>>(data, n/2);
    }
}
```

### 9.2 Memory Model

| Memory | Parent -> Child | Child -> Parent |
|---|---|---|
| Global | Visible (weak consistency) | Visible after sync |
| Shared | NOT accessible | NOT accessible |
| Local | NOT accessible | NOT accessible |
| Constant | Visible | N/A (set at launch) |
| Texture | Visible (coherent at grid boundaries) | N/A |

### 9.3 Synchronization (CDP2)

CDP2 (CUDA 12.0+) removes `cudaDeviceSynchronize()` from device code. Instead, use tail launches:

```c
__global__ void parentKernel() {
    // Do work...

    // Launch child that starts after parent grid completes
    childKernel<<<grid, block, 0, cudaStreamTailLaunch>>>();

    // Or fire-and-forget (concurrent with parent):
    childKernel<<<grid, block, 0, cudaStreamFireAndForget>>>();
}
```

- `cudaStreamTailLaunch`: Child starts after the ENTIRE parent grid completes
- `cudaStreamFireAndForget`: Child is independent, may run concurrently

### 9.4 When to Use Dynamic Parallelism

**Good use cases:**
- Adaptive mesh refinement (subdivide only where needed)
- Recursive algorithms (quicksort, tree traversal) with data-dependent branching
- Hierarchical data structures (octrees, BVH)
- Algorithms where work discovery happens on GPU

**When to avoid:**
- Small child grids (launch overhead ~10-50 us dominates)
- Predictable work that can be structured as flat kernels
- Performance-critical inner loops (overhead is not negligible)

### 9.5 Overhead Considerations

- Device-side kernel launch overhead is ~10-50 us (much higher than host-side ~5 us with graphs)
- The device runtime reserves memory for pending launches (configurable via `cudaLimitDevRuntimePendingLaunchCount`)
- System software overhead applies to ALL kernels when device runtime is linked, even those not using CDP
- Compilation requires `-rdc=true` and linking with `-lcudadevrt`
- Max nesting depth: typically 24 levels (configurable via `cudaLimitDevRuntimeSyncDepth`)

---

## 10. Profiling with Nsight Tools

### 10.1 Nsight Compute (Kernel-Level Profiling)

Nsight Compute (`ncu`) provides detailed per-kernel performance analysis.

**Basic usage:**

```bash
# Profile all kernels with full metrics
ncu --set full -o profile_output ./my_application

# Profile specific kernel
ncu --kernel-name myKernel --launch-count 5 ./my_application

# Collect specific sections
ncu --section SpeedOfLight --section MemoryWorkloadAnalysis ./my_application
```

### 10.2 Key Metrics That Matter

**Speed of Light (SOL%) -- The first thing to check:**

| Metric | What It Tells You |
|---|---|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | Compute SOL% -- how close to peak compute |
| `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` | Memory SOL% -- how close to peak memory bandwidth |
| `l1tex__throughput.avg.pct_of_peak_sustained_elapsed` | L1/Tex cache throughput |
| `lts__throughput.avg.pct_of_peak_sustained_elapsed` | L2 cache throughput |

**Interpretation framework:**
- **Compute SOL > 60%, Memory SOL < 60%**: Compute-bound. Optimize ALU usage, consider lower precision.
- **Memory SOL > 60%, Compute SOL < 60%**: Memory-bound. Improve data reuse, reduce memory traffic.
- **Both < 60%**: Latency-bound. Increase occupancy, reduce stalls, check warp stall reasons.
- **Both > 80%**: Kernel is well-optimized, hitting hardware limits.

**Memory workload metrics:**

| Metric | Meaning |
|---|---|
| `dram__bytes_read.sum` / `dram__bytes_write.sum` | Total HBM bytes transferred |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | L1 sectors for global loads |
| `l1tex__data_bank_conflicts_pipe_lsu.sum` | L1/shared memory bank conflicts |
| `lts__t_sectors_srcunit_tex_op_read_hit_rate.pct` | L2 hit rate |
| `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct` | Global load efficiency |

**Compute metrics:**

| Metric | Meaning |
|---|---|
| `sm__inst_executed.avg.per_cycle_active` | IPC (instructions per cycle) |
| `sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active` | FMA pipe utilization |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | Tensor Core utilization |

### 10.3 Warp Stall Reasons

When a warp cannot issue an instruction, it is stalled. The key stall categories:

| Stall Reason | Description | Likely Fix |
|---|---|---|
| `stall_barrier` | Waiting at `__syncthreads()` or barrier | Reduce sync frequency, balance work |
| `stall_long_scoreboard` | Waiting for global/local memory load | Prefetch, increase occupancy, better coalescing |
| `stall_short_scoreboard` | Waiting for shared memory or L1 | Reduce bank conflicts, pipeline accesses |
| `stall_math_pipe_throttle` | Compute pipeline full | Good! Means compute-bound. Consider lower precision |
| `stall_membar` | Waiting for memory fence | Reduce fence scope if possible |
| `stall_not_selected` | Ready but not scheduled (others chosen) | Often benign, indicates sufficient occupancy |
| `stall_wait` | Waiting for fixed-latency dependency | Common with register dependencies |
| `stall_sleeping` | Thread explicitly sleeping | Dynamic parallelism or cooperative group sync |
| `stall_imc_miss` | Instruction cache miss | Reduce code size, improve locality |
| `stall_tex_throttle` | Texture/L1 pipe full | Reduce texture requests |
| `stall_lg_throttle` | Local/global pipe throttled | Reduce memory requests per cycle |

### 10.4 Occupancy Analysis

Nsight Compute reports:
- **Theoretical occupancy**: Based on register/shared memory/block limits
- **Achieved occupancy**: Actual active warps during execution
- **Limiting factor**: Which resource (registers, shared memory, block count) caps occupancy

### 10.5 Roofline Analysis

The roofline chart plots arithmetic intensity (FLOPs/byte) vs. performance (FLOP/s). A kernel below the roofline ceiling has room for optimization. The ridge point (where compute ceiling meets memory ceiling) tells you the minimum arithmetic intensity needed to be compute-bound.

### 10.6 Nsight Systems (Timeline Profiling)

Nsight Systems (`nsys`) provides system-level timeline visualization.

```bash
# Basic profiling
nsys profile -o timeline_output ./my_application

# With CUDA API tracing
nsys profile --trace=cuda,nvtx,osrt -o timeline_output ./my_application

# GPU metrics sampling
nsys profile --gpu-metrics-device=all ./my_application
```

**What to look for in Nsight Systems:**
- **Gaps between kernels**: CPU launch overhead, unnecessary synchronization
- **Serial kernel execution**: Missing concurrency opportunities
- **H2D/D2H overlap**: Whether transfers overlap with compute
- **NVTX markers**: Custom annotations for logical regions
- **Stream concurrency**: Whether multiple streams actually execute concurrently
- **CPU thread activity**: Whether host code is the bottleneck

**NVTX annotations for targeted profiling:**

```c
#include <nvtx3/nvToolsExt.h>

nvtxRangePush("Forward Pass");
forward_kernel<<<grid, block, 0, stream>>>(args);
nvtxRangePop();

nvtxRangePush("Backward Pass");
backward_kernel<<<grid, block, 0, stream>>>(args);
nvtxRangePop();
```

### 10.7 Profiling Workflow

1. **Start with Nsight Systems** to get the big picture: timeline, kernel durations, gaps, concurrency
2. **Identify hotspot kernels** consuming the most GPU time
3. **Profile hotspots with Nsight Compute** using `--set full` for comprehensive metrics
4. **Check SOL%** to classify as compute-bound, memory-bound, or latency-bound
5. **For memory-bound**: Check coalescing efficiency, cache hit rates, bank conflicts
6. **For compute-bound**: Check pipe utilization, consider lower precision or algorithmic changes
7. **For latency-bound**: Check warp stall reasons, occupancy, and consider increasing parallelism
8. **Iterate**: Make changes, re-profile, compare

---

## Quick Reference: Architecture Feature Matrix

| Feature | Volta (7.0) | Turing (7.5) | Ampere (8.0) | Ada (8.9) | Hopper (9.0) | Blackwell (10.0) |
|---|---|---|---|---|---|---|
| Independent thread scheduling | Yes | Yes | Yes | Yes | Yes | Yes |
| cp.async (LDGSTS) | No | No | Yes | Yes | Yes | Yes |
| Async barriers | No | No | Yes | Yes | Yes | Yes |
| TMA | No | No | No | No | Yes | Yes |
| Thread block clusters | No | No | No | No | Yes | Yes |
| Distributed shared memory | No | No | No | No | Yes | Yes |
| Conditional graph nodes | No | No | No | No | Yes | Yes |
| Max shared memory/SM | 96 KB | 64 KB | 164 KB | 100 KB | 228 KB | 228 KB |
| Max registers/thread | 255 | 255 | 255 | 255 | 255 | 255 |
| Warp size | 32 | 32 | 32 | 32 | 32 | 32 |
| FP8 Tensor Cores | No | No | No | Yes | Yes | Yes |
| DPX instructions | No | No | No | No | Yes | Yes |

---

## Sources

- [Using Shared Memory in CUDA C/C++ -- NVIDIA Blog](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)
- [Using CUDA Warp-Level Primitives -- NVIDIA Blog](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)
- [CUDA Programming Guide: Asynchronous Execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)
- [CUDA Programming Guide: CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
- [CUDA Programming Guide: Async Data Copies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)
- [CUDA Programming Guide: Dynamic Parallelism](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html)
- [CUDA Programming Guide: Cooperative Groups](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html)
- [Understanding PTX -- NVIDIA Blog](https://developer.nvidia.com/blog/understanding-ptx-the-assembly-language-of-cuda-gpu-computing/)
- [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [CUDA Occupancy API -- NVIDIA Blog](https://developer.nvidia.com/blog/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/)
- [GPU Pro Tip: CUDA 7 Streams Simplify Concurrency](https://developer.nvidia.com/blog/gpu-pro-tip-cuda-7-streams-simplify-concurrency/)
- [Dynamic Control Flow in CUDA Graphs with Conditional Nodes](https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes/)
- [Cooperative Groups: Flexible CUDA Thread Programming](https://developer.nvidia.com/blog/cooperative-groups/)
- [CUTLASS Tutorial: Mastering the NVIDIA Tensor Memory Accelerator (TMA)](https://research.colfax-intl.com/tutorial-hopper-tma/)
- [Deep Dive on the Hopper TMA Unit for FP8 GEMMs -- PyTorch](https://pytorch.org/blog/hopper-tma-unit/)
- [How to Improve CUDA Kernel Performance with Shared Memory Register Spilling](https://developer.nvidia.com/blog/how-to-improve-cuda-kernel-performance-with-shared-memory-register-spilling)
- [A Gentle Introduction to CUDA PTX](https://philipfabianek.com/posts/cuda-ptx-introduction/)
- [Adaptive Parallel Computation with CUDA Dynamic Parallelism](https://developer.nvidia.com/blog/introduction-cuda-dynamic-parallelism/)
