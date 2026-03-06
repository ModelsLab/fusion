---
id: nvidia-cutlass-cute-deep-dive
kind: source
title: "NVIDIA CUTLASS and CuTe: Comprehensive Deep Dive"
type: official-doc
category: kernel-lib
summary: "Exhaustive technical reference covering CUTLASS 2.x-4.x architecture, CuTe layout algebra, MMA/copy atoms, GEMM kernel design, epilogue fusion (EVT), CuTe DSL (Python), the CUTLASS profiler, fused attention, Blackwell support, and performance tuning."
support_level: ""
reliability: official
review_status: reviewed
url: https://github.com/NVIDIA/cutlass
tags:
  - nvidia
  - cutlass
  - cute
  - cute-dsl
  - gemm
  - tensor-core
  - epilogue-fusion
  - evt
  - flash-attention
  - blackwell
  - hopper
  - ampere
  - performance-tuning
  - autotuning
  - kernel-lib
aliases: []
family: ""
market: ""
compute_capability: ""
memory_gb: 0
memory_bandwidth_gbps: 0
preferred_precisions: []
experimental_precisions: []
strengths: []
constraints: []
source_ids: []
workloads: []
operators: []
gpu_families: []
gpu_ids: []
precision: []
bottlenecks: []
goals: []
priority: 0
preconditions: []
actions: []
metrics: []
tradeoffs: []
preferred_backends: []
required_tools: []
steps: []
verification: []
benchmark_rubric: []
failure_recovery: []
artifacts_to_save: []
runtime_adapters: []
reference_source_ids:
  - nvidia-cutlass-overview
  - nvidia-cute-dsl
  - nvidia-blackwell-cutlass
backends: []
path: ""
runtimes: []
use_cases: []
notes: []
reference_paths: []
---

# NVIDIA CUTLASS and CuTe: Comprehensive Deep Dive

---

## 1. CUTLASS Overview

### What Is CUTLASS

CUTLASS (CUDA Templates for Linear Algebra Subroutines) is NVIDIA's open-source C++ template library (and, since CUTLASS 4.0, Python DSL) for high-performance dense linear algebra on NVIDIA GPUs. It provides reusable building blocks for GEMM (General Matrix Multiply), convolution, and related operations, targeting every generation of NVIDIA Tensor Cores from Volta (SM70) through Blackwell (SM100/SM120).

CUTLASS is **not** a drop-in replacement for cuBLAS. Instead, it provides composable template primitives that let developers:

- Build custom GEMM/convolution kernels with fused pre- and post-processing
- Target specific tile sizes, data types, and scheduling strategies
- Achieve performance comparable to or exceeding cuBLAS for specialized workloads
- Compile only the kernels they need, avoiding the large binary overhead of cuBLAS

### CUTLASS vs cuBLAS

| Aspect | cuBLAS | CUTLASS |
|--------|--------|---------|
| Interface | Runtime library (shared object) | Header-only C++ templates + Python DSL |
| Customization | Fixed set of GEMM variants | Fully customizable tile sizes, epilogues, data types |
| Epilogue Fusion | Limited (alpha/beta scaling) | Arbitrary fusion via Epilogue Visitor Trees |
| Binary Size | Large precompiled library | Compile only what you need |
| Performance | Highly tuned by NVIDIA | Comparable; often identical kernels |
| Ease of Use | Simple API call | Requires understanding of GEMM hierarchy |
| Mixed Precision | Supported but fixed | Fully flexible, including block-scaled FP4/FP6/FP8 |

### Version History

| Version | Year | Key Features |
|---------|------|-------------|
| CUTLASS 1.x | 2017 | Initial release; SIMT GEMM on Volta |
| CUTLASS 2.x | 2019-2022 | Tensor Core support (Volta/Turing/Ampere), iterator-based design, epilogue functors, convolution support |
| CUTLASS 3.0 | 2023 | CuTe core library, Hopper TMA/WGMMA support, warp-specialized kernels, Epilogue Visitor Trees |
| CUTLASS 3.5+ | 2024 | Blackwell SM100 support, FP4/FP6, Tensor Memory, 2-SM cooperative MMA |
| CUTLASS 4.0 | 2025 | CuTe DSL (Python), AOT compilation, DLPack integration, nvMatmulHeuristics |
| CUTLASS 4.x | 2025-2026 | CuTe DSL graduating from beta, expanded Blackwell support, SM120 |

### Design Philosophy

CUTLASS 3.x fundamentally changed the design philosophy from CUTLASS 2.x:

**CUTLASS 2.x** decomposed GEMM along a hierarchy mirroring GPU hardware (device -> threadblock -> warp -> thread). This created tight coupling between abstractions and specific hardware features. Hopper's warp-group-wide WGMMA instructions did not fit naturally into any warp or thread layer concept.

**CUTLASS 3.x** detaches interface layers from hardware, centering them around the natural structure of GEMM algorithms. The key change: replacing all iterator concepts for all memory domains with `cute::Tensor`, using CuTe's formalized layout algebra at every layer. This makes code robust to GPU architecture evolution and provides a consistent interface regardless of hardware details.

### Supported Architectures

- **Volta** (SM70): 1st gen Tensor Cores, FP16 MMA
- **Turing** (SM75): 2nd gen Tensor Cores, INT8/INT4 MMA
- **Ampere** (SM80): 3rd gen Tensor Cores, TF32, BF16, sparse MMA, cp.async
- **Ada Lovelace** (SM89): Enhanced Ampere, FP8
- **Hopper** (SM90): 4th gen Tensor Cores, TMA, WGMMA, warp specialization, thread block clusters
- **Blackwell** (SM100/SM120): 5th gen Tensor Cores, UMMA, Tensor Memory, FP4/FP6, 2-SM cooperative execution

---

## 2. CUTLASS 3.x Architecture

### The Five-Layer GEMM Hierarchy

CUTLASS 3.x organizes GEMM computation through five hierarchical layers:

```
Layer 5: Device       cutlass::gemm::device::GemmUniversalAdapter<>
  |                   Host-side interface, argument marshaling, grid launch
  v
Layer 4: Kernel       cutlass::gemm::kernel::GemmUniversal<>
  |                   Device code, grid-level execution, tile scheduling
  v
Layer 3: Collective   CollectiveMma<> + CollectiveEpilogue<>
  |                   Temporal micro-kernels with arch-specific synchronization
  v
Layer 2: Tiled        cute::TiledMma<> + cute::TiledCopy<>
  |                   Spatial micro-kernels for arbitrary tiling
  v
Layer 1: Atom         cute::MMA_Atom<> + cute::Copy_Atom<>
                      Single hardware instruction wrappers
```

#### Layer 1: Atom

The atom is the smallest collection of threads and data that must cooperatively participate in a hardware-accelerated math or copy operation. It wraps a single PTX instruction (e.g., `mma.sync`, `wgmma.mma_async`, `cp.async`) with metadata about thread shape and arrangement.

#### Layer 2: Tiled MMA/Copy

Tiled operations replicate atoms across threads and values to cover arbitrary tile dimensions. `cute::TiledMma<>` defines how MMA atoms tile over a thread block's M, N, K dimensions. `cute::TiledCopy<>` does the same for data movement.

#### Layer 3: Collective

The collective layer provides temporal micro-kernels that orchestrate spatial micro-kernels using architecture-specific synchronization:

- `cutlass::gemm::collective::CollectiveMma<>`: The mainloop implementing MMA with pipelined data loading
- `cutlass::epilogue::collective::CollectiveEpilogue<>`: Post-GEMM processing (scaling, activation, store)

Collectives are parameterized by **dispatch policies** that specialize behavior per architecture:

```cpp
// Example dispatch policy for Hopper warp-specialized GEMM
MainloopSm90TmaGmmaWarpSpecialized<
  Stages,           // Number of pipeline stages
  ClusterShape,     // Thread block cluster dimensions
  KernelSchedule    // e.g., KernelTmaWarpSpecializedPingpong
>
```

#### Layer 4: Kernel

The kernel layer implements the CUDA `__global__` function. `GemmUniversal` manages:
- Grid-level tile scheduling (basic, persistent, stream-K)
- Cluster launch configuration
- Argument validation and workspace management

#### Layer 5: Device

The device layer provides the host-side interface. `GemmUniversalAdapter` handles argument construction, workspace allocation, grid dimension computation, and kernel launch.

### CollectiveBuilder

CUTLASS provides high-level builder interfaces that automatically deduce TiledMma, TiledCopy, and SMEM layouts from high-level parameters:

```cpp
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cute::arch::Sm90,                    // Architecture tag
    cute::arch::OpClassTensorOp,         // Operation class
    ElementA, LayoutA, AlignmentA,       // Operand A specification
    ElementB, LayoutB, AlignmentB,       // Operand B specification
    ElementAccumulator,                  // Accumulator type
    TileShape,                           // e.g., Shape<_128, _128, _64>
    ClusterShape,                        // e.g., Shape<_1, _1, _1>
    cutlass::gemm::collective::StageCountAuto,    // Auto-deduce stages
    cutlass::gemm::collective::KernelScheduleAuto // Auto-deduce schedule
>::CollectiveOp;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    EpilogueScheduleType,
    FusionOp                             // e.g., LinCombEltAct<ReLU, ...>
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>,     // Problem shape (M, N, K, L)
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
```

### Kernel Scheduling Modes

CUTLASS 3.x supports multiple scheduling strategies for Hopper and Blackwell:

#### Basic (Non-Persistent)

One CTA computes exactly one output tile. Simple but may underutilize the GPU when the tile grid is smaller than the number of SMs.

#### Persistent

One CTA per SM computes multiple output tiles across its lifetime, reducing kernel launch overhead and improving occupancy for small problems.

```
KernelTmaWarpSpecialized          // Basic warp-specialized
KernelTmaWarpSpecializedCooperative  // Cooperative
KernelTmaWarpSpecializedPingpong     // Ping-pong
```

#### Cooperative Scheduling

Both consumer warp groups work on the **same** output C tile. This enables larger tile sizes (e.g., 256x128) because the accumulator is distributed across two warp groups. Both groups share the MMA work, then cooperatively write the epilogue.

#### Ping-Pong Scheduling

Two consumer warp groups alternate: while one computes MMA on tile N, the other writes the epilogue for tile N-1. This hides epilogue latency behind MMA computation. The producer warps ping-pong between feeding each consumer group.

**Key difference from cooperative**: Each consumer group works on a **different** output tile, enabling MMA/epilogue overlap at the cost of smaller per-group tile sizes.

#### Warp Specialization

On Hopper, warps are split into dedicated roles:
- **Producer warps** (typically 1 warp): Issue TMA loads from GMEM to SMEM
- **Consumer warps** (typically 2 warp groups of 128 threads): Execute WGMMA on SMEM data

This is enabled by Hopper's hardware features:
- TMA requires minimal registers (single-threaded execution)
- WGMMA sources operands directly from SMEM
- `setmaxnreg` instruction allows dynamic per-warpgroup register (de)allocation

```cpp
if (isProducerWarp(threadIdx.x)) {
    // Issue TMA loads, signal full_barrier
    pipeline.producer_acquire(smem_pipe_write);
    copy(tma_load.with(*tmaBar), gA_tile, sA_tile);
    pipeline.producer_commit(smem_pipe_write);
} else {
    // Consumer: wait for data, execute WGMMA
    pipeline.consumer_wait(smem_pipe_read);
    gemm(tiled_mma, sA_fragment, sB_fragment, accumulator);
    pipeline.consumer_release(smem_pipe_read);
}
```

#### Stream-K Scheduling

A persistent approach that divides K-dimension work across CTAs for improved load balancing. Instead of each CTA computing a full K-reduction for one output tile, work is distributed more evenly across SMs.

---

## 3. CuTe (CUDA Templates)

### What Is CuTe

CuTe is the core library introduced in CUTLASS 3.0. It provides:
- A **Layout** type as a universal vocabulary for describing multidimensional data and thread arrangements
- A **Tensor** type combining data pointers with Layout metadata
- A **layout algebra** for manipulating layouts through composition, complement, inverse, etc.
- **Atom** abstractions wrapping hardware MMA and copy instructions

CuTe replaces all of CUTLASS 2.x's iterator types with a single, mathematically grounded abstraction.

### Layout

A CuTe Layout is a function from a logical coordinate space to a 1D index space, defined by a pair **(Shape, Stride)**.

```
Layout L = Shape : Stride
```

The layout function maps coordinate `c` to index via inner product:

```
L(c) = sum_i (c_i * stride_i)
```

#### Basic Examples

```cpp
// 8-element contiguous vector: shape=8, stride=1
auto layout_vec = make_layout(Int<8>{});                // 8:1

// 4x2 column-major matrix
auto layout_col = make_layout(make_shape(4, 2), make_stride(1, 4));  // (4,2):(1,4)
// Maps: (row, col) -> row * 1 + col * 4

// 4x2 row-major matrix
auto layout_row = make_layout(make_shape(4, 2), make_stride(2, 1));  // (4,2):(2,1)
// Maps: (row, col) -> row * 2 + col * 1

// Column-major via tag
auto layout_left = make_layout(make_shape(4, 2), LayoutLeft{});   // (4,2):(1,4)

// Row-major via tag
auto layout_right = make_layout(make_shape(4, 2), LayoutRight{});  // (4,2):(2,1)
```

#### Hierarchical (Nested) Layouts

Shapes and strides can be arbitrarily nested tuples (IntTuples):

```cpp
// Shape: (2, (2,2))  Stride: (4, (2,1))
auto nested = make_layout(
    make_shape(2, make_shape(2, 2)),
    make_stride(4, make_stride(2, 1))
);
// This has 3 modes but rank-2 at top level
// Accepts coordinates: (i, (j, k)) -> i*4 + j*2 + k*1
```

#### Coordinate Systems

A layout with shape `(3, (2, 3))` accepts three coordinate types:

| Coordinate Type | Example | Description |
|----------------|---------|-------------|
| 1-D (flat) | `0` through `17` | Colexicographic ordering of all elements |
| 2-D | `(0, 0)` through `(2, 5)` | Flattened inner dimensions |
| Natural (hierarchical) | `(0, (0, 0))` through `(2, (1, 2))` | Matches the shape hierarchy |

Conversion: `idx2crd(index, shape)` converts from flat to natural coordinates. `crd2idx(coord, shape, stride)` computes the inner product.

#### Layout Query Functions

| Function | Description |
|----------|-------------|
| `rank(L)` | Number of top-level modes |
| `size(L)` | Product of all shape elements = domain cardinality |
| `cosize(L)` | Codomain size (max index + 1) |
| `depth(L)` | Maximum nesting depth |
| `get<I>(L)` | Extract sublayout for mode I |

#### Layout Manipulation

| Operation | Description |
|-----------|-------------|
| `flatten(L)` | Remove all nesting |
| `group<B,E>(L)` | Nest modes [B, E) into a single mode |
| `select<I...>(L)` | Pick specific modes |
| `take<B,E>(L)` | Extract mode range |
| `append(L, M)` | Concatenate layouts |

#### Key Distinction from mdspan

Unlike C++23's `mdspan`, CuTe layouts are first-class citizens that are natively hierarchical, supporting coordinates at any compatible level of the hierarchy. This is essential for representing the complex thread-to-data mappings in GPU kernels.

---

## 4. CuTe Layout Algebra

The layout algebra is the mathematical foundation of CuTe, providing operations for composing, partitioning, and transforming layouts.

### Coalescence

Simplifies a layout without changing its function. Rules applied recursively:
1. Remove modes with shape = 1
2. Merge adjacent modes where `stride[i+1] = shape[i] * stride[i]`

```
coalesce((4, 1, 2):(1, 4, 4)) = (4, 2):(1, 4) = coalesce -> (8):(1)
```

This is the layout's "canonical form."

### Composition

Functional composition of layouts: `R := A o B` where `R(c) = A(B(c))`.

For a simple case with `B = (N):(r)`:

If the shape of A is left-divisible by the stride `r`, and the quotient is weakly left-divisible by `N`:

```
A o B = (N):(c * d_i)
```

where `c` and `d_i` come from the divisibility decomposition.

For multi-modal B, composition applies independently to each mode:

```
A o B = (A o B_0, A o B_1, ..., A o B_b)
```

**Critical property**: `size(A o B) = size(B)`, and the layout functions satisfy `f_{A o B} = f_A o f_B`.

### Complement

The complement of layout A with respect to size M finds a layout B that "completes" A's codomain:

```
complement(A, M) = (d_0, d_1/N_0*d_0, ..., M/N_a*d_a) : (1, N_0*d_0, N_1*d_1, ..., N_a*d_a)
```

**Key property**: When concatenated `(A, complement(A, M))`, the layout functions partition `[0, M)` bijectively. The complement and A have disjoint codomains (except 0).

### Inverse

The inverse reverses the coordinate-to-index mapping:

- **Left-inverse**: `L^{-1}` such that `L^{-1}(L(c)) = c` for all valid coordinates
- **Right-inverse**: `L^{-1}` such that `L(L^{-1}(i)) = i` for all valid indices

For a simple layout with all strides being powers of a common base, the inverse swaps shape and stride roles.

### Logical Division

Combines complementation and composition:

```
A / B := A o (B, complement(B, size(A)))
```

This partitions layout A according to B's structure with the remainder. This is the foundation of CuTe's tiling operations.

### Product Operations

Layout products combine thread layouts with value layouts:

- **Logical product**: `(thread_layout, value_layout) -> tiled_layout`
- **Blocked product**: Values are contiguous per thread
- **Raked product**: Values are interleaved across threads

These are used internally by `TiledMma` and `TiledCopy` to map threads to data.

---

## 5. CuTe Atoms

### Architecture

CuTe atoms follow a four-level hierarchy:

```
PTX Instruction
    -> Operation struct    (include/cute/arch/mma_*.hpp, copy_*.hpp)
        -> Traits struct   (include/cute/atom/mma_traits_*.hpp, copy_traits_*.hpp)
            -> Atom        (MMA_Atom, Copy_Atom)
                -> Tiled   (TiledMma, TiledCopy)
```

### MMA Atoms

#### Operation Struct

Wraps a raw PTX instruction without layout dependencies. Defines four register arrays:

```cpp
// Example: SM70_8x8x4_F32F16F16F32_NT
struct SM70_8x8x4_F32F16F16F32_NT {
    using DRegisters = float[8];       // Destination
    using ARegisters = uint32_t[2];    // Source A (packed FP16 pairs)
    using BRegisters = uint32_t[2];    // Source B
    using CRegisters = float[8];       // Accumulator

    CUTE_HOST_DEVICE static void fma(
        float      & d0, float      & d1, ...,
        uint32_t const& a0, uint32_t const& a1,
        uint32_t const& b0, uint32_t const& b1,
        float const& c0, float const& c1, ...
    ) {
        #if defined(CUTE_ARCH_MMA_SM70_ENABLED)
        asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 ...");
        #endif
    }
};
```

**Naming convention**: `SM{arch}_{M}x{N}x{K}_{TypeD}{TypeA}{TypeB}{TypeC}_{TransposeMode}`

#### Traits Struct

Provides logical metadata for the operation:

```cpp
template<> struct MMA_Traits<SM70_8x8x4_F32F16F16F32_NT> {
    using ValTypeD = float;
    using ValTypeA = half_t;
    using ValTypeB = half_t;
    using ValTypeC = float;

    using Shape_MNK = Shape<_8, _8, _4>;

    using ThrID = Layout<Shape<_4, _2>, Stride<_1, _16>>;
    // Maps logical thread IDs to physical warp lanes

    using ALayout = Layout<Shape<Shape<_4,_2>,_4>, Stride<Stride<_8,_4>,_1>>;
    using BLayout = ...;
    using CLayout = Layout<Shape<Shape<_2,_2,_2>, Shape<_2,_2,_2>>,
                           Stride<Stride<_1,_16,_4>, Stride<_8,_2,_32>>>;
};
```

The `(Thread, Value) -> MatrixCoordinate` layouts decouple the logical structure from hardware instruction specifics.

#### Hopper MMA Atoms

Hopper introduces warp-group-wide GMMA (128 threads):

```cpp
// SM90 GMMA example - 64x128x16 FP16
using MMA_Atom = MMA_Atom<SM90_64x128x16_F32F16F16F32_SS>;
// ThrID = Layout<_128, _1>  (trivial - all 128 threads)
// Sources A,B from SMEM (SS suffix), accumulator in registers
```

#### Blackwell MMA Atoms

Blackwell uses UMMA (tcgen05.mma) with single-threaded execution and Tensor Memory:

```cpp
// SM100 UMMA - both operands from SMEM
using MMA_Atom = MMA_Atom<SM100_MMA_F16BF16_SS>;
// ThrID repurposed as CTA layout (not thread indices)
// Accumulator stored in TMEM, not registers
```

### Copy Atoms

Copy atoms wrap data movement instructions:

| Copy Atom | Source | Destination | Architecture |
|-----------|--------|-------------|--------------|
| `UniversalCopy<>` | Any | Any | All (fallback) |
| `SM80_CP_ASYNC_CACHEALWAYS<>` | GMEM | SMEM | Ampere+ |
| `SM80_CP_ASYNC_CACHEGLOBAL<>` | GMEM | SMEM | Ampere+ |
| `SM90_TMA_LOAD` | GMEM | SMEM | Hopper+ |
| `SM90_TMA_LOAD_MULTICAST` | GMEM | SMEM (cluster) | Hopper+ |
| `SM90_TMA_STORE` | SMEM | GMEM | Hopper+ |
| `SM100_TMEM_LOAD` | SMEM/GMEM | TMEM | Blackwell |

### TiledMMA Construction

`make_tiled_mma()` replicates atoms across threads and values:

```cpp
// Create a 128x128x16 tiled MMA from a 64x128x16 atom
// by replicating 2x in the M dimension
auto tiled_mma = make_tiled_mma(
    MMA_Atom<SM90_64x128x16_F32F16F16F32_SS>{},
    Layout<Shape<_2, _1, _1>>{}   // 2x1x1 atom arrangement
);
```

The atom arrangement layout specifies how atoms are replicated across (M, N, K) dimensions, creating larger tiled operations.

---

## 6. CuTe Tensor and Algorithms

### Tensor

A CuTe Tensor combines an Engine (data pointer/iterator) and a Layout:

```cpp
// Global memory tensor
Tensor gA = make_tensor(
    make_gmem_ptr(ptr_A),
    make_layout(make_shape(M, K), make_stride(lda, 1))
);

// Shared memory tensor
Tensor sA = make_tensor(
    make_smem_ptr(smem_A),
    SmemLayoutA{}
);

// Register fragment
Tensor rA = make_tensor<half_t>(Shape<_4, _8>{});
```

Tensors carry layout information at compile time when possible, enabling the compiler to optimize indexing.

### Copy Algorithm

```cpp
// Default implementation based on tensor types
copy(src_tensor, dst_tensor);

// With explicit copy atom
copy(Copy_Atom<SM90_TMA_LOAD>{}, src_tensor, dst_tensor);

// Conditional copy with predication
copy_if(pred_tensor, src_tensor, dst_tensor);
```

The copy algorithm dispatches based on tensor parameter types:
- Source/destination memory spaces determine available instructions
- Tagging memory (e.g., `make_gmem_ptr`, `make_smem_ptr`) enables CuTe to select the fastest implementation
- May be sequential per-thread, parallel across warp/block/cluster, synchronous or asynchronous

### GEMM Algorithm

CuTe's `gemm()` dispatches to five modes based on tensor dimensionality:

| Source A | Source B | Accumulator | Operation |
|----------|----------|-------------|-----------|
| (V) | (V) | (V) | Vector dot product |
| (M) | (N) | (M,N) | Outer product |
| (M,K) | (N,K) | (M,N) | Matrix multiplication |
| (V,M) | (V,N) | (V,M,N) | Batched outer product |
| (V,M,K) | (V,N,K) | (V,M,N) | Batched matrix multiply |

With an MMA atom:

```cpp
gemm(tiled_mma, tensor_A, tensor_B, tensor_C);
// Dispatches to hardware MMA (mma.sync, wgmma, umma) based on atom type
```

### Additional Algorithms

- `axpby(alpha, x, beta, y)`: Computes alpha*x + beta*y
- `fill(tensor, value)`: Set all elements to a scalar
- `clear(tensor)`: Zero all elements

---

## 7. CUTLASS GEMM Kernels: Pipeline Design

### Pipeline Abstraction

CUTLASS provides pipeline classes for managing asynchronous producer-consumer patterns:

| Class | Use Case |
|-------|----------|
| `PipelineAsync<Stages>` | General async pipeline |
| `PipelineTmaAsync<Stages>` | TMA-based pipeline (Hopper) |
| `PipelineTransactionAsync<Stages>` | Transaction-based pipeline |

Each pipeline manages two barrier arrays (`full_barrier[]` and `empty_barrier[]`) of size N (number of stages), implemented as `mbarrier` objects in shared memory.

### Pipeline State

```cpp
PipelineState state;
// state.index()  - current stage (0 to N-1)
// state.phase()  - synchronization phase bit (0 or 1)
// ++state        - increment index mod N, flip phase on wraparound
```

### Four Key Methods

```cpp
pipeline.producer_acquire(state);  // Wait for empty buffer
// ... produce data (TMA load or cp.async) ...
pipeline.producer_commit(state);   // Signal buffer is full

pipeline.consumer_wait(state);     // Wait for full buffer
// ... consume data (WGMMA or thread math) ...
pipeline.consumer_release(state);  // Signal buffer is empty
```

### Multistage Pipeline (Ampere/Hopper)

Each warp performs both producer and consumer roles with hardware asynchrony:

```cpp
// Prologue: fill pipeline
for (int stage = 0; stage < Stages; ++stage) {
    pipeline.producer_acquire(pipe_write);
    copy(tma_load.with(*barrier), gmem_tile, smem_tile);
    pipeline.producer_commit(pipe_write);
    ++pipe_write;
}

// Mainloop: overlap load and compute
for (int k = Stages; k < k_tiles; ++k) {
    // Issue next load
    pipeline.producer_acquire(pipe_write);
    copy(tma_load.with(*barrier), gmem_tile_next, smem_next);
    pipeline.producer_commit(pipe_write);
    ++pipe_write;

    // Compute on current data
    pipeline.consumer_wait(pipe_read);
    gemm(tiled_mma, sA_frag, sB_frag, accumulator);
    pipeline.consumer_release(pipe_read);
    ++pipe_read;
}
```

### Warp-Specialized Pipeline (Hopper)

Warps split into dedicated producer and consumer roles:

```cpp
if (warp_role == Producer) {
    // Producer warp: TMA loads only
    for (int k = 0; k < k_tiles; ++k) {
        pipeline.producer_acquire(pipe_write);
        copy(tma_load.with(*tmaBar), gA(_,k), sA(_,pipe_write.index()));
        copy(tma_load.with(*tmaBar), gB(_,k), sB(_,pipe_write.index()));
        pipeline.producer_commit(pipe_write);
        ++pipe_write;
    }
} else {
    // Consumer warp group: WGMMA computation
    // Can set max registers via setmaxnreg for more register space
    for (int k = 0; k < k_tiles; ++k) {
        pipeline.consumer_wait(pipe_read);
        warpgroup_arrive();
        gemm(tiled_mma, sA_frag, sB_frag, accumulator);
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        pipeline.consumer_release(pipe_read);
        ++pipe_read;
    }
}
```

### Performance Impact

On H200 SXM: Tensor cores deliver 3,958 TFLOPS while memory bandwidth is 4.8 TB/s. Pipelining is essential to keep tensor cores fed. Warp-specialized kernels on Hopper achieve ~65% GPU utilization with FP16 through pipelining alone.

---

## 8. CUTLASS Epilogue Fusion and EVT

### Motivation

Many AI workloads require post-GEMM processing: bias addition, activation functions (ReLU, GELU, SiLU), scaling, quantization, or reductions (layernorm, rmsnorm). Without fusion, each operation requires a separate kernel launch with additional global memory round-trips.

Epilogue fusion applies these operations while the GEMM result is still in registers, before the final store to global memory, eliminating intermediate memory traffic and kernel launch overhead.

### Three Approaches to Epilogue Fusion

#### 1. DefaultEpilogue (Simplest)

For elementwise-only operations. Does not use the visitor tree:

```cpp
using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<
    cutlass::gemm::TagToStrideC_t<LayoutC>,
    cutlass::gemm::TagToStrideC_t<LayoutD>,
    cutlass::epilogue::thread::LinearCombination<ElementD, 1, ElementAccum, ElementCompute>
>;
```

#### 2. Built-in EVT Operations via CollectiveBuilder

Pre-constructed operations from `cutlass/epilogue/fusion/operations.hpp`:

```cpp
// Linear combination + ReLU activation
using FusionOp = cutlass::epilogue::fusion::LinCombEltAct<
    cutlass::epilogue::thread::ReLU,
    ElementD, ElementCompute, ElementC, ElementScalar
>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    EpilogueScheduleType,
    FusionOp
>::CollectiveOp;
```

Available built-in operations include:
- `LinCombEltAct<Activation>`: alpha * acc + beta * C, then activation
- `ScaledLinCombPerRowBiasEltAct<>`: With per-row bias
- `ScaledLinCombPerRowBiasEltActAmaxAux<>`: With absolute max tracking for FP8

#### 3. Custom Epilogue Visitor Trees

Hand-constructed trees for novel operations using the `Sm90EVT` (alias for `Sm90TreeVisitor`) template:

### EVT Node Types

| Node | Purpose |
|------|---------|
| `Sm90AccFetch` | Load GEMM accumulator output |
| `Sm90SrcFetch<ElementC>` | Load C input matrix |
| `Sm90ScalarBroadcast<ElementScalar>` | Broadcast a scalar value |
| `Sm90RowBroadcast<ElementBias>` | Broadcast a row vector (bias) |
| `Sm90ColBroadcast<Element>` | Broadcast a column vector |
| `Sm90AuxLoad<Element>` | Load auxiliary matrix |
| `Sm90AuxStore<Element>` | Store auxiliary output |
| `Sm90ScalarReduction<Op>` | Reduce to scalar |
| `Sm90RowReduction<Op>` | Reduce to row vector |
| `Sm90ColReduction<Op>` | Reduce to column vector |
| `Sm90Compute<Op, Output, Input, Round>` | Elementwise compute |
| `Sm90TopologicalVisitor<...>` | DAG (non-tree) computation graphs |

### EVT Composition Example

Linear combination: `D = activation(alpha * acc + beta * C)`

```cpp
using EVT_LinearCombination =
    Sm90EVT<Sm90Compute<homogeneous_multiply_add, ElementD, ElementCompute, RoundStyle>,
        Sm90ScalarBroadcast<ElementScalar>,  // beta
        Sm90SrcFetch<ElementC>,              // C
        Sm90EVT<Sm90Compute<multiplies, ElementCompute, ElementCompute, RoundStyle>,
            Sm90ScalarBroadcast<ElementScalar>,  // alpha
            Sm90AccFetch                         // accumulator
        >
    >;

// With activation applied on top:
using EVT_LinearCombActivation =
    Sm90EVT<Sm90Compute<cutlass::epilogue::thread::ReLU, ElementD, ElementCompute>,
        EVT_LinearCombination
    >;
```

### EVT Argument Structure

Arguments mirror the tree structure, with operation arguments appearing **after** child arguments:

```cpp
arguments.epilogue.thread = {
    {   // multiply_add node (root)
        {{beta}, {beta_ptr}},     // ScalarBroadcast args for beta
        {},                        // SrcFetch args (no extra args)
        {                          // Inner multiply node
            {{alpha}, {alpha_ptr}}, // ScalarBroadcast args for alpha
            {},                     // AccFetch args
            {}                      // multiplies op args
        },
        {}  // multiply_add op args
    },
    activation_args  // activation op args
};
```

### Topological Visitors (DAG Graphs)

When outputs of one node feed multiple consumers, the standard tree structure requires duplication. `Sm90TopologicalVisitor` enables DAG (Directed Acyclic Graph) structures where "the output of a node could be required by multiple other nodes."

### Quantization Epilogues

EVT supports quantization fusion. Example for INT8/FP8 GEMM:

```
ScaledEpilogue: acc_val (INT32/FP32)
    -> multiply by b_scales (row-wise broadcast)
    -> multiply by a_scales (column-wise broadcast)
    -> convert to output type (FP16/BF16) with rounding
```

### Limitations

EVT currently supports only Hopper (SM90) and Blackwell (SM100) architectures with warp-specialized kernels. For older architectures, use CUTLASS 2.x epilogue visitors.

---

## 9. CuTe DSL (Python)

### Overview

CuTe DSL, released with CUTLASS 4.0, is a Python-native programming model for writing high-performance CUDA kernels. It is fully consistent with CuTe C++ abstractions, exposing layouts, tensors, hardware atoms, and full control over the hardware thread and data hierarchy.

### Key Benefits

- **No C++ template metaprogramming**: Write kernels in Python syntax
- **33-116x faster compilation** than C++ CUTLASS templates (on Blackwell dense GEMM)
- **Performance parity** with C++ CUTLASS across dense GEMM, grouped GEMM, and FMHA
- **DLPack integration**: Directly accept PyTorch/JAX tensors without memory copies
- **JIT and AOT compilation**: Flexible deployment options

### Compilation Pipeline

CuTe DSL uses a three-stage hybrid pipeline:

#### Stage 1: Pre-Staging (AST Rewrite)

The `@jit` decorator triggers AST preprocessing. The preprocessor rewrites the decorated function, inserting callbacks around control-flow constructs (loops, branches, function boundaries) so program structure is captured explicitly.

#### Stage 2: Meta-Stage (Python Interpreter + Tracing)

The rewritten function executes with proxy tensor arguments:
- Callbacks emit structured IR for control flow
- Tensor operations are traced through overloaded operators
- Compile-time constants undergo partial evaluation

#### Stage 3: Object-Stage (Compiler Backend)

The IR undergoes optimization passes (tiling, vectorization, memory promotion) then lowering through MLIR to PTX/SASS assembly.

### Hybrid AST Rewrite + Tracing

The hybrid approach solves limitations of pure approaches:

- **Pure tracing** loses branches not taken, flattens loops, freezes control flow to a single path
- **Pure AST rewrite** is complex and slow for arithmetic
- **Hybrid**: AST rewrite captures structure (loops/branches); tracing captures arithmetic within each structured region

### JIT Decorator Modes

```python
@cute.jit(preprocess=True)   # Default: hybrid AST rewrite + tracing
def my_kernel(...):
    ...

@cute.jit(preprocess=False)  # Pure tracing (faster, straight-line only)
def simple_kernel(...):
    ...
```

### Core Abstractions in Python

```python
# Layouts
layout = cute.make_layout((128, 64), (64, 1))  # Row-major 128x64

# Tensors
tensor = cute.make_tensor(ptr, layout)

# TiledMMA
atom = tcgen05.MmaF16BF16Op(...)
tiled_mma = cute.make_tiled_mma(atom)
frag_A = tiled_mma.make_fragment_A(shared_A)

# TiledCopy
copy_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(...))
tiled_copy = tcgen05.make_tmem_copy(copy_atom, tensor)

# Compilation
compiled = cute.compile(my_kernel, *args)
compiled(*runtime_args)
```

### JIT Caching

`cute.compile()` compiles a kernel once, caches the compiled result, and reuses it for subsequent executions with the same signature.

### AOT (Ahead-of-Time) Compilation

For production deployment:

```python
# Export to compiled artifact
cute.export(my_kernel, *args, output_path="kernel.cubin")
```

Examples in `examples/python/CuTeDSL/cute/export/`.

### Constexpr Specialization

Values declared as `Constexpr` enable compile-time evaluation:

```python
@cute.jit
def kernel(x: cute.Tensor, TILE_M: cute.Constexpr = 128):
    # TILE_M is folded into generated code at compile time
    ...
```

### Meta-Stage vs Object-Stage

| Python `print()` | `cute.printf()` |
|-------------------|-----------------|
| Executes during compilation (meta-stage) | Compiles into kernel, executes on GPU |
| Shows what compiler sees | Shows runtime values |

---

## 10. CUTLASS Profiler

### Overview

The CUTLASS Profiler is a command-line tool for testing and benchmarking CUTLASS kernels from the CUTLASS Instance Library. It supports GEMM, Sparse GEMM, Conv2d, and Conv3d operations.

### Building

```bash
make cutlass_profiler -j

# For broader kernel coverage:
cmake .. -DCUTLASS_LIBRARY_KERNELS=all \
         -DCUTLASS_UNITY_BUILD_ENABLED=ON
```

### Modes

| Mode | Description |
|------|-------------|
| `--mode=profile` | Full verification and profiling (default) |
| `--mode=dry_run` | No kernel launches |
| `--mode=enumerate` | List all available operations |
| `--mode=trace` | Single device-side computation |

### GEMM Profiling

```bash
# Basic SGEMM
./cutlass_profiler --kernels=sgemm --m=3456 --n=4096 --k=4096

# Tensor Core GEMM
./cutlass_profiler --op_class=tensorop --m=3456 --n=4096 --k=8192

# Problem size sweep
./cutlass_profiler --kernels=cutlass_simt_sgemm_128x128_nn \
    --m=3456 --n=4096 --k=8:4096:8 --output=report.csv

# Filter kernels by name pattern
./cutlass_profiler --kernels=cutlass_tensorop_s*fprop_optimized_f16

# Exhaustive performance search
./cutlass_profiler --enable-kernel-performance-search \
    --sort-results-flops-per-sec
```

### Key Command-Line Options

| Option | Description |
|--------|-------------|
| `--m`, `--n`, `--k` | Problem dimensions (accept ranges `start:end:step`) |
| `--A=type:layout` | Operand A data type and layout |
| `--alpha`, `--beta` | Scaling factors |
| `--cta_m/n/k` | Thread block tile shape |
| `--cluster_m/n/k` | Cluster configuration |
| `--stages` | Pipeline stages |
| `--split_k_mode` | `serial` or `parallel` |
| `--split_k_slices` | Number of K splits |
| `--raster_order` | Tile rasterization direction |
| `--dist` | Data distribution: `uniform`, `gaussian`, `sequential`, `identity` |

### Data Initialization

```bash
--dist=uniform,min:-1,max:1,scale:-1
--dist=gaussian,mean:0,stddev:3
--dist=sequential,start:0,delta:1
```

### Performance Control

```bash
--profiling-iterations=100          # Fixed iteration count
--profiling-duration=500            # Duration in ms
--warmup-iterations=10              # Warmup runs
--workspace-count=4                 # Avoid cache residency
--use-cuda-graphs=true              # CUDA graph wrapping
```

### Output and Verification

```bash
--output=results.csv                # CSV export
--verification-enabled=true         # Enable correctness check
--verification-providers=cublas     # Compare against cuBLAS
--epsilon=1e-3                      # Error tolerance
--save-workspace=incorrect          # Save failing cases
```

### Reported Metrics

- **Runtime**: Kernel execution time (ms)
- **Throughput**: GFLOP/s
- **Bandwidth**: Effective GiB/s
- **FLOPs/Bytes**: Total operations and data movement

### Hopper (SM90) Instantiation Levels

The four-digit level `XYZW` controls kernel generation:

| Digit | Controls | Range |
|-------|----------|-------|
| W | Instruction shape | 0-3 |
| Z | MMA multiplier | 0-9 |
| Y | Cluster shape | 0-5 (1 to 16 CTAs) |
| X | Schedule pruning | 0 (pruned), >=1 (all) |

Example: `CUTLASS_LIBRARY_INSTANTIATION_LEVEL=0500`

### Convolution Profiling

```bash
./cutlass_profiler --operation=Conv2d \
    --Activation=f16:nhwc --Filter=f16:nhwc \
    --n=8 --h=224 --w=224 --c=128 --k=128 --r=3 --s=3

# Conv modes: --conv_kind={fprop|dgrad|wgrad}
```

---

## 11. CUTLASS for Attention

### Fused Multi-Head Attention (FMHA)

CUTLASS provides optimized FMHA implementations in `examples/41_fused_multi_head_attention/`:
- Forward pass: `fused_multi_head_attention_forward.cu`
- Backward pass: `fused_multi_head_attention_backward.cu`

### Algorithm

FMHA fuses three operations into a single kernel:
1. **GEMM-I**: `S = Q * K^T` (Query times Key-transpose)
2. **Online Softmax**: Row-wise softmax on S with running max/sum
3. **GEMM-II**: `O = softmax(S) * V` (Attention weights times Value)

Instead of materializing S and P to global memory, the algorithm tiles Q and K, computes partial softmax results in registers/SMEM, and rescales accumulators between iterations.

### FlashAttention-2 on Hopper with CUTLASS

Key implementation details from the research paper (arXiv:2312.11918):

#### Kernel Fusion

All three operations execute in a single CUDA kernel, bypassing intermediate global memory writes:
- S values reside in registers after GEMM-I
- Online softmax computes local max/sum using warp shuffle instructions
- GEMM-II sources operand A from registers (softmax output), operand B (V) from SMEM

#### Rescaling

Between K-tile iterations, the accumulator is rescaled:

```
O_new = exp(m_old - m_new) * O_old + softmax(S_partial) * V_partial
```

Where `m_old` and `m_new` are the running row-wise maximum values.

#### Layout Transformation

The accumulator layout from GEMM-I must be reshuffled for GEMM-II's operand A. `ReshapeTStoTP()` converts between tile dimensions (e.g., 64x128 to 64x16) by adjusting strides in registers.

#### Cross-GEMM Pipelining

Rather than standard software pipelining within one GEMM, the implementation overlaps loads for GEMM-II with GEMM-I computation and vice versa, exploiting the two-GEMM structure.

#### TMA Usage

- Asynchronous GMEM-to-SMEM copies via TMA
- K-major 128-byte swizzling for bank conflict avoidance
- Single-threaded TMA execution model

#### Shared Memory Layout

```cpp
// K-major swizzled layout for Q in SMEM
using SmemLayoutQ = decltype(
    composition(Swizzle<3,3,3>{},
    Layout<Shape<_64,_64>, Stride<_64,_1>>{}));
```

V is transposed in SMEM without data movement by composing with a transposing layout.

#### Performance

On H100 PCIe with FP16:
- 2.5-3x faster than CUTLASS 3.3 reference
- 20-50% faster than Flash Attention 2
- Optimal tile shapes: 64x128 or 128x64 depending on head dimension
- 128x128 tiles suffer register spills

### Key Examples

- `examples/41_fused_multi_head_attention/` - Forward and backward FMHA
- `examples/python/CuTeDSL/blackwell/fmha.py` - CuTe DSL Python FMHA for Blackwell
- Meta's xFormers uses CUTLASS-based attention kernels

---

## 12. CUTLASS on Blackwell

### Fifth-Generation Tensor Cores

Blackwell (SM100/SM120) introduces:

| Feature | Details |
|---------|---------|
| Data Types | FP64, TF32, BF16, FP16, FP8, FP6, FP4, INT8 |
| Block Scaling | Native NVFP4 (micro-block scaling with FP8 E4M3 factors) |
| Peak FP4 | 7,702.5 TFLOPS at 2.4 GHz |
| Peak FP6 | 5,134.8 TFLOPS at 2.4 GHz |
| Instruction | UMMA (tcgen05.mma) replacing Hopper's WGMMA |
| Memory | Tensor Memory (TMEM) - 256KB per SM |

### Tensor Memory (TMEM)

TMEM is a dedicated 256KB per-SM resource for tensor core accumulation:

- **Organization**: 2D structure with 512 columns x 128 rows (lanes) of 32-bit cells
- **Address format**: Lane ID in bits 31-16, column in bits 15-0
- **Allocation**: Dynamic via `tcgen05.alloc` in column units (minimum 32 columns, power-of-2)
- **Deallocation**: Explicit via `tcgen05.dealloc`

Access instructions:
- `tcgen05.ld`: TMEM to registers (warp-synchronized)
- `tcgen05.st`: Registers to TMEM
- `tcgen05.cp`: SMEM to TMEM

### UMMA Instructions (tcgen05.mma)

UMMA replaces Hopper's deprecated WGMMA with key differences:

| Feature | WGMMA (Hopper) | UMMA (Blackwell) |
|---------|----------------|-------------------|
| Execution | Warp-group (128 threads) | Single-threaded, asynchronous |
| Accumulator | Registers | TMEM |
| Operand A | SMEM or registers | TMEM or SMEM |
| Operand B | SMEM | SMEM |
| Max dimensions | 64xNx16 | 64xNx16 or 128xNx16 (N up to 256) |
| Register pressure | High (accumulator in regs) | Minimal (accumulator in TMEM) |

### 2-SM Cooperative Execution

Two adjacent CTAs within a cluster ("CTA pair") collaborate on a single UMMA:

```cpp
// Extract peer CTA coordinate
auto mma_v = get<0>(mma_coord_vmnk);  // 0 or 1
ThrMMA cta_mma = tiled_mma.get_slice(mma_v);
```

Only one thread from one CTA launches the UMMA instruction, but the work is distributed across two SMs' tensor cores.

### Block Scaling

Blackwell natively supports block-scaled operations through UMMA instruction descriptors:

- **NVFP4**: Values grouped in blocks of 16, each with FP8 (E4M3) scale factor, plus per-tensor FP32 scale
- **MXFP4/MXFP6/MXFP8**: OCP (Open Compute Project) standard block-scaled formats
- Scale factors are applied automatically during the MMA operation

### Blackwell GEMM Pipeline

The instruction model decouples GEMM phases:

1. **Pre-processing** (registers): Prepare descriptors
2. **Data load to TMEM** (TMA): GMEM -> SMEM -> TMEM for operand A
3. **UMMA computation** (single thread, no registers consumed): Accumulate in TMEM
4. **TMEM extraction** (`tcgen05.ld`): TMEM -> registers for epilogue
5. **Epilogue** (registers, full warpgroup): Post-processing and store

### CuTe Abstractions for Blackwell

```cpp
// MMA Atom for Blackwell
using MMA_Atom = SM100_MMA_F16BF16_SS;  // Both operands from SMEM
// SS = SMEM-SMEM, TS = TMEM-SMEM

// ThrID is repurposed as CTA layout (not thread indices)
// For single-CTA: Layout<_1>
// For 2-SM cooperative: Layout<_2>

// SMEM layouts with Blackwell-specific swizzle
using SmemLayoutA = decltype(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<ElementA>{},
    make_shape(TileM, TileK, Int<Stages>{})));
```

### Key Blackwell Examples in CUTLASS

- `examples/cute/tutorial/blackwell/` - CuTe Blackwell tutorials
  - `01_mma_tma_1sm_sm100.cu` - Single-SM UMMA
  - `04_mma_tma_2sm_sm100.cu` - 2-SM cooperative UMMA
- `examples/72_blackwell_narrow_precision_gemm/` - FP4/FP6/FP8 GEMM
  - `72a_blackwell_narrow_precision_gemm.cu` - Basic narrow precision
  - `72c_blackwell_mixed_mxfp8_bf16_gemm.cu` - Mixed MXFP8+BF16
- `examples/python/CuTeDSL/blackwell/` - Python DSL examples
  - `grouped_gemm.py` - Grouped GEMM
  - `fmha.py` - Fused multi-head attention

---

## 13. CUTLASS Examples

### Key Examples from the Repository

| Example | Description | Architecture |
|---------|-------------|-------------|
| `00_basic_gemm` | Simplest GEMM using device-level API | All |
| `14_ampere_tf32_tensorop_gemm` | TF32 Tensor Core GEMM | Ampere |
| `24_gemm_grouped` | Grouped GEMM with distinct problem sizes | Ampere+ |
| `35_gemm_softmax` | GEMM with fused softmax epilogue | Ampere |
| `41_fused_multi_head_attention` | Forward + backward FMHA | Hopper |
| `48_hopper_warp_specialized_gemm` | Warp-specialized persistent GEMM | Hopper |
| `49_hopper_gemm_with_collective_builder` | CollectiveBuilder-based GEMM | Hopper |
| `50_hopper_gemm_with_epilogue_swizzle` | Epilogue swizzle patterns | Hopper |
| `55_hopper_mixed_dtype_gemm` | Mixed precision GEMM | Hopper |
| `56_hopper_gemm_stream_k` | Stream-K scheduling | Hopper |
| `58_ada_fp8_gemm` | FP8 GEMM | Ada |
| `62_hopper_sparse_gemm` | Structured sparsity | Hopper |
| `72_blackwell_narrow_precision_gemm` | FP4/FP6/FP8 GEMM | Blackwell |
| `cute/tutorial/` | CuTe tutorials (layouts, copy, MMA) | Various |
| `cute/tutorial/blackwell/` | Blackwell-specific CuTe tutorials | Blackwell |

### Grouped GEMM

Computes a batch of GEMM operations with distinct problem sizes. Pointers to matrices in global memory are passed in an array:

```cpp
// Each problem has its own M, N, K, pointers, and strides
struct GemmGroupedProblem {
    cutlass::gemm::GemmCoord problem_size;
    ElementA* ptr_A;
    ElementB* ptr_B;
    ElementC* ptr_C;
    ElementD* ptr_D;
    int64_t lda, ldb, ldc, ldd;
};
```

Blackwell adds `acc_scale` grouped mixed-input GEMM for better decode performance.

### Mixed Precision GEMM

CUTLASS supports extensive mixed-precision configurations:

- FP16 A/B with FP32 accumulator
- BF16 A/B with FP32 accumulator
- FP8 (E4M3/E5M2) A/B with FP32 accumulator
- INT8 A/B with INT32 accumulator
- FP4 A/B with FP32 accumulator (Blackwell)
- Mixed A/B types (e.g., FP8 A + BF16 B)
- Block-scaled formats (NVFP4, MXFP4, MXFP6, MXFP8)

---

## 14. Performance Tuning

### Key Tuning Parameters

| Parameter | Description | Impact |
|-----------|-------------|--------|
| **Tile Shape** (CTA_M x CTA_N x CTA_K) | Thread block's output tile and K-slice | Determines parallelism and register/SMEM usage |
| **Cluster Shape** | Multi-CTA cluster dimensions (Hopper+) | Controls TMA multicast and inter-CTA cooperation |
| **Pipeline Stages** | Number of SMEM buffers for pipelining | More stages = better latency hiding, more SMEM |
| **Kernel Schedule** | Cooperative, ping-pong, basic | Affects MMA/epilogue overlap |
| **Warp Count** | Number of warps per CTA | Affects occupancy and register pressure |
| **Swizzle Pattern** | SMEM layout permutation | Reduces bank conflicts |
| **Split-K** | Partition K-dimension across CTAs | Improves parallelism for small M, N |
| **Raster Order** | CTA traversal order | Affects L2 cache hit rates |

### Tile Size Guidelines

| Architecture | Typical Tile Shapes | Notes |
|-------------|-------------------|-------|
| Ampere | 128x128x32, 256x64x32, 64x64x64 | Balance register usage and parallelism |
| Hopper | 128x128x64, 128x256x64, 256x128x64 | Warp-group MMA prefers 64 in K |
| Blackwell | 128x256x128, 256x128x128 | Larger K-tiles with TMEM accumulation |

### Swizzle Patterns

Swizzle reorders SMEM addresses to avoid bank conflicts during tensor core loads:

| Swizzle | Bytes | Use Case |
|---------|-------|----------|
| None | 0 | Simple cases, debugging |
| `Swizzle<1,0,3>` | 32B | Minimal swizzle |
| `Swizzle<2,0,3>` | 64B | Medium swizzle |
| `Swizzle<3,3,3>` | 128B | Maximum swizzle (recommended for Hopper WGMMA) |

### Pipeline Stage Selection

- More stages: Better memory latency hiding, but consumes more SMEM
- Fewer stages: Less SMEM, higher occupancy potential
- `StageCountAuto`: CUTLASS automatically selects based on SMEM budget
- Rule of thumb: 2-4 stages for Ampere, 4-8 stages for Hopper

### Autotuning

#### nvMatmulHeuristics (CUTLASS 4.2+)

A GPU kernel meta-parameter optimization module providing fast heuristics:

```bash
cmake .. -DCUTLASS_LIBRARY_HEURISTICS_PROBLEMS_FILE=problems.json \
         -DCUTLASS_LIBRARY_HEURISTICS_CONFIGS_PER_PROBLEM=16
```

Problem specification:
```json
{
    "m": 4096, "n": 4096, "k": 4096,
    "layout": "tnn",
    "dtype_a": "f16", "dtype_b": "f16",
    "dtype_c": "f16", "dtype_d": "f16"
}
```

Results: Achieves 96% of peak performance in ~150 minutes vs 700+ minutes for exhaustive search (Llama 3 405B on H100). On DeepSeek-R1 671B on B200: 99% of exhaustive with 5x speedup.

#### CuTe DSL Autotuning

```python
# Define search space
search_space = {
    "TILE_M": [64, 128, 256],
    "TILE_N": [64, 128, 256],
    "TILE_K": [32, 64, 128],
    "STAGES": [2, 3, 4, 5, 6],
    "CLUSTER_M": [1, 2],
    "CLUSTER_N": [1, 2],
}

# Exhaustive or heuristic search across configurations
for config in itertools.product(*search_space.values()):
    kernel = cute.compile(gemm_kernel, *args, **dict(zip(search_space.keys(), config)))
    time_ms = benchmark(kernel, *runtime_args)
```

#### Profiler-Based Autotuning

```bash
# Sweep tile sizes and find best kernel
./cutlass_profiler --op_class=tensorop \
    --m=4096 --n=4096 --k=4096 \
    --enable-kernel-performance-search \
    --sort-results-flops-per-sec \
    --output=sweep_results.csv
```

### Benchmarking Best Practices

1. **Lock GPU clocks** for consistent measurements
2. **Use multiple workspace allocations** (`--workspace-count=4`) to avoid cache residency effects
3. **Warm up** (minimum 10 iterations) before measuring
4. **Use duration-based profiling** (50-500ms) rather than fixed iteration counts
5. **Disable verification** during performance sweeps
6. **Test with realistic data distributions** (uniform random, not zeros)

---

## References

- [NVIDIA CUTLASS GitHub Repository](https://github.com/NVIDIA/cutlass)
- [NVIDIA CUTLASS Documentation](https://docs.nvidia.com/cutlass/latest/)
- [CUTLASS 3.x Design](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cutlass_3x_design.html)
- [CUTLASS 3.0 GEMM API](https://docs.nvidia.com/cutlass/media/docs/cpp/gemm_api_3x.html)
- [CuTe Layout Algebra Documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html)
- [CuTe MMA Atoms Documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0t_mma_atom.html)
- [CuTe Tensor Algorithms](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/04_algorithms.html)
- [CuTe DSL Overview](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html)
- [CuTe DSL Code Generation](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_code_generation.html)
- [CUTLASS Profiler Documentation](https://docs.nvidia.com/cutlass/media/docs/cpp/profiler.html)
- [CUTLASS 3.x Blog Post](https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design)
- [CuTe Blog Post](https://developer.nvidia.com/blog/cutlass-principled-abstractions-for-handling-multidimensional-data-through-tensors-and-spatial-microkernels)
- [CuTe DSL Blog Post](https://developer.nvidia.com/blog/achieve-cutlass-c-performance-with-python-apis-using-cute-dsl/)
- [Epilogue Fusion with EVT - Colfax Research](https://research.colfax-intl.com/epilogue_visitor_tree/)
- [CUTLASS Pipelining Tutorial - Colfax Research](https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/)
- [CUTLASS Persistent Kernels and Stream-K - Colfax Research](https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/)
- [CUTLASS Blackwell GEMM Tutorial - Colfax Research](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/)
- [FlashAttention-2 on Hopper with CUTLASS (arXiv:2312.11918)](https://arxiv.org/html/2312.11918v1)
- [EVT ASPLOS'24 Paper](https://dl.acm.org/doi/10.1145/3620666.3651369)
- [CuTe Layout Algebra Paper (arXiv:2603.02298)](https://arxiv.org/abs/2603.02298)
- [CuTe Layout Algebra - Lei Mao](https://leimao.github.io/article/CuTe-Layout-Algebra/)
- [CuTe Tiled MMA - Lei Mao](https://leimao.github.io/blog/CuTe-Tiled-MMA/)
- [Categorical Foundations for CuTe Layouts - Colfax Research](https://research.colfax-intl.com/categorical-foundations-for-cute-layouts/)
- [Deep Dive on CUTLASS Ping-Pong GEMM - PyTorch Blog](https://pytorch.org/blog/cutlass-ping-pong-gemm-kernel/)
- [Autotuning with nvMatmulHeuristics - NVIDIA Blog](https://developer.nvidia.com/blog/improving-gemm-kernel-auto-tuning-efficiency-on-nvidia-gpus-with-heuristics-and-cutlass-4-2)
- [NVIDIA/cutlass DeepWiki](https://deepwiki.com/NVIDIA/cutlass)
- [MMA Atoms in CuTe - Veitner Blog](https://veitner.bearblog.dev/mma-atoms-in-cute/)
- [Hopper/Blackwell MMA Layouts - VJ Krish](https://vjkrish.com/2026/01/19/Mma_Layouts.html)
