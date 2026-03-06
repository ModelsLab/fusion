---
id: cutlass_cute_deep_dive
kind: document
title: CUTLASS and CuTe - Complete Deep Dive
category: kernel-lib
summary: Comprehensive guide to NVIDIA CUTLASS 3.x and CuTe covering architecture, layout algebra, MMA atoms, GEMM kernel design, epilogue fusion, Python DSL, and Blackwell support.
tags:
  - cutlass
  - cute
  - tensor-cores
  - gemm
  - epilogue
  - evt
  - tma
  - warp-specialization
source_ids:
  - nvidia-cutlass-overview
  - nvidia-cute-dsl
  - nvidia-blackwell-cutlass
gpu_families:
  - Ampere
  - Ada
  - Hopper
  - Blackwell
operators:
  - matmul
  - gemm
  - attention
precision:
  - fp16
  - bf16
  - fp8
  - int8
  - int4
  - fp4
---

# CUTLASS and CuTe - Complete Deep Dive

## CUTLASS Overview

CUTLASS (CUDA Templates for Linear Algebra Subroutines) is NVIDIA's open-source library for high-performance GEMM and related operations. It provides:

1. **Production-quality GEMM kernels** matching or exceeding cuBLAS
2. **Composable epilogue system** (EVT) for fusing post-GEMM operations
3. **CuTe layout algebra** for expressing data layouts and transformations
4. **Multi-architecture support**: Ampere, Ada, Hopper, Blackwell
5. **Python DSL** for writing kernels without C++

### CUTLASS 2.x vs 3.x

| Feature | CUTLASS 2.x | CUTLASS 3.x |
|---------|------------|-------------|
| Layout system | Manual, template-heavy | CuTe (algebraic) |
| Hopper support | Limited | Full (TMA, WGMMA, clusters) |
| Epilogue | Fixed templates | EVT (composable tree) |
| Python DSL | No | Yes (CuTe DSL) |
| Code style | Deeply nested templates | Cleaner CuTe abstractions |
| Kernel types | Cooperative | Cooperative, Persistent, Warp-Specialized |

## CuTe Fundamentals

### Layouts: The Core Abstraction

A CuTe Layout maps logical coordinates to physical offsets:
```
Layout = (Shape, Stride)

Shape: tuple of dimensions, e.g., (M, N) or ((M1, M2), N)
Stride: tuple of strides, e.g., (N, 1) for row-major or (1, M) for col-major
```

Examples:
```cpp
// Row-major 4x8 matrix:
Layout row_major = make_layout(make_shape(4, 8), make_stride(8, 1));
// offset(i, j) = i * 8 + j * 1

// Column-major 4x8 matrix:
Layout col_major = make_layout(make_shape(4, 8), make_stride(1, 4));
// offset(i, j) = i * 1 + j * 4

// Hierarchical (tiled) layout - 4x8 tiled into 2x4 tiles:
Layout tiled = make_layout(
    make_shape(make_shape(2, 2), make_shape(4, 2)),  // ((tile_rows, num_tiles_m), (tile_cols, num_tiles_n))
    make_stride(make_stride(4, 8), make_stride(1, 16))
);
```

### Layout Operations

**Composition**: `compose(Layout_A, Layout_B)` - apply B's indexing to A
```
// If A maps logical coords to intermediate, and B maps intermediate to physical:
// compose(A, B) maps logical directly to physical
```

**Complement**: `complement(Layout, size)` - find the "missing" strides
```
// For a layout that covers some elements, complement gives the layout that
// covers the remaining elements in the co-domain
```

**Product**: `logical_product(Layout_A, Layout_B)` - tile A by B
```
// Creates a hierarchical layout where A is the inner tile and B tiles over it
```

**Coalesce**: `coalesce(Layout)` - simplify by merging adjacent modes with compatible strides
```
// Layout((4, 2), (1, 4)) → Layout(8, 1)  // consecutive can be merged
```

### Tensors in CuTe
```cpp
// A Tensor = data pointer + Layout
auto tensor = make_tensor(data_ptr, layout);

// Access element:
tensor(i, j)  // returns reference to element at logical (i, j)

// Partition tensor for thread block:
auto tiled = local_tile(tensor, tile_shape, thread_block_coord);

// Partition for individual thread:
auto thread_frag = local_partition(tiled, thread_layout, thread_idx);
```

## CuTe Atoms

### Copy Atoms
Describe how to move data between memory levels:

```cpp
// Ampere: cp.async for global → shared memory
using CopyAtom = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, half_t>;
// Copies 128 bits (8 half values) per thread asynchronously

// Hopper: TMA for bulk async copies
using CopyAtom = Copy_Atom<SM90_TMA_LOAD, half_t>;
// TMA handles entire tile copy without warp involvement

// Register ↔ Shared: done implicitly during MMA or explicitly
using CopyAtom = Copy_Atom<DefaultCopy, half_t>;
```

### MMA Atoms
Describe tensor core operations:

```cpp
// Ampere: warp-level MMA
using MmaAtom = MMA_Atom<SM80_16x8x16_F16F16F16F16>;
// m16n8k16: 16 rows, 8 cols, 16 reduction per instruction
// Per warp: 256 FMA ops

// Hopper: warp-group-level WGMMA
using MmaAtom = MMA_Atom<SM90_64x256x16_F16F16F16F16_SS>;
// m64n256k16: much larger tile per instruction
// _SS: both operands from shared memory
// _RS: A from registers, B from shared memory

// Blackwell: fifth-gen tensor core
using MmaAtom = MMA_Atom<SM100_64x256x16_F16F16F16F16>;
```

## CUTLASS GEMM Kernel Architecture

### Layer Structure (CUTLASS 3.x)

```
┌─────────────────────────────┐
│     GemmUniversal<Config>    │ ← Top-level API
├─────────────────────────────┤
│     CollectiveMainloop       │ ← Main GEMM loop (loads + MMA)
│     CollectiveEpilogue       │ ← Post-GEMM operations
├─────────────────────────────┤
│     TiledMma                 │ ← MMA atom tiled to CTA level
│     TiledCopy                │ ← Copy atom tiled for loads
├─────────────────────────────┤
│     MMA_Atom, Copy_Atom      │ ← Hardware instruction wrappers
└─────────────────────────────┘
```

### Kernel Scheduling Policies

**Cooperative** (default, Ampere/Ada):
```
All warps in CTA participate in both loading and computing
load_A(); load_B(); sync(); mma(); sync(); // repeat
```

**Warp-Specialized** (Hopper):
```
Producer warps (1-2 warps): Load data via TMA
Consumer warps (remaining): Compute via WGMMA
Connected by async barriers (named barriers)

Producer loop:                 Consumer loop:
  tma_load(smem_A, gmem_A)     barrier_wait(stage)
  tma_load(smem_B, gmem_B)     wgmma(smem_A, smem_B, accum)
  barrier_arrive(stage)         barrier_arrive(consumed_stage)
  advance_stage()               advance_stage()
```

**Persistent** (Hopper):
```
Kernel launches with num_SMs CTAs (one per SM)
Each CTA loops over multiple output tiles
Work is distributed via atomic counter

while (tile_idx = atomicAdd(&global_counter, 1) < total_tiles):
    compute_tile(tile_idx)

Benefits: reduces launch overhead, enables load balancing
```

**Ping-Pong** (Hopper, for maximum throughput):
```
Two warp groups alternate between loading and computing:
WG0: load stage A → compute stage B → load stage C → compute stage D
WG1: compute stage A → load stage B → compute stage C → load stage D
Maximum overlap of memory and compute
```

## Epilogue Visitor Tree (EVT)

The EVT system allows composing arbitrary epilogue operations as a tree:

### Built-in Visitors
```cpp
// Scalar broadcast: alpha * D
using Alpha = ScalarBroadcast<float>;

// Matrix multiplication (C * beta)
using BetaC = ScalarBroadcast<float, Multiply, Load<C_type>>;

// Bias addition (per-row or per-column)
using Bias = Sm90RowBroadcast<half_t>;

// Activation functions
using Activation = Sm90EVT<ReLU, InputTree>;
using Activation = Sm90EVT<GeLU, InputTree>;
using Activation = Sm90EVT<SiLU, InputTree>;

// Quantization
using Quantize = Sm90EVT<FloatToInt8, InputTree>;
```

### Composing EVT
```cpp
// D = gelu(alpha * Acc + bias)
using EVT =
    Sm90EVT<GeLU,            // outer: gelu(...)
        Sm90EVT<Add,         //   inner: (...) + bias
            Sm90EVT<Multiply,//     alpha * Acc
                ScalarBroadcast<float>,  // alpha
                Sm90AccFetch             // accumulator
            >,
            Sm90RowBroadcast<half_t>    // bias (per-row)
        >
    >;
```

### Example: GEMM + Bias + GeLU + Quantize to INT8
```cpp
using Epilogue =
    Sm90EVT<Store<int8_t>,
        Sm90EVT<FloatToInt8,
            Sm90EVT<GeLU,
                Sm90EVT<Add,
                    Sm90EVT<Multiply,
                        ScalarBroadcast<float>,
                        Sm90AccFetch
                    >,
                    Sm90RowBroadcast<half_t>
                >
            >,
            ScalarBroadcast<float>  // quantization scale
        >
    >;
```

## CuTe Python DSL

### Overview
Write CuTe kernels in Python, compile to CUDA:

```python
from cutlass.cute import *
from cutlass.cute.runtime import *

@cute_kernel
def my_gemm(A: Tensor, B: Tensor, C: Tensor, M: int, N: int, K: int):
    # Get thread block and thread indices
    bx = blockIdx.x
    by = blockIdx.y
    tx = threadIdx.x

    # Define tiles
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32

    # ... CuTe operations in Python
```

### Compilation Options
```python
# JIT compilation (during development)
kernel = compile(my_gemm, target="sm_90")

# AOT compilation (for deployment)
kernel = aot_compile(my_gemm, target="sm_90", output="my_kernel.cubin")

# With autotuning
configs = [
    {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
    {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64},
    {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64},
]
best_kernel = autotune(my_gemm, configs, sample_inputs)
```

## CUTLASS for Attention

### Fused Multi-Head Attention
CUTLASS provides FMHA (Fused Multi-Head Attention) examples:

```
Structure:
1. Q @ K^T GEMM (with online softmax during accumulation)
2. Softmax(scores) @ V GEMM

Key optimizations:
- Tile across sequence length (like FlashAttention)
- Online softmax: track running max and sum
- Accumulate in FP32, output in FP16
- Optional: causal masking applied during score computation
- Optional: FP8 for Q, K (with FP32 accumulation)
```

### CUTLASS 3.x FMHA on Hopper
```
Uses warp-specialized kernel:
- Producer warps: TMA loads for Q, K, V tiles
- Consumer warps: WGMMA for matmuls
- Pipelined: overlap K,V loads with Q@K^T computation

Performance: competitive with FlashAttention-3 for certain configs
```

## Blackwell Support in CUTLASS

### Fifth-Gen Tensor Cores
- FP4 (E2M1) support: 2x throughput vs FP8
- Larger MMA shapes for higher throughput
- Second-gen Transformer Engine integration

### New Features
- Enhanced TMA with decompression engine
- Larger shared memory / L2 cache utilization
- Updated kernel scheduling for GB100/GB202

### FP4 GEMM in CUTLASS
```cpp
// Blackwell FP4 GEMM:
using ElementA = cutlass::float_e2m1_t;  // FP4
using ElementB = cutlass::float_e2m1_t;
using ElementC = cutlass::half_t;
using ElementAccumulator = float;

using MmaAtom = MMA_Atom<SM100_64x256x64_F16E2M1E2M1_SS>;  // FP4 MMA
```

## CUTLASS Performance Tuning Guide

### Tile Size Selection
```
Rule 1: CTA tile should be large enough for high arithmetic intensity
  AI = 2 * TILE_M * TILE_N / ((TILE_M + TILE_N) * dtype_size)
  Want AI > ridge_point for compute-bound execution

Rule 2: CTA tile should fit in shared memory
  smem = (TILE_M * TILE_K + TILE_K * TILE_N) * dtype_size * num_stages
  Must be < max_smem_per_SM (228 KB on H100)

Rule 3: Enough CTAs to fill the GPU
  num_CTAs = ceil(M/TILE_M) * ceil(N/TILE_N)
  Want num_CTAs >= num_SMs (132 on H100) for full occupancy

Rule 4: Warp tile should align with MMA instruction
  For m16n8k16: warp tile should be multiple of 16x8
  Common: 64x64, 32x64, 64x32 per warp
```

### Common Configurations

**H100 FP16 GEMM (large shapes)**:
```
TILE_M=128, TILE_N=256, TILE_K=64
Stages=4, Warps=8
Kernel: warp-specialized persistent
Expected: 85-95% of peak TC throughput
```

**H100 FP8 GEMM**:
```
TILE_M=128, TILE_N=256, TILE_K=128  (larger K because FP8 is half the bytes)
Stages=4, Warps=8
Expected: 85-95% of peak FP8 throughput
```

**A100 FP16 GEMM**:
```
TILE_M=128, TILE_N=128, TILE_K=32
Stages=3, Warps=4
Kernel: cooperative
Expected: 80-90% of peak
```

### When CUTLASS vs cuBLAS

| Scenario | Winner | Why |
|----------|--------|-----|
| Standard dense GEMM | Tie | Both highly optimized |
| GEMM + custom epilogue | CUTLASS | EVT is unmatched |
| Grouped GEMM (MoE) | CUTLASS | GroupedGemm kernel |
| Mixed precision (W4A16) | CUTLASS/Custom | Need dequant fusion |
| FP8 on Hopper | Tie | Both excellent |
| FP4 on Blackwell | CUTLASS | First to support |
| Non-standard shapes | CUTLASS | Can autotune for specific shapes |
| Quick integration | cuBLAS | Simpler API |
