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

## Practical CUTLASS Recipes

### Recipe 1: Build and Run CUTLASS Examples

```bash
# Clone and build
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass
mkdir build && cd build
cmake .. -DCUTLASS_NVCC_ARCHS="89;90" -DCUTLASS_ENABLE_EXAMPLES=ON
make -j$(nproc) 00_basic_gemm

# Run basic GEMM example
./examples/00_basic_gemm/00_basic_gemm
```

**Flag breakdown:**

- `-DCUTLASS_NVCC_ARCHS="89;90"` -- Semicolon-separated list of SM architectures to compile for. `89` = Ada Lovelace (RTX 4090, L40), `90` = Hopper (H100). Other common values: `80` (A100), `86` (RTX 3090), `100` (Blackwell). Only include architectures you actually need; each one increases compile time significantly.
- `-DCUTLASS_ENABLE_EXAMPLES=ON` -- Builds the examples under `examples/`. Off by default to speed up builds.
- `make -j$(nproc) 00_basic_gemm` -- Builds only the `00_basic_gemm` target. Building all of CUTLASS takes a long time (30+ minutes); always specify the target you need.

**Which architectures to target:**

| GPU | SM Arch | Notes |
|-----|---------|-------|
| A100 | `80` | CUTLASS 2.x and 3.x both work |
| RTX 3090 | `86` | Same as A100 codegen mostly |
| RTX 4090 / L40 | `89` | Ada; FP8 tensor cores |
| H100 / H200 | `90` / `90a` | Hopper; TMA, WGMMA, warp-specialization. Use `90a` for architecture-specific features. |
| B100 / B200 | `100` | Blackwell; FP4, 5th-gen tensor cores |

**Tip:** Start with `examples/00_basic_gemm` to validate your setup, then move to `examples/48_hopper_warp_specialized_gemm` for Hopper-specific kernels or `examples/55_hopper_mixed_dtype_gemm` for mixed-precision work.

### Recipe 2: Run the CUTLASS Profiler

```bash
# Build profiler
make -j$(nproc) cutlass_profiler

# Profile all GEMM configurations for a specific shape
./tools/profiler/cutlass_profiler --operation=gemm \
  --m=4096 --n=4096 --k=4096 \
  --A=f16 --B=f16 --C=f16 \
  --arch=sm_89

# Sweep multiple shapes at once
./tools/profiler/cutlass_profiler --operation=gemm \
  --m=1024,2048,4096 --n=1024,2048,4096 --k=4096 \
  --A=f16 --B=f16 --C=f16 \
  --arch=sm_89 \
  --output=results.csv
```

**Interpreting profiler output:**

The profiler emits a table with columns including:
- **Operation**: The full CUTLASS kernel configuration (tile size, stages, warp count).
- **Runtime (ms)**: Wall-clock execution time.
- **GFLOPS**: Achieved throughput. Compare against theoretical peak (e.g., H100 FP16 = 989 TFLOPS with sparsity, ~495 TFLOPS dense).
- **Bytes**: Total memory traffic. Lower is better for bandwidth-bound shapes.

**Finding optimal tile sizes from profiler output:**

1. Sort by GFLOPS descending. The top entry is the fastest kernel config for that shape.
2. Look at the operation name -- it encodes tile sizes like `cutlass_gemm_f16_128x256x64_...`. These numbers are TILE_M x TILE_N x TILE_K.
3. For small M (e.g., batch=1 inference, M=1-16), you will see skinny tiles like `64x64x64` or `64x128x64` win. For large square shapes, `128x256x64` typically dominates on Hopper.
4. Export to CSV with `--output` and analyze across shapes to find a single config that works well for your workload distribution.

**Useful profiler flags:**
- `--warmup-iterations=5 --profiling-iterations=20` -- Control measurement accuracy.
- `--operation=gemm_grouped` -- Profile grouped GEMM for MoE workloads.
- `--split-k=1,2,4` -- Test split-K parallelism for tall-skinny shapes.

### Recipe 3: Use CUTLASS from Python (CuTe DSL)

**Setup:**

```bash
# CUTLASS Python packages require CUDA 12.x and Python 3.9+
pip install nvidia-cutlass

# Or install from source for latest features
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass
pip install -e python/

# Verify installation
python -c "import cutlass; print(cutlass.__version__)"
```

**Simple GEMM example in Python CuTe:**

```python
import torch
import cutlass

# Define GEMM plan
plan = cutlass.op.Gemm(
    element_A=cutlass.DataType.f16,
    element_B=cutlass.DataType.f16,
    element_C=cutlass.DataType.f16,
    element_D=cutlass.DataType.f16,
    layout_A=cutlass.LayoutType.RowMajor,
    layout_B=cutlass.LayoutType.ColumnMajor,
    element_accumulator=cutlass.DataType.f32,
)

# Compile for a specific tile config (optional -- CUTLASS picks defaults)
plan.run(
    torch.randn(4096, 4096, dtype=torch.float16, device="cuda"),
    torch.randn(4096, 4096, dtype=torch.float16, device="cuda"),
    torch.zeros(4096, 4096, dtype=torch.float16, device="cuda"),
    torch.zeros(4096, 4096, dtype=torch.float16, device="cuda"),
)
```

**Benchmarking against cuBLAS:**

```python
import torch
import time

M, N, K = 4096, 4096, 4096
A = torch.randn(M, K, dtype=torch.float16, device="cuda")
B = torch.randn(K, N, dtype=torch.float16, device="cuda")

# Warmup
for _ in range(10):
    torch.mm(A, B)
torch.cuda.synchronize()

# cuBLAS baseline (via torch.mm)
start = time.perf_counter()
for _ in range(100):
    torch.mm(A, B)
torch.cuda.synchronize()
cublas_time = (time.perf_counter() - start) / 100

# CUTLASS GEMM
import cutlass
plan = cutlass.op.Gemm(
    element_A=cutlass.DataType.f16, element_B=cutlass.DataType.f16,
    element_C=cutlass.DataType.f16, element_D=cutlass.DataType.f16,
    layout_A=cutlass.LayoutType.RowMajor, layout_B=cutlass.LayoutType.ColumnMajor,
    element_accumulator=cutlass.DataType.f32,
)
C = torch.zeros(M, N, dtype=torch.float16, device="cuda")
D = torch.zeros(M, N, dtype=torch.float16, device="cuda")

for _ in range(10):
    plan.run(A, B, C, D)
torch.cuda.synchronize()

start = time.perf_counter()
for _ in range(100):
    plan.run(A, B, C, D)
torch.cuda.synchronize()
cutlass_time = (time.perf_counter() - start) / 100

print(f"cuBLAS: {cublas_time*1000:.2f} ms")
print(f"CUTLASS: {cutlass_time*1000:.2f} ms")
print(f"CUTLASS / cuBLAS: {cutlass_time/cublas_time:.2%}")
```

For standard dense GEMMs, expect CUTLASS and cuBLAS to be within 5% of each other. CUTLASS wins when you add custom epilogues that cuBLAS cannot fuse.

### Recipe 4: CUTLASS for Custom Epilogues

**Complete example: GEMM + bias + ReLU fused**

```cpp
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"

using namespace cute;
using namespace cutlass;

// Define the fused epilogue: D = relu(alpha * Acc + bias)
using FusionOp =
    epilogue::fusion::LinCombPerRowBiasEltAct<
        cutlass::epilogue::thread::ReLu,  // activation
        half_t,                            // element output
        float,                             // element compute (accumulator type)
        half_t                             // element bias
    >;

// Full GEMM config with fused epilogue
using GemmKernel = gemm::kernel::GemmUniversal<
    Shape<_128, _256, _64>,                // CTA tile shape (M, N, K)
    half_t,                                // Element A
    cutlass::layout::RowMajor,             // Layout A
    half_t,                                // Element B
    cutlass::layout::ColumnMajor,          // Layout B
    half_t,                                // Element C
    cutlass::layout::RowMajor,             // Layout C
    float,                                 // Accumulator
    cutlass::arch::OpClassTensorOp,        // Use tensor cores
    cutlass::arch::Sm90,                   // Target Hopper
    // ... collective mainloop and epilogue configs ...
    FusionOp                               // Fused epilogue
>;

using Gemm = gemm::device::GemmUniversalAdapter<GemmKernel>;

// Launch: pass bias pointer in epilogue params
typename Gemm::Arguments args{
    gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    {ptr_A, stride_A, ptr_B, stride_B},
    {{alpha, beta}, ptr_C, stride_C, ptr_D, stride_D, ptr_bias}
};

Gemm gemm_op;
gemm_op.initialize(args);
gemm_op.run(stream);
```

**How to compose EVT nodes:**

The EVT is a tree where leaves are data sources (accumulators, scalars, tensors) and internal nodes are operations:

1. **Start from the output** and work inward. If you want `D = relu(alpha * Acc + bias)`, your tree root is `ReLU`.
2. **Each node wraps its children.** `Sm90EVT<Op, Child1, Child2>` means `Op(Child1, Child2)`.
3. **Leaf nodes** are data sources:
   - `Sm90AccFetch` -- the GEMM accumulator
   - `ScalarBroadcast<float>` -- a scalar (alpha, beta, scale)
   - `Sm90RowBroadcast<half_t>` -- a per-row vector (bias)
   - `Sm90ColBroadcast<half_t>` -- a per-column vector
   - `Load<half_t>` -- load from a matrix (e.g., source C)
4. **Composing custom trees:**
```cpp
// Example: D = silu(alpha * Acc + row_bias) * gate_matrix
using Inner =
    Sm90EVT<SiLU,
        Sm90EVT<Add,
            Sm90EVT<Multiply,
                ScalarBroadcast<float>,   // alpha
                Sm90AccFetch              // accumulator
            >,
            Sm90RowBroadcast<half_t>     // bias
        >
    >;
using Full =
    Sm90EVT<Multiply,
        Inner,                            // silu(alpha * Acc + bias)
        Load<half_t>                      // gate matrix element-wise
    >;
```

**When to use CUTLASS epilogue vs a separate kernel:**

- **Use CUTLASS EVT when**: Your post-GEMM ops are element-wise or broadcast (bias, activation, quantization, scaling). EVT fuses these into the GEMM epilogue at zero extra memory traffic cost -- the accumulator never leaves registers before the epilogue runs.
- **Use a separate kernel when**: Your post-GEMM op requires a global reduction (e.g., layernorm, softmax across the N dimension), needs to read from other GEMMs' outputs, or involves control flow that EVT cannot express. In those cases, write the GEMM output to global memory and launch a follow-up kernel.

## When to Use CUTLASS vs Alternatives

| Scenario | Best Choice | Why |
|----------|------------|-----|
| Standard GEMM | cuBLAS | Tuned, stable, easy API -- zero configuration needed |
| GEMM + custom epilogue | CUTLASS EVT | Fused epilogue saves memory bandwidth; no separate kernel launch |
| Triton can't match perf | CUTLASS | Better tensor core utilization, explicit shared memory control, warp-specialization |
| Mixed precision W4A16 | Marlin (built on CUTLASS) | Specialized for quantized inference with fused dequant |
| Research / prototyping | Triton | Faster development cycle; Python-native; good enough for most shapes |
| Production serving kernel | CUTLASS or cuBLAS | Battle-tested, deterministic results, well-documented performance characteristics |
| Grouped GEMM (MoE) | CUTLASS | Native `GroupedGemm` kernel handles variable-size batches efficiently |
| FP8 training on Hopper | cuBLAS or Transformer Engine | Integrated scaling and amax tracking out of the box |
| FP4 inference on Blackwell | CUTLASS | First library with FP4 tensor core support |
| Custom attention variant | CUTLASS FMHA or FlashAttention | CUTLASS gives more control; FA is easier to integrate via PyTorch |
