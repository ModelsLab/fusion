# GEMM (General Matrix Multiply) Optimization on GPUs: The Definitive Guide

> GEMM is the single most important kernel in deep learning. It dominates compute in every transformer layer -- linear projections, attention QKV, MLP up/down/gate, and output heads. Mastering GEMM optimization is mastering GPU performance.

---

## Table of Contents

1. [GEMM Fundamentals on GPU](#1-gemm-fundamentals-on-gpu)
2. [Tiling Strategy](#2-tiling-strategy)
3. [Tensor Core Programming](#3-tensor-core-programming)
4. [Memory Access Optimization for GEMM](#4-memory-access-optimization-for-gemm)
5. [cuBLAS and cuBLASLt](#5-cublas-and-cublaslt)
6. [Warp-Specialized GEMM (Hopper)](#6-warp-specialized-gemm-hopper)
7. [Split-K GEMM](#7-split-k-gemm)
8. [Batched and Grouped GEMM](#8-batched-and-grouped-gemm)
9. [Mixed Precision GEMM Patterns](#9-mixed-precision-gemm-patterns)
10. [GEMM Epilogue Fusion](#10-gemm-epilogue-fusion)
11. [Autotuning GEMM](#11-autotuning-gemm)
12. [Specific GEMM Shapes in LLMs](#12-specific-gemm-shapes-in-llms)

---

## 1. GEMM Fundamentals on GPU

### 1.1 Why GEMM Dominates Deep Learning Compute

The General Matrix Multiply operation `C = alpha * A * B + beta * C` where `A` is `(M x K)`, `B` is `(K x N)`, and `C` is `(M x N)` is the foundational primitive for nearly every computation in neural networks:

| Layer Type | GEMM Operation | Typical Shape |
|---|---|---|
| Linear / Dense | `Y = XW + b` | `(batch, in_features) x (in_features, out_features)` |
| QKV Projection | `QKV = X * W_qkv` | `(batch*seq, hidden) x (hidden, 3*head_dim*num_heads)` |
| Attention Output | `O = Attn * W_o` | `(batch*seq, hidden) x (hidden, hidden)` |
| MLP Up/Gate | `Up = X * W_up` | `(batch*seq, hidden) x (hidden, intermediate)` |
| MLP Down | `Down = X * W_down` | `(batch*seq, intermediate) x (intermediate, hidden)` |
| Convolution (im2col) | Implicit GEMM | Unfolded patches x filters |
| Embedding lookup (large vocab) | `logits = hidden * W_vocab^T` | `(batch*seq, hidden) x (hidden, vocab_size)` |

In a standard transformer forward pass, GEMM operations account for **>95% of all FLOPs**. The attention softmax, layer norms, and activations are memory-bound elementwise operations that pale in comparison to the compute demands of matrix multiplication.

### 1.2 The Roofline Model: Compute-Bound vs Memory-Bound GEMM

The **roofline model** is the fundamental framework for understanding GEMM performance on GPUs. It relates achievable performance to the ratio of computation to memory movement.

#### Arithmetic Intensity

For a GEMM `C(M,N) = A(M,K) * B(K,N)`:

```
Total FLOPs = 2 * M * N * K                   (multiply-accumulate = 2 ops per element)
Total Bytes = sizeof(dtype) * (M*K + K*N + M*N)  (load A, load B, store C)

Arithmetic Intensity (AI) = (2 * M * N * K) / (sizeof(dtype) * (M*K + K*N + M*N))   [FLOP/Byte]
```

#### Ridge Point (Machine Balance)

Each GPU has a characteristic **ridge point** where the transition from memory-bound to compute-bound occurs:

```
Ridge Point = Peak Compute (FLOP/s) / Peak Memory Bandwidth (Byte/s)   [FLOP/Byte]
```

**Examples for FP16 Tensor Core operations:**

| GPU | Peak TC TFLOPS (FP16) | HBM BW (TB/s) | Ridge Point (FLOP/B) |
|---|---|---|---|
| A100 SXM | 312 | 2.0 | 156 |
| H100 SXM | 990 | 3.35 | 296 |
| H200 SXM | 990 | 4.8 | 206 |
| B200 SXM | 2,250 | 8.0 | 281 |

**Classification rule:**
- If `AI > Ridge Point`: **Compute-bound** -- limited by peak TFLOPS
- If `AI < Ridge Point`: **Memory-bound** -- limited by HBM bandwidth
- `Attainable Performance = min(Peak_Compute, AI * Peak_Bandwidth)`

#### Arithmetic Intensity for Common GEMM Shapes (FP16, 2 bytes per element)

| Shape (M x N x K) | AI (FLOP/B) | Regime on H100 |
|---|---|---|
| 1 x 4096 x 4096 | 0.99 | Extremely memory-bound (GEMV) |
| 8 x 4096 x 4096 | 7.8 | Memory-bound |
| 32 x 4096 x 4096 | 30.1 | Memory-bound |
| 128 x 4096 x 4096 | 110.7 | Memory-bound |
| 512 x 4096 x 4096 | 336.8 | Compute-bound |
| 1024 x 4096 x 4096 | 546.1 | Compute-bound |
| 4096 x 4096 x 4096 | 1365.3 | Compute-bound |
| 8192 x 8192 x 8192 | 2730.7 | Compute-bound |

**Key insight:** For an H100, GEMM becomes compute-bound roughly when `M >= 256-512` for typical hidden dimensions (4096-8192). This is exactly the boundary between LLM decode (memory-bound, M=batch_size, often 1-64) and prefill (compute-bound, M=sequence_length*batch, often 512-8192+).

### 1.3 GEMM Shape Taxonomy

#### Square GEMM (M ~ N ~ K)
- Highest arithmetic intensity and efficiency
- Both operands can be tiled and reused effectively
- Typical in training (large batch sizes)
- Example: `(4096, 4096, 4096)` -- AI = 1365 FLOP/B

#### Tall-Skinny GEMM (M >> N or M >> K)
- One dimension is very large, others are small
- Common in prefill with long sequences
- Potential load-imbalance across SMs
- Example: `(8192, 128, 4096)` -- lower AI, may need Split-K

#### Skinny GEMM / GEMV (M = 1 or very small M)
- Degenerates to matrix-vector product
- Always memory-bound: AI < 1 for M=1
- Dominates LLM decode latency
- Cannot effectively use tensor cores (insufficient parallelism along M)
- Example: `(1, 4096, 4096)` -- AI = 0.99 FLOP/B

#### Batched GEMM
- Many independent GEMMs of the same shape
- Typical in multi-head attention: `batch_size * num_heads` independent GEMMs
- Can saturate GPU even with small individual GEMM sizes
- Example: `batched(batch=32*32, M=128, N=128, K=64)`

---

## 2. Tiling Strategy

### 2.1 The Hierarchical Tiling Model

GEMM on GPU is implemented through a hierarchical decomposition that maps onto the GPU execution model:

```
Grid Level:     Problem (M x N x K)  -->  Thread Block (CTA) tiles
Block Level:    CTA tile              -->  Warp tiles
Warp Level:     Warp tile             -->  Instruction tiles (MMA)
Thread Level:   Instruction tile      -->  Per-thread accumulation
```

This hierarchy exists because of the physical memory hierarchy:

```
Global Memory (HBM)     -- TB/s bandwidth, hundreds of ns latency
     |
L2 Cache                -- ~10 TB/s bandwidth, ~200 cycles
     |
Shared Memory (SMEM)    -- ~20 TB/s bandwidth, ~30 cycles
     |
Register File (RMEM)    -- ~80 TB/s bandwidth, ~1 cycle
     |
Tensor Cores            -- peak TFLOPS
```

The goal is to **move data from slow memory to fast memory once, then reuse it as many times as possible** before discarding.

### 2.2 CTA (Thread Block) Tiles

The CTA tile defines how much of the output matrix `C` each thread block is responsible for, plus the chunk of the K dimension it processes per mainloop iteration.

**CTA tile dimensions: `(CtaTileM, CtaTileN, CtaTileK)`**

Common CTA tile sizes:

| CTA Tile (M x N) | CtaTileK | Use Case |
|---|---|---|
| 256 x 128 | 32-64 | Large GEMMs, high throughput |
| 128 x 256 | 32-64 | Large GEMMs, column-major favored |
| 128 x 128 | 32-64 | General purpose, good balance |
| 64 x 256 | 32-64 | Tall-skinny problems |
| 256 x 64 | 32-64 | Wide problems |
| 64 x 128 | 32-64 | Moderate problems |
| 64 x 64 | 32-64 | Small GEMMs, high parallelism |

**Tradeoffs:**
- **Larger tiles**: More data reuse (higher arithmetic intensity per tile), fewer tiles total, better efficiency per tile, but risk under-utilizing SMs for small problems
- **Smaller tiles**: More tiles (more parallelism), better for small GEMMs, but lower data reuse and efficiency per tile

**Data reuse analysis for a CTA tile `(TM, TN, TK)`:**
```
Data loaded per K-iteration:  TM*TK + TN*TK  bytes (A tile + B tile)
Compute per K-iteration:      TM * TN * TK * 2  FLOPs
Reuse factor:                 (TM * TN * TK * 2) / ((TM + TN) * TK * sizeof(dtype))
                            = (2 * TM * TN) / ((TM + TN) * sizeof(dtype))
```

For `TM = TN = 128, dtype = fp16 (2 bytes)`:
```
Reuse = (2 * 128 * 128) / ((128 + 128) * 2) = 32768 / 512 = 64 FLOP/B
```

For `TM = TN = 256, dtype = fp16`:
```
Reuse = (2 * 256 * 256) / ((256 + 256) * 2) = 131072 / 1024 = 128 FLOP/B
```

### 2.3 Tile Quantization and Wave Quantization

**Tile Quantization**: When matrix dimensions are not divisible by the CTA tile size, wasted computation occurs:

```
Effective tiles_M = ceil(M / CtaTileM)
Effective tiles_N = ceil(N / CtaTileN)
Total CTA tiles   = tiles_M * tiles_N

Wasted ratio = 1 - (M * N) / (tiles_M * CtaTileM * tiles_N * CtaTileN)
```

Example: M=129, N=257 with 128x128 tiles requires `2 * 3 = 6` tiles instead of `1 * 2 = 2` tiles, executing **3x** the computation for only 0.39% more data.

**Wave Quantization**: CTA tiles are scheduled in "waves" across SMs:

```
Total CTAs    = tiles_M * tiles_N
SMs available = (e.g., 108 on A100, 132 on H100)
Num waves     = ceil(Total CTAs / SMs)

Efficiency = Total CTAs / (Num waves * SMs)
```

Going from 132 CTAs to 133 CTAs on H100 doubles the number of waves (from 1 to 2), halving throughput for the marginal tile. This is why **problem dimensions should be chosen to produce CTA counts that are multiples of the SM count**.

### 2.4 Warp Tiles

Within a CTA, the work on the `(CtaTileM x CtaTileN)` output region is divided among warps:

```
Warps per CTA = (CtaTileM / WarpTileM) * (CtaTileN / WarpTileN)
```

Common warp tile configurations (for 128x128 CTA tile, 4 warps):
- `WarpTile = 64x64`: 2x2 warp grid
- `WarpTile = 32x128`: 4x1 warp grid
- `WarpTile = 128x32`: 1x4 warp grid

**Tradeoffs for warp tiles:**
- Larger warp tiles = better instruction-level parallelism (ILP) and register reuse
- Larger warp tiles = fewer warps per CTA = less latency hiding through warp scheduling
- Typical: 4-8 warps per CTA for balanced occupancy and ILP

On Hopper with warp groups (4 warps per warp group = 128 threads):
- WGMMA operates at the warp-group level
- Typical: 2-3 warp groups per CTA (1 producer + 1-2 consumers)

### 2.5 Thread Tiles

For non-tensor-core (CUDA core) GEMM, each thread computes a small sub-tile of the warp tile:

```
ThreadTileM x ThreadTileN per thread
Typical: 8x8 or 4x8 elements per thread
```

For tensor core GEMM, the "thread tile" maps to MMA instruction fragments:
- Volta/Turing: `wmma::fragment` of 16x16x16
- Ampere: `mma.sync` m16n8k16
- Hopper: `wgmma.mma_async` m64n{8-256}k16

### 2.6 Multi-Stage Pipelining (Software Pipeline)

The mainloop of a GEMM kernel iterates over the K dimension in chunks of `CtaTileK`. Each iteration requires:
1. **Load**: Copy A and B tiles from global memory to shared memory
2. **Compute**: Perform MMA operations on the loaded tiles

Without pipelining, compute stalls while waiting for memory loads. **Multi-stage pipelining** overlaps loads and compute by maintaining multiple buffers:

```
Stage 0: LOAD[0]  COMPUTE[0]
Stage 1: LOAD[1]  COMPUTE[0]  LOAD[2]
Stage 2:          COMPUTE[1]  LOAD[3]  COMPUTE[2]
...

With N stages:  N shared memory buffers
                At any time: 1 buffer being computed, N-1 being loaded
```

**Pipeline depth (number of stages):**

| Stages | SMEM Usage | Latency Hiding | Notes |
|---|---|---|---|
| 2 (double buffer) | 2x | Basic overlap | Minimum for pipelining |
| 3 (triple buffer) | 3x | Good | Common on Ampere |
| 4-7 | 4-7x | Excellent | Common on Hopper |

**SMEM budget calculation:**
```
SMEM per stage = CtaTileM * CtaTileK * sizeof(A_dtype) + CtaTileN * CtaTileK * sizeof(B_dtype)

For 128x128x64 tile, FP16, 4 stages:
  Per stage = 128*64*2 + 128*64*2 = 16384 + 16384 = 32 KB
  Total SMEM = 4 * 32 KB = 128 KB
```

On H100: 228 KB shared memory per SM, so 4 stages fits comfortably for this tile size.

**Register budget for accumulators:**
```
Accumulator registers = CtaTileM * CtaTileN / warps_per_CTA / 32 (threads per warp)

For 128x128 tile, 4 warps, FP32 accumulator:
  Per thread = 128*128 / 4 / 32 = 128 FP32 registers
```

This is a major constraint -- the register file is 65536 32-bit registers per SM on H100, limiting occupancy when accumulator registers are large.

### 2.7 How to Choose Tile Sizes

Decision factors:

1. **Problem size**: Small M/N -> smaller tiles for parallelism; large M/N -> larger tiles for efficiency
2. **GPU architecture**: Available SMEM, register count, SM count
3. **Data type**: FP16 needs less SMEM per element than FP32
4. **Tensor core shape**: Tile dimensions must be multiples of MMA instruction shape
5. **Occupancy target**: Balance SMEM usage and register pressure
6. **Wave quantization**: Choose tiles so `ceil(M/TM) * ceil(N/TN)` is a good multiple of SM count

**Rule of thumb for Hopper (H100):**
- Square/large GEMM: `128x256x64` or `256x128x64` with 4+ stages
- Medium GEMM: `128x128x64` with 3-4 stages
- Small GEMM: `64x64x64` or `64x128x64` with 2-3 stages, consider Split-K

---

## 3. Tensor Core Programming

### 3.1 Tensor Core Evolution

| Generation | Architecture | Key Instruction | Unit of Execution | Shapes |
|---|---|---|---|---|
| 1st Gen | Volta (SM70) | `wmma` | Warp (32 threads) | m16n16k16 (FP16) |
| 2nd Gen | Turing (SM75) | `mma.sync` | Warp (32 threads) | m16n8k8, m8n8k16 (INT) |
| 3rd Gen | Ampere (SM80) | `mma.sync` | Warp (32 threads) | m16n8k16 (FP16/BF16/TF32) |
| 4th Gen | Hopper (SM90) | `wgmma.mma_async` | Warp Group (128 threads) | m64n{8-256}k16 (FP16/BF16/FP8) |
| 5th Gen | Blackwell (SM100) | `tcgen05.mma` | Thread Block Cluster | m64-256, various K |

### 3.2 WMMA API (High-Level)

The WMMA (Warp Matrix Multiply-Accumulate) C++ API provides a portable, high-level interface:

```cpp
#include <mma.h>
using namespace nvcuda::wmma;

// Declare fragments
fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
fragment<accumulator, 16, 16, 16, float> c_frag;

// Initialize accumulator
fill_fragment(c_frag, 0.0f);

// Load from shared memory
load_matrix_sync(a_frag, smem_a + offset_a, lda);
load_matrix_sync(b_frag, smem_b + offset_b, ldb);

// Compute D = A * B + C
mma_sync(c_frag, a_frag, b_frag, c_frag);

// Store result
store_matrix_sync(smem_c + offset_c, c_frag, ldc, mem_row_major);
```

**Limitations of WMMA:**
- Opaque fragment layout (cannot directly access per-thread elements portably)
- Limited shape support
- Lower performance than direct PTX MMA on newer architectures
- Recommended mainly for portability

### 3.3 MMA PTX Instructions (Ampere and Earlier)

Direct PTX `mma.sync` provides finer control and higher performance:

```
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    {d0, d1, d2, d3},        // D registers (f32 accumulator, 4 per thread)
    {a0, a1, a2, a3},        // A registers (f16 elements, 4 per thread)
    {b0, b1},                // B registers (f16 elements, 2 per thread)
    {c0, c1, c2, c3};        // C registers (f32 accumulator, 4 per thread)
```

**Supported MMA shapes (Ampere SM80):**

| Input Type | Instruction Shape | A regs/thread | B regs/thread | C regs/thread |
|---|---|---|---|---|
| FP16 | m16n8k16 | 4 (8 fp16) | 2 (4 fp16) | 4 (4 fp32) |
| BF16 | m16n8k16 | 4 (8 bf16) | 2 (4 bf16) | 4 (4 fp32) |
| TF32 | m16n8k8 | 4 (4 tf32) | 2 (2 tf32) | 4 (4 fp32) |
| INT8 | m16n8k32 | 4 (16 int8) | 2 (8 int8) | 4 (4 int32) |
| INT4 | m16n8k64 | 4 (32 int4) | 2 (16 int4) | 4 (4 int32) |

### 3.4 Fragment Layout: How Data Maps to Registers

For `mma.sync.m16n8k16.f16.f16.f32`, the 32 threads in a warp each hold specific elements:

**Matrix A (16x16, row-major, FP16):**
```
Thread T (0-31) holds:
  Register a0: A[row0][col0], A[row0][col1]    (packed fp16x2)
  Register a1: A[row1][col0], A[row1][col1]
  Register a2: A[row0][col2], A[row0][col3]
  Register a3: A[row1][col2], A[row1][col3]

Where:
  row0 = (T % 4) * 2 + (T / 16) * 8           -- rows 0-15 distributed across threads
  row1 = row0 + 1
  col0,col1,col2,col3 derived from T's position in the warp
```

The layout distributes a 16x16 tile across 32 threads such that each thread holds 8 FP16 values (4 registers of fp16x2). This non-trivial mapping is why **swizzle patterns** are essential for conflict-free shared memory access.

**Matrix C/D (16x8 output, FP32):**
```
Thread T holds 4 FP32 values:
  d0 = C[row0][col0]
  d1 = C[row0][col1]
  d2 = C[row1][col0]
  d3 = C[row1][col1]
```

### 3.5 WGMMA on Hopper (SM90)

WGMMA (Warp Group Matrix Multiply-Accumulate) is the Hopper tensor core instruction that operates on a **warp group** of 128 contiguous threads (4 warps):

```
wgmma.mma_async.sync.aligned.m64n{N}k16.f32.f16.f16
```

**Key properties:**
- **M dimension**: Fixed at 64
- **N dimension**: Multiples of 8, from 8 to 256. Larger N is more efficient
- **K dimension**: 16 for 16-bit types (f16, bf16), 32 for 8-bit types (fp8, int8)
- **Operand A**: Can be in registers (RS variant) or shared memory (SS variant)
- **Operand B**: Always in shared memory (accessed via 64-bit descriptors)
- **Accumulator**: Always in registers

**WGMMA uses shared memory descriptors instead of loading data into registers:**
- A 64-bit descriptor encodes the SMEM address, layout, and swizzle pattern
- Only the descriptor (1 register) needs to be in registers, not the data
- This dramatically reduces register pressure compared to mma.sync

**Synchronization for WGMMA:**
```cpp
cute::warpgroup_arrive();           // Signal start
cute::warpgroup_commit_batch();     // Submit MMA batch
cute::warpgroup_wait<0>();          // Wait for completion
```

**Supported precision modes on Hopper:**

| Input A | Input B | Accumulator | K-dim | Notes |
|---|---|---|---|---|
| FP16 | FP16 | FP16 or FP32 | 16 | Standard half-precision |
| BF16 | BF16 | FP16 or FP32 | 16 | Common for training |
| TF32 | TF32 | FP32 | 8 | Higher precision path |
| FP8 (E4M3) | FP8 (E4M3) | FP32 | 32 | Inference-optimized |
| FP8 (E4M3) | FP8 (E5M2) | FP32 | 32 | Mixed FP8 |
| INT8 | INT8 | INT32 | 32 | Integer quantized |

### 3.6 Accumulator Precision Considerations

Tensor cores perform multiply-accumulate with limited internal precision:

- **FP16 input, FP32 accumulator**: Full precision, standard for training
- **FP16 input, FP16 accumulator**: Lower precision, risk of accumulation error for large K
- **FP8 input, FP32 accumulator**: FP8 tensor cores have **imprecise internal accumulation** -- the accumulator may lose precision over many K iterations
- **Mitigation for FP8**: Two-level accumulation (promotion) -- accumulate in TC registers for a few K-steps, then promote to CUDA core FP32 registers periodically

DeepGEMM's approach to FP8 accumulation:
```
For every chunk of K-steps:
  1. Accumulate in tensor core registers (limited precision)
  2. Promote partial sum to FP32 on CUDA cores
  3. Add to FP32 running accumulator
```

This is essential for numerical correctness with FP8 and is a key innovation in DeepSeek's DeepGEMM library, which achieves up to 1550 TFLOPS on H800 with this technique.

---

## 4. Memory Access Optimization for GEMM

### 4.1 Global Memory to Shared Memory: Coalesced and Vectorized Loads

**Coalescing requirement**: Consecutive threads should access consecutive memory addresses. For a warp of 32 threads accessing FP16 data, ideal access is 32 * 2 = 64 bytes per transaction (one 128-byte cache line with 50% utilization, or better with vectorized loads).

**Vectorized loads** maximize bandwidth utilization:

```cpp
// Scalar load: 2 bytes per thread per instruction
half val = smem[thread_offset];

// float4 vectorized load: 16 bytes per thread per instruction (8x)
float4 val = reinterpret_cast<float4*>(gmem_ptr)[thread_offset];
// Equivalent to loading 8 FP16 values at once (128 bits)
```

**Optimal access pattern for loading A tile (M x K) into SMEM:**
```
// 128 threads in a CTA, loading 128x64 tile of FP16
// Each thread loads 128*64*2 / 128 = 128 bytes = 8 float4 loads
// Arrange threads to load consecutive K-elements for coalescing

int thread_row = threadIdx.x / (K_TILE / 8);  // 8 FP16 per float4
int thread_col = threadIdx.x % (K_TILE / 8) * 8;
float4 data = *reinterpret_cast<float4*>(&A[row + thread_row][col + thread_col]);
```

### 4.2 Shared Memory Bank Conflicts and Swizzle Patterns

Shared memory is organized into **32 banks**, each 4 bytes wide. A bank conflict occurs when two threads in a warp access different addresses in the same bank simultaneously.

**The problem for GEMM:**
When threads in a warp read a column of a matrix stored row-major in shared memory, consecutive threads access addresses with a stride equal to the row length. If the row length is a multiple of 32 banks * 4 bytes = 128 bytes, every thread hits the same bank -- a **32-way bank conflict**.

**Solution 1: Padding**
```cpp
// Add padding to break bank conflict pattern
__shared__ half smem_a[TILE_M][TILE_K + PAD];  // PAD = 8 for FP16
```
Wastes shared memory but simple to implement. Used with WMMA API.

**Solution 2: Swizzling (preferred for MMA PTX)**
Swizzling permutes the storage layout so that column accesses hit different banks:

```
Logical index:  (row, col)
Physical index: (row, col XOR f(row))
```

The XOR-based swizzle function maps matrix elements to different banks without wasting memory. CUTLASS implements multiple swizzle modes:

| Swizzle Mode | Granularity | Pattern Repeat | Use Case |
|---|---|---|---|
| No swizzle | 16 bytes | N/A | Simple cases |
| Swizzle 32B | 32 bytes | 32 bytes | Small tiles |
| Swizzle 64B | 64 bytes | 64 bytes | Medium tiles |
| Swizzle 128B | 128 bytes | 128 bytes | Large tiles, best for WGMMA |

**Swizzle formula (128B mode):**
```
Within a 128-byte chunk (eight 16-byte units):
  physical_unit = logical_unit XOR (row_within_chunk % 8)
```

This ensures that consecutive rows access different 16-byte units, eliminating bank conflicts for both row and column access patterns.

### 4.3 Double Buffering and Multi-Stage Pipeline

**Double buffering** (2 stages) is the minimum for overlapping load and compute:

```
Allocate: smem_buffer[2][TILE_M][TILE_K]

Iteration 0: Load A[0] -> smem_buffer[0]
             Sync
Iteration 1: Load A[1] -> smem_buffer[1]    |  Compute on smem_buffer[0]
             Sync
Iteration 2: Load A[2] -> smem_buffer[0]    |  Compute on smem_buffer[1]
             ...
```

**Triple buffering** (3 stages) provides better latency hiding:
```
Can overlap 2 loads with 1 compute, hiding more memory latency
Requires 3x shared memory for tiles
```

**General N-stage pipeline synchronization (CUTLASS abstraction):**
```
Pipeline state: (index, phase_bit)
  index:     0..N-1 (which buffer stage)
  phase_bit: 0 or 1 (alternates when index wraps)

Producer:
  producer_acquire(state)    // Wait for empty buffer
  issue_load(smem[state.index])
  producer_commit(state)     // Signal buffer is full
  ++state                    // Advance to next stage

Consumer:
  consumer_wait(state)       // Wait for full buffer
  compute(smem[state.index])
  consumer_release(state)    // Signal buffer is empty
  ++state
```

### 4.4 cp.async for Ampere+ (SM80+)

Before Ampere, global-to-shared-memory copies required an intermediate step through registers:
```
Old path: GMEM -> Registers -> SMEM  (wastes register bandwidth)
```

`cp.async` enables direct global-to-shared memory copy:
```
New path: GMEM -> SMEM  (bypasses registers)
```

```cpp
// cp.async: copy 16 bytes from global to shared memory
asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
    :: "r"(smem_addr), "l"(gmem_addr));

// Commit the group
asm volatile("cp.async.commit_group;\n");

// Wait for completion
asm volatile("cp.async.wait_group 0;\n");
```

**Benefits for GEMM:**
- Frees registers for accumulation
- Enables true asynchronous overlap of memory copy and compute
- Foundation for multi-stage pipelines on Ampere

### 4.5 TMA (Tensor Memory Accelerator) for Hopper+

TMA is a dedicated hardware unit on Hopper that handles multi-dimensional data movement:

**Key capabilities:**
- **2D/3D bulk copies**: Understands tensor dimensions and strides
- **Automatic swizzling**: Applies swizzle patterns during copy to prevent bank conflicts
- **Automatic out-of-bounds handling**: Predication handled in hardware
- **Single-thread issuance**: Only thread 0 needs to issue the TMA, freeing other threads
- **mbarrier integration**: Completion signaled via hardware barriers

**TMA setup (host side):**
```cpp
// Create a TMA descriptor encoding the tensor layout
auto tma_desc = cute::make_tma_copy(
    cute::SM90_TMA_LOAD{},        // Copy operation
    gmem_tensor,                   // Global memory tensor
    smem_layout,                   // Target shared memory layout
    tile_shape,                    // Tile dimensions
    cluster_shape);                // Thread block cluster shape
```

**TMA execution (device side):**
```cpp
// Only thread 0 in the warp group issues the TMA
if (thread_rank == 0) {
    cute::copy(tma_desc, gmem_coord, smem_tensor);
}
// All threads synchronize via mbarrier
cute::cp_async_wait<0>();
```

**TMA swizzle modes:**
```
CU_TENSOR_MAP_SWIZZLE_NONE     -- No swizzling
CU_TENSOR_MAP_SWIZZLE_32B      -- 32-byte swizzle pattern
CU_TENSOR_MAP_SWIZZLE_64B      -- 64-byte swizzle pattern
CU_TENSOR_MAP_SWIZZLE_128B     -- 128-byte swizzle pattern (best for WGMMA)
```

**Constraints:**
- Contiguous dimension must have stride = 1
- Other strides must be multiples of 16 bytes
- Global memory must be 128-byte aligned for swizzled modes
- Shared memory alignment must match the swizzle repeat size

**Why TMA enables warp specialization:** TMA offloads all address computation and data movement to hardware. This means producer warps need minimal registers and instructions, allowing the compiler to allocate more registers to consumer (compute) warps.

### 4.6 L2 Cache Residency Control

On Ampere+ (compute capability 8.0+), the L2 cache can be partitioned into a **persisting** region that retains data across kernel invocations:

```cpp
// Set aside portion of L2 for persistent data
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, persistent_size);

// Configure access policy for a stream
cudaStreamAttrValue attr;
attr.accessPolicyWindow.base_ptr  = data_ptr;
attr.accessPolicyWindow.num_bytes = data_size;
attr.accessPolicyWindow.hitRatio  = 0.6f;  // 60% of accesses persist
attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
```

**Application to GEMM:**
- Pin one operand (e.g., weight matrix B) in L2 cache
- Stream the other operand (activation matrix A) through
- Particularly effective for decode-phase GEMM where the same weights are used across many small batches
- Speedups of 5-15% reported for memory-bound GEMM shapes

**Threadblock rasterization for L2 locality:**
Instead of linear row-major tile ordering, CUTLASS uses swizzled rasterization patterns so that nearby thread blocks access overlapping tiles of the input matrices, improving L2 cache hit rates.

---

## 5. cuBLAS and cuBLASLt

### 5.1 API Overview

**cuBLAS** provides the standard BLAS GEMM interface:
```cpp
cublasGemmEx(handle,
    CUBLAS_OP_N, CUBLAS_OP_N,   // transA, transB
    M, N, K,                     // dimensions
    &alpha,
    A, CUDA_R_16F, lda,         // A matrix
    B, CUDA_R_16F, ldb,         // B matrix
    &beta,
    C, CUDA_R_16F, ldc,         // C matrix
    CUBLAS_COMPUTE_32F,          // compute type
    CUBLAS_GEMM_DEFAULT);        // algorithm (no-op on Ampere+)
```

**cuBLASLt** provides the advanced interface with full control:
```cpp
cublasLtMatmulDesc_t matmulDesc;
cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_16F, M, K, lda);
// ... create B and C layouts

cublasLtMatmulPreference_t preference;
cublasLtMatmulPreferenceCreate(&preference);
cublasLtMatmulPreferenceSetAttribute(preference,
    CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size));

// Get algorithm heuristics
cublasLtMatmulHeuristicResult_t heuristicResult[8];
int returnedAlgoCount;
cublasLtMatmulAlgoGetHeuristic(ltHandle, matmulDesc,
    layoutA, layoutB, layoutC, layoutC,
    preference, 8, heuristicResult, &returnedAlgoCount);

// Execute with best algorithm
cublasLtMatmul(ltHandle, matmulDesc,
    &alpha, A, layoutA, B, layoutB,
    &beta, C, layoutC, C, layoutC,
    &heuristicResult[0].algo,
    workspace, workspace_size, stream);
```

### 5.2 Workspace and Algorithm Selection

**Workspace**: cuBLASLt highly recommends providing at least **32 MiB** of workspace memory on Hopper for optimal kernel selection. Without sufficient workspace, many high-performance kernels are excluded from consideration.

**Algorithm selection strategies:**

| Strategy | Approach | Quality | Time |
|---|---|---|---|
| Default heuristic | `cublasLtMatmulAlgoGetHeuristic` | Good (93% accuracy) | Microseconds |
| Top-N heuristic + profile | Get top-N candidates, benchmark each | Very good | Seconds |
| Exhaustive search | Try all valid algorithms | Optimal | Minutes-hours |

The runtime heuristic system considers:
- Matrix dimensions and data types
- Memory layout (row-major, column-major)
- Available workspace size
- GPU architecture
- Previously cached results

### 5.3 Epilogue Fusion in cuBLASLt

cuBLASLt supports fusing common post-GEMM operations into the GEMM kernel:

```cpp
// Set epilogue
cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_RELU_BIAS;
cublasLtMatmulDescSetAttribute(matmulDesc,
    CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));

// Set bias pointer
cublasLtMatmulDescSetAttribute(matmulDesc,
    CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, sizeof(bias_ptr));
```

**Supported epilogues (Hopper):**

| Epilogue | Operation | FP8 Support |
|---|---|---|
| `CUBLASLT_EPILOGUE_DEFAULT` | D = alpha*A*B + beta*C | Yes |
| `CUBLASLT_EPILOGUE_BIAS` | D = alpha*A*B + beta*C + bias | Yes |
| `CUBLASLT_EPILOGUE_RELU` | D = ReLU(alpha*A*B + beta*C) | Yes |
| `CUBLASLT_EPILOGUE_RELU_BIAS` | D = ReLU(alpha*A*B + beta*C + bias) | Yes |
| `CUBLASLT_EPILOGUE_GELU` | D = GELU(alpha*A*B + beta*C) | Yes |
| `CUBLASLT_EPILOGUE_GELU_BIAS` | D = GELU(alpha*A*B + beta*C + bias) | Yes |
| `CUBLASLT_EPILOGUE_GELU_AUX` | D = GELU(...), aux = pre-GELU values | Yes |
| `CUBLASLT_EPILOGUE_DGELU` | Backward GELU for training | No |

**FP8 specific epilogues on Hopper:** Support scaling, amax computation, and conversion to FP8 output -- critical for FP8 training workflows.

### 5.4 Performance Characteristics

**When cuBLAS excels:**
- Standard GEMM shapes (M, N, K all > 256)
- Well-aligned dimensions (multiples of 64 or 128)
- FP16/BF16/FP8 with tensor cores
- Large GEMMs where occupancy is not a concern
- Production workloads where reliability matters

**When custom kernels can beat cuBLAS:**
- Fused operations beyond cuBLASLt's epilogue set (e.g., SiLU, group-norm)
- Quantized formats not supported by cuBLAS (W4A16, W4A8, mixed-group quantization)
- Very small or irregular GEMM shapes (M < 32)
- Custom memory layouts or tiling patterns
- MoE grouped GEMM with variable sizes

**cuBLAS 12.0+ Hopper performance (H100 SXM):**
- FP16 GEMM: Up to 3x speedup over A100
- FP8 GEMM: Up to 4.8x speedup over A100 BF16
- Peak sustained: ~800+ TFLOPS for large FP16 GEMMs

### 5.5 Migration: cublasGemmEx to cublasLtMatmul

NVIDIA recommends migration from `cublasGemmEx` to `cublasLtMatmul` for Ampere+ GPUs because:
- `cublasGemmAlgo_t` algorithm selection is a no-op on Ampere+
- cuBLASLt unlocks epilogue fusion
- Better support for mixed-precision operations
- Access to the full heuristic and tuning API

---

## 6. Warp-Specialized GEMM (Hopper)

### 6.1 The Producer-Consumer Warp Pattern

On Hopper, the most performant GEMM kernels use **warp specialization**: different warp groups are assigned different roles:

```
Thread Block (CTA):
  |
  +-- Producer Warp Group (128 threads):   Data movement (TMA loads)
  |     - Issues TMA async copies GMEM -> SMEM
  |     - Minimal register usage
  |     - Manages pipeline barriers
  |
  +-- Consumer Warp Group 0 (128 threads): Computation (WGMMA)
  |     - Reads from SMEM via descriptors
  |     - Executes wgmma.mma_async
  |     - Accumulates in registers
  |     - Maximum register allocation
  |
  +-- Consumer Warp Group 1 (128 threads): Computation (WGMMA)
        - Same as Consumer 0, different output tile
        - Enables epilogue overlap
```

**Why warp specialization works on Hopper:**
1. **TMA**: Producer warps need very few registers (just TMA descriptors and barrier management). Before TMA, loading data required many registers for address computation and staging.
2. **WGMMA with SMEM descriptors**: Consumer warps access operand B via SMEM descriptors rather than loading into registers, reducing register pressure.
3. **`setmaxnreg`**: Hopper allows per-warp-group register allocation. Producer groups can be limited to ~40 registers, freeing registers for consumer groups (up to ~232 registers each).
4. **Warp scheduler**: The hardware warp scheduler can efficiently interleave producer and consumer warp groups, hiding latency of both memory and compute.

### 6.2 Cooperative vs Ping-Pong Scheduling

**Cooperative kernel**: Both consumer warp groups work on the **same** output tile:
```
Consumer WG0: Computes even K-chunks of output tile T
Consumer WG1: Computes odd K-chunks of output tile T
Both reduce into the same accumulator (requires synchronization)
```
- Advantage: Faster time-to-completion for a single tile
- Disadvantage: Reduction overhead, cannot overlap epilogue

**Ping-Pong kernel**: Consumer warp groups work on **different** output tiles:
```
Consumer WG0: Computes tile T0 (all K-iterations)
Consumer WG1: Computes tile T1 (all K-iterations)

While WG0 does epilogue(T0), WG1 does MMA(T1) -- full overlap!
While WG1 does epilogue(T1), WG0 does MMA(T2) -- full overlap!
```
- Advantage: Epilogue and MMA fully overlap, higher sustained throughput
- Disadvantage: Higher latency for individual tiles, needs 2x output tile registers
- This is the **highest performance** GEMM design on Hopper

### 6.3 Persistent Kernels

Traditional GEMM launches a grid of thread blocks where each block computes one output tile and exits. **Persistent kernels** launch exactly `num_SMs` blocks that stay alive and process multiple tiles:

```python
# Traditional
grid = (ceil(M/TM) * ceil(N/TN), 1, 1)  # One block per tile

# Persistent
grid = (num_SMs, 1, 1)                    # One block per SM, stays alive
```

Each persistent CTA uses a **tile scheduler** to determine which tile to process next:

```cpp
// Persistent kernel mainloop
while (tile_scheduler.has_work()) {
    auto tile_info = tile_scheduler.get_next_tile();

    // Producer: load tiles for this output position
    // Consumer: compute WGMMA for this output position
    // Epilogue: store results

    tile_scheduler.advance();
}
```

**Benefits of persistent kernels:**
- Eliminates kernel launch overhead for processing many tiles
- Enables sophisticated scheduling (Stream-K, grouped ordering)
- Better L2 cache control (CTAs on same SM process nearby tiles)
- Foundation for warp specialization (warps maintain state across tiles)

### 6.4 How TMA Enables Warp Specialization

Before TMA (Ampere), loading data required:
```
Per-thread: compute address, load 16 bytes, store to SMEM
128 threads doing address computation + register staging
```

With TMA (Hopper):
```
Thread 0: issue TMA descriptor (1 instruction)
TMA hardware: handles all address computation, swizzling, predication
Barrier: signals completion when all bytes arrive
```

This asymmetry in work -- 1 thread for TMA vs 128 threads for WGMMA -- is what makes warp specialization viable. The producer warp group's 128 threads mostly idle (only thread 0 works), but they serve as a "reservation" of warp-scheduling slots that the hardware uses for latency hiding.

---

## 7. Split-K GEMM

### 7.1 When to Use Split-K

Split-K is needed when the GEMM has **small M and N but large K**, causing insufficient output tiles to fill the GPU:

```
Example: M=128, N=128, K=16384 with 128x128 tiles
  Output tiles = 1x1 = 1 tile
  Only 1 SM active out of 132! (0.76% utilization)

With Split-K = 8:
  K-partitions = 8 tiles along K dimension
  Total tiles = 1 * 8 = 8
  Better SM utilization (8/132 = 6%, still not great)
```

**Use Split-K when:**
- `ceil(M/TM) * ceil(N/TN) < num_SMs` (wave quantization problem)
- K is much larger than M and N
- The GEMM is "tall-skinny" in the K dimension

### 7.2 Parallel Reduction Strategies

Each K-partition computes a **partial sum** of the output tile. These must be reduced:

**Strategy 1: Atomic reduction (simple)**
```cpp
// Each K-partition atomically adds its partial result
atomicAdd(&output[m][n], partial_result);
```
- Simple but suffers from atomic contention
- Works well for small Split-K factors

**Strategy 2: Turnstile reduction (CUTLASS)**
```
Phase 1: All K-partitions compute and store partial sums to workspace
Phase 2: One designated CTA reduces all partial sums and runs epilogue

Synchronization via global memory workspace:
  - Each CTA atomically increments a counter after storing partial sum
  - The CTA that sees counter == split_k_factor runs the reduction
```

**Strategy 3: Separate reduction kernel**
```
Kernel 1: Compute partial sums -> workspace[split_k][M][N]
Kernel 2: Reduce workspace -> output[M][N]
```
- Two kernel launches but no atomic contention
- Reduction kernel can be highly optimized

### 7.3 Stream-K Scheduling

Stream-K generalizes Split-K by assigning **fractional tiles** to CTAs:

```
Traditional tiling: Each CTA gets exactly 1 output tile, processes all K
Split-K:            Each output tile split into split_k CTAs along K
Stream-K:           Each CTA gets a fractional number of tiles (varies per CTA)
```

**How Stream-K works:**
```
Total work units = tiles_M * tiles_N * K_iterations_per_tile
CTAs             = num_SMs (persistent)

Work per CTA     = total_work_units / num_SMs (approximately)
Some CTAs handle partial tiles that span a K-boundary
```

**Stream-K advantages:**
- Eliminates wave quantization entirely (always exactly 1 wave of work)
- Maintains large complete tiles where possible (high arithmetic intensity)
- Only splits tiles at the boundaries

**Stream-K disadvantages:**
- Partial tiles require reduction (same overhead as Split-K)
- K-offset skew between CTAs reduces L2 cache hit rates
- More complex scheduling logic

**Hybrid Stream-K** (best overall):
```
Phase 1 (Stream-K):     Process 1-2 waves worth of work, handling partial tiles
Phase 2 (Data-parallel): Remaining tiles evenly divisible by SM count, standard tiling

Benefits: L2 cache efficiency of data-parallel + load balancing of Stream-K
```

**Stream-K++** (recent research, 2024): Expands from 3 to 7 scheduling policies and uses Bloom filters for efficient policy selection, achieving up to 43% improvement in select scenarios.

### 7.4 Performance Comparison

For a GEMM with M=256, N=256, K=8192 on H100 (132 SMs):

| Strategy | Tiles | Waves | SM Util | Notes |
|---|---|---|---|---|
| Standard (128x128) | 4 | 1 | 3% | Terrible |
| Split-K=8 | 32 | 1 | 24% | Better, reduction overhead |
| Split-K=32 | 128 | 1 | 97% | Good util, high reduction overhead |
| Stream-K | 132 | 1 | 100% | Optimal, minimal partial tiles |
| Split-K + smaller tiles (64x64) | 16 | 1 | 12% | Tiles too small, low efficiency |

---

## 8. Batched and Grouped GEMM

### 8.1 Batched GEMM

Batched GEMM performs B independent GEMMs of the same shape:

```
C[b] = A[b] * B[b]   for b = 0..batch_size-1
```

**Two API variants:**

**Strided batched** (uniform strides):
```cpp
cublasGemmStridedBatchedEx(handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    M, N, K,
    &alpha,
    A, CUDA_R_16F, lda, strideA,    // strideA = M*K
    B, CUDA_R_16F, ldb, strideB,    // strideB = K*N
    &beta,
    C, CUDA_R_16F, ldc, strideC,    // strideC = M*N
    batch_count,
    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
```
- All matrices same shape, uniformly spaced in memory
- Most efficient: no pointer indirection

**Pointer-array batched:**
```cpp
// Array of pointers to each matrix
half** A_array, **B_array, **C_array;  // device arrays of pointers

cublasGemmBatchedEx(handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    M, N, K, &alpha,
    (const void**)A_array, CUDA_R_16F, lda,
    (const void**)B_array, CUDA_R_16F, ldb,
    &beta,
    (void**)C_array, CUDA_R_16F, ldc,
    batch_count, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
```
- Matrices can be at arbitrary memory locations
- Slightly less efficient due to pointer loads

**Common uses:**
- Multi-head attention: `batch = batch_size * num_heads`
- LoRA: batched low-rank updates across sequence positions
- Batch inference: multiple requests processed simultaneously

### 8.2 Grouped GEMM for MoE (Mixture of Experts)

In MoE models (e.g., DeepSeek-V3, Mixtral), tokens are routed to different experts, each with their own weight matrices. This creates **variable-size** GEMMs:

```
Expert 0: C0 (M0 x N) = A0 (M0 x K) * W0 (K x N)   // M0 tokens routed here
Expert 1: C1 (M1 x N) = A1 (M1 x K) * W1 (K x N)   // M1 tokens routed here
...
Expert E: CE (ME x N) = AE (ME x K) * WE (K x N)     // ME tokens routed here

Where M0 + M1 + ... + ME = total_tokens (variable per expert)
N and K are the same for all experts (shared architecture)
```

**Naive approach**: Loop over experts, launch one GEMM per expert
```python
for i in range(num_experts):
    torch.mm(activations[i], weights[i], out=output[i])
# Problem: many small kernel launches, poor GPU utilization
```

**Grouped GEMM**: Single kernel launch processing all experts:

**cuBLAS 12.5+ Grouped GEMM API:**
```cpp
cublasLtMatmulDescCreate(&matmulDesc, ...);
// Set grouped GEMM attributes
// Supports variable M per group, fixed N and K
cublasLtGroupedGemm(ltHandle, ...);
```

**DeepGEMM approach** (DeepSeek):
- Groups only the M-axis (variable tokens per expert)
- N and K fixed (all experts share the same architecture)
- FP8 with fine-grained scaling
- ~300 lines of core kernel code
- JIT compilation for specific shapes
- Achieves up to 1550 TFLOPS on H800

**CUTLASS Grouped GEMM:**
```cpp
// Problem sizes specified as array
cutlass::gemm::GroupProblemSize problems[num_groups] = {
    {M0, N, K}, {M1, N, K}, ..., {ME, N, K}
};
```

### 8.3 Cache-Aware Grouped GEMM Scheduling

The PyTorch team's Triton persistent cache-aware grouped GEMM kernel achieves up to **2.62x speedup** over naive PyTorch loops by:

1. **Persistent kernel**: Launches exactly `num_SMs` programs (132 on H100), each processing multiple tiles
2. **Grouped launch ordering**: Instead of row-major tile traversal, groups tiles by expert for L2 cache locality
3. **Single-wave execution**: All programs complete in one GPU wave

```python
# Persistent kernel structure
grid = (NUM_SMS, 1, 1)

@triton.jit
def grouped_gemm_kernel(...):
    pid = tl.program_id(0)
    for tile_id in tl.range(pid, num_tiles, NUM_SMS):
        # Map tile_id to (expert_id, tile_m, tile_n)
        # using grouped ordering for cache locality
        expert_id, local_tile = map_tile(tile_id)
        # ... compute GEMM tile for this expert
```

### 8.4 TMA-Adaptive Grouped GEMM

Recent work (2025) eliminates padding requirements in grouped GEMM:
- Traditional approaches pad variable-M to fixed sizes, wasting compute
- TMA-adaptive approach adjusts TMA descriptors per-group
- Eliminates padding overhead entirely
- Particularly important for FP8 MoE where padding costs are amplified

---

## 9. Mixed Precision GEMM Patterns

### 9.1 W4A16: 4-bit Weights, 16-bit Activations

The most common quantization format for LLM inference. Weights are stored in INT4, activations remain in FP16/BF16.

**How it works:**
```
For each tile:
  1. Load INT4 weights from global memory (4x compression)
  2. Dequantize to FP16 in shared memory or registers:
     weight_fp16 = (weight_int4 - zero_point) * scale
  3. Perform FP16 GEMM with tensor cores
  4. Accumulate in FP32
```

**Memory bandwidth analysis:**
```
Standard FP16 GEMM: Load W(K*N*2 bytes) + X(M*K*2 bytes)
W4A16 GEMM:         Load W(K*N/2 bytes) + scales + X(M*K*2 bytes)

For decode (M=1, K=4096, N=4096):
  FP16:  4096*4096*2 + 4096*2 = 33.6 MB
  W4A16: 4096*4096/2 + scales + 4096*2 = 8.4 MB + scales
  ~4x memory reduction -> ~4x speedup for memory-bound decode
```

#### The Marlin Kernel

Marlin is the benchmark-setting W4A16 GEMM kernel, achieving near-ideal 4x speedup over FP16:

**Key design principles:**
1. **Offline weight reshuffling**: Weights and group scales are reordered before inference into layouts that enable direct dequantization into tensor core format. No runtime permutation needed.
2. **Asynchronous global loads**: Weight loads via `cp.async` with streaming eviction to avoid polluting L2 cache.
3. **Activation L2 caching**: Activations are pinned in L2 and reused across multiple warp computations.
4. **Pipeline interleaving**: Dequantization on CUDA cores overlaps with tensor core MMA instructions -- both pipelines kept saturated simultaneously.
5. **Multi-warp partial tiling**: Multiple warps compute partial results for the same output tile, maximizing warp count for latency hiding without increasing tile size.
6. **Striped SM partitioning**: Workload spans multiple column slices for better utilization on realistic matrix shapes.

**Performance (on A10 GPU):**
- Achieves ~3.87x speedup (theoretical max ~3.9x accounting for scale overhead)
- Maintains speedup from batch size 1 through 16-32 tokens
- Competitors degrade significantly at batch > 1-2

**Requirements**: CUDA >= 11.8, Compute capability >= 8.0 (Ampere+)

#### The Machete Kernel (vLLM)

Machete is Neural Magic's successor to Marlin, optimized specifically for Hopper:

**Key innovations over Marlin:**
1. **WGMMA instructions** instead of `mma.sync` -- recovers ~37% performance lost by Marlin on Hopper
2. **TMA integration**: Asynchronous multidimensional data movement
3. **Warp specialization**: Producer/consumer pattern for overlapping memory and compute
4. **Weight pre-shuffling with CuTe layout algebra**: Architecture-agnostic, easily adaptable (vs Marlin's hand-derived shuffling)
5. **Transpose computation**: Computes `Y^T = W^T * X^T` so upconverted weights stay register-resident

**Performance:**
- Single H100: Serves Llama 3.1 70B at 5 RPS, <250ms TTFT, <100ms TPOT
- 4x H100: Serves Llama 3.1 405B at 3 RPS
- At batch >= 128, approaches FP16 performance

### 9.2 W8A8: INT8 Weights and Activations

Both weights and activations quantized to INT8:

```
Y_int32 = X_int8 * W_int8           (INT8 tensor core GEMM)
Y_fp32  = Y_int32 * scale_x * scale_w  (dequantize in epilogue)
```

**Two scaling granularities:**
- **Per-tensor**: One scale for entire matrix. Simple but less accurate.
- **Per-channel/per-token**: scale_x per token row, scale_w per output channel. Higher accuracy.

**Performance characteristics:**
- INT8 tensor cores produce INT32 partial sums
- Dequantization (INT32 -> FP32) happens in the epilogue on CUDA cores
- On Blackwell (RTX 5090), the INT32->FP32 conversion consumes ~85% of execution time, making FP8 preferred on newer architectures

### 9.3 FP8 GEMM: E4M3 x E4M3

FP8 is the preferred low-precision format on Ada, Hopper, and Blackwell (all have FP8 tensor cores):

```
Y_fp32 = X_fp8_e4m3 * W_fp8_e4m3    (FP8 tensor core GEMM)
// With fine-grained scaling:
Y_fp32 = sum_blocks( X_block * W_block * scale_x_block * scale_w_block )
```

**Fine-grained scaling** (DeepSeek approach):
- Instead of one scale per tensor, use one scale per 128-element block
- Dramatically improves accuracy for FP8
- Requires two-level accumulation (tensor core + CUDA core promotion)

**DeepGEMM performance:**
- Up to 1550 TFLOPS on H800 (FP8)
- ~300 lines of core kernel code
- Supports normal GEMM and MoE grouped GEMM
- Fine-grained scaling gives 10%+ improvement by creating overlap opportunities between MMA and promotion FFMA instructions

### 9.4 W4A8: Emerging Approaches

4-bit weights with 8-bit activations (FP8):

**LiquidGEMM** (SC'25):
- Dequantizes INT4 weights on CUDA cores during the mainloop, before MMA on tensor cores
- In contrast to W8A8 where dequantization is deferred to epilogue
- Hardware-efficient design for Hopper

**Challenges:**
- No native INT4 x FP8 tensor core instruction
- Must dequantize INT4 -> FP8 or FP16 before tensor core MMA
- Dequantization overhead competes with compute

### 9.5 FP4 (Blackwell)

Blackwell introduces native FP4 tensor cores:
- FP4 multiplication allows direct dequantization on tensor cores
- Eliminates the CUDA-core dequantization bottleneck
- E2M1 format with microscaling (MXFP4)
- ~2x theoretical speedup over FP8 at same memory bandwidth

---

## 10. GEMM Epilogue Fusion

### 10.1 Why Epilogue Fusion Matters

After computing `D = A * B`, typical deep learning operations include:

```python
# Unfused: 3 separate kernels, 3 global memory round-trips
D = matmul(A, B)           # GEMM kernel: write D to HBM
D = D + bias               # Elementwise kernel: read D, write D
D = gelu(D)                # Activation kernel: read D, write D

# Fused: 1 kernel, 1 write to HBM
D = gelu(matmul(A, B) + bias)  # Everything in GEMM epilogue
```

**Memory savings:**
```
Unfused: 3 * M * N * sizeof(dtype) bytes of extra HBM traffic
Fused:   0 extra bytes (result stays in registers/SMEM during epilogue)

For M=4096, N=11008, FP16:
  Extra traffic per unfused op = 4096 * 11008 * 2 = 90 MB
  Two extra ops = 180 MB wasted bandwidth
```

### 10.2 What Can Be Fused

| Category | Operations | Complexity |
|---|---|---|
| Scalar broadcast | alpha, beta scaling | Trivial |
| Bias | Row bias (per-output-channel), column bias | Simple |
| Activation | ReLU, GeLU, SiLU/Swish, Sigmoid, Tanh | Simple |
| Residual | Add residual connection | Simple |
| Quantization | FP32->FP8 conversion + amax tracking | Moderate |
| Normalization | RMSNorm, LayerNorm (requires reduction) | Complex |
| Softmax | Row-wise softmax (requires reduction) | Complex |
| Custom | Arbitrary elementwise DAGs | Via EVT |

### 10.3 CUTLASS Epilogue Visitor Tree (EVT)

EVT enables **arbitrary epilogue fusion** through a tree-based composition of operations, available for Hopper (SM90) warp-specialized kernels:

**Tree structure:**
```
Root (Store to GMEM)
  |
  +-- Sm90Compute<ReLU>
        |
        +-- Sm90EVT (subtree)
        |     |
        |     +-- Sm90Compute<multiply>
        |     |     |
        |     |     +-- Sm90ScalarBroadcast (alpha)
        |     |     +-- Sm90AccFetch (GEMM accumulator)
        |     |
        |     +-- Sm90Compute<add>
        |           |
        |           +-- Sm90ScalarBroadcast (beta)
        |           +-- Sm90SrcFetch (source C matrix)
        |
        +-- Sm90RowBroadcast (bias vector)
```

This computes: `D = ReLU(alpha * (A*B) + beta * C + bias)`

**EVT node types:**

| Node | Purpose |
|---|---|
| `Sm90AccFetch` | Read GEMM accumulator |
| `Sm90SrcFetch` | Read source matrix C |
| `Sm90ScalarBroadcast` | Broadcast scalar value |
| `Sm90RowBroadcast` | Broadcast row vector (bias) |
| `Sm90ColBroadcast` | Broadcast column vector |
| `Sm90AuxLoad` | Load additional matrix |
| `Sm90Compute<Op>` | Elementwise operation |
| `Sm90ScalarReduction` | Reduce to scalar (e.g., amax) |
| `Sm90EVT` / `Sm90TreeVisitor` | Composite subtree node |

**Advanced: Topological visitors** for DAGs (non-tree graphs where a node has multiple consumers):
```
When the sigmoid output is used by both the loss and the gradient:
  Use topological visitor to compute sigmoid once, share result
  Avoids recomputation in tree-based evaluation
```

**Example: Binary Cross-Entropy Loss fusion:**
```
Loss = -sum[C * log(sigmoid(F)) + (1-C) * log(1-sigmoid(F))]
Where F = A*B + bias

Fused entirely in epilogue using EVT with:
  - Sm90AccFetch (GEMM result)
  - Sm90RowBroadcast (bias)
  - Sm90Compute<sigmoid>, Sm90Compute<log>
  - Sm90AuxLoad (labels C)
  - Sm90ScalarReduction (sum)
```

### 10.4 cuBLASLt Epilogue Options

More limited than EVT but zero development effort:

```cpp
// Available epilogue operations (combinable):
CUBLASLT_EPILOGUE_RELU          // ReLU activation
CUBLASLT_EPILOGUE_RELU_AUX      // ReLU + save pre-activation
CUBLASLT_EPILOGUE_BIAS          // Add bias vector
CUBLASLT_EPILOGUE_GELU          // GELU activation
CUBLASLT_EPILOGUE_GELU_AUX      // GELU + save pre-activation (for backward)
CUBLASLT_EPILOGUE_DGELU         // GELU backward
CUBLASLT_EPILOGUE_DRELU         // ReLU backward
```

**Decision matrix:**

| Need | Use |
|---|---|
| Standard bias + ReLU/GELU | cuBLASLt epilogue |
| SiLU/Swish, custom activation | CUTLASS EVT or custom kernel |
| Quantize output to FP8 | cuBLASLt (Hopper) or CUTLASS |
| RMSNorm fusion | Custom CUTLASS EVT |
| Arbitrary elementwise DAG | Custom CUTLASS EVT |

---

## 11. Autotuning GEMM

### 11.1 What Needs to Be Tuned

GEMM has many **compile-time** and **runtime** parameters:

**Compile-time (require recompilation):**
- CTA tile shape (CtaTileM, CtaTileN, CtaTileK)
- Warp tile shape
- Number of pipeline stages
- Cluster shape (Hopper)
- Instruction type (mma.sync vs wgmma)
- Epilogue type

**Runtime (can be changed per-invocation):**
- Split-K factor
- Rasterization strategy (row-major, column-major, swizzled)
- Algorithm ID (cuBLASLt)

### 11.2 cuBLASLt Heuristic Search

```cpp
// Step 1: Query heuristics for top candidates
cublasLtMatmulHeuristicResult_t results[32];
int returnedCount;
cublasLtMatmulAlgoGetHeuristic(ltHandle, matmulDesc,
    layoutA, layoutB, layoutC, layoutD,
    preference, 32, results, &returnedCount);

// Step 2: Profile top candidates
float best_time = FLT_MAX;
int best_algo = 0;
for (int i = 0; i < returnedCount; i++) {
    float time = benchmark(results[i].algo);
    if (time < best_time) {
        best_time = time;
        best_algo = i;
    }
}

// Step 3: Use best algorithm
cublasLtMatmul(ltHandle, ..., &results[best_algo].algo, ...);
```

The heuristic system achieves **93% accuracy** across a large problem space, meaning the top-1 recommendation is within 93% of optimal performance in most cases.

### 11.3 CUTLASS Profiler

CUTLASS provides a built-in profiler for autotuning:

```bash
# Profile all FP16 GEMM kernels for a specific shape
./tools/profiler/cutlass_profiler \
    --operation=gemm \
    --m=4096 --n=4096 --k=4096 \
    --A=f16:row --B=f16:column --C=f32 \
    --warmup-iterations=5 \
    --profiling-iterations=20
```

The profiler enumerates valid kernel configurations and benchmarks each one.

### 11.4 nvMatmulHeuristics (CUTLASS 4.2+)

NVIDIA's new heuristic system dramatically reduces tuning time:

**Traditional workflow:**
```
1. Generate thousands of kernel configurations
2. Compile ALL of them (hours)
3. Benchmark ALL of them (more hours)
4. Select best
Total: 700+ minutes for a set of GEMM shapes
```

**nvMatmulHeuristics workflow:**
```
1. Heuristic predicts top-16 configurations (microseconds)
2. Compile ONLY those 16 (minutes)
3. Benchmark 16 candidates
4. Select best
Total: ~150 minutes, achieving 96% of exhaustive search performance
```

**Performance results:**
- H100, Llama 3 405B training shapes: 96% peak in ~150 min (4.7x faster than exhaustive)
- B200, DeepSeek-R1 671B: 99% peak, >5x faster
- Supports all tensor core precisions: FP4, FP8, FP16/BF16, TF32, INT8

### 11.5 Triton Autotuning

Triton uses a Python-level autotuning mechanism:

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(A, B, C, M, N, K, ...):
    ...
```

**Triton autotuning limitations:**
- Must compile and benchmark every candidate (50-80 us per heuristic call vs 10-50 seconds for Triton's full autotuning of 75 configs)
- No analytical model -- pure empirical
- Retuning needed per GPU and shape

**tritonBLAS** (recent research): Analytical tile selection in 50-80 microseconds, independent of GEMM size -- orders of magnitude faster than empirical autotuning while achieving competitive performance.

---

## 12. Specific GEMM Shapes in LLMs

### 12.1 Transformer Layer GEMM Breakdown

For a typical LLM (e.g., Llama-3-8B: hidden=4096, intermediate=11008, num_heads=32, head_dim=128, num_kv_heads=8):

#### QKV Projection
```
Input:  X (batch*seq, 4096)
Weight: W_q (4096, 4096) + W_k (4096, 1024) + W_v (4096, 1024)
        Often fused: W_qkv (4096, 6144) or separate

GEMM shape: (M, N, K) = (batch*seq, 6144, 4096)
  Prefill (seq=2048, batch=1): M=2048, N=6144, K=4096 -> AI=699 FLOP/B [compute-bound]
  Decode  (seq=1, batch=1):    M=1,    N=6144, K=4096 -> AI=1.0 FLOP/B [memory-bound]
  Decode  (seq=1, batch=32):   M=32,   N=6144, K=4096 -> AI=31 FLOP/B  [memory-bound on H100]
```

#### Output Projection
```
Input:  O (batch*seq, 4096)
Weight: W_o (4096, 4096)

GEMM shape: (batch*seq, 4096, 4096)
  Similar compute characteristics to QKV but smaller N
```

#### MLP Up + Gate Projection (SwiGLU)
```
Input:  X (batch*seq, 4096)
Weight: W_up (4096, 11008) + W_gate (4096, 11008)
        Often fused: W_upgate (4096, 22016)

GEMM shape: (batch*seq, 22016, 4096)
  Prefill: M=2048 -> AI=754 FLOP/B [compute-bound]
  Decode:  M=1    -> AI=1.0 FLOP/B [memory-bound]
```

#### MLP Down Projection
```
Input:  X (batch*seq, 11008)
Weight: W_down (11008, 4096)

GEMM shape: (batch*seq, 4096, 11008)
  Prefill: M=2048 -> AI=546 FLOP/B [compute-bound]
  Decode:  M=1    -> AI=0.5 FLOP/B [extremely memory-bound]
```

### 12.2 Prefill vs Decode: The Two Regimes

| Property | Prefill | Decode |
|---|---|---|
| M dimension | batch * seq_len (hundreds to thousands) | batch_size (1 to ~64) |
| Compute regime | Compute-bound | Memory-bound |
| Arithmetic intensity | 55-100+ FLOP/B | 1-10 FLOP/B |
| Limiting factor | Peak TFLOPS | HBM bandwidth |
| SM utilization | High (>80%) | Low to moderate |
| Bandwidth utilization | Moderate | High (>80%) |
| Optimal kernel | Large tiles, high occupancy GEMM | Skinny GEMM, GEMV, Split-K |
| Key optimization | Maximize tensor core utilization | Minimize memory loads (quantization) |

**Empirical findings (2025 research across Llama-3-8B, Qwen2.5-32B):**
- Prefill GEMM kernels (QKV, FFN): AI = 55-100 FLOP/B, stalled primarily by compute
- Decode GEMM kernels: AI = 1-10 FLOP/B, stalled primarily by memory
- Prefill attention (FlashAttention): AI up to 319 (long context on Llama-3-8B), 382 (Qwen2.5-32B)
- Decode FFN transitions from ~95 FLOP/B (compute-bound) to ~8 FLOP/B (memory-bound) between phases

### 12.3 Decode GEMM: The M=1 Problem

When M=1 (single token decode, batch=1), GEMM degenerates to GEMV:

```
y(1, N) = x(1, K) * W(K, N)

This is a matrix-vector product:
  - AI = 2*N*K / (2*(N*K + K + N)) ~ 1.0 for large N, K
  - Always memory-bound regardless of GPU
  - Tensor cores cannot be efficiently utilized (M too small)
```

**Optimization strategies for decode GEMM:**

1. **Batching**: Increase M by batching multiple requests
   ```
   M=1: AI=1.0 (memory-bound)
   M=8: AI=7.8 (still memory-bound but better)
   M=64: AI=54 (approaching compute-bound on H100)
   ```

2. **Weight quantization**: Reduce memory footprint
   ```
   FP16 weights: Load K*N*2 bytes
   INT4 weights: Load K*N/2 bytes -> 4x less memory traffic
   Since decode is memory-bound, 4x less traffic -> ~4x speedup
   ```

3. **GEMV-specific kernels**: Use GEMV_SPLITK algorithm
   - Iterates over K-dimension computing dot product
   - Uses tl.dot alternatives optimized for skinny shapes
   - Avoids tensor core overhead for M=1

4. **GEMM-SPLITK for small M (2-64)**:
   - "Skinny" matrix regime
   - SplitK enables better SM utilization
   - FlashDecoding++ uses partial-softmax with zero-padding reduction

5. **Speculative decoding**: Generate multiple candidate tokens, verify in batch
   - Converts M=1 to M=k (where k is number of candidates)
   - Improves arithmetic intensity

### 12.4 Shape Table for Common Models

| Model | Hidden | Intermediate | Heads | KV Heads | QKV GEMM (M, N, K) | MLP Up (M, N, K) |
|---|---|---|---|---|---|---|
| Llama-3-8B | 4096 | 14336 | 32 | 8 | (M, 6144, 4096) | (M, 28672, 4096) |
| Llama-3-70B | 8192 | 28672 | 64 | 8 | (M, 10240, 8192) | (M, 57344, 8192) |
| Qwen2.5-32B | 5120 | 27648 | 40 | 8 | (M, 6400, 5120) | (M, 55296, 5120) |
| DeepSeek-V3 MoE | 7168 | 2048* | 128 | 128 | (M, 18432, 7168) | (M, 4096, 7168)* |
| Mixtral-8x7B | 4096 | 14336 | 32 | 8 | (M, 6144, 4096) | (M, 28672, 4096)* |

*MoE models: intermediate dimension is per-expert; total compute distributed across active experts

**Where M =**
- **Training**: `batch_size * sequence_length` (typically 2048-8192+)
- **Prefill**: `sequence_length` (512-128000)
- **Decode**: `batch_size` (1-256, typically 1-64)

### 12.5 The Crossover Point

The batch size at which decode transitions from memory-bound to compute-bound varies by GPU:

```
Crossover batch size ~ Ridge Point * sizeof(dtype) * (N + K) / (2 * N * K)

For H100 (ridge=296), FP16, N=K=4096:
  Crossover M ~ 296 * 2 * 8192 / (2 * 4096 * 4096) ~ 145

For H100, FP16, N=4096, K=14336 (MLP up):
  Crossover M ~ 296 * 2 * 18432 / (2 * 4096 * 14336) ~ 93
```

Below this batch size, **quantization** (reducing memory traffic) helps more than faster compute. Above it, higher TFLOPS and better tensor core utilization matter more.

---

## Performance Summary: GEMM Optimization Checklist

### Architecture-Independent

- [ ] Choose tile sizes to minimize wave quantization: `ceil(M/TM) * ceil(N/TN) % num_SMs == 0`
- [ ] Align matrix dimensions to multiples of 128 bytes (64 for FP16, 128 for INT8)
- [ ] Use vectorized loads (float4 / uint4) for global memory access
- [ ] Ensure coalesced global memory access patterns
- [ ] Multi-stage pipeline (minimum 2 stages, prefer 3-4+)
- [ ] Epilogue fusion for bias, activation, quantization
- [ ] Use Split-K or Stream-K for small-M large-K shapes
- [ ] Profile: compare cuBLAS baseline before writing custom kernels

### Ampere-Specific (SM80)

- [ ] Use `cp.async` for global-to-shared memory copies
- [ ] Use `mma.sync.m16n8k16` for FP16/BF16 tensor cores
- [ ] Shared memory swizzle (padding or XOR) to eliminate bank conflicts
- [ ] Double or triple buffer shared memory tiles
- [ ] L2 cache residency control (`cuAccessPolicyWindow`) for weight matrices

### Hopper-Specific (SM90)

- [ ] Use TMA for all global-to-shared memory transfers
- [ ] Use WGMMA (`wgmma.mma_async`) instead of `mma.sync`
- [ ] Warp specialization: producer warp group (TMA) + consumer warp groups (WGMMA)
- [ ] 128B swizzle mode for SMEM layouts
- [ ] `setmaxnreg` for asymmetric register allocation
- [ ] Ping-pong kernel for epilogue/MMA overlap
- [ ] Persistent kernel with tile scheduler
- [ ] Thread block clusters for cross-SM shared memory access
- [ ] FP8 with two-level accumulation for precision

### Blackwell-Specific (SM100)

- [ ] Use `tcgen05.mma` for 5th-gen tensor cores
- [ ] FP4 (MXFP4) for maximum throughput
- [ ] Tensor Memory (TMEM) for register-like persistent storage
- [ ] Thread block cluster GEMM for multi-SM tiles

---

## Sources

- [NVIDIA Matrix Multiplication Performance Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)
- [CUTLASS Efficient GEMM Documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)
- [Colfax: CUTLASS Tutorial - Efficient GEMM Kernel Designs with Pipelining](https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/)
- [Colfax: WGMMA on Hopper GPUs](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/)
- [Colfax: Mastering the TMA](https://research.colfax-intl.com/tutorial-hopper-tma/)
- [Colfax: Persistent Kernels and Stream-K](https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/)
- [Colfax: Epilogue Fusion with EVT](https://research.colfax-intl.com/epilogue_visitor_tree/)
- [Colfax: GEMM with Thread Block Clusters on Blackwell](https://research.colfax-intl.com/cutlass-tutorial-gemm-with-thread-block-clusters-on-nvidia-blackwell-gpus/)
- [PyTorch: Deep Dive on CUTLASS Ping-Pong GEMM Kernel](https://pytorch.org/blog/cutlass-ping-pong-gemm-kernel/)
- [PyTorch: Accelerating MoEs with Triton Persistent Grouped GEMM](https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/)
- [NVIDIA: cuBLAS 12.0 Features on Hopper](https://developer.nvidia.com/blog/new-cublas-12-0-features-and-matrix-multiplication-performance-on-nvidia-hopper-gpus/)
- [NVIDIA: Introducing Grouped GEMM APIs in cuBLAS](https://developer.nvidia.com/blog/introducing-grouped-gemm-apis-in-cublas-and-more-performance-updates/)
- [NVIDIA: Improving GEMM Auto-Tuning with nvMatmulHeuristics and CUTLASS 4.2](https://developer.nvidia.com/blog/improving-gemm-kernel-auto-tuning-efficiency-on-nvidia-gpus-with-heuristics-and-cutlass-4-2)
- [NVIDIA: cuBLAS 13.1 Documentation](https://docs.nvidia.com/cuda/cublas/)
- [NVIDIA: CUTLASS 3.x Abstractions for GEMM](https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design)
- [NVIDIA: CUDA L2 Cache Control](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html)
- [Marlin: FP16xINT4 LLM Inference Kernel (GitHub)](https://github.com/IST-DASLab/marlin)
- [Introducing Machete: Mixed-Input GEMM for Hopper (Red Hat)](https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel)
- [DeepGEMM: Clean and Efficient FP8 GEMM (DeepSeek)](https://github.com/deepseek-ai/DeepGEMM)
- [Stream-K++: Adaptive GPU GEMM Scheduling](https://arxiv.org/abs/2408.11417)
- [LiquidGEMM: W4A8 GEMM Kernel](https://arxiv.org/html/2509.01229)
- [A Systematic Characterization of LLM Inference on GPUs](https://arxiv.org/html/2512.01644v1)
- [Siboehm: How to Optimize a CUDA Matmul for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM)
- [Hamza: Optimising GEMM on H100](https://hamzaelshafie.bearblog.dev/worklog-optimising-gemm-on-nvidia-h100-for-cublas-like-performance-wip/)
- [Persistent GEMM in CuTeDSL on Hopper](https://veitner.bearblog.dev/persistent-gemm-in-cutedsl-on-hopper/)
- [Swizzles in CuTeDSL Kernels](https://veitner.bearblog.dev/swizzles-and-their-usage-in-cutedsl-kernels/)
- [Aleksa Gordic: Anatomy of High Performance Matmul Kernels](https://www.aleksagordic.com/blog/matmul)
- [CUDA HGEMM Optimization (Bruce-Lee-LY GitHub)](https://github.com/Bruce-Lee-LY/cuda_hgemm)
- [PyTorch: Accelerating LLM Inference with GemLite](https://pytorch.org/blog/accelerating-llm-inference/)
- [tritonBLAS: Analytical GEMM Parameter Selection](https://arxiv.org/html/2512.04226v1)
- [NVIDIA Tensor Core Evolution: Volta to Blackwell (SemiAnalysis)](https://newsletter.semianalysis.com/p/nvidia-tensor-core-evolution-from-volta-to-blackwell)
- [vLLM INT4 W4A16 Documentation](https://docs.vllm.ai/en/latest/features/quantization/int4/)
- [TMA-Adaptive FP8 Grouped GEMM](https://arxiv.org/html/2508.16584v1)
