---
id: gemm_optimization_deep_dive
kind: document
title: GEMM Optimization Deep Dive - The Most Critical GPU Kernel
category: kernel
summary: Comprehensive guide to General Matrix Multiply optimization on GPUs, covering tiling, tensor cores, memory access patterns, mixed precision, and autotuning strategies.
tags:
  - gemm
  - tensor-cores
  - tiling
  - performance
  - cutlass
  - cublas
  - triton
source_ids:
  - nvidia-cutlass-overview
  - nvidia-cute-dsl
  - triton-tutorials
gpu_families:
  - Ampere
  - Ada
  - Hopper
  - Blackwell
operators:
  - matmul
  - linear
  - gemm
precision:
  - fp32
  - tf32
  - fp16
  - bf16
  - fp8
  - int8
  - int4
  - fp4
---

# GEMM Optimization Deep Dive

## Why GEMM Dominates Deep Learning Compute

In transformer-based models, GEMM accounts for 60-80% of total compute time. Every linear layer is a GEMM:

- **QKV Projection**: `(B*S, H) x (H, 3*D)` where B=batch, S=seq_len, H=hidden_dim, D=head_dim*num_heads
- **Attention Output Projection**: `(B*S, D) x (D, H)`
- **MLP Up/Gate Projection**: `(B*S, H) x (H, I)` where I=intermediate_size (typically 4*H or 8/3*H for SwiGLU)
- **MLP Down Projection**: `(B*S, I) x (I, H)`
- **LM Head**: `(B*S, H) x (H, V)` where V=vocab_size

For a LLaMA-70B forward pass with batch=1, seq_len=2048:
- Total GEMMs per layer: 4 (QKV, O, up+gate, down)
- Total layers: 80
- Each GEMM involves matrices with hidden_dim=8192
- Total FLOPS dominated by these matrix multiplies

## The Roofline Model for GEMM

### Arithmetic Intensity Calculation

For GEMM C = A*B where A is (M,K) and B is (K,N):

```
FLOPs = 2 * M * N * K  (multiply-add = 2 ops)
Bytes = (M*K + K*N + M*N) * sizeof(element)
Arithmetic Intensity = FLOPs / Bytes = 2*M*N*K / ((M*K + K*N + M*N) * elem_size)
```

For large square GEMM (M=N=K=4096, fp16):
```
FLOPs = 2 * 4096^3 = 137.4 GFLOP
Bytes = (4096^2 * 3) * 2 = 100.7 MB
AI = 137.4G / 100.7M = 1365 FLOP/byte
```
This is compute-bound on every modern GPU.

For decode GEMM (M=1, K=4096, N=4096, fp16):
```
FLOPs = 2 * 1 * 4096 * 4096 = 33.6 MFLOP
Bytes = (1*4096 + 4096*4096 + 1*4096) * 2 = 33.6 MB
AI = 33.6M / 33.6M = 1 FLOP/byte
```
This is severely memory-bound. The weight matrix dominates - you read 32MB of weights to do 33 MFLOP.

### Ridge Points for Common GPUs

The ridge point is where compute and memory bandwidth are equally utilized:
```
Ridge Point = Peak FLOPS / Peak Bandwidth (FLOP/byte)
```

| GPU | Peak FP16 TC TFLOPS | Bandwidth GB/s | Ridge Point |
|-----|---------------------|----------------|-------------|
| RTX 3090 | 142 (sparse) | 936 | 152 |
| RTX 4090 | 330 | 1008 | 327 |
| A100 SXM | 312 | 2039 | 153 |
| H100 SXM | 990 (FP16) | 3350 | 296 |
| H100 SXM | 1979 (FP8) | 3350 | 591 |
| B200 | 2250 (FP16) | 8000 | 281 |
| B200 | 4500 (FP8) | 8000 | 563 |
| B200 | 9000 (FP4) | 8000 | 1125 |

**Implication**: For decode (AI~1), you're using <1% of compute. For large prefill GEMMs (AI>300), you're compute-bound.

## GPU GEMM Tiling Strategy

### Three-Level Tiling Hierarchy

```
Global Memory (HBM)
  |
  v  [CTA Tile: e.g., 128x256xK]
Shared Memory (SMEM)
  |
  v  [Warp Tile: e.g., 64x64xK]
Register File
  |
  v  [Thread Tile / MMA: e.g., 16x8x16]
Tensor Cores
```

#### Level 1: CTA (Cooperative Thread Array) Tile
- The tile of output matrix C computed by one thread block
- Typical sizes: 64x64, 128x128, 128x256, 256x128, 256x256
- Larger tiles = better compute/memory ratio but more shared memory needed
- Must fit in shared memory: `(TILE_M + TILE_N) * TILE_K * elem_size * num_stages`

#### Level 2: Warp Tile
- The portion of the CTA tile computed by one warp (32 threads)
- Each warp issues MMA instructions on its portion
- Typical: 64x64, 32x64, 64x32
- Number of warps per CTA: 4-16 typically

#### Level 3: MMA (Matrix Multiply-Accumulate) Instruction
- The hardware tensor core instruction
- Ampere: `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16`
- Hopper: `wgmma.mma_async.sync.aligned.shape.dtype` (warp group level)
- Blackwell: Fifth-gen tensor core MMA instructions

### Choosing Tile Sizes

Rules of thumb:
1. **CTA tile M and N** should be multiples of the warp tile
2. **Warp tile** should be multiples of the MMA instruction shape
3. **TILE_K** (reduction dimension slice) should be large enough to amortize shared memory loads
4. **Number of stages** (pipeline depth) trades shared memory for latency hiding

Common configurations for H100 (FP16):
```
TILE_M=128, TILE_N=256, TILE_K=64, STAGES=3, WARPS=8
TILE_M=256, TILE_N=128, TILE_K=64, STAGES=3, WARPS=8
TILE_M=128, TILE_N=128, TILE_K=64, STAGES=4, WARPS=4
```

## Software Pipelining (Multi-Stage)

The key optimization to hide memory latency:

### Double Buffering (2 stages)
```
Stage 0: Load tile[k+1] from global → smem buffer 1
         Compute on tile[k] from smem buffer 0
Stage 1: Load tile[k+2] from global → smem buffer 0
         Compute on tile[k+1] from smem buffer 1
```

### Triple Buffering (3 stages) - Standard on Ampere+
```
Each iteration:
  1. Issue async copy for tile[k+2] into smem buffer (k+2)%3
  2. Wait for tile[k] to arrive (should already be done)
  3. Compute MMA on tile[k]
  4. Advance k
```

### Hopper: Up to 7-8 stages with TMA
TMA (Tensor Memory Accelerator) issues bulk async copies without consuming warp cycles:
```python
# Pseudocode for Hopper GEMM pipeline
for k in range(num_k_tiles):
    # TMA issues happen in background
    tma_load(smem_A[k % STAGES], global_A[k])
    tma_load(smem_B[k % STAGES], global_B[k])
    barrier_arrive(barriers[k % STAGES])

    # Meanwhile, compute on previously loaded tiles
    barrier_wait(barriers[(k - STAGES + 1) % STAGES])
    wgmma(smem_A[(k-STAGES+1) % STAGES], smem_B[(k-STAGES+1) % STAGES], accum)
```

## Tensor Core Programming

### Ampere: mma.sync (Warp-Level)
Each warp (32 threads) collectively executes one MMA instruction:
```
mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {d0,d1,d2,d3}, {a0,a1,a2,a3}, {b0,b1}, {c0,c1,c2,c3};
```
- 16x8x16 shape: 16 rows of A, 8 columns of B, 16 reduction
- 4 FP16x2 registers for A fragment, 2 for B, 4 for accumulator
- Throughput: 256 FP16 FMA ops per warp per clock (on A100)

### Hopper: wgmma (Warp Group-Level)
Four warps (128 threads) form a warp group for wgmma:
```
wgmma.mma_async.sync.aligned.m64n256k16.f16.f16 {d}, desc_a, desc_b, ...
```
- Much larger shapes: 64x256x16 in a single instruction
- Operands come from shared memory via descriptors (not registers for B)
- Asynchronous: doesn't block the warp group
- 4x more throughput per instruction vs mma.sync

### Fragment Layouts

For mma.sync m16n8k16 with FP16:
- A matrix (16x16): distributed across 32 threads, each holds 8 FP16 values (4 registers of FP16x2)
- B matrix (16x8): each thread holds 4 FP16 values (2 registers)
- C/D matrix (16x8): each thread holds 4 FP32 values (4 registers)

Thread-to-element mapping follows a specific pattern:
```
Thread T within warp (0-31):
  Row group = T / 4
  Column = (T % 4) * 2  // for A matrix

  The exact mapping depends on the matrix and instruction variant.
  CUTLASS CuTe handles this with Layout algebra.
```

## Shared Memory Optimization for GEMM

### Bank Conflicts in GEMM
Shared memory has 32 banks with 4-byte stride. When loading tiles:

Problem: If threads in a warp access the same bank, serialization occurs.

For a 128-wide FP16 tile stored row-major:
```
Row 0: banks 0,1,2,...,63 (128 elements * 2 bytes = 256 bytes = 64 banks worth)
Row 1: banks 0,1,2,...,63
```
When threads load column-major from shared memory → bank conflicts!

### Swizzle Patterns
Solution: apply a swizzle to the shared memory layout:
```
// XOR-based swizzle
smem_addr = base + row * stride + (col ^ (row % swizzle_period)) * elem_size
```

CUTLASS swizzle modes:
- `Swizzle<3, 3, 3>`: 8-byte XOR swizzle
- `Swizzle<2, 3, 3>`: 4-byte XOR swizzle
- Identity (no swizzle) for debugging

### Padding Alternative
Simpler but wastes memory:
```
// Instead of smem[128][64], use smem[128][65]
// The +1 shifts each row by one bank
__shared__ half smem_A[128][65]; // padded
```

## Mixed Precision GEMM Patterns

### W16A16 (Standard FP16/BF16)
- A and B in FP16/BF16, accumulate in FP32
- Convert back to FP16/BF16 for output
- Supported by all tensor core generations

### W8A8 (INT8 / SmoothQuant)
```
C_fp32 = dequant(A_int8 @ B_int8)
C_fp32 = (scale_A * scale_B) * (A_int8 @ B_int8)_int32
```
- Tensor core computes INT8 GEMM with INT32 accumulation
- Scale factors applied in epilogue
- 2x throughput vs FP16 on same hardware

### W4A16 (GPTQ/AWQ style)
```
# Weights stored as INT4, activations are FP16
# During GEMM, dequantize weight on the fly
for each tile of K:
    w_fp16 = dequant(w_int4, scale, zero_point)  # in registers
    accumulator += activation_fp16 @ w_fp16       # tensor core MMA
```
Key insight: dequantization is "free" because it overlaps with memory latency.

The Marlin kernel achieves near-optimal W4A16 performance:
- 4-bit weights packed: 8 int4 values in one int32
- Asynchronous global→shared memory with cp.async
- Dequantize in registers while waiting for next tile
- Near cuBLAS FP16 throughput despite 4-bit weights
- Achieves 80-90% of theoretical memory bandwidth

### FP8 GEMM (Hopper native)
```
# E4M3 for forward, E5M2 for backward
C_fp32 = scale_A * scale_B * fp8_gemm(A_e4m3, B_e4m3)
```
- Native FP8 tensor cores on Hopper: 2x throughput vs FP16
- Per-tensor or per-token scaling
- Transformer Engine handles scale management (delayed scaling with amax history)

### FP4 GEMM (Blackwell)
```
# NV FP4 (E2M1) with block scaling
C = block_scale_A * block_scale_B * fp4_gemm(A_fp4, B_fp4)
```
- 2x throughput vs FP8 on Blackwell
- Requires block scaling (shared exponent per group of elements)
- MX format compatibility

## Epilogue Fusion

After the GEMM accumulation, the epilogue writes results. Fusing operations into the epilogue avoids extra memory round-trips:

### What Can Be Fused
1. **Bias addition**: `C = A@B + bias`
2. **Activation**: `C = gelu(A@B + bias)`
3. **Residual connection**: `C = A@B + residual`
4. **Quantization**: `C_int8 = quantize(A@B, scale)`
5. **Dequantization + bias**: `C = dequant(A_int8@B_int8) + bias`
6. **Custom chains**: `C = quantize(gelu(dequant(A@B) + bias))`

### CUTLASS EVT (Epilogue Visitor Tree)
CUTLASS 3.x provides a composable epilogue system:
```cpp
// Define epilogue as a tree of operations
using EpilogueOp = cutlass::epilogue::fusion::LinCombEltAct<
    cutlass::epilogue::thread::GELU,    // activation
    float,                               // compute type
    float,                               // scale type
    cutlass::half_t                      // output type
>;
```

### cuBLASLt Epilogue Options
```c
// Available epilogues in cublasLtMatmulDescSetAttribute:
CUBLASLT_EPILOGUE_DEFAULT      // D = alpha*A*B + beta*C
CUBLASLT_EPILOGUE_RELU         // D = relu(alpha*A*B + beta*C)
CUBLASLT_EPILOGUE_GELU         // D = gelu(alpha*A*B + beta*C)
CUBLASLT_EPILOGUE_BIAS         // D = alpha*A*B + beta*C + bias
CUBLASLT_EPILOGUE_RELU_BIAS    // D = relu(alpha*A*B + beta*C + bias)
CUBLASLT_EPILOGUE_GELU_BIAS    // D = gelu(alpha*A*B + beta*C + bias)
CUBLASLT_EPILOGUE_DGELU        // backward GELU
CUBLASLT_EPILOGUE_BGRADA       // bias gradient for A
```

## Specialized GEMM Variants

### Split-K GEMM
When M is small (decode), parallelize across K dimension:
```
# Instead of 1 CTA doing full K reduction:
# Split into split_k CTAs, each does K/split_k
# Then reduce partial results

# Good for: M=1 (decode), M=small batch
# Overhead: extra reduction kernel or atomic adds
```

### Stream-K GEMM
More balanced workload distribution:
```
total_tiles = ceil(M/TILE_M) * ceil(N/TILE_N)
total_k_iters = ceil(K/TILE_K)
total_work = total_tiles * total_k_iters

# Distribute total_work evenly across SMs
work_per_sm = total_work / num_SMs
# Each SM does a contiguous chunk of work
# May span multiple output tiles
```
Benefits: better load balancing than static tile assignment, especially for irregular shapes.

### Grouped GEMM (for MoE)
Execute multiple GEMMs with different sizes in one kernel launch:
```
# For MoE with E experts, each processing different numbers of tokens:
problems = [
    (M_expert_0, N, K),  # expert 0 gets M_0 tokens
    (M_expert_1, N, K),  # expert 1 gets M_1 tokens
    ...
    (M_expert_E, N, K),  # expert E gets M_E tokens
]
grouped_gemm(problems, A_ptrs, B_ptrs, C_ptrs)
```
CUTLASS GroupedGemm kernel handles this efficiently.

### Batched GEMM
Same shapes, different data (e.g., multi-head attention):
```
# Strided batched: regular memory layout
cublasGemmStridedBatchedEx(handle,
    transA, transB,
    M, N, K,
    alpha, A, lda, strideA,
    B, ldb, strideB,
    beta, C, ldc, strideC,
    batchCount, computeType)
```

## Autotuning GEMM

### Triton Autotuning
```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(A, B, C, M, N, K, ...):
    ...
```

### cuBLASLt Algorithm Search
```c
// Find best algorithm
cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc,
                                preference, requestedAlgoCount, heuristicResults, &returnedAlgoCount);

// Or exhaustive search
for (int i = 0; i < numAlgos; i++) {
    cublasLtMatmul(ltHandle, operationDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, D, Ddesc,
                   &algo[i], workspace, workspaceSize, stream);
    // Time each algorithm
}
```

### Key Autotuning Parameters
1. **Tile sizes** (BLOCK_M, BLOCK_N, BLOCK_K)
2. **Number of stages** (pipeline depth)
3. **Number of warps** per CTA
4. **Split-K factor** (for small M)
5. **Swizzle mode** for shared memory
6. **Cluster size** (Hopper)
7. **Persistent vs non-persistent** kernel

## GEMM Performance Targets

### What to Expect (% of Peak)

| Scenario | Shape | Expected % of Peak |
|----------|-------|-------------------|
| Large square | M=N=K=4096+ | 80-95% |
| Large rectangular | M=4096, N=11008, K=4096 | 85-95% |
| Medium batch decode | M=32, N=4096, K=4096 | 40-60% |
| Single decode | M=1, N=4096, K=4096 | 5-15% (bandwidth bound) |
| Small batch prefill | M=128, N=4096, K=4096 | 50-70% |
| MoE grouped (variable M) | varies | 30-60% |

### When cuBLAS vs CUTLASS vs Triton

| Use Case | Best Choice | Why |
|----------|------------|-----|
| Standard dense GEMM, large shapes | cuBLAS/cuBLASLt | Exhaustively tuned, fast heuristics |
| GEMM + custom epilogue | CUTLASS (EVT) | Composable epilogue fusion |
| Quantized GEMM (W4A16, W8A8) | Custom/Marlin/CUTLASS | Need dequant fusion |
| Grouped GEMM (MoE) | CUTLASS GroupedGemm | Purpose-built kernel |
| Rapid prototyping | Triton | Python-native, fast iteration |
| Blackwell-specific (FP4, new features) | CUTLASS 3.x / CuTe | First to support new hardware |
| Memory-bound decode GEMM | Custom with split-K | Need K-parallelism |

## LLM-Specific GEMM Shapes

### Common Model Dimensions

| Model | hidden_dim | intermediate | num_heads | head_dim | vocab |
|-------|-----------|-------------|-----------|----------|-------|
| LLaMA-7B | 4096 | 11008 | 32 | 128 | 32000 |
| LLaMA-13B | 5120 | 13824 | 40 | 128 | 32000 |
| LLaMA-70B | 8192 | 28672 | 64 (8 KV) | 128 | 32000 |
| Mistral-7B | 4096 | 14336 | 32 (8 KV) | 128 | 32000 |
| Mixtral-8x7B | 4096 | 14336 | 32 (8 KV) | 128 | 32000 |
| DeepSeek-V2 | 5120 | 12288 | 128 (MLA) | 128 | 102400 |
| Qwen2.5-72B | 8192 | 29568 | 64 (8 KV) | 128 | 152064 |

### Shape Analysis for Prefill vs Decode

**Prefill** (B=1, S=2048, model=LLaMA-70B):
- QKV: M=2048, K=8192, N=8192+2*1024=10240 → compute-bound
- MLP up+gate: M=2048, K=8192, N=57344 → compute-bound
- MLP down: M=2048, K=28672, N=8192 → compute-bound

**Decode** (B=1, S=1, model=LLaMA-70B):
- QKV: M=1, K=8192, N=10240 → severely memory-bound
- MLP up+gate: M=1, K=8192, N=57344 → severely memory-bound
- MLP down: M=1, K=28672, N=8192 → severely memory-bound

**Batched Decode** (B=64, S=1):
- All shapes become M=64, which helps but still memory-bound
- Need B>256 to approach compute-bound on H100

This is why decode optimization focuses on:
1. Reducing weight memory reads (quantization)
2. Increasing batch size (continuous batching)
3. Speculative decoding (process multiple tokens)
4. CUDA graphs (reduce launch overhead)
