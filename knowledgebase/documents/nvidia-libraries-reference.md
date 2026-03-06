---
id: nvidia_libraries_reference
kind: document
title: NVIDIA CUDA Libraries - Complete Reference
category: libraries
summary: Comprehensive reference for cuBLAS, cuBLASLt, cuDNN, NCCL, Transformer Engine, cuSPARSELt, CUB/Thrust, and when to use each library vs custom kernels.
tags:
  - cublas
  - cudnn
  - nccl
  - transformer-engine
  - cusparselt
  - cub
  - thrust
source_ids:
  - nvidia-transformer-engine
gpu_families:
  - Ampere
  - Ada
  - Hopper
  - Blackwell
---

# NVIDIA CUDA Libraries - Complete Reference

## cuBLAS and cuBLASLt

### cuBLAS (Basic)
```c
// Standard GEMM: C = alpha * A * B + beta * C
cublasGemmEx(handle,
    CUBLAS_OP_N, CUBLAS_OP_N,   // transA, transB
    M, N, K,                     // dimensions
    &alpha,                      // scalar
    A, CUDA_R_16F, lda,         // A matrix
    B, CUDA_R_16F, ldb,         // B matrix
    &beta,
    C, CUDA_R_16F, ldc,         // C matrix
    CUBLAS_COMPUTE_32F,          // compute type
    CUBLAS_GEMM_DEFAULT          // algorithm
);

// Strided Batched GEMM (for multi-head attention, batch processing):
cublasGemmStridedBatchedEx(handle,
    transa, transb,
    M, N, K,
    &alpha,
    A, CUDA_R_16F, lda, strideA,
    B, CUDA_R_16F, ldb, strideB,
    &beta,
    C, CUDA_R_16F, ldc, strideC,
    batchCount,
    CUBLAS_COMPUTE_32F,
    CUBLAS_GEMM_DEFAULT
);
```

### cuBLASLt (Advanced)

cuBLASLt provides more control: workspace, algorithm selection, and **epilogue fusion**.

```c
// Setup
cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16F, M, K, lda);
cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16F, K, N, ldb);
cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, M, N, ldc);

// Set epilogue (fused operations after GEMM):
cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_GELU_BIAS;
cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE,
                               &epilogue, sizeof(epilogue));
// Set bias pointer:
cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                               &bias_ptr, sizeof(bias_ptr));

// Algorithm selection (heuristic):
cublasLtMatmulPreferenceCreate(&preference);
cublasLtMatmulPreferenceSetAttribute(preference,
    CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize));

int returnedResults;
cublasLtMatmulHeuristicResult_t heuristicResult[8];
cublasLtMatmulAlgoGetHeuristic(handle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc,
                                preference, 8, heuristicResult, &returnedResults);

// Execute with best algorithm:
cublasLtMatmul(handle, operationDesc,
    &alpha, A, Adesc, B, Bdesc, &beta, C, Cdesc, D, Ddesc,
    &heuristicResult[0].algo, workspace, workspaceSize, stream);
```

### cuBLASLt Epilogue Options
```
CUBLASLT_EPILOGUE_DEFAULT          // D = alpha*A*B + beta*C
CUBLASLT_EPILOGUE_RELU             // D = relu(alpha*A*B + beta*C)
CUBLASLT_EPILOGUE_RELU_BIAS        // D = relu(alpha*A*B + beta*C + bias)
CUBLASLT_EPILOGUE_BIAS             // D = alpha*A*B + beta*C + bias
CUBLASLT_EPILOGUE_GELU             // D = gelu(alpha*A*B + beta*C)
CUBLASLT_EPILOGUE_GELU_BIAS        // D = gelu(alpha*A*B + beta*C + bias)
CUBLASLT_EPILOGUE_GELU_AUX         // gelu with auxiliary output (for backward)
CUBLASLT_EPILOGUE_GELU_AUX_BIAS    // gelu+bias with auxiliary output
CUBLASLT_EPILOGUE_DGELU            // derivative of gelu (backward pass)
CUBLASLT_EPILOGUE_DGELU_BGRAD      // dgelu + bias gradient
CUBLASLT_EPILOGUE_BGRADA           // bias gradient for A
CUBLASLT_EPILOGUE_BGRADB           // bias gradient for B
```

### FP8 GEMM with cuBLAS
```c
// Hopper FP8 GEMM:
cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

// Set FP8 types:
cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8F_E4M3, M, K, lda);
cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8F_E4M3, K, N, ldb);

// Scale factors (per-tensor):
cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                               &scale_A, sizeof(scale_A));
cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                               &scale_B, sizeof(scale_B));
cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER,
                               &scale_D, sizeof(scale_D));

// Result: C = scale_A * scale_B * (A_fp8 @ B_fp8), accumulated in FP32
```

## cuDNN

### Graph API (Modern)
```c
// Build operation graph:
cudnnBackendDescriptor_t graph;
cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &graph);

// Add operations to graph (matmul, bias, activation, etc.)
// cuDNN finds optimal fusion plan

// Execute:
cudnnBackendExecute(handle, executionPlan, variantPack);
```

### cuDNN Flash Attention
```python
# Via PyTorch SDPA:
import torch.nn.functional as F

# cuDNN backend is automatically selected when available:
with torch.backends.cuda.sdp_kernel(
    enable_flash=True,      # FlashAttention (Triton-based)
    enable_math=False,       # Naive implementation
    enable_mem_efficient=True,# xFormers-style
    enable_cudnn=True,       # cuDNN Flash Attention (Hopper)
):
    output = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

# cuDNN Flash Attention is often fastest on Hopper
# Supports: FP16, BF16, FP8 (Hopper), causal/non-causal
```

## NCCL

### Collective Operations
```python
import torch.distributed as dist

# AllReduce: sum across all GPUs
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

# AllGather: gather full tensor from all GPUs
output_list = [torch.empty_like(tensor) for _ in range(world_size)]
dist.all_gather(output_list, tensor)

# ReduceScatter: reduce then scatter
dist.reduce_scatter(output, input_list, op=dist.ReduceOp.SUM)

# AllToAll: for expert parallelism in MoE
dist.all_to_all(output_split, input_split)

# Broadcast: from one GPU to all
dist.broadcast(tensor, src=0)
```

### NCCL Transport Types
```
P2P (NVLink): Direct GPU-to-GPU, lowest latency
  - NVLink 4.0 (H100): 900 GB/s bidirectional per GPU
  - NVLink 5.0 (B200): 1800 GB/s bidirectional per GPU

SHM (Shared Memory): Through CPU shared memory
  - Useful when NVLink not available
  - Limited by PCIe bandwidth

NET (Network): InfiniBand/RoCE
  - For multi-node communication
  - InfiniBand HDR: 200 Gb/s per port
  - InfiniBand NDR: 400 Gb/s per port
```

### NCCL Tuning Environment Variables
```bash
# Algorithm selection:
NCCL_ALGO=Ring          # Ring algorithm (good for small messages)
NCCL_ALGO=Tree          # Tree algorithm (good for large messages)
NCCL_ALGO=CollnetDirect # SHARP acceleration (InfiniBand)

# Protocol:
NCCL_PROTO=Simple       # Simple protocol
NCCL_PROTO=LL           # Low-Latency protocol (small messages)
NCCL_PROTO=LL128        # Low-Latency 128-byte protocol

# Channel count:
NCCL_MIN_NCHANNELS=4    # Minimum channels (more = more bandwidth)
NCCL_MAX_NCHANNELS=32   # Maximum channels

# NVLink SHARP (Hopper+):
NCCL_NVLS_ENABLE=1      # Enable NVLink SHARP for faster collectives

# Debug:
NCCL_DEBUG=INFO          # Print NCCL debug information
NCCL_DEBUG_SUBSYS=ALL    # All subsystems
```

## Transformer Engine

### Overview
Automatic mixed-precision with FP8 for transformer models.

```python
import transformer_engine.pytorch as te

# Drop-in replacement for nn.Linear:
linear = te.Linear(in_features, out_features, bias=True)

# FP8 context manager:
with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
    output = linear(input)  # Automatically uses FP8 GEMM

# Recipe configuration:
from transformer_engine.common.recipe import DelayedScaling, Format

recipe = DelayedScaling(
    margin=0,              # Scale factor margin
    fp8_format=Format.HYBRID,  # E4M3 for forward, E5M2 for backward
    amax_history_len=1024,     # History window for amax tracking
    amax_compute_algo="max",   # How to compute amax from history
)
```

### Delayed Scaling Algorithm
```python
# Each tensor maintains:
# - amax_history: deque of recent maximum absolute values
# - scale: current quantization scale
# - scale_inv: inverse scale (for dequantization)

# Per forward pass:
current_amax = tensor.abs().amax()
amax_history.append(current_amax)

# Scale is computed from PREVIOUS iteration's amax (delayed by 1):
prev_amax = max(amax_history)  # max of history window
scale = fp8_max / prev_amax     # fp8_max = 448 for E4M3

# Why delayed? Avoids GPU sync point in current iteration
# Assumption: amax doesn't change drastically between steps
```

## cuSPARSELt (Structured Sparsity)

```c
// 2:4 structured sparse GEMM:
cusparseLtInit(&handle);

// Create sparse matrix descriptor:
cusparseLtStructuredDescriptorInit(&matA, rows_A, cols_A, ld_A,
    alignment, CUDA_R_16F, CUSPARSE_ORDER_ROW,
    CUSPARSELT_SPARSITY_50_PERCENT);  // 2:4 sparsity

// Create matmul descriptor:
cusparseLtMatmulDescriptorInit(&matmulDesc, ...);

// Prune the matrix to 2:4 pattern:
cusparseLtSpMMAPrune(&handle, &matmulDesc, A_dense, A_pruned, ...);

// Compress pruned matrix:
cusparseLtSpMMACompress(&handle, &matmulDesc, A_pruned, A_compressed, ...);

// Execute sparse GEMM (2x throughput):
cusparseLtMatmul(&handle, &matmulDesc, &alpha,
    A_compressed, B_dense, &beta, C, D, workspace, ...);
```

## CUB (CUDA Unbound)

### Device-Level Primitives
```cpp
#include <cub/cub.cuh>

// Sort (for top-k sampling):
cub::DeviceRadixSort::SortPairsDescending(
    temp_storage, temp_bytes,
    keys_in, keys_out,    // sort by these (logits)
    vals_in, vals_out,    // carry these (token indices)
    num_items);

// Reduce (for sum, max):
cub::DeviceReduce::Sum(temp, temp_bytes, input, output, n);
cub::DeviceReduce::Max(temp, temp_bytes, input, output, n);

// Scan (prefix sum):
cub::DeviceScan::InclusiveSum(temp, temp_bytes, input, output, n);

// Select (filtering):
cub::DeviceSelect::If(temp, temp_bytes, input, output, num_selected, predicate, n);

// Histogram:
cub::DeviceHistogram::HistogramEven(temp, temp_bytes, samples, histogram, levels, n);
```

### Block-Level Primitives
```cpp
// Inside a CUDA kernel:
__global__ void my_kernel(float* input, float* output) {
    // Block-level reduce:
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    float val = input[threadIdx.x + blockIdx.x * blockDim.x];
    float block_sum = BlockReduce(temp).Sum(val);

    // Block-level sort:
    typedef cub::BlockRadixSort<float, BLOCK_SIZE, ITEMS_PER_THREAD> BlockSort;
    __shared__ typename BlockSort::TempStorage sort_temp;
    float items[ITEMS_PER_THREAD];
    // ... load items
    BlockSort(sort_temp).Sort(items);  // in-place sort
}
```

## Library Selection Decision Tree

```
What operation do you need?

GEMM (Matrix Multiply):
├── Standard dense GEMM → cuBLAS (cublasGemmEx) or cuBLASLt
├── GEMM + bias + activation → cuBLASLt with epilogue
├── Custom epilogue chain → CUTLASS EVT
├── Quantized GEMM (W4A16) → Custom kernel (Marlin) or CUTLASS
├── FP8 GEMM → cuBLAS FP8 or CUTLASS
├── Sparse GEMM (2:4) → cuSPARSELt
├── Grouped GEMM (MoE) → CUTLASS GroupedGemm
├── Rapid prototyping → Triton tl.dot
└── Blackwell FP4 → CUTLASS 3.x

Attention:
├── Standard training/inference → FlashAttention-2/3
├── Serving with paged KV → FlashInfer or PagedAttention
├── Custom attention pattern → FlexAttention (torch.nn.attention)
├── cuDNN Flash Attention → via PyTorch SDPA (Hopper)
└── Blackwell attention → CUTLASS FMHA / cuTile

Normalization (RMSNorm, LayerNorm):
├── Standard → Triton fused kernel (fastest to write)
├── With residual fusion → Custom Triton or CUDA
├── cuDNN fused norm → via cuDNN graph API
└── Apex fused norm → apex.normalization

Collective Communication:
├── Standard (AllReduce, etc.) → NCCL
├── Custom patterns → NCCL + manual overlap
└── NVLink SHARP (Hopper+) → NCCL with NVLS

Sort/Scan/Reduce:
├── Device-level → CUB
├── Block-level (inside kernel) → CUB block primitives
└── High-level → Thrust
```
