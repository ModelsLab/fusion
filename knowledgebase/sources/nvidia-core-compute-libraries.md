# NVIDIA Core Compute Libraries for GPU Kernel Optimization

> Comprehensive reference covering cuBLAS, cuDNN, NCCL, Transformer Engine, cuSPARSE/cuSPARSELt, cuFFT, CUTLASS, Thrust/CUB, cuRAND, and library selection guidance.

---

## Table of Contents

1. [cuBLAS and cuBLASLt](#1-cublas-and-cublaslt)
2. [cuDNN](#2-cudnn)
3. [NCCL](#3-nccl)
4. [Transformer Engine](#4-transformer-engine)
5. [cuSPARSE and cuSPARSELt](#5-cusparse-and-cusparselt)
6. [cuFFT](#6-cufft)
7. [CUTLASS](#7-cutlass)
8. [Thrust / CUB](#8-thrust--cub)
9. [cuRAND](#9-curand)
10. [Library Selection Decision Tree](#10-library-selection-decision-tree)

---

## 1. cuBLAS and cuBLASLt

### 1.1 Overview

cuBLAS is NVIDIA's implementation of the Basic Linear Algebra Subprograms (BLAS) on GPU. It provides highly optimized implementations of matrix multiplication (GEMM) and other linear algebra operations. cuBLASLt (Lightweight) is an extended API offering finer control over algorithm selection, epilogue fusion, and mixed-precision operations.

### 1.2 Core API: cublasGemmEx

`cublasGemmEx` is the extended precision GEMM function supporting mixed-precision computation:

```c
cublasStatus_t cublasGemmEx(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void *alpha,
    const void *A, cudaDataType_t Atype, int lda,
    const void *B, cudaDataType_t Btype, int ldb,
    const void *beta,
    void *C, cudaDataType_t Ctype, int ldc,
    cublasComputeType_t computeType,
    cublasGemmAlgo_t algo
);
```

**Key Parameters:**
- `Atype`, `Btype`, `Ctype`: Independent data types for each matrix (FP16, BF16, FP8 E4M3, FP8 E5M2, FP32, FP64, INT8)
- `computeType`: Controls internal accumulation precision
- `algo`: Algorithm selection (heuristic or explicit)

### 1.3 cublasGemmStridedBatchedEx

Batched GEMM with constant stride between matrices in each batch:

```c
cublasStatus_t cublasGemmStridedBatchedEx(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void *alpha,
    const void *A, cudaDataType_t Atype, int lda, long long strideA,
    const void *B, cudaDataType_t Btype, int ldb, long long strideB,
    const void *beta,
    void *C, cudaDataType_t Ctype, int ldc, long long strideC,
    int batchCount,
    cublasComputeType_t computeType,
    cublasGemmAlgo_t algo
);
```

Critical for attention computation (batched BMM across heads) and MoE expert execution.

### 1.4 Math Modes

| Math Mode | Description | Use Case |
|-----------|-------------|----------|
| `CUBLAS_DEFAULT_MATH` | Default precision for the data type | General purpose, maximum accuracy |
| `CUBLAS_TF32_TENSOR_OP_MATH` | TF32 Tensor Core operations on FP32 data | Training with Ampere+ GPUs, ~8x speedup over FP32 with minimal accuracy loss |
| `CUBLAS_PEDANTIC_MATH` | Strict IEEE compliance, no reduced precision | Numerical validation, debugging |
| `CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION` | Prevents reduced precision in accumulation | When intermediate precision matters |

**Compute Type Enumeration (for cublasGemmEx/cublasLtMatmul):**

| Compute Type | Description |
|-------------|-------------|
| `CUBLAS_COMPUTE_16F` | FP16 compute |
| `CUBLAS_COMPUTE_16F_PEDANTIC` | Strict FP16 |
| `CUBLAS_COMPUTE_32F` | FP32 compute |
| `CUBLAS_COMPUTE_32F_PEDANTIC` | Strict FP32, no TF32 |
| `CUBLAS_COMPUTE_32F_FAST_TF32` | FP32 inputs, TF32 Tensor Core math |
| `CUBLAS_COMPUTE_32F_FAST_16F` | FP32 inputs, FP16 Tensor Core math |
| `CUBLAS_COMPUTE_32F_FAST_16BF` | FP32 inputs, BF16 Tensor Core math |
| `CUBLAS_COMPUTE_64F` | FP64 compute |
| `CUBLAS_COMPUTE_64F_PEDANTIC` | Strict FP64 |
| `CUBLAS_COMPUTE_32I` | INT32 compute |
| `CUBLAS_COMPUTE_32I_PEDANTIC` | Strict INT32 |

**TF32 Override:** Setting `NVIDIA_TF32_OVERRIDE=0` disables TF32 tensor core acceleration globally regardless of programmatic configuration.

### 1.5 Algorithm Selection

`cublasGemmAlgo_t` controls kernel selection:

| Algorithm | Description |
|-----------|-------------|
| `CUBLAS_GEMM_DEFAULT` | Library heuristic selects best kernel |
| `CUBLAS_GEMM_DEFAULT_TENSOR_OP` | Prefer Tensor Core kernels |
| `CUBLAS_GEMM_ALGO0` through `CUBLAS_GEMM_ALGO23` | Specific algorithm IDs for benchmarking |
| `CUBLAS_GEMM_ALGO0_TENSOR_OP` through `CUBLAS_GEMM_ALGO15_TENSOR_OP` | Specific Tensor Core algorithm IDs |

**Best Practice:** Use `CUBLAS_GEMM_DEFAULT` or `CUBLAS_GEMM_DEFAULT_TENSOR_OP` in production. Explicit algorithm IDs are for autotuning benchmarks only -- they may become invalid across cuBLAS versions.

### 1.6 Performance: When cuBLAS is Optimal vs When Custom Kernels Win

**cuBLAS is optimal when:**
- Standard GEMM shapes (large M, N, K) without fused operations
- You need broad hardware compatibility across GPU generations
- Problem is purely matrix multiplication with no custom pre/post-processing
- Batch sizes are standard and strided

**Custom kernels beat cuBLAS when:**
- You need epilogue fusion beyond what cuBLASLt supports (custom activation functions, complex scaling)
- Non-standard memory layouts or tiling patterns are required
- Very small matrix dimensions where cuBLAS kernel launch overhead dominates
- Fused operations spanning multiple GEMM calls (e.g., multi-head attention)
- Custom quantization/dequantization schemes fused with GEMM
- Grouped GEMM with irregular problem sizes (MoE workloads)

### 1.7 cuBLASLt: The Lightweight API

cuBLASLt provides lower-level control over GEMM execution:

#### Core Objects

```c
// Handle
cublasLtHandle_t lightHandle;
cublasLtCreate(&lightHandle);

// Matrix Layout Descriptor
cublasLtMatrixLayout_t layoutA, layoutB, layoutC, layoutD;
cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_16F, m, k, lda);

// Matmul Descriptor
cublasLtMatmulDesc_t operationDesc;
cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

// Algorithm selection via preference
cublasLtMatmulPreference_t preference;
cublasLtMatmulPreferenceCreate(&preference);
cublasLtMatmulPreferenceSetAttribute(preference,
    CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
    &workspaceSize, sizeof(workspaceSize));

// Heuristic query
cublasLtMatmulHeuristicResult_t heuristicResult[8];
int returnedResults;
cublasLtMatmulAlgoGetHeuristic(lightHandle, operationDesc,
    layoutA, layoutB, layoutC, layoutD,
    preference, 8, heuristicResult, &returnedResults);

// Execute
cublasLtMatmul(lightHandle, operationDesc,
    alpha, A, layoutA, B, layoutB,
    beta, C, layoutC, D, layoutD,
    &heuristicResult[0].algo, workspace, workspaceSize, stream);
```

#### Workspace Requirements

H100/Hopper native kernels require significantly more workspace:
- **Minimum recommended: 32 MiB (33,554,432 bytes)**
- Performance degrades substantially with insufficient workspace
- Workspace is used for split-K reductions, algorithm-specific scratch space, and data rearrangement

#### Epilogue Fusion

cuBLASLt supports fusing post-GEMM operations into the GEMM kernel, avoiding separate kernel launches:

| Epilogue | Operation | Notes |
|----------|-----------|-------|
| `CUBLASLT_EPILOGUE_DEFAULT` | D = alpha * A * B + beta * C | Standard GEMM |
| `CUBLASLT_EPILOGUE_BIAS` | D = alpha * A * B + beta * C + bias | Bias vector addition |
| `CUBLASLT_EPILOGUE_RELU` | D = ReLU(alpha * A * B + beta * C) | ReLU activation |
| `CUBLASLT_EPILOGUE_RELU_BIAS` | D = ReLU(alpha * A * B + beta * C + bias) | Fused bias + ReLU |
| `CUBLASLT_EPILOGUE_RELU_AUX` | D = ReLU(...), aux = mask | ReLU with bitmask for backward |
| `CUBLASLT_EPILOGUE_RELU_AUX_BIAS` | D = ReLU(... + bias), aux = mask | Full fused forward |
| `CUBLASLT_EPILOGUE_GELU` | D = GELU(alpha * A * B + beta * C) | GELU activation |
| `CUBLASLT_EPILOGUE_GELU_BIAS` | D = GELU(... + bias) | Fused bias + GELU |
| `CUBLASLT_EPILOGUE_GELU_AUX` | D = GELU(...), aux = pre-activation | GELU with auxiliary output |
| `CUBLASLT_EPILOGUE_GELU_AUX_BIAS` | D = GELU(... + bias), aux = pre-act | Full fused forward |
| `CUBLASLT_EPILOGUE_DGELU` | Backward GELU gradient | Training backward pass |
| `CUBLASLT_EPILOGUE_DRELU` | Backward ReLU gradient | Training backward pass |
| `CUBLASLT_EPILOGUE_DRELU_BGRAD` | dReLU + bias gradient | Fused backward |
| `CUBLASLT_EPILOGUE_DGELU_BGRAD` | dGELU + bias gradient | Fused backward |

**Hopper FP8 Epilogues:** BIAS (BF16/FP16), ReLU, GELU, with and without auxiliary output buffers are supported for FP8 data types on Hopper.

#### Mixed-Type GEMM

cuBLASLt supports GEMMs where A and B have different data types:
- FP8 E4M3 x FP8 E5M2 with FP16/BF16/FP32 accumulation
- FP16 x INT8 (select configurations)
- Per-matrix scaling factors for A, B, C, D independently
- Absolute maximum (amax) computation for output matrices (critical for FP8 scale factor management)

### 1.8 cuBLAS with FP8 on Hopper

**FP8 Formats:**
- **E4M3** (1 sign, 4 exponent, 3 mantissa): Range [-448, 448], higher precision, used for forward pass
- **E5M2** (1 sign, 5 exponent, 2 mantissa): Range [-57344, 57344], wider dynamic range, used for backward pass

**Performance on Hopper (H100 SXM vs A100 PCIe):**
- FP8 GEMM: **4.8x speedup**
- FP16 GEMM: **3x speedup** (compute-bound)
- BF16 GEMM: **2.8x speedup**

**cuBLAS 12.9 FP8 Enhancements:**
- **Channel-wide / outer vector scaling:** A single scaling factor per matrix row of A[MxK] or column of B[KxN]
- **Block scaling:** Scaling factor per 128-element 1D block in K dimension, or 128x128 2D block
- Up to **1.75x speedup** over BF16 baselines on H200

**Blackwell (cuBLAS 12.9+):**
- Native block-scaled FP4 (`CUDA_R_4F_E2M1`) with 16-element blocks
- Block-scaled FP8 with 32-element blocks using unsigned exponent-only (E8M0) scaling
- FP4 achieves **4.6x speedup over FP8** on Blackwell
- cuBLAS can compute scaling factors for output tensors inline, eliminating pre-estimation overhead
- FP32 emulation via BF16 Tensor Cores: **3-4x faster** than native FP32

---

## 2. cuDNN

### 2.1 Overview

cuDNN (CUDA Deep Neural Network library) provides highly tuned implementations of deep learning primitives: convolutions, normalization, pooling, activation functions, attention mechanisms, and more. Since cuDNN 8, the Graph API enables describing computation as a dataflow graph for automatic fusion and optimization.

### 2.2 Frontend API vs Legacy API

| Aspect | Frontend API (C++/Python) | Legacy/Backend API (C) |
|--------|---------------------------|------------------------|
| Language | C++17, Python | C only |
| Abstraction | High-level convenience nodes | Low-level descriptor-based |
| Recommended | Yes (for new code) | For existing integrations |
| Installation | `pip install nvidia-cudnn-cu12` | CUDA Toolkit / manual |
| SDPA Support | Single-node convenience call | Manual BMM-Softmax-BMM graph construction |
| Error Reporting | Enhanced with `cuDNNGetLastErrorString` | Status codes only |
| Forward Compatibility | Large API subset compatible across architectures | Version-locked |

### 2.3 Graph API: Building Operation Graphs

A cuDNN graph represents computation as a dataflow graph where **operations are nodes** and **tensors are edges**. Tensors implicitly connect operations via producer-consumer relationships.

**Workflow:**
1. Create a graph object
2. Add operation nodes (convolution, matmul, pointwise, reduction, normalization)
3. Set I/O tensors with data types, dimensions, strides
4. Build the graph (validates the pattern)
5. Query heuristics for candidate engine configurations
6. Check support status
7. Allocate workspace
8. Execute

**Heuristic Modes:**
| Mode | Description | CPU Latency | Quality |
|------|-------------|-------------|---------|
| Mode A | Fast heuristic, handles most patterns | Low | Good (ranked configs) |
| Mode B | More accurate performance prediction | Higher | Better |
| Fallback | Functional fallback | Lowest | No optimization guarantee |

**Autotuning:** Time each candidate engine configuration and select the fastest. Both C++ and Python APIs provide utility functions for this.

### 2.4 Fusion Engines

cuDNN provides three categories of fusion engines:

#### Pre-Compiled Single Operation Engines
- Individual convolution operations: `ConvolutionFwd`, `ConvolutionBwdFilter`, `ConvolutionBwdData`
- Normalization: `NormalizationForward`, `NormalizationBackward`

#### Generic Runtime Fusion Engines
JIT-compiled kernels generated at runtime based on the graph pattern:
- Matmul fusions with preprocessing (g1) and postprocessing (g2) DAGs
- Convolution fusions with pointwise pre/post operations
- Pointwise and reduction operations
- Three support surfaces based on compute capability (SM90, SM80, SM70)
- SM90 (Hopper): Full FP8, grouped convolutions
- SM80 (Ampere): Standard mixed precision
- SM70 (Volta): Limited precision and layout options

#### Specialized Runtime Fusion Engines
Optimized for common deep learning patterns:
- **BnAddRelu**: Batch norm + addition + ReLU (ResNet-like architectures)
- **DReluForkDBn**: Backward pass complement to BnAddRelu
- **Fused Attention**: Forward and backward multi-head attention
- **Flash Attention**: Optimized SDPA with dropout, masking, FP8

### 2.5 cuDNN Flash Attention / SDPA

cuDNN implements flash fused attention for the BMM-Softmax-BMM pattern:

**Supported Configurations:**
- Data types: FP16, BF16, FP8 (E4M3 on Hopper)
- Head dimensions: Commonly 64, 128, 256
- Sequence lengths: Arbitrary (with padding support)
- Masking: Causal, padding, sliding window, ALiBi
- Optional: Dropout (with Philox RNG), attention scale, relative positional encoding

**Performance:**
- Up to **1.2 PFLOPS in FP8** on H200
- Up to **2x faster** than PyTorch eager BF16
- Up to **3x faster** in FP8 vs PyTorch eager
- **1.15x speedup** for Llama2 70B LoRA fine-tuning on 8-GPU H200 node

**Frontend Convenience API (Python):**
```python
import cudnn

graph = cudnn.pygraph()
Q = graph.tensor(name="Q", dim=[B, H, S, D], stride=[...], data_type=cudnn.data_type.HALF)
K = graph.tensor(name="K", dim=[B, H, S, D], stride=[...], data_type=cudnn.data_type.HALF)
V = graph.tensor(name="V", dim=[B, H, S, D], stride=[...], data_type=cudnn.data_type.HALF)

O, stats = graph.sdpa(
    q=Q, k=K, v=V,
    is_inference=False,
    attn_scale=1.0 / math.sqrt(D),
    use_causal_mask=True
)
O.set_output(True).set_data_type(cudnn.data_type.HALF)
graph.validate()
graph.build_operation_graph()
graph.create_execution_plans([cudnn.heur_mode.A])
graph.check_support()
graph.build_plans()
```

**Native Sparse Attention (NSA):** cuDNN v9.13.0+ supports NSA as described in "Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention" with C++ API samples for sink attention forward and backward propagation.

### 2.6 Convolution Algorithms

| Algorithm | Description | Best For | Memory |
|-----------|-------------|----------|--------|
| **Implicit GEMM** | Operates directly on input tensors, transforms on-the-fly | General purpose, 1x1 and 5x5 filters with Tensor Cores | Low |
| **Implicit GEMM (Precomp)** | Precomputes offsets for the implicit transform | Large batches, 1x1/5x5 filters | Medium |
| **FFT** | Transforms to frequency domain, pointwise multiply, inverse transform | Large filters, large inputs, large batch sizes | High (workspace) |
| **FFT Tiled** | Tiled FFT variant reducing workspace requirements | Same as FFT with memory constraints | Medium |
| **Winograd** | Minimal filtering algorithm, reduces multiply count | **3x3 filters** (optimal choice), Tensor Core accelerated | Medium |
| **Winograd Non-Fused** | Separate transform/multiply/inverse stages | When fused Winograd cannot be used | Medium |
| **Direct** | Straightforward convolution computation | Small/uncommon filter sizes | Low |

**Algorithm Selection Guidance:**
- **3x3 filters:** Winograd with Tensor Cores yields highest performance
- **1x1 and 5x5 filters:** GEMM-impl-precomp with Tensor Cores is fastest
- **Small inputs, large batches:** GEMM algorithm
- **Large filters, large inputs:** FFT variants
- **Hardware utilization:** Both Winograd and naive convolution achieve ~43.4% of theoretical peak; Winograd does so with fewer FLOPs

### 2.7 Normalization (BN, LN, RMSNorm) Fused Kernels

cuDNN Graph API supports fused normalization operations:

**Forward Operations:**
- Layer Norm, Batch Norm, RMS Norm: Supported in `GRAPH_JIT_ONLY` and `FULL` library configs
- Runtime-compiled specialized engines (Engine Index 3 for training, Index 4 for inference)

**Backward Operations:**
- All three normalization types supported for backward pass
- Specialized runtime-compiled engines for Layer Norm and RMS Norm backward

**Fusion Patterns:**
- **BnAddRelu**: Batch norm + element-wise add + ReLU (ResNet blocks)
- **DReluForkDBn**: Backward dReLU forking into dBatchNorm

### 2.8 cuDNN 9.x Key Features

- **Reorganized sub-libraries:** Legacy functionality separated from graph API and engine implementations
- **Mixed input precision:** Matmuls and convolutions where A and B have different types (e.g., FP16 x INT8), with online fused type conversion
- **Python frontend:** `pip install nvidia-cudnn-cu12`
- **Programmatic error access:** `cuDNNGetLastErrorString()`
- **Forward compatibility:** Large API subset works across future architectures
- **NSA (v9.13.0+):** Native Sparse Attention support

---

## 3. NCCL (NVIDIA Collective Communications Library)

### 3.1 Overview

NCCL provides multi-GPU and multi-node collective communication primitives optimized for NVIDIA GPU topologies. It automatically detects the interconnect topology and selects optimal algorithms and transport protocols.

### 3.2 All Collectives

| Collective | Operation | Communication Pattern |
|------------|-----------|----------------------|
| `ncclAllReduce` | Reduce across all ranks, result on all ranks | All-to-all reduction + broadcast |
| `ncclAllGather` | Gather data from all ranks to all ranks | All-to-all gather |
| `ncclReduceScatter` | Reduce + scatter result across ranks | Reduce then distribute |
| `ncclBroadcast` | Send from one rank to all others | One-to-all |
| `ncclReduce` | Reduce across all ranks, result on one rank | All-to-one |
| `ncclAllToAll` | Each rank sends distinct data to every other rank | Personalized all-to-all |
| `ncclSend` / `ncclRecv` | Point-to-point communication | Direct rank-to-rank |

**Reduction Operations:** Sum, Product, Min, Max, Avg (FP types), PreMulSum (pre-multiply then sum)

### 3.3 Transport Layers

NCCL abstracts data transfer through multiple transport backends, selecting the optimal one based on GPU topology:

| Transport | When Used | Bandwidth | Latency |
|-----------|-----------|-----------|---------|
| **NVLink (P2P)** | Same node, NVLink-connected GPUs | Up to 900 GB/s (NVLink 5) | Lowest |
| **PCIe P2P** | Same node, PCIe-connected, no NVLink | PCIe Gen5: ~64 GB/s | Low |
| **SHM (Shared Memory)** | Same node, P2P suboptimal or disabled | System memory bandwidth | Medium |
| **NET (TCP/IP Sockets)** | Cross-node, no RDMA | 10-100 Gbps | High |
| **NET (InfiniBand/RoCE)** | Cross-node, RDMA available | 200-400 Gbps (NDR) | Medium |
| **GPU Direct RDMA** | Cross-node, NIC-GPU direct path | Full NIC bandwidth | Low |

**Blackwell Direct NIC (NCCL 2.27+):** On Grace Blackwell platforms, CX8 NIC exposes two virtual PCIe trees. One connects directly to GPU via PCIe Gen6 x16, bypassing CPU -- up to 800 Gb/s network bandwidth.

### 3.4 Algorithms: Ring vs Tree vs NVLS vs PAT

| Algorithm | Description | Best For | Bandwidth Utilization |
|-----------|-------------|----------|----------------------|
| **Ring** | Data flows in a ring through all ranks | Large messages, bandwidth-bound | Optimal for 2 ranks, degrades at scale |
| **Tree** | Hierarchical reduction/broadcast tree | Small-medium messages, latency-sensitive | Sub-optimal bandwidth, better latency |
| **CollNet** | Network-offloaded reduction (SHARP-enabled switches) | Large-scale AllReduce | Offloads compute to switches |
| **NVLS** | NVLink SHARP: intra-node NVLink reduction + inter-node CollNet | Hopper+ with NVSwitch | Hardware-accelerated intra-node |
| **NVLSTree** | NVLS intra-node + Tree inter-node | When CollNet unavailable | Best hybrid approach |
| **PAT** | Point-to-all-to-point algorithm | Specific topology patterns | Topology-dependent |

### 3.5 NVLS (NVLink SHARP) for Hopper+

NVLS leverages NVSwitch for hardware-accelerated intra-node reduction:
- Uses NVLink SHARP for intra-node collective operations
- Supports NVL72 (GB200/GB300 systems) and NVL8 (DGX/HGX systems)
- **Up to 2.5x higher performance** for small-medium messages on NVL8
- Combined with SHARP for inter-node (NVLS) or Tree for inter-node (NVLSTree)

### 3.6 NCCL 2.27 Key Features

- **Symmetric Memory:** Buffers with identical virtual addresses across GPUs enable optimized collective operations. Up to **9x reduction in latency** for small messages.
- **SHARP Extension:** AllGather and ReduceScatter now SHARP-accelerated. Reduces SM usage from 16+ to **6 or fewer SMs**.
- **Communicator Shrink:** Dynamic GPU exclusion for fault tolerance during training. Two modes: default (planned) and error (failure recovery).
- **FP32 Accumulation:** Reductions computed with FP32 accumulators (FP16 for FP8 on NVSwitch).

**Symmetric Memory API:**
```c
ncclCommWindowRegister(comm, buffer, size, NCCL_WIN_COLL_SYMMETRIC, &handle);
// ... perform collectives ...
ncclCommWindowDeregister(comm, handle);
// Buffers must be allocated via CUDA Virtual Memory Management APIs
```

### 3.7 NCCL Environment Variables for Tuning

#### Algorithm and Protocol Selection

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `NCCL_ALGO` | Ring, Tree, CollnetChain, CollnetDirect, NVLS, NVLSTree, PAT | All | Allowed algorithms (comma-separated, `^` to exclude) |
| `NCCL_PROTO` | LL, LL128, Simple | All | Communication protocols |

**Protocol Details:**
- **LL (Low Latency):** 8-byte inline data with flag, best for small messages
- **LL128:** 128-byte chunks with flag checking, better bandwidth than LL
- **Simple:** Full buffer-based protocol, highest bandwidth for large messages

#### Channel and Thread Configuration

| Variable | Range | Default | Description |
|----------|-------|---------|-------------|
| `NCCL_MAX_NCHANNELS` | >= 1 | Platform-dependent | Max communication channels |
| `NCCL_MIN_NCHANNELS` | 1-32 | Platform-dependent | Min channels |
| `NCCL_NTHREADS` | 64, 128, 256, 512 | 256-512 | CUDA threads per block |
| `NCCL_BUFFSIZE` | Bytes (power of 2) | 4 MB | Inter-GPU buffer size |
| `NCCL_MAX_CTAS` | 0-64 | Auto | Max cooperative thread arrays |
| `NCCL_MIN_CTAS` | 0-64 | Auto | Min cooperative thread arrays |
| `NCCL_CGA_CLUSTER_SIZE` | 0-8 | Auto | CUDA cluster dim (SM90+) |

#### Network Configuration

| Variable | Values | Description |
|----------|--------|-------------|
| `NCCL_SOCKET_IFNAME` | Interface names | IP interface selection (`eth0`, `^docker0`, `=eth0` for exact) |
| `NCCL_IB_HCA` | HCA names | InfiniBand HCA filter (`mlx5`, `=mlx5_0:1`) |
| `NCCL_IB_DISABLE` | 0/1 | Disable InfiniBand, fall back to sockets |
| `NCCL_NET_GDR_LEVEL` | LOC, PIX, PXB, PHB, SYS | GPU Direct RDMA distance cutoff |
| `NCCL_NET_GDR_READ` | 0/1 | Enable GDR for sends (default 1 on NVLink) |
| `NCCL_P2P_LEVEL` | LOC, NVL, PIX, PXB, PHB, SYS | Max GPU distance for P2P |
| `NCCL_P2P_DISABLE` | 0/1 | Disable all P2P |
| `NCCL_SHM_DISABLE` | 0/1 | Disable shared memory transport |
| `NCCL_NVLS_ENABLE` | 0/1/2 | NVLink SHARP (0=off, 1=on, 2=auto) |

#### Debugging

| Variable | Values | Description |
|----------|--------|-------------|
| `NCCL_DEBUG` | VERSION, WARN, INFO, TRACE | Verbosity level |
| `NCCL_DEBUG_FILE` | Path | Output file (`%h` hostname, `%p` PID) |
| `NCCL_DEBUG_SUBSYS` | INIT, COLL, P2P, SHM, NET, GRAPH, TUNING | Filter INFO by subsystem |

### 3.8 Custom Plugins and External Networks

NCCL supports external network plugins via `libnccl-net.so`:
- Custom RDMA implementations
- Proprietary interconnects (e.g., AWS EFA, Google TPCI)
- Plugin API provides send/recv/flush/poll primitives

### 3.9 MSCCL (Microsoft Collective Communication Library)

MSCCL is built on top of NCCL, using its building blocks to execute custom collective communication algorithms:

- **DSL:** MSCCLang high-level language for defining custom algorithms
- **Compiler:** Generates IR for the MSCCL runtime
- **Performance:** AllReduce up to **48% faster**, AllToAll up to **20% faster** than vendor implementations
- **Custom Alltoall:** Native support added, accessible via PyTorch `torch.distributed` patching
- **MSCCL++:** Newer abstraction (2025) with separation of concerns between primitive and higher-level portable interfaces

### 3.10 Performance Characteristics per Collective per Interconnect

| Collective | NVLink (intra-node) | InfiniBand (inter-node) | PCIe (intra-node) |
|------------|-------------------|----------------------|------------------|
| AllReduce | Ring/NVLS: near line-rate | Tree/CollNet: scales well | Ring: bandwidth-limited |
| AllGather | Near line-rate | Tree: latency-optimized | Moderate |
| ReduceScatter | Similar to AllReduce | SHARP-accelerated (2.27+) | Moderate |
| Broadcast | Direct copy | Tree: log(N) steps | Tree: log(N) steps |
| AllToAll | Good for MoE | Bandwidth-demanding | Poor scaling |

---

## 4. Transformer Engine

### 4.1 Overview

Transformer Engine (TE) is a library for accelerating Transformer models on NVIDIA GPUs using FP8 (and FP4) precision on Hopper, Ada, and Blackwell architectures. It provides drop-in replacements for PyTorch, JAX, and PaddlePaddle modules that automatically manage mixed-precision FP8 training.

### 4.2 Delayed Scaling Mechanism

The core innovation of TE is **delayed scaling** -- computing FP8 scale factors from historical data rather than requiring costly just-in-time multi-pass computation.

**How It Works:**
1. Each FP8 tensor maintains an **amax history** buffer (configurable length, typically 16 iterations)
2. After each iteration, the current amax (max absolute value) of the tensor is recorded
3. The **scale factor** for the next iteration is computed from the history:
   - `scale = FP8_MAX / amax_from_history`
   - Where `amax_from_history` is computed via the configured algorithm (e.g., `max` of history)
4. This scale factor is applied to quantize the tensor to FP8 for the next forward/backward pass

**Advantages:**
- Eliminates the need for a separate pass to compute amax before quantization
- Introduces only a one-iteration lag in scale factor estimation
- Works well because tensor value distributions change gradually between iterations

### 4.3 FP8 Formats

| Format | Bits | Exponent | Mantissa | Range | Use Case |
|--------|------|----------|----------|-------|----------|
| E4M3 | 8 | 4 | 3 | [-448, 448] | Forward pass (weights, activations) |
| E5M2 | 8 | 5 | 2 | [-57344, 57344] | Backward pass (gradients) |

**Hybrid Format:** `Format.HYBRID` uses E4M3 for forward, E5M2 for backward -- the most common configuration.

### 4.4 te.Linear: Drop-in FP8 Linear Layer

```python
import transformer_engine.pytorch as te

# Drop-in replacement for torch.nn.Linear
linear = te.Linear(in_features=4096, out_features=4096, bias=True)
```

**Constraints:**
- Both tensor dimensions must be divisible by 16
- Sequence length padding may be required for transformer applications

### 4.5 Recipe API

```python
from transformer_engine.common.recipe import Format, DelayedScaling

# Standard delayed scaling recipe
fp8_recipe = DelayedScaling(
    fp8_format=Format.HYBRID,       # E4M3 forward, E5M2 backward
    amax_history_len=16,            # Number of iterations to track
    amax_compute_algo="max"         # How to aggregate amax history
)

# Usage with autocast context manager
with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    output = linear(input_tensor)    # Automatically uses FP8

# IMPORTANT: Backward pass should occur OUTSIDE autocast context
loss = output.mean()
loss.backward()
```

### 4.6 MXFP8 Block Scaling (Blackwell)

Blackwell introduces MXFP8 with per-block scaling (32-element blocks):

```python
from transformer_engine.common.recipe import MXFP8BlockScaling, Format

mxfp8_recipe = MXFP8BlockScaling(fp8_format=Format.E4M3)
```

**Key Differences from Standard FP8:**
| Aspect | Standard FP8 (Hopper) | MXFP8 (Blackwell) |
|--------|----------------------|-------------------|
| Scaling Granularity | Per-tensor (single FP32 scale) | Per-block (32 consecutive elements) |
| Scale Factor Format | FP32 (E8M23) | E8M0 (8-bit power-of-2) |
| Forward Format | E4M3 | E4M3 everywhere |
| Backward Format | E5M2 | E4M3 (higher precision gradients) |
| Hardware Scaling | Software-managed | Hardware-accelerated in GEMM path |
| Transpose | Scale preserved | Requires requantization (block orientation changes) |

### 4.7 NVFP4 (Blackwell)

4-bit format using E2M1 representation (values up to +/-6):
- **Block scaling:** 1 scale factor per 16 elements (finer than MXFP8's 32)
- **Per-block dtype:** E4M3
- **Dual-level scaling:** Per-block E4M3 + per-tensor FP32
- **Weight variant:** 2D blocking (16x16), mirrors DeepSeek-v3 approach
- **Training enhancements:** Stochastic rounding, random Hadamard transforms for outlier smoothing

### 4.8 Integration with Frameworks

| Framework | Module | Key Classes |
|-----------|--------|-------------|
| **PyTorch** | `transformer_engine.pytorch` | `te.Linear`, `te.LayerNorm`, `te.TransformerLayer`, `te.MultiheadAttention` |
| **JAX** | `transformer_engine.jax` | `te.DenseGeneral`, `te.LayerNorm`, `te.TransformerLayer` |
| **PaddlePaddle** | `transformer_engine.paddle` | Similar drop-in modules |

### 4.9 Performance

- FP8 training typically achieves **1.2-1.5x speedup** over BF16 on Hopper with negligible accuracy loss
- Memory savings from FP8 weights/activations enable larger batch sizes
- MXFP8 on Blackwell provides additional speedups via hardware-accelerated block scaling
- Best gains on compute-bound workloads (large batch, large model)

### 4.10 Interaction with Tensor Cores

TE automatically:
1. Quantizes weights and activations to FP8 using the computed scale factors
2. Calls cuBLASLt with FP8 data types, which dispatches to Tensor Core FP8 instructions
3. Accumulates in FP32 (higher precision) on Tensor Cores
4. Dequantizes the output back to the master precision (FP32/BF16)
5. Updates amax history for the next iteration

The Tensor Core FP8 matrix multiply-accumulate instruction on Hopper: `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`

---

## 5. cuSPARSE and cuSPARSELt

### 5.1 Overview

cuSPARSE provides general-purpose sparse matrix operations. cuSPARSELt is specialized for **structured sparsity** (2:4 pattern) leveraging Tensor Core sparse matrix multiply-accumulate (SpMMA) instructions introduced with Ampere.

### 5.2 Structured Sparsity (2:4)

**The 2:4 Pattern:** Out of every 4 consecutive elements, at least 2 must be zero.

```
Original:  [1.2, 0.0, 3.4, 0.0, 0.0, 5.6, 0.0, 7.8]
            ---- group 1 ----   ---- group 2 ----
Valid 2:4: 2 nonzeros per group of 4 elements
```

### 5.3 How 2:4 Sparsity Works on Tensor Cores

1. **Sparse operand A** has 2:4 structure
2. **Compression:** Only nonzero values are stored (50% data reduction) plus metadata indices
3. **Hardware execution:** Sparse Tensor Cores skip multiplications by zero values using metadata to select corresponding elements from the dense operand B
4. **Result:** 2x math throughput compared to dense computation

### 5.4 Compression Format and Metadata

The compressed representation has two components:

| Component | Description | Size |
|-----------|-------------|------|
| **Data** | Nonzero values in dense format | 50% of original |
| **Metadata** | 2-bit indices per nonzero indicating position in the group of 4 | 1/8 of original data size |

For each group of 4 elements with 2 nonzeros:
- 2 data values stored contiguously
- 2 x 2-bit index stored in metadata (positions 0-3)

**Memory Savings:**
- Data footprint: **2x reduction** in the sparse operand
- Bandwidth requirement: Proportionally reduced
- Metadata overhead: Small (~6.25% of original data)

### 5.5 cuSPARSELt API Workflow

```c
// 1. Initialize library handle
cusparseLtHandle_t handle;
cusparseLtInit(&handle);

// 2. Create matrix descriptors
cusparseLtMatDescriptor_t matA, matB, matC;
cusparseLtStructuredDescriptorInit(&handle, &matA,
    rows_A, cols_A, lda, alignment, CUDA_R_16F,
    CUSPARSE_ORDER_ROW, CUSPARSELT_SPARSITY_50_PERCENT);
cusparseLtDenseDescriptorInit(&handle, &matB,
    rows_B, cols_B, ldb, alignment, CUDA_R_16F, CUSPARSE_ORDER_ROW);
cusparseLtDenseDescriptorInit(&handle, &matC,
    rows_C, cols_C, ldc, alignment, CUDA_R_16F, CUSPARSE_ORDER_ROW);

// 3. Create matmul descriptor
cusparseLtMatmulDescriptor_t matmulDesc;
cusparseLtMatmulDescriptorInit(&handle, &matmulDesc,
    CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
    &matA, &matB, &matC, &matC, CUSPARSE_COMPUTE_16F);

// 4. Algorithm selection
cusparseLtMatmulAlgSelection_t algSelection;
cusparseLtMatmulAlgSelectionInit(&handle, &algSelection,
    &matmulDesc, CUSPARSELT_MATMUL_ALG_DEFAULT);

// 5. Create execution plan
cusparseLtMatmulPlan_t plan;
size_t workspaceSize;
cusparseLtMatmulPlanInit(&handle, &plan, &matmulDesc, &algSelection);
cusparseLtMatmulGetWorkspace(&handle, &plan, &workspaceSize);

// 6. Prune the matrix to 2:4 sparsity (if not already pruned)
cusparseLtSpMMAPrune(&handle, &matmulDesc, d_A, d_A_pruned,
    CUSPARSELT_PRUNE_SPMMA_STRIP, stream);
// Verify sparsity
int valid;
cusparseLtSpMMAPruneCheck(&handle, &matmulDesc, d_A_pruned, &valid, stream);

// 7. Compress the pruned matrix
size_t compressedSize, compressedBufferSize;
cusparseLtSpMMACompressedSize(&handle, &plan, &compressedSize, &compressedBufferSize);
cusparseLtSpMMACompress(&handle, &plan, d_A_pruned, d_A_compressed,
    d_compressBuffer, stream);

// 8. Execute sparse matmul
float alpha = 1.0f, beta = 0.0f;
cusparseLtMatmul(&handle, &plan, &alpha, d_A_compressed, d_B,
    &beta, d_C, d_D, d_workspace, &stream, 1);

// 9. Cleanup
cusparseLtMatmulPlanDestroy(&plan);
cusparseLtDestroy(&handle);
```

### 5.6 Performance Gains

| Workload | Speedup vs Dense | Notes |
|----------|-----------------|-------|
| BERT-Large FC2 | **1.6x** | Best case for large linear layers |
| BERT-Large general | **1.3-1.6x** | Varies by layer |
| FP16 GEMM (large K) | **~1.5-1.8x** | Compute-bound workloads |
| INT8 inference | **~1.3-1.5x** | TN layout optimized |

**Supported Precision Combinations:**
- FP16 data / FP16 or FP32 accumulation
- BF16 data / FP32 accumulation
- INT8 data / INT32 accumulation
- FP8 (Hopper+): E4M3/E5M2 with FP32 accumulation

### 5.7 Pruning Strategies

- **`CUSPARSELT_PRUNE_SPMMA_STRIP`**: Prune based on magnitude within each group of 4 (retain 2 largest)
- **`CUSPARSELT_PRUNE_SPMMA_TILE`**: Tile-based pruning for better accuracy
- **External pruning:** NVIDIA ASP (Automatic SParsity) in PyTorch, gradual magnitude pruning during training
- **Accuracy preservation:** Fine-tuning after pruning typically recovers accuracy to within 1% of dense baseline

---

## 6. cuFFT

### 6.1 Overview

cuFFT provides GPU-accelerated Fast Fourier Transform implementations. While primarily used in scientific computing, FFT has specific applications in machine learning.

### 6.2 ML Applications of FFT

| Application | Description | Typical Use |
|-------------|-------------|-------------|
| **Spectral convolutions** | FFT-based convolution for large kernels | Signal processing models, FNO (Fourier Neural Operator) |
| **Audio/speech processing** | STFT, mel spectrograms | Whisper, WaveNet preprocessing |
| **Image processing** | Frequency domain filtering | Denoising, super-resolution |
| **Fourier Neural Operators** | Neural operators in frequency domain | PDE solving, physics-informed ML |
| **Positional embeddings** | Rotary position encoding computation | RoPE in transformers (minor use) |

### 6.3 cuFFT API

```c
// 1D FFT
cufftHandle plan;
cufftPlan1d(&plan, n, CUFFT_C2C, batch);
cufftExecC2C(plan, d_input, d_output, CUFFT_FORWARD);
cufftExecC2C(plan, d_output, d_input, CUFFT_INVERSE);
cufftDestroy(plan);

// 2D FFT
cufftPlan2d(&plan, nx, ny, CUFFT_R2C);
cufftExecR2C(plan, d_real_input, d_complex_output);

// Batched FFT (most common in ML)
cufftPlanMany(&plan, rank, n, inembed, istride, idist,
              onembed, ostride, odist, CUFFT_C2C, batch);
```

**Supported Transforms:**
- `CUFFT_R2C` / `CUFFT_C2R`: Real-to-complex / complex-to-real
- `CUFFT_C2C`: Complex-to-complex
- `CUFFT_D2Z` / `CUFFT_Z2D`: Double precision variants
- `CUFFT_Z2Z`: Double complex-to-complex
- Half-precision FFT support (FP16)

### 6.4 Performance Characteristics

- **Batch mode optimization:** cuFFT combines signals from different batches for processing, improving GPU utilization
- **Multi-GPU:** Supports up to 16 GPUs in a single node via cuFFTXt APIs
- **Multi-node:** cuFFTMp extension for multi-node, multi-process FFT on exascale platforms
- **Architecture scaling:** Modern GPUs significantly outperform multi-core CPUs (e.g., surpassing FFTW on 2x12 core Haswell)
- **Performance tip:** Powers of 2 for FFT dimensions yield best performance; use padding when possible

---

## 7. CUTLASS

### 7.1 Overview

CUTLASS (CUDA Templates for Linear Algebra Subroutines) is a collection of CUDA C++ template abstractions for implementing high-performance GEMM and related computations. It provides the building blocks used internally by cuBLAS while exposing them for customization.

### 7.2 CUTLASS 3.x Architecture (5-Layer Hierarchy)

| Layer | Abstraction | Key Types | Role |
|-------|------------|-----------|------|
| **Atom** | Architecture-specific instructions | `cute::Mma_Atom<>`, `cute::Copy_Atom<>` | Lowest-level hardware ISA mapping |
| **Tiled MMA/Copy** | Spatial micro-kernels | `cute::TiledMma<>`, `cute::TiledCopy<>` | Thread-to-data mapping with arbitrary interleaving |
| **Collective** | Temporal organization + sync | `CollectiveMma<>`, `CollectiveEpilogue<>` | Multi-stage pipelining, producer-consumer |
| **Kernel** | Grid-level device code | `cutlass::gemm::kernel::GemmUniversal<>` | Entry point composing mainloop + epilogue |
| **Device** | Host-side interface | `cutlass::gemm::device::GemmUniversalAdapter<>` | Launch configuration, argument setup |

### 7.3 GemmUniversal

The stateless universal entry point for GEMM kernels:

```cpp
// CUTLASS 3.x GemmUniversal
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,          // cute::Shape<M, N, K> or rank-4 for batched
    CollectiveMainloop,    // Mainloop implementation
    CollectiveEpilogue     // Epilogue implementation
>;

// Host-side adapter
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Arguments
typename Gemm::Arguments args{
    problem_size,          // {M, N, K} or {M, N, K, L} for batched
    {ptr_A, stride_A, ptr_B, stride_B},  // mainloop args
    {{alpha, beta}, ptr_C, stride_C, ptr_D, stride_D}  // epilogue args
};

Gemm gemm;
gemm.initialize(args, workspace);
gemm.run(stream);
```

**Problem Shapes:**
- Rank-3: `cute::Shape<int, int, int>` for standard GEMM (M, N, K)
- Rank-4: `cute::Shape<int, int, int, int>` for batched GEMM (M, N, K, L)

### 7.4 CollectiveBuilder Interface

High-level deduction of mainloop and epilogue configurations:

```cpp
using CollectiveMainloop = typename
    cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90,                    // Architecture
        cutlass::arch::OpClassTensorOp,         // Operation class
        ElementA, LayoutA, AlignmentA,          // A matrix config
        ElementB, LayoutB, AlignmentB,          // B matrix config
        ElementAccumulator,                     // Accumulator type
        TileShape_MNK,                          // Tile dimensions
        ClusterShape_MNK,                       // Thread block cluster
        cutlass::gemm::collective::StageCountAutoCarveout<
            sizeof(typename CollectiveEpilogue::SharedStorage)>,
        KernelSchedule                          // Scheduling policy
    >::CollectiveOp;
```

### 7.5 GemmGrouped (CUTLASS 2.x)

For executing multiple GEMMs of different sizes in a single kernel launch (critical for MoE):

```cpp
using GemmGrouped = cutlass::gemm::device::GemmGrouped<
    cutlass::gemm::kernel::DefaultGemmGrouped<
        ElementA, LayoutA,            // A config
        ElementB, LayoutB,            // B config
        ElementC, LayoutC,            // C config
        ElementAccumulator,           // Accumulator
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOp
    >
>;
```

**Key Feature:** Launches fewer threadblocks than total tiles across all problems. Each threadblock computes one or more tiles, with the grouped kernel scheduler assigning tile sequences.

### 7.6 GemmSplitK

Partitions the K dimension for reduction-bound problems:

```cpp
using GemmSplitK = cutlass::gemm::device::GemmSplitKParallel<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape, WarpShape, InstructionShape,
    EpilogueOp,
    SplitKSlices  // Number of K-dimension partitions
>;
// Launches a separate reduction kernel after the GEMM
```

**When to use:** Small M and N but large K dimensions (e.g., attention output projection with small batch).

### 7.7 EpilogueVisitorTree (EVT)

CUTLASS 3.2+ provides composable epilogue fusion via visitor trees:

```cpp
// Example: D = alpha * acc + beta * C + bias
using EpilogueEVT = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90AccFetch,           // Fetch accumulator
    cutlass::epilogue::fusion::Sm90ScalarBroadcast<float>,  // alpha
    cutlass::epilogue::fusion::Sm90EVT<
        cutlass::epilogue::fusion::Sm90SrcFetch,       // Fetch C
        cutlass::epilogue::fusion::Sm90ScalarBroadcast<float>  // beta
    >,
    cutlass::epilogue::fusion::Sm90ColBroadcast<0, TileShape, float>  // bias
>;
```

EVTs allow custom fusion patterns without writing new collective epilogue classes. Operations compose as a tree of visitors.

### 7.8 TensorRef and Memory Abstractions

```cpp
// TensorRef: pointer + stride abstraction
cutlass::TensorRef<ElementA, LayoutA> ref_A(ptr_A, LayoutA(lda));

// HostTensor: manages host+device memory
cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor> tensor_A({M, K});
tensor_A.sync_device();  // Copy host -> device

// In CUTLASS 3.x, CuTe Tensors replace TensorRef:
auto tensor = cute::make_tensor(ptr, cute::make_layout(cute::make_shape(M, K), cute::make_stride(K, 1)));
```

### 7.9 Tile Scheduling Variants

| Scheduler | Description | Best For |
|-----------|-------------|----------|
| **Basic** | One CTA per output tile | Simple workloads |
| **Persistent** | One CTA per SM, computes multiple tiles | Large problems, reduced launch overhead |
| **Stream-K** | Persistent + K-dimension decomposition | Load balancing across SMs for irregular shapes |

### 7.10 CUTLASS 3.x vs 2.x API Differences

| Aspect | CUTLASS 2.x | CUTLASS 3.x |
|--------|-------------|-------------|
| Core abstraction | Threadblock/Warp/Thread hierarchy | CuTe Atom/Tiled/Collective hierarchy |
| Layout system | `cutlass::layout::*` | `cute::Layout` with hierarchical shapes/strides |
| Memory movement | `cutlass::transform::threadblock::*` | `cute::TiledCopy` with `Copy_Atom` |
| MMA operations | `cutlass::gemm::warp::MmaTensorOp` | `cute::TiledMma` with `Mma_Atom` |
| Epilogues | Fixed epilogue classes | EpilogueVisitorTree (composable) |
| Pipelining | Manual pipeline stages | Dispatch policies (pingpong, cooperative) |
| TMA support | Not available | Native TMA integration via CuTe |
| Target architectures | Volta through Hopper | Hopper (SM90) and Blackwell (SM100) primary |
| Code reuse | Monolithic kernel variants | Orthogonal composition maximizes reuse |

### 7.11 CuTe Library

CuTe (CUDA Templates) provides the foundational tensor abstractions for CUTLASS 3.x:

```cpp
// Layout: describes mapping from logical coordinates to linear index
auto layout = cute::make_layout(
    cute::make_shape(M, K),      // Logical shape
    cute::make_stride(K, 1)      // Row-major stride
);

// Tensor: combines pointer + layout
auto tensor = cute::make_tensor(ptr, layout);

// Hierarchical shapes for thread/value decomposition
auto thr_val_layout = cute::make_layout(
    cute::make_shape(cute::make_shape(8, 4), cute::make_shape(2, 2)),
    cute::make_stride(...)
);
```

**Key Abstractions:**
- `Layout`: Shape + Stride, compactly describes multidimensional indexing
- `Tensor`: Data pointer + Layout
- `TiledMma`: Spatial MMA micro-kernel with thread-to-data mapping
- `TiledCopy`: Spatial copy micro-kernel (global->shared, shared->register)

---

## 8. Thrust / CUB

### 8.1 CUB Overview

CUB (CUDA UnBound) provides reusable CUDA primitives at multiple granularities. It is lower-level than Thrust and specific to CUDA C++, accommodating CUDA-specific features like shared memory, warp-level operations, and architecture specialization.

CUB and Thrust are now part of the **CUDA Core Compute Libraries (CCCL)** project.

### 8.2 CUB Device-Wide Primitives

| Primitive | Operation | When to Use |
|-----------|-----------|-------------|
| `cub::DeviceReduce::Sum` | Sum all elements | Loss computation, norm calculation |
| `cub::DeviceReduce::Min/Max` | Find min/max element | Finding amax for FP8 scaling |
| `cub::DeviceReduce::ReduceByKey` | Reduce segments by key | Grouped operations |
| `cub::DeviceScan::ExclusiveSum` | Prefix sum | Cumulative operations, histogram equalization |
| `cub::DeviceScan::InclusiveSum` | Inclusive prefix sum | Running totals |
| `cub::DeviceRadixSort::SortKeys` | Radix sort | Top-k selection, histogram creation |
| `cub::DeviceRadixSort::SortPairs` | Sort key-value pairs | Sorting with associated data |
| `cub::DeviceSelect::If` | Compact elements by predicate | Filtering, sparse operations |
| `cub::DeviceSelect::Flagged` | Compact by flag array | Masked operations |
| `cub::DeviceSelect::Unique` | Remove duplicates | Deduplication |
| `cub::DeviceHistogram::HistogramEven` | Uniform-width histogram | Distribution analysis |
| `cub::DeviceHistogram::HistogramRange` | Custom-width bins | Non-uniform histogram |
| `cub::DeviceRunLengthEncode::Encode` | Run-length encoding | Compression, sparse representation |
| `cub::DeviceSegmentedReduce::Sum` | Per-segment reductions | Batch-wise operations |
| `cub::DeviceSegmentedSort::SortKeys` | Per-segment sorting | Batch-wise sorting |

### 8.3 CUB Block-Level Primitives

Block-level primitives operate within a single CUDA thread block using shared memory:

```cpp
// BlockReduce: Reduce within a thread block
__global__ void kernel(float* input, float* output) {
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    float val = input[threadIdx.x + blockIdx.x * blockDim.x];
    float block_sum = BlockReduce(temp_storage).Sum(val);

    if (threadIdx.x == 0) output[blockIdx.x] = block_sum;
}
```

| Primitive | Operation | Common Use |
|-----------|-----------|------------|
| `cub::BlockReduce` | Reduction within a block | Computing block-level statistics (mean, variance) |
| `cub::BlockScan` | Prefix scan within a block | Cumulative operations, attention masking |
| `cub::BlockRadixSort` | Radix sort within a block | Local top-k, histogram construction |
| `cub::BlockLoad` | Efficient block-wide load from global memory | Coalesced data loading patterns |
| `cub::BlockStore` | Efficient block-wide store to global memory | Coalesced data writing patterns |
| `cub::BlockExchange` | Rearrange data between threads | Transpose, blocked-to-striped conversion |
| `cub::BlockDiscontinuity` | Detect discontinuities in sorted data | Unique counts, segment boundaries |
| `cub::BlockHistogram` | Block-level histogram | Local distribution analysis |

**Composition:** Complex primitives compose from simpler ones. For example, `BlockRadixSort` is built from `BlockExchange` and `BlockRadixRank`, which itself uses `BlockScan`, which uses `WarpScan`.

### 8.4 CUB Warp-Level Primitives

| Primitive | Operation |
|-----------|-----------|
| `cub::WarpReduce` | Warp-wide reduction |
| `cub::WarpScan` | Warp-wide prefix scan |
| `cub::WarpExchange` | Warp-wide data exchange |
| `cub::WarpMergeSort` | Warp-wide merge sort |

### 8.5 When to Use CUB vs Custom Kernels

**Use CUB when:**
- You need standard parallel primitives (reduce, scan, sort, select)
- Performance portability across GPU architectures is important
- You want to compose within your own custom kernels using block/warp-level primitives
- The operation maps cleanly to CUB's supported patterns

**Use custom kernels when:**
- CUB's primitives don't match your access pattern
- You need fused operations that span multiple CUB primitives (the overhead of multiple calls dominates)
- You have domain-specific knowledge that enables better optimization
- Register pressure or shared memory requirements are unusual

### 8.6 Thrust: High-Level Parallel Algorithms

Thrust provides STL-like parallel algorithms with automatic backend selection:

```cpp
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/device_vector.h>

thrust::device_vector<float> d_vec(N);
// ... fill d_vec ...

// Sort
thrust::sort(d_vec.begin(), d_vec.end());

// Reduce
float sum = thrust::reduce(d_vec.begin(), d_vec.end(), 0.0f, thrust::plus<float>());

// Transform
thrust::transform(d_vec.begin(), d_vec.end(), d_out.begin(),
    [] __device__ (float x) { return x * x; });

// Scatter/Gather
thrust::scatter(d_values.begin(), d_values.end(), d_map.begin(), d_output.begin());

// Prefix sum
thrust::inclusive_scan(d_vec.begin(), d_vec.end(), d_out.begin());
```

**Key Algorithms:** `sort`, `sort_by_key`, `reduce`, `reduce_by_key`, `transform`, `transform_reduce`, `scan` (inclusive/exclusive), `scatter`, `gather`, `copy_if`, `remove_if`, `unique`, `set_operations`, `merge`, `partition`, `for_each`, `count_if`, `min_element`, `max_element`.

**When to use Thrust vs CUB:**
- Thrust for rapid prototyping and high-level operations
- CUB when you need to embed primitives inside custom kernels or need finer control
- CUB for performance-critical inner loops; Thrust for orchestration

---

## 9. cuRAND

### 9.1 Overview

cuRAND provides GPU-accelerated random number generation, critical for dropout, stochastic rounding, sampling, and weight initialization in deep learning.

### 9.2 Generators

| Generator | State Size | Quality | Speed | Use Case |
|-----------|-----------|---------|-------|----------|
| **CURAND_RNG_PSEUDO_PHILOX4_32_10** | 4x32 bits | Excellent | Fastest | **Dropout, sampling** (default for DL) |
| **CURAND_RNG_PSEUDO_XORWOW** | 5x32 bits | Good | Fast | General purpose PRNG |
| **CURAND_RNG_PSEUDO_MRG32K3A** | 6x32 bits | Very good | Moderate | Statistical simulations |
| **CURAND_RNG_PSEUDO_MT19937** | Large | Excellent | Moderate | When Mersenne Twister is required |
| **CURAND_RNG_PSEUDO_MTGP32** | Large | Excellent | Moderate | GPU-optimized Mersenne Twister |
| **CURAND_RNG_QUASI_SOBOL32** | Varies | N/A (quasi) | Fast | Monte Carlo integration |
| **CURAND_RNG_QUASI_SCRAMBLED_SOBOL32** | Varies | N/A (quasi) | Fast | Better quasi-random coverage |

### 9.3 Philox Generator (Deep Learning Standard)

Philox (Philox-4x32-10) is the preferred generator for deep learning because:

1. **Counter-based:** Stateless design -- no need to transfer state from/to global memory
2. **Parallel-friendly:** Keys can be derived from intrinsic variables (thread ID, block ID)
3. **Reproducible:** Same counter + key always produces same output
4. **Efficient:** Generates 4 random numbers per call (`curand4()`)
5. **Small state:** Only 4x32 bits for number generation

**Device API Usage:**

```c
__global__ void dropout_kernel(float* data, float* mask, int n,
                               float dropout_prob, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curandStatePhilox4_32_10_t state;
        curand_init(seed, idx, 0, &state);

        float rand_val = curand_uniform(&state);
        mask[idx] = (rand_val > dropout_prob) ? 1.0f : 0.0f;
        data[idx] *= mask[idx] / (1.0f - dropout_prob);
    }
}
```

**Efficient 4-wide generation:**

```c
// Generate 4 random floats at once (matches Philox's natural output width)
float4 rand4 = curand_uniform4(&state);
// This is more efficient than 4 separate curand_uniform() calls
```

### 9.4 Host API vs Device API

| API | Description | Use Case |
|-----|-------------|----------|
| **Host API** | `curandGenerate*()` fills device buffers from host code | Bulk generation, weight initialization |
| **Device API** | `curand_init()` + `curand()` inside kernels | Inline RNG (dropout, stochastic rounding) |

**Host API Example:**

```c
curandGenerator_t gen;
curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10);
curandSetPseudoRandomGeneratorSeed(gen, seed);
curandGenerateUniform(gen, d_output, n);  // Fill buffer with uniform [0,1)
curandGenerateNormal(gen, d_output, n, mean, stddev);  // Normal distribution
curandDestroyGenerator(gen);
```

### 9.5 Performance Considerations for Sampling Kernels

- **Philox is fastest** for device-side inline RNG in custom kernels
- **Use `curand4()` / `curand_uniform4()`** for 4-wide generation matching hardware width
- **Avoid re-initialization per element** -- initialize state once per thread, advance with subsequence/offset
- **Memory vs compute trade-off:** Host API pre-generates large buffers; device API generates on-the-fly (saves memory, adds compute)
- **Determinism:** Same seed + same configuration = deterministic output (critical for debugging)
- **cuRANDDx:** New device-side extension library with more generator options and improved performance for inline generation

---

## 10. CUDA Math Libraries Performance Comparison

### 10.1 When to Use Which Library

| Operation | Recommended Library | Why |
|-----------|-------------------|-----|
| Standard GEMM (large M, N, K) | **cuBLAS** | Best heuristics, broadest optimization |
| GEMM with custom epilogue | **cuBLASLt** (if supported epilogue) or **CUTLASS** (custom) | cuBLASLt for standard fusions; CUTLASS for custom |
| Batched GEMM (uniform size) | **cuBLAS** (`cublasGemmStridedBatchedEx`) | Optimized batched kernel dispatch |
| Grouped GEMM (varied sizes) | **CUTLASS** (`GemmGrouped`) or **cuBLAS 12.9+** | CUTLASS for full control; cuBLAS for ease |
| FP8 GEMM | **cuBLAS/cuBLASLt** or **Transformer Engine** | cuBLAS for raw GEMM; TE for training integration |
| Convolution (standard) | **cuDNN** | Best algorithm selection + autotuning |
| Fused attention (SDPA) | **cuDNN** or **FlashAttention** | cuDNN for vendor-optimized; FA for cutting-edge |
| Custom fused kernels | **Triton** or **CUTLASS** or **raw CUDA** | Triton for productivity; CUTLASS for perf |
| Sparse GEMM (2:4) | **cuSPARSELt** | Only option for hardware-accelerated 2:4 sparsity |
| Parallel primitives (sort, reduce, scan) | **CUB** (in kernels) or **Thrust** (standalone) | CUB for embedding in kernels |
| Collective communication | **NCCL** | Industry standard for multi-GPU |
| Random number generation | **cuRAND** (Philox) | Fastest GPU RNG for DL workloads |
| FFT | **cuFFT** | Only vendor-optimized GPU FFT |
| Mixed-precision training management | **Transformer Engine** | Automatic FP8 scaling and format management |

### 10.2 Decision Tree: cuBLAS vs CUTLASS vs Triton vs Custom CUDA

```
Is the operation a standard GEMM?
├── YES: Is epilogue fusion needed?
│   ├── NO: Use cuBLAS (best heuristics, zero effort)
│   ├── YES, standard (BIAS, GELU, RELU): Use cuBLASLt
│   └── YES, custom: Is the custom epilogue simple pointwise?
│       ├── YES: Use CUTLASS EVT or Triton
│       └── NO: Use CUTLASS with custom epilogue or raw CUDA
├── Is it grouped/variable-size GEMM?
│   ├── YES: Use CUTLASS GemmGrouped or cuBLAS 12.9+ grouped API
│   └── Specialized shape (very small M/N, huge K)?
│       └── Use CUTLASS GemmSplitK
├── Is it a fused multi-operation kernel (e.g., attention)?
│   ├── YES: Is it standard attention pattern?
│   │   ├── YES: Use cuDNN SDPA / FlashAttention
│   │   └── NO: Use Triton (fastest iteration) or CUTLASS (best peak perf)
│   └── Memory-bound fusion (elementwise + reduction)?
│       └── Use Triton (best productivity/performance ratio)
└── Need absolute maximum performance with full hardware control?
    └── Use raw CUDA C++ with CuTe/CUB primitives
```

### 10.3 Complexity vs Performance vs Flexibility Trade-off

| Library | Development Effort | Peak Performance | Flexibility | Hardware Portability |
|---------|-------------------|------------------|-------------|---------------------|
| cuBLAS | Lowest | Highest (for GEMM) | Lowest | Best |
| cuBLASLt | Low | High | Medium (epilogue selection) | Good |
| cuDNN | Low-Medium | Highest (for DL ops) | Medium (graph API) | Good |
| Triton | Medium | High (80-95% of peak) | High | Good (NVIDIA GPUs) |
| CUTLASS | High | Near-peak | Very High | NVIDIA only, arch-specific |
| Raw CUDA | Highest | Peak possible | Maximum | Manual per-architecture |
| Transformer Engine | Lowest (drop-in) | High (FP8) | Low (fixed patterns) | Hopper/Blackwell |

### 10.4 Key Performance Insights

1. **cuBLAS contains CUTLASS kernels internally** -- for standard GEMM, cuBLAS already selects from hundreds of CUTLASS-derived implementations via trained heuristics.

2. **CUTLASS advantage: binary size** -- cuBLAS loads all kernels; CUTLASS lets you compile only the kernels you need, reducing binary size and load time.

3. **Triton sweet spot: memory-bound fused kernels** -- for operations like fused softmax, fused layer norm, or fused attention variants, Triton provides 80-95% of handwritten CUDA performance with much less development time.

4. **cuDNN SDPA outperforms manual implementations** -- cuDNN's flash attention implementation includes architecture-specific optimizations and heuristics that are difficult to replicate.

5. **FP8 ecosystem:** Transformer Engine > cuBLASLt > manual FP8, in terms of ease of use. Performance is similar across all three when configured correctly.

6. **For inference serving:** cuBLAS/cuBLASLt for GEMM, cuDNN or FlashInfer for attention, NCCL for tensor parallelism. This is what TensorRT-LLM and vLLM use internally.

---

## Sources

- [cuBLAS 13.1 Documentation](https://docs.nvidia.com/cuda/cublas/)
- [cuBLAS 12.0 Features and Hopper Performance](https://developer.nvidia.com/blog/new-cublas-12-0-features-and-matrix-multiplication-performance-on-nvidia-hopper-gpus/)
- [cuBLAS 12.9 Performance Updates](https://developer.nvidia.com/blog/boosting-matrix-multiplication-speed-and-flexibility-with-nvidia-cublas-12-9)
- [Grouped GEMM APIs in cuBLAS](https://developer.nvidia.com/blog/introducing-grouped-gemm-apis-in-cublas-and-more-performance-updates/)
- [Accelerating Transformers with cuDNN 9](https://developer.nvidia.com/blog/accelerating-transformers-with-nvidia-cudnn-9)
- [cuDNN Graph API Documentation](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.5.0/developer/graph-api.html)
- [cuDNN Frontend Graph API](https://docs.nvidia.com/deeplearning/cudnn/frontend/v1.9.0/developer/graph-api.html)
- [cuDNN Convolution Performance Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html)
- [NCCL 2.27 Features](https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27)
- [NCCL Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [NCCL Cross Data Center Communication](https://developer.nvidia.com/blog/nccl-deep-dive-cross-data-center-communication-and-network-topology-awareness)
- [Demystifying NCCL (Research Paper)](https://arxiv.org/html/2507.04786v1)
- [Transformer Engine Documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html)
- [FP8 Primer with Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
- [Per-Tensor and Per-Block FP8 Scaling](https://developer.nvidia.com/blog/per-tensor-and-per-block-scaling-strategies-for-effective-fp8-training/)
- [Transformer Engine GitHub](https://github.com/NVIDIA/TransformerEngine)
- [cuSPARSELt Structured Sparsity](https://developer.nvidia.com/blog/exploiting-ampere-structured-sparsity-with-cusparselt/)
- [Accelerating Sparse DNNs (Whitepaper)](https://arxiv.org/pdf/2104.08378)
- [cuFFT Documentation](https://docs.nvidia.com/cuda/cufft/)
- [cuFFTMp Multi-Node FFT](https://developer.nvidia.com/blog/multinode-multi-gpu-using-nvidia-cufftmp-ffts-at-scale/)
- [CUTLASS 3.x Design Blog](https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design)
- [CUTLASS GEMM API (2.x)](https://docs.nvidia.com/cutlass/4.2.1/media/docs/cpp/gemm_api.html)
- [CUTLASS 3.0 GEMM API](https://docs.nvidia.com/cutlass/media/docs/cpp/gemm_api_3x.html)
- [CUTLASS GitHub](https://github.com/NVIDIA/cutlass)
- [CUB Documentation](https://nvidia.github.io/cccl/cub/index.html)
- [CCCL GitHub](https://github.com/NVIDIA/cccl)
- [cuRAND Device API](https://docs.nvidia.com/cuda/curand/device-api-overview.html)
- [MSCCL GitHub](https://github.com/microsoft/msccl)
- [MSCCL++ Paper](https://arxiv.org/html/2504.09014v2)
- [cuBLAS vs CUTLASS Discussion](https://github.com/NVIDIA/cutlass/issues/109)
- [TF32 Tensor Cores Blog](https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/)
- [Structured Sparsity in Ampere](https://developer.nvidia.com/blog/structured-sparsity-in-the-nvidia-ampere-architecture-and-applications-in-search-engines/)
- [MXFP8 Training Blog](https://jianyuh.github.io/mxfp8/2025/12/07/MXFP8-Train.html)
