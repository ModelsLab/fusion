# Non-NVIDIA GPU Programming & Accelerator Ecosystem Reference

> Comprehensive reference for AMD, Intel, Google TPU, Apple Silicon, Qualcomm, and cross-platform GPU/accelerator programming for AI/ML workloads.
> Last updated: 2026-03-06

---

## Table of Contents

1. [AMD GPU Architecture (CDNA/RDNA)](#1-amd-gpu-architecture-cdnardna)
2. [AMD MI300X Deep Dive](#2-amd-mi300x-deep-dive)
3. [AMD MI325X, MI350, and MI400](#3-amd-mi325x-mi350-and-mi400)
4. [ROCm Ecosystem](#4-rocm-ecosystem)
5. [Triton on AMD](#5-triton-on-amd)
6. [PyTorch on AMD](#6-pytorch-on-amd)
7. [vLLM / SGLang on AMD](#7-vllm--sglang-on-amd)
8. [Intel GPU for ML](#8-intel-gpu-for-ml)
9. [Google TPU](#9-google-tpu)
10. [Apple Silicon for ML](#10-apple-silicon-for-ml)
11. [Qualcomm / Mobile AI](#11-qualcomm--mobile-ai)
12. [Cross-Platform Kernel Development](#12-cross-platform-kernel-development)

---

## 1. AMD GPU Architecture (CDNA/RDNA)

AMD maintains two distinct GPU architecture lines for different markets:

- **CDNA** (Compute DNA): Data center AI/HPC accelerators (Instinct MI series)
- **RDNA** (Radeon DNA): Consumer/gaming GPUs (Radeon RX series)

### 1.1 CDNA 3 (MI300X/MI300A) Architecture

CDNA 3 is AMD's third-generation data center compute architecture, introduced in late 2023.

**Core Architecture:**

| Component | Specification |
|---|---|
| **Wavefront Size** | 64 work-items (Wave64) |
| **CU (Compute Unit) Structure** | 4x SIMD16 units per CU = 64 vector ops/cycle |
| **Matrix Cores per CU** | 4 Matrix Cores |
| **L1 Data Cache per CU** | 32 KB (128B cache line), 2x over MI250 |
| **L1 Instruction Cache** | 64 KB, shared by 2 CUs |
| **Scalar Cache per CU** | 16 KB |
| **L2 Cache per XCD** | 4 MB (writeback, write-allocate) |
| **Infinity Cache (MALL)** | 256 MB shared across all XCDs |
| **Supported Precisions** | FP64, FP32, TF32, BF16, FP16, FP8, INT8 |

**Wavefront Execution Model:**

A wavefront (wave) is AMD's equivalent of an NVIDIA warp. In CDNA architectures, Wave64 is the standard execution width: 64 work-items execute in lockstep across 4 SIMD16 units over 4 cycles. This contrasts with NVIDIA's 32-thread warp. Key implications:

- Kernels need more parallelism to fill wavefronts efficiently
- Occupancy calculations differ from NVIDIA (fewer concurrent wavefronts needed per CU)
- Branch divergence affects 64 threads at once, making it more costly
- Some workloads may benefit from Wave32 mode (available on RDNA, limited on CDNA)

**Matrix Core Operations:**

CDNA 3 Matrix Cores perform matrix fused multiply-add (MFMA) instructions. Key supported operations:

| Instruction | Input Type | Output Type | Matrix Shape |
|---|---|---|---|
| `v_mfma_f32_16x16x32_f16` | FP16 | FP32 | 16x16x32 |
| `v_mfma_f32_16x16x32_bf16` | BF16 | FP32 | 16x16x32 |
| `v_mfma_f32_16x16x32_fp8` | FP8 (E4M3/E5M2) | FP32 | 16x16x32 |
| `v_mfma_i32_16x16x64_i8` | INT8 | INT32 | 16x16x64 |
| `v_mfma_f64_16x16x4_f64` | FP64 | FP64 | 16x16x4 |

**Generational Matrix Core Improvements over MI250X (CDNA 2):**

| Precision | Speedup vs MI250X |
|---|---|
| FP16 / BF16 | 3x |
| INT8 | 6.8x |
| FP8 | 16x (vs FP32 baseline) |
| TF32 | 4x (vs FP32 baseline) |

### 1.2 CDNA 2 (MI250X) Architecture

CDNA 2 is the previous-generation data center architecture, powering the Frontier exascale supercomputer.

| Property | MI250X |
|---|---|
| **Die Configuration** | 2 GCDs (Graphics Compute Dies) per package |
| **Process Node** | TSMC N6 (6nm) |
| **CUs per GCD** | 110 (220 total) |
| **Stream Processors** | 14,080 (7,040 per GCD) |
| **Matrix Cores per CU** | 4 MFMA units |
| **L2 Cache per GCD** | 8 MB (16 MB total) |
| **HBM Type** | HBM2e |
| **HBM Capacity** | 128 GB (64 GB per GCD) |
| **HBM Bandwidth** | 3.2 TB/s aggregate |
| **Peak FP64** | 47.9 TFLOPS |
| **Peak FP16/BF16** | 383 TFLOPS |
| **Peak INT8** | 383 TOPS |
| **Interconnect** | Infinity Fabric (GCD-to-GCD, GPU-to-GPU) |
| **TDP** | 560W |

**CDNA 2 Key Characteristics:**
- Full-rate FP64 (64-bit ALUs throughout), critical for HPC/scientific computing
- Each GCD is essentially a separate GPU; software must manage cross-GCD communication
- The two GCDs share a package but have separate memory domains
- Infinity Fabric connects GCDs at ~200 GB/s bidirectional

### 1.3 RDNA 3 (Consumer) -- Differences from CDNA for ML

RDNA 3 (Radeon RX 7000 series) is AMD's consumer GPU architecture. Key differences from CDNA for ML:

| Feature | CDNA 3 (MI300X) | RDNA 3 (RX 7900 XTX) |
|---|---|---|
| **Primary Target** | AI/HPC compute | Gaming/graphics |
| **Wavefront Size** | Wave64 (fixed) | Wave32 (default), Wave64 supported |
| **Matrix Cores** | Full MFMA units | WMMA (Wave Matrix Multiply Accumulate) |
| **FP64 Performance** | Full rate | 1/16 rate |
| **HBM** | Yes (HBM3, 192 GB) | No (GDDR6, 24 GB) |
| **ECC Memory** | Yes | No |
| **Infinity Cache** | 256 MB | 96 MB |
| **ROCm Support** | Full, officially supported | Limited, community-driven |
| **AI Accelerators** | Matrix Cores (MFMA) | 1st-gen AI accelerators (WMMA) |
| **Supported Precisions (Matrix)** | FP64/FP32/TF32/BF16/FP16/FP8/INT8 | FP16/BF16/INT8 (WMMA only) |

**RDNA 3 ML Limitations:**
- WMMA instructions are less flexible and lower throughput than MFMA
- No HBM means much lower memory bandwidth (960 GB/s vs 5.3 TB/s)
- ROCm support is unofficial for most consumer RDNA 3 GPUs (gfx1100, gfx1101, gfx1102)
- Smaller VRAM (24 GB max) limits model sizes
- No native FP8 support in RDNA 3

### 1.4 RDNA 4 (RX 9070) -- ML Capabilities

RDNA 4 represents a significant upgrade for consumer AI capabilities:

| Feature | Specification |
|---|---|
| **Process Node** | TSMC 4nm |
| **Architecture** | RDNA 4 with 2nd-gen AI Accelerators |
| **CUs (RX 9070 XT)** | 64 |
| **Stream Processors** | 4,096 |
| **AI Accelerators** | 128 (2 per CU) |
| **VRAM** | 16 GB GDDR6 |
| **Memory Bandwidth** | 576 GB/s |
| **Ray Accelerators** | 64 |

**RDNA 4 AI Improvements:**

| Feature | RDNA 3 | RDNA 4 |
|---|---|---|
| AI Accelerator Generation | 1st gen | 2nd gen |
| FP16 Throughput per Accelerator | Baseline | Up to 4x |
| INT8 Throughput per Accelerator | Baseline | Up to 8x |
| FP8 WMMA Support | No | Yes (hardware-accelerated) |
| Sparsity Acceleration | No | Yes (for AI workloads) |

**Key RDNA 4 ML Capabilities:**
- FP8 Wave Matrix Multiply Accumulate (WMMA) for the first time on consumer AMD GPUs
- 2nd-generation AI accelerators with sparsity support enabling up to 8x INT8 throughput
- AMD FSR 4 uses ML-based upscaling trained on AMD Instinct accelerators, running on RDNA 4's FP8 WMMA
- ROCm 7.0.2+ officially supports RDNA 4 (gfx1200)

### 1.5 Infinity Cache

AMD Infinity Cache is a Memory Attached Last Level (MALL) cache present across RDNA and CDNA architectures.

**MI300X Infinity Cache Details:**

| Property | Specification |
|---|---|
| Total Capacity | 256 MB |
| Location | Distributed across 4 I/O dies (IODs) |
| Structure | 128 slices |
| Peak Bandwidth | Up to 17.2 TB/s |
| Latency | ~218 ns |
| Cache Line Size | 64B |
| Coherency | Inclusive of L2 caches |

**How Infinity Cache Works on MI300X:**
- Sits between L2 and HBM as a memory-side cache (not between compute and L2)
- Reduces HBM traffic by caching frequently accessed data
- L2 writeback/write-allocate design coalesces traffic before it reaches Infinity Cache
- Data in Infinity Cache must traverse die boundaries (IOD to XCD), adding latency vs L2
- Provides 1.6x better bandwidth than H100's L1, 3.49x better from L2, 3.12x from LLC

**Performance Impact:**
- Critical for memory-bound kernels where working set fits in 256 MB
- Especially effective for attention mechanisms with reused KV-cache data
- Less impactful for large GEMM operations that stream through HBM

### 1.6 MI300A APU -- Unified Memory Architecture

The MI300A is AMD's first CPU+GPU APU for data center AI/HPC:

| Component | Specification |
|---|---|
| **CPU** | Up to 24 Zen 4 EPYC cores (3 CCD chiplets) |
| **GPU** | 6 CDNA 3 XCDs (228 CUs) |
| **HBM** | 128 GB HBM3, 8 stacks |
| **HBM Bandwidth** | 5.3 TB/s |
| **Infinity Cache** | 256 MB (128 slices) |
| **Infinity Cache Bandwidth** | Up to 17.2 TB/s |
| **Memory Model** | Unified Physical Memory (UPM) |
| **TDP** | 760W |

**Unified Memory Architecture Details:**
- CPU and GPU share a single physical address space (Unified Physical Memory / UPM)
- No separate device DRAM; all memory is shared HBM3
- Hardware cache coherency between CPU and GPU eliminates explicit data transfers
- Zero-copy access: CPU and GPU can access same memory without `memcpy` operations
- Shared Last Level Cache (LLC) enables efficient data sharing
- Direct CPU-GPU Infinity Fabric interconnects for low-latency synchronization
- Eliminates PCIe copy overhead that plagues discrete GPU architectures

**Programming Implications:**
- `hipMalloc` and `malloc` can potentially share the same address space
- No need for `hipMemcpy` between host and device for many workloads
- Cache coherency means CPU can observe GPU writes without explicit synchronization barriers
- Ideal for workloads with frequent CPU-GPU data exchange (e.g., sparse computation, dynamic batching)

---

## 2. AMD MI300X Deep Dive

### 2.1 Chiplet Architecture

The MI300X is a massive multi-chiplet design with 12 dies in a single package:

```
┌─────────────────────────────────────────────────────────┐
│                   MI300X Package                         │
│                                                          │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐      │
│  │XCD 0│ │XCD 1│ │XCD 2│ │XCD 3│ │XCD 4│ │XCD 5│      │
│  │38CU │ │38CU │ │38CU │ │38CU │ │38CU │ │38CU │      │
│  │5nm  │ │5nm  │ │5nm  │ │5nm  │ │5nm  │ │5nm  │      │
│  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘      │
│  ┌─────┐ ┌─────┐                                        │
│  │XCD 6│ │XCD 7│     ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐│
│  │38CU │ │38CU │     │IOD 0 │ │IOD 1 │ │IOD 2 │ │IOD 3 ││
│  │5nm  │ │5nm  │     │6nm   │ │6nm   │ │6nm   │ │6nm   ││
│  └──┬──┘ └──┬──┘     │64MB  │ │64MB  │ │64MB  │ │64MB  ││
│     │       │        │IFC   │ │IFC   │ │IFC   │ │IFC   ││
│     │       │        └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘│
│     │       │           │        │        │        │     │
│  ┌──┴──┐ ┌──┴──┐ ┌──┴──┐ ┌──┴──┐ ┌──┴──┐ ┌──┴──┐ ┌──┴──┐ ┌──┴──┐│
│  │HBM3 │ │HBM3 │ │HBM3 │ │HBM3 │ │HBM3 │ │HBM3 │ │HBM3 │ │HBM3 ││
│  │24GB │ │24GB │ │24GB │ │24GB │ │24GB │ │24GB │ │24GB │ │24GB ││
│  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘│
│                                                          │
│  Total: 8 XCDs + 4 IODs + 8 HBM3 stacks                │
│  = 304 CUs, 192 GB HBM3, 256 MB Infinity Cache         │
└─────────────────────────────────────────────────────────┘
```

### 2.2 NUMA Architecture

The MI300X exposes **8 NUMA domains** (one per XCD), which has significant programming implications:

- Each XCD has its own 4 MB L2 cache and connects to a specific HBM3 stack (24 GB)
- Cross-XCD access goes through Infinity Fabric links (128 GB/s bidirectional each)
- Local HBM access: 662.4 GB/s per stack
- Remote HBM access incurs additional latency through the Infinity Fabric mesh

**NUMA-Aware Programming:**
- Memory allocation should consider XCD locality
- Kernels that access data across multiple XCDs will see reduced bandwidth
- The ROCm runtime attempts to manage NUMA placement, but explicit placement can help
- Use `hipExtMallocWithFlags` with `hipDeviceMallocUncached` for cross-XCD shared data

### 2.3 Full Specifications

| Specification | Value |
|---|---|
| **Architecture** | CDNA 3 |
| **Compute Dies (XCDs)** | 8 (TSMC 5nm) |
| **I/O Dies (IODs)** | 4 (TSMC 6nm) |
| **Total Transistors** | 153 billion |
| **Total CUs** | 304 (38 per XCD, 2 disabled per XCD for yield) |
| **Stream Processors** | 19,456 |
| **Matrix Cores** | 1,216 |
| **Peak FP64** | 81.7 TFLOPS |
| **Peak FP32** | 163.4 TFLOPS |
| **Peak FP16/BF16** | 1,307 TFLOPS |
| **Peak INT8** | 2,615 TOPS |
| **Peak FP8** | 2,615 TFLOPS |
| **HBM3 Capacity** | 192 GB (8 x 24 GB stacks) |
| **Memory Bus** | 8,192-bit |
| **Memory Data Rate** | 5.2 Gbps |
| **Peak HBM Bandwidth** | 5,325 GB/s (5.3 TB/s) |
| **L1 Cache per CU** | 32 KB |
| **L2 Cache per XCD** | 4 MB (32 MB total) |
| **Infinity Cache** | 256 MB |
| **Infinity Fabric Links** | 128 GB/s bidirectional per link |
| **PCIe** | Gen 5 x16 |
| **TDP** | 750W |

### 2.4 MI300X vs H100 for LLM Inference

| Metric | MI300X | H100 SXM |
|---|---|---|
| **HBM Capacity** | 192 GB | 80 GB |
| **HBM Bandwidth** | 5.3 TB/s | 3.35 TB/s |
| **FP16 TFLOPS** | 1,307 | 1,979 |
| **FP8 TFLOPS** | 2,615 | 3,958 |
| **L2 + LLC** | 32 MB + 256 MB | 50 MB |
| **Interconnect** | IF 128 GB/s/link | NVLink 900 GB/s total |
| **TDP** | 750W | 700W |

**Real-World LLM Inference Performance:**

- **Small batch sizes (1-4):** MI300X competitive or faster, benefiting from higher memory bandwidth for decode-bound workloads
- **Medium batch sizes (8-128):** H100 typically wins due to more mature CUDA kernel optimization, better compute utilization
- **Large batch sizes (256+):** MI300X catches up and can surpass H100, as 192 GB VRAM enables larger batches without tensor parallelism
- **Cost-effectiveness:** MI300X can be 57%+ cheaper per hour than H100, with better cost-per-token at small and large batch sizes
- **Software gap:** Despite MI300X offering 1.5x theoretical compute vs H100, real-world achieves only 37-66% of H100/H200 performance due to less mature ROCm kernel optimization
- **Key advantage:** 192 GB VRAM allows running 70B+ parameter models on a single GPU, where H100 requires 2+ GPUs

**Memory-Bound vs Compute-Bound:**
- For memory-bound workloads (autoregressive decode, small batch): MI300X has the edge (5.3 vs 3.35 TB/s)
- For compute-bound workloads (prefill, large batch GEMM): H100 has the edge (3,958 vs 2,615 FP8 TFLOPS)

---

## 3. AMD MI325X, MI350, and MI400

### 3.1 MI325X (CDNA 3 Memory Refresh)

| Property | MI300X | MI325X |
|---|---|---|
| **Architecture** | CDNA 3 | CDNA 3 (same) |
| **CUs** | 304 | 304 (same) |
| **HBM Type** | HBM3 | HBM3e |
| **HBM Capacity** | 192 GB | 256 GB (288 GB some SKUs) |
| **HBM Bandwidth** | 5.3 TB/s | 6.0 TB/s |
| **Compute (FP16)** | 1,307 TFLOPS | 1,307 TFLOPS (same) |
| **TDP** | 750W | 1,000W |
| **Release** | Q4 2023 | Q4 2024 |

**Key Takeaway:** MI325X is a drop-in memory upgrade over MI300X, using the same compute architecture but with faster, higher-capacity HBM3e. Same Universal Baseboard (UBB) server design for easy deployment alongside MI300X.

### 3.2 MI350 Series (CDNA 4)

The MI350 series represents a full architectural generation jump to CDNA 4:

| Property | MI350X | MI355X |
|---|---|---|
| **Architecture** | CDNA 4 | CDNA 4 |
| **Process** | TSMC 3nm | TSMC 3nm |
| **Transistors** | ~185 billion | ~185 billion |
| **CUs** | 256 | 320 |
| **Stream Processors** | 16,384 | 20,480 |
| **HBM Type** | HBM3e | HBM3e |
| **HBM Capacity** | 288 GB | 288 GB |
| **HBM Bandwidth** | 8 TB/s | 8 TB/s |
| **TDP** | 1,000W (air) | 1,400W (liquid) |
| **Release** | Mid 2025 | Mid 2025 |

**MI355X Peak Performance:**

| Precision | Performance |
|---|---|
| FP64 | 79 TFLOPS |
| FP16 | ~5 PFLOPS |
| FP8 | ~10 PFLOPS |
| FP6 | ~14 PFLOPS |
| FP4 | ~20 PFLOPS |

**CDNA 4 Architectural Innovations:**
- **FP4 and FP6 native support:** First AMD GPU with sub-8-bit precision for inference
- **Up to 4x peak performance vs MI300X:** Through architectural improvements + new precision formats
- **3nm process:** Higher transistor density and energy efficiency
- **2.8x faster training vs MI300X; 2.1x vs MI325X** (AMD benchmark claims)
- **ROCm 7.1+ optimizations** enhance kernel performance and communication efficiency
- **MLPerf Training v5.1:** MI350 demonstrated breakthrough performance and efficiency

### 3.3 MI400 Series (2026 Roadmap)

| Property | MI400 (Preliminary) |
|---|---|
| **Architecture** | CDNA Next |
| **HBM** | HBM4, 432 GB |
| **HBM Bandwidth** | 19.6 TB/s |
| **Peak FP4** | 40 PFLOPS |
| **Peak FP8** | 20 PFLOPS |
| **Scale-out** | 300 GB/s per GPU |
| **Rack Configuration** | Up to 72 GPUs per rack ("Helios") |
| **Scale-up Bandwidth** | 260 TB/s per rack |
| **Expected Release** | 2026 |

---

## 4. ROCm Ecosystem

### 4.1 Overview

ROCm (Radeon Open Compute) is AMD's open-source software platform for GPU-accelerated computing, analogous to NVIDIA's CUDA ecosystem.

**Stack Architecture (Bottom-Up):**

```
┌──────────────────────────────────────────────────────┐
│  Applications: PyTorch, TensorFlow, JAX, vLLM, etc. │
├──────────────────────────────────────────────────────┤
│  ML Libraries: MIOpen, rocBLAS, hipBLASLt, CK       │
├──────────────────────────────────────────────────────┤
│  HIP Runtime (CUDA-like API)                         │
├──────────────────────────────────────────────────────┤
│  ROCm Runtime (HSA, ROCr)                            │
├──────────────────────────────────────────────────────┤
│  AMDGPU Kernel Driver (Linux)                        │
├──────────────────────────────────────────────────────┤
│  AMD GPU Hardware (CDNA / RDNA)                      │
└──────────────────────────────────────────────────────┘
```

### 4.2 HIP (Heterogeneous-Compute Interface for Portability)

HIP is AMD's CUDA-like programming API. It provides:

- Near-identical API to CUDA (same function names with `hip` prefix instead of `cuda`)
- `hipMalloc`, `hipMemcpy`, `hipLaunchKernelGGL`, `hipDeviceSynchronize`, etc.
- Kernel syntax using `__global__`, `__device__`, `__shared__` qualifiers
- Same thread/block/grid execution model as CUDA
- Can compile for both AMD (via hipcc/AMDCLANG) and NVIDIA (via nvcc) backends

**HIP vs CUDA API Mapping (Selected):**

| CUDA | HIP |
|---|---|
| `cudaMalloc` | `hipMalloc` |
| `cudaMemcpy` | `hipMemcpy` |
| `cudaFree` | `hipFree` |
| `cudaDeviceSynchronize` | `hipDeviceSynchronize` |
| `cudaStream_t` | `hipStream_t` |
| `cudaEvent_t` | `hipEvent_t` |
| `__syncthreads()` | `__syncthreads()` (same) |
| `__shfl_sync` | `__shfl` (no `_sync` needed, Wave64 is implicitly sync) |
| `atomicAdd` | `atomicAdd` (same) |
| `cooperative_groups` | Limited support |

**Key Differences from CUDA:**
- Wave64 vs Warp32: All wavefront operations execute 64 threads, not 32
- No equivalent of `__ballot_sync` with mask parameter (all 64 threads participate)
- Shared memory bank conflicts: 32 banks (same as NVIDIA), but 64-thread wavefront means different access patterns
- No direct equivalent of CUDA's cooperative groups (limited support)
- No TMA (Tensor Memory Accelerator) equivalent
- No equivalent of `wgmma` (Hopper warpgroup MMA); uses MFMA instructions instead

### 4.3 HIPIFY -- CUDA to HIP Conversion

HIPIFY is AMD's tool for automatically converting CUDA source code to HIP:

**Two modes:**
1. **hipify-perl:** Perl-based text replacement (fast, handles most simple conversions)
2. **hipify-clang:** Clang-based source-to-source translation (more accurate, handles complex cases)

**What HIPIFY handles:**
- CUDA API calls to HIP equivalents
- CUDA kernel launch syntax (`<<<>>>`) to HIP equivalents
- cuBLAS/cuDNN/cuFFT calls to rocBLAS/MIOpen/rocFFT
- CUDA header includes to HIP headers
- Device intrinsics mapping

**What HIPIFY does NOT handle:**
- Warp-level programming differences (warp size 32 vs 64)
- Architecture-specific optimizations (e.g., Tensor Core code)
- PTX inline assembly (requires rewriting in GCN/CDNA assembly)
- CUDA-specific features without HIP equivalents (cooperative groups, TMA)
- Performance tuning (block sizes, shared memory usage patterns)

### 4.4 Core Libraries

#### rocBLAS
- AMD's implementation of BLAS (Basic Linear Algebra Subprograms)
- Equivalent to cuBLAS
- Optimized GEMM kernels for CDNA and RDNA architectures
- Supports FP64, FP32, FP16, BF16, INT8 data types
- Uses Tensile (AMD's GEMM code generator) for kernel selection

#### hipBLASLt
- AMD's equivalent of cuBLASLt (lightweight BLAS with extended features)
- Preferred over rocBLAS for newer GPUs and FP8 workloads
- Supports fused GEMM + epilogue operations (bias, activation, scaling)
- FP8 GEMM support (E4M3, E5M2 formats)
- hipblaslt-bench for offline GEMM tuning to find optimal kernels per problem size
- Produces "solution indices" that can be reused for production workloads

**hipBLASLt FP8 Performance (MI300X):**
- Fused FP8 rowwise scaled GEMM achieves up to 2.5x speedup over naive implementation
- Per-token activation scaling + per-channel weight scaling (PTPC-FP8)
- Known issue: performance can degrade at very large matrix dimensions (16K+)

#### MIOpen
- AMD's deep learning primitives library (equivalent to cuDNN)
- Provides optimized implementations of:
  - Convolution (forward, backward data, backward weights)
  - Pooling, BatchNorm, LayerNorm, Dropout
  - Activation functions (ReLU, Sigmoid, Tanh, etc.)
  - Softmax, RNN/LSTM
  - Fused operations (attention, layer norm, RoPE)
- Uses rocBLAS or hipBLASLt as backend for GEMM operations
- Auto-tuning: MIOpen can benchmark multiple algorithm implementations and cache the fastest
- Recently integrated into the unified ROCm/rocm-libraries repository

#### rocSPARSE
- Sparse matrix operations library (equivalent to cuSPARSE)
- SpMV, SpMM, sparse format conversions
- Supports CSR, CSC, COO, BSR, ELL formats

#### rocFFT
- FFT library (equivalent to cuFFT)
- 1D, 2D, 3D complex and real transforms
- Supports FP32, FP64, FP16

### 4.5 Composable Kernel (CK) -- AMD's CUTLASS Equivalent

The Composable Kernel library is AMD's answer to NVIDIA's CUTLASS:

**Overview:**
- C++ templated device library for building performance-critical GPU kernels
- Provides composable building blocks for GEMM, fused attention, MoE, and other operations
- Template-based design allows customization of data types, layouts, tile shapes, and fusion patterns

**CK-Tile API (Latest):**
- Vendor-optimized kernel templates for common operations
- To build a kernel: define template classes for IO tensor types, layouts, tile shapes, and flags
- Supports: GEMM, BatchGemm, Fused MHA (Multi-Head Attention), Fused MoE, SmoothQuant, element-wise kernels

**CK vs CUTLASS Comparison:**

| Feature | CK (AMD) | CUTLASS (NVIDIA) |
|---|---|---|
| Language | HIP C++ | CUDA C++ |
| Architecture | Template-based composable tiles | Template-based with 5-layer hierarchy |
| GEMM Support | Yes (multiple precisions) | Yes (FP4 through FP64) |
| Fused Attention | Yes (Flash Attention style) | Yes (via CuTe/FMHA) |
| Epilogue Fusion | Yes | Yes (CollectiveEpilogue) |
| Auto-tuning | Offline tuning support | Profile-driven tuning |
| Maturity | Growing, less documentation | Mature, extensive docs |
| Ecosystem | Used by MIOpen, vLLM-ROCm | Used by PyTorch, TensorRT |

**CK-Tile Kernel Structure:**
```
Template Parameters:
├── IO Tensor Data Types (e.g., fp16_t, bf16_t, fp8_t)
├── Tensor Layouts (row-major, col-major)
├── Workgroup-level Tile Shapes (e.g., 128x128x32)
├── Warp-level Tile Shapes (e.g., 32x32x16)
└── Global Flags (pipeline stages, epilogue type)
```

### 4.6 ROCm Version History and Compatibility

| ROCm Version | Release | Key Features |
|---|---|---|
| **5.x** | 2022-2023 | CDNA 2 (MI250X) support, initial ML framework support |
| **6.0** | Q1 2024 | CDNA 3 (MI300X) launch support, CK improvements |
| **6.1** | Q2 2024 | rocDecode (GPU video decode), expanded framework support |
| **6.2** | Q3 2024 | Radeon GPU support (RDNA 3), performance improvements |
| **6.4** | Q4 2024 | Megatron-LM fused kernels, expanded CK-Tile |
| **7.0** | Q1 2025 | Major release: MI350 support, PyTorch 2.6+ |
| **7.0.2** | Q2 2025 | RDNA 4 (RX 9070) official support, Debian 13, RHEL 10 |
| **7.1** | Q3 2025 | MI350 optimization, PyTorch 2.9 support |
| **7.1.1** | Q4 2025 | Framework updates, kernel performance improvements |
| **7.2** | Q1 2026 | JAX 0.8.0 support, continued optimization |

**GPU Architecture Support Matrix:**

| GPU Architecture | Target | ROCm Support |
|---|---|---|
| CDNA 3 (MI300X/MI300A/MI325X) | gfx940, gfx941, gfx942 | ROCm 6.0+ (full) |
| CDNA 2 (MI250X/MI250/MI210) | gfx90a | ROCm 5.0+ (full) |
| CDNA 1 (MI100) | gfx908 | ROCm 4.0+ (deprecated in 7.x) |
| CDNA 4 (MI350X/MI355X) | gfx950 | ROCm 7.0+ |
| RDNA 3 (RX 7900 XTX) | gfx1100 | ROCm 6.2+ (limited) |
| RDNA 4 (RX 9070) | gfx1200 | ROCm 7.0.2+ |

**Compatibility Notes:**
- ROCm technology preview stream will replace production stream by mid-2026
- New minor versions enable additional GPU targets at a 6-week cadence
- Linux kernel compatibility is version-specific (check compatibility matrix)
- WSL (Windows Subsystem for Linux) support available for some configurations

---

## 5. Triton on AMD

### 5.1 Architecture and Compilation Pipeline

Triton on AMD uses a different backend from NVIDIA but shares the same frontend DSL:

**Compilation Pipeline:**
```
Triton Python DSL
       │
       ▼
Triton IR (MLIR-based)
       │
       ▼
LLVM IR
       │
       ├── NVIDIA path: PTX → cubin (via NVVM)
       │
       └── AMD path: AMDGCN assembly → HSACO binary (via AMD LLVM)
```

**AMD-Specific Compiler Passes:**
- AMD has developed GPU-specific optimization passes for the Triton compiler
- These passes optimize register usage, memory access patterns, and instruction scheduling for CDNA/RDNA
- The AMD JIT compiler translates AMDGCN assembly to HSACO (Heterogeneous System Architecture Code Object)

### 5.2 Performance on AMD

**Key Results (2025):**
- Triton Flash Attention v2 achieves comparable performance to vendor-specific libraries on both NVIDIA A100 and AMD MI250, using the same kernel code
- vLLM v1 with Triton attention backend delivers 10% higher throughput on MI300X vs. vLLM v0 with custom C++/HIP kernels
- Triton is now the default attention backend in vLLM for AMD deployments
- A single Triton codebase can produce state-of-the-art performance on both NVIDIA and AMD

**Performance Comparison (Triton on AMD vs NVIDIA):**

| Kernel | AMD MI300X (Triton) | NVIDIA H100 (Triton) | Notes |
|---|---|---|---|
| Flash Attention v2 | Competitive | Baseline | Same kernel code |
| GEMM (FP16) | 85-95% of CK | 90-95% of CUTLASS | Depends on shape |
| Fused Softmax | Competitive | Baseline | Good on both |
| Quantized GEMM (FP8) | Improving | Mature | AMD catching up |

### 5.3 Compatibility and Known Issues

**What Works Well on AMD Triton:**
- Standard tiled GEMM kernels
- Flash Attention v2 (both forward and backward)
- Element-wise fused kernels (activation, normalization)
- Paged attention for inference
- Basic reductions and scans

**Known Issues and Limitations:**
- Some NVIDIA-specific Triton intrinsics (e.g., `tl.extra.cuda.*`) are not available
- Warp-level primitives differ (64-thread wavefronts vs 32-thread warps)
- Auto-tuning configurations that work on NVIDIA may not be optimal on AMD
- Some atomic operations may have different performance characteristics
- Debugging tools are less mature than NVIDIA's Nsight Compute
- `tl.dot` performance can vary significantly depending on matrix shapes and data types

**Workarounds:**
- Use AMD-specific tuning configs (different tile sizes, num_warps, num_stages)
- Triton's `TRITON_AMD_ENABLE_TUNING=1` environment variable for AMD-specific tuning
- For maximum performance, consider CK-Tile for critical kernels and Triton for everything else
- Use `triton.autotune` with AMD-appropriate parameter ranges

### 5.4 Triton-Distributed (2025)

A novel extension enabling concurrent computation and cross-GPU data transfers:
- Overlaps communication and computation within Triton kernels
- Critical for multi-GPU LLM serving where all-reduce is a bottleneck
- Works across both NVIDIA and AMD platforms

---

## 6. PyTorch on AMD

### 6.1 How It Works

PyTorch on AMD uses ROCm's HIP backend but exposes the same `torch.cuda` API:

```python
# This code works identically on NVIDIA and AMD GPUs
import torch

device = torch.device("cuda")  # Works on AMD via ROCm/HIP
x = torch.randn(1024, 1024, device=device)
y = torch.matmul(x, x.T)
```

**Under the hood:**
- `torch.cuda.*` calls are intercepted by the HIP runtime
- HIP translates CUDA-like API calls to AMD GPU operations
- PyTorch ATen operators have SYCL/HIP kernel implementations
- rocBLAS/hipBLASLt handle BLAS operations, MIOpen handles DNN primitives

### 6.2 Performance Gaps vs NVIDIA

As of early 2026, known performance gaps include:

| Area | Gap vs NVIDIA | Notes |
|---|---|---|
| **GEMM (FP16)** | 5-15% slower | hipBLASLt improving, needs per-shape tuning |
| **GEMM (FP8)** | 10-30% slower | PTPC-FP8 closes gap, but less mature |
| **Flash Attention** | 5-20% slower | CK backend competitive, Triton backend improving |
| **Compilation (torch.compile)** | 10-20% slower | Triton AMD backend overhead |
| **Custom CUDA Extensions** | Requires porting | HIPIFY can automate much of it |
| **Communication (NCCL)** | RCCL is 5-15% slower | QuickReduce helps for small messages |
| **Overall E2E Training** | 10-30% slower | Software maturity is main factor |
| **Overall E2E Inference** | 37-66% of H100 | Depends heavily on batch size and model |

### 6.3 Flash Attention on AMD

**Available Backends:**

1. **CK (Composable Kernel) Backend:**
   - Most mature and performant Flash Attention implementation on AMD
   - Flash Attention 2 support
   - Integrated into PyTorch's `F.scaled_dot_product_attention()`
   - Supports causal masks, variable sequence lengths
   - Best for production use

2. **Triton Backend:**
   - Cross-platform (same code for NVIDIA and AMD)
   - Flash Attention 2 implementation
   - Good performance, slightly below CK on AMD
   - Easier to customize and extend
   - Now default in vLLM for AMD

3. **Torch SDPA:**
   - PyTorch's built-in `scaled_dot_product_attention`
   - Works well on ROCm since PyTorch 2.3
   - Good speedup with minimal code changes
   - Automatically selects best backend

**What Does NOT Work Well:**
- Flash Attention 3 (Hopper-specific optimizations, TMA, wgmma) has no direct AMD equivalent
- Sliding window attention lacks robust Triton support on ROCm
- Flex Attention shows minimal gains on AMD at smaller scales

**Installation Challenges:**
- Often requires specific forked versions rather than `pip install flash-attn`
- Dependency hell: matching `pytorch-triton-rocm` versions with ROCm driver versions
- Docker containers are the most reliable installation method
- Manual compiler flag adjustments sometimes needed

### 6.4 Known Issues and Workarounds

**Common Issues:**
1. **HIPIFY incomplete:** Some CUDA extensions need manual porting beyond HIPIFY
2. **Memory allocator differences:** ROCm's memory allocator behaves differently; use `PYTORCH_HIP_ALLOC_CONF` to tune
3. **torch.compile instabilities:** Some models fail to compile on ROCm; use `torch.compile(backend="eager")` as fallback
4. **NCCL → RCCL:** ROCm uses RCCL (ROCm Communication Collectives Library); same API but different performance profile
5. **Random number generation:** RNG state may differ between CUDA and ROCm, affecting reproducibility

**Recommended Setup:**
```bash
# Use official ROCm Docker image for best compatibility
docker pull rocm/pytorch:latest

# Key environment variables
export PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO  # Works with RCCL
export HSA_FORCE_FINE_GRAIN_PCIE=1  # For some PCIe configurations
```

---

## 7. vLLM / SGLang on AMD

### 7.1 vLLM on AMD

**Support Status (2026):**
- vLLM has first-class AMD MI300X support
- vLLM v1 (Triton-based) is the recommended version for AMD
- Triton attention is the default backend (replacing custom C++/HIP in v0)
- PagedAttention fully optimized for MI300X
- Continuous batching works without modification

**vLLM AMD-Specific Optimizations:**
- PTPC-FP8 quantization (per-token per-channel FP8) supported since vLLM v0.7.3+
- QuickReduce: Up to 3x faster all-reduce for multi-GPU serving
- hipBLASLt tuned GEMM kernels for common LLM shapes
- ROCm 7.0+ with vLLM 0.8.5+ shows up to 2x improvement over earlier versions

**Performance Trajectory:**
- December 2025 to January 2026: AMD software improved up to 2x in vLLM performance
- vLLM v1 on MI300X delivers 10% higher throughput than vLLM v0 (which used C++/HIP kernels)
- Continuous improvement with each ROCm release

**AMD-Specific Kernel Paths in vLLM:**

| Kernel | NVIDIA Path | AMD Path |
|---|---|---|
| Attention | FlashInfer / Flash-Attn CUDA | Triton attention (default) |
| GEMM | cuBLAS / cuBLASLt | hipBLASLt |
| FP8 Quantization | CUDA kernels | Triton + hipBLASLt |
| All-Reduce | NCCL | RCCL + QuickReduce |
| Sampling | CUDA kernels | Triton kernels |
| RoPE | CUDA fused kernel | Triton fused kernel |

### 7.2 SGLang on AMD

**Support Status:**
- SGLang has official AMD MI300X support
- First-class citizen on ROCm ecosystem
- All advanced inference techniques available on AMD hardware
- RadixAttention and other SGLang innovations work on MI300X

**SGLang on MI300X Performance:**
- Competitive with vLLM for most workloads
- Benefits from same Triton kernel optimizations
- Multi-GPU scaling works via RCCL

### 7.3 Cost-Performance Analysis

| Metric | MI300X (8-GPU) | H100 SXM (8-GPU) |
|---|---|---|
| **VRAM Total** | 1.5 TB | 640 GB |
| **Cost/Hour** | ~$16-24 (cloud) | ~$24-32 (cloud) |
| **Llama-70B Single GPU** | Yes (192 GB) | No (requires 2+) |
| **Cost per M tokens (small batch)** | ~$22/M tokens | ~$28/M tokens |
| **Throughput (large batch)** | Competitive | Slightly higher |

**When MI300X Wins:**
- Large models that fit in 192 GB but not 80 GB (single GPU deployment)
- Memory-bandwidth-bound decode workloads
- Cost-sensitive deployments
- Batch sizes of 1-4 or 256+

**When H100 Wins:**
- Compute-bound prefill with medium batch sizes
- Workloads requiring mature CUDA kernel ecosystem
- Multi-GPU scaling (NVLink 900 GB/s >> Infinity Fabric)
- FP8 GEMM-heavy workloads

### 7.4 Remaining Gaps

AMD's own acknowledgment of remaining work needed:
- Upstream vLLM needs ~20 more MI300 machines, ~20 MI325 machines, and ~20 MI355X machines for CI/testing to reach CUDA-level usability
- Some model architectures may have untested code paths on AMD
- Multi-node scaling (across servers) is less tested than single-node
- Speculative decoding support on AMD is less optimized
- Vision-language model (VLM) support lags NVIDIA

---

## 8. Intel GPU for ML

### 8.1 Intel Gaudi 2 Architecture

| Property | Gaudi 2 |
|---|---|
| **Process Node** | TSMC 7nm |
| **Compute Engines** | 2 MMEs + 24 TPCs |
| **HBM** | 96 GB HBM2e |
| **HBM Bandwidth** | 2.45 TB/s |
| **Peak FP8** | ~430 TFLOPS |
| **Peak BF16** | ~430 TFLOPS |
| **Network** | 24x 100 GbE RDMA ports |
| **Data Types** | FP32, TF32, BF16, FP16, FP8 (E4M3, E5M2) |
| **TDP** | 600W |

**Compute Architecture:**
- **MME (Matrix Multiplication Engine):** Fixed-function engine for large matrix multiplications. Handles all operations that can be lowered to GEMM
- **TPC (Tensor Processor Core):** Fully programmable VLIW SIMD processor, custom-designed for deep learning. Handles element-wise ops, activations, normalization, and non-GEMM computation

### 8.2 Intel Gaudi 3 Architecture

| Property | Gaudi 3 |
|---|---|
| **Process Node** | TSMC 5nm |
| **Die Configuration** | 2 compute dies (connected by high-bandwidth bridge) |
| **MME Engines** | 8 (4 per die) |
| **TPC Engines** | 64 (32 per die) |
| **On-die SRAM** | 96 MB (48 MB per die) |
| **SRAM Bandwidth** | 12.8 TB/s |
| **HBM** | 128 GB HBM2e |
| **HBM Bandwidth** | 3.7 TB/s |
| **Peak FP8/BF16** | 1,835 TFLOPS |
| **Network** | 24x 200 GbE RDMA ports (integrated) |
| **Total Network BW** | 1,200 GB/s bidirectional |
| **Media Engines** | 14 (H.265/H.264/JPEG/VP9) |
| **TDP** | 900W (air), 1,200W (liquid) |

**Gaudi 3 vs H100 Performance:**
- BF16 TFLOPS: 1,835 vs 1,979 (Gaudi 3 is ~7% lower)
- FP8 TFLOPS: 1,835 vs 3,958 (Gaudi 3 is ~54% lower, significant gap)
- HBM capacity: 128 GB vs 80 GB (Gaudi 3 has 60% more)
- HBM bandwidth: 3.7 TB/s vs 3.35 TB/s (Gaudi 3 ~10% more)
- **Price: ~$15,000 vs ~$30,000 (Gaudi 3 is half the cost)**

**Real-World Performance:**
- LLM inference: 95-170% of H100 depending on model (Falcon 180B up to 4x faster)
- Training: Intel claims up to 50% faster for GPT-3 style models
- Sweet spot: large models where memory capacity matters, cost-sensitive deployments

### 8.3 Habana SynapseAI Software Stack

```
┌──────────────────────────────────────────────────┐
│  Frameworks: PyTorch, TensorFlow, Hugging Face   │
├──────────────────────────────────────────────────┤
│  Habana SynapseAI SDK                            │
│  ├── Graph Compiler (optimizes execution graphs)  │
│  ├── Runtime (manages device execution)           │
│  ├── Optimized Kernel Library                     │
│  ├── Collective Communication Library             │
│  └── Profiling & Debugging Tools                  │
├──────────────────────────────────────────────────┤
│  Habana Driver                                   │
├──────────────────────────────────────────────────┤
│  Gaudi Hardware (HPU)                            │
└──────────────────────────────────────────────────┘
```

**Key Software Features:**
- PyTorch programs run on Gaudi with minimal modifications (use `hpu` device)
- Graph compiler automatically optimizes execution graphs
- Integrated with Hugging Face Optimum-Habana for easy model deployment
- DeepSpeed support for distributed training
- vLLM support for LLM serving
- SynapseAI provides auto-mixed-precision support

**Programming Model:**
```python
import torch
import habana_frameworks.torch as ht

# Move model and data to HPU
model = model.to("hpu")
x = x.to("hpu")

# Run inference
with torch.no_grad():
    output = model(x)

# Or use lazy mode for graph compilation
ht.hpu.init()  # Enable lazy mode
```

### 8.4 Intel Data Center GPU Max (Ponte Vecchio)

| Property | GPU Max 1550 (2-stack) |
|---|---|
| **Architecture** | Xe-HPC |
| **Process** | Hybrid 7nm (Intel 7 + TSMC N5/N7, Foveros 3D + EMIB) |
| **Tiles in Package** | Up to 63 |
| **Xe-Cores** | 128 (2 stacks x 8 slices x 8 Xe-cores) |
| **Vector Engines per Xe-core** | 8 (512-bit SIMD, 16 FP32 ops) |
| **Matrix Engines per Xe-core** | 8 (XMX: Xe Matrix Extensions) |
| **L1 Cache/SLM per Xe-core** | 512 KB |
| **L2 Cache** | 408 MB |
| **HBM** | 128 GB HBM2e |
| **Xe-Link** | 16 links for multi-GPU |
| **Peak FP32** | ~52 TFLOPS |
| **Peak BF16/FP16** | ~840 TFLOPS |
| **Peak INT8** | ~1,680 TOPS |

**Status (2025-2026):**
- Deployed in Aurora supercomputer at Argonne National Lab
- Software support via oneAPI/SYCL
- Limited commercial adoption outside HPC
- Intel's focus has shifted more toward Gaudi for AI workloads

### 8.5 Intel ARC for Inference

Intel ARC GPUs (consumer/workstation) have growing ML support:

**ARC GPU ML Capabilities:**
- llama.cpp SYCL backend supports ARC GPUs for LLM inference
- PyTorch Intel Extension (IPEX) provides SYCL kernel implementations
- vLLM support for Intel ARC Pro B-Series GPUs (enterprise)

**ARC Pro B-Series (Workstation):**
- Large memory capacity for workstation-class GPUs
- Multi-GPU scalability for local AI model serving
- Exceptional price-to-performance for inference
- vLLM blog (Nov 2025): "Fast and Affordable LLM serving on Intel Arc Pro B-Series GPUs"

**Software Stack:**
- oneAPI DPC++/C++ Compiler with SYCL support
- Intel Extension for PyTorch (IPEX)
- oneMKL (Math Kernel Library) for optimized linear algebra
- llama.cpp SYCL backend works across Arc, Data Center GPU Max, and Flex Series

### 8.6 SYCL Programming Model

SYCL is the Khronos Group's cross-platform C++ abstraction for heterogeneous computing:

**Key Characteristics:**
- Single-source programming: host and device code in same C++17 source file
- No separate kernel language (unlike CUDA's `.cu` or OpenCL's kernel strings)
- Uses `sycl::queue`, `sycl::buffer`, and lambda-based kernels
- Portable across Intel, AMD, NVIDIA GPUs (via appropriate backends)

**SYCL vs CUDA Comparison:**

| Feature | SYCL | CUDA |
|---|---|---|
| Language | Standard C++17 | C++ with extensions |
| Portability | Multi-vendor | NVIDIA only |
| Kernel Syntax | Lambda functions | `__global__` functions |
| Memory Model | Buffers + Accessors or USM | Explicit malloc/memcpy |
| Compiler | DPC++, AdaptiveCpp | nvcc |
| Ecosystem Maturity | Growing | Dominant |

**SYCL Code Example:**
```cpp
#include <sycl/sycl.hpp>

sycl::queue q;
float* data = sycl::malloc_shared<float>(N, q);

q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
    data[i] = data[i] * 2.0f;
}).wait();
```

### 8.7 oneAPI Math Kernel Library (oneMKL)

- Cross-platform math library supporting SYCL
- Provides BLAS, LAPACK, sparse algebra, FFT, RNG
- Optimized for Intel CPUs, Intel GPUs, and (via backends) other vendors
- Used by llama.cpp's SYCL backend for GEMM operations
- Supports Intel Arc, Data Center GPU Max, and Xe2-based GPUs

---

## 9. Google TPU

### 9.1 TPU Architecture Overview

Google TPUs are custom ASICs designed specifically for neural network workloads, fundamentally different from GPUs.

**Core Architecture:**

```
┌─────────────────────────────────┐
│         TPU Chip                 │
│  ┌───────────────────────────┐  │
│  │      TensorCore           │  │
│  │  ┌─────┐ ┌─────┐         │  │
│  │  │MXU 0│ │MXU 1│ ...     │  │
│  │  └─────┘ └─────┘         │  │
│  │  ┌──────────────────────┐ │  │
│  │  │    Vector Unit (VPU)  │ │  │
│  │  └──────────────────────┘ │  │
│  │  ┌──────────────────────┐ │  │
│  │  │    Scalar Unit        │ │  │
│  │  └──────────────────────┘ │  │
│  │  ┌──────────────────────┐ │  │
│  │  │    VMEM (SRAM)        │ │  │
│  │  └──────────────────────┘ │  │
│  └───────────────────────────┘  │
│  ┌──────────┐  ┌─────────────┐  │
│  │   HBM    │  │    ICI      │  │
│  └──────────┘  └─────────────┘  │
└─────────────────────────────────┘
```

### 9.2 TPU Generations

| Spec | TPU v5e | TPU v5p | TPU v6e (Trillium) | TPU v7 (Ironwood) |
|---|---|---|---|---|
| **Release** | 2023 | 2023 | 2024-2025 | 2025 |
| **Target** | Inference/training (cost) | Training (performance) | Training/inference | Inference (primarily) |
| **MXUs per Chip** | 4 | 4 | 2 (4x larger) | TBD |
| **MXU Array Size** | 128x128 | 128x128 | 256x256 | TBD |
| **BF16 TFLOPS** | 197 | 459 | ~920 | TBD |
| **HBM Capacity** | 16 GB | 95 GB | 32 GB | 192 GB |
| **HBM Bandwidth** | ~820 GB/s | 2,800 GB/s | ~1,640 GB/s | 7,400 GB/s |
| **ICI Topology** | 2D Torus | 3D Torus | 2D Torus | 3D Torus |
| **ICI BW/Chip** | ~400 Gbps/axis | 4,800 Gbps total | ~800 Gbps/axis | 1.2 TB/s |
| **Max Pod (chips)** | 256 | 8,960 | 256 | 9,216 |
| **FP8 Support** | No | No | No | Yes (first TPU) |
| **Perf/Watt vs Prior** | Baseline | ~2x v4 | 67% more efficient | 2x vs Trillium |

### 9.3 MXU (Matrix Multiply Unit) Deep Dive

The MXU is the heart of TPU compute:

**Architecture:**
- Systolic array of multiply-accumulators
- v5e/v5p: 128x128 array = 16,384 multiply-accumulate ops per cycle
- v6e: 256x256 array = 65,536 multiply-accumulate ops per cycle (4x)
- Input precision: BF16 (primary), INT8
- Accumulation: Always FP32 (no precision loss during accumulation)
- v7 (Ironwood): First TPU with FP8 support

**How Systolic Arrays Differ from Tensor Cores:**

| Feature | TPU MXU (Systolic) | NVIDIA Tensor Core |
|---|---|---|
| Architecture | 2D systolic array | Matrix multiply-accumulate unit |
| Data Flow | Data flows through array | Data loaded per instruction |
| Array Size | 128x128 or 256x256 | 4x4 (conceptually, larger with wgmma) |
| Parallelism | Single large array per MXU | Many small units per SM |
| Flexibility | Fixed operation size | Configurable via instructions |
| Efficiency | Very high for large matrices | Good for various sizes |

**Key Implication:** TPU MXUs are most efficient when matrix dimensions are multiples of the array size (128 or 256). Undersized matrices waste compute.

### 9.4 ICI (Inter-Chip Interconnect) Topology

TPUs are designed for massive scale-out via ICI:

**Topology Types:**
- **2D Torus** (v5e, v6e): Each chip connects to 4 neighbors. Max 256 chips
- **3D Torus** (v4, v5p, v7): Each chip connects to 6 neighbors. Max 8,960-9,216 chips

**v7 (Ironwood) ICI Details:**
- 1.2 TB/s per chip bidirectional
- Optical transceivers connect to Optical Circuit Switches (OCS)
- OCS enable reconfigurable connections between 4x4x4 cubes (64 chips)
- Full Superpod: 9,216 chips with ~10 MW power
- ICI resiliency: can route around optical link faults

**ICI vs NVLink:**

| Feature | TPU ICI | NVIDIA NVLink 5 |
|---|---|---|
| Topology | Torus (2D/3D) | Full mesh (within NVL72) |
| Max Scale | 9,216 chips | 72 GPUs (single domain) |
| Per-chip BW | 1.2 TB/s (v7) | 1,800 GB/s |
| Protocol | Custom | NVLink |
| Reconfigurability | OCS for dynamic routing | Fixed |

### 9.5 JAX/XLA Programming Model

**JAX:**
- Primary framework for TPU programming
- NumPy-like API with automatic differentiation and JIT compilation
- Uses XLA (Accelerated Linear Algebra) compiler as backend
- Functional programming paradigm: pure functions, immutable arrays

**XLA Compiler:**
- Traces Python code into HLO (High Level Operations) graph
- Fuses operations into optimized kernels that saturate MXU and VPU
- Explicitly schedules data movement between HBM and VMEM (on-chip SRAM)
- Performs operation fusion, layout optimization, and memory planning
- Automatically parallelizes across TPU chips when sharding annotations are provided

**Sharding and Parallelism:**

```python
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, Mesh, PartitionSpec as P

# Define a mesh of devices
devices = jax.devices()
mesh = Mesh(devices, axis_names=('data', 'model'))

# Shard a tensor across devices
sharding = NamedSharding(mesh, P('data', 'model'))
x = jax.device_put(x, sharding)

# JAX + XLA automatically insert communication (AllGather, ReduceScatter)
```

**Parallelism Strategies:**
- **GSPMD (General and Scalable Parallelization for ML):** XLA's automatic partitioning system
- **Shardy:** GSPMD's successor (2025), improved constraint propagation and dynamic shape handling
- **shard_map:** Manual parallelism where developer writes device-local code and explicit communication
- **pjit:** Partitioned JIT compilation with sharding annotations (mostly superseded by simpler APIs)

### 9.6 TPU Programming Considerations and Limitations

**Advantages Over GPUs:**
- Compiler-centric approach: XLA provides good "out of the box" performance
- Massive scale-out (thousands of chips in a single pod)
- High ICI bandwidth enables efficient data parallelism
- BF16 is the native training precision (no mixed-precision complexity)
- Software-managed SRAM (VMEM) allows predictable performance
- Cost-effective for large-scale training (especially via Google Cloud)

**Limitations vs GPUs:**

| Limitation | Description |
|---|---|
| **Static shapes required** | JIT compilation caches by input shape; dynamic shapes cause recompilation |
| **Limited custom kernels** | No CUDA/HIP equivalent for writing arbitrary GPU kernels |
| **XLA fallbacks** | Some PyTorch ops lack XLA equivalents, falling back to CPU |
| **No dynamic control flow** | Loops with input-dependent bounds compile poorly |
| **Padding waste** | Matrix dims should be multiples of 128/256; padding wastes MXU compute |
| **No sparsity support** | No equivalent of NVIDIA's 2:4 structured sparsity |
| **FP8 only on v7** | Older TPUs limited to BF16/INT8 |
| **Debugging difficulty** | XLA's graph compilation makes step-through debugging challenging |
| **Vendor lock-in** | Only available on Google Cloud |
| **Limited framework support** | JAX is primary; PyTorch/XLA works but less mature |

**Best Practices for TPU Performance:**
1. Pad tensor dimensions to multiples of 128 (v5) or 256 (v6e)
2. Use BF16 as default training precision
3. Minimize Python-level control flow; use `jax.lax.scan` instead of Python loops
4. Annotate sharding for key tensors; let XLA propagate to intermediates
5. Use `jax.jit` liberally; avoid eager execution
6. Profile with TPU profiler and check MXU utilization
7. For variable-length sequences, bucket by length to reduce recompilation

### 9.7 TPU v7 (Ironwood) Deep Dive

Ironwood is Google's 7th-generation TPU, announced in 2025:

| Specification | Value |
|---|---|
| **Target Workload** | Inference (first inference-focused TPU) |
| **HBM Capacity** | 192 GB |
| **HBM Bandwidth** | 7.4 TB/s |
| **FP8 Support** | Yes (first TPU with FP8) |
| **Peak FP8 TFLOPS** | ~500 per chip |
| **ICI Bandwidth** | 1.2 TB/s per chip |
| **Superpod Size** | 9,216 chips |
| **Superpod Peak FP8** | 4,614 PFLOPS |
| **Superpod Power** | ~10 MW |
| **Topology** | 3D torus with OCS |
| **Perf/Watt** | 2x vs Trillium |
| **Cooling** | Liquid-cooled |

**Key Innovations:**
- First TPU with FP8 compute, enabling competitive inference performance
- 192 GB HBM matches MI300X capacity for large model deployment
- 7.4 TB/s HBM bandwidth approaches B200 levels
- Co-designed with XLA compiler for optimal performance
- Optical Circuit Switches enable dynamic topology reconfiguration

---

## 10. Apple Silicon for ML

### 10.1 Architecture Overview

Apple Silicon M-series chips use a System-on-Chip (SoC) architecture with unified memory:

```
┌─────────────────────────────────────────┐
│           Apple Silicon SoC              │
│  ┌──────────┐ ┌──────────┐ ┌─────────┐  │
│  │ CPU Cores│ │ GPU Cores│ │  ANE    │  │
│  │ (P+E)   │ │          │ │ (Neural │  │
│  │          │ │          │ │ Engine) │  │
│  └────┬─────┘ └────┬─────┘ └────┬────┘  │
│       │            │            │        │
│  ┌────┴────────────┴────────────┴────┐   │
│  │     Unified Memory (LPDDR5/5X)    │   │
│  └───────────────────────────────────┘   │
│  ┌───────────────────────────────────┐   │
│  │  Media Engine, ISP, etc.          │   │
│  └───────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

### 10.2 M-Series Chip Specifications for ML

| Spec | M4 | M4 Pro | M4 Max | M5 |
|---|---|---|---|---|
| **CPU Cores** | 10 (4P+6E) | 14 (10P+4E) | 16 (12P+4E) | 10 (4P+6E) |
| **GPU Cores** | 10 | 20 | 40 | 10 (w/ Neural Accel) |
| **Neural Engine Cores** | 16 | 16 | 16 | 16 (improved) |
| **Neural Engine TOPS** | 38 | 38 | 38 | TBD (3x+ M1) |
| **Max Unified Memory** | 32 GB | 64 GB | 128 GB | 32 GB |
| **Memory Bandwidth** | 120 GB/s | 273 GB/s | 546 GB/s | 153 GB/s |
| **Process** | TSMC 3nm (2nd gen) | TSMC 3nm | TSMC 3nm | TSMC 3nm (3rd gen) |

**M5 GPU Neural Accelerators (2025):**
- Each of the 10 GPU cores has a dedicated Neural Accelerator
- Matrix-multiplication acceleration directly in the GPU pipeline
- Over 4x peak GPU compute vs M4 for AI workloads
- Over 6x peak GPU AI compute vs M1

### 10.3 Apple Neural Engine (ANE)

The ANE is a dedicated neural network inference accelerator:

| Property | M4 ANE |
|---|---|
| **Cores** | 16 |
| **Peak TOPS** | 38 (INT8) |
| **Supported Ops** | Convolution, MatMul, pooling, normalization, activations |
| **Precision** | INT8, INT16, FP16 |
| **Access** | Via Core ML framework only |

**ANE Limitations:**
- Not directly programmable (no Metal/SYCL/OpenCL access)
- Only accessible through Core ML (Apple's ML framework)
- Limited operation support compared to GPU
- No BF16 support
- Transformer support improving but not as flexible as GPU execution
- Some models may fall back to GPU for unsupported operations

### 10.4 Metal Performance Shaders (MPS)

Metal is Apple's low-level GPU API, and MPS is the ML/compute abstraction:

**MPS for ML:**
- PyTorch MPS backend: `device = torch.device("mps")`
- Translates standard GPU operations into optimized Metal instructions
- Supports most common ML operations (matmul, convolution, attention, etc.)
- JAX experimental MPS support (via PJRT plugin)

**Metal Compute Capabilities:**
- Fine-grained GPU resource control
- Threadgroup shared memory (equivalent to CUDA shared memory)
- SIMD group operations (equivalent to warp-level primitives)
- Metal Performance Shaders for optimized primitives
- Metal 3 adds mesh shaders and hardware ray tracing

### 10.5 MLX Framework

MLX is Apple's purpose-built array framework for ML on Apple Silicon:

**Key Features:**
- NumPy-like API with automatic differentiation
- Lazy evaluation with operation fusion
- JIT compilation for Metal GPU
- Native unified memory support (zero-copy between CPU and GPU)
- Distributed training support
- Designed specifically for Apple Silicon's unified memory architecture

**MLX Performance Advantages:**
- Zero-copy tensor operations (no CPU-GPU memory transfer overhead)
- Lazy evaluation enables kernel fusion, reducing launch overhead
- Specifically optimized for Metal GPU backend
- MLX consistently exceeds llama.cpp throughput by 21-87% on Apple Silicon

**MLX vs llama.cpp on Apple Silicon:**

| Metric | MLX | llama.cpp (Metal) | Ollama |
|---|---|---|---|
| Token Generation Speed | ~230 tok/s | ~150 tok/s | ~150 tok/s |
| Advantage | Baseline | 35% slower | 35% slower |
| Memory Efficiency | Best (zero-copy) | Good | Good |
| Model Support | Growing | Extensive (GGUF) | Extensive |
| Ease of Use | Python API | CLI/API | CLI/API |

**MLX Code Example:**
```python
import mlx.core as mx
import mlx.nn as nn

# Tensors live in unified memory, accessible by CPU and GPU
x = mx.random.normal((1024, 1024))
y = mx.matmul(x, x.T)  # Runs on GPU via Metal
mx.eval(y)  # Trigger lazy evaluation
```

### 10.6 llama.cpp Metal Backend

llama.cpp's Metal backend provides LLM inference on Apple Silicon:

**Capabilities:**
- Full GPU offloading of model layers to Metal GPU
- Quantized inference: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K through Q6_K
- Flash attention support via Metal compute shaders
- Multi-GPU support (for Mac Pro with multiple GPUs)

**Performance (Approximate, M4 Max):**
- Llama 2 7B (Q4_0): ~100+ tokens/sec generation
- Llama 2 13B (Q4_0): ~50-70 tokens/sec generation
- 70B models: Fits in 128 GB unified memory (M4 Max)

**Limitations:**
- No FP8 support in Metal (limited to FP16/FP32 and integer quantization)
- Memory bandwidth is the bottleneck (546 GB/s max vs 5.3 TB/s on MI300X)
- No equivalent of Tensor Cores (except M5's Neural Accelerators)
- Single-user inference only (not designed for serving)

---

## 11. Qualcomm / Mobile AI

### 11.1 Qualcomm AI Engine Architecture

Qualcomm's AI Engine is a heterogeneous computing platform spanning multiple processors:

```
┌────────────────────────────────────────────┐
│         Snapdragon SoC                      │
│  ┌─────────────┐  ┌──────────────────────┐  │
│  │  Kryo CPU   │  │  Hexagon NPU (HTP)   │  │
│  │  (ARM cores)│  │  - Tensor accelerator │  │
│  │             │  │  - Scalar/Vector DSP  │  │
│  └──────┬──────┘  └──────────┬───────────┘  │
│         │                    │               │
│  ┌──────┴──────┐  ┌─────────┴────────────┐  │
│  │  Adreno GPU │  │  Sensing Hub DSP     │  │
│  │             │  │                      │  │
│  └──────┬──────┘  └──────────┬───────────┘  │
│         │                    │               │
│  ┌──────┴────────────────────┴───────────┐  │
│  │        Shared Memory (LPDDR5X)        │  │
│  └───────────────────────────────────────┘  │
└────────────────────────────────────────────┘
```

### 11.2 Hexagon NPU (Neural Processing Unit)

The Hexagon NPU is Qualcomm's dedicated AI accelerator:

**Snapdragon 8 Gen 3 / 8 Elite:**

| Property | Specification |
|---|---|
| **Architecture** | Hexagon Tensor Processor (HTP) |
| **Performance** | 45+ TOPS (INT8) |
| **Supported Precisions** | INT4, INT8, INT16, FP16 |
| **Key Feature** | Micro-NPU for always-on AI |
| **Power Efficiency** | 40% improved perf/watt (vs Gen 2) |
| **Prefill Speed** | 10x faster than CPU/GPU on same SoC |

**Hexagon HTP Architecture:**
- Dedicated tensor accelerator for matrix operations
- Scalar and vector DSP units for non-matrix operations
- Hardware support for weight decompression (quantized models)
- Micro-tile execution for efficient memory access

### 11.3 On-Device LLM Inference

**Current Capabilities (2025-2026):**

| Model | Device | Performance | Backend |
|---|---|---|---|
| Llama 2 7B | Snapdragon 8 Gen 3 | ~15-25 tok/s | Hexagon NPU |
| Phi-2 (2.7B) | Snapdragon 8 Gen 3 | ~30-50 tok/s | Hexagon NPU |
| Gemma 2B | Snapdragon 8 Elite | ~40-60 tok/s | Hexagon NPU |
| Llama 3.2 3B | Snapdragon 8 Elite | ~20-35 tok/s | NPU + GPU |

**Software Frameworks for Mobile AI:**
- **Qualcomm AI Engine Direct:** Low-level API for direct NPU access
- **LiteRT (formerly TFLite):** Google's mobile inference framework with Qualcomm delegate
- **ONNX Runtime Mobile:** Microsoft's cross-platform inference with Qualcomm EP
- **MLC-LLM:** Open-source LLM deployment framework with Qualcomm support
- **Qualcomm AI Hub Models:** Pre-optimized model collection for Qualcomm devices

### 11.4 Mobile Inference Considerations

**Memory Constraints:**
- Mobile devices typically have 8-16 GB shared RAM (LPDDR5X)
- LLM must fit in available memory alongside OS and apps
- 4-bit quantization (INT4/GPTQ/AWQ) essential for 7B+ models
- KV cache memory limits context length

**Power and Thermal:**
- Mobile TDP: 5-15W sustained (vs 750W for MI300X)
- Thermal throttling reduces performance over sustained inference
- NPU is 10-40x more power-efficient than CPU for AI workloads
- Battery impact is critical consideration

**Optimization Strategies:**
1. Aggressive quantization (INT4 weights, INT8 activations)
2. Speculative decoding for faster generation
3. KV cache quantization to extend context length
4. Model pruning and distillation for smaller models
5. Heterogeneous execution: NPU for compute, CPU for coordination

---

## 12. Cross-Platform Kernel Development

### 12.1 Triton as Portable Kernel Language

Triton has emerged as the leading portable GPU kernel language in 2025:

**Multi-Platform Support:**

| Platform | Backend | Status (2026) |
|---|---|---|
| NVIDIA (Ampere, Hopper, Blackwell) | PTX/cubin via NVVM | Production-ready |
| AMD (CDNA 2/3/4, RDNA 3/4) | AMDGCN via AMD LLVM | Production-ready |
| Intel (Xe HPC, Arc) | SPIR-V via Intel LLVM | Experimental |
| Qualcomm | In development | Early stage |
| RISC-V CPU | LLVM | Research/experimental |

**Triton's Portability Strengths:**
- Single codebase produces near-optimal code for multiple GPU vendors
- vLLM's Triton backend achieves state-of-the-art on both NVIDIA and AMD
- Higher-level abstraction than CUDA/HIP eliminates most vendor-specific code
- Auto-tuning adapts tile sizes and configurations per target hardware

**Triton's Limitations for Portability:**
- Performance ceiling: hand-tuned CUDA/HIP can still be 10-20% faster for critical kernels
- Vendor-specific features (TMA, wgmma) not expressible in standard Triton
- Debugging and profiling tools are less mature than CUDA's
- Not all Triton features work identically across backends

### 12.2 OpenCL Status in 2025-2026

OpenCL remains relevant but is not the future for ML:

**Current Status:**
- OpenCL 3.0 is the latest specification
- Still the broadest hardware support of any compute API
- Supported on CPUs, GPUs, DSPs, FPGAs, and dedicated accelerators
- Used in embedded and mobile compute workloads

**For ML Specifically:**
- Largely superseded by vendor-specific frameworks (CUDA, ROCm, Metal)
- No modern ML framework uses OpenCL as primary backend
- llama.cpp has an OpenCL backend (CLBlast) but it's less maintained
- Kompute (Vulkan compute framework) is preferred for new cross-platform work

**When OpenCL Still Makes Sense:**
- Embedded systems with diverse accelerators
- FPGAs and DSPs (where CUDA/Metal/ROCm don't exist)
- Legacy code maintenance
- Maximum hardware breadth (runs on virtually everything)

### 12.3 Vulkan Compute for ML

Vulkan's compute pipeline is growing in ML relevance:

**Capabilities:**
- First-class compute pipeline (not a hack like WebGL compute)
- Storage buffers for arbitrary read/write data
- Explicit GPU resource management
- Subgroup operations (similar to warp/wavefront primitives)
- Cross-platform: Windows, Linux, Android, macOS (via MoltenVK)

**Vulkan for ML:**
- **Kompute:** Vulkan compute framework for ML/data processing
- **llama.cpp Vulkan backend:** LLM inference via Vulkan compute shaders
- **ncnn:** Tencent's neural network framework with Vulkan backend
- Best for mobile/Android where CUDA isn't available

**Vulkan vs CUDA for ML:**

| Feature | Vulkan Compute | CUDA |
|---|---|---|
| Hardware Support | Any GPU vendor | NVIDIA only |
| API Complexity | Very high (verbose) | Moderate |
| Performance | 70-90% of native | Baseline (100%) |
| ML Framework Support | Limited | Dominant |
| Mobile Support | Excellent (Android) | None |
| Tensor Operations | Manual implementation | cuBLAS/cuDNN |

### 12.4 WebGPU for ML Inference

WebGPU has reached critical mass for browser-based ML:

**Browser Support (2025-2026):**

| Browser | Platform | Status |
|---|---|---|
| Chrome | Windows, macOS, Android | Shipped (v113+) |
| Firefox | Windows | Shipped (v141+), Linux in Nightly |
| Safari | macOS, iOS | Shipped (Safari 26 beta) |
| Edge | Windows, macOS | Shipped (Chromium-based) |

**WebGPU for ML:**
- Compute shaders in WGSL (WebGPU Shading Language) for GPU-accelerated inference
- Separate compute pipeline (not a rendering hack)
- Storage buffers for arbitrary data
- Explicit workgroup dispatch

**Performance:**
- 3-5x speedup over WebGL for transformer models
- 20x over multi-threaded CPU, 550x over single-threaded (Microsoft benchmarks)
- Quantized models up to ~3B parameters run well in browser
- Larger models (7B+) limited by browser memory constraints

**Production Frameworks:**
- **Transformers.js:** Hugging Face's browser ML framework with WebGPU backend
- **ONNX Runtime Web:** Microsoft's inference runtime with WebGPU execution provider
- **Web-LLM:** In-browser LLM inference using WebGPU
- **MediaPipe:** Google's on-device ML framework with WebGPU support

**WGSL Compute Shader Example:**
```wgsl
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    output[idx] = input[idx] * 2.0;
}
```

**WebGPU Limitations for ML:**
- No shared memory equivalent as flexible as CUDA's `__shared__`
- Workgroup size limits (256 invocations typically)
- No subgroup operations in all browsers yet
- 32-bit float only (no FP16/BF16 compute in WGSL)
- Memory limited by browser sandbox
- No direct access to tensor cores/matrix engines

### 12.5 SYCL as Cross-Platform Alternative

SYCL's position in the cross-platform landscape (2025-2026):

**Implementations:**
- **Intel oneAPI DPC++:** Primary SYCL implementation, supports Intel/NVIDIA/AMD GPUs
- **AdaptiveCpp (formerly hipSYCL):** Open-source SYCL for AMD, NVIDIA, Intel
- **Codeplay ComputeCpp:** Commercial SYCL (being sunset in favor of DPC++)

**SYCL for ML:**
- llama.cpp SYCL backend works across Intel Arc, Data Center GPU Max, and Flex GPUs
- PyTorch Intel Extension (IPEX) uses SYCL kernels
- DeepSpeed has SYCL kernel support for Intel GPUs
- oneMKL provides optimized BLAS/LAPACK via SYCL

**SYCL vs Other Approaches:**

| Feature | SYCL | HIP | CUDA | Triton | Vulkan |
|---|---|---|---|---|---|
| Multi-vendor | Yes | AMD (+NVIDIA via thin layer) | NVIDIA only | Yes (growing) | Yes |
| Language | C++17 | C++ (CUDA-like) | C++ extensions | Python DSL | C/Vulkan API |
| Abstraction | High | Medium | Medium | Very High | Very Low |
| ML Ecosystem | Growing | Growing | Dominant | Growing fast | Limited |
| Performance | Good | Good | Best | Good | Good |
| Kernel Complexity | Medium | Medium | Medium | Low | Very High |

### 12.6 Practical Cross-Platform Strategy (2026)

**Recommended Approach for Cross-Platform ML Kernels:**

1. **Use Triton for most custom kernels:**
   - Write once, runs on NVIDIA and AMD (Intel support improving)
   - Sufficient performance for 80%+ of use cases
   - Lower development cost than vendor-specific kernels

2. **Fall back to vendor-specific for critical paths:**
   - CUDA/CUTLASS for NVIDIA-specific optimizations (TMA, wgmma)
   - CK-Tile/HIP for AMD-specific optimizations (MFMA tuning)
   - SYCL for Intel GPU paths

3. **Use framework-level abstractions where possible:**
   - PyTorch's `torch.compile` handles most operations
   - `F.scaled_dot_product_attention` routes to best backend per hardware
   - vLLM/SGLang handle kernel selection automatically

4. **For mobile/edge:**
   - Vulkan compute for Android
   - Metal for iOS/macOS
   - WebGPU for browser deployment
   - Qualcomm AI Engine Direct for Snapdragon NPU

---

## Sources

### AMD Architecture & Hardware
- [AMD CDNA 3 White Paper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf)
- [AMD MI300 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
- [AMD MI300X Hot Chips 2024](https://hc2024.hotchips.org/assets/program/conference/day1/23_HC2024.AMD.MI300X.ASmith(MI300X).v1.Final.20240817.pdf)
- [AMD MI300X Product Page](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)
- [AMD MI300X Data Sheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf)
- [AMD MI300A Data Sheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300a-data-sheet.pdf)
- [AMD MI350 Series Blog](https://www.amd.com/en/blogs/2025/amd-instinct-mi350-series-game-changer.html)
- [AMD MI350 Product Page](https://www.amd.com/en/products/accelerators/instinct/mi350.html)
- [AMD RDNA 4 Launch Press Release](https://www.amd.com/en/newsroom/press-releases/2025-2-28-amd-unveils-next-generation-amd-rdna-4-architectu.html)
- [AMD MI300 Series Microarchitecture (ROCm Docs)](https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi300.html)
- [Chips and Cheese: AMD CDNA 3 Architecture](https://chipsandcheese.com/p/amds-cdna-3-compute-architecture)
- [Chips and Cheese: Testing AMD MI300X](https://chipsandcheese.com/p/testing-amds-giant-mi300x)
- [Chips and Cheese: MI300A Memory Subsystem](https://chipsandcheese.com/p/inside-the-amd-radeon-instinct-mi300as)
- [Dissecting CPU-GPU Unified Physical Memory on MI300A (arXiv)](https://arxiv.org/abs/2508.12743)

### ROCm Ecosystem
- [ROCm Documentation](https://rocm.docs.amd.com/en/latest/)
- [ROCm Compatibility Matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)
- [ROCm 7.2.0 Release Notes](https://rocm.docs.amd.com/en/latest/about/release-notes.html)
- [Composable Kernel Documentation](https://rocm.docs.amd.com/projects/composable_kernel/en/docs-6.0.2/what-is-ck.html)
- [CK-Tile GEMM Blog](https://rocm.blogs.amd.com/software-tools-optimization/building-efficient-gemm-kernels-with-ck-tile-vendo/README.html)
- [hipBLASLt GEMM Tuning](https://rocm.blogs.amd.com/software-tools-optimization/hipblaslt-offline-tuning-part1/README.html)
- [ROCm Revisited: Ecosystem Evolution](https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-revisited-ecosy/README.html)

### Triton on AMD
- [Triton Kernel Optimizations on AMD (ROCm Blog)](https://rocm.blogs.amd.com/software-tools-optimization/kernel-development-optimizations-with-triton-on-/README.html)
- [Enabling vLLM V1 on AMD GPUs with Triton (PyTorch Blog)](https://pytorch.org/blog/enabling-vllm-v1-on-amd-gpus-with-triton/)
- [Triton Developer Hub (ROCm)](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/gpu_dev_optimize/triton_kernel_dev.html)
- [Anatomy of a Triton Attention Kernel (arXiv)](https://arxiv.org/abs/2511.11581)
- [vllm-triton-backend (IBM Research)](https://research.ibm.com/publications/vllm-triton-backend-how-to-get-state-of-the-art-performance-on-nvidia-and-amd-with-just-triton)
- [Triton-Distributed (ROCm Blog)](https://rocm.blogs.amd.com/software-tools-optimization/triton-distributed-c/README.html)

### PyTorch & vLLM on AMD
- [Flash Attention on AMD (ROCm Blog)](https://rocm.blogs.amd.com/artificial-intelligence/flash-attention/README.html)
- [State of Flash Attention on ROCm](https://zdtech.substack.com/p/the-state-of-flash-attention-on-rocm)
- [PTPC-FP8 on ROCm (vLLM Blog)](https://blog.vllm.ai/2025/02/24/ptpc-fp8-rocm.html)
- [QuickReduce for vLLM/SGLang (ROCm Blog)](https://rocm.blogs.amd.com/artificial-intelligence/quick-reduce/README.html)
- [SGLang on MI300X (Stephen Diehl)](https://www.stephendiehl.com/posts/sglang_mI300x/)
- [AMD vs NVIDIA Inference Benchmark (SemiAnalysis)](https://newsletter.semianalysis.com/p/amd-vs-nvidia-inference-benchmark-who-wins-performance-cost-per-million-tokens)

### Intel
- [Intel Gaudi 2 Processor](https://www.intel.com/content/www/us/en/developer/articles/technical/habana-gaudi2-processor-for-deep-learning.html)
- [Gaudi Architecture Documentation](https://docs.habana.ai/en/latest/Gaudi_Overview/Gaudi_Architecture.html)
- [Intel Gaudi 3 Performance Positioning](https://cdrdv2-public.intel.com/854746/gaudi-3-ai-accelerator-performance-and-positioning.pdf)
- [Intel Xe GPU Architecture](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/intel-xe-gpu-architecture.html)
- [vLLM on Intel Arc Pro B-Series](https://blog.vllm.ai/2025/11/11/intel-arc-pro-b.html)
- [llama.cpp SYCL Backend](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md)
- [oneAPI Overview](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html)

### Google TPU
- [TPU v5e Documentation](https://docs.cloud.google.com/tpu/docs/v5e)
- [TPU v5p Documentation](https://docs.cloud.google.com/tpu/docs/v5p)
- [TPU v6e Documentation](https://docs.cloud.google.com/tpu/docs/v6e)
- [TPU v7 (Ironwood) Documentation](https://docs.cloud.google.com/tpu/docs/tpu7x)
- [Ironwood Blog Post](https://blog.google/products/google-cloud/ironwood-tpu-age-of-inference/)
- [TPU System Architecture](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm)
- [Introducing Trillium (6th Gen TPUs)](https://cloud.google.com/blog/products/compute/introducing-trillium-6th-gen-tpus)
- [Inside Ironwood TPU AI Stack](https://cloud.google.com/blog/products/compute/inside-the-ironwood-tpu-codesigned-ai-stack)
- [How to Think About TPUs (JAX Scaling Book)](https://jax-ml.github.io/scaling-book/tpus/)
- [Programming TPUs in JAX](https://jax-ml.github.io/scaling-book/jax-stuff/)
- [Cloud TPU Performance Guide](https://docs.cloud.google.com/tpu/docs/performance-guide)

### Apple Silicon
- [Apple M4 Introduction](https://www.apple.com/newsroom/2024/05/apple-introduces-m4-chip/)
- [Apple M4 Pro and M4 Max](https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/)
- [Apple M5 Announcement](https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/)
- [MLX Framework (GitHub)](https://github.com/ml-explore/mlx)
- [Exploring LLMs with MLX and M5 Neural Accelerators](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
- [Deploying Transformers on Apple Neural Engine](https://machinelearning.apple.com/research/neural-engine-transformers)
- [Production-Grade Local LLM Inference on Apple Silicon (arXiv)](https://arxiv.org/abs/2511.05502)
- [Inside M4 Apple Neural Engine Benchmarks](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615)

### Qualcomm / Mobile AI
- [Qualcomm On-Device Gen AI White Paper](https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/Unlocking-on-device-generative-AI-with-an-NPU-and-heterogeneous-computing.pdf)
- [Qualcomm AI Hub Models (GitHub)](https://github.com/qualcomm/ai-hub-models)
- [Qualcomm Hexagon AI Accelerators (The Chip Letter)](https://thechipletter.substack.com/p/qualcomms-hexagon-ai-accelerators)
- [Scaling LLM Test-Time Compute with Mobile NPU (arXiv)](https://arxiv.org/abs/2509.23324)
- [LiteRT on Qualcomm NPU](https://ai.google.dev/edge/litert/android/npu/qualcomm)

### Cross-Platform
- [SYCL vs OpenCL vs Vulkan Compute](https://tillcode.com/sycl-vs-opencl-vs-vulkan-compute-cross-platform-gpu-api/)
- [Choosing GPU Compute APIs (TechnoLynx)](https://www.technolynx.com/post/choosing-vulkan-opencl-sycl-or-cuda-for-gpu-compute)
- [OpenCL Standard (Khronos)](https://www.khronos.org/opencl/)
- [SYCL Wikipedia](https://en.wikipedia.org/wiki/SYCL)
- [WebGPU Browser AI (SitePoint)](https://www.sitepoint.com/webgpu-browser-based-ai-future/)
- [WebGPU for On-Device AI Inference](https://makitsol.com/webgpu-for-on-device-ai-inference/)
- [WebGPU All Major Browsers](https://www.webgpu.com/news/webgpu-hits-critical-mass-all-major-browsers/)
- [Triton Language (GitHub)](https://github.com/triton-lang/triton)
- [TritonForge: Cross-Platform Kernel Synthesis](https://github.com/RLsys-Foundation/TritonForge)
