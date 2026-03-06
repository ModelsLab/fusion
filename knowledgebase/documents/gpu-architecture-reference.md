# Comprehensive GPU Architecture Reference

> A detailed technical reference for GPU architectures relevant to AI model optimization.
> Last updated: 2026-03-06

---

## Table of Contents

1. [NVIDIA Ampere (GA100 / GA102 / GA104)](#1-nvidia-ampere-ga100--ga102--ga104)
2. [NVIDIA Hopper (GH100)](#2-nvidia-hopper-gh100)
3. [NVIDIA Ada Lovelace (AD102 / AD103 / AD104)](#3-nvidia-ada-lovelace-ad102--ad103--ad104)
4. [NVIDIA Blackwell (GB100 / GB202)](#4-nvidia-blackwell-gb100--gb202)
5. [NVIDIA Rubin (Next-Gen)](#5-nvidia-rubin-next-gen)
6. [AMD Instinct MI300X / MI300A](#6-amd-instinct-mi300x--mi300a)
7. [AMD Instinct MI325X and MI350](#7-amd-instinct-mi325x-and-mi350)
8. [Intel Gaudi 3](#8-intel-gaudi-3)
9. [Google TPU v5e / v5p / v6e (Trillium)](#9-google-tpu-v5e--v5p--v6e-trillium)

---

## 1. NVIDIA Ampere (GA100 / GA102 / GA104)

### Overview

| Property | GA100 (A100) | GA102 (RTX 3090) | GA104 (RTX 3070) |
|---|---|---|---|
| **Process Node** | TSMC N7 (7nm) | Samsung 8nm (8N) | Samsung 8nm (8N) |
| **Transistors** | 54.2 billion | 28.3 billion | 17.4 billion |
| **Die Size** | 826 mm^2 | 628.4 mm^2 | 392.5 mm^2 |
| **Compute Capability** | 8.0 | 8.6 | 8.6 |

### SM Structure

**GA100 (Data Center):**
- 8 GPCs, 16 SMs per GPC = 128 SMs (full die), 108 SMs enabled (A100)
- Each SM: 64 FP32 CUDA cores, 32 FP64 cores, 4 third-gen Tensor Cores
- Total CUDA cores (A100): 6,912 FP32, 3,456 FP64
- Total Tensor Cores (A100): 432

**GA102 (Consumer/Pro):**
- 7 GPCs, 12 SMs per GPC = 84 SMs (full die), 82 enabled (RTX 3090)
- Each SM: 128 FP32 CUDA cores (dual FP32 datapaths), 4 third-gen Tensor Cores
- Total CUDA cores (RTX 3090): 10,496
- Total Tensor Cores (RTX 3090): 328

**GA104 (Consumer):**
- 6 GPCs, 8 SMs per GPC = 48 SMs (full die), 46 enabled (RTX 3070 Ti)
- Each SM: 128 FP32 CUDA cores, 4 third-gen Tensor Cores
- Total CUDA cores (RTX 3070 Ti): 6,144

### Tensor Core Specifications (Third Generation)

**GA100 Tensor Core Throughput per SM per clock:**
- FP16/BF16: 256 FMA ops
- TF32: 128 FMA ops
- FP64: 128 FMA ops
- INT8: 512 ops
- INT4: 1024 ops

**A100 Peak Tensor Core Performance (SXM, 80GB):**

| Precision | Dense (TFLOPS) | Sparse (TFLOPS) |
|---|---|---|
| FP64 | 19.5 | N/A |
| TF32 | 156 | 312 |
| BF16 | 312 | 624 |
| FP16 | 312 | 624 |
| INT8 | 624 (TOPS) | 1,248 (TOPS) |
| INT4 | 1,248 (TOPS) | 2,496 (TOPS) |
| FP32 (non-TC) | 19.5 | N/A |

### Memory Hierarchy

**GA100 (A100 SXM 80GB):**

| Component | Specification |
|---|---|
| HBM Type | HBM2e |
| Capacity | 80 GB |
| Bus Width | 5120-bit |
| Bandwidth | 2,039 GB/s |
| L2 Cache | 40 MB |
| L1/Shared Memory per SM | 192 KB combined (configurable) |
| Max Shared Memory per SM | 164 KB |
| Register File per SM | 256 KB |
| NVLink | 3rd-gen, 12 links, 600 GB/s bidirectional |
| PCIe | Gen 4 x16 |
| TDP (SXM) | 400W |
| TDP (PCIe) | 250W / 300W |

**GA102 (RTX 3090):**

| Component | Specification |
|---|---|
| Memory Type | GDDR6X |
| Capacity | 24 GB |
| Bus Width | 384-bit |
| Bandwidth | 936 GB/s |
| L2 Cache | 6 MB |
| L1/Shared Memory per SM | 128 KB combined |
| Register File per SM | 256 KB |
| NVLink | 3rd-gen, 1 link (RTX 3090 only) |
| PCIe | Gen 4 x16 |
| TDP | 350W |

### Key Architectural Innovations

- **TF32 (TensorFloat-32):** New 19-bit format (8-bit exponent of FP32, 10-bit mantissa of FP16) that accelerates FP32 data through Tensor Cores transparently, delivering 8x throughput over FP32
- **Fine-Grained Structured Sparsity:** 2:4 pattern (2 values must be zero in every group of 4), hardware-accelerated pruning that doubles Tensor Core throughput for supported precisions
- **Asynchronous Memory Copy (cp.async):** Hardware-accelerated data movement from global memory to shared memory, bypassing register file, enabling overlap of compute and data movement
- **Third-Gen NVLink:** 600 GB/s bidirectional per GPU (A100), 12 links
- **Multi-Instance GPU (MIG):** Partitions A100 into up to 7 isolated GPU instances with dedicated memory bandwidth, cache, and compute
- **BF16 Tensor Core Support:** First NVIDIA GPU with native BF16 Tensor Core support at same throughput as FP16

### Warp Scheduling

- 32 threads per warp (unchanged)
- 4 warp schedulers per SM (GA100), each managing up to 16 warps = 64 warps max per SM (GA100) / 48 warps (GA102)
- Each scheduler dispatches one instruction per clock cycle
- Register file: 65,536 32-bit registers per SM

---

## 2. NVIDIA Hopper (GH100)

### Overview

| Property | GH100 (H100) |
|---|---|
| **Process Node** | TSMC 4N (custom 4nm) |
| **Transistors** | 80 billion |
| **Die Size** | 814 mm^2 |
| **Compute Capability** | 9.0 |

### SM Structure

- 8 GPCs, 9 TPCs per GPC, 2 SMs per TPC = 144 SMs (full die)
- H100 SXM5: 132 SMs enabled
- H100 PCIe: 114 SMs enabled
- Each SM: 128 FP32 CUDA cores, 64 FP64 cores, 4 fourth-gen Tensor Cores
- Total CUDA cores (H100 SXM): 16,896 FP32, 8,448 FP64 (full die: 18,432 / 9,216)
- Total Tensor Cores (H100 SXM): 528 (full die: 576)

### Tensor Core Specifications (Fourth Generation)

**H100 SXM5 Peak Tensor Core Performance:**

| Precision | Dense (TFLOPS) | Sparse (TFLOPS) |
|---|---|---|
| FP64 | 67 | N/A |
| TF32 | 989 | 1,979 |
| BF16 | 1,979 | 3,958 |
| FP16 | 1,979 | 3,958 |
| FP8 | 3,958 | 7,916 |
| INT8 | 3,958 (TOPS) | 7,916 (TOPS) |
| FP32 (non-TC) | 67 | N/A |

**H100 PCIe Peak Tensor Core Performance:**

| Precision | Dense (TFLOPS) | Sparse (TFLOPS) |
|---|---|---|
| FP64 | 51 | N/A |
| TF32 | 756 | 1,513 |
| BF16 | 1,513 | 3,026 |
| FP16 | 1,513 | 3,026 |
| FP8 | 3,026 | 6,052 |
| INT8 | 3,026 (TOPS) | 6,052 (TOPS) |

### Memory Hierarchy

| Component | H100 SXM5 | H100 PCIe |
|---|---|---|
| HBM Type | HBM3 | HBM2e |
| Capacity | 80 GB | 80 GB |
| Bus Width | 5120-bit | 5120-bit |
| Bandwidth | 3,352 GB/s | 2,039 GB/s |
| L2 Cache | 50 MB | 50 MB |
| L1/Shared Memory per SM | 256 KB combined | 256 KB combined |
| Max Shared Memory per SM | 228 KB | 228 KB |
| Register File per SM | 256 KB | 256 KB |
| NVLink | 4th-gen, 18 links, 900 GB/s | 4th-gen, 18 links, 900 GB/s |
| PCIe | Gen 5 x16 | Gen 5 x16 |
| TDP | 700W | 350W |

### Key Architectural Innovations

- **FP8 Tensor Cores:** Native support for E4M3 and E5M2 FP8 formats, delivering 2x throughput over FP16 per SM, 4x over A100 FP16
- **Transformer Engine:** Hardware-managed dynamic precision switching between FP8 and FP16 on a per-tensor basis. Automatically selects precision per layer, maintaining accuracy while maximizing throughput
- **Tensor Memory Accelerator (TMA):** Dedicated hardware unit for asynchronous bulk data transfers of up to 5D tensors between global and shared memory. Supports element-wise reductions (add/min/max) during transfer. Frees warps from address calculation overhead
- **Thread Block Clusters:** New programming hierarchy level enabling multiple thread blocks across multiple SMs to synchronize and share data through Distributed Shared Memory. Extends CUDA hierarchy: threads > thread blocks > clusters > grids
- **Distributed Shared Memory (DSMEM):** Allows thread blocks within a cluster to directly access shared memory of other SMs via hardware-supported load/store/atomics, without going through global memory
- **DPX Instructions:** New instructions accelerating dynamic programming algorithms (e.g., Smith-Waterman, Floyd-Warshall, Viterbi) by up to 7x over A100. Hardware-accelerated fused comparison/selection operations
- **Asynchronous Execution:** Extended async capabilities including TMA, thread block cluster barrier operations, and async transaction barriers
- **Fourth-Gen NVLink:** 900 GB/s bidirectional per GPU, 18 links. NVLink Switch System enables up to 256 GPUs in a single NVLink domain

### Warp Scheduling

- 32 threads per warp
- 4 warp schedulers per SM, each managing a set of warps
- Max 64 warps per SM
- Enhanced warp scheduling with support for asynchronous warp specialization (producer/consumer warp pattern)
- New wgmma (warpgroup matrix multiply-accumulate) instruction operates on groups of 4 warps (128 threads)

---

## 3. NVIDIA Ada Lovelace (AD102 / AD103 / AD104)

### Overview

| Property | AD102 | AD103 | AD104 |
|---|---|---|---|
| **Process Node** | TSMC 4N | TSMC 4N | TSMC 4N |
| **Transistors** | 76.3 billion | 45.9 billion | 35.8 billion |
| **Die Size** | 608.4 mm^2 | 378.6 mm^2 | 294.5 mm^2 |
| **Compute Capability** | 8.9 | 8.9 | 8.9 |

### SM Structure

- **AD102:** 12 GPCs, 12 SMs per GPC = 144 SMs (full die), 128 enabled (RTX 4090)
- **AD103:** 7 GPCs = 80 SMs (full die), 76 enabled (RTX 4080)
- **AD104:** 5 GPCs = 60 SMs (full die), 58 enabled (RTX 4070 Ti)
- Each SM: 128 FP32 CUDA cores, 4 fourth-gen Tensor Cores, 1 third-gen RT Core
- Total CUDA cores (RTX 4090): 16,384
- Total Tensor Cores (RTX 4090): 512

### Tensor Core Specifications (Fourth Generation - Consumer)

**RTX 4090 (AD102) Peak Tensor Core Performance:**

| Precision | Dense (TFLOPS) | With Sparsity (TFLOPS) |
|---|---|---|
| TF32 | 82.6 | 165.2 |
| BF16 | 165.2 | 330.3 |
| FP16 | 165.2 | 330.3 |
| FP8 | 330.3 | 660.6 |
| INT8 | 660.6 (TOPS) | 1,321.2 (TOPS) |
| FP32 (non-TC) | 82.6 | N/A |

### Memory Hierarchy

| Component | RTX 4090 (AD102) | RTX 4080 (AD103) | RTX 4070 Ti (AD104) |
|---|---|---|---|
| Memory Type | GDDR6X | GDDR6X | GDDR6X |
| Capacity | 24 GB | 16 GB | 12 GB |
| Bus Width | 384-bit | 256-bit | 192-bit |
| Bandwidth | 1,008 GB/s | 717 GB/s | 504 GB/s |
| L2 Cache | 96 MB | 64 MB | 48 MB |
| L1/Shared per SM | 128 KB combined | 128 KB combined | 128 KB combined |
| Max Shared per SM | 100 KB | 100 KB | 100 KB |
| Register File per SM | 256 KB | 256 KB | 256 KB |
| PCIe | Gen 4 x16 | Gen 4 x16 | Gen 4 x16 |
| TDP | 450W | 320W | 285W |

### Key Architectural Innovations

- **Shader Execution Reordering (SER):** Dynamically reorders ray tracing shader workloads for better execution coherence. Reschedules divergent shader work on-the-fly, improving ray tracing performance by up to 3x in certain scenarios. Analogous to out-of-order execution for shaders
- **Fourth-Gen Tensor Cores (Consumer):** FP8 support (first in consumer GPUs), same architecture as Hopper but at consumer clocks. Supports all Hopper precisions: FP8, FP16, BF16, TF32, INT8, plus sparsity
- **Massively Expanded L2 Cache:** Up to 96 MB L2 (AD102) vs. 6 MB (GA102), a 16x increase. Dramatically reduces DRAM bandwidth pressure for inference workloads
- **Optical Flow Accelerator (OFA):** Hardware-accelerated optical flow estimation for DLSS 3 frame generation
- **DLSS 3 Frame Generation:** AI-powered frame generation using optical flow and neural networks
- **Third-Gen RT Cores:** 2x ray-triangle intersection throughput, Opacity Micro-Map (OMM) engine, Displaced Micro-Mesh (DMM) engine
- **AV1 Dual Encode/Decode:** Two NVENC encoders with AV1 support
- **High Clock Speeds:** Boost clocks up to 2.52 GHz (RTX 4090), enabled by TSMC 4N process efficiency

---

## 4. NVIDIA Blackwell (GB100 / GB202)

### Overview

| Property | GB100 (Data Center) | GB202 (Consumer) |
|---|---|---|
| **Process Node** | TSMC 4NP (custom) | TSMC 4N |
| **Transistors** | 104B per die (208B dual-die) | 92.2 billion |
| **Die Size** | ~800 mm^2 per die (dual) | 744 mm^2 |
| **Compute Capability** | 10.0 | 12.0 |

### SM Structure

**GB100 (Data Center - B100/B200/B300):**
- Dual-die design: two GB100 dies connected via 10 TB/s chip-to-chip interconnect
- 148 SMs per die, 296 total (full dual-die)
- B200: ~296 SMs enabled; B100: ~280 SMs
- Each SM: 128 FP32 CUDA cores, 4 fifth-gen Tensor Cores
- Total CUDA cores (B200): ~37,888 (full die)
- Total Tensor Cores: ~592 per die

**GB202 (Consumer - RTX 5090):**
- 192 SMs (full die), 170 SMs enabled (RTX 5090)
- Each SM: 128 FP32 CUDA cores, 4 fifth-gen Tensor Cores
- Total CUDA cores (RTX 5090): 21,760
- Total Tensor Cores: 680

### Tensor Core Specifications (Fifth Generation)

**B200 Peak Performance (Data Center):**

| Precision | Performance |
|---|---|
| FP4 | 20 PFLOPS (sparse: 40 PFLOPS) |
| FP6 | ~14 PFLOPS |
| FP8 | 10 PFLOPS (sparse: 20 PFLOPS) |
| FP16 / BF16 | 5 PFLOPS (sparse: 10 PFLOPS) |
| TF32 | 2.5 PFLOPS (sparse: 5 PFLOPS) |
| FP64 | 45 TFLOPS |
| FP32 (non-TC) | ~90 TFLOPS |

**B100 Peak Performance (Data Center):**

| Precision | Performance |
|---|---|
| FP4 | 18 PFLOPS |
| FP8 | 9 PFLOPS |
| FP16 / BF16 | 4.5 PFLOPS |
| TF32 | 2.25 PFLOPS |
| FP64 | 40 TFLOPS |

### Memory Hierarchy

**Data Center (B100/B200/B300):**

| Component | B100 | B200 | B300 |
|---|---|---|---|
| HBM Type | HBM3e | HBM3e | HBM3e |
| Capacity | 192 GB | 192 GB | 288 GB |
| Bandwidth | 8 TB/s | 8 TB/s | 8 TB/s |
| L2 Cache | ~192 MB (combined) | ~192 MB | ~192 MB |
| NVLink | 5th-gen, 1.8 TB/s | 5th-gen, 1.8 TB/s | 5th-gen, 1.8 TB/s |
| PCIe | Gen 6 | Gen 6 | Gen 6 |
| TDP | 700W | 1,000W | 1,100W |

**Consumer (RTX 5090 / GB202):**

| Component | RTX 5090 |
|---|---|
| Memory Type | GDDR7 |
| Capacity | 32 GB |
| Bus Width | 512-bit |
| Bandwidth | 1,792 GB/s |
| L2 Cache | 98 MB |
| L1/Shared per SM | 128 KB combined |
| PCIe | Gen 5 x16 |
| TDP | 575W |

### Key Architectural Innovations

- **Dual-Die Architecture (Data Center):** Two GB100 dies connected by 10 TB/s NVLink-C2C chip-to-chip interconnect, presenting as a single unified GPU to software
- **Fifth-Gen Tensor Cores:** Native FP4 and FP6 support (MXFP4/MXFP6 microscaling formats per OCP spec), delivering 2x throughput over FP8
- **Second-Gen Transformer Engine:** Micro-tensor scaling with per-block dynamic precision adjustment at sub-tensor granularity. Enables FP4 inference with near-FP8 accuracy. Dynamically selects between FP4, FP6, and FP8 per tensor block
- **NVLink 5:** 1.8 TB/s bidirectional per GPU. NVLink Switch supports up to 576-GPU single NVLink domain. NVL72 rack: 130 TB/s aggregate GPU bandwidth
- **Decompression Engine:** Hardware-accelerated database decompression supporting LZ4, Snappy, and Deflate formats at line rate
- **Confidential Computing:** First GPU with full TEE-I/O (Trusted Execution Environment with I/O) capability. Near-zero performance overhead for encrypted/protected workloads
- **RAS Engine:** Dedicated reliability, availability, and serviceability engine with AI-driven predictive failure management
- **Neural Rendering:** Hardware-accelerated neural shader execution, mega-geometry engine

### Warp Scheduling

- 32 threads per warp
- Enhanced warpgroup MMA instructions (wgmma) for 128-thread cooperative matrix operations
- Improved asynchronous execution with enhanced TMA support

---

## 5. NVIDIA Rubin (Next-Gen)

### Overview

| Property | Rubin GPU |
|---|---|
| **Process Node** | TSMC 3nm |
| **Expected Release** | H2 2026 |
| **Successor To** | Blackwell |
| **Succeeded By** | Feynman (roadmap) |

### Known Specifications

**Compute Performance:**
- 50 PFLOPS FP4 per GPU (vs. 20 PFLOPS for Blackwell B200)
- Rubin Ultra: 100 PFLOPS FP4 per GPU
- Rubin NVL144: 3.6 EFLOPS dense FP4 compute (vs. 1.1 EFLOPS for B300 NVL72)
- 1.2 EFLOPS FP8 training (vs. 0.36 EFLOPS for B300)

**Memory:**
- HBM4 (Rubin standard), HBM4e (Rubin Ultra)
- 288 GB per GPU capacity (same as B300)
- Bandwidth: 13 TB/s per GPU (vs. 8 TB/s for Blackwell)

**Interconnect:**
- NVLink 6: 3.6 TB/s per GPU (2x Blackwell's 1.8 TB/s)
- Rack-level (NVL72): 260 TB/s total (vs. 130 TB/s for Blackwell NVL72)
- ConnectX-9 (CX9) inter-rack link: 28.8 TB/s

**Third-Gen Transformer Engine:**
- Hardware-accelerated adaptive compression

### Platform Integration

The Rubin platform combines six chips:

| Chip | Role |
|---|---|
| **Rubin GPU** | AI compute accelerator |
| **Vera CPU** | 88 custom Olympus cores, Armv9.2, NVLink-C2C connected |
| **NVLink 6 Switch** | In-network compute for collective operations |
| **ConnectX-9 SuperNIC** | High-speed networking |
| **BlueField-4 DPU** | Data processing unit |
| **Spectrum-6** | Ethernet switch |

### System Configurations

| Configuration | Details |
|---|---|
| **Vera Rubin NVL72** | 72 Rubin GPUs + 36 Vera CPUs, full NVLink 6 mesh |
| **HGX Rubin NVL8** | 8 Rubin GPUs, server board for x86 platforms |

### Performance Claims vs. Blackwell

- Up to 10x reduction in inference token cost
- 4x reduction in number of GPUs to train MoE models
- Up to 5x greater inference performance

### Roadmap

- **Rubin:** H2 2026
- **Rubin Ultra:** 2027
- **Feynman:** Post-Rubin (announced on roadmap)

---

## 6. AMD Instinct MI300X / MI300A

### Overview

| Property | MI300X | MI300A |
|---|---|---|
| **Architecture** | CDNA 3 | CDNA 3 |
| **Process Node** | TSMC 5nm (compute) + 6nm (I/O) | TSMC 5nm + 6nm |
| **Transistors** | 153 billion | 146 billion |
| **Form Factor** | OAM | OAM |

### Chiplet Architecture

**MI300X (Pure GPU):**
- 8 XCDs (Accelerated Compute Dies) on 5nm
- 4 IODs (I/O Dies) on 6nm
- Each XCD: 38 CDNA 3 Compute Units (CUs), 4 MB L2 cache
- Total: 304 CUs, 1,216 Matrix Cores (AI), 19,456 Stream Processors

**MI300A (APU - CPU+GPU):**
- 6 XCDs (up to 228 CUs)
- 3 CCDs (Zen 4 CPU chiplets, up to 24 cores)
- Total GPU CUs: 228, Total CPU cores: 24

### Matrix Core Specifications

**MI300X Peak Performance:**

| Precision | Peak TFLOPS |
|---|---|
| FP64 | 81.7 |
| FP32 | 163.4 |
| FP16 | 1,307.4 |
| BF16 | 1,307.4 |
| INT8 | 2,614.9 (TOPS) |
| FP8 | 2,614.9 |

Generational improvement: 3x over MI250X for FP16/BF16, 6.8x for INT8.

### Memory Hierarchy

| Component | MI300X | MI300A |
|---|---|---|
| HBM Type | HBM3 | HBM3 |
| HBM Stacks | 8 | 8 |
| Capacity | 192 GB | 128 GB |
| Bus Width | 8,192-bit | 8,192-bit |
| Data Rate | 5.2 Gbps | 5.2 Gbps |
| Bandwidth | 5,325 GB/s (5.3 TB/s) | 5,325 GB/s |
| L1 Cache per CU | 32 KB | 32 KB |
| L2 Cache per XCD | 4 MB | 4 MB |
| Total L2 Cache | 32 MB | 24 MB |
| Infinity Cache (LLC) | 256 MB | 256 MB |
| TDP | 750W | 760W |

### Interconnect

| Feature | Specification |
|---|---|
| **Infinity Fabric (GPU-GPU)** | 128 GB/s per link between GPUs |
| **Platform (8-GPU)** | Fully connected 8x MI300X via 4th-gen Infinity Fabric |
| **Total Platform Memory** | 1.5 TB HBM3 (8-GPU) |
| **PCIe** | Gen 5 x16 |
| **Unified Memory** | MI300A: CPU and GPU share same HBM3 pool, eliminating PCIe copy overhead |

### Key Architectural Innovations

- **Chiplet Design:** Industry's first GPU with 3D chiplet stacking (compute dies stacked on I/O dies with TSV)
- **Unified Memory (MI300A):** CPU (Zen 4) and GPU (CDNA 3) share same HBM3 address space with hardware coherency
- **256 MB Infinity Cache:** Last-level cache shared across all compute dies, reducing HBM traffic
- **CDNA 3 Matrix Cores:** Optimized for AI with 8-bit and sub-8-bit precision support
- **Massive HBM Capacity:** 192 GB enables running larger models without model parallelism (e.g., 70B parameter models single-GPU)

---

## 7. AMD Instinct MI325X and MI350

### MI325X

| Property | MI325X |
|---|---|
| **Architecture** | CDNA 3 |
| **Process Node** | TSMC 5nm + 6nm |
| **Compute Units** | 304 |
| **Stream Processors** | 19,456 |
| **Matrix Cores** | 1,216 |
| **Release** | Q4 2024 |

**Performance:**

| Precision | Peak TFLOPS |
|---|---|
| FP64 | 81.7 |
| FP32 | 163.4 |
| FP16 | 1,307.4 |
| BF16 | 1,307.4 |
| FP8 | 2,614.9 |

**Memory:**

| Component | Specification |
|---|---|
| HBM Type | HBM3e |
| Capacity | 256 GB |
| Bandwidth | 6 TB/s |
| L2 Cache | 16 MB |
| Infinity Cache | 256 MB |
| TDP | 1,000W |

**Key Differentiator:** Same CDNA 3 compute as MI300X but with upgraded HBM3e memory (256 GB capacity, 6 TB/s bandwidth vs. 192 GB / 5.3 TB/s on MI300X).

### MI350 Series (MI350X / MI355X)

| Property | MI350X | MI355X |
|---|---|---|
| **Architecture** | CDNA 4 | CDNA 4 |
| **Process Node** | TSMC 3nm | TSMC 3nm |
| **Transistors** | 185 billion | 185 billion |
| **Compute Units** | 256 | 320 |
| **Stream Processors** | 16,384 | 20,480 |
| **Matrix Cores** | 1,024 | 1,280 |
| **Release** | June 2025 | June 2025 |

**Performance (MI355X):**

| Precision | Peak Performance |
|---|---|
| FP64 | 79 TFLOPS |
| FP16 | 5 PFLOPS |
| FP8 | 10 PFLOPS |
| FP6 | ~14 PFLOPS |
| FP4 | 20 PFLOPS |

**Memory (Both MI350X and MI355X):**

| Component | Specification |
|---|---|
| HBM Type | HBM3e |
| Capacity | 288 GB |
| Bandwidth | 8 TB/s |
| Memory Interface | 8,192-bit |
| L1 Cache per CU | 32 KB |
| L2 Cache per XCD | 4 MB |
| Infinity Cache (LLC) | 256 MB |

**Power:**

| Variant | TDP |
|---|---|
| MI350X | 1,000W (air-cooled) |
| MI355X | 1,400W (liquid-cooled) |

### Key Architectural Innovations (CDNA 4)

- **FP4 and FP6 Support:** First AMD GPU with native FP4 and FP6 precision for inference, matching Blackwell
- **4x Generational Performance:** Up to 4x peak theoretical performance over MI300X
- **3nm Process:** First AMD GPU on TSMC 3nm, enabling higher transistor density and efficiency
- **Claimed 2.2x Faster Than B200:** AMD claims MI355X delivers 2.2x better inference throughput vs. NVIDIA B200 for certain LLM workloads

### MI400 Series (2026 Roadmap)

- Next-generation after MI350
- "Helios" reference design for AI rack infrastructure
- Supports up to 72 MI400 series GPUs per rack
- 260 TB/s scale-up bandwidth
- Expected dramatic generational leap for large-scale training

---

## 8. Intel Gaudi 3

### Overview

| Property | Gaudi 3 |
|---|---|
| **Architecture** | Dual-die heterogeneous |
| **Process Node** | TSMC 5nm |
| **Form Factor** | OAM mezzanine card, PCIe add-in card |
| **Target** | AI training and inference |

### Compute Architecture

Gaudi 3 uses a dual-die design connected by a high-bandwidth bridge:

| Component | Per Die | Total (Both Dies) |
|---|---|---|
| **MME (Matrix Multiplication Engine)** | 4 | 8 |
| **TPC (Tensor Processor Core)** | 32 | 64 |
| **On-die SRAM Cache** | 48 MB | 96 MB |

**Compute Engine Design:**
- **MME:** Fixed-function matrix multiplication engine optimized for large GEMM operations. Handles all operations reducible to matrix multiply
- **TPC:** Fully programmable VLIW processor for element-wise ops, activations, normalization, and any non-GEMM computation. Custom ISA with native support for tensor operations

### Peak Performance

| Precision | TFLOPS |
|---|---|
| FP8 | 1,835 |
| BF16 (Matrix) | 1,835 |
| BF16 (Vector) | 28.7 |
| FP32 (Vector) | 14.3 |

### Memory Hierarchy

| Component | Specification |
|---|---|
| HBM Type | HBM2e |
| HBM Stacks | 8 |
| Capacity | 128 GB |
| Bandwidth | 3.7 TB/s |
| On-die SRAM (L2) | 96 MB total (48 MB per die) |
| SRAM Bandwidth | 12.8 TB/s |

### Networking (Integrated)

| Feature | Specification |
|---|---|
| RDMA NICs | 24x 200 GbE ports |
| Total Network Bandwidth | 1,200 GB/s bidirectional |
| Protocol | RoCE v2 (RDMA over Converged Ethernet) |
| Topology | Direct chip-to-chip without external switches |

### Media Processing

| Feature | Specification |
|---|---|
| Media Engines | 14 |
| Supported Codecs | H.265 (HEVC), H.264 (AVC), JPEG, VP9 |
| Capabilities | Decode + post-processing of video streams |
| Use Case | Vision AI, video preprocessing pipelines |

### Power

| Configuration | TDP |
|---|---|
| OAM (Air-cooled) | Up to 900W |
| OAM (Liquid-cooled) | Up to 1,200W |
| PCIe | Lower TDP variant available |
| Host Interface | PCIe Gen 5 x16 |

### Key Architectural Innovations

- **Integrated Networking:** 24x 200GbE RDMA ports built into the chip, eliminating need for external NICs. Enables direct scale-out without network switches for small clusters
- **Heterogeneous Compute:** Separate MME (matrix) and TPC (tensor processor) engines allow independent optimization of GEMM vs. non-GEMM operations
- **Large On-die SRAM:** 96 MB SRAM with 12.8 TB/s bandwidth serves as massive scratchpad, reducing HBM pressure
- **Media Processing Integration:** Built-in video decode engines for end-to-end vision AI pipelines
- **Software:** Uses PyTorch with Habana's SynapseAI SDK; supports vLLM, DeepSpeed, Hugging Face Optimum-Habana
- **Cost Positioning:** Priced significantly below H100/H200, targeting price-performance ratio advantage

### Comparison vs. NVIDIA H100

| Metric | Gaudi 3 | H100 SXM |
|---|---|---|
| BF16 TFLOPS | 1,835 | 1,979 |
| FP8 TFLOPS | 1,835 | 3,958 |
| HBM Capacity | 128 GB (HBM2e) | 80 GB (HBM3) |
| HBM Bandwidth | 3.7 TB/s | 3.35 TB/s |
| Network | 24x 200GbE (integrated) | NVLink 900 GB/s + InfiniBand |
| TDP | 900W (air) | 700W |

---

## 9. Google TPU v5e / v5p / v6e (Trillium)

### Overview

| Property | TPU v5e | TPU v5p | TPU v6e (Trillium) |
|---|---|---|---|
| **Release** | 2023 | 2023 | 2024 (preview), 2025 (GA) |
| **Process Node** | Not disclosed | Not disclosed | TSMC N5 (5nm) |
| **Target** | Cost-efficient inference/training | High-performance training | Next-gen training/inference |
| **Generation** | 5th | 5th | 6th |

### Per-Chip Compute

| Specification | TPU v5e | TPU v5p | TPU v6e (Trillium) |
|---|---|---|---|
| TensorCores per chip | 1 | 1 | 1 |
| MXUs per TensorCore | 4 | 4 | 2 (but 4x larger each) |
| MXU Systolic Array Size | 128 x 128 | 128 x 128 | 256 x 256 |
| BF16 Peak TFLOPS | 197 | 459 | ~920 (4.7x v5e) |
| INT8 Peak TOPS | 393 | 918 | ~1,840 (estimated) |

### Memory

| Specification | TPU v5e | TPU v5p | TPU v6e (Trillium) |
|---|---|---|---|
| HBM Type | HBM2e | HBM2e | HBM (likely HBM2e/HBM3) |
| HBM Capacity per chip | 16 GB | 95 GB | 32 GB |
| HBM Bandwidth | ~820 GB/s | 2,800 GB/s | ~1,640 GB/s (2x v5e) |

### Interconnect (ICI - Inter-Chip Interconnect)

| Specification | TPU v5e | TPU v5p | TPU v6e (Trillium) |
|---|---|---|---|
| Topology | 2D Torus | 3D Torus | 2D Torus |
| Neighbors | 4 (2D) | 6 (3D) | 4 (2D) |
| ICI Bandwidth per chip | ~400 Gbps/axis | 4,800 Gbps total | ~800 Gbps/axis (2x v5e) |
| Max Pod Size (chips) | 256 | 8,960 | 256 |

### Power

| Specification | TPU v5e | TPU v5p | TPU v6e (Trillium) |
|---|---|---|---|
| TDP per chip | ~200W (est.) | 450W | ~250W (est., 67% more efficient) |

### Architecture Details

**Matrix Multiply Unit (MXU):**
- Systolic array architecture performing matrix multiply-accumulate
- v5e/v5p: 128x128 multiply-accumulators per MXU
- v6e: 256x256 multiply-accumulators per MXU (4x compute density)
- Supports BF16, INT8 inputs with FP32 accumulation
- Each MXU can process one matrix multiplication per cycle

**Vector Unit:**
- Handles element-wise operations (activations, normalization, softmax)
- Operates on 1D vectors
- Supports FP32, BF16, INT8/32

**Scalar Unit:**
- Control flow, address calculation, scalar operations
- Manages data movement between HBM, VMEM, and MXU

**On-chip Memory (VMEM/CMEM):**
- Vector Memory (VMEM): High-bandwidth SRAM for vector/matrix data staging
- v5e: ~37 MB SRAM per chip
- v5p: ~95 MB SRAM per chip (estimated)
- Acts as software-managed scratchpad (not hardware cache)

### Key Architectural Innovations

**TPU v5e:**
- Cost-optimized for inference and training of models up to ~200B parameters
- 2x cost-efficiency vs. TPU v4 for inference
- Dynamic tiling support for variable-size workloads
- Optimized for serving large language models

**TPU v5p:**
- Highest per-chip performance in TPU v5 family
- 3D torus topology enabling 8,960-chip pods
- 2x FLOPS and 3x HBM vs. TPU v4
- Designed for training frontier models (Gemini, PaLM)
- SparseCore for embedding-heavy recommendation models

**TPU v6e (Trillium):**
- 4.7x peak compute over v5e through enlarged MXUs (256x256)
- 2x HBM capacity and bandwidth over v5e
- 2x ICI bandwidth for better scaling
- 67% more energy-efficient than v5e
- Supports INT8 quantized inference natively
- Enhanced SparseCore for recommendation workloads
- Scales to 256-chip pods

### Software Ecosystem

- **Framework:** JAX (primary), TensorFlow, PyTorch/XLA
- **Compiler:** XLA (Accelerated Linear Algebra)
- **Orchestration:** Google Cloud TPU VMs, GKE (Google Kubernetes Engine)
- **Libraries:** Pax, MaxText, Flax, Orbax
- **Key Feature:** Software-managed memory (explicit data tiling and placement vs. hardware caching)

---

## Cross-Architecture Comparison

### Data Center AI Accelerator Comparison

| Metric | A100 SXM 80GB | H100 SXM | B200 | MI300X | MI355X | Gaudi 3 |
|---|---|---|---|---|---|---|
| **FP16 TFLOPS** | 312 | 1,979 | 5,000 | 1,307 | 5,000 | 1,835 (BF16) |
| **FP8 TFLOPS** | N/A | 3,958 | 10,000 | 2,615 | 10,000 | 1,835 |
| **FP4 TFLOPS** | N/A | N/A | 20,000 | N/A | 20,000 | N/A |
| **HBM Capacity** | 80 GB | 80 GB | 192 GB | 192 GB | 288 GB | 128 GB |
| **HBM BW** | 2,039 GB/s | 3,352 GB/s | 8,000 GB/s | 5,325 GB/s | 8,000 GB/s | 3,700 GB/s |
| **L2 Cache** | 40 MB | 50 MB | ~192 MB | 32 MB + 256 MB LLC | 32 MB + 256 MB LLC | 96 MB SRAM |
| **Interconnect BW** | 600 GB/s NVLink | 900 GB/s NVLink | 1,800 GB/s NVLink | 128 GB/s IF/link | TBD | 1,200 GB/s Ethernet |
| **TDP** | 400W | 700W | 1,000W | 750W | 1,400W | 900W |
| **Process** | 7nm | 4nm | 4nm (4NP) | 5nm/6nm | 3nm | 5nm |

### Compute Capability Reference (NVIDIA)

| Compute Capability | Architecture | Key GPUs |
|---|---|---|
| 8.0 | Ampere (GA100) | A100, A30 |
| 8.6 | Ampere (GA102/104) | RTX 3090/3080/3070, A5000/A4000 |
| 8.9 | Ada Lovelace (AD10x) | RTX 4090/4080/4070, L40, L4 |
| 9.0 | Hopper (GH100) | H100, H200 |
| 10.0 | Blackwell (GB100) | B100, B200, B300 |
| 12.0 | Blackwell (GB202) | RTX 5090, RTX 5080 |

### Memory Hierarchy Evolution (NVIDIA Data Center)

| Feature | A100 | H100 | B200 |
|---|---|---|---|
| **L1+Shared/SM** | 192 KB | 256 KB | 256 KB |
| **Max Shared/SM** | 164 KB | 228 KB | 228 KB+ |
| **L2 Cache** | 40 MB | 50 MB | ~192 MB |
| **HBM Gen** | HBM2e | HBM3 | HBM3e |
| **HBM BW** | 2.0 TB/s | 3.35 TB/s | 8.0 TB/s |
| **NVLink BW** | 600 GB/s | 900 GB/s | 1,800 GB/s |

### Tensor Core / Matrix Engine Precision Support

| Precision | A100 | H100 | B200 | Ada | MI300X | MI350X | Gaudi 3 | TPU v6e |
|---|---|---|---|---|---|---|---|---|
| FP64 | Yes | Yes | Yes | No | Yes | Yes | No | No |
| TF32 | Yes | Yes | Yes | Yes | No | No | No | No |
| FP32 | CUDA | CUDA | CUDA | CUDA | Yes | Yes | Vector | No |
| BF16 | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| FP16 | Yes | Yes | Yes | Yes | Yes | Yes | No | No |
| FP8 | No | Yes | Yes | Yes | Yes | Yes | Yes | No |
| FP6 | No | No | Yes | No | No | Yes | No | No |
| FP4 | No | No | Yes | No | No | Yes | No | No |
| INT8 | Yes | Yes | Yes | Yes | Yes | Yes | No | Yes |
| INT4 | Yes | No | No | No | No | No | No | No |
| Sparsity | 2:4 | 2:4 | 2:4 | 2:4 | No | No | No | No |

---

## Optimization Implications

### Precision Selection Guide

| Workload | Recommended Precision | Best Hardware |
|---|---|---|
| LLM Inference (latency) | FP4/FP8 with micro-tensor scaling | B200, MI355X |
| LLM Inference (throughput) | FP8 | H100, B200, MI300X |
| LLM Training | BF16 mixed-precision | H100, B200, MI300X |
| Fine-tuning (QLoRA) | INT4 weights + BF16 compute | Any with BF16 TC |
| Vision Models | FP16/TF32 | A100, H100, RTX 4090 |
| HPC (Scientific) | FP64 | A100, H100, B200, MI300X |
| Cost-sensitive Inference | INT8/FP8 | TPU v5e, Gaudi 3, L4 |

### Key Optimization Strategies by Architecture

**NVIDIA Ampere (A100):**
- Use TF32 for transparent FP32 speedup (no code changes)
- Enable 2:4 structured sparsity for 2x throughput (requires sparse training/pruning)
- Use async copy (cp.async) for compute/memory overlap
- Leverage MIG for multi-tenant inference

**NVIDIA Hopper (H100):**
- Use FP8 with Transformer Engine for automatic mixed-precision
- Leverage TMA for efficient tensor data movement
- Use thread block clusters for cross-SM cooperation (flash attention)
- Use wgmma instructions for large matrix operations
- Overlap communication with compute using async execution

**NVIDIA Blackwell (B200/B300):**
- Use FP4 with micro-tensor scaling for maximum inference throughput
- Leverage second-gen Transformer Engine for dynamic FP4/FP8 selection
- Use decompression engine for database/RAG workloads
- Scale to 576 GPUs via NVLink 5 for massive model training

**AMD MI300X/MI350:**
- Leverage large HBM capacity (192-288 GB) for fewer-GPU deployments
- Use 256 MB Infinity Cache to reduce memory traffic
- Target FP8 for inference on CDNA 3; FP4 on CDNA 4
- Use ROCm + PyTorch for software stack

**Google TPU:**
- Use BF16 as primary training precision
- Leverage software-managed memory with explicit tiling
- Use large pod configurations for distributed training
- Optimize for XLA compilation patterns

---

## Sources

### NVIDIA
- [NVIDIA Ampere Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)
- [NVIDIA A100 Whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)
- [NVIDIA GA102 Whitepaper](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.1.pdf)
- [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [NVIDIA H100 Whitepaper](https://www.advancedclustering.com/wp-content/uploads/2022/03/gtc22-whitepaper-hopper.pdf)
- [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)
- [NVIDIA Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)
- [NVIDIA Ada Lovelace Whitepaper](https://images.nvidia.com/aem-dam/Solutions/Data-Center/l4/nvidia-ada-gpu-architecture-whitepaper-v2.1.pdf)
- [NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)
- [NVIDIA RTX Blackwell Architecture Whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)
- [NVIDIA Rubin Platform Announcement](https://nvidianews.nvidia.com/news/rubin-platform-ai-supercomputer)
- [NVIDIA A100 Datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf)
- [NVIDIA H100 Datasheet](https://www.megware.com/fileadmin/user_upload/LandingPage%20NVIDIA/nvidia-h100-datasheet.pdf)

### AMD
- [AMD MI300X Product Page](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)
- [AMD MI300X Datasheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf)
- [AMD MI325X Product Page](https://www.amd.com/en/products/accelerators/instinct/mi300/mi325x.html)
- [AMD MI350 Series](https://www.amd.com/en/products/accelerators/instinct/mi350.html)
- [AMD MI350 Blog](https://www.amd.com/en/blogs/2025/amd-instinct-mi350-series-game-changer.html)
- [AMD MI300X Hot Chips 2024 Presentation](https://hc2024.hotchips.org/assets/program/conference/day1/23_HC2024.AMD.MI300X.ASmith(MI300X).v1.Final.20240817.pdf)

### Intel
- [Intel Gaudi 3 White Paper](https://cdrdv2-public.intel.com/817486/gaudi-3-ai-accelerator-white-paper.pdf)
- [Gaudi Architecture Documentation](https://docs.habana.ai/en/latest/Gaudi_Overview/Gaudi_Architecture.html)
- [Intel Gaudi 3 Hot Chips 2024](https://hc2024.hotchips.org/assets/program/conference/day1/60_HC2024.Intel.RomanKaplan.Gaudi3-0826.pdf)

### Google
- [TPU v5e Documentation](https://docs.google.com/tpu/docs/v5e)
- [TPU v5p Documentation](https://docs.cloud.google.com/tpu/docs/v5p)
- [TPU v6e Documentation](https://docs.cloud.google.com/tpu/docs/v6e)
- [Introducing Trillium Blog](https://cloud.google.com/blog/products/compute/introducing-trillium-6th-gen-tpus)
- [Google TPU System Architecture](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm)

### Third-Party Analysis
- [Benchmarking the NVIDIA Hopper GPU Architecture (arXiv)](https://arxiv.org/pdf/2402.13499)
- [Chips and Cheese: Testing AMD MI300X](https://chipsandcheese.com/p/testing-amds-giant-mi300x)
- [Tom's Hardware: Rubin GPUs](https://www.tomshardware.com/pc-components/gpus/nvidia-announces-rubin-gpus-in-2026-rubin-ultra-in-2027-feynam-after)
- [ServeTheHome: Rubin Launch at CES 2026](https://www.servethehome.com/nvidia-launches-next-generation-rubin-ai-compute-platform-at-ces-2026/)
- [Exxact: Blackwell vs Hopper Comparison](https://www.exxactcorp.com/blog/hpc/comparing-nvidia-tensor-core-gpus)
