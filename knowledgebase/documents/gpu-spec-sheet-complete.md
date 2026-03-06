# Definitive GPU Specification Reference for Kernel Optimization

> The exhaustive spec sheet for every GPU relevant to AI/ML workloads.
> Last updated: 2026-03-06

---

## Table of Contents

1. [Architecture Constants (Per Compute Capability)](#1-architecture-constants-per-compute-capability)
2. [NVIDIA Consumer GPUs - Ampere (RTX 30 Series)](#2-nvidia-consumer-gpus---ampere-rtx-30-series)
3. [NVIDIA Consumer GPUs - Ada Lovelace (RTX 40 Series)](#3-nvidia-consumer-gpus---ada-lovelace-rtx-40-series)
4. [NVIDIA Consumer GPUs - Blackwell (RTX 50 Series)](#4-nvidia-consumer-gpus---blackwell-rtx-50-series)
5. [NVIDIA Workstation GPUs - Ampere (RTX Axxxx)](#5-nvidia-workstation-gpus---ampere-rtx-axxxx)
6. [NVIDIA Workstation GPUs - Ada Lovelace (RTX xxxx Ada)](#6-nvidia-workstation-gpus---ada-lovelace-rtx-xxxx-ada)
7. [NVIDIA Workstation GPUs - Blackwell (RTX PRO 6000)](#7-nvidia-workstation-gpus---blackwell-rtx-pro-6000)
8. [NVIDIA Datacenter GPUs - Turing (T4)](#8-nvidia-datacenter-gpus---turing-t4)
9. [NVIDIA Datacenter GPUs - Ampere (A-Series)](#9-nvidia-datacenter-gpus---ampere-a-series)
10. [NVIDIA Datacenter GPUs - Ada Lovelace (L-Series)](#10-nvidia-datacenter-gpus---ada-lovelace-l-series)
11. [NVIDIA Datacenter GPUs - Hopper (H-Series)](#11-nvidia-datacenter-gpus---hopper-h-series)
12. [NVIDIA Datacenter GPUs - Blackwell (B-Series)](#12-nvidia-datacenter-gpus---blackwell-b-series)
13. [AMD Instinct Accelerators](#13-amd-instinct-accelerators)
14. [Intel Gaudi Accelerators](#14-intel-gaudi-accelerators)
15. [Master Comparison Tables](#15-master-comparison-tables)

---

## 1. Architecture Constants (Per Compute Capability)

These are fixed hardware limits determined by architecture, not by specific GPU model.

| Parameter | CC 7.5 (Turing) | CC 8.0 (Ampere DC) | CC 8.6 (Ampere Consumer) | CC 8.9 (Ada) | CC 9.0 (Hopper) | CC 10.0 (Blackwell DC) | CC 12.0 (Blackwell Consumer) |
|---|---|---|---|---|---|---|---|
| **Warp Size** | 32 | 32 | 32 | 32 | 32 | 32 | 32 |
| **Max Threads/Block** | 1,024 | 1,024 | 1,024 | 1,024 | 1,024 | 1,024 | 1,024 |
| **Max Threads/SM** | 1,024 | 2,048 | 1,536 | 1,536 | 2,048 | 2,048 | 2,048 |
| **Max Warps/SM** | 32 | 64 | 48 | 48 | 64 | 64 | 64 |
| **Max Registers/Thread** | 255 | 255 | 255 | 255 | 255 | 255 | 255 |
| **Max Registers/SM** | 65,536 | 65,536 | 65,536 | 65,536 | 65,536 | 65,536 | 65,536 |
| **Register File Size/SM** | 256 KB | 256 KB | 256 KB | 256 KB | 256 KB | 256 KB | 256 KB |
| **Max Shared Mem/SM** | 64 KB | 164 KB | 100 KB | 100 KB | 228 KB | 228 KB | 228 KB |
| **Max Shared Mem/Block** | 48 KB | 164 KB | 100 KB | 99 KB | 227 KB | 227 KB | 227 KB |
| **L1+Shared/SM (Combined)** | 96 KB | 192 KB | 128 KB | 128 KB | 256 KB | 256 KB | 128 KB |
| **CUDA Cores/SM** | 64 | 64 (FP32) + 64 (FP64) | 128 (dual FP32) | 128 (dual FP32) | 128 (FP32) + 64 (FP64) | 128 (FP32) | 128 (FP32) |
| **Tensor Cores/SM** | 8 (2nd gen) | 4 (3rd gen) | 4 (3rd gen) | 4 (4th gen) | 4 (4th gen) | 4 (5th gen) | 4 (5th gen) |
| **FP8 TC Support** | No | No | No | Yes | Yes | Yes | Yes |
| **FP4 TC Support** | No | No | No | No | No | Yes | Yes |
| **Sparsity (2:4)** | No | Yes | Yes | Yes | Yes | Yes | Yes |
| **Key GPUs** | T4 | A100, A30 | RTX 30xx, A2, A10, A10G, A40, RTX Axxxx | RTX 40xx, L4, L40, L40S, RTX xxxx Ada | H100, H200 | B100, B200, B300 | RTX 50xx, RTX PRO 6000 |

---

## 2. NVIDIA Consumer GPUs - Ampere (RTX 30 Series)

| Specification | RTX 3060 (12GB) | RTX 3070 | RTX 3080 (10GB) | RTX 3090 |
|---|---|---|---|---|
| **Full Name** | GeForce RTX 3060 12GB | GeForce RTX 3070 | GeForce RTX 3080 10GB | GeForce RTX 3090 |
| **GPU Die** | GA106 | GA104 | GA102 | GA102 |
| **Architecture** | Ampere | Ampere | Ampere | Ampere |
| **Compute Capability** | 8.6 | 8.6 | 8.6 | 8.6 |
| **Process Node** | Samsung 8nm | Samsung 8nm | Samsung 8nm | Samsung 8nm |
| **SM Count** | 28 | 46 | 68 | 82 |
| **CUDA Cores** | 3,584 | 5,888 | 8,704 | 10,496 |
| **Tensor Cores (3rd Gen)** | 112 | 184 | 272 | 328 |
| **RT Cores (2nd Gen)** | 28 | 46 | 68 | 82 |
| **Base Clock** | 1,320 MHz | 1,500 MHz | 1,440 MHz | 1,395 MHz |
| **Boost Clock** | 1,780 MHz | 1,730 MHz | 1,710 MHz | 1,695 MHz |
| **Memory Type** | GDDR6 | GDDR6 | GDDR6X | GDDR6X |
| **Memory Size** | 12 GB | 8 GB | 10 GB | 24 GB |
| **Memory Bus Width** | 192-bit | 256-bit | 320-bit | 384-bit |
| **Memory Bandwidth** | 360 GB/s | 448 GB/s | 760 GB/s | 936 GB/s |
| **L2 Cache** | 3 MB | 4 MB | 5 MB | 6 MB |
| **L1/Shared per SM** | 128 KB | 128 KB | 128 KB | 128 KB |
| **Max Shared Mem/SM** | 100 KB | 100 KB | 100 KB | 100 KB |
| **TDP** | 170W | 220W | 320W | 350W |
| **PCIe** | Gen 4 x16 | Gen 4 x16 | Gen 4 x16 | Gen 4 x16 |
| **NVLink** | No | No | No | 3rd Gen (1 link, 112 GB/s) |
| **Release Date** | Feb 2021 | Oct 2020 | Sep 2020 | Sep 2020 |

### RTX 30 Series Tensor Core Performance (TFLOPS, Dense | With Sparsity)

| Precision | RTX 3060 | RTX 3070 | RTX 3080 | RTX 3090 |
|---|---|---|---|---|
| **FP32 (non-TC)** | 12.7 | 20.3 | 29.8 | 35.6 |
| **FP16 Tensor** | 51 \| 101 | 81 \| 163 | 119 \| 238 | 142 \| 285 |
| **BF16 Tensor** | 51 \| 101 | 81 \| 163 | 119 \| 238 | 142 \| 285 |
| **TF32 Tensor** | 25.4 \| 50.9 | 40.6 \| 81 | 59.5 \| 119 | 71 \| 142 |
| **INT8 Tensor (TOPS)** | 101 \| 203 | 163 \| 326 | 238 \| 477 | 285 \| 570 |
| **INT4 Tensor (TOPS)** | 203 \| 406 | 326 \| 652 | 477 \| 954 | 570 \| 1,140 |
| **FP64** | 0.20 | 0.32 | 0.47 | 0.56 |
| **FP8** | Not supported | Not supported | Not supported | Not supported |

---

## 3. NVIDIA Consumer GPUs - Ada Lovelace (RTX 40 Series)

| Specification | RTX 4060 | RTX 4070 | RTX 4080 | RTX 4090 |
|---|---|---|---|---|
| **Full Name** | GeForce RTX 4060 | GeForce RTX 4070 | GeForce RTX 4080 16GB | GeForce RTX 4090 |
| **GPU Die** | AD107 | AD104 | AD103 | AD102 |
| **Architecture** | Ada Lovelace | Ada Lovelace | Ada Lovelace | Ada Lovelace |
| **Compute Capability** | 8.9 | 8.9 | 8.9 | 8.9 |
| **Process Node** | TSMC 4N | TSMC 4N | TSMC 4N | TSMC 4N |
| **SM Count** | 24 | 46 | 76 | 128 |
| **CUDA Cores** | 3,072 | 5,888 | 9,728 | 16,384 |
| **Tensor Cores (4th Gen)** | 96 | 184 | 304 | 512 |
| **RT Cores (3rd Gen)** | 24 | 46 | 76 | 128 |
| **Base Clock** | 1,830 MHz | 1,920 MHz | 2,205 MHz | 2,235 MHz |
| **Boost Clock** | 2,460 MHz | 2,475 MHz | 2,505 MHz | 2,520 MHz |
| **Memory Type** | GDDR6 | GDDR6X | GDDR6X | GDDR6X |
| **Memory Size** | 8 GB | 12 GB | 16 GB | 24 GB |
| **Memory Bus Width** | 128-bit | 192-bit | 256-bit | 384-bit |
| **Memory Bandwidth** | 272 GB/s | 504 GB/s | 717 GB/s | 1,008 GB/s |
| **L2 Cache** | 24 MB | 36 MB | 64 MB | 72 MB |
| **L1/Shared per SM** | 128 KB | 128 KB | 128 KB | 128 KB |
| **Max Shared Mem/SM** | 100 KB | 100 KB | 100 KB | 100 KB |
| **TDP** | 115W | 200W | 320W | 450W |
| **PCIe** | Gen 4 x8 | Gen 4 x16 | Gen 4 x16 | Gen 4 x16 |
| **NVLink** | No | No | No | No |
| **Release Date** | Jun 2023 | Apr 2023 | Nov 2022 | Oct 2022 |

### RTX 40 Series Tensor Core Performance (TFLOPS, Dense | With Sparsity)

| Precision | RTX 4060 | RTX 4070 | RTX 4080 | RTX 4090 |
|---|---|---|---|---|
| **FP32 (non-TC)** | 15.1 | 29.1 | 48.7 | 82.6 |
| **FP16 Tensor** | 30.2 \| 60.5 | 58.3 \| 116.5 | 97.5 \| 194.9 | 165.2 \| 330.3 |
| **BF16 Tensor** | 30.2 \| 60.5 | 58.3 \| 116.5 | 97.5 \| 194.9 | 165.2 \| 330.3 |
| **TF32 Tensor** | 15.1 \| 30.2 | 29.1 \| 58.3 | 48.7 \| 97.5 | 82.6 \| 165.2 |
| **FP8 Tensor** | 60.5 \| 120.9 | 116.5 \| 233 | 194.9 \| 389.9 | 330.3 \| 660.6 |
| **INT8 Tensor (TOPS)** | 120.9 \| 241.9 | 233 \| 466 | 389.9 \| 779.8 | 660.6 \| 1,321.2 |
| **FP64** | 0.24 | 0.45 | 0.76 | 1.29 |

---

## 4. NVIDIA Consumer GPUs - Blackwell (RTX 50 Series)

| Specification | RTX 5070 | RTX 5080 | RTX 5090 |
|---|---|---|---|
| **Full Name** | GeForce RTX 5070 | GeForce RTX 5080 | GeForce RTX 5090 |
| **GPU Die** | GB205 | GB203 | GB202 |
| **Architecture** | Blackwell | Blackwell | Blackwell |
| **Compute Capability** | 12.0 | 12.0 | 12.0 |
| **Process Node** | TSMC 4N | TSMC 4N | TSMC 4N |
| **SM Count** | 48 | 84 | 170 |
| **CUDA Cores** | 6,144 | 10,752 | 21,760 |
| **Tensor Cores (5th Gen)** | 192 | 336 | 680 |
| **RT Cores (4th Gen)** | 48 | 84 | 170 |
| **Base Clock** | 2,162 MHz | 2,235 MHz | 2,017 MHz |
| **Boost Clock** | 2,512 MHz | 2,520 MHz | 2,407 MHz |
| **Memory Type** | GDDR7 | GDDR7 | GDDR7 |
| **Memory Size** | 12 GB | 16 GB | 32 GB |
| **Memory Bus Width** | 192-bit | 256-bit | 512-bit |
| **Memory Bandwidth** | 672 GB/s | 960 GB/s | 1,792 GB/s |
| **L2 Cache** | 36 MB | 64 MB | 98 MB |
| **L1/Shared per SM** | 128 KB | 128 KB | 128 KB |
| **Max Shared Mem/SM** | 228 KB | 228 KB | 228 KB |
| **TDP** | 250W | 360W | 575W |
| **PCIe** | Gen 5 x16 | Gen 5 x16 | Gen 5 x16 |
| **NVLink** | No | No | No |
| **Release Date** | Mar 2025 | Jan 2025 | Jan 2025 |

### RTX 50 Series Tensor Core Performance (TFLOPS, Dense | With Sparsity)

| Precision | RTX 5070 | RTX 5080 | RTX 5090 |
|---|---|---|---|
| **FP32 (non-TC)** | 30.8 | 54.2 | 104.8 |
| **FP16 Tensor** | 61.7 \| 123.4 | 108.4 \| 216.8 | 209.5 \| 419.0 |
| **BF16 Tensor** | 61.7 \| 123.4 | 108.4 \| 216.8 | 209.5 \| 419.0 |
| **TF32 Tensor** | 30.8 \| 61.7 | 54.2 \| 108.4 | 104.8 \| 209.5 |
| **FP8 Tensor** | 123.4 \| 246.8 | 216.8 \| 433.6 | 419.0 \| 838.0 |
| **FP4 Tensor** | 246.8 \| 493.6 | 433.6 \| 867.2 | 838.0 \| 1,676 |
| **INT8 Tensor (TOPS)** | 246.8 \| 493.6 | 433.6 \| 867.2 | 838.0 \| 1,676 |
| **FP64** | 0.48 | 0.85 | 1.64 |

### Key Blackwell Consumer Features
- Fifth-gen Tensor Cores with native FP4 support
- DLSS 4 with Multi Frame Generation
- Fourth-gen RT Cores with 2x performance
- Neural Rendering support
- GDDR7 memory (first consumer GPUs with GDDR7)

---

## 5. NVIDIA Workstation GPUs - Ampere (RTX Axxxx)

| Specification | RTX A4000 | RTX A5000 | RTX A6000 |
|---|---|---|---|
| **Full Name** | NVIDIA RTX A4000 | NVIDIA RTX A5000 | NVIDIA RTX A6000 |
| **GPU Die** | GA104 | GA102 | GA102 |
| **Architecture** | Ampere | Ampere | Ampere |
| **Compute Capability** | 8.6 | 8.6 | 8.6 |
| **Process Node** | Samsung 8nm | Samsung 8nm | Samsung 8nm |
| **SM Count** | 48 | 64 | 84 |
| **CUDA Cores** | 6,144 | 8,192 | 10,752 |
| **Tensor Cores (3rd Gen)** | 192 | 256 | 336 |
| **RT Cores (2nd Gen)** | 48 | 64 | 84 |
| **Base Clock** | 735 MHz | 1,170 MHz | 930 MHz |
| **Boost Clock** | 1,560 MHz | 1,695 MHz | 1,800 MHz |
| **Memory Type** | GDDR6 | GDDR6 | GDDR6 |
| **Memory Size** | 16 GB | 24 GB | 48 GB |
| **Memory Bus Width** | 256-bit | 384-bit | 384-bit |
| **Memory Bandwidth** | 448 GB/s | 768 GB/s | 768 GB/s |
| **Memory ECC** | Yes | Yes | Yes |
| **L2 Cache** | 4 MB | 6 MB | 6 MB |
| **TDP** | 140W | 230W | 300W |
| **PCIe** | Gen 4 x16 | Gen 4 x16 | Gen 4 x16 |
| **NVLink** | No | 3rd Gen (1 link, 112 GB/s) | 3rd Gen (1 link, 112 GB/s) |
| **Form Factor** | Single-slot | Dual-slot | Dual-slot |
| **Release Date** | Apr 2021 | Apr 2021 | Dec 2020 |

---

## 6. NVIDIA Workstation GPUs - Ada Lovelace (RTX xxxx Ada)

| Specification | RTX 4000 Ada | RTX 5000 Ada | RTX 6000 Ada |
|---|---|---|---|
| **Full Name** | NVIDIA RTX 4000 Ada Generation | NVIDIA RTX 5000 Ada Generation | NVIDIA RTX 6000 Ada Generation |
| **GPU Die** | AD104 | AD103 | AD102 |
| **Architecture** | Ada Lovelace | Ada Lovelace | Ada Lovelace |
| **Compute Capability** | 8.9 | 8.9 | 8.9 |
| **Process Node** | TSMC 4N | TSMC 4N | TSMC 4N |
| **SM Count** | 48 | 100 | 142 |
| **CUDA Cores** | 6,144 | 12,800 | 18,176 |
| **Tensor Cores (4th Gen)** | 192 | 400 | 568 |
| **RT Cores (3rd Gen)** | 48 | 100 | 142 |
| **Boost Clock** | 2,175 MHz | 2,550 MHz | 2,505 MHz |
| **Memory Type** | GDDR6 | GDDR6 | GDDR6 |
| **Memory Size** | 20 GB | 32 GB | 48 GB |
| **Memory Bus Width** | 160-bit | 256-bit | 384-bit |
| **Memory Bandwidth** | 360 GB/s | 576 GB/s | 960 GB/s |
| **Memory ECC** | Yes | Yes | Yes |
| **L2 Cache** | 32 MB | 64 MB | 96 MB |
| **FP32 TFLOPS** | 26.7 | 65.3 | 91.1 |
| **TDP** | 130W | 250W | 300W |
| **PCIe** | Gen 4 x16 | Gen 4 x16 | Gen 4 x16 |
| **NVLink** | No | No | No |
| **Form Factor** | Dual-slot | Dual-slot | Dual-slot |
| **Release Date** | 2023 | 2023 | 2023 |

---

## 7. NVIDIA Workstation GPUs - Blackwell (RTX PRO 6000)

| Specification | RTX PRO 6000 Blackwell |
|---|---|
| **Full Name** | NVIDIA RTX PRO 6000 Blackwell Workstation Edition |
| **GPU Die** | GB202 |
| **Architecture** | Blackwell |
| **Compute Capability** | 12.0 |
| **Process Node** | TSMC 4N |
| **SM Count** | 188 |
| **CUDA Cores** | 24,064 |
| **Tensor Cores (5th Gen)** | 752 |
| **RT Cores (4th Gen)** | 188 |
| **Memory Type** | GDDR7 ECC |
| **Memory Size** | 96 GB |
| **Memory Bus Width** | 512-bit |
| **Memory Bandwidth** | 1,792 GB/s |
| **L2 Cache** | 128 MB |
| **FP32 TFLOPS** | 125 |
| **AI TOPS (FP4)** | 4,000 |
| **RT TFLOPS** | 380 |
| **TDP** | 600W |
| **PCIe** | Gen 5 x16 |
| **NVLink** | No |
| **Display Outputs** | 4x DisplayPort 2.1b |
| **Form Factor** | Dual-slot |
| **Release Date** | 2025 |

### Key Features
- Fifth-gen Tensor Cores with FP4 support (3x performance over previous gen)
- Fourth-gen RT Cores (2x performance over previous gen)
- GDDR7 with ECC for professional reliability
- 96 GB VRAM enables massive datasets and models in-memory
- PCIe Gen 5 for 128 GB/s host bandwidth

---

## 8. NVIDIA Datacenter GPUs - Turing (T4)

| Specification | T4 |
|---|---|
| **Full Name** | NVIDIA Tesla T4 |
| **GPU Die** | TU104 |
| **Architecture** | Turing |
| **Compute Capability** | 7.5 |
| **Process Node** | TSMC 12nm FFN |
| **SM Count** | 40 |
| **CUDA Cores** | 2,560 |
| **Tensor Cores (2nd Gen)** | 320 |
| **RT Cores (1st Gen)** | 40 |
| **Base Clock** | 585 MHz |
| **Boost Clock** | 1,590 MHz |
| **Memory Type** | GDDR6 |
| **Memory Size** | 16 GB |
| **Memory Bus Width** | 256-bit |
| **Memory Bandwidth** | 320 GB/s |
| **L2 Cache** | 4 MB |
| **L1/Shared per SM** | 96 KB |
| **Max Shared Mem/SM** | 64 KB |
| **TDP** | 70W |
| **PCIe** | Gen 3 x16 |
| **NVLink** | No |
| **Form Factor** | Single-slot, low-profile |
| **Release Date** | Sep 2018 |

### T4 Performance (TFLOPS)

| Precision | TFLOPS |
|---|---|
| **FP32** | 8.1 |
| **FP16 Tensor** | 65 |
| **INT8 Tensor** | 130 TOPS |
| **INT4 Tensor** | 260 TOPS |
| **FP64** | 0.25 |

### Key Features
- Ultra-low 70W power envelope (single PCIe slot, no external power)
- Most widely deployed inference GPU (AWS G4, GCP T4)
- INT8 inference optimized
- No FP8, no BF16 support (Turing limitation)

---

## 9. NVIDIA Datacenter GPUs - Ampere (A-Series)

### A2, A10, A10G

| Specification | A2 | A10 | A10G |
|---|---|---|---|
| **Full Name** | NVIDIA A2 | NVIDIA A10 | NVIDIA A10G |
| **GPU Die** | GA107 | GA102 | GA102 |
| **Architecture** | Ampere | Ampere | Ampere |
| **Compute Capability** | 8.6 | 8.6 | 8.6 |
| **Process Node** | Samsung 8nm | Samsung 8nm | Samsung 8nm |
| **SM Count** | 10 | 72 | 72 |
| **CUDA Cores** | 1,280 | 9,216 | 9,216 |
| **Tensor Cores (3rd Gen)** | 40 | 288 | 320 |
| **RT Cores (2nd Gen)** | 10 | 72 | 80 |
| **Boost Clock** | 1,770 MHz | 1,695 MHz | 1,695 MHz |
| **Memory Type** | GDDR6 | GDDR6 | GDDR6 |
| **Memory Size** | 16 GB | 24 GB | 24 GB |
| **Memory Bus Width** | 128-bit | 384-bit | 384-bit |
| **Memory Bandwidth** | 200 GB/s | 600 GB/s | 600 GB/s |
| **L2 Cache** | 2 MB | 6 MB | 6 MB |
| **TDP** | 60W (configurable 42-60W) | 150W | 150W |
| **PCIe** | Gen 4 x8 | Gen 4 x16 | Gen 4 x16 |
| **NVLink** | No | No | No |
| **Form Factor** | Low-profile, single-slot | Single-slot | Single-slot |
| **Key Difference** | Edge inference | Datacenter inference | AWS-exclusive variant (more RT cores) |
| **Release Date** | Nov 2021 | Apr 2021 | 2021 |

### A10 Performance (TFLOPS)

| Precision | A10 |
|---|---|
| **FP32** | 31.2 |
| **FP16 Tensor** | 125 \| 250 (sparse) |
| **BF16 Tensor** | 125 \| 250 (sparse) |
| **TF32 Tensor** | 62.5 \| 125 (sparse) |
| **INT8 Tensor (TOPS)** | 250 \| 500 (sparse) |
| **INT4 Tensor (TOPS)** | 500 \| 1,000 (sparse) |

### A30, A40

| Specification | A30 | A40 |
|---|---|---|
| **Full Name** | NVIDIA A30 | NVIDIA A40 |
| **GPU Die** | GA100 | GA102 |
| **Architecture** | Ampere | Ampere |
| **Compute Capability** | 8.0 | 8.6 |
| **Process Node** | TSMC 7nm | Samsung 8nm |
| **SM Count** | 56 | 84 |
| **CUDA Cores** | 3,584 | 10,752 |
| **Tensor Cores (3rd Gen)** | 224 | 336 |
| **RT Cores** | N/A | 84 (2nd Gen) |
| **Memory Type** | HBM2e | GDDR6 ECC |
| **Memory Size** | 24 GB | 48 GB |
| **Memory Bus Width** | 3072-bit | 384-bit |
| **Memory Bandwidth** | 933 GB/s | 696 GB/s |
| **L2 Cache** | 24 MB | 6 MB |
| **TDP** | 165W | 300W |
| **PCIe** | Gen 4 x16 | Gen 4 x16 |
| **NVLink** | 3rd Gen (1 link) | 3rd Gen (1 link) |
| **MIG Support** | Yes (up to 4 instances) | No |
| **Form Factor** | Dual-slot, passive | Dual-slot, passive |
| **Release Date** | Apr 2021 | Oct 2020 |

### A30 Performance (TFLOPS)

| Precision | Dense | Sparse |
|---|---|---|
| **FP64** | 5.2 | N/A |
| **FP64 Tensor** | 10.3 | N/A |
| **FP32** | 10.3 | N/A |
| **TF32 Tensor** | 82 | 165 |
| **BF16 Tensor** | 165 | 330 |
| **FP16 Tensor** | 165 | 330 |
| **INT8 Tensor (TOPS)** | 330 | 661 |
| **INT4 Tensor (TOPS)** | 661 | 1,321 |

### A40 Performance (TFLOPS)

| Precision | Dense | Sparse |
|---|---|---|
| **FP32** | 37.4 | N/A |
| **TF32 Tensor** | 74.8 | 149.7 |
| **BF16 Tensor** | 149.7 | 299.4 |
| **FP16 Tensor** | 149.7 | 299.4 |
| **INT8 Tensor (TOPS)** | 299.4 | 598.7 |
| **INT4 Tensor (TOPS)** | 598.7 | 1,197.4 |

### A100 (40GB and 80GB)

| Specification | A100 SXM 40GB | A100 SXM 80GB | A100 PCIe 40GB | A100 PCIe 80GB |
|---|---|---|---|---|
| **Full Name** | NVIDIA A100-SXM4-40GB | NVIDIA A100-SXM4-80GB | NVIDIA A100-PCIE-40GB | NVIDIA A100-PCIE-80GB |
| **GPU Die** | GA100 | GA100 | GA100 | GA100 |
| **Architecture** | Ampere | Ampere | Ampere | Ampere |
| **Compute Capability** | 8.0 | 8.0 | 8.0 | 8.0 |
| **Process Node** | TSMC 7nm | TSMC 7nm | TSMC 7nm | TSMC 7nm |
| **Transistors** | 54.2 billion | 54.2 billion | 54.2 billion | 54.2 billion |
| **Die Size** | 826 mm^2 | 826 mm^2 | 826 mm^2 | 826 mm^2 |
| **SM Count** | 108 | 108 | 108 | 108 |
| **CUDA Cores (FP32)** | 6,912 | 6,912 | 6,912 | 6,912 |
| **FP64 Cores** | 3,456 | 3,456 | 3,456 | 3,456 |
| **Tensor Cores (3rd Gen)** | 432 | 432 | 432 | 432 |
| **Boost Clock** | 1,410 MHz | 1,410 MHz | 1,410 MHz | 1,410 MHz |
| **Memory Type** | HBM2e | HBM2e | HBM2e | HBM2e |
| **Memory Size** | 40 GB | 80 GB | 40 GB | 80 GB |
| **Memory Bus Width** | 5120-bit | 5120-bit | 5120-bit | 5120-bit |
| **Memory Bandwidth** | 1,555 GB/s | 2,039 GB/s | 1,555 GB/s | 2,039 GB/s |
| **L2 Cache** | 40 MB | 40 MB | 40 MB | 40 MB |
| **L1/Shared per SM** | 192 KB | 192 KB | 192 KB | 192 KB |
| **Max Shared Mem/SM** | 164 KB | 164 KB | 164 KB | 164 KB |
| **TDP** | 400W | 400W | 250W | 300W |
| **PCIe** | N/A (SXM) | N/A (SXM) | Gen 4 x16 | Gen 4 x16 |
| **NVLink** | 3rd Gen, 12 links, 600 GB/s | 3rd Gen, 12 links, 600 GB/s | 3rd Gen, optional bridge | 3rd Gen, optional bridge |
| **MIG** | Up to 7 instances | Up to 7 instances | Up to 7 instances | Up to 7 instances |
| **Form Factor** | SXM4 | SXM4 | Dual-slot PCIe | Dual-slot PCIe |
| **Release Date** | Jun 2020 | Nov 2020 | Jun 2020 | Nov 2020 |

### A100 Performance (TFLOPS, SXM 80GB)

| Precision | Dense | Sparse |
|---|---|---|
| **FP64** | 9.7 | N/A |
| **FP64 Tensor** | 19.5 | N/A |
| **FP32** | 19.5 | N/A |
| **TF32 Tensor** | 156 | 312 |
| **BF16 Tensor** | 312 | 624 |
| **FP16 Tensor** | 312 | 624 |
| **INT8 Tensor (TOPS)** | 624 | 1,248 |
| **INT4 Tensor (TOPS)** | 1,248 | 2,496 |
| **FP8** | Not supported | Not supported |

---

## 10. NVIDIA Datacenter GPUs - Ada Lovelace (L-Series)

| Specification | L4 | L40 | L40S |
|---|---|---|---|
| **Full Name** | NVIDIA L4 | NVIDIA L40 | NVIDIA L40S |
| **GPU Die** | AD104 | AD102 | AD102 |
| **Architecture** | Ada Lovelace | Ada Lovelace | Ada Lovelace |
| **Compute Capability** | 8.9 | 8.9 | 8.9 |
| **Process Node** | TSMC 4N | TSMC 4N | TSMC 4N |
| **SM Count** | 58 | 142 | 142 |
| **CUDA Cores** | 7,424 | 18,176 | 18,176 |
| **Tensor Cores (4th Gen)** | 232 | 568 | 568 |
| **RT Cores (3rd Gen)** | 58 | 142 | 142 |
| **Boost Clock** | 2,040 MHz | 2,490 MHz | 2,520 MHz |
| **Memory Type** | GDDR6 | GDDR6 ECC | GDDR6 ECC |
| **Memory Size** | 24 GB | 48 GB | 48 GB |
| **Memory Bus Width** | 192-bit | 384-bit | 384-bit |
| **Memory Bandwidth** | 300 GB/s | 864 GB/s | 864 GB/s |
| **L2 Cache** | 48 MB | 96 MB | 96 MB |
| **TDP** | 72W | 300W | 350W |
| **PCIe** | Gen 4 x16 | Gen 4 x16 | Gen 4 x16 |
| **NVLink** | No | No | No |
| **MIG** | No | No | No |
| **Form Factor** | Single-slot, low-profile | Dual-slot, passive | Dual-slot, passive |
| **Release Date** | Mar 2023 | 2023 | Oct 2023 |

### L-Series Performance (TFLOPS, Dense | Sparse)

| Precision | L4 | L40 | L40S |
|---|---|---|---|
| **FP32** | 30.3 | 90.5 | 91.6 |
| **TF32 Tensor** | 60 \| 120 | 181 \| 362 | 183 \| 366 |
| **BF16 Tensor** | 120 \| 242 | 362 \| 724 | 366 \| 733 |
| **FP16 Tensor** | 120 \| 242 | 362 \| 724 | 366 \| 733 |
| **FP8 Tensor** | 242 \| 485 | 724 \| 1,448 | 733 \| 1,466 |
| **INT8 Tensor (TOPS)** | 485 \| 970 | 1,448 \| 2,896 | 1,466 \| 2,932 |

### Key Differences: L40 vs L40S
- L40S has higher TDP (350W vs 300W) and slightly higher clocks, optimized for AI compute
- L40 is optimized for professional visualization (graphics + AI)
- Both share the same AD102 die with identical SM/core counts
- L40S has enhanced Tensor Core throughput scheduling for AI workloads

---

## 11. NVIDIA Datacenter GPUs - Hopper (H-Series)

### H100 Variants

| Specification | H100 SXM5 | H100 PCIe | H100 NVL |
|---|---|---|---|
| **Full Name** | NVIDIA H100 SXM5 80GB | NVIDIA H100 PCIe 80GB | NVIDIA H100 NVL 94GB |
| **GPU Die** | GH100 | GH100 | GH100 |
| **Architecture** | Hopper | Hopper | Hopper |
| **Compute Capability** | 9.0 | 9.0 | 9.0 |
| **Process Node** | TSMC 4N | TSMC 4N | TSMC 4N |
| **Transistors** | 80 billion | 80 billion | 80 billion |
| **Die Size** | 814 mm^2 | 814 mm^2 | 814 mm^2 |
| **SM Count** | 132 | 114 | 114 (per GPU) |
| **CUDA Cores (FP32)** | 16,896 | 14,592 | 14,592 (per GPU) |
| **FP64 Cores** | 8,448 | 7,296 | 7,296 (per GPU) |
| **Tensor Cores (4th Gen)** | 528 | 456 | 456 (per GPU) |
| **Boost Clock** | 1,830 MHz | 1,755 MHz | 1,785 MHz |
| **Memory Type** | HBM3 | HBM2e | HBM3 |
| **Memory Size** | 80 GB | 80 GB | 94 GB (per GPU) |
| **Memory Bus Width** | 5120-bit | 5120-bit | 5120-bit |
| **Memory Bandwidth** | 3,352 GB/s | 2,039 GB/s | 3,938 GB/s |
| **L2 Cache** | 50 MB | 50 MB | 50 MB |
| **L1/Shared per SM** | 256 KB | 256 KB | 256 KB |
| **Max Shared Mem/SM** | 228 KB | 228 KB | 228 KB |
| **TDP** | 700W | 350W | 400W (per GPU) |
| **PCIe** | Gen 5 x16 | Gen 5 x16 | Gen 5 x16 |
| **NVLink** | 4th Gen, 18 links, 900 GB/s | 4th Gen, optional | 4th Gen, 600 GB/s (NVLink bridge) |
| **MIG** | Up to 7 instances (10GB each) | Up to 7 instances | Up to 7 instances (12GB each) |
| **Form Factor** | SXM5 | Dual-slot PCIe | Dual PCIe (bridged pair) |
| **Release Date** | Mar 2023 | Mar 2023 | Mar 2024 |

### H100 Performance (TFLOPS)

| Precision | H100 SXM5 (Dense \| Sparse) | H100 PCIe (Dense \| Sparse) | H100 NVL (per GPU, Dense \| Sparse) |
|---|---|---|---|
| **FP64** | 34 | 26 | 30 |
| **FP64 Tensor** | 67 | 51 | 60 |
| **FP32** | 67 | 51 | 60 |
| **TF32 Tensor** | 989 \| 1,979 | 756 \| 1,513 | 835 \| 1,671 |
| **BF16 Tensor** | 1,979 \| 3,958 | 1,513 \| 3,026 | 1,671 \| 3,341 |
| **FP16 Tensor** | 1,979 \| 3,958 | 1,513 \| 3,026 | 1,671 \| 3,341 |
| **FP8 Tensor** | 3,958 \| 7,916 | 3,026 \| 6,052 | 3,341 \| 6,682 |
| **INT8 Tensor (TOPS)** | 3,958 \| 7,916 | 3,026 \| 6,052 | 3,341 \| 6,682 |

### H200

| Specification | H200 SXM |
|---|---|
| **Full Name** | NVIDIA H200 Tensor Core GPU |
| **GPU Die** | GH100 (same as H100) |
| **Architecture** | Hopper |
| **Compute Capability** | 9.0 |
| **Process Node** | TSMC 4N |
| **SM Count** | 132 |
| **CUDA Cores (FP32)** | 16,896 |
| **FP64 Cores** | 8,448 |
| **Tensor Cores (4th Gen)** | 528 |
| **Memory Type** | HBM3e |
| **Memory Size** | 141 GB |
| **Memory Bus Width** | 6144-bit |
| **Memory Bandwidth** | 4,800 GB/s |
| **L2 Cache** | 50 MB |
| **L1/Shared per SM** | 256 KB |
| **Max Shared Mem/SM** | 228 KB |
| **TDP** | 700W |
| **PCIe** | Gen 5 x16 |
| **NVLink** | 4th Gen, 18 links, 900 GB/s |
| **MIG** | Up to 7 instances |
| **Form Factor** | SXM5 |
| **Release Date** | Q1 2024 |

### H200 Performance (TFLOPS)

Same compute die as H100 SXM5, so identical TFLOPS numbers. The improvement is entirely in memory:

| Metric | H100 SXM5 | H200 SXM | Improvement |
|---|---|---|---|
| **Memory Capacity** | 80 GB HBM3 | 141 GB HBM3e | 1.76x |
| **Memory Bandwidth** | 3,352 GB/s | 4,800 GB/s | 1.43x |
| **FP8 Tensor** | 3,958 TFLOPS | 3,958 TFLOPS | Same |
| **FP16 Tensor** | 1,979 TFLOPS | 1,979 TFLOPS | Same |

### Key Hopper Features
- Transformer Engine with automatic FP8/FP16 dynamic precision
- Tensor Memory Accelerator (TMA) for efficient 5D tensor data movement
- Thread Block Clusters and Distributed Shared Memory
- Fourth-gen NVLink (900 GB/s) with NVLink Switch (up to 256 GPUs)
- Multi-Instance GPU (MIG) for up to 7 isolated partitions
- DPX instructions for dynamic programming acceleration

---

## 12. NVIDIA Datacenter GPUs - Blackwell (B-Series)

### B100, B200

| Specification | B100 | B200 |
|---|---|---|
| **Full Name** | NVIDIA B100 | NVIDIA B200 |
| **GPU Die** | GB100 (dual-die) | GB100 (dual-die) |
| **Architecture** | Blackwell | Blackwell |
| **Compute Capability** | 10.0 | 10.0 |
| **Process Node** | TSMC 4NP | TSMC 4NP |
| **Transistors** | 208 billion (dual-die) | 208 billion (dual-die) |
| **Die Architecture** | 2 dies, 10 TB/s NV-HBI interconnect | 2 dies, 10 TB/s NV-HBI interconnect |
| **SM Count** | ~280 | ~296 (148 per die) |
| **CUDA Cores** | 16,896 | 20,480 |
| **Tensor Cores (5th Gen)** | ~560 | ~592 |
| **Memory Type** | HBM3e | HBM3e |
| **Memory Size** | 192 GB | 192 GB |
| **Memory Bandwidth** | 8 TB/s | 8 TB/s |
| **TDP** | 700W | 1,000W |
| **NVLink** | 5th Gen, 1.8 TB/s | 5th Gen, 1.8 TB/s |
| **PCIe** | Gen 5 x16 | Gen 6 x16 |
| **MIG** | Up to 7 instances | Up to 7 instances |
| **Form Factor** | SXM (drop-in for H100 SXM) | SXM |
| **Release Date** | 2024 | 2024 |

### B100 Performance (TFLOPS/PFLOPS)

| Precision | Dense | Sparse |
|---|---|---|
| **FP64** | 30 TF | N/A |
| **FP64 Tensor** | 30 TF | N/A |
| **FP32** | 60 TF | N/A |
| **TF32 Tensor** | 1.8 PF | 3.5 PF |
| **BF16 Tensor** | 3.5 PF | 7 PF |
| **FP16 Tensor** | 3.5 PF | 7 PF |
| **FP8 Tensor** | 7 PF | 14 PF |
| **INT8 (TOPS)** | 7 PO | 14 PO |
| **FP4 Tensor** | 14 PF | 28 PF |

### B200 Performance (TFLOPS/PFLOPS)

| Precision | Dense | Sparse |
|---|---|---|
| **FP64** | 40 TF | N/A |
| **FP64 Tensor** | 40 TF | N/A |
| **FP32** | 80 TF | N/A |
| **TF32 Tensor** | 2.2 PF | 4.5 PF |
| **BF16 Tensor** | 4.5 PF | 9 PF |
| **FP16 Tensor** | 4.5 PF | 9 PF |
| **FP8 Tensor** | 9 PF | 18 PF |
| **INT8 (TOPS)** | 9 PO | 18 PO |
| **FP4 Tensor** | 18 PF | 36 PF |

### B300 (Blackwell Ultra)

| Specification | B300 |
|---|---|
| **Full Name** | NVIDIA B300 (Blackwell Ultra) |
| **GPU Die** | GB100 Ultra (dual-die) |
| **Architecture** | Blackwell Ultra |
| **Compute Capability** | 10.0 |
| **Process Node** | TSMC 4NP |
| **Transistors** | 208 billion (dual-die) |
| **SM Count** | 160 (80 per die) |
| **CUDA Cores** | 20,480 |
| **Tensor Cores (5th Gen)** | 640 |
| **Memory Type** | HBM3e (12-high stacks) |
| **Memory Size** | 288 GB |
| **Memory Bandwidth** | 8 TB/s |
| **TDP** | 1,400W |
| **NVLink** | 5th Gen, 1.8 TB/s |
| **PCIe** | Gen 6 x16 (256 GB/s) |
| **MIG** | Up to 7 instances |
| **Boost Clock** | 2,600 MHz |
| **Form Factor** | SXM |
| **Release Date** | H1 2025 |

### B300 Performance (TFLOPS/PFLOPS)

| Precision | Dense | Sparse |
|---|---|---|
| **FP4 (NVFP4)** | 15 PF | 30 PF |
| **FP8** | 10 PF | 20 PF |
| **FP16/BF16** | 5 PF | 10 PF |
| **TF32** | 2.5 PF | 5 PF |
| **FP32** | ~80 TF | N/A |
| **FP64** | ~40 TF | N/A |

### GB200 Grace Blackwell Superchip

| Specification | GB200 |
|---|---|
| **Full Name** | NVIDIA GB200 Grace Blackwell Superchip |
| **Components** | 1x Grace CPU + 2x B200 GPUs |
| **CPU** | 72x Arm Neoverse V2 cores, 480 GB LPDDR5x, 512 GB/s |
| **GPU Memory (total)** | 384 GB HBM3e (192 GB per GPU) |
| **GPU Bandwidth (total)** | 16 TB/s (8 TB/s per GPU) |
| **CPU-GPU Interconnect** | NVLink-C2C, 900 GB/s per GPU |
| **GPU-GPU NVLink** | 5th Gen, 1.8 TB/s per GPU |
| **FP8 Performance** | 10 PFLOPS (per superchip, without sparsity) |
| **FP4 Performance** | 20 PFLOPS (per superchip, without sparsity) |
| **TDP** | ~2,700W (total system) |
| **NVL72 System** | 36x GB200 = 72 GPUs + 36 CPUs |
| **NVL72 FP4** | 720 PFLOPS |
| **NVL72 Memory** | 13.5 TB HBM3e |
| **Form Factor** | Liquid-cooled rack |
| **Release Date** | 2024 |

### GB300 Grace Blackwell Ultra Superchip

| Specification | GB300 |
|---|---|
| **Full Name** | NVIDIA GB300 Grace Blackwell Ultra Superchip |
| **Components** | 1x Grace CPU + 2x B300 GPUs |
| **CPU** | 72x Arm Neoverse V2 cores @ 3.1 GHz |
| **GPU Memory (total)** | 576 GB HBM3e (288 GB per GPU) |
| **CPU Memory** | Up to 496 GB LPDDR5x |
| **Unified Memory** | Up to 784 GB (CPU + GPU coherent) |
| **GPU Bandwidth (total)** | 16 TB/s |
| **CPU-GPU Interconnect** | NVLink-C2C, 900 GB/s per GPU |
| **GPU-GPU NVLink** | 5th Gen, 1.8 TB/s per GPU |
| **FP4 Performance** | ~20 PFLOPS (per superchip) |
| **TDP** | ~3,500W (estimated per superchip) |
| **NVL72 System** | 36x GB300 = 72 GPUs + 36 CPUs |
| **NVL72 FP4** | 1.1 EFLOPS |
| **NVL72 Memory** | 20.7 TB HBM3e |
| **NVL72 NVLink BW** | 130 TB/s aggregate |
| **NVL72 Power** | ~120 kW |
| **Form Factor** | Liquid-cooled rack |
| **Release Date** | H1 2025 |

### Key Blackwell Datacenter Features
- Dual-die design with 10 TB/s NV-HBI chip-to-chip interconnect
- Fifth-gen Tensor Cores with FP4/FP6/FP8 native support
- Second-gen Transformer Engine with micro-tensor scaling
- NVLink 5 at 1.8 TB/s per GPU (576-GPU NVLink domain)
- Hardware decompression engine (LZ4, Snappy, Deflate)
- Confidential Computing with TEE-I/O
- RAS Engine with AI-driven predictive failure management

---

## 13. AMD Instinct Accelerators

### MI250X

| Specification | MI250X |
|---|---|
| **Full Name** | AMD Instinct MI250X |
| **Architecture** | CDNA 2 |
| **Process Node** | TSMC 6nm (compute), TSMC 7nm (I/O) |
| **Design** | Multi-Chip Module (MCM), 2 GCDs |
| **Compute Units (Total)** | 220 (110 per die) |
| **Stream Processors** | 14,080 (64 per CU) |
| **Matrix Cores** | 880 |
| **Boost Clock** | 1,700 MHz |
| **Memory Type** | HBM2e |
| **Memory Size** | 128 GB (64 GB per die) |
| **Memory Bus Width** | 8192-bit |
| **Memory Bandwidth** | 3,276 GB/s |
| **L2 Cache** | 16 MB (8 MB per die) |
| **Infinity Cache** | 0 (CDNA 2 does not have Infinity Cache) |
| **TDP** | 560W |
| **Interconnect** | AMD Infinity Fabric (IF), 4th Gen |
| **PCIe** | Gen 4 x16 |
| **Form Factor** | OAM |
| **Release Date** | Nov 2021 |

### MI250X Performance (TFLOPS)

| Precision | Peak TFLOPS |
|---|---|
| **FP64** | 47.9 |
| **FP64 Matrix** | 95.7 |
| **FP32** | 47.9 |
| **FP32 Matrix** | 95.7 |
| **FP16** | 383 |
| **BF16** | 383 |
| **INT8 (TOPS)** | 383 |
| **INT4 (TOPS)** | 383 |

### MI300X

| Specification | MI300X |
|---|---|
| **Full Name** | AMD Instinct MI300X |
| **Architecture** | CDNA 3 |
| **Process Node** | TSMC 5nm (compute) + 6nm (I/O) |
| **Transistors** | 153 billion |
| **Design** | 3D Chiplet (8 XCDs + 4 IODs) |
| **Compute Units** | 304 |
| **Stream Processors** | 19,456 (64 per CU) |
| **Matrix Cores** | 1,216 |
| **Boost Clock** | 2,100 MHz |
| **Memory Type** | HBM3 |
| **Memory Size** | 192 GB |
| **Memory Bus Width** | 8192-bit |
| **Memory Bandwidth** | 5,325 GB/s (5.3 TB/s) |
| **L1 Cache per CU** | 32 KB |
| **L2 Cache per XCD** | 4 MB |
| **Total L2 Cache** | 32 MB |
| **Infinity Cache (LLC)** | 256 MB |
| **TDP** | 750W |
| **Interconnect** | 4th Gen Infinity Fabric |
| **PCIe** | Gen 5 x16 |
| **Form Factor** | OAM |
| **Release Date** | Dec 2023 |

### MI300X Performance (TFLOPS)

| Precision | Dense | Sparse (2:4) |
|---|---|---|
| **FP64** | 81.7 | N/A |
| **FP64 Matrix** | 163.4 | N/A |
| **FP32** | 163.4 | N/A |
| **TF32 (TP32)** | 653.7 | 1,307.4 |
| **FP16** | 1,307.4 | 2,614.9 |
| **BF16** | 1,307.4 | 2,614.9 |
| **FP8** | 2,614.9 | 5,229.8 |
| **INT8 (TOPS)** | 2,614.9 | 5,229.8 |

### MI300A (APU)

| Specification | MI300A |
|---|---|
| **Full Name** | AMD Instinct MI300A |
| **Architecture** | CDNA 3 + Zen 4 |
| **Process Node** | TSMC 5nm + 6nm |
| **Transistors** | 146 billion |
| **Design** | APU: 6 XCDs + 3 Zen 4 CCDs + 4 IODs |
| **GPU Compute Units** | 228 (38 per XCD) |
| **GPU Stream Processors** | 14,592 |
| **GPU Matrix Cores** | 912 |
| **CPU Cores** | 24x Zen 4 |
| **Boost Clock (GPU)** | 2,100 MHz |
| **Memory Type** | HBM3 (unified CPU+GPU) |
| **Memory Size** | 128 GB |
| **Memory Bus Width** | 8192-bit |
| **Memory Bandwidth** | 5,325 GB/s (5.3 TB/s) |
| **Total L2 Cache** | 24 MB (GPU) + 32 MB (CPU L3) |
| **Infinity Cache** | 256 MB |
| **TDP** | 550W (air) / 760W (liquid) |
| **PCIe** | Gen 5 x16 |
| **Form Factor** | OAM |
| **Release Date** | Dec 2023 |

### MI300A Performance (TFLOPS)

| Precision | Peak TFLOPS |
|---|---|
| **FP64** | 61.3 |
| **FP64 Matrix** | 122.6 |
| **FP32** | 122.6 |
| **FP16** | 980.6 |
| **BF16** | 980.6 |
| **FP8** | 1,961.2 |
| **INT8 (TOPS)** | 1,961.2 |

### MI325X

| Specification | MI325X |
|---|---|
| **Full Name** | AMD Instinct MI325X |
| **Architecture** | CDNA 3 |
| **Process Node** | TSMC 5nm + 6nm |
| **Design** | 3D Chiplet (8 XCDs + 4 IODs) |
| **Compute Units** | 304 |
| **Stream Processors** | 19,456 |
| **Matrix Cores** | 1,216 |
| **Boost Clock** | 2,100 MHz |
| **Memory Type** | HBM3e (upgraded from MI300X) |
| **Memory Size** | 256 GB |
| **Memory Bus Width** | 8192-bit |
| **Memory Bandwidth** | 6,000 GB/s (6 TB/s) |
| **L2 Cache** | 32 MB |
| **Infinity Cache** | 256 MB |
| **TDP** | 1,000W |
| **Interconnect** | 4th Gen Infinity Fabric |
| **PCIe** | Gen 5 x16 |
| **Form Factor** | OAM |
| **Release Date** | Q4 2024 |

### MI325X Performance (TFLOPS)

Same compute die as MI300X. Performance numbers identical:

| Precision | Dense | Sparse |
|---|---|---|
| **FP16** | 1,307.4 | 2,614.9 |
| **BF16** | 1,307.4 | 2,614.9 |
| **FP8** | 2,614.9 | 5,229.8 |
| **INT8 (TOPS)** | 2,614.9 | 5,229.8 |

Key difference vs MI300X: Memory upgrade (256 GB HBM3e @ 6 TB/s vs 192 GB HBM3 @ 5.3 TB/s) and higher TDP (1,000W vs 750W).

### MI350X (CDNA 4)

| Specification | MI350X |
|---|---|
| **Full Name** | AMD Instinct MI350X |
| **Architecture** | CDNA 4 |
| **Process Node** | TSMC 3nm (compute) + 6nm (I/O) |
| **Transistors** | 185 billion |
| **Design** | 3D Multi-Chiplet (8 XCDs) |
| **Compute Units** | 256 (32 per XCD) |
| **Stream Processors** | 32,768 (128 per CU) |
| **Matrix Cores** | 1,024 |
| **Memory Type** | HBM3e |
| **Memory Size** | 288 GB |
| **Memory Bus Width** | 8192-bit |
| **Memory Bandwidth** | 8,000 GB/s (8 TB/s) |
| **L1 Cache per CU** | 32 KB |
| **L2 Cache per XCD** | 4 MB |
| **Total L2 Cache** | 32 MB |
| **Infinity Cache** | 256 MB |
| **TDP** | 1,000W (air-cooled) |
| **Boost Clock** | 2,200 MHz |
| **PCIe** | Gen 5 x16 |
| **Form Factor** | OAM |
| **Release Date** | H1 2025 |

### MI350X Performance (TFLOPS)

| Precision | Peak Performance |
|---|---|
| **FP64** | ~80 TFLOPS |
| **FP32** | ~160 TFLOPS |
| **FP16/BF16** | ~4,600 TFLOPS (per-platform: 36.9 PF for 8-GPU) |
| **FP8** | ~9,200 TFLOPS (per-platform: 73.9 PF for 8-GPU) |
| **FP6** | ~14,000 TFLOPS |
| **FP4** | ~20,000 TFLOPS |

### Key AMD Instinct Features
- **CDNA 3 (MI300X/MI325X):** 3D chiplet stacking with TSV, 256 MB Infinity Cache, massive HBM capacity
- **CDNA 4 (MI350X):** Native FP4/FP6 support, 128 stream processors per CU (2x CDNA 3), 3nm process
- **Unified Memory (MI300A):** CPU and GPU share HBM3 address space with hardware coherency
- **Software:** ROCm, PyTorch, vLLM, DeepSpeed, JAX support

---

## 14. Intel Gaudi Accelerators

### Gaudi 2

| Specification | Gaudi 2 |
|---|---|
| **Full Name** | Intel Gaudi 2 AI Accelerator (HL-225H) |
| **Architecture** | Heterogeneous (MME + TPC) |
| **Process Node** | 7nm |
| **MME (Matrix Multiply Engines)** | 2 |
| **TPC (Tensor Processor Cores)** | 24 |
| **On-die SRAM** | 48 MB |
| **Memory Type** | HBM2e |
| **Memory Size** | 96 GB |
| **Memory Bandwidth** | 2,450 GB/s (2.45 TB/s) |
| **TDP** | 600W |
| **Networking (Integrated)** | 24x 100 GbE RoCE v2 RDMA NICs |
| **Network Bandwidth** | 2,400 Gb/s (300 GB/s) bidirectional |
| **Host Interface** | PCIe Gen 4 x16 |
| **Form Factor** | OAM mezzanine card |
| **Release Date** | 2022 |

### Gaudi 2 Performance (TFLOPS)

| Precision | Peak TFLOPS |
|---|---|
| **FP32 (vector)** | ~7 |
| **BF16 (matrix)** | ~432 |
| **FP16** | ~432 |
| **FP8** | ~865 |
| **TF32** | Supported |

### Gaudi 3

| Specification | Gaudi 3 |
|---|---|
| **Full Name** | Intel Gaudi 3 AI Accelerator (HL-325L) |
| **Architecture** | Dual-die heterogeneous |
| **Process Node** | TSMC 5nm |
| **MME (Matrix Multiply Engines)** | 8 (4 per die) |
| **TPC (Tensor Processor Cores)** | 64 (32 per die) |
| **On-die SRAM** | 96 MB (48 MB per die) |
| **SRAM Bandwidth** | 12.8 TB/s |
| **Memory Type** | HBM2e |
| **Memory Size** | 128 GB |
| **Memory Bandwidth** | 3,700 GB/s (3.7 TB/s) |
| **TDP (air-cooled)** | 900W |
| **TDP (liquid-cooled)** | 1,200W |
| **Networking (Integrated)** | 24x 200 GbE RoCE v2 RDMA NICs |
| **Network Bandwidth** | 4,800 Gb/s (600 GB/s) bidirectional |
| **Host Interface** | PCIe Gen 5 x16 |
| **Form Factor** | OAM mezzanine card |
| **Media Engines** | 14 (H.265, H.264, JPEG, VP9) |
| **Release Date** | Q4 2024 |

### Gaudi 3 Performance (TFLOPS)

| Precision | Peak TFLOPS |
|---|---|
| **FP32 (vector)** | 14.3 |
| **BF16 (matrix)** | 1,835 |
| **FP16** | Supported |
| **FP8** | 1,835 |
| **TF32** | Supported |

### Gaudi 3 vs Gaudi 2 Improvement

| Metric | Gaudi 2 | Gaudi 3 | Improvement |
|---|---|---|---|
| **FP8 TFLOPS** | ~865 | 1,835 | 2.1x |
| **BF16 Matrix TFLOPS** | ~432 | 1,835 | 4.2x |
| **Memory** | 96 GB HBM2e | 128 GB HBM2e | 1.33x |
| **Memory BW** | 2.45 TB/s | 3.7 TB/s | 1.51x |
| **Network BW** | 300 GB/s | 600 GB/s | 2x |
| **TDP** | 600W | 900W | 1.5x |
| **Process** | 7nm | 5nm | -- |

### Key Intel Gaudi Features
- **Integrated Networking:** 24x Ethernet RDMA ports per chip (no external NICs needed)
- **MME + TPC Design:** Separate matrix (MME) and tensor processor (TPC) engines
- **Large SRAM:** 96 MB on-die scratchpad (Gaudi 3)
- **Media Processing:** Built-in video decode for vision AI pipelines
- **Software:** PyTorch with SynapseAI SDK, vLLM, DeepSpeed, Hugging Face Optimum-Habana
- **Cost Advantage:** Priced below H100/H200 for price-performance ratio

---

## 15. Master Comparison Tables

### 15.1 Consumer GPU Comparison

| GPU | Arch | CC | SMs | CUDA | Tensor | Mem (GB) | Mem Type | BW (GB/s) | L2 (MB) | TDP (W) | FP32 TF | FP16 TC TF (Dense) | FP8 TC TF (Dense) | Release |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RTX 3060 | Ampere | 8.6 | 28 | 3,584 | 112 | 12 | GDDR6 | 360 | 3 | 170 | 12.7 | 51 | -- | Feb 2021 |
| RTX 3070 | Ampere | 8.6 | 46 | 5,888 | 184 | 8 | GDDR6 | 448 | 4 | 220 | 20.3 | 81 | -- | Oct 2020 |
| RTX 3080 | Ampere | 8.6 | 68 | 8,704 | 272 | 10 | GDDR6X | 760 | 5 | 320 | 29.8 | 119 | -- | Sep 2020 |
| RTX 3090 | Ampere | 8.6 | 82 | 10,496 | 328 | 24 | GDDR6X | 936 | 6 | 350 | 35.6 | 142 | -- | Sep 2020 |
| RTX 4060 | Ada | 8.9 | 24 | 3,072 | 96 | 8 | GDDR6 | 272 | 24 | 115 | 15.1 | 30.2 | 60.5 | Jun 2023 |
| RTX 4070 | Ada | 8.9 | 46 | 5,888 | 184 | 12 | GDDR6X | 504 | 36 | 200 | 29.1 | 58.3 | 116.5 | Apr 2023 |
| RTX 4080 | Ada | 8.9 | 76 | 9,728 | 304 | 16 | GDDR6X | 717 | 64 | 320 | 48.7 | 97.5 | 194.9 | Nov 2022 |
| RTX 4090 | Ada | 8.9 | 128 | 16,384 | 512 | 24 | GDDR6X | 1,008 | 72 | 450 | 82.6 | 165.2 | 330.3 | Oct 2022 |
| RTX 5070 | BW | 12.0 | 48 | 6,144 | 192 | 12 | GDDR7 | 672 | 36 | 250 | 30.8 | 61.7 | 123.4 | Mar 2025 |
| RTX 5080 | BW | 12.0 | 84 | 10,752 | 336 | 16 | GDDR7 | 960 | 64 | 360 | 54.2 | 108.4 | 216.8 | Jan 2025 |
| RTX 5090 | BW | 12.0 | 170 | 21,760 | 680 | 32 | GDDR7 | 1,792 | 98 | 575 | 104.8 | 209.5 | 419.0 | Jan 2025 |

### 15.2 Workstation GPU Comparison

| GPU | Arch | CC | SMs | CUDA | Tensor | Mem (GB) | Mem Type | BW (GB/s) | L2 (MB) | TDP (W) | ECC | FP32 TF | Release |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RTX A4000 | Ampere | 8.6 | 48 | 6,144 | 192 | 16 | GDDR6 | 448 | 4 | 140 | Yes | 19.2 | Apr 2021 |
| RTX A5000 | Ampere | 8.6 | 64 | 8,192 | 256 | 24 | GDDR6 | 768 | 6 | 230 | Yes | 27.8 | Apr 2021 |
| RTX A6000 | Ampere | 8.6 | 84 | 10,752 | 336 | 48 | GDDR6 | 768 | 6 | 300 | Yes | 38.7 | Dec 2020 |
| RTX 4000 Ada | Ada | 8.9 | 48 | 6,144 | 192 | 20 | GDDR6 | 360 | 32 | 130 | Yes | 26.7 | 2023 |
| RTX 5000 Ada | Ada | 8.9 | 100 | 12,800 | 400 | 32 | GDDR6 | 576 | 64 | 250 | Yes | 65.3 | 2023 |
| RTX 6000 Ada | Ada | 8.9 | 142 | 18,176 | 568 | 48 | GDDR6 | 960 | 96 | 300 | Yes | 91.1 | 2023 |
| RTX PRO 6000 BW | BW | 12.0 | 188 | 24,064 | 752 | 96 | GDDR7 | 1,792 | 128 | 600 | Yes | 125 | 2025 |

### 15.3 Datacenter GPU Comparison

| GPU | Arch | CC | SMs | CUDA | Tensor | Mem (GB) | Mem Type | BW (GB/s) | L2 (MB) | TDP (W) | NVLink (GB/s) | PCIe | Release |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T4 | Turing | 7.5 | 40 | 2,560 | 320 | 16 | GDDR6 | 320 | 4 | 70 | -- | Gen 3 | Sep 2018 |
| A2 | Ampere | 8.6 | 10 | 1,280 | 40 | 16 | GDDR6 | 200 | 2 | 60 | -- | Gen 4 | Nov 2021 |
| A10 | Ampere | 8.6 | 72 | 9,216 | 288 | 24 | GDDR6 | 600 | 6 | 150 | -- | Gen 4 | Apr 2021 |
| A10G | Ampere | 8.6 | 72 | 9,216 | 320 | 24 | GDDR6 | 600 | 6 | 150 | -- | Gen 4 | 2021 |
| A30 | Ampere | 8.0 | 56 | 3,584 | 224 | 24 | HBM2e | 933 | 24 | 165 | 1 link | Gen 4 | Apr 2021 |
| A40 | Ampere | 8.6 | 84 | 10,752 | 336 | 48 | GDDR6 | 696 | 6 | 300 | 1 link | Gen 4 | Oct 2020 |
| A100 40 SXM | Ampere | 8.0 | 108 | 6,912 | 432 | 40 | HBM2e | 1,555 | 40 | 400 | 600 | -- | Jun 2020 |
| A100 80 SXM | Ampere | 8.0 | 108 | 6,912 | 432 | 80 | HBM2e | 2,039 | 40 | 400 | 600 | -- | Nov 2020 |
| A100 40 PCIe | Ampere | 8.0 | 108 | 6,912 | 432 | 40 | HBM2e | 1,555 | 40 | 250 | bridge | Gen 4 | Jun 2020 |
| A100 80 PCIe | Ampere | 8.0 | 108 | 6,912 | 432 | 80 | HBM2e | 2,039 | 40 | 300 | bridge | Gen 4 | Nov 2020 |
| L4 | Ada | 8.9 | 58 | 7,424 | 232 | 24 | GDDR6 | 300 | 48 | 72 | -- | Gen 4 | Mar 2023 |
| L40 | Ada | 8.9 | 142 | 18,176 | 568 | 48 | GDDR6 | 864 | 96 | 300 | -- | Gen 4 | 2023 |
| L40S | Ada | 8.9 | 142 | 18,176 | 568 | 48 | GDDR6 | 864 | 96 | 350 | -- | Gen 4 | Oct 2023 |
| H100 SXM5 | Hopper | 9.0 | 132 | 16,896 | 528 | 80 | HBM3 | 3,352 | 50 | 700 | 900 | Gen 5 | Mar 2023 |
| H100 PCIe | Hopper | 9.0 | 114 | 14,592 | 456 | 80 | HBM2e | 2,039 | 50 | 350 | opt | Gen 5 | Mar 2023 |
| H100 NVL | Hopper | 9.0 | 114 | 14,592 | 456 | 94 | HBM3 | 3,938 | 50 | 400 | 600 | Gen 5 | Mar 2024 |
| H200 | Hopper | 9.0 | 132 | 16,896 | 528 | 141 | HBM3e | 4,800 | 50 | 700 | 900 | Gen 5 | Q1 2024 |
| B100 | BW | 10.0 | ~280 | 16,896 | ~560 | 192 | HBM3e | 8,000 | ~192 | 700 | 1,800 | Gen 5 | 2024 |
| B200 | BW | 10.0 | ~296 | 20,480 | ~592 | 192 | HBM3e | 8,000 | ~192 | 1,000 | 1,800 | Gen 6 | 2024 |
| B300 | BW Ultra | 10.0 | 160 | 20,480 | 640 | 288 | HBM3e | 8,000 | ~192 | 1,400 | 1,800 | Gen 6 | H1 2025 |

### 15.4 Datacenter Tensor Core Performance (TFLOPS, Dense)

| GPU | FP64 | FP64 TC | FP32 | TF32 TC | BF16 TC | FP16 TC | FP8 TC | INT8 (TOPS) | FP4 TC |
|---|---|---|---|---|---|---|---|---|---|
| T4 | 0.25 | -- | 8.1 | -- | -- | 65 | -- | 130 | -- |
| A10 | 0.49 | -- | 31.2 | 62.5 | 125 | 125 | -- | 250 | -- |
| A30 | 5.2 | 10.3 | 10.3 | 82 | 165 | 165 | -- | 330 | -- |
| A40 | 0.58 | -- | 37.4 | 74.8 | 149.7 | 149.7 | -- | 299.4 | -- |
| A100 SXM 80 | 9.7 | 19.5 | 19.5 | 156 | 312 | 312 | -- | 624 | -- |
| L4 | 0.12 | -- | 30.3 | 60 | 120 | 120 | 242 | 485 | -- |
| L40S | 0.36 | -- | 91.6 | 183 | 366 | 366 | 733 | 1,466 | -- |
| H100 SXM | 34 | 67 | 67 | 989 | 1,979 | 1,979 | 3,958 | 3,958 | -- |
| H100 PCIe | 26 | 51 | 51 | 756 | 1,513 | 1,513 | 3,026 | 3,026 | -- |
| H200 | 34 | 67 | 67 | 989 | 1,979 | 1,979 | 3,958 | 3,958 | -- |
| B100 | 30 | 30 | 60 | 1,800 | 3,500 | 3,500 | 7,000 | 7,000 | 14,000 |
| B200 | 40 | 40 | 80 | 2,200 | 4,500 | 4,500 | 9,000 | 9,000 | 18,000 |
| B300 | ~40 | ~40 | ~80 | 2,500 | 5,000 | 5,000 | 10,000 | 10,000 | 15,000 |

### 15.5 AMD Instinct Comparison

| GPU | Arch | CUs | Stream Proc | Mem (GB) | Mem Type | BW (TB/s) | TDP (W) | FP64 TF | FP16 TF | FP8 TF | FP4 TF | Release |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MI250X | CDNA 2 | 220 | 14,080 | 128 | HBM2e | 3.3 | 560 | 47.9 | 383 | -- | -- | Nov 2021 |
| MI300X | CDNA 3 | 304 | 19,456 | 192 | HBM3 | 5.3 | 750 | 81.7 | 1,307 | 2,615 | -- | Dec 2023 |
| MI300A | CDNA 3 | 228 | 14,592 | 128 | HBM3 | 5.3 | 760 | 61.3 | 981 | 1,961 | -- | Dec 2023 |
| MI325X | CDNA 3 | 304 | 19,456 | 256 | HBM3e | 6.0 | 1,000 | 81.7 | 1,307 | 2,615 | -- | Q4 2024 |
| MI350X | CDNA 4 | 256 | 32,768 | 288 | HBM3e | 8.0 | 1,000 | ~80 | ~4,600 | ~9,200 | ~20,000 | H1 2025 |

### 15.6 Intel Gaudi Comparison

| Specification | Gaudi 2 | Gaudi 3 |
|---|---|---|
| **Architecture** | Heterogeneous | Dual-die heterogeneous |
| **Process** | 7nm | 5nm |
| **MME Engines** | 2 | 8 |
| **TPC Cores** | 24 | 64 |
| **SRAM** | 48 MB | 96 MB |
| **Memory** | 96 GB HBM2e | 128 GB HBM2e |
| **Memory BW** | 2.45 TB/s | 3.7 TB/s |
| **SRAM BW** | -- | 12.8 TB/s |
| **BF16 Matrix TFLOPS** | ~432 | 1,835 |
| **FP8 TFLOPS** | ~865 | 1,835 |
| **Network (Integrated)** | 24x 100 GbE | 24x 200 GbE |
| **Network BW** | 300 GB/s | 600 GB/s |
| **TDP (air)** | 600W | 900W |
| **TDP (liquid)** | -- | 1,200W |
| **PCIe** | Gen 4 x16 | Gen 5 x16 |
| **Release** | 2022 | Q4 2024 |

### 15.7 Memory Bandwidth Ranking (All GPUs)

| Rank | GPU | Bandwidth | Memory Type | Capacity |
|---|---|---|---|---|
| 1 | B100 / B200 / B300 / MI350X | 8,000 GB/s | HBM3e | 192-288 GB |
| 2 | MI325X | 6,000 GB/s | HBM3e | 256 GB |
| 3 | MI300X / MI300A | 5,325 GB/s | HBM3 | 128-192 GB |
| 4 | H200 | 4,800 GB/s | HBM3e | 141 GB |
| 5 | H100 NVL | 3,938 GB/s | HBM3 | 94 GB |
| 6 | Gaudi 3 | 3,700 GB/s | HBM2e | 128 GB |
| 7 | H100 SXM5 | 3,352 GB/s | HBM3 | 80 GB |
| 8 | MI250X | 3,276 GB/s | HBM2e | 128 GB |
| 9 | Gaudi 2 | 2,450 GB/s | HBM2e | 96 GB |
| 10 | A100 80GB SXM/PCIe | 2,039 GB/s | HBM2e | 80 GB |
| 11 | H100 PCIe | 2,039 GB/s | HBM2e | 80 GB |
| 12 | RTX 5090 / RTX PRO 6000 | 1,792 GB/s | GDDR7 | 32-96 GB |
| 13 | A100 40GB | 1,555 GB/s | HBM2e | 40 GB |
| 14 | RTX 4090 | 1,008 GB/s | GDDR6X | 24 GB |
| 15 | RTX 6000 Ada / RTX 5080 | 960 GB/s | GDDR6/7 | 16-48 GB |
| 16 | RTX 3090 | 936 GB/s | GDDR6X | 24 GB |
| 17 | A30 | 933 GB/s | HBM2e | 24 GB |
| 18 | L40 / L40S | 864 GB/s | GDDR6 | 48 GB |
| 19 | RTX A5000 / RTX A6000 | 768 GB/s | GDDR6 | 24-48 GB |
| 20 | RTX 3080 | 760 GB/s | GDDR6X | 10 GB |
| 21 | RTX 4080 | 717 GB/s | GDDR6X | 16 GB |
| 22 | A40 | 696 GB/s | GDDR6 | 48 GB |
| 23 | RTX 5070 | 672 GB/s | GDDR7 | 12 GB |
| 24 | A10 / A10G | 600 GB/s | GDDR6 | 24 GB |
| 25 | RTX 5000 Ada | 576 GB/s | GDDR6 | 32 GB |
| 26 | RTX 4070 | 504 GB/s | GDDR6X | 12 GB |
| 27 | RTX A4000 / RTX 3070 | 448 GB/s | GDDR6 | 8-16 GB |
| 28 | RTX 3060 / RTX 4000 Ada | 360 GB/s | GDDR6 | 12-20 GB |
| 29 | T4 | 320 GB/s | GDDR6 | 16 GB |
| 30 | L4 | 300 GB/s | GDDR6 | 24 GB |
| 31 | RTX 4060 | 272 GB/s | GDDR6 | 8 GB |
| 32 | A2 | 200 GB/s | GDDR6 | 16 GB |

### 15.8 FLOPS/Watt Efficiency (FP16 Tensor Core Dense TFLOPS per Watt)

| GPU | FP16 TC (Dense) | TDP (W) | TFLOPS/Watt |
|---|---|---|---|
| L4 | 120 | 72 | 1.67 |
| T4 | 65 | 70 | 0.93 |
| RTX 4060 | 30.2 | 115 | 0.26 |
| RTX 4090 | 165.2 | 450 | 0.37 |
| RTX 5090 | 209.5 | 575 | 0.36 |
| A100 SXM 80 | 312 | 400 | 0.78 |
| H100 SXM | 1,979 | 700 | 2.83 |
| H200 | 1,979 | 700 | 2.83 |
| B200 | 4,500 | 1,000 | 4.50 |
| B300 | 5,000 | 1,400 | 3.57 |
| MI300X | 1,307 | 750 | 1.74 |
| MI350X | ~4,600 | 1,000 | 4.60 |
| Gaudi 3 | 1,835 (BF16) | 900 | 2.04 |

### 15.9 Compute Capability Quick Reference

| CC | Architecture | Key GPUs | Max Threads/SM | Max Shared/SM |
|---|---|---|---|---|
| 7.5 | Turing | T4 | 1,024 | 64 KB |
| 8.0 | Ampere (DC) | A100, A30 | 2,048 | 164 KB |
| 8.6 | Ampere (Consumer) | RTX 30xx, A2, A10, A10G, A40, RTX Axxxx | 1,536 | 100 KB |
| 8.9 | Ada Lovelace | RTX 40xx, L4, L40, L40S, RTX xxxx Ada | 1,536 | 100 KB |
| 9.0 | Hopper | H100, H200 | 2,048 | 228 KB |
| 10.0 | Blackwell (DC) | B100, B200, B300 | 2,048 | 228 KB |
| 12.0 | Blackwell (Consumer) | RTX 50xx, RTX PRO 6000 | 2,048 | 228 KB |

### 15.10 Precision Support Matrix

| Precision | T4 | A100 | RTX 30xx | RTX 40xx | L4/L40S | H100/H200 | B100/B200/B300 | RTX 50xx | MI300X | MI350X | Gaudi 3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **FP64** | CUDA | TC | CUDA | CUDA | CUDA | TC | TC | CUDA | TC | TC | No |
| **FP32** | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | Vector |
| **TF32** | No | TC | TC | TC | TC | TC | TC | TC | No | No | TC |
| **BF16** | No | TC | TC | TC | TC | TC | TC | TC | TC | TC | TC |
| **FP16** | TC | TC | TC | TC | TC | TC | TC | TC | TC | TC | TC |
| **FP8** | No | No | No | TC | TC | TC | TC | TC | TC | TC | TC |
| **FP6** | No | No | No | No | No | No | TC | No | No | TC | No |
| **FP4** | No | No | No | No | No | No | TC | TC | No | TC | No |
| **INT8** | TC | TC | TC | TC | TC | TC | TC | TC | TC | TC | No |
| **INT4** | TC | TC | TC | No | No | No | No | No | No | No | No |
| **Sparsity (2:4)** | No | Yes | Yes | Yes | Yes | Yes | Yes | Yes | No | No | No |

---

## Sources

### NVIDIA Official
- [NVIDIA A100 Datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf)
- [NVIDIA H100 Product Page](https://www.nvidia.com/en-us/data-center/h100/)
- [NVIDIA H200 Product Page](https://www.nvidia.com/en-us/data-center/h200/)
- [NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)
- [NVIDIA GB300 NVL72](https://www.nvidia.com/en-us/data-center/gb300-nvl72/)
- [NVIDIA RTX PRO 6000 Blackwell](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/)
- [NVIDIA GeForce RTX 50 Series](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/)
- [NVIDIA GeForce RTX 40 Series](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/)
- [NVIDIA GeForce RTX 30 Series](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/)
- [NVIDIA T4 Datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf)
- [NVIDIA A10 Datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a10/pdf/datasheet-new/nvidia-a10-datasheet.pdf)
- [NVIDIA A30 Datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/data-center/products/a30-gpu/pdf/a30-datasheet.pdf)
- [NVIDIA L4 Product Page](https://www.nvidia.com/en-us/data-center/l4/)
- [NVIDIA L40S Product Page](https://www.nvidia.com/en-us/data-center/l40s/)
- [NVIDIA CUDA Compute Capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)
- [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [NVIDIA Ampere Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)
- [NVIDIA Blackwell Ultra Blog](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era)
- [NVIDIA RTX Blackwell GPU Architecture Whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)

### AMD Official
- [AMD MI300X Product Page](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)
- [AMD MI300X Datasheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf)
- [AMD MI325X Product Page](https://www.amd.com/en/products/accelerators/instinct/mi300/mi325x.html)
- [AMD MI325X Datasheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/instinct-mi325x-datasheet.pdf)
- [AMD MI350 Product Page](https://www.amd.com/en/products/accelerators/instinct/mi350.html)
- [AMD MI350 Blog](https://www.amd.com/en/blogs/2025/amd-instinct-mi350-series-and-beyond-accelerating-the-future-of-ai-and-hpc.html)
- [AMD MI250X Product Page](https://www.amd.com/en/products/accelerators/instinct/mi200/mi250x.html)
- [AMD Accelerator Specifications](https://www.amd.com/en/products/specifications/accelerators.html)

### Intel Official
- [Intel Gaudi 3 White Paper](https://cdrdv2-public.intel.com/817486/gaudi-3-ai-accelerator-white-paper.pdf)
- [Intel Gaudi 2 Product Page](https://habana.ai/products/gaudi2/)
- [Intel Gaudi 3 Product Page](https://habana.ai/products/gaudi3/)
- [Gaudi Architecture Documentation](https://docs.habana.ai/en/latest/Gaudi_Overview/Gaudi_Architecture.html)

### Third-Party References
- [BIZON: NVIDIA Tensor Core GPU Comparison](https://bizon-tech.com/blog/nvidia-b200-b100-h200-h100-a100-comparison)
- [IntuitionLabs: NVIDIA Data Center GPU Specs](https://intuitionlabs.ai/articles/nvidia-data-center-gpu-specs)
- [Exxact: Blackwell vs Hopper Comparison](https://www.exxactcorp.com/blog/hpc/comparing-nvidia-tensor-core-gpus)
- [Verda: B300 vs B200 Comparison](https://verda.com/blog/nvidia-b300-vs-b200-complete-gpu-comparison-to-date)
- [Verda: GB300 NVL72 Architecture](https://verda.com/blog/gb300-nvl72-architecture)
- [BentoML: AMD Data Center GPUs Explained](https://www.bentoml.com/blog/amd-data-center-gpus-mi250x-mi300x-mi350x-and-beyond)
- [ServeTheHome: AMD CDNA 4 Architecture](https://www.servethehome.com/amd-dives-deep-on-cdna-4-architecture-and-mi350-accelerator-at-hot-chips-2025/)
- [Modal: H200 vs H100 vs A100](https://modal.com/blog/h200-vs-h100-vs-a100)
- [Modal: NVIDIA Blackwell Products](https://modal.com/blog/nvidia-blackwell)
