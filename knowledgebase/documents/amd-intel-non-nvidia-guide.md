---
id: amd_intel_non_nvidia_guide
kind: document
title: AMD, Intel, and Non-NVIDIA GPU Programming Guide
category: hardware
summary: Comprehensive guide to programming AMD GPUs (ROCm/HIP), Intel GPUs (SYCL/oneAPI), Google TPUs, and Apple Silicon for AI workloads.
tags:
  - amd
  - rocm
  - hip
  - mi300x
  - intel
  - gaudi
  - tpu
  - apple-silicon
  - cross-platform
gpu_families:
  - CDNA
  - RDNA
  - Xe
  - TPU
---

# AMD, Intel, and Non-NVIDIA GPU Programming

## AMD GPU Architecture (CDNA 3 / MI300X)

### Architecture Overview
```
MI300X:
- 8 XCDs (Accelerated Compute Dies) on chiplet design
- 304 Compute Units (CUs) total
- Wavefront size: 64 threads (vs NVIDIA warp of 32)
- Matrix cores (equivalent to tensor cores)
- 4 HBM3 stacks, 192 GB total
- 5.3 TB/s memory bandwidth
- 256 MB Infinity Cache (L3)
- 750W TDP
```

### CDNA 3 vs NVIDIA Hopper Comparison

| Feature | MI300X (CDNA 3) | H100 (Hopper) |
|---------|-----------------|---------------|
| Memory | 192 GB HBM3 | 80 GB HBM3 |
| Bandwidth | 5.3 TB/s | 3.35 TB/s |
| FP16 TFLOPS | 1307 | 990 |
| FP8 TFLOPS | 2615 | 1979 |
| L2/L3 Cache | 256 MB | 50 MB |
| Process | 5nm/6nm | 4nm |
| Interconnect | Infinity Fabric | NVLink 4.0 |
| TDP | 750W | 700W |

MI300X has 2.4x more memory and 1.6x more bandwidth than H100.
H100 has stronger per-SM efficiency and better software ecosystem.

### ROCm Software Stack
```
Application Layer: PyTorch, TensorFlow, JAX
         ↓
Framework Layer: MIOpen (≈cuDNN), rocBLAS (≈cuBLAS), hipBLAS
         ↓
Runtime Layer: HIP (≈CUDA Runtime API)
         ↓
Driver Layer: ROCm kernel driver, KFD
         ↓
Hardware: CDNA/RDNA GPU
```

### HIP Programming (CUDA-like API)
```cpp
// HIP is nearly identical to CUDA:
#include <hip/hip_runtime.h>

__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

int main() {
    hipMalloc(&d_a, size);
    hipMemcpy(d_a, h_a, size, hipMemcpyHostToDevice);
    vector_add<<<grid, block>>>(d_a, d_b, d_c, n);
    hipDeviceSynchronize();
}
```

### Key Differences from CUDA
```
1. Wavefront size: 64 threads (NVIDIA warp = 32)
   - Implications: shared memory bank patterns differ
   - Some algorithms tuned for warp=32 need adjustment

2. No equivalent to TMA (Tensor Memory Accelerator)
   - Hopper/Blackwell advantage for async bulk copies

3. Composable Kernel (CK) library instead of CUTLASS
   - AMD's equivalent for optimized GEMM kernels
   - Different API but similar concepts

4. Matrix core (MFMA) instructions instead of MMA/WMMA
   - Different shapes and precisions
   - MFMA_F32_32x32x8_F16: 32x32x8 FP16 matrix multiply

5. LDS (Local Data Share) instead of Shared Memory
   - 64 KB per CU (vs configurable on NVIDIA)
   - Similar bank structure but 64 banks (matching wavefront size)
```

### PyTorch on AMD
```python
# ROCm builds of PyTorch use the same API:
import torch
device = torch.device('cuda')  # Works with ROCm! HIP maps to CUDA API

# Check ROCm:
print(torch.version.hip)  # ROCm version

# Known gaps vs NVIDIA:
# - FlashAttention: needs AMD-specific fork (flash-attn for ROCm)
# - Some custom CUDA kernels need hipification
# - Triton on AMD: works but less optimized than NVIDIA
# - torch.compile: works but Inductor generates less optimal code
```

### vLLM on AMD
```bash
# Install vLLM for ROCm:
pip install vllm  # ROCm wheel
# or build from source with ROCm

# Known support:
# - PagedAttention: works
# - FlashAttention for ROCm: supported
# - AWQ/GPTQ: supported via ROCm-compatible kernels
# - Tensor parallelism: works with RCCL (AMD's NCCL)
```

## AMD MI325X and MI350

### MI325X
```
- CDNA 3 architecture (same as MI300X)
- 256 GB HBM3e (vs 192 GB HBM3)
- 6 TB/s bandwidth (vs 5.3 TB/s)
- Same compute as MI300X
- Key advantage: more memory for larger models
```

### MI350X (CDNA 4)
```
- New CDNA 4 architecture (3nm process)
- 288 GB HBM3e
- 8 TB/s bandwidth
- FP4 support (matching Blackwell)
- Up to 20 PFLOPS FP4 (estimated)
- Expected: competing directly with B200
```

## Intel Gaudi 3

### Architecture
```
Intel Gaudi 3:
- 8 Matrix Math Engines (MME) - equivalent to tensor cores
- 64 Tensor Processing Cores (TPC) - programmable VLIW processors
- 128 GB HBM2e, 3.7 TB/s bandwidth
- 96 MB on-chip SRAM
- 24x 200GbE integrated networking
- 900W TDP
- ~1835 BF16 TFLOPS (via MMEs)
```

### Software Stack
```
Application: PyTorch (via Habana plugin)
Framework: Intel Gaudi PyTorch plugin
Runtime: SynapseAI SDK
Hardware: Gaudi 3
```

### Programming Model
```python
# PyTorch with Gaudi:
import habana_frameworks.torch as ht

# Move model to Gaudi:
model = model.to('hpu')  # Habana Processing Unit
input = input.to('hpu')

# Lazy mode (graph compilation):
ht.hpu.lazy_mode()  # similar to torch.compile
output = model(input)
ht.hpu.synchronize()

# Or eager mode:
ht.hpu.eager_mode()
output = model(input)
```

## Google TPU

### Architecture (v5p)
```
TPU v5p:
- 2 MXU (Matrix Multiply Units): 128x128 systolic arrays
- 459 TFLOPS BF16 per chip
- 95 GB HBM2e, 2.76 TB/s per chip
- ICI (Inter-Chip Interconnect): 4.8 TB/s per chip
- 4096 chips per pod
- Good at: large-batch training, serving with high batch sizes
```

### TPU v6e (Trillium)
```
- Next-gen TPU
- ~67% higher peak compute vs v5e
- 256x256 MXU (larger)
- 32 GB HBM per chip
- Better ICI bandwidth
```

### Programming TPUs (JAX)
```python
import jax
import jax.numpy as jnp

# TPU uses XLA compiler:
@jax.jit
def matmul(a, b):
    return jnp.dot(a, b)

# Distributed across TPU pod:
from jax.sharding import Mesh, PartitionSpec as P

devices = jax.devices()
mesh = Mesh(devices, axis_names=('data', 'model'))

# Shard tensors across TPU mesh:
x_sharded = jax.device_put(x, NamedSharding(mesh, P('data', None)))
```

### TPU vs GPU for Kernel Work
- TPU kernels written in Pallas (JAX's GPU kernel language) or XLA custom calls
- Less flexibility than CUDA - designed for regular, high-throughput workloads
- Excellent for: large-batch training, large-batch serving
- Weaker for: small-batch inference, irregular computation (MoE routing)

## Apple Silicon for ML

### Architecture
```
M4 Max:
- 40 GPU cores (unified architecture)
- 128 GB unified memory, 546 GB/s bandwidth
- 16-core Neural Engine (ANE)
- Unified memory: CPU and GPU share the same memory (no copies!)
```

### MLX Framework
```python
import mlx.core as mx
import mlx.nn as nn

# MLX is numpy-like but GPU-accelerated:
x = mx.array([1, 2, 3])
y = mx.matmul(a, b)  # GPU-accelerated

# Lazy evaluation (like JAX):
mx.eval(y)  # forces computation

# Model inference:
model = load_model("llama-7b-4bit")
output = model.generate(prompt, max_tokens=100)
```

### llama.cpp Metal Backend
```
- Best option for LLM inference on Apple Silicon
- Metal shaders for GEMM, attention, norm, etc.
- Leverages unified memory (no CPU↔GPU copies)
- Supports all GGUF quantization formats
- Performance: LLaMA-7B Q4_K at ~30-40 tok/s on M2 Ultra
```

## Cross-Platform Kernel Development

### Triton as Portable GPU Language
```python
# Triton kernels work on:
# - NVIDIA GPUs (primary target, best optimized)
# - AMD GPUs (via HIP backend, less optimized)
# - Intel GPUs (experimental)

# Same kernel code, different backends:
@triton.jit
def my_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < n)
    tl.store(y_ptr + offs, x * 2, mask=offs < n)

# Runs on NVIDIA with CUDA, AMD with HIP
```

### When to Target Non-NVIDIA

| Scenario | Recommendation | Why |
|----------|---------------|-----|
| Maximum inference perf | NVIDIA (H100/B200) | Best software stack |
| Maximum memory (single chip) | AMD MI300X (192 GB) | 2.4x H100 memory |
| Cost-effective training | Google TPU | Good price/perf at scale |
| Edge/mobile inference | Apple Silicon / Qualcomm | Low power, unified memory |
| Budget inference | AMD MI250X (used) | Cheaper than A100 |
| Multi-modal / media | Intel Gaudi 3 | Integrated media processing |
