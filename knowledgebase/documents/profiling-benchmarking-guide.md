---
id: profiling_benchmarking_guide
kind: document
title: GPU Profiling and Benchmarking - Complete Guide
category: profiling
summary: Comprehensive guide to GPU profiling with Nsight Compute/Systems, roofline analysis, benchmarking methodology, and performance anti-patterns with concrete diagnosis strategies.
tags:
  - profiling
  - nsight-compute
  - nsight-systems
  - roofline
  - benchmarking
  - performance
source_ids:
  - nsight-python
gpu_families:
  - Ampere
  - Ada
  - Hopper
  - Blackwell
---

# GPU Profiling and Benchmarking Guide

## Nsight Compute (Kernel-Level Profiling)

### Essential Metrics

#### Speed of Light (SOL%)
The most important metric: what fraction of theoretical peak your kernel achieves.

```
SM SOL% = actual_compute_throughput / peak_compute_throughput * 100
Memory SOL% = actual_memory_throughput / peak_memory_throughput * 100
```

| SOL% | Assessment |
|------|-----------|
| >80% | Excellent - near optimal |
| 60-80% | Good - room for improvement |
| 40-60% | Moderate - significant optimization possible |
| <40% | Poor - likely a fundamental issue |

#### Determining Bound Type
```
If Memory SOL% >> Compute SOL%: → Memory bound
If Compute SOL% >> Memory SOL%: → Compute bound
If both are low: → Latency bound (stalls, low occupancy, etc.)
If both are high: → Well balanced (best case)
```

### Warp Stall Reasons

| Stall Reason | Meaning | Fix |
|-------------|---------|-----|
| stall_long_scoreboard | Waiting for global memory load | More pipeline stages, prefetching |
| stall_short_scoreboard | Waiting for shared memory / L1 | Fix bank conflicts, better layout |
| stall_barrier | Waiting at __syncthreads() | Reduce sync frequency, smaller tiles |
| stall_not_selected | Eligible but not scheduled | More warps (higher occupancy) |
| stall_math_pipe | Math pipeline full | Already compute-saturated (good) |
| stall_mio_throttle | Memory I/O throttle | Reduce concurrent memory ops |
| stall_tex_throttle | Texture unit throttle | Reduce texture fetches |
| stall_imc_miss | Instruction cache miss | Simplify control flow |
| stall_wait | Waiting for warp group (Hopper) | Check wgmma pipeline |

### Memory Workload Analysis

**L1/Shared Memory**:
```
Metric: l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum (sectors loaded from L1)
Metric: l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum (shared memory loads)

Bank conflicts: l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum
  - 0 = no conflicts (ideal)
  - >0 = conflicts exist, redesign shared memory layout
```

**L2 Cache**:
```
Hit rate: lts__t_sector_hit_rate.pct
  - High hit rate (>80%): good data reuse
  - Low hit rate: working set doesn't fit in L2

Throughput: lts__t_bytes.sum / kernel_time
```

**DRAM (HBM)**:
```
Read throughput: dram__bytes_read.sum / kernel_time
Write throughput: dram__bytes_write.sum / kernel_time
Total: should approach theoretical peak for memory-bound kernels

H100 SXM: 3,350 GB/s theoretical, ~2,900 GB/s achievable
A100 SXM: 2,039 GB/s theoretical, ~1,800 GB/s achievable
RTX 4090: 1,008 GB/s theoretical, ~900 GB/s achievable
```

### Compute Workload Analysis

**Pipe Utilization**:
```
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active  → Tensor core usage
sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active     → FP32 FMA pipe
sm__pipe_fp16_cycles_active.avg.pct_of_peak_sustained_active    → FP16 pipe (non-TC)
sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_active  → Shared memory pipe
```

If tensor core utilization is low for a GEMM kernel → not using tensor cores properly.

### Launch Statistics

```
Registers per thread: launch__registers_per_thread
  - More registers = less occupancy
  - Max 255 per thread, but >128 often hurts occupancy
  - Use __launch_bounds__ to control

Shared memory per block: launch__shared_mem_per_block_driver
  - More shared memory = fewer blocks per SM

Theoretical occupancy: sm__warps_active.avg.pct_of_peak_sustained_active
Achieved occupancy: sm__warps_active.avg.pct_of_peak_sustained_elapsed
```

### Common Nsight Compute Commands

```bash
# Full profiling
ncu --set full -o profile_output ./my_application

# Specific metrics
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed \
./my_application

# Profile specific kernel
ncu --kernel-name "my_kernel" --launch-count 5 ./my_app

# Compare two profiles
ncu --import profile_baseline.ncu-rep --import profile_optimized.ncu-rep

# Python integration
ncu --mode=launch-and-attach python my_script.py
```

## Nsight Systems (Timeline Profiling)

### What It Shows
- CPU thread activity
- CUDA API calls (launch, sync, memcpy)
- Kernel execution timeline on GPU
- Memory transfers (H2D, D2H)
- NVTX annotation regions
- NCCL communications
- Multi-GPU activity

### Common Bottleneck Patterns

**Pattern: Gaps Between Kernels**
```
Timeline: [kernel1]---gap---[kernel2]---gap---[kernel3]
Problem: CPU overhead between kernel launches
Fix: CUDA graphs, torch.compile with reduce-overhead mode
```

**Pattern: Serialized Kernels on Default Stream**
```
Timeline: [kernel1][memcpy][kernel2][memcpy]
Problem: All ops on default stream → serialized
Fix: Use multiple streams, pipeline compute and memory
```

**Pattern: Large Sync Stalls**
```
Timeline: [kernel1]...[cudaDeviceSynchronize = 50ms]...[kernel2]
Problem: Unnecessary synchronization
Fix: Remove sync, use events, async operations
```

**Pattern: Long Prefill Blocking Decode**
```
Timeline: [prefill 200ms][decode][decode]...
Problem: Prefill starves decode requests
Fix: Chunked prefill
```

### Nsight Systems Commands

```bash
# Basic profiling
nsys profile --trace=cuda,nvtx,osrt --output=report ./my_app

# With NCCL tracing (multi-GPU)
nsys profile --trace=cuda,nvtx,nccl --output=report ./my_app

# Python profiling
nsys profile --trace=cuda,nvtx python my_script.py

# Specific time range
nsys profile --duration=10 --delay=5 ./my_app

# Export to JSON for custom analysis
nsys export --type=json report.nsys-rep
```

### NVTX Annotations

```python
import torch
import nvtx

# Mark regions in code
with nvtx.annotate("prefill", color="green"):
    output = model.prefill(input_ids)

with nvtx.annotate("decode", color="blue"):
    for _ in range(max_tokens):
        with nvtx.annotate("decode_step"):
            token = model.decode_step()

# In PyTorch:
with torch.cuda.nvtx.range("attention"):
    attn_output = self.attention(hidden_states)
```

## Roofline Analysis

### Building a Roofline

```
                    |
Peak Compute -------|.........................__________
   (TFLOPS)        |                       /
                    |                     /
                    |                   /  ← slope = bandwidth
                    |                 /
                    |               /
                    |             /
                    |           /
                    |         /
                    |       /
                    |     /
                    |   /
                    | /
                    |________________________
                    Arithmetic Intensity (FLOP/Byte)
                         Ridge Point ↑
```

### GPU Roofline Parameters

```
H100 SXM:
  Peak FP16 TC: 990 TFLOPS
  Peak FP8 TC: 1,979 TFLOPS
  Peak BW: 3,350 GB/s
  Ridge FP16: 990T / 3.35T = 296 FLOP/byte
  Ridge FP8: 1979T / 3.35T = 591 FLOP/byte

A100 SXM 80GB:
  Peak FP16 TC: 312 TFLOPS
  Peak BW: 2,039 GB/s
  Ridge FP16: 312T / 2.039T = 153 FLOP/byte

RTX 4090:
  Peak FP16 TC: 330 TFLOPS
  Peak BW: 1,008 GB/s
  Ridge FP16: 330T / 1.008T = 327 FLOP/byte
```

### Arithmetic Intensity for Common Operations

| Operation | Formula | AI (FP16) | Bound |
|-----------|---------|-----------|-------|
| Vector add | 1 FLOP / (3 * 2 bytes) | 0.17 | Memory |
| RMSNorm | ~5 FLOP / (3 * 2 bytes) | 0.83 | Memory |
| Softmax | ~5 FLOP / (2 * 2 bytes) | 1.25 | Memory |
| GEMM (4096x4096x4096) | 2*4096 / (3*2) | 1365 | Compute |
| GEMM (1x4096x4096) | 2*4096 / ((1+4096+1)*2) | 1.0 | Memory |
| GEMM (32x4096x4096) | 2*32*4096 / ((32+4096+32)*4096*2) | 7.7 | Memory |
| GEMM (256x4096x4096) | 2*256 / ((256/4096+1+256/4096)*2) | 52 | Memory-ish |
| GEMM (1024x4096x4096) | 2*1024 / ((1024/4096+1+1024/4096)*2) | 174 | Compute-ish |
| Attention prefill S=2048 | ~2*2048 / (4*2) | 512 | Compute |
| Attention decode S=1 | ~2 / (2*2) | 0.5 | Memory |

### Practical Optimization Based on Roofline

**Memory-bound kernel** (AI < ridge point):
1. Reduce memory accesses: kernel fusion, avoid intermediates
2. Use lower precision: FP8 → half the bytes, same FLOPs
3. Compress data: quantize weights, KV cache
4. Improve memory access pattern: coalescing, vectorized loads
5. Use L2 cache residency hints

**Compute-bound kernel** (AI > ridge point):
1. Use tensor cores (if not already)
2. Use lower-precision tensor cores: FP8, INT8
3. Increase tile sizes for better MMA utilization
4. Reduce non-MMA compute (softmax rescaling, etc.)
5. Use sparsity (2:4) for 2x throughput

**Latency-bound kernel** (both SOL% low):
1. Increase occupancy (fewer registers, less shared memory)
2. Add pipeline stages to hide memory latency
3. Use CUDA graphs to reduce launch overhead
4. Reduce thread divergence
5. Check for unnecessary synchronization

## Benchmarking Methodology

### Proper Timing

```python
import torch
import time

# WRONG - doesn't account for GPU async execution:
start = time.time()
output = model(input)
elapsed = time.time() - start  # Measures CPU time, not GPU time!

# CORRECT - CUDA event timing:
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# Warmup (CRITICAL - first runs are always slower)
for _ in range(5):
    _ = model(input)

torch.cuda.synchronize()
start_event.record()
for _ in range(num_iterations):
    _ = model(input)
end_event.record()
torch.cuda.synchronize()

elapsed_ms = start_event.elapsed_time(end_event) / num_iterations
```

### Triton Benchmarking Utility
```python
import triton

ms = triton.testing.do_bench(
    lambda: my_kernel[grid](args),
    warmup=25,        # warmup iterations
    rep=100,          # measurement iterations
    quantiles=[0.5, 0.2, 0.8],  # median, p20, p80
)

# Returns: [median_ms, p20_ms, p80_ms]
```

### Statistical Analysis
```python
# Collect multiple measurements
times = []
for _ in range(100):
    t = benchmark_one_iteration()
    times.append(t)

import numpy as np
print(f"Mean: {np.mean(times):.3f} ms")
print(f"Median: {np.median(times):.3f} ms")
print(f"Std: {np.std(times):.3f} ms")
print(f"P99: {np.percentile(times, 99):.3f} ms")
print(f"Min: {np.min(times):.3f} ms")
print(f"Max: {np.max(times):.3f} ms")

# Check for thermal throttling:
# If later measurements are slower, GPU may be throttling
```

### Bandwidth Calculation
```python
def calculate_bandwidth(bytes_accessed, time_ms):
    """Calculate achieved bandwidth in GB/s"""
    return bytes_accessed / (time_ms * 1e-3) / 1e9

# Example: RMSNorm on (batch=32, hidden=4096) in FP16
bytes_read = 32 * 4096 * 2  # input
bytes_written = 32 * 4096 * 2  # output
bytes_weight = 4096 * 2  # norm weight
total_bytes = bytes_read + bytes_written + bytes_weight

time_ms = 0.015  # measured
bw = calculate_bandwidth(total_bytes, time_ms)
print(f"Achieved bandwidth: {bw:.1f} GB/s")
# Compare to peak: H100 = 3350 GB/s → {bw/3350*100:.1f}% of peak
```

### FLOPS Calculation
```python
def calculate_tflops(flops, time_ms):
    """Calculate achieved TFLOPS"""
    return flops / (time_ms * 1e-3) / 1e12

# Example: GEMM M=2048, N=4096, K=4096
flops = 2 * 2048 * 4096 * 4096  # multiply-add = 2 ops
time_ms = 0.5  # measured
tflops = calculate_tflops(flops, time_ms)
print(f"Achieved: {tflops:.1f} TFLOPS")
# Compare to peak: H100 FP16 TC = 990 TFLOPS → {tflops/990*100:.1f}% of peak
```

## Common Performance Anti-Patterns

### 1. Kernel Launch Overhead
```
Problem: Many small kernels → CPU-side launch overhead dominates
Diagnosis: Nsight Systems shows gaps between kernels
Solution:
  - CUDA graphs (batch multiple kernels into single launch)
  - Kernel fusion (combine small ops into one kernel)
  - torch.compile with reduce-overhead mode
  - Persistent kernels for decode
```

### 2. Host-Device Synchronization
```
Problem: Unnecessary cudaDeviceSynchronize() calls
Diagnosis: Nsight Systems shows CPU idle after sync
Solution:
  - Remove explicit syncs
  - Use CUDA events for timing instead of CPU timing
  - Use async memory copies
  - Use cuda graph capture (inherently async)
```

### 3. Uncoalesced Memory Access
```
Problem: Threads in a warp access non-contiguous memory
Diagnosis: Nsight Compute shows high sectors_per_request (ideal: 1)
Solution:
  - Ensure threads access consecutive addresses
  - Transpose data if needed for better access pattern
  - Use vectorized loads (float4)
  - SoA instead of AoS layout
```

### 4. Shared Memory Bank Conflicts
```
Problem: Multiple threads in warp access same shared memory bank
Diagnosis: Nsight Compute l1tex__data_bank_conflicts_pipe_lsu > 0
Solution:
  - Add padding: smem[N][M+1] instead of smem[N][M]
  - Use swizzle patterns (CUTLASS style)
  - Rearrange access pattern
```

### 5. Register Spilling
```
Problem: Kernel uses too many registers → spills to local memory (slow)
Diagnosis: Nsight Compute shows local memory usage, low occupancy
Solution:
  - Reduce variables in kernel
  - Use __launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)
  - Recompute values instead of storing
  - Reduce BLOCK sizes
```

### 6. Low Occupancy
```
Problem: Too few active warps → can't hide memory latency
Diagnosis: Achieved occupancy << theoretical occupancy
Root causes:
  - Too many registers per thread (limit to 128 for good occupancy)
  - Too much shared memory per block
  - Too many threads per block
Solution:
  - Reduce register usage
  - Smaller shared memory allocation
  - More blocks with fewer threads
```

### 7. Warp Divergence
```
Problem: Threads in same warp take different branches → serialization
Diagnosis: sm__sass_branch_targets_threads_divergent high
Solution:
  - Restructure code to minimize divergence within warps
  - Sort data so adjacent threads take same branch
  - Use predication instead of branching for simple cases
```

## GPU Performance Reference Table

### Theoretical Peak FLOPS (TFLOPS)

| GPU | FP32 | FP16 TC | BF16 TC | TF32 TC | FP8 TC | INT8 TC | FP4 TC |
|-----|------|---------|---------|---------|--------|---------|--------|
| RTX 3090 | 35.6 | 71 | 71 | 71 | - | 142 | - |
| RTX 4090 | 82.6 | 165 | 165 | 165 | 330 | 330 | - |
| RTX 5090 | ~105 | ~210 | ~210 | ~210 | ~420 | ~420 | ~840 |
| A100 SXM | 19.5 | 312 | 312 | 156 | - | 624 | - |
| H100 SXM | 67 | 990 | 990 | 495 | 1979 | 1979 | - |
| H200 | 67 | 990 | 990 | 495 | 1979 | 1979 | - |
| B200 | ~90 | 2250 | 2250 | 1125 | 4500 | 4500 | 9000 |

### Memory Bandwidth (GB/s)

| GPU | Type | Capacity | Bandwidth |
|-----|------|----------|-----------|
| RTX 3090 | GDDR6X | 24 GB | 936 |
| RTX 4090 | GDDR6X | 24 GB | 1008 |
| RTX 5090 | GDDR7 | 32 GB | ~1792 |
| A100 SXM 80GB | HBM2e | 80 GB | 2039 |
| H100 SXM | HBM3 | 80 GB | 3350 |
| H200 | HBM3e | 141 GB | 4800 |
| B200 | HBM3e | 192 GB | 8000 |
| MI300X | HBM3 | 192 GB | 5300 |
