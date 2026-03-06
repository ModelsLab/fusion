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

---

## Practical Profiling Recipes

### Recipe 1: Profile a PyTorch Model End-to-End

Use `torch.profiler` to capture CPU ops, CUDA kernels, and memory usage in one pass.

#### Complete Profiling Script

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity, schedule

# --- Setup ---
device = torch.device("cuda")
model = YourModel().to(device).eval()  # replace with your model
input_ids = torch.randint(0, 32000, (1, 128), device=device)

# --- Warmup (always do this before profiling) ---
for _ in range(3):
    with torch.no_grad():
        _ = model(input_ids)
torch.cuda.synchronize()

# --- Profile with schedule: skip 2, warmup 2, active 3, repeat 1 ---
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=2, warmup=2, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler_logs"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    with_flops=True,
) as prof:
    for step in range(7):  # wait(2) + warmup(2) + active(3)
        with record_function(f"step_{step}"):
            with torch.no_grad():
                _ = model(input_ids)
        prof.step()

# --- Print summary table sorted by CUDA time ---
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# --- Export Chrome trace ---
prof.export_chrome_trace("trace.json")
print("Trace saved to trace.json")
```

#### Export and Open the Chrome Trace

```bash
# The script above writes trace.json. Open it in Chrome:
# 1. Open Chrome browser
# 2. Navigate to: chrome://tracing
# 3. Click "Load" and select trace.json
# 4. Use WASD to navigate, scroll to zoom

# Alternative: use Perfetto (better for large traces):
# 1. Go to https://ui.perfetto.dev
# 2. Click "Open trace file"
# 3. Select trace.json
```

#### How to Read the Output Table

```
---------------------------------  ----------  ----------  ----------  ----------
                             Name    CPU total  CUDA total     # Calls   CUDA Mem
---------------------------------  ----------  ----------  ----------  ----------
                     aten::linear      1.200ms     3.500ms          12       0 b
                      aten::addmm      0.800ms     3.200ms          12       0 b
                     aten::matmul      0.300ms     2.100ms           6       0 b
              aten::layer_norm      0.200ms     0.400ms           6       0 b
                    aten::softmax      0.050ms     0.300ms           6       0 b
---------------------------------  ----------  ----------  ----------  ----------
```

Key columns:
- **CPU total**: Time the CPU spent dispatching this op (should be small)
- **CUDA total**: Time the GPU spent executing this op (the real cost)
- **# Calls**: Number of invocations (high count + low time = fine; high count + high time = fusion candidate)
- **CUDA Mem**: Memory delta (positive = allocation, negative = free)

**What to look for**:
1. The top rows are your optimization targets (sorted by CUDA time)
2. If CPU total >> CUDA total for many ops, you have launch overhead -- use `torch.compile` or CUDA Graphs
3. If one op dominates (e.g., `aten::addmm` is 80% of total), focus there
4. Many small ops with few microseconds each = fusion opportunity

---

### Recipe 2: Nsight Compute - Profile a Single Kernel

#### Step 1: Find the Slowest Kernel

```bash
# Quick summary: list all kernels sorted by time
ncu --set basic --csv python my_script.py 2>&1 | head -50

# Or use Nsight Systems first to identify the kernel name:
nsys profile --stats=true python my_script.py 2>&1 | grep "gpukernels"
```

#### Step 2: Profile That Kernel in Detail

```bash
# Profile a specific kernel by name (regex supported)
# --launch-skip: skip N launches (skip warmup)
# --launch-count: profile only N launches
# --set full: collect ALL metrics (slower but complete)
ncu \
  --kernel-name "regex:ampere_fp16_s1688gemm" \
  --launch-skip 3 \
  --launch-count 1 \
  --set full \
  --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  --section ComputeWorkloadAnalysis \
  --section Occupancy \
  -o kernel_profile \
  python my_script.py

# Flags explained:
#   --kernel-name "regex:..."   Filter by kernel name (use regex)
#   --launch-skip 3             Skip first 3 launches (warmup)
#   --launch-count 1            Profile only 1 launch
#   --set full                  Collect all metric sections
#   --section X                 Collect specific section (faster than full)
#   -o kernel_profile           Output file (creates kernel_profile.ncu-rep)
```

#### Step 3: Read SOL% and Diagnose the Bottleneck

```bash
# Print SOL metrics to terminal (no GUI needed)
ncu --import kernel_profile.ncu-rep \
    --page details \
    --csv | grep -i "sol"
```

**Decision matrix based on SOL% readings:**

```
SOL Memory %    SOL Compute %    Diagnosis              Action
-----------     -------------    ---------              ------
80%             20%              Memory-bound           Quantize (FP16→FP8/INT8), fuse kernels,
                                                        reduce memory traffic
20%             80%              Compute-bound          Use tensor cores, lower precision (FP8),
                                                        enable sparsity (2:4)
20%             20%              Latency-bound          Increase occupancy, add prefetching,
                                                        use CUDA Graphs, check warp stalls
75%             70%              Well-balanced          Already near optimal; try lower precision
                                                        for both axes
```

#### Worked Example

```
Scenario: You profile a Llama-7B decode attention kernel.

Nsight Compute reports:
  SM [%]:         18.5%    (Compute SOL)
  Memory [%]:     82.3%    (Memory SOL)
  Achieved Occupancy: 45%

Diagnosis: Memory-bound (Memory SOL >> Compute SOL)
  - The kernel spends most of its time waiting for data from HBM
  - Compute units are idle 80%+ of the time

Action plan (in order of impact):
  1. Quantize KV cache: FP16 → FP8 cuts memory reads in half
     → Expected speedup: ~1.6-1.8x
  2. Fuse QKV projection + RoPE + attention into a single kernel
     → Eliminates intermediate memory writes
  3. Use FlashAttention (if not already) for IO-aware tiling
  4. If batch size is 1: use FlashDecoding to parallelize across KV length
```

---

### Recipe 3: Nsight Systems - Profile Full Application

#### Step 1: Capture a Trace

```bash
# Profile a Python inference script with CUDA + NVTX + OS runtime tracing
nsys profile \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --cuda-memory-usage=true \
  --output=app_profile \
  --force-overwrite=true \
  --duration=30 \
  python my_inference.py

# Flags explained:
#   --trace=cuda,nvtx,osrt,cudnn,cublas   What to trace
#   --cuda-memory-usage=true               Track GPU memory allocations
#   --output=app_profile                   Output file (app_profile.nsys-rep)
#   --force-overwrite=true                 Overwrite existing file
#   --duration=30                          Stop after 30 seconds

# For multi-GPU:
nsys profile \
  --trace=cuda,nvtx,nccl,osrt \
  --output=multi_gpu_profile \
  torchrun --nproc_per_node=4 my_script.py
```

#### Step 2: Get Summary Statistics (No GUI)

```bash
# Print kernel execution time summary
nsys stats --report cuda_gpu_kern_sum app_profile.nsys-rep

# Print CUDA API call summary (shows sync stalls)
nsys stats --report cuda_api_sum app_profile.nsys-rep

# Print memory operation summary
nsys stats --report cuda_gpu_mem_size_sum app_profile.nsys-rep

# Export everything to SQLite for custom queries
nsys export --type=sqlite --output=app_profile.sqlite app_profile.nsys-rep
```

#### Step 3: Identify CPU-GPU Sync Stalls

Look for these in the `cuda_api_sum` report:

```
Time (%)  Total Time (ns)  Num Calls  Avg (ns)    Name
--------  ---------------  ---------  ----------  -------------------------
  45.2%    1,230,000,000        150   8,200,000  cudaDeviceSynchronize   <<<< STALL
  30.1%      820,000,000      5,000     164,000  cudaLaunchKernel
  12.3%      335,000,000         50   6,700,000  cudaMemcpy              <<<< STALL
   8.4%      228,000,000        200   1,140,000  cudaStreamSynchronize
```

**Red flags:**
- `cudaDeviceSynchronize` taking >1ms per call or >10% total time
- `cudaMemcpy` (synchronous) instead of `cudaMemcpyAsync`
- High total time in any synchronization API

**Fixes:**
```python
# BAD: synchronous copy forces CPU to wait
tensor_cpu = tensor_gpu.cpu()  # implicit cudaDeviceSynchronize

# GOOD: async copy with pinned memory
tensor_cpu = torch.empty_like(tensor_gpu, device="cpu").pin_memory()
tensor_cpu.copy_(tensor_gpu, non_blocking=True)

# BAD: print/logging forces sync
output = model(x)
print(output.shape)  # forces sync to materialize tensor

# GOOD: defer sync
output = model(x)
# ... queue more work ...
torch.cuda.synchronize()  # sync only when needed
print(output.shape)
```

#### Step 4: Spot Kernel Launch Overhead

In the Nsight Systems timeline (GUI or stats output):

```
Symptom: Thousands of tiny kernels with gaps between them.

nsys stats report shows:
  Kernel count: 12,000 kernels in 1 second
  Average kernel duration: 5 us
  Average gap between kernels: 8 us  <<<< Launch overhead > kernel time!

Fix:
  1. torch.compile(model, mode="reduce-overhead")  # uses CUDA Graphs
  2. Manual CUDA Graph capture:
```

```python
# CUDA Graph capture to eliminate launch overhead
# Step 1: Warmup
static_input = torch.randn(1, 128, 4096, device="cuda")
for _ in range(3):
    static_output = model(static_input)

# Step 2: Capture
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    static_output = model(static_input)

# Step 3: Replay (near-zero launch overhead)
for new_input in input_stream:
    static_input.copy_(new_input)  # copy into captured buffer
    graph.replay()                 # replay all kernels at once
    result = static_output.clone() # copy out
```

---

### Recipe 4: Measure Memory-Bound vs Compute-Bound

#### Calculate Arithmetic Intensity for Any Layer

```python
def arithmetic_intensity(flops: int, bytes_accessed: int) -> float:
    """
    Arithmetic intensity = FLOPs / Bytes accessed.

    If AI < ridge_point → memory-bound
    If AI > ridge_point → compute-bound

    Ridge points (FP16 tensor cores):
      H100 SXM: 296 FLOP/byte
      A100 SXM: 153 FLOP/byte
      RTX 4090: 327 FLOP/byte
    """
    return flops / bytes_accessed


def gemm_ai(M: int, N: int, K: int, dtype_bytes: int = 2) -> dict:
    """Calculate arithmetic intensity for a GEMM: C[M,N] = A[M,K] x B[K,N]"""
    flops = 2 * M * N * K  # multiply-add = 2 ops per element
    bytes_a = M * K * dtype_bytes
    bytes_b = K * N * dtype_bytes
    bytes_c = M * N * dtype_bytes
    total_bytes = bytes_a + bytes_b + bytes_c
    ai = flops / total_bytes
    return {
        "flops": flops,
        "bytes": total_bytes,
        "arithmetic_intensity": ai,
        "bound": "compute" if ai > 200 else "memory" if ai < 50 else "borderline",
    }


def attention_ai(batch: int, heads: int, seq_len: int, head_dim: int,
                 kv_seq_len: int = None, dtype_bytes: int = 2) -> dict:
    """Calculate arithmetic intensity for attention (Q @ K^T then @ V)"""
    if kv_seq_len is None:
        kv_seq_len = seq_len
    # Q @ K^T: [B*H, S, D] x [B*H, D, KV_S] → [B*H, S, KV_S]
    flops_qk = 2 * batch * heads * seq_len * head_dim * kv_seq_len
    # Attn @ V: [B*H, S, KV_S] x [B*H, KV_S, D] → [B*H, S, D]
    flops_av = 2 * batch * heads * seq_len * kv_seq_len * head_dim
    total_flops = flops_qk + flops_av

    bytes_q = batch * heads * seq_len * head_dim * dtype_bytes
    bytes_k = batch * heads * kv_seq_len * head_dim * dtype_bytes
    bytes_v = batch * heads * kv_seq_len * head_dim * dtype_bytes
    bytes_o = batch * heads * seq_len * head_dim * dtype_bytes
    total_bytes = bytes_q + bytes_k + bytes_v + bytes_o

    ai = total_flops / total_bytes
    return {
        "flops": total_flops,
        "bytes": total_bytes,
        "arithmetic_intensity": ai,
    }


# --- Examples ---
# Decode GEMM: batch=1, projecting hidden_dim
print(gemm_ai(M=1, N=4096, K=4096))
# → {'flops': 33554432, 'bytes': 67117056, 'arithmetic_intensity': 0.5, 'bound': 'memory'}

# Prefill GEMM: batch=512
print(gemm_ai(M=512, N=4096, K=4096))
# → {'arithmetic_intensity': 174.8, 'bound': 'borderline'}

# Large GEMM: batch=2048
print(gemm_ai(M=2048, N=4096, K=4096))
# → {'arithmetic_intensity': 571.7, 'bound': 'compute'}

# Decode attention: batch=1, 32 heads, seq_len=1, kv_len=2048, head_dim=128
print(attention_ai(batch=1, heads=32, seq_len=1, head_dim=128, kv_seq_len=2048))
# → AI ≈ 1.0 → heavily memory-bound
```

#### Plot a Roofline Diagram (Copy-Paste Ready)

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_roofline(
    peak_compute_tflops: float,
    peak_bandwidth_gbps: float,
    gpu_name: str = "GPU",
    points: dict = None,
):
    """
    Plot a roofline diagram.

    Args:
        peak_compute_tflops: Peak compute in TFLOPS (e.g., 990 for H100 FP16 TC)
        peak_bandwidth_gbps: Peak memory bandwidth in GB/s (e.g., 3350 for H100)
        gpu_name: Label for the GPU
        points: Dict of {"label": (arithmetic_intensity, achieved_tflops)} to plot

    Example:
        plot_roofline(
            peak_compute_tflops=990,
            peak_bandwidth_gbps=3350,
            gpu_name="H100 SXM",
            points={
                "Decode Attn":   (1.0, 2.8),
                "Prefill Attn":  (512, 750),
                "Decode GEMM":   (0.5, 1.5),
                "Prefill GEMM":  (175, 450),
            },
        )
    """
    peak_compute = peak_compute_tflops  # TFLOPS
    peak_bw = peak_bandwidth_gbps / 1000  # TB/s (to match TFLOPS / (FLOP/byte))

    ridge_point = peak_compute / peak_bw  # FLOP/byte

    fig, ax = plt.subplots(figsize=(10, 6))

    # X-axis: arithmetic intensity (FLOP/byte), log scale
    ai = np.logspace(-2, 4, 1000)

    # Roofline: min(peak_compute, bandwidth * AI)
    roofline = np.minimum(peak_compute, peak_bw * ai)

    ax.loglog(ai, roofline, "b-", linewidth=2, label=f"{gpu_name} Roofline")

    # Ridge point marker
    ax.axvline(x=ridge_point, color="gray", linestyle="--", alpha=0.5)
    ax.annotate(
        f"Ridge: {ridge_point:.0f} FLOP/B",
        xy=(ridge_point, peak_compute),
        xytext=(ridge_point * 2, peak_compute * 0.6),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=9,
        color="gray",
    )

    # Label regions
    ax.text(0.05, peak_compute * 0.3, "Memory\nBound", fontsize=12, alpha=0.4)
    ax.text(ridge_point * 5, peak_compute * 0.3, "Compute\nBound", fontsize=12, alpha=0.4)

    # Plot measured points
    if points:
        colors = plt.cm.Set1(np.linspace(0, 1, len(points)))
        for (label, (pt_ai, pt_tflops)), color in zip(points.items(), colors):
            ax.plot(pt_ai, pt_tflops, "o", markersize=10, color=color, label=label)
            # Draw vertical line to roofline to show gap
            roofline_at_ai = min(peak_compute, peak_bw * pt_ai)
            ax.plot(
                [pt_ai, pt_ai],
                [pt_tflops, roofline_at_ai],
                "--",
                color=color,
                alpha=0.4,
            )
            efficiency = pt_tflops / roofline_at_ai * 100
            ax.annotate(
                f"{efficiency:.0f}% eff",
                xy=(pt_ai, pt_tflops),
                xytext=(pt_ai * 1.5, pt_tflops * 0.7),
                fontsize=8,
            )

    ax.set_xlabel("Arithmetic Intensity (FLOP/Byte)", fontsize=12)
    ax.set_ylabel("Performance (TFLOPS)", fontsize=12)
    ax.set_title(f"Roofline Model - {gpu_name}", fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(0.01, 10000)
    ax.set_ylim(0.1, peak_compute * 2)
    plt.tight_layout()
    plt.savefig("roofline.png", dpi=150)
    plt.show()
    print("Saved to roofline.png")


# --- Example usage ---
plot_roofline(
    peak_compute_tflops=990,
    peak_bandwidth_gbps=3350,
    gpu_name="H100 SXM (FP16 TC)",
    points={
        "Decode Attn (bs=1, seq=2048)":  (1.0, 3.0),
        "Prefill Attn (bs=1, seq=2048)": (512, 720),
        "Decode GEMM (bs=1)":            (0.5, 1.6),
        "Prefill GEMM (bs=512)":         (175, 520),
        "RMSNorm":                       (0.83, 2.5),
    },
)
```

#### Decision Table: What to Do Based on Arithmetic Intensity

| Arithmetic Intensity | Classification | Root Cause | Optimization Strategy |
|---------------------|---------------|------------|----------------------|
| AI < 1 | Heavily memory-bound | Elementwise ops, small batch decode | Fuse kernels, quantize to FP8/INT4, vectorized loads |
| 1 < AI < 10 | Memory-bound | Small batch GEMMs, attention decode | Increase batch size, quantize KV cache, use FlashDecoding |
| 10 < AI < ridge | Transitional | Medium batch GEMMs | Increase batch until compute-bound, try INT8/FP8 |
| AI > ridge | Compute-bound | Large GEMMs, prefill attention | Use tensor cores, lower precision (FP8), enable 2:4 sparsity |

---

### Recipe 5: Benchmark Inference Speed

#### Complete Benchmarking Script

```python
"""
Inference speed benchmark: measures tokens/sec with statistical analysis.
Copy-paste ready. Replace model loading with your model.
"""

import torch
import time
import numpy as np
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    name: str
    latencies_ms: list
    num_tokens: int

    @property
    def mean_ms(self): return np.mean(self.latencies_ms)
    @property
    def std_ms(self): return np.std(self.latencies_ms)
    @property
    def median_ms(self): return np.median(self.latencies_ms)
    @property
    def p99_ms(self): return np.percentile(self.latencies_ms, 99)
    @property
    def p50_ms(self): return np.percentile(self.latencies_ms, 50)
    @property
    def min_ms(self): return np.min(self.latencies_ms)
    @property
    def max_ms(self): return np.max(self.latencies_ms)
    @property
    def tokens_per_sec(self): return self.num_tokens / (self.mean_ms / 1000)

    def report(self):
        print(f"\n{'='*60}")
        print(f"  Benchmark: {self.name}")
        print(f"{'='*60}")
        print(f"  Iterations:     {len(self.latencies_ms)}")
        print(f"  Tokens/step:    {self.num_tokens}")
        print(f"  Mean latency:   {self.mean_ms:.2f} ms")
        print(f"  Std latency:    {self.std_ms:.2f} ms")
        print(f"  Median (p50):   {self.p50_ms:.2f} ms")
        print(f"  p99 latency:    {self.p99_ms:.2f} ms")
        print(f"  Min latency:    {self.min_ms:.2f} ms")
        print(f"  Max latency:    {self.max_ms:.2f} ms")
        print(f"  Tokens/sec:     {self.tokens_per_sec:.1f}")
        print(f"{'='*60}")


def benchmark_decode(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 128,
    warmup_steps: int = 5,
    bench_steps: int = 50,
    name: str = "model",
) -> BenchmarkResult:
    """
    Benchmark autoregressive decode speed.

    Args:
        model: HuggingFace-compatible model with .generate()
        input_ids: Prompt token IDs [1, seq_len] on CUDA
        max_new_tokens: Number of tokens to generate per step
        warmup_steps: Number of warmup iterations (not measured)
        bench_steps: Number of measured iterations
        name: Label for this benchmark
    """
    device = input_ids.device
    latencies = []

    # Warmup
    print(f"Warming up ({warmup_steps} steps)...")
    for _ in range(warmup_steps):
        with torch.no_grad():
            _ = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
    torch.cuda.synchronize(device)

    # Benchmark
    print(f"Benchmarking ({bench_steps} steps)...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for i in range(bench_steps):
        torch.cuda.synchronize(device)
        start_event.record()
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        end_event.record()
        torch.cuda.synchronize(device)
        latency_ms = start_event.elapsed_time(end_event)
        latencies.append(latency_ms)

        if (i + 1) % 10 == 0:
            tps = max_new_tokens / (latency_ms / 1000)
            print(f"  Step {i+1}/{bench_steps}: {latency_ms:.1f} ms ({tps:.1f} tok/s)")

    return BenchmarkResult(
        name=name,
        latencies_ms=latencies,
        num_tokens=max_new_tokens,
    )


def compare_models(results: list[BenchmarkResult]):
    """Print side-by-side comparison of benchmark results."""
    print(f"\n{'='*80}")
    print(f"  COMPARISON")
    print(f"{'='*80}")
    header = f"{'Metric':<20}"
    for r in results:
        header += f"  {r.name:>18}"
    print(header)
    print("-" * 80)

    metrics = [
        ("Mean (ms)", lambda r: f"{r.mean_ms:.2f}"),
        ("Std (ms)", lambda r: f"{r.std_ms:.2f}"),
        ("p50 (ms)", lambda r: f"{r.p50_ms:.2f}"),
        ("p99 (ms)", lambda r: f"{r.p99_ms:.2f}"),
        ("Min (ms)", lambda r: f"{r.min_ms:.2f}"),
        ("Max (ms)", lambda r: f"{r.max_ms:.2f}"),
        ("Tokens/sec", lambda r: f"{r.tokens_per_sec:.1f}"),
    ]
    for label, fn in metrics:
        row = f"{label:<20}"
        for r in results:
            row += f"  {fn(r):>18}"
        print(row)

    # Speedup relative to first model
    if len(results) >= 2:
        base = results[0]
        print("-" * 80)
        for r in results[1:]:
            speedup = base.mean_ms / r.mean_ms
            print(f"  {r.name} is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than {base.name}")
    print(f"{'='*80}")


# --- Example usage ---
if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda")
    prompt = "The future of AI is"

    # Model A: baseline
    print("Loading Model A...")
    tokenizer_a = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    model_a = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    input_ids_a = tokenizer_a(prompt, return_tensors="pt").input_ids.to(device)
    result_a = benchmark_decode(model_a, input_ids_a, max_new_tokens=128,
                                 warmup_steps=3, bench_steps=20, name="Llama2-7B-FP16")
    result_a.report()

    # Model B: optimized (e.g., quantized)
    print("\nLoading Model B...")
    model_b = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_8bit=True,  # INT8 quantization
    )
    input_ids_b = tokenizer_a(prompt, return_tensors="pt").input_ids.to(device)
    result_b = benchmark_decode(model_b, input_ids_b, max_new_tokens=128,
                                 warmup_steps=3, bench_steps=20, name="Llama2-7B-INT8")
    result_b.report()

    # Compare
    compare_models([result_a, result_b])
```

---

## Common Profiling Mistakes and Fixes

| # | Mistake | What Happens | Fix |
|---|---------|-------------|-----|
| 1 | **No warmup before timing** | First iteration includes CUDA context init, JIT compilation, memory allocation -- reports 10-100x slower than real speed | Always run 3-5 warmup iterations before measuring |
| 2 | **Using `time.time()` instead of CUDA events** | Measures CPU dispatch time, not GPU execution time; reports faster than reality for async ops | Use `torch.cuda.Event(enable_timing=True)` with `.record()` and `.elapsed_time()` |
| 3 | **Missing `torch.cuda.synchronize()`** | GPU work is async; timing ends before GPU finishes; reports impossibly fast times | Call `torch.cuda.synchronize()` before starting and after ending the timer |
| 4 | **Profiling with gradient computation enabled** | Model stores activations for backward pass, uses 2-3x more memory, different kernels execute | Wrap in `torch.no_grad()` or `torch.inference_mode()` for inference benchmarks |
| 5 | **Thermal throttling during long benchmarks** | GPU heats up, clocks drop, later iterations are slower; mean is lower than true steady-state | Monitor `nvidia-smi -l 1` during benchmark; check if p99 >> p50 (sign of throttling); allow cooldown between runs |
| 6 | **Benchmarking with `print()` or logging in the loop** | `print(tensor)` forces `cudaDeviceSynchronize()`; serializes GPU pipeline; adds seconds of overhead | Remove all print/log statements from the timed loop; log results after timing completes |
| 7 | **Not controlling input shapes** | Different sequence lengths or batch sizes hit different code paths; results are not comparable | Fix input shape explicitly; do not use variable-length inputs unless measuring that specifically |
| 8 | **Running other GPU workloads during benchmark** | Shared GPU memory and compute; noisy, unreproducible results | Use `nvidia-smi` to verify no other processes; use `CUDA_VISIBLE_DEVICES=X` to isolate GPU |
| 9 | **Profiling with Nsight on too many kernels** | `ncu --set full` on entire application takes hours; generates GBs of data | Use `--kernel-name` to target specific kernels; use `--launch-skip` / `--launch-count` to limit scope |
| 10 | **Comparing models with different `torch.compile` states** | One model is compiled (fused kernels, CUDA Graphs) and the other is not; unfair comparison | Either compile both or neither; state it explicitly in results |
