# GPU Profiling, Benchmarking, and Performance Analysis

## Complete Reference for Kernel Optimization

---

## Table of Contents

1. [NVIDIA Nsight Compute](#1-nvidia-nsight-compute)
2. [NVIDIA Nsight Systems](#2-nvidia-nsight-systems)
3. [Roofline Model Deep Dive](#3-roofline-model-deep-dive)
4. [Benchmarking Methodology](#4-benchmarking-methodology)
5. [Performance Counters and Hardware Metrics](#5-performance-counters-and-hardware-metrics)
6. [Common Performance Anti-Patterns](#6-common-performance-anti-patterns)
7. [NVIDIA DALI](#7-nvidia-dali)
8. [GPU Bandwidth/FLOPS Reference Table](#8-gpu-bandwidthflops-reference-table)

---

## 1. NVIDIA Nsight Compute

Nsight Compute is the definitive kernel-level GPU profiler for CUDA applications (Compute Capability 7.0+). It replaces nvprof and the NVIDIA Visual Profiler. It works by replaying kernels multiple times to collect the full set of hardware performance counters.

### 1.1 Speed of Light (SOL%) Metrics

The **Speed of Light** section is the first place to look when analyzing any kernel. It reports achieved performance as a percentage of the GPU's theoretical maximum for both compute and memory subsystems.

| SOL Metric | What It Measures | Target Values |
|---|---|---|
| **SM Throughput (%)** | Achieved compute utilization vs. peak | >60% is good, >80% is excellent |
| **Memory Throughput (%)** | Achieved memory bandwidth vs. peak | >60% is good, >80% is excellent |
| **Achieved Occupancy** | Ratio of active warps to max warps per SM | Depends on kernel; higher is usually better |
| **SM Frequency** | Actual clock rate during kernel execution | Compare to base/boost clocks |
| **Elapsed Cycles** | Total GPU cycles for kernel execution | Lower is better; use for A/B comparison |
| **Duration** | Wall-clock kernel time | Primary optimization target |

**Rules of Thumb for SOL Analysis:**

- **High SM%, Low Memory%**: Kernel is compute-bound. Optimize arithmetic (reduce instructions, use faster ops, leverage tensor cores).
- **Low SM%, High Memory%**: Kernel is memory-bound. Optimize memory access patterns (coalescing, caching, reduce traffic).
- **Low SM%, Low Memory%**: Kernel is latency-bound. Increase occupancy, reduce stalls, improve instruction-level parallelism.
- **High SM%, High Memory%**: Kernel is well-balanced and near hardware limits. Focus on algorithmic improvements.

**SOL Roofline Chart:** If enabled, this section includes a roofline chart plotting the kernel's achieved FLOP/s against its arithmetic intensity. The kernel appears as a dot; its position relative to the roofline ceilings indicates the primary bottleneck.

### 1.2 Memory Workload Analysis

This section examines data movement efficiency across the entire memory hierarchy.

#### L1/TEX Cache Metrics

| Metric | Description | Optimization Guidance |
|---|---|---|
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | L1 sectors loaded for global loads | Compare to requests for coalescing efficiency |
| `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` | L1 requests for global loads | Ideal ratio: 4 sectors per request (32-bit, coalesced) |
| `l1tex__t_sector_hit_rate.pct` | L1 cache hit rate | Higher is better; >50% indicates good locality |
| `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum` | Shared memory load wavefronts | Compare to requests to detect bank conflicts |

**Coalescing Check:** For 32-bit (4-byte) loads with a full warp (32 threads):
- **Ideal:** 4 sectors per request (128 bytes = 32 threads x 4 bytes, served in 4x32-byte sectors)
- **Worst case:** 32 sectors per request (each thread accesses a different 32-byte sector)
- **Formula:** Coalescing efficiency = (ideal sectors) / (actual sectors) x 100%

#### L2 Cache Metrics

| Metric | Description |
|---|---|
| `lts__t_sector_hit_rate.pct` | L2 cache hit rate |
| `lts__t_sectors.sum` | Total L2 sectors accessed |
| `lts__t_sectors_srcunit_tex_op_read.sum` | L2 sectors read from TEX unit |
| `lts__t_sectors_srcunit_tex_op_write.sum` | L2 sectors written from TEX unit |
| `lts__throughput.avg.pct_of_peak_sustained` | L2 throughput as % of peak |

#### DRAM (HBM/GDDR) Metrics

| Metric | Description |
|---|---|
| `dram__bytes.sum` | Total DRAM bytes transferred |
| `dram__bytes_read.sum` | DRAM bytes read |
| `dram__bytes_write.sum` | DRAM bytes written |
| `dram__throughput.avg.pct_of_peak_sustained` | DRAM throughput as % of peak |

#### Shared Memory Bank Conflicts

Shared memory has 32 banks. Successive 32-bit words map to successive banks. A bank conflict occurs when two or more threads in a warp access different addresses in the same bank.

- **N-way conflict:** N threads access the same bank -> serialized into N rounds
- **Worst case:** 32-way conflict -> 32x throughput reduction
- **Detection metric:** Compare `shared_op_ld wavefronts` to `shared_op_ld requests`. Ratio > 1.0 indicates bank conflicts.
- **Fix:** Pad shared memory arrays (e.g., `__shared__ float smem[32][33]` instead of `[32][32]`)

### 1.3 Compute Workload Analysis

| Metric | Description |
|---|---|
| `sm__inst_executed.avg.per_cycle_active` | Instructions per cycle (IPC), executed |
| `sm__inst_issued.avg.per_cycle_active` | Instructions per cycle (IPC), issued |
| `sm__warps_active.avg.per_cycle_active` | Active warps per cycle |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | Tensor core pipe utilization |
| `sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active` | FP32 (FMA) pipe utilization |
| `sm__pipe_fp16_cycles_active.avg.pct_of_peak_sustained_active` | FP16 pipe utilization |
| `sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active` | Integer (ALU) pipe utilization |
| `sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_active` | Shared memory pipe utilization |

**Pipe Utilization Interpretation:**
- **High tensor pipe, low others:** Kernel effectively uses tensor cores (ideal for GEMM)
- **High FP32 pipe:** Standard compute-bound kernel
- **High memory pipe (LSU):** Memory-bound; most instructions are loads/stores
- **All pipes low:** Stall-bound kernel; check warp state statistics

### 1.4 Warp State Statistics

At each cycle, the warp scheduler samples the state of each resident warp. Stall reasons explain why eligible warps cannot be issued.

| Stall Reason | Description | Common Cause | Fix |
|---|---|---|---|
| **long_scoreboard** | Waiting for result of L1TEX (global/local/tex/surface) memory operation | Global memory latency | Increase occupancy, prefetch, improve locality |
| **stall_barrier** | Waiting at `__syncthreads()` or other barrier | Load imbalance within block | Reduce barrier use, balance work across threads |
| **stall_not_selected** | Warp was eligible but scheduler chose another | High occupancy (actually good) | Not a real stall; indicates healthy scheduling |
| **stall_math_pipe_throttle** | Waiting because math pipe is full | Compute-bound (actually good if intended) | Use tensor cores, reduce instruction count |
| **stall_lg_throttle** | Waiting for free entry in LSU instruction queue | Too many outstanding loads/stores | Combine narrow loads into wider ones, interleave math |
| **stall_short_scoreboard** | Waiting for result of MIO (shared memory, constant, immediate) operation | Shared memory latency | Reduce shared memory bank conflicts |
| **stall_tex_throttle** | Waiting for TEX unit to be available | Texture/surface instruction backlog | Reduce texture requests, use global loads |
| **stall_memory_throttle** | Waiting because memory is backed up | Memory subsystem saturated | Reduce memory traffic, improve access patterns |
| **stall_dispatch** | Waiting for dispatch unit | Instruction scheduling conflict | Compiler/architecture limitation |
| **stall_imc_miss** | Instruction cache miss | Large kernel code | Reduce code size, improve instruction locality |
| **stall_drain** | Waiting for all memory writes to complete before exit | End-of-kernel fence | Normal at kernel end; not actionable |
| **stall_sleeping** | Warp explicitly sleeping | `nanosleep()` or cooperative groups | Intentional; check if necessary |

**Key Insight:** `long_scoreboard` and `lg_throttle` are related but distinct:
- `long_scoreboard`: The warp issued a load and is now at an instruction that *depends on* the load result (waiting for data).
- `lg_throttle`: The warp wants to *issue* a new load/store but the LSU queue is full (too many outstanding requests).
- Fixing `lg_throttle` (by reducing loads) often reveals `long_scoreboard` as the new bottleneck, because execution reaches the dependent instruction faster.

### 1.5 Launch Statistics

| Metric | Description | Impact |
|---|---|---|
| `launch__registers_per_thread` | Registers allocated per thread | More registers -> lower occupancy but fewer spills |
| `launch__shared_mem_per_block_static` | Static shared memory per block | Limits blocks per SM |
| `launch__shared_mem_per_block_dynamic` | Dynamic shared memory per block | Limits blocks per SM |
| `launch__block_dim_x/y/z` | Block dimensions | Must be multiple of 32 for full warps |
| `launch__grid_dim_x/y/z` | Grid dimensions | Should be >> SM count for full utilization |
| `launch__occupancy_theoretical` | Max occupancy given resource limits | Upper bound on active warps |
| `launch__occupancy_achieved` | Measured average active warps / max | Gap from theoretical indicates imbalance |

**Occupancy Limiters (check which one caps your kernel):**

1. **Registers per thread:** Each SM has a fixed register file (e.g., 65536 on Ampere). More registers per thread -> fewer concurrent threads.
   - A100: 65536 registers/SM, max 2048 threads/SM
   - At 128 registers/thread: only 512 threads = 25% occupancy
   - At 32 registers/thread: 2048 threads = 100% occupancy
2. **Shared memory per block:** Each SM has limited shared memory (e.g., 164KB configurable on A100). Large shared memory allocations reduce blocks per SM.
3. **Threads per block:** Max 1024 threads per block. If block size is 1024 and only 1 block fits per SM, occupancy is limited.
4. **Blocks per SM:** Hardware limit (e.g., 32 on Ampere). Many small blocks can hit this limit.

### 1.6 Source-Level Analysis

When profiled with `--set source` or `--set full`, Nsight Compute provides source-correlated metrics:
- **Per-line** instruction counts (executed, predicated-off)
- **Per-line** memory traffic (global loads/stores, shared loads/stores)
- **Hotspot identification:** Lines consuming the most cycles
- **Stall attribution:** Which source lines cause which stalls
- **Unrolling visualization:** See how loops map to instructions

### 1.7 Comparing Kernels (Baseline A/B)

In the Nsight Compute GUI:
1. Profile the baseline kernel -> save report (.ncu-rep)
2. Click **"Add Baseline"** on the baseline result
3. Profile the optimized kernel
4. All metrics automatically show **percentage change** vs. baseline
5. Green = improvement, Red = regression

Baselines can be saved to `.ncu-bln` files and shared across team members.

### 1.8 CLI Usage

```bash
# Full profile with all metrics (slowest, most comprehensive)
ncu --set full -o profile_output ./my_app

# Detailed profile (includes roofline, recommended default)
ncu --set detailed -o profile_output ./my_app

# Default (fastest, high-level overview only)
ncu -o profile_output ./my_app

# Source-level analysis
ncu --set source -o profile_output ./my_app

# Specific sections only
ncu --section SpeedOfLight --section MemoryWorkloadAnalysis -o profile_output ./my_app

# Specific metrics only
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed ./my_app

# Profile specific kernel (by name regex)
ncu --kernel-name "regex:matmul.*" -o profile_output ./my_app

# Profile specific kernel launch (e.g., 5th launch)
ncu --launch-skip 4 --launch-count 1 -o profile_output ./my_app

# CSV output for scripting
ncu --csv --set full ./my_app > metrics.csv

# Query all available metrics
ncu --query-metrics --chip sm_80 > all_metrics.txt

# Compare two runs
ncu --diff baseline.ncu-rep optimized.ncu-rep

# Profile Python/PyTorch application
ncu --set full -o profile_output python my_script.py

# Roofline chart specifically
ncu --section SpeedOfLight_RooflineChart -o profile_output ./my_app
```

### 1.9 Rules of Thumb for Kernel Optimization Based on Metrics

| Observation | Diagnosis | Action |
|---|---|---|
| SOL SM% > 80%, Memory% < 30% | Compute-bound | Use tensor cores, reduce FLOPs, use lower precision |
| SOL SM% < 30%, Memory% > 80% | Memory-bound | Improve coalescing, use shared memory tiling, reduce traffic |
| SOL SM% < 30%, Memory% < 30% | Latency-bound | Increase occupancy, reduce stalls, increase ILP |
| Achieved occupancy << theoretical | Resource limits or imbalance | Check registers, shared memory; use occupancy calculator |
| Transaction/request ratio >> 4 | Uncoalesced access | Fix memory access pattern; ensure stride-1 access |
| High `long_scoreboard` stalls | Waiting on global memory | More arithmetic between loads, prefetch, increase occupancy |
| High `stall_barrier` | Sync overhead | Reduce `__syncthreads()`, balance work within block |
| High `lg_throttle` | LSU queue full | Widen loads (float4), interleave math with loads |
| High `stall_math_pipe_throttle` | Compute pipe saturated | Good if intended; try tensor cores for higher throughput |
| Low tensor pipe utilization for GEMM | Not using tensor cores | Use WMMA/MMA instructions, ensure correct data layout |
| Register spilling (local memory traffic) | Too many registers | Reduce register pressure, accept lower ILP, use `__launch_bounds__` |

---

## 2. NVIDIA Nsight Systems

Nsight Systems is a **system-wide** profiler that provides a timeline view of all CPU and GPU activity. Use it to understand the big picture before diving into kernel-level analysis with Nsight Compute.

### 2.1 Timeline Analysis

The timeline displays multiple rows of information:

| Row | Contents |
|---|---|
| **CUDA HW** | GPU utilization %, kernel execution bars, memory copy bars |
| **Process (e.g., python)** | CPU utilization per core |
| **OS Runtime Libraries** | Thread activities, semaphores, mutexes, sleep calls |
| **CUDA API** | `cudaLaunchKernel`, `cudaMemcpy`, `cudaMalloc`, `cudaDeviceSynchronize` calls |
| **NVTX** | Custom user-defined annotation ranges |
| **Kernels** | Individual kernel launches with grid/block dimensions, registers, occupancy |
| **Memory Operations** | H2D, D2H, D2D transfers with size and throughput |

**What to look for in the timeline:**
- **Gaps between kernels:** Launch overhead, synchronization stalls, or CPU work
- **Thin kernel bars:** Very short kernels (< 10us) -> consider fusing
- **Long memory copy bars overlapping nothing:** Transfers blocking compute
- **Staircase pattern:** Sequential kernel launches (no overlap) -> use streams/graphs
- **CPU bars extending past GPU bars:** CPU is the bottleneck, not GPU

### 2.2 CPU-GPU Overlap Analysis

Ideal GPU utilization means the GPU is never idle. Common problems visible in timeline:

1. **No overlap between compute and memory transfers:** Use `cudaMemcpyAsync` with multiple streams.
2. **CPU bottleneck:** CPU takes longer to prepare next batch than GPU takes to process current one. Visible as GPU idle gaps between kernel bursts.
3. **Synchronization stalls:** `cudaDeviceSynchronize()` or `torch.cuda.synchronize()` forces CPU to wait for GPU. Visible as long CPU bars in CUDA API row.
4. **Default stream serialization:** All operations on the default stream execute sequentially. Use multiple streams for overlap.

### 2.3 NVTX Annotations

NVTX annotations label regions of code that appear in the timeline, enabling code-to-trace mapping.

**Python (PyTorch):**
```python
import nvtx

# Context manager style
with nvtx.annotate("forward_pass", color="blue"):
    output = model(input)

with nvtx.annotate("loss_computation", color="green"):
    loss = criterion(output, target)

with nvtx.annotate("backward_pass", color="red"):
    loss.backward()

# Decorator style
@nvtx.annotate("data_preprocessing")
def preprocess(data):
    return transform(data)

# PyTorch built-in NVTX
torch.cuda.nvtx.range_push("my_region")
# ... work ...
torch.cuda.nvtx.range_pop()
```

**C/C++:**
```cpp
#include <nvtx3/nvToolsExt.h>

nvtxRangePushA("kernel_execution");
my_kernel<<<grid, block>>>(args);
nvtxRangePop();
```

### 2.4 Multi-GPU Profiling

```bash
# Profile all GPUs in a multi-GPU run
nsys profile --trace=cuda,nvtx,cudnn,cublas \
    --gpu-metrics-devices=all \
    --cuda-memory-usage=true \
    -o multi_gpu_profile \
    torchrun --nproc_per_node=4 train.py

# Profile specific GPU rank with wrapper script
# wrapper.sh:
if [ "$LOCAL_RANK" = "0" ]; then
    nsys profile --trace=cuda,nvtx -o gpu0_profile "$@"
else
    "$@"
fi

# Usage:
torchrun --nproc_per_node=4 wrapper.sh python train.py
```

### 2.5 Network Profiling (NCCL)

```bash
# Trace NCCL communication alongside CUDA
nsys profile --trace=cuda,nvtx,nccl \
    --gpu-metrics-devices=all \
    -o nccl_profile \
    python distributed_train.py
```

In the timeline, NCCL operations appear as distinct bars showing:
- AllReduce, AllGather, ReduceScatter durations
- Overlap (or lack thereof) between communication and computation
- NVLink vs. PCIe transfer patterns

### 2.6 CLI Usage Reference

```bash
# Basic CUDA + NVTX trace
nsys profile --trace=cuda,nvtx -o output python my_script.py

# Comprehensive trace with statistics
nsys profile \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --stats=true \
    --force-overwrite=true \
    --sample=cpu \
    --cudabacktrace=true \
    --gpu-metrics-devices=all \
    --cuda-memory-usage=true \
    -o comprehensive_profile \
    python my_script.py

# Delayed start (skip initialization)
nsys profile --trace=cuda,nvtx --delay=10 -o output python my_script.py

# Duration-limited profiling
nsys profile --trace=cuda,nvtx --duration=30 -o output python my_script.py

# Generate summary statistics from existing report
nsys stats output.nsys-rep

# Export to SQLite for custom analysis
nsys export --type=sqlite output.nsys-rep

# View report in terminal
nsys stats --report cuda_gpu_kern_sum output.nsys-rep
```

### 2.7 Common Bottleneck Patterns in Timeline

| Pattern | Visual Signature | Root Cause | Fix |
|---|---|---|---|
| **GPU idle gaps** | White space between kernel bars | CPU bottleneck or sync stalls | Overlap CPU/GPU work, use async APIs |
| **Many tiny kernels** | Dense thin bars with gaps | Launch overhead dominates | Fuse kernels, use CUDA graphs |
| **Long memcpy bars** | Wide bars in memory row | Excessive data transfer | Pin memory, use async transfers, reduce transfers |
| **Staircase kernels** | Sequential non-overlapping bars | Default stream serialization | Use multiple streams |
| **Sync spikes** | Long cudaDeviceSynchronize bars | Unnecessary synchronization | Remove or defer syncs |
| **Compilation bursts** | Long initial CPU activity, no GPU | torch.compile / JIT | Expected on first run; use warmup |
| **Uneven multi-GPU** | One GPU busy, others idle | Load imbalance | Rebalance data/compute across GPUs |

---

## 3. Roofline Model Deep Dive

The roofline model is the most important mental framework for understanding kernel performance. It plots achieved performance (FLOP/s) against arithmetic intensity (FLOP/byte) on a log-log chart.

### 3.1 Core Formula

```
Arithmetic Intensity (AI) = FLOPs / Bytes_transferred

Achievable Performance = min(Peak_Compute, Peak_Bandwidth x AI)

Ridge Point = Peak_Compute / Peak_Bandwidth
```

- **Below ridge point:** Memory-bound (performance limited by bandwidth)
- **Above ridge point:** Compute-bound (performance limited by compute throughput)
- **At ridge point:** Perfectly balanced between compute and memory

### 3.2 Arithmetic Intensity by Kernel Type

#### GEMM (General Matrix Multiply): C[M,N] = A[M,K] x B[K,N]

```
FLOPs = 2 * M * N * K                    (multiply-add for each output element)
Bytes  = (M*K + K*N + M*N) * element_size (read A, read B, write C)

AI_gemm = 2*M*N*K / ((M*K + K*N + M*N) * element_size)
```

**Example (square matrices, FP16):**
```
M = N = K = 4096, element_size = 2 bytes
FLOPs = 2 * 4096^3 = 137.4 GFLOPs
Bytes = (4096^2 + 4096^2 + 4096^2) * 2 = 100.7 MB
AI = 137.4e9 / 100.7e6 = 1365 FLOP/byte  -> Compute-bound on any GPU
```

**Example (skinny matrices, FP16):**
```
M = 1, N = 4096, K = 4096 (matrix-vector multiply)
FLOPs = 2 * 1 * 4096 * 4096 = 33.6 MFLOPs
Bytes = (1*4096 + 4096*4096 + 1*4096) * 2 = 33.6 MB
AI = 33.6e6 / 33.6e6 = 1.0 FLOP/byte  -> Extremely memory-bound
```

**Key insight:** GEMM arithmetic intensity scales with matrix dimensions. Small batch sizes (M=1 in decode) make GEMM memory-bound. Large batch sizes make it compute-bound.

#### Attention

**Naive Attention:**
```
Q, K, V are [B, H, N, d] where B=batch, H=heads, N=seq_len, d=head_dim

Step 1: S = Q @ K^T     -> FLOPs: 2*B*H*N*N*d,  Output: B*H*N*N
Step 2: P = softmax(S)   -> FLOPs: ~3*B*H*N*N,    I/O: read+write B*H*N*N
Step 3: O = P @ V        -> FLOPs: 2*B*H*N*N*d,  Output: B*H*N*d

Total FLOPs ~ 4*B*H*N^2*d + 3*B*H*N^2
Total Bytes ~ B*H*(4*N*d + 2*N^2 + 2*N*d) * element_size  (if materializing S)
```

**Attention Arithmetic Intensity (simplified per head):**
```
AI_attention ~ (4*N*d + 3*N) / (4*N*d + 4*N^2 + 2*N*d) * (1/element_size)

For N >> d (long sequences): AI ~ (4d)/(4N) = d/N  -> Very low, memory-bound
For N << d (short sequences): AI ~ 2  -> Still low
```

**Prefill (self-attention, S = T = sequence_length):**
```
AI_prefill ~ T/2  (simplified)
For T = 2048: AI ~ 1024 FLOP/byte -> Compute-bound
For T = 128:  AI ~ 64 FLOP/byte   -> Borderline
```

**Decode (autoregressive, T = 1, S = context_length):**
```
AI_decode ~ S/(S+1) ~ 1 FLOP/byte -> Always memory-bound
```

This is precisely why FlashAttention exists: it avoids materializing the N x N attention matrix to HBM, reducing memory traffic from O(N^2) to O(N) while keeping FLOPs the same (actually slightly higher due to recomputation).

#### Element-wise Operations (ReLU, GELU, SiLU, add, multiply)

```
FLOPs = 1-5 per element (depending on operation)
Bytes = 2 * element_size per element (read input + write output)

AI = FLOPs_per_element / (2 * element_size)

For FP16 ReLU: AI = 1 / (2 * 2) = 0.25 FLOP/byte -> Extremely memory-bound
For FP16 GELU: AI = 5 / (2 * 2) = 1.25 FLOP/byte -> Still memory-bound
```

**Element-wise operations are ALWAYS memory-bound.** The only optimization is to fuse them with adjacent operations to avoid extra reads/writes.

#### Reduction Operations (LayerNorm, Softmax, RMSNorm)

```
LayerNorm over dim D:
  FLOPs ~ 5*D per row (mean, variance, normalize, scale, bias)
  Bytes ~ 3*D*element_size per row (read input, read params, write output)
  AI ~ 5/(3*element_size) ~ 0.83 FLOP/byte (FP16)

Softmax over dim D:
  FLOPs ~ 3*D per row (max, exp, sum, divide)
  Bytes ~ 2*D*element_size per row
  AI ~ 3/(2*element_size) ~ 0.75 FLOP/byte (FP16)
```

**Reductions are always memory-bound.** Fuse with upstream/downstream operations whenever possible.

### 3.3 Ridge Point Calculation for Major GPUs

The ridge point is where the memory roofline meets the compute roofline:

```
Ridge Point = Peak_FLOPS / Peak_Bandwidth
```

**FP16 Tensor Core Ridge Points (FLOP/byte):**

| GPU | Peak FP16 TC (TFLOPS) | Peak BW (TB/s) | Ridge Point (FLOP/byte) |
|---|---|---|---|
| **A100 SXM** | 312 | 2.039 | 153 |
| **H100 SXM** | 1979 (with sparsity) / 990 (dense) | 3.35 | 296 (dense) |
| **H200 SXM** | 990 (dense) | 4.89 | 202 (dense) |
| **B200 SXM** | 2250 (dense) | 8.0 | 281 (dense) |
| **RTX 3090** | 71 (dense) | 0.936 | 76 |
| **RTX 4090** | 165 (dense) | 1.008 | 164 |
| **RTX 5090** | 838 (dense) | 1.792 | 468 |
| **L40S** | 366 (dense) | 0.864 | 424 |

**Interpretation:** A kernel with arithmetic intensity of 100 FLOP/byte is:
- Compute-bound on RTX 3090 (ridge = 76)
- Memory-bound on A100 (ridge = 153)
- Memory-bound on H100 (ridge = 296)

### 3.4 Hierarchical Roofline (L1/L2/DRAM)

Nsight Compute provides hierarchical roofline analysis with three ceilings:

```
L1 Cache Roofline:  AI_L1 = FLOPs / L1_bytes_transferred
L2 Cache Roofline:  AI_L2 = FLOPs / L2_bytes_transferred
DRAM Roofline:      AI_DRAM = FLOPs / DRAM_bytes_transferred
```

Each level has a different bandwidth ceiling:
- **L1:** ~19 TB/s on A100 (per SM, ~128 bytes/clock * 108 SMs * 1.4 GHz)
- **L2:** ~5 TB/s on A100
- **DRAM:** ~2 TB/s on A100

A kernel may be:
- **DRAM-bound** but far from L2 ceiling (good L2 hit rate helps)
- **L2-bound** (L2 is the bottleneck, not DRAM; improve L2 hit rate)
- **L1-bound** (rare; indicates very hot shared memory / L1 access)

### 3.5 How to Shift a Kernel from Memory-Bound to Compute-Bound

1. **Kernel Fusion:** Combine element-wise/reduction ops with GEMM epilogue to avoid intermediate writes to global memory.
2. **Tiling:** Load data into shared memory once, compute multiple operations on it before writing back.
3. **Increase Batch Size:** For GEMM, larger M dimension increases arithmetic intensity linearly.
4. **Lower Precision:** FP16/FP8 halves/quarters memory traffic while compute throughput stays same or increases (tensor cores).
5. **Operator Reordering:** Perform high-AI operations (GEMM) first, fuse low-AI operations (norm, activation) into epilogue.
6. **Recomputation:** Like FlashAttention -- recompute intermediate values instead of storing them (trades compute for memory).

---

## 4. Benchmarking Methodology

### 4.1 Warm-Up Runs

**Always discard the first N iterations.** First-run overhead includes:
- CUDA context initialization (can be 100ms+)
- JIT compilation (PTX to SASS assembly)
- `torch.compile` compilation passes (can be 10-200 seconds)
- cuDNN autotuning (`torch.backends.cudnn.benchmark=True` microbenchmarks algorithms)
- CUDA lazy kernel loading (CUDA 11.7+)
- Memory allocator pool initialization (PyTorch caching allocator)
- GPU clock ramp-up from idle to boost frequency

**Recommended:** 10-50 warm-up iterations before measurement. For `torch.compile`, 3+ full iterations (first for compilation, second for secondary traces, third+ for steady state).

### 4.2 CUDA Event Timing (Most Accurate GPU-Side Timing)

**C/C++:**
```c
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

// Warm up
for (int i = 0; i < WARMUP; i++) {
    my_kernel<<<grid, block, 0, stream>>>(args);
}

// Measure
float times[NUM_ITERS];
for (int i = 0; i < NUM_ITERS; i++) {
    cudaEventRecord(start, stream);  // MUST be same stream as kernel
    my_kernel<<<grid, block, 0, stream>>>(args);
    cudaEventRecord(stop, stream);   // MUST be same stream as kernel
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&times[i], start, stop);
}

cudaEventDestroy(start);
cudaEventDestroy(stop);
```

**Python (PyTorch):**
```python
import torch

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

# Warm up
for _ in range(50):
    output = model(input_tensor)

# Measure
times = []
for _ in range(100):
    start.record()
    output = model(input_tensor)
    end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end))  # milliseconds
```

### 4.3 torch.cuda.synchronize() Timing

```python
import time
import torch

torch.cuda.synchronize()  # Ensure GPU is idle before starting
start = time.perf_counter()
output = model(input_tensor)
torch.cuda.synchronize()  # Wait for GPU to finish
elapsed = time.perf_counter() - start
```

**Warning:** This measures wall-clock time including kernel launch overhead (~5-10us per launch). For fast kernels (<100us), CUDA events are more accurate. For application-level benchmarks, wall-clock timing is more representative.

### 4.4 Statistical Analysis

Never report a single measurement. Collect 100+ samples and report:

```python
import numpy as np

times = np.array(measured_times)

stats = {
    "mean": np.mean(times),
    "median": np.median(times),
    "std": np.std(times),
    "min": np.min(times),
    "max": np.max(times),
    "p95": np.percentile(times, 95),
    "p99": np.percentile(times, 99),
    "cv": np.std(times) / np.mean(times) * 100,  # coefficient of variation %
    "trimmed_mean": scipy.stats.trim_mean(times, 0.1),  # trim 10% from each tail
}
```

**Which statistic to use:**
- **Median:** Best single number for "typical" performance (robust to outliers)
- **Minimum:** Theoretical best case (unrealistic for sustained workloads)
- **P99:** Worst-case latency for SLA-bound applications
- **Trimmed mean (10%):** Good balance between mean and median
- **Coefficient of variation:** If CV > 5%, measurements are noisy; investigate clock throttling, contention

### 4.5 Power Measurement

```bash
# One-shot power query
nvidia-smi -q -d POWER

# Continuous monitoring (1 sample/second)
nvidia-smi dmon -s p -d 1

# Log power to file during benchmark
nvidia-smi dmon -s p -d 1 -f power_log.csv &
PID=$!
./my_benchmark
kill $PID

# Detailed power breakdown
nvidia-smi -q -d POWER
# Reports: Power Draw, Power Limit, Default Power Limit, Min/Max Power Limit,
#          Current Power Limit, Enforced Power Limit

# Set power limit (requires root)
sudo nvidia-smi -pl 300   # Set to 300W
```

### 4.6 Thermal Throttling Detection

```bash
# Check current clocks and throttle reasons
nvidia-smi -q -d CLOCK,PERFORMANCE

# Throttle reasons bitmask
nvidia-smi --query-gpu=clocks_throttle_reasons.active \
    --format=csv,noheader

# Common throttle reasons:
# clocks_throttle_reasons.gpu_idle
# clocks_throttle_reasons.sw_power_cap      (software power limit)
# clocks_throttle_reasons.hw_slowdown       (thermal or power protection)
# clocks_throttle_reasons.hw_thermal_slowdown
# clocks_throttle_reasons.sw_thermal_slowdown

# Lock clocks for reproducible benchmarking
sudo nvidia-smi -lgc 1800,1800  # Lock GPU clock to 1800 MHz
sudo nvidia-smi -lmc 1593       # Lock memory clock (if supported)

# Reset clocks
sudo nvidia-smi -rgc
sudo nvidia-smi -rmc
```

**Important caveat:** Power constraints override manual clock settings. Even locked clocks will throttle if the workload exceeds the power limit. Set clock frequency below maximum to ensure it stays locked.

### 4.7 L2 Cache Flushing

For fair comparison between kernels (cold-cache measurement):

```c
// C/CUDA
int l2_size;
cudaDeviceGetAttribute(&l2_size, cudaDevAttrL2CacheSize, 0);
void* flush_buf;
cudaMalloc(&flush_buf, l2_size * 2);  // 2x L2 size to ensure full flush

// Before each timed iteration:
cudaMemsetAsync(flush_buf, 0, l2_size * 2, stream);
cudaStreamSynchronize(stream);
```

```python
# PyTorch
l2_size = 40 * 1024 * 1024  # 40MB for A100, 50MB for H100
flush_buf = torch.zeros(l2_size // 4, dtype=torch.float32, device='cuda')

# Before each timed iteration:
flush_buf.zero_()
torch.cuda.synchronize()
```

**When to flush:** Memory-bound kernels (GEMV, element-wise) are highly sensitive to cache state. Compute-bound kernels (large GEMM) are mostly unaffected.

### 4.8 triton.testing.do_bench

```python
import triton

# Basic usage
ms = triton.testing.do_bench(lambda: my_kernel(args))

# With warmup and repetition control
ms = triton.testing.do_bench(
    lambda: my_kernel(args),
    warmup=100,       # Warmup iterations (default: 25)
    rep=1000,         # Measurement iterations (default: 100)
    quantiles=[0.5, 0.2, 0.8],  # Return median, 20th, 80th percentiles
)

# Benchmarking across configurations
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M'],
        x_vals=[128 * i for i in range(2, 33)],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'PyTorch'],
        ylabel='TFLOPS',
        plot_name='matmul-performance',
        args={'N': 4096, 'K': 4096},
    )
)
def benchmark(M, N, K, provider):
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: triton_matmul(a, b))
    else:
        ms = triton.testing.do_bench(lambda: torch.matmul(a, b))
    tflops = 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return tflops
```

`do_bench` handles: CUDA synchronization, warm-up iterations, L2 cache flushing (optional), and returns time in milliseconds.

### 4.9 Benchmarking Across Batch Sizes and Sequence Lengths

```python
import itertools

batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192]

results = {}
for bs, seq_len in itertools.product(batch_sizes, seq_lengths):
    input_tensor = torch.randn(bs, seq_len, hidden_dim, device='cuda', dtype=torch.float16)

    # Warm up
    for _ in range(10):
        _ = model(input_tensor)

    # Measure
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(50):
        start.record()
        _ = model(input_tensor)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    results[(bs, seq_len)] = {
        "median_ms": np.median(times),
        "tokens_per_sec": bs * seq_len / (np.median(times) / 1000),
        "tflops": compute_tflops(bs, seq_len, np.median(times)),
    }
```

### 4.10 Best Practices Checklist

1. Enable persistent GPU mode: `nvidia-smi -pm 1`
2. Lock GPU clocks below max boost: `nvidia-smi -lgc <freq>,<freq>`
3. Run 10-50 warm-up iterations (3+ for torch.compile)
4. Use CUDA events for kernel-level timing
5. Collect 100+ measurement samples
6. Report median + p99 + standard deviation
7. Flush L2 cache for cold-cache benchmarks of memory-bound kernels
8. Monitor power draw and temperature throughout
9. Ensure no other GPU workloads are running
10. Record: GPU model, driver version, CUDA version, compilation flags
11. For cloud GPUs: verify exact SKU (SXM vs PCIe, power cap, virtualization)
12. Compile for exact target architecture: `nvcc -arch=sm_80` (avoid JIT)

---

## 5. Performance Counters and Hardware Metrics

### 5.1 Clock Monitoring

```bash
# Real-time clock frequencies
nvidia-smi dmon -s c -d 1
# Columns: gpu, sm_clk(MHz), mem_clk(MHz)

# Query specific clocks
nvidia-smi --query-gpu=clocks.current.sm,clocks.current.memory,clocks.max.sm,clocks.max.memory \
    --format=csv

# Application clocks (for data center GPUs)
nvidia-smi --query-gpu=clocks.applications.graphics,clocks.applications.memory \
    --format=csv
```

### 5.2 GPU Utilization vs. SM Active Cycles

```bash
# GPU utilization (coarse, percentage of time at least one kernel is running)
nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv -l 1

# More detailed: use nvidia-smi dmon
nvidia-smi dmon -s u -d 1
# Columns: gpu, sm%, mem%, enc%, dec%
```

**Important:** `nvidia-smi` GPU utilization only reports whether the GPU is doing *anything*, not how efficiently. A kernel using 1 SM on a 108-SM GPU shows 100% utilization. Use Nsight Compute's SM throughput for actual SM efficiency.

### 5.3 PCIe Bandwidth Measurement

```bash
# Real-time PCIe throughput
nvidia-smi dmon -s t -d 1
# Columns: gpu, pcie_tx(KB/s), pcie_rx(KB/s)

# PCIe link info
nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current \
    --format=csv

# Theoretical PCIe bandwidth:
# Gen3 x16: ~15.75 GB/s per direction
# Gen4 x16: ~31.5 GB/s per direction
# Gen5 x16: ~63 GB/s per direction
```

**Programmatic measurement:**
```python
import torch
import time

# Measure H2D bandwidth
size = 1024 * 1024 * 1024  # 1 GB
cpu_tensor = torch.randn(size // 4, dtype=torch.float32).pin_memory()
gpu_tensor = torch.empty_like(cpu_tensor, device='cuda')

torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(10):
    gpu_tensor.copy_(cpu_tensor)
    torch.cuda.synchronize()
elapsed = time.perf_counter() - start

bandwidth_gbps = (size * 10) / elapsed / 1e9
print(f"H2D Bandwidth: {bandwidth_gbps:.2f} GB/s")
```

### 5.4 NVLink Bandwidth Measurement

```bash
# NVLink throughput
nvidia-smi nvlink -gt d    # Data throughput
nvidia-smi nvlink -gt r    # Raw throughput
nvidia-smi nvlink -sc 0    # Show counters for link 0

# NVLink status
nvidia-smi nvlink -s       # Link status
nvidia-smi nvlink -c        # Capabilities

# Programmatic: peer-to-peer bandwidth
# Use CUDA bandwidthTest sample or custom benchmark
```

**NVLink Theoretical Bandwidth:**
| Generation | Per-Link (bidirectional) | Total (typical config) |
|---|---|---|
| NVLink 3.0 (A100) | 50 GB/s | 600 GB/s (12 links) |
| NVLink 4.0 (H100) | 50 GB/s | 900 GB/s (18 links) |
| NVLink 5.0 (B200) | 100 GB/s | 1800 GB/s (18 links) |

### 5.5 ECC Error Monitoring

```bash
# Query ECC status and error counts
nvidia-smi -q -d ECC

# Specific ECC queries
nvidia-smi --query-gpu=ecc.errors.corrected.volatile.total,\
ecc.errors.uncorrected.volatile.total \
    --format=csv -l 1

# Reset ECC counters
nvidia-smi --reset-ecc-errors=volatile

# ECC modes
nvidia-smi --query-gpu=ecc.mode.current,ecc.mode.pending --format=csv
```

**ECC Impact:** ECC enabled reduces available memory by ~6% (e.g., A100 80GB -> ~75GB usable) and reduces memory bandwidth by ~3-5%. For ML workloads, ECC is typically disabled on consumer GPUs and enabled on data center GPUs.

### 5.6 Power Consumption Profiling

```bash
# Continuous power monitoring
nvidia-smi dmon -s p -d 1
# Columns: pwr(W), temp(C), gtemp(C)

# Detailed power info
nvidia-smi -q -d POWER
# Reports: GPU Power Readings, Power Draw, Current Power Limit,
#          Default/Min/Max Power Limits, Enforced Power Limit

# Monitor power during benchmark (script)
nvidia-smi dmon -s p -d 1 -f power.csv &
MONITOR_PID=$!
python benchmark.py
kill $MONITOR_PID

# Compute energy efficiency
# energy_per_token = avg_power_watts * time_per_token_seconds
# TFLOPS_per_watt = achieved_tflops / avg_power_watts
```

---

## 6. Common Performance Anti-Patterns

### 6.1 Kernel Launch Overhead (Too Many Small Kernels)

**Problem:** Each CUDA kernel launch has ~5-20us of CPU-side overhead. If your kernel runs for <50us, launch overhead is >10% of execution time.

**Symptoms in Nsight Systems:** Dense thin bars with visible gaps between them. CPU CUDA API row shows `cudaLaunchKernel` consuming significant time.

**Solutions:**
- **Kernel fusion:** Combine multiple small operations into one kernel
- **CUDA Graphs:** Capture a sequence of kernels and launch them as a single graph
- **torch.compile:** Automatically fuses eligible operations
- **Triton:** Write fused kernels that combine element-wise + reduction + activation

```python
# CUDA Graphs in PyTorch
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    output = model(static_input)
# Replay without launch overhead:
g.replay()
```

### 6.2 Host-Device Synchronization Stalls

**Problem:** `cudaDeviceSynchronize()`, `torch.cuda.synchronize()`, or implicit syncs (e.g., `.item()`, `.cpu()`, `print(tensor)`) force the CPU to wait for all GPU work to finish, creating a bubble.

**Symptoms in Nsight Systems:** Long `cudaDeviceSynchronize` bars in CUDA API row, with GPU going idle after the sync returns.

**Common hidden syncs in PyTorch:**
```python
tensor.item()           # Syncs to copy scalar to CPU
tensor.cpu()            # Syncs to copy tensor to CPU
print(tensor)           # Syncs to read values
tensor.numpy()          # Syncs to copy to CPU
if tensor > threshold:  # Syncs to evaluate condition on CPU
loss_list.append(loss)  # If loss is on GPU, prevents garbage collection
```

**Fix:** Batch all GPU->CPU transfers, use non-blocking transfers, avoid reading GPU tensors until necessary.

### 6.3 Uncoalesced Memory Access

**Problem:** When threads in a warp access non-contiguous memory addresses, the hardware must issue multiple memory transactions instead of one, wasting bandwidth.

**Coalesced (good):** Thread i accesses `array[base + i]` -- one 128-byte transaction serves all 32 threads.
**Uncoalesced (bad):** Thread i accesses `array[base + i * stride]` where stride >> 1 -- up to 32 separate 32-byte transactions.

**Impact:** Up to **32x** bandwidth waste in worst case (12.5% memory utilization).

**Detection in Nsight Compute:** Check `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` / `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum`. Ratio >> 4 (for 32-bit) indicates poor coalescing.

**Fixes:**
- Ensure innermost loop index maps to contiguous memory
- Use Structure of Arrays (SoA) instead of Array of Structures (AoS)
- For 2D arrays: `C[row][col]` where `col = threadIdx.x` (row-major)
- Use `float4` / `int4` vector loads for wider transactions

### 6.4 Shared Memory Bank Conflicts

**Problem:** Shared memory is divided into 32 banks (4 bytes per bank). When multiple threads in a warp access different addresses in the same bank, accesses are serialized.

**N-way conflict:** N threads hit same bank -> N serial rounds -> throughput / N.
**Worst case:** 32-way conflict -> 1/32 throughput.
**Exception:** All threads accessing the *same* address is a broadcast, not a conflict.

**Detection in Nsight Compute:** Compare `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum` to `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld_ideal.sum`. Ratio > 1 indicates bank conflicts.

**Fix:** Add padding to shared memory arrays:
```c
// BAD: column access causes 32-way bank conflicts
__shared__ float smem[32][32];
// GOOD: padding eliminates bank conflicts
__shared__ float smem[32][33];  // +1 padding per row
```

### 6.5 Register Spilling

**Problem:** When a kernel uses more registers than available per thread, the compiler "spills" excess to local memory (actually global memory, cached in L1/L2). Local memory access is ~100x slower than register access.

**Detection in Nsight Compute:**
- `launch__registers_per_thread` shows register count
- `l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum` > 0 indicates spilling
- `lmem_used` in kernel launch statistics

**Fixes:**
- Use `__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)` to guide compiler
- Reduce variable scope and lifetime
- Trade register pressure for recomputation
- Use shared memory for frequently accessed values
- Accept lower ILP if register pressure is extreme

```c
// Guide compiler to use fewer registers
__global__ void __launch_bounds__(256, 4)
my_kernel(float* data) {
    // Compiler targets 256 threads/block, 4 blocks/SM
    // -> max 64 registers/thread on SM 8.0 (65536 / 256 / 4)
}
```

### 6.6 Low Occupancy from Resource Limits

**Problem:** Too few active warps per SM means the GPU cannot hide memory latency effectively.

**Occupancy limiters (check each):**

| Limiter | A100 Limit | H100 Limit | How to Check |
|---|---|---|---|
| Registers/thread | 65536/SM | 65536/SM | `launch__registers_per_thread` |
| Shared mem/block | 164 KB configurable | 228 KB configurable | `launch__shared_mem_per_block` |
| Threads/block | Max 1024 | Max 1024 | Block dimensions |
| Blocks/SM | Max 32 | Max 32 | Grid vs. SM count |
| Warps/SM | Max 64 (2048 threads) | Max 64 (2048 threads) | Theoretical occupancy |

**Rules of thumb:**
- 50% occupancy is often sufficient for compute-bound kernels
- Memory-bound kernels benefit more from higher occupancy (more warps to hide latency)
- Use NVIDIA's Occupancy Calculator spreadsheet or Nsight Compute's occupancy section

### 6.7 Warp Divergence in Conditional Code

**Problem:** When threads in a warp take different branch paths, the warp executes *both* paths serially, with threads not on the active path masked off.

```c
// BAD: High divergence -- even/odd threads take different paths
if (threadIdx.x % 2 == 0) {
    do_expensive_work_A();
} else {
    do_expensive_work_B();
}

// BETTER: Minimize divergence -- warp 0 does A, warp 1 does B
if (threadIdx.x / 32 % 2 == 0) {
    do_expensive_work_A();
} else {
    do_expensive_work_B();
}
```

**Detection in Nsight Compute:** Compare `sm__inst_executed.sum` (actual instructions) to `sm__inst_issued.sum` (issued with mask). High ratio of predicated-off instructions indicates divergence.

### 6.8 Unnecessary Data Type Conversions

**Problem:** Converting between FP32 and FP16 at kernel boundaries wastes bandwidth and adds instructions.

```python
# BAD: Repeated casting
x = x.half()       # Convert to FP16
x = linear(x)      # Compute in FP16
x = x.float()      # Convert back to FP32
x = relu(x)        # Compute in FP32
x = x.half()       # Convert to FP16 again
```

**Fix:** Use `torch.autocast` / AMP to manage precision automatically, or keep tensors in FP16/BF16 throughout.

### 6.9 cudaMalloc in Hot Path

**Problem:** `cudaMalloc` / `cudaFree` are expensive (100us-1ms) and synchronize the device. Calling them in the inference loop destroys performance.

**Fix:**
- Pre-allocate all buffers before the hot loop
- Use PyTorch's caching allocator (default behavior)
- Use memory pools: `cudaMemPoolCreate` / `cudaMallocAsync`

### 6.10 Default Stream Serialization

**Problem:** All CUDA operations on the default (null) stream are serialized. Independent operations cannot overlap.

**Fix:** Use multiple CUDA streams for independent work:
```python
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

with torch.cuda.stream(stream1):
    output1 = model_part1(input1)

with torch.cuda.stream(stream2):
    output2 = model_part2(input2)

# Both execute concurrently on GPU
```

---

## 7. NVIDIA DALI

NVIDIA Data Loading Library (DALI) offloads data preprocessing from CPU to GPU, eliminating the CPU bottleneck in training pipelines.

### 7.1 Why DALI

- **CPU bottleneck:** Standard PyTorch DataLoader uses CPU for decoding, resizing, augmentation. With fast GPUs, the CPU cannot feed data fast enough.
- **GPU-accelerated:** DALI runs decoding (JPEG->RGB), resizing, cropping, normalization, augmentation on GPU.
- **Zero-copy:** Data stays on GPU from decode through training, avoiding PCIe transfers.
- **Prefetching:** Built-in async prefetching hides data loading latency.

### 7.2 Pipeline Definition

```python
from nvidia.dali import pipeline_def, Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types

@pipeline_def
def training_pipeline(data_dir, num_gpus):
    device_id = Pipeline.current().device_id

    # Read data (CPU)
    jpegs, labels = fn.readers.file(
        name="Reader",
        file_root=data_dir,
        random_shuffle=True,
        shard_id=device_id,
        num_shards=num_gpus,
    )

    # Decode on GPU (mixed = CPU input, GPU output)
    images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)

    # GPU operations
    images = fn.resize(
        images,
        resize_shorter=fn.random.uniform(range=(256, 480)),
        interp_type=types.INTERP_LINEAR,
    )

    images = fn.crop_mirror_normalize(
        images,
        crop_pos_x=fn.random.uniform(range=(0.0, 1.0)),
        crop_pos_y=fn.random.uniform(range=(0.0, 1.0)),
        dtype=types.FLOAT,
        crop=(224, 224),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        mirror=fn.random.coin_flip(probability=0.5),
    )

    return images, labels
```

### 7.3 PyTorch Integration

```python
from nvidia.dali.plugin.pytorch import DALIGenericIterator

BATCH_SIZE = 256
NUM_GPUS = 4

# Build pipelines (one per GPU)
pipes = [
    training_pipeline(
        batch_size=BATCH_SIZE,
        num_threads=4,
        device_id=device_id,
        data_dir="/data/imagenet/train",
        num_gpus=NUM_GPUS,
    )
    for device_id in range(NUM_GPUS)
]

for pipe in pipes:
    pipe.build()

# Create PyTorch iterator
train_loader = DALIGenericIterator(
    pipes,
    ["data", "label"],
    reader_name="Reader",
    auto_reset=True,
)

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        images = batch[0]["data"]   # Already on GPU as torch.Tensor
        labels = batch[0]["label"]  # Already on GPU

        output = model(images)
        loss = criterion(output, labels.squeeze().long())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    train_loader.reset()  # Reset for next epoch
```

### 7.4 Key DALI Operators

| Operator | Description | Device |
|---|---|---|
| `fn.readers.file` | Read files from directory | CPU |
| `fn.readers.tfrecord` | Read TFRecord files | CPU |
| `fn.readers.caffe` | Read LMDB (Caffe format) | CPU |
| `fn.decoders.image` | JPEG/PNG decode | CPU or Mixed (GPU decode) |
| `fn.resize` | Resize images | CPU or GPU |
| `fn.crop_mirror_normalize` | Crop, mirror, normalize | CPU or GPU |
| `fn.color_twist` | Brightness, contrast, hue | GPU |
| `fn.rotate` | Image rotation | GPU |
| `fn.gaussian_blur` | Gaussian blur | GPU |
| `fn.random.coin_flip` | Random boolean | CPU |
| `fn.random.uniform` | Random uniform distribution | CPU |

### 7.5 DALI Proxy (Selective Offloading)

For multi-modal or complex pipelines, DALI Proxy lets you offload only specific preprocessing steps:

```python
from nvidia.dali.plugin.pytorch import DALIProxy

# Define which preprocessing to offload to DALI
proxy = DALIProxy(pipeline=my_dali_pipeline)

# Use within existing PyTorch DataLoader
dataset = MyCustomDataset(transform=proxy)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=4)
```

---

## 8. GPU Bandwidth/FLOPS Reference Table

### 8.1 Data Center GPUs

| GPU | Arch | CUDA Cores | Tensor Cores | Memory | Mem BW | FP64 | FP64 TC | FP32 | TF32 TC | FP16/BF16 TC | FP8 TC | INT8 TC | TDP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **V100 SXM** | Volta | 5120 | 640 | 32GB HBM2 | 900 GB/s | 7.8 TF | 7.8 TF | 15.7 TF | -- | 125 TF | -- | -- | 300W |
| **A100 SXM** | Ampere | 6912 | 432 | 80GB HBM2e | 2039 GB/s | 9.7 TF | 19.5 TF | 19.5 TF | 156 TF / 312* | 312 TF / 624* | -- | 624 TO / 1248* | 400W |
| **H100 SXM** | Hopper | 16896 | 528 | 80GB HBM3 | 3350 GB/s | 34 TF | 67 TF | 67 TF | 495 TF / 990* | 990 TF / 1979* | 1979 TF / 3958* | 1979 TO / 3958* | 700W |
| **H200 SXM** | Hopper | 16896 | 528 | 141GB HBM3e | 4890 GB/s | 34 TF | 67 TF | 67 TF | 495 TF / 990* | 990 TF / 1979* | 1979 TF / 3958* | 1979 TO / 3958* | 700W |
| **B100** | Blackwell | -- | -- | 192GB HBM3e | 8000 GB/s | 30 TF | 30 TF | 60 TF | 1125 TF / 2250* | 1800 TF / 3500* | 3500 TF / 7000* | 3500 TO / 7000* | 700W |
| **B200 SXM** | Blackwell | -- | -- | 192GB HBM3e | 8000 GB/s | 40 TF | 40 TF | 80 TF | 1125 TF / 2250* | 2250 TF / 4500* | 4500 TF / 9000* | 4500 TO / 9000* | 1000W |
| **L4** | Ada | 7680 | 240 | 24GB GDDR6 | 300 GB/s | -- | -- | 30.3 TF | 120 TF / 242* | 242 TF / 485* | 485 TF / 970* | 485 TO / 970* | 72W |
| **L40S** | Ada | 18176 | 568 | 48GB GDDR6 | 864 GB/s | -- | -- | 91.6 TF | 183 TF / 366* | 366 TF / 733* | 733 TF / 1466* | 733 TO / 1466* | 350W |
| **T4** | Turing | 2560 | 320 | 16GB GDDR6 | 300 GB/s | -- | -- | 8.1 TF | -- | 65 TF | -- | 130 TO | 70W |

*\* = with structured sparsity (2:4)*

### 8.2 Consumer / Workstation GPUs

| GPU | Arch | CUDA Cores | Tensor Cores | Memory | Mem BW | FP32 | TF32 TC | FP16/BF16 TC | FP8 TC | INT8 TC | TDP |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **RTX 3090** | Ampere | 10496 | 328 | 24GB GDDR6X | 936 GB/s | 35.6 TF | 35.6 TF / 71.2* | 71.2 TF / 142.3* | -- | 284.7 TO / 569.3* | 350W |
| **RTX 4090** | Ada | 16384 | 512 | 24GB GDDR6X | 1008 GB/s | 82.6 TF | 165 TF / 331* | 331 TF / 661* | 661 TF / 1321* | 661 TO / 1321* | 450W |
| **RTX 5090** | Blackwell | 21760 | 680 | 32GB GDDR7 | 1792 GB/s | 105 TF | 419 TF / 838* | 838 TF / 1676* | 1676 TF / 3352* | 1676 TO / 3352* | 575W |
| **RTX A6000** | Ampere | 10752 | 336 | 48GB GDDR6 | 768 GB/s | 38.7 TF | 77.4 TF / 154.8* | 154.8 TF / 309.7* | -- | 309.7 TO / 619.3* | 300W |
| **RTX 6000 Ada** | Ada | 18176 | 568 | 48GB GDDR6 | 960 GB/s | 91.1 TF | 182.2 TF / 364.4* | 364.4 TF / 728.8* | 728.8 TF / 1457.7* | 728.8 TO / 1457.7* | 300W |

*\* = with structured sparsity (2:4)*

### 8.3 Ridge Points (FP16 Tensor Core Dense, FLOP/byte)

| GPU | Peak FP16 TC Dense (TFLOP/s) | Peak BW (GB/s) | Ridge Point |
|---|---|---|---|
| V100 SXM | 125 | 900 | 139 |
| A100 SXM | 312 | 2039 | 153 |
| H100 SXM | 990 | 3350 | 296 |
| H200 SXM | 990 | 4890 | 202 |
| B200 SXM | 2250 | 8000 | 281 |
| RTX 3090 | 71 | 936 | 76 |
| RTX 4090 | 331 | 1008 | 328 |
| RTX 5090 | 838 | 1792 | 468 |
| L40S | 366 | 864 | 424 |
| L4 | 242 | 300 | 807 |
| T4 | 65 | 300 | 217 |

### 8.4 Quick Reference: Interconnect Bandwidth

| Interconnect | Bandwidth (per direction) | Typical Use |
|---|---|---|
| PCIe Gen3 x16 | 15.75 GB/s | Older GPUs, T4 |
| PCIe Gen4 x16 | 31.5 GB/s | A100 PCIe, L40S, RTX 4090 |
| PCIe Gen5 x16 | 63 GB/s | H100 PCIe, RTX 5090 |
| NVLink 3.0 (A100) | 600 GB/s total (12 links) | 8-GPU NVLink mesh |
| NVLink 4.0 (H100) | 900 GB/s total (18 links) | 8-GPU NVLink mesh |
| NVLink 5.0 (B200) | 1800 GB/s total (18 links) | 8-GPU NVLink mesh |
| NVSwitch (H100) | Full bisection bandwidth | DGX H100 scale-up |
| InfiniBand NDR | 400 Gbps (50 GB/s) | Multi-node scale-out |
| InfiniBand NDR400 | 400 Gbps per port | Blackwell generation |

### 8.5 Memory Hierarchy Bandwidth (Approximate)

| Level | A100 | H100 | Notes |
|---|---|---|---|
| Registers | ~19 TB/s per SM | ~33 TB/s per SM | Fastest; limited per thread |
| Shared Memory / L1 | ~19 TB/s aggregate | ~33 TB/s aggregate | Configurable split |
| L2 Cache | ~5 TB/s | ~12 TB/s | 40MB on A100, 50MB on H100 |
| HBM (DRAM) | 2.0 TB/s | 3.35 TB/s | Main bottleneck for memory-bound kernels |
| PCIe | 31.5 GB/s (Gen4) | 63 GB/s (Gen5) | Host-device; 100x slower than HBM |

---

## Appendix: Decision Tree for Kernel Optimization

```
Start: Profile with Nsight Compute (ncu --set full)
  |
  v
Check SOL% for SM and Memory
  |
  +---> SM% high, Mem% low --> COMPUTE-BOUND
  |       |
  |       +-- Use tensor cores (WMMA/MMA)
  |       +-- Reduce instruction count
  |       +-- Lower precision (FP16->FP8)
  |       +-- Algorithmic optimization (fewer FLOPs)
  |
  +---> SM% low, Mem% high --> MEMORY-BOUND
  |       |
  |       +-- Check coalescing (transaction/request ratio)
  |       +-- Check L1/L2 hit rates
  |       +-- Use shared memory tiling
  |       +-- Fuse with adjacent kernels
  |       +-- Reduce precision (less bytes/element)
  |       +-- Increase arithmetic intensity (recompute vs. store)
  |
  +---> SM% low, Mem% low --> LATENCY-BOUND
  |       |
  |       +-- Check warp stall reasons
  |       +-- Increase occupancy (reduce registers, shared memory)
  |       +-- Increase ILP (more independent instructions)
  |       +-- Check for barriers/syncs
  |       +-- Check for warp divergence
  |
  +---> SM% high, Mem% high --> WELL-BALANCED (near hardware limits)
          |
          +-- Algorithmic changes only
          +-- Consider different hardware
          +-- Check if lower precision is acceptable
```

---

## Sources

- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [Nsight Compute CLI Documentation](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
- [Using Nsight Compute to Inspect Your Kernels (NVIDIA Blog)](https://developer.nvidia.com/blog/using-nsight-compute-to-inspect-your-kernels/)
- [Accelerating HPC Applications with Nsight Compute Roofline Analysis (NVIDIA Blog)](https://developer.nvidia.com/blog/accelerating-hpc-applications-with-nsight-compute-roofline-analysis/)
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
- [Navigating NVIDIA Nsight Systems for Efficient Profiling](https://henryhmko.github.io/posts/profiling/profiling.html)
- [How to Benchmark CUDA Kernels](https://guillesanbri.com/CUDA-Benchmarks/)
- [How to Accurately Time CUDA Kernels in PyTorch (Speechmatics)](https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch)
- [In Pursuit of High-Fidelity GPU Kernel Benchmarking (Standard Kernel)](https://standardkernel.com/blog/in-pursuit-of-high-fidelity-gpu-kernel-benchmarking/)
- [Understanding the Roofline Model (Daniel Nichols)](https://dando18.github.io/posts/2020/04/02/roofline-model)
- [All About Rooflines (JAX Scaling Book)](https://jax-ml.github.io/scaling-book/roofline/)
- [Transformer Inference Arithmetic Intensity (Saurabh Yadav)](https://www.yadavsaurabh.com/transformer-inference-arithmetic-intensity-cost-and-optimization/)
- [FlashAttention & LLM Inference on GPUs (Tyler Crosse)](https://www.tylercrosse.com/ideas/2026/gpu-p6)
- [NVIDIA DALI Documentation](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/index.html)
- [DALI PyTorch Basic Example](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/frameworks/pytorch/pytorch-basic_example.html)
- [NVIDIA Tensor Core GPU Comparison (BIZON)](https://bizon-tech.com/blog/nvidia-b200-b100-h200-h100-a100-comparison)
- [NVIDIA Data Center GPU Specs Comparison (IntuitionLabs)](https://intuitionlabs.ai/articles/nvidia-data-center-gpu-specs)
- [NVIDIA GeForce RTX 3090 Specs (WareDB)](https://www.waredb.com/processor/nvidia-geforce-rtx-3090)
- [NVIDIA GeForce RTX 5090 vs RTX 4090 (BOXX)](https://boxx.com/blog/hardware/nvidia-geforce-rtx-5090-vs-rtx-4090)
- [nvidia-smi Documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html)
- [NVIDIA Register Spilling Optimization (NVIDIA Blog)](https://developer.nvidia.com/blog/how-to-improve-cuda-kernel-performance-with-shared-memory-register-spilling)
- [Analysis-Driven Optimization Part 3 (NVIDIA Blog)](https://developer.nvidia.com/blog/analysis-driven-optimization-finishing-the-analysis-with-nvidia-nsight-compute-part-3/)
- [Profiling CUDA Kernels with NVIDIA NSight Compute (UW-Madison)](https://www.hep.wisc.edu/cms/comp/gpuprofiling.html)
- [NVIDIA Nsight Compute CERN Presentation (PDF)](https://indico.cern.ch/event/962112/contributions/4110591/attachments/2159863/3643851/CERN_Nsight_Compute.pdf)
- [Nsight Compute Performance Analysis (NASA HECC)](https://www.nas.nasa.gov/hecc/support/kb/performance-analysis-of-your-gpu-cuda-kernels-with-nsight-compute-cli_706.html)
