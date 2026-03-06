# GPU Memory Optimization: Comprehensive Deep Dive

> The definitive reference for GPU memory hierarchy, optimization techniques, and memory management for ML/AI workloads. All numbers are sourced from official NVIDIA documentation and verified benchmarks.

---

## 1. GPU Memory Hierarchy Deep Dive

### 1.1 Overview: Memory Levels, Latency, and Bandwidth

| Memory Level | Latency (cycles) | Latency (approx ns) | Bandwidth | Size (per SM) | Scope |
|---|---|---|---|---|---|
| **Registers** | ~1 | <1 ns | ~8 TB/s | 256 KB | Per-thread |
| **L1 / Shared Memory** | ~28-35 | ~5-10 ns | ~15-20 TB/s | 128-256 KB | Per-SM (threadblock) |
| **L2 Cache** | ~150-200 | ~30-50 ns | ~3-5.5 TB/s | 6-126 MB (global) | Global (all SMs) |
| **HBM/GDDR** | ~400-800 | ~200-400 ns | 0.9-8.0 TB/s | 32-192 GB | Global |
| **Constant Memory** | ~1 (cached) | <1 ns (cached) | Broadcast | 64 KB | Global (read-only) |
| **Texture Memory** | ~30 (cached) | ~5-10 ns (cached) | Cached | Varies | Global (read-only) |

The performance gap between L1 and HBM is approximately **15x in latency** -- this is the fundamental reason memory optimization matters so much.

### 1.2 Registers

Registers are the fastest storage available on the GPU. Each thread has its own private registers, invisible to other threads.

**Key Specifications Across Architectures:**

| Property | V100 (Volta) | A100 (Ampere) | H100 (Hopper) | B200 (Blackwell) |
|---|---|---|---|---|
| Register file per SM | 256 KB | 256 KB | 256 KB | 256 KB |
| Total 32-bit registers per SM | 65,536 | 65,536 | 65,536 | 65,536 |
| Max registers per thread | 255 | 255 | 255 | 255 |
| Number of SMs | 80 | 108 | 132 | 148 |
| Total register file (chip) | 20 MB | 27 MB | 33 MB | 37 MB |

**Register Pressure:**
- Each 32-bit register = 4 bytes
- A thread using 128 registers = 512 bytes of register state
- With 255 max registers/thread, max per-thread register state = 1,020 bytes
- Higher register usage per thread reduces **occupancy** (fewer concurrent warps per SM)
- The compiler can "spill" registers to local memory (actually in HBM), causing massive slowdowns

**Occupancy Calculation:**
```
Max warps per SM = Register file per SM / (registers_per_thread * warp_size * 4 bytes)
Example: 256 KB / (128 registers * 32 threads * 4 bytes) = 256 KB / 16 KB = 16 warps
vs. max of 64 warps on Blackwell CC 10.0
Occupancy = 16/64 = 25%
```

### 1.3 L1 Cache / Shared Memory

L1 cache and shared memory physically reside in the **same SRAM** on each SM. On most modern architectures, the split is configurable.

**Shared Memory Capacity Per SM:**

| GPU | Max Shared Memory/SM | Max per Thread Block | L1+SMEM+Tex Total |
|---|---|---|---|
| V100 | 96 KB | 96 KB | 128 KB |
| A100 | 164 KB | 163 KB | 192 KB |
| H100 | 228 KB | 227 KB | 256 KB |
| B200 (CC 10.0) | 228 KB | 227 KB | 256 KB |
| B200 (CC 12.0) | 128 KB | 99 KB | 128 KB |

**Total Shared Memory (Chip-wide):**

| GPU | SMs | Total SMEM |
|---|---|---|
| V100 | 80 | ~7.5 MB |
| A100 | 108 | ~17 MB |
| H100 | 132 | ~29 MB |
| B200 | 148 | ~33 MB |

**Static vs. Dynamic Allocation:**
- **Static**: Declared at compile time with `__shared__ float arr[SIZE]`. Limited to 48 KB by default; opt-in required for larger amounts via `cudaFuncSetAttribute()`
- **Dynamic**: Size specified at kernel launch via the third execution configuration parameter `<<<blocks, threads, sharedMemBytes>>>`; pointer obtained via `extern __shared__`

**Configurable Split (Legacy):** On older architectures (Kepler, Maxwell), the L1/shared memory split was explicitly configurable (e.g., 48KB/16KB or 16KB/48KB). On Volta+ architectures, the combined L1 + shared memory pool is unified and shared memory carve-out is set per-kernel.

### 1.4 Tensor Memory (TMEM) -- Blackwell Only

Blackwell introduces **256 KB of TMEM per SM**, a new memory space for tensor core inputs that no longer fit in traditional shared memory. This is separate from the 256 KB L1/SMEM pool, giving Blackwell SMs access to 512 KB of fast on-chip memory total.

### 1.5 L2 Cache

L2 cache is a **global, shared cache** accessible by all SMs. It sits between L1 and HBM in the hierarchy.

| GPU | L2 Cache Size | L2 Bandwidth (approx) |
|---|---|---|
| V100 | 6 MB | ~1.5 TB/s |
| A100 | 40 MB | ~3 TB/s |
| H100 | 50 MB | ~5.5 TB/s |
| H200 | 50 MB | ~5.5 TB/s |
| B200 | 126 MB | ~8+ TB/s |

**L2 Persistence Control (Ampere+):**
- `cudaAccessPolicyWindow` API allows setting persistence hints for specific address ranges
- Data marked as "persisting" stays in L2 longer, reducing HBM accesses for hot data
- Up to a configurable fraction of L2 can be reserved for persistent data

### 1.6 HBM / GDDR: Main GPU Memory

**HBM Specifications by GPU:**

| GPU | Memory Type | Capacity | Bandwidth (Theoretical) | Interface Width | Stacks |
|---|---|---|---|---|---|
| V100 | HBM2 | 32 GB | 900 GB/s | 4096-bit | 4 |
| A100 | HBM2e | 80 GB | 2.0 TB/s | 5120-bit | 5 |
| H100 SXM | HBM3 | 80 GB | 3.35 TB/s | 5120-bit | 5 |
| H200 | HBM3e | 141 GB | 4.8 TB/s | 6144-bit | 6 |
| B200 | HBM3e | 192 GB | 8.0 TB/s | -- | 8 |
| RTX 4090 | GDDR6X | 24 GB | 1.0 TB/s | 384-bit | N/A |
| RTX 5090 | GDDR7 | 32 GB | 1.8 TB/s | 512-bit | N/A |

**HBM Channel Architecture:**
- HBM3: 1024-bit interface per stack, divided into 16 independent 64-bit channels (or 32 pseudo-channels of 32-bit each)
- HBM3 at 6.4 Gb/s: (16 channels x 64 bits x 6.4 Gbps) / 8 = **819 GB/s per stack**
- HBM3e at 9.6 Gb/s: up to **1,229 GB/s per stack**
- HBM4 (2026+): 2048-bit interface, up to **2 TB/s per stack**

**Effective vs. Theoretical Bandwidth:**
- Theoretical bandwidth is the peak achievable with perfectly sequential, fully coalesced access patterns
- Real-world effective bandwidth is typically **60-85%** of theoretical for well-optimized kernels
- Poorly written kernels may achieve only **10-30%** of theoretical bandwidth
- Factors: access pattern, coalescing, memory controller contention, channel imbalance
- Benchmark tool: `bandwidthTest` from CUDA samples, or `ncu --set full`

### 1.7 Constant Memory

- **Size**: 64 KB global constant memory
- **Caching**: Cached in a dedicated **constant cache** (8 KB per SM on most architectures)
- **Broadcast**: When all threads in a warp read the **same** address, the access is broadcast -- effectively 1 memory transaction for 32 threads
- **Serial**: When threads read **different** addresses, accesses are serialized (worst case: 32 serial reads)
- **Best use cases**: Filter coefficients, hyperparameters, lookup tables that are read-only and uniform across threads
- **Declaration**: `__constant__ float params[1024];`

### 1.8 Texture Memory

- **Spatial locality caching**: Optimized for 2D spatial access patterns (neighboring x,y coordinates)
- **Hardware features**: Free interpolation (bilinear/trilinear), automatic boundary handling (clamp/wrap)
- **Cache**: Separate texture cache per SM, optimized for 2D locality
- **When useful for ML**:
  - Image preprocessing in CNNs (2D spatial access patterns)
  - Grid-based computations (e.g., NeRF voxel grids)
  - Any 2D or 3D data with spatial locality in access patterns
  - NOT useful for standard GEMM, attention, or linear layers (these have row/column patterns better served by L1/shared memory)

---

## 2. Memory Coalescing

### 2.1 What Makes an Access Coalesced

A **coalesced** memory access occurs when threads in a warp (32 threads) access **consecutive, aligned** addresses in global memory. The hardware combines these into the minimum number of memory transactions.

**On modern GPUs (compute capability 6.0+):**
- The fundamental unit of memory access is a **32-byte sector**
- A warp's memory requests are serviced by fetching the minimum number of 32-byte sectors needed
- For 4-byte (float) elements, a perfectly coalesced access by 32 threads = 128 bytes = **4 sectors**
- No alignment to 128-byte boundaries is required (unlike pre-Kepler GPUs)

### 2.2 Sector-Based Coalescing

```
Example: 32 threads each read a float (4 bytes) at consecutive addresses

Thread 0 reads address 0x1000  -> Sector at 0x1000 (covers 0x1000-0x101F)
Thread 1 reads address 0x1004  -> Same sector
...
Thread 7 reads address 0x101C  -> Same sector
Thread 8 reads address 0x1020  -> Sector at 0x1020 (covers 0x1020-0x103F)
...

Total: 4 sectors fetched, 128 bytes transferred, 128 bytes used = 100% efficiency
```

**Strided access disaster:**
```
Thread 0 reads address 0x1000  -> Sector at 0x1000 (only 4 bytes used out of 32)
Thread 1 reads address 0x1080  -> Sector at 0x1080 (only 4 bytes used out of 32)
...

Total: 32 sectors fetched, 1024 bytes transferred, 128 bytes used = 12.5% efficiency
```

### 2.3 Impact of Misalignment -- Measured Numbers

From NVIDIA benchmarks on GH200:

| Access Pattern | Sectors Fetched | Efficiency | Relative Performance |
|---|---|---|---|
| Coalesced (stride 1) | 8,388,900 | 100% | 1.0x (baseline) |
| Stride 2 | ~16,777,000 | 50% | ~0.5x |
| Stride 4 | ~33,554,000 | 25% | ~0.25x |
| Uncoalesced (stride 32) | 67,110,368 | 12.5% | 0.125x (**8x slower**) |

**Key metric**: Sector-to-request ratio. Healthy = **4:1** for float accesses. Unhealthy = **8:1** or higher.

### 2.4 Vectorized Loads (float2, float4, int4)

Vectorized loads allow a single thread to load 8 or 16 bytes in one instruction instead of 4.

| Type | Bytes per Load | Load Width | Benefit |
|---|---|---|---|
| `float` / `int` | 4 bytes | 32-bit | Baseline |
| `float2` / `int2` | 8 bytes | 64-bit | 2x fewer instructions |
| `float4` / `int4` | 16 bytes | 128-bit | 4x fewer instructions |

**When vectorized loads help:**
- Bandwidth-bound kernels (most ML kernels)
- Data is naturally aligned to 8 or 16 bytes
- Reduces instruction count and instruction cache pressure
- Fewer memory transactions needed per warp

**When NOT to use:**
- High register pressure kernels (vectorized loads increase register usage)
- Data not aligned to vector size boundaries
- Element size not a power of two

**Implementation pattern:**
```cuda
// Instead of:
float val = input[tid];

// Use:
float4 vals = reinterpret_cast<float4*>(input)[tid];  // loads 4 floats at once
// Process vals.x, vals.y, vals.z, vals.w
```

### 2.5 Checking Coalescing in Nsight Compute

**Key metrics:**
```
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum    # Number of memory requests
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum      # Number of sectors fetched
```

**Healthy ratio:** `sectors / requests = 4` (for 4-byte elements)

**Command:**
```bash
ncu --metrics l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum ./my_kernel
# Or use the memory table:
ncu --metrics group:memory__dram_table ./my_kernel
```

**For 2D arrays in row-major order:**
```cuda
// GOOD (coalesced): consecutive threadIdx.x maps to consecutive columns
int idx = row * width + col;  // where col = threadIdx.x

// BAD (strided): consecutive threadIdx.x maps to consecutive rows
int idx = col * height + row; // where row = threadIdx.x -- creates stride of 'height'
```

---

## 3. Shared Memory Optimization

### 3.1 Bank Conflicts

Shared memory is organized into **32 banks** of **4 bytes** each (32-bit words). Successive 32-bit words map to successive banks.

```
Address 0x00-0x03 -> Bank 0
Address 0x04-0x07 -> Bank 1
Address 0x08-0x0B -> Bank 2
...
Address 0x7C-0x7F -> Bank 31
Address 0x80-0x83 -> Bank 0   (wraps around)
```

**Bank conflict** occurs when two or more threads in the same warp access **different addresses** in the **same bank** simultaneously. The accesses are serialized.

**Conflict-free cases:**
- Each thread accesses a different bank (stride-1 for 4-byte elements)
- Multiple threads access the **exact same** address (broadcast, no conflict)

**Conflict cases and their impact:**
- **2-way conflict**: 2x serialization (half throughput)
- **4-way conflict**: 4x serialization
- **8-way conflict**: 8x serialization
- **32-way conflict**: 32x serialization (worst case, all threads hit same bank)

**Detection in Nsight Compute:**
```
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum  # Load bank conflicts
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum  # Store bank conflicts
derived__memory_l1_wavefronts_shared_excessive            # Excess wavefronts
```

### 3.2 Bank Conflict with 16-byte (Vectorized) Access

When using 16-byte vectorized loads/stores (`float4`, `int4`), the banking structure changes:
- Access is split into **4 phases** with **8 threads** each
- Creates **8 effective banks of 16 bytes** each (instead of 32 banks of 4 bytes)
- Threads from different phases can access the same bank without conflict

### 3.3 Swizzle Patterns for Bank-Conflict-Free Access

Swizzling rearranges the shared memory layout using XOR operations to distribute accesses across banks.

**Core idea:**
```
swizzled_col = row XOR col
```

This creates a "sudoku-like" mapping where each row uses unique banks and each column uses unique banks.

**Implementation:**
```cuda
// Writing to shared memory with swizzle
size_t swizzled_x = (x ^ y) % TILE_SIZE;
smem[y][swizzled_x] = global_data[global_idx];

// Reading back: apply same swizzle to reconstruct
size_t swizzled_x = (x ^ y) % TILE_SIZE;
float val = smem[y][swizzled_x];
```

**Performance impact (RTX 3090, 8192x8192 matrix transpose):**
- Without swizzle (bank conflicts): 1.10 ms
- With swizzle: 0.92 ms (**~20% faster**)

**Flash Attention kernel results:**
- Before swizzle (Kernel 1): 33.28 TFLOPs, 8-way bank conflicts
- After swizzle (Kernel 2): 66.12 TFLOPs (**2x improvement**), no conflicts

**Hardware-accelerated swizzle (Hopper+):**
- `cp.async.bulk` PTX instructions support built-in swizzle modes
- `CU_TENSOR_MAP_SWIZZLE_128B` layout eliminates conflicts for both row-wise and column-wise access
- No manual XOR computation needed

### 3.4 Padding Technique

The simplest bank conflict avoidance: add 1 extra element per row.

```cuda
// Without padding: column stride = 32 elements = stride of 32 banks = conflicts!
__shared__ float smem[32][32];       // Bank conflict on column access

// With padding: column stride = 33 elements = offset by 1 bank each row = no conflicts
__shared__ float smem[32][32 + 1];   // Conflict-free!
```

**Trade-off vs. swizzle:**
- Padding wastes ~3% shared memory (1/33 per row)
- Padding is simpler to implement
- Swizzling preserves alignment for vectorized access
- Swizzling wastes no memory

### 3.5 Async Shared Memory Fill (cp.async on Ampere+)

Introduced in CUDA 11.1 with the Ampere architecture, `cp.async` enables **hardware-accelerated** direct copy from global memory to shared memory, bypassing registers entirely.

**Benefits:**
- **Decouples** data loading from computation (no register intermediary)
- Enables **pipelining**: multiple batches of data can be "in flight" simultaneously
- Frees registers that would otherwise hold intermediate load values
- On Ampere+, if data is aligned to 4+ bytes, `cp.async` instructions are used automatically

**Usage pattern:**
```cuda
// Pipeline stages: load batch N+1 while computing on batch N
cuda::memcpy_async(smem_ptr, global_ptr, num_bytes, barrier);
barrier.arrive_and_wait();  // Wait for copy to complete
// Now compute on smem_ptr data while loading next batch
```

**Hopper TMA (Tensor Memory Accelerator):**
- Can transfer 1D-5D tensor blocks from global to shared memory in a single instruction
- Supports built-in swizzle for bank-conflict-free layouts
- Hardware address generation -- no warp-level address calculation needed

### 3.6 Shared Memory Atomics

- Atomic operations (`atomicAdd`, `atomicCAS`, `atomicExch`, etc.) work on both global and shared memory
- Shared memory atomics are **much faster** than global memory atomics (~10-100x)
- Still imply serialization at the bank level
- Best practice: use shared memory atomics for intra-block reductions, then a single global atomic per block

### 3.7 Distributed Shared Memory (Hopper+)

Hopper introduced **Distributed Shared Memory (DSMEM)**, enabling direct SM-to-SM loads, stores, and atomics across thread blocks in a cluster.
- ~7x faster than routing through global memory for inter-block communication
- Accessible via `cluster.map_shared_rank()` APIs

---

## 4. Memory Bandwidth Optimization Techniques

### 4.1 Hardware Data Compression

NVIDIA GPUs (Ampere+) include **hardware compression/decompression** for data in HBM and L2 cache:
- Lossless compression on data transfers between L2 and HBM
- Can effectively increase memory bandwidth when data is compressible
- Works transparently -- no code changes needed
- Compression ratio depends on data patterns (zeros, repeated values compress well)
- Most effective for sparse activations and zero-padded tensors

### 4.2 Mixed Precision to Reduce Memory Footprint

| Data Type | Bytes | Relative to FP32 | Typical Use |
|---|---|---|---|
| FP32 | 4 | 1.0x | Master weights (training) |
| BF16/FP16 | 2 | 0.5x | Activations, forward/backward pass |
| FP8 (E4M3/E5M2) | 1 | 0.25x | Hopper+ forward pass, KV cache |
| INT8 | 1 | 0.25x | Inference weights (PTQ/QAT) |
| INT4 | 0.5 | 0.125x | Inference weights (GPTQ, AWQ) |

**Memory bandwidth savings:**
- Halving precision (FP32 -> FP16) halves memory traffic, effectively doubling bandwidth
- FP8 on Hopper/Blackwell: 4x less traffic than FP32, plus 2x higher compute throughput

### 4.3 Kernel Fusion

Fusing multiple operations into a single kernel eliminates intermediate memory writes/reads to HBM.

**Example: unfused vs. fused**
```
Unfused:
  Kernel 1: Read A from HBM, compute B = f(A), write B to HBM     [2 HBM accesses]
  Kernel 2: Read B from HBM, compute C = g(B), write C to HBM     [2 HBM accesses]
  Total: 4 HBM accesses

Fused:
  Kernel: Read A from HBM, compute C = g(f(A)), write C to HBM    [2 HBM accesses]
  Total: 2 HBM accesses (2x less memory traffic)
```

**Measured speedups:** Fused BLAS kernels achieve up to **2.61x** speedup over unfused CUBLAS sequences.

**Common fusion opportunities in ML:**
- LayerNorm + Activation + Dropout
- Attention score computation + softmax + value projection (FlashAttention)
- Quantize + GEMM (fused dequant-GEMM)
- Bias + GELU/ReLU (epilogue fusion)

### 4.4 Data Layout: AoS vs SoA

**Array of Structures (AoS):**
```c
struct Particle { float x, y, z, w; };
Particle particles[N];
// Memory: x0 y0 z0 w0 x1 y1 z1 w1 x2 y2 z2 w2 ...
```

**Structure of Arrays (SoA):**
```c
struct Particles { float x[N], y[N], z[N], w[N]; };
// Memory: x0 x1 x2 ... xN y0 y1 y2 ... yN z0 z1 z2 ... zN ...
```

**SoA is almost always better on GPUs** because:
- Consecutive threads access consecutive memory locations = **coalesced**
- AoS with 4 float fields: stride of 16 bytes between same-field accesses = poor coalescing
- Exception: when all fields are always accessed together AND fit in a `float4` vector load

### 4.5 Prefetching Strategies

**Software prefetching:**
```cuda
// Pipeline: prefetch next tile while computing current tile
__shared__ float tile_current[TILE], tile_next[TILE];

// Double-buffering loop
for (int i = 0; i < num_tiles; i++) {
    // Prefetch tile i+1 into tile_next
    cuda::memcpy_async(tile_next, &global[offset_next], size, barrier);

    // Compute on tile_current (already loaded)
    compute(tile_current);

    barrier.arrive_and_wait();
    swap(tile_current, tile_next);
}
```

**L2 prefetching (Ampere+):**
```cuda
cudaMemPrefetchAsync(ptr, size, device, stream);  // Prefetch unified memory
```

**Register-level prefetching:**
- Unroll loops to load data into registers before it's needed
- Compiler does this automatically with `#pragma unroll`

---

## 5. KV Cache Memory Management

### 5.1 Memory Requirements Calculation

**Formula:**
```
KV_cache_bytes = 2 * num_layers * num_kv_heads * head_dim * seq_len * batch_size * bytes_per_element
```

The `2` accounts for both Key and Value tensors.

**Per-token KV cache size (single request):**
```
bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * bytes_per_element
```

### 5.2 KV Cache Size for Popular Models

**LLaMA 3.1 70B (80 layers, 8 KV heads (GQA), head_dim=128, FP16):**
```
bytes_per_token = 2 * 80 * 8 * 128 * 2 = 327,680 bytes = 320 KB/token

At 8K context:    320 KB * 8,192    = 2.56 GB per request
At 128K context:  320 KB * 131,072  = 40.96 GB per request
Batch of 32, 8K:  2.56 GB * 32      = 81.92 GB total KV cache
```

**LLaMA 3.1 8B (32 layers, 8 KV heads (GQA), head_dim=128, FP16):**
```
bytes_per_token = 2 * 32 * 8 * 128 * 2 = 131,072 bytes = 128 KB/token

At 8K context:    128 KB * 8,192   = 1.0 GB per request
At 128K context:  128 KB * 131,072 = 16.0 GB per request
```

**Mixtral 8x7B (32 layers, 8 KV heads (GQA), head_dim=128, FP16):**
```
bytes_per_token = 2 * 32 * 8 * 128 * 2 = 131,072 bytes = 128 KB/token
(Same as LLaMA 8B -- MoE only affects FFN, not attention)

At 8K context:   128 KB * 8,192  = 1.0 GB per request
At 32K context:  128 KB * 32,768 = 4.0 GB per request
```

**Effect of KV cache quantization:**

| KV Precision | Bytes/Element | Memory vs FP16 | Quality Impact |
|---|---|---|---|
| FP16 | 2 | 1.0x (baseline) | None |
| FP8 | 1 | 0.5x | Minimal (native on Hopper+) |
| INT8 | 1 | 0.5x | Low |
| INT4 | 0.5 | 0.25x | Moderate (experimental) |

### 5.3 PagedAttention (vLLM)

**Problem:** Traditional systems allocate contiguous memory for the maximum possible sequence length, wasting 60-80% of KV cache memory on average.

**Solution:** PagedAttention divides KV cache into fixed-size **pages** (blocks), allocated on-demand like OS virtual memory.

**Key details:**
- Default block size: 16 tokens per block
- Each block stores KV tensors for 16 tokens across all layers/heads
- Blocks are stored in a block table (lookup table mapping logical to physical blocks)
- Non-contiguous physical memory, contiguous logical view
- **Memory waste reduction**: 60-80% down to under **4%**
- **Throughput improvement**: 2-4x due to higher batch sizes from reduced waste
- **Copy-on-write**: Enables efficient beam search and parallel sampling

### 5.4 Block-Based vs. Contiguous Allocation

| Property | Contiguous | Block-Based (PagedAttention) |
|---|---|---|
| Allocation | Pre-allocate max seq_len | On-demand, per block |
| Fragmentation | High (variable request sizes) | Near-zero |
| Memory waste | 60-80% typical | <4% |
| Copy overhead | None | Block table indirection |
| Beam search | Duplicate entire KV cache | Copy-on-write (share blocks) |

### 5.5 KV Cache Eviction Strategies

#### H2O (Heavy-Hitter Oracle)
- **Observation**: Accumulated attention scores follow a power-law distribution; a small set of "heavy-hitter" tokens are critical
- **Policy**: Retain top-K "heavy hitter" tokens + recent tokens; evict the rest
- **Scoring**: Sum of attention scores across all previous decoding steps
- **Results**: With only **20%** heavy hitters retained, achieves up to **29x throughput** improvement on OPT-6.7B and OPT-30B
- **Quality**: Maintains >95% of full-cache accuracy on most benchmarks

#### StreamingLLM
- **Observation**: First few tokens ("attention sinks") receive disproportionately high attention regardless of content
- **Policy**: Keep first **4 tokens** (attention sinks) + sliding window of most recent **N** tokens
- **Result**: Enables theoretically **unlimited generation length** with fixed memory budget
- **Limitation**: Cannot recall information from evicted middle context

#### SnapKV
- **Approach**: Decoder-side KV cache compression that selects or merges important past states
- **Method**: Uses attention patterns from a "probe" layer to identify important tokens, then compresses KV cache for all layers
- **Advantage**: More selective than sliding window, preserves important long-range dependencies

#### PagedEviction (2025)
- **Designed for PagedAttention**: Operates at block granularity (not token-level)
- **Scoring**: One score per block rather than per token
- **Benefit**: No need to move tokens between blocks; enables full-block eviction
- **Integration**: Works without modifying CUDA kernels

### 5.6 Automatic Prefix Caching (APC)

vLLM's APC detects shared prefixes across requests and **shares KV cache blocks** automatically:
- System prompt cached once, shared across all requests
- Radix tree structure for efficient prefix matching
- Significant memory savings for chatbot workloads with common system prompts

---

## 6. Activation Memory

### 6.1 Activation Checkpointing (Gradient Checkpointing)

**Problem:** During training, all intermediate activations from the forward pass must be stored for the backward pass. For a model with `n` layers, this requires O(n) memory.

**Solution:** Only store activations at selected "checkpoint" layers. Recompute the discarded activations during the backward pass.

**Memory complexity:** O(sqrt(n)) instead of O(n)

**Trade-off:** ~20-33% increase in computation time (one extra forward pass per segment)

**PyTorch implementation:**
```python
from torch.utils.checkpoint import checkpoint

# Wrap specific layers
output = checkpoint(layer, input, use_reentrant=False)
```

### 6.2 Selective Activation Checkpointing (SAC)

Instead of checkpointing everything or nothing, selectively choose which operations to recompute:

| Strategy | Memory Reduction | Compute Overhead |
|---|---|---|
| No checkpointing | 0% | 0% |
| Recompute pointwise ops only | ~50% | ~5-10% |
| Recompute pointwise + small matmuls | ~65% | ~15-20% |
| Recompute everything (full) | ~80-90% | ~30-33% |
| Recompute attention last | Best trade-off | Minimal overhead |

**Key insight:** Attention matrix materialization is the most expensive to recompute, so checkpoint (save) attention and recompute cheaper pointwise operations.

### 6.3 Memory-Efficient Attention (FlashAttention)

**Problem:** Standard attention materializes the full N x N attention matrix, requiring O(N^2) memory.

**FlashAttention solution:**
- Never materializes the full attention matrix
- Computes attention in **tiles**, keeping only the current tile in SRAM
- Memory: **O(N)** instead of O(N^2)
- Also faster due to reduced HBM access (kernel fusion effect)

**Memory comparison for sequence length 8192:**
```
Standard attention matrix: 8192 * 8192 * 2 bytes (FP16) = 128 MB per head per batch
FlashAttention:            ~O(tile_size) = ~KB per head (tiles in SRAM)
```

### 6.4 In-Place Operations

Certain operations can be done **in-place**, overwriting input tensors to save memory:
```python
# Out-of-place (allocates new tensor):
y = torch.relu(x)         # x still in memory

# In-place (overwrites x):
x.relu_()                 # No new allocation
F.relu(x, inplace=True)   # Same effect
```

**Caveats:**
- Cannot use in-place on tensors that require gradients in the backward pass
- Autograd may raise errors if in-place operation modifies a tensor needed for gradient computation
- Safe for the final operation in a chain (e.g., activation functions after residual adds)

---

## 7. Model Memory Footprint

### 7.1 Weight Memory Calculation

```
Weight memory = num_parameters * bytes_per_parameter
```

| Precision | Bytes/Param | 7B Model | 13B Model | 70B Model |
|---|---|---|---|---|
| FP32 | 4 | 28 GB | 52 GB | 280 GB |
| FP16/BF16 | 2 | 14 GB | 26 GB | 140 GB |
| FP8 | 1 | 7 GB | 13 GB | 70 GB |
| INT8 | 1 | 7 GB | 13 GB | 70 GB |
| INT4 | 0.5 | 3.5 GB | 6.5 GB | 35 GB |

### 7.2 Optimizer State Memory

| Optimizer | States Stored | Bytes/Param | 7B Model | 70B Model |
|---|---|---|---|---|
| SGD (no momentum) | None | 0 | 0 GB | 0 GB |
| SGD (with momentum) | Momentum (FP32) | 4 | 28 GB | 280 GB |
| Adam/AdamW | m + v (FP32 each) | 8 | 56 GB | 560 GB |
| 8-bit Adam (bitsandbytes) | m + v (INT8 each) | 2 | 14 GB | 140 GB |
| Adafactor | Row + Col factors | ~4 | 28 GB | 280 GB |

### 7.3 Gradient Memory

```
Gradient memory = num_parameters * bytes_per_gradient
```

- FP32 training: 4 bytes/param
- Mixed precision: gradients computed in FP16, stored in FP32 for optimizer = 4 bytes/param
- Gradient accumulation does NOT save gradient memory (same buffer reused)

### 7.4 Peak Memory Estimation Formulas

**Standard FP32 Training with AdamW:**
```
Peak memory = 16 * params + activation_memory + CUDA_kernel_overhead
             (4 weights + 4 gradients + 8 optimizer states)

Example 7B: 16 * 7B = 112 GB + activations
```

**Mixed Precision (BF16/FP16) Training with AdamW:**
```
Peak memory = 18 * params + activation_memory + CUDA_kernel_overhead
             (2 FP16 weights + 4 FP32 master weights + 4 FP32 gradients + 8 FP32 optimizer)

Example 7B: 18 * 7B = 126 GB + activations
Note: Actually ~16-18 bytes/param because FP16 weights can be discarded after master copy update
```

**Practical breakdown for LLaMA 7B mixed-precision training with AdamW:**
```
Model weights (BF16):     14 GB
Master weights (FP32):    28 GB  (maintained by optimizer)
Gradients (FP32):         28 GB
Optimizer states (FP32):  56 GB  (Adam m + v)
Activations:              ~2-8 GB (depends on batch size, seq_len, checkpointing)
CUDA overhead:            ~1-2 GB
-----------------------------------------
Total:                    ~58-65 GB (matches reported ~58 GB)
```

**Mixed Precision with 8-bit Adam:**
```
Peak memory = 12 * params + activation_memory
             (2 FP16 + 4 FP32 master + 4 gradients + 2 INT8 optimizer)

Example 7B: 12 * 7B = 84 GB + activations
```

**Inference Only (no optimizer, no gradients):**
```
Peak memory = weight_bytes + KV_cache + activation_buffer + framework_overhead
```

### 7.5 Memory Impact Summary Table

| Method | Speed Impact | Memory Impact |
|---|---|---|
| Gradient Accumulation | None | ~33% reduction (smaller batch per step) |
| Gradient Checkpointing | ~20% slower | ~20-60% activation reduction |
| Mixed Precision (FP16) | ~2x faster | ~50% activation reduction |
| 8-bit Adam | Similar | 75% optimizer state reduction |
| Adafactor | Similar | 50% optimizer state reduction |

---

## 8. CUDA Memory Management

### 8.1 cudaMalloc vs cudaMallocAsync

**cudaMalloc:**
- Synchronous allocation on the GPU
- Implicitly synchronizes the device (blocks until allocation completes)
- Every allocation/free pair incurs driver overhead
- Simple but slow for workloads with many allocations

**cudaMallocAsync (CUDA 11.2+):**
- Stream-ordered, asynchronous allocation
- Uses a **memory pool** under the hood
- Can reuse memory from previous frees in the same pool
- No implicit synchronization -- much better for pipelined workloads
- Default pool can be configured per-device

### 8.2 Memory Pools (cudaMemPool)

```cuda
cudaMemPool_t pool;
cudaDeviceGetDefaultMemPool(&pool, device);

// Configure pool properties
cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);

// All cudaMallocAsync calls use the pool
cudaMallocAsync(&ptr, size, stream);
cudaFreeAsync(ptr, stream);
```

**Benefits:**
- Memory reuse without explicit management
- Reduced allocation overhead (pool pre-allocates from OS)
- Framework integration (PyTorch uses CUDA memory pools internally via `CUDACachingAllocator`)

### 8.3 Unified Memory (cudaMallocManaged)

Provides a single pointer accessible from both CPU and GPU. The CUDA runtime automatically migrates pages between host and device.

**How it works:**
- Pages start on CPU or GPU (depending on first touch)
- On access from the "wrong" side, a page fault triggers migration
- Migration granularity: typically 4 KB or 64 KB pages

**When useful:**
- Prototyping (simplicity)
- Data structures with irregular access patterns
- When data size exceeds GPU memory (automatic oversubscription)
- NOT recommended for performance-critical production code

**Performance optimization:**
```cuda
// Prefetch to avoid page fault overhead
cudaMemPrefetchAsync(ptr, size, device, stream);

// Advise the runtime about access patterns
cudaMemAdvise(ptr, size, cudaMemAdviseSetReadMostly, device);
```

### 8.4 Pinned Memory (cudaMallocHost / cudaHostAlloc)

**Page-locked (pinned) host memory** cannot be swapped to disk by the OS.

**Benefits:**
- **Required** for asynchronous H2D/D2H transfers (`cudaMemcpyAsync`)
- ~2x faster synchronous transfers compared to pageable memory
- Enables overlapping transfers with computation using streams
- Can be mapped into GPU address space (`cudaHostAllocMapped`) for zero-copy access

**Cost:**
- Reduces available system memory (not pageable)
- Excessive pinning can cause OS memory pressure
- Allocation is slower than regular `malloc`

**Usage pattern:**
```cuda
float* h_pinned;
cudaHostAlloc(&h_pinned, size, cudaHostAllocDefault);

// Async copy (overlaps with kernel execution)
cudaMemcpyAsync(d_ptr, h_pinned, size, cudaMemcpyHostToDevice, stream);
kernel<<<grid, block, 0, stream>>>(d_ptr);
```

### 8.5 CUDA IPC for Multi-Process Memory Sharing

Enable multiple processes on the same node to share GPU memory without copying.

**Legacy API:**
```cuda
// Process A: export
cudaIpcMemHandle_t handle;
cudaIpcGetMemHandle(&handle, d_ptr);
// Send handle to Process B via shared memory, pipe, socket, etc.

// Process B: import
void* d_ptr_B;
cudaIpcOpenMemHandle(&d_ptr_B, handle, cudaIpcMemLazyEnablePeerAccess);
```

**Limitation:** Legacy `cudaIpc*` does NOT work with `cudaMallocAsync` or `cuMemCreate` allocations.

### 8.6 CUDA Virtual Memory Management API

CUDA 10.2+ introduced low-level VMM APIs that separate virtual address reservation from physical memory allocation:

**Step-by-step:**
```cuda
// 1. Reserve virtual address range (no physical memory yet)
CUdeviceptr ptr;
cuMemAddressReserve(&ptr, size, alignment, 0, 0);

// 2. Allocate physical memory handle
CUmemGenericAllocationHandle handle;
CUmemAllocationProp prop = { .type = CU_MEM_ALLOCATION_TYPE_PINNED, ... };
cuMemCreate(&handle, chunk_size, &prop, 0);

// 3. Map physical memory to virtual address
cuMemMap(ptr, chunk_size, 0, handle, 0);

// 4. Set access permissions
CUmemAccessDesc access = { .location = { .type = CU_MEM_LOCATION_TYPE_DEVICE, .id = device },
                           .flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE };
cuMemSetAccess(ptr, chunk_size, &access, 1);
```

**Advantages over cudaMalloc:**
- **Growable allocations**: Map additional chunks to the same VA range without copying old data
- **No synchronization**: `cuMemMap` doesn't force device sync (unlike `cudaFree`)
- **Selective peer access**: Map memory to specific GPUs with O(lg N) complexity
- **IPC sharing**: Export/import via OS-native handles (`cuMemExportToShareableHandle`)
- **Ideal for**: Memory pools, dynamic data structures (hash tables, graphs), database join buffers

---

## 9. Multi-GPU Memory Patterns

### 9.1 NVLink: Peer-to-Peer Memory Access

NVLink provides direct high-bandwidth connections between GPUs, enabling peer-to-peer memory access.

| Generation | Bandwidth (per link) | Links per GPU | Total Bidirectional |
|---|---|---|---|
| NVLink 3.0 (A100) | 50 GB/s | 12 | 600 GB/s |
| NVLink 4.0 (H100) | 50 GB/s | 18 | 900 GB/s |
| NVLink 5.0 (B200) | 50 GB/s | 18 | 1,800 GB/s (via NVLink Switch) |

**Peer-to-peer access:**
```cuda
// Enable direct memory access between GPU 0 and GPU 1
cudaSetDevice(0);
cudaDeviceEnablePeerAccess(1, 0);

// GPU 0 can now read/write GPU 1's memory directly
kernel<<<grid, block>>>(gpu1_ptr);  // gpu1_ptr allocated on GPU 1
```

### 9.2 GPUDirect RDMA

Enables direct data transfer between GPU memory and a third-party network adapter (NIC/IB HCA), bypassing CPU memory entirely.

**Data path without GPUDirect RDMA:**
```
GPU Memory -> CPU Memory (PCIe) -> NIC -> Network -> NIC -> CPU Memory (PCIe) -> GPU Memory
```

**Data path with GPUDirect RDMA:**
```
GPU Memory -> NIC -> Network -> NIC -> GPU Memory
```

**Performance:** 84 GB/s RDMA bandwidth at 2 nodes, ~53-54 GB/s at 8 nodes (BF16).

### 9.3 GPUDirect Storage

Enables direct DMA transfers between storage (NVMe, NFS) and GPU memory, bypassing CPU bounce buffers.

**Use cases:**
- Loading large model weights directly to GPU
- Checkpoint loading/saving
- Training on datasets larger than CPU memory

### 9.4 NCCL Collective Operations Memory Usage

| Operation | Memory Overhead | Communication Pattern |
|---|---|---|
| AllReduce | In-place (0 extra) | Ring/tree: O(2 * (N-1)/N * data_size) traffic |
| AllGather | N * chunk_size output | Each GPU sends chunk to all others |
| ReduceScatter | chunk_size output | Reduce + scatter, inverse of AllGather |
| Broadcast | 0 extra | One-to-all |
| All-to-All | N * chunk_size I/O | Full mesh exchange |

### 9.5 AllGather vs ReduceScatter Memory Patterns

These are the core operations for ZeRO-style distributed training:

**AllGather (used in ZeRO-3 forward pass):**
```
Before: Each GPU has 1/N of the parameters
After:  Each GPU has ALL parameters (temporarily)
Memory: Temporarily increases by N-1 chunks during forward pass
Communication: Each GPU sends its chunk to all others
```

**ReduceScatter (used in ZeRO backward pass):**
```
Before: Each GPU has full gradient tensor
After:  Each GPU has 1/N of the reduced gradient
Memory: Reduces by (N-1)/N after the operation
Communication: Reduce (sum) + distribute chunks
```

---

## 10. Memory-Efficient Training Techniques

### 10.1 ZeRO Stages

ZeRO (Zero Redundancy Optimizer) eliminates memory redundancy in data-parallel training.

**Memory breakdown for a model with P parameters using mixed-precision Adam:**
```
Per-GPU memory WITHOUT ZeRO (data parallelism):
  FP16 parameters:     2P bytes
  FP32 master weights: 4P bytes
  FP32 gradients:      4P bytes
  FP32 Adam m state:   4P bytes
  FP32 Adam v state:   4P bytes
  Total:               18P bytes per GPU (all duplicated!)
```

| Stage | What's Partitioned | Memory/GPU (N GPUs) | Communication Overhead |
|---|---|---|---|
| **ZeRO-0** | Nothing (standard DDP) | 18P | Baseline |
| **ZeRO-1** | Optimizer states (m, v) | 2P + 4P + 4P + 8P/N = 10P + 8P/N | Same as DDP |
| **ZeRO-2** | Optimizer states + Gradients | 2P + 4P + (4P + 8P)/N = 6P + 12P/N | Same as DDP |
| **ZeRO-3** | Optimizer + Gradients + Parameters | (2P + 4P + 4P + 8P)/N = 18P/N | 1.5x DDP |

**Concrete example: 7B model, 8 GPUs, mixed-precision Adam:**
```
ZeRO-0: 18 * 7B = 126 GB per GPU
ZeRO-1: 10 * 7B + 8 * 7B / 8 = 77 GB per GPU
ZeRO-2: 6 * 7B + 12 * 7B / 8 = 52.5 GB per GPU
ZeRO-3: 18 * 7B / 8 = 15.75 GB per GPU
```

### 10.2 CPU Offloading (ZeRO-Offload)

Moves optimizer states and optionally gradients to CPU memory:
- Adam optimizer step executed on CPU (sufficient for backward-bound training)
- GPU only needs: parameters + gradients + activations
- Enables training 13B models on a single GPU

**Memory on GPU with ZeRO-Offload:**
```
GPU: 2P (FP16 params) + 4P (FP32 gradients) + activations
CPU: 4P (FP32 master weights) + 8P (Adam m + v)
```

### 10.3 NVMe Offloading (ZeRO-Infinity)

Extends offloading to NVMe SSDs for virtually unlimited model size:
- All model states can be offloaded to NVMe
- NVMe bandwidth: ~3-7 GB/s per drive (PCIe 4.0 x4)
- Enables training trillion-parameter models on limited GPU count
- Higher latency than CPU offloading; requires careful pipelining

### 10.4 Mixed Precision Training Memory Savings

```
Standard FP32:           16P bytes/GPU (params + grads + optimizer)
Mixed Precision (AMP):   18P bytes/GPU (adds FP16 copy, but activations halved)
  -- Activation memory:  ~50% reduction (stored in FP16)
  -- Compute throughput: ~2x (FP16 tensor cores)
  -- Net effect:         Faster training, similar or lower peak memory
```

### 10.5 Gradient Accumulation

Reduces **activation** memory by processing smaller micro-batches:
```
Without accumulation: batch_size = 64, activation memory for 64 samples
With accumulation:    micro_batch = 8, 8 accumulation steps
                      Activation memory for only 8 samples (~8x less)
                      Gradient buffer: same size (accumulated in-place)
                      Training result: mathematically identical
```

---

## 11. Practical Memory Budgets

### 11.1 LLaMA Model Family Inference Requirements

**LLaMA 3.1 7B/8B:**

| Precision | Weight Memory | Min VRAM (weights only) | Recommended VRAM (+ KV cache + overhead) |
|---|---|---|---|
| FP16 | 16 GB | 16 GB | 20-24 GB |
| FP8/INT8 | 8 GB | 8 GB | 10-14 GB |
| INT4 (AWQ/GPTQ) | 4 GB | 4 GB | 6-8 GB |

**LLaMA 3.1 13B:**

| Precision | Weight Memory | Min VRAM | Recommended VRAM |
|---|---|---|---|
| FP16 | 26 GB | 26 GB | 32-40 GB |
| INT8 | 13 GB | 13 GB | 16-20 GB |
| INT4 | 6.5 GB | 6.5 GB | 8-12 GB |

**LLaMA 3.1 70B:**

| Precision | Weight Memory | Min VRAM | Recommended VRAM | GPU Config |
|---|---|---|---|---|
| FP16 | 140 GB | 140 GB | 160-180 GB | 2x H100 80GB or 4x A100 40GB |
| INT8 | 70 GB | 70 GB | 80-100 GB | 1x H100 80GB |
| INT4 (AWQ) | 35 GB | 35 GB | 40-48 GB | 1x A100 80GB or 1x RTX 6000 Ada 48GB |

### 11.2 Mixtral 8x7B (MoE)

Despite only activating ~13B parameters per token, **all 46.7B parameters must reside in memory**:

| Precision | Weight Memory | Min VRAM | Recommended VRAM |
|---|---|---|---|
| FP16/BF16 | ~87 GB | 87 GB | ~100 GB (1x H100 80GB tight) |
| INT8 | ~47 GB | 47 GB | ~55-60 GB |
| INT4 | ~22 GB | 22 GB | ~28-32 GB |

### 11.3 Memory Per Token During Inference

**Components per generated token:**
```
1. KV cache update:  bytes_per_token (see Section 5.2)
2. Activation buffer: ~hidden_dim * batch_size * 2-4 bytes (reused each step)
3. Logits:           vocab_size * batch_size * 4 bytes (temporary)
```

**Incremental memory per new token (KV cache growth):**

| Model | FP16 | FP8 | INT4 |
|---|---|---|---|
| LLaMA 8B | 128 KB | 64 KB | 32 KB |
| LLaMA 70B | 320 KB | 160 KB | 80 KB |
| Mixtral 8x7B | 128 KB | 64 KB | 32 KB |

### 11.4 Maximum Batch Size Estimation

```
max_batch_size = (Total_VRAM - Weight_Memory - Framework_Overhead) / KV_cache_per_request

Example: LLaMA 70B INT4 on A100 80GB
  Available: 80 GB - 35 GB (weights) - 3 GB (overhead) = 42 GB
  KV cache per request (4K context, FP16): 320 KB * 4096 = 1.28 GB
  Max batch size: 42 / 1.28 = ~32 concurrent requests

Example: LLaMA 8B FP16 on RTX 4090 24GB
  Available: 24 GB - 16 GB (weights) - 2 GB (overhead) = 6 GB
  KV cache per request (4K context, FP16): 128 KB * 4096 = 512 MB
  Max batch size: 6 / 0.512 = ~11 concurrent requests
```

### 11.5 Training Memory Budgets

| Model | GPUs Needed (FP16 + Adam) | With ZeRO-3 (8 GPUs) | With ZeRO-3 + Offload |
|---|---|---|---|
| 7B | 2x A100 80GB | ~16 GB/GPU + activations | 1x A100 40GB feasible |
| 13B | 4x A100 80GB | ~30 GB/GPU + activations | 1x A100 80GB feasible |
| 70B | 16x A100 80GB | ~16 GB/GPU (on 128 GPUs) | 8x A100 80GB feasible |

### 11.6 Complete Memory Budget Worksheet

```
=== INFERENCE MEMORY BUDGET ===

Input:
  model_params      = ___B
  precision_bytes   = ___ (FP16=2, INT8=1, INT4=0.5)
  num_layers        = ___
  num_kv_heads      = ___
  head_dim          = ___
  max_seq_len       = ___
  max_batch_size    = ___

Calculations:
  weight_memory     = model_params * precision_bytes
  kv_per_token      = 2 * num_layers * num_kv_heads * head_dim * kv_precision_bytes
  kv_per_request    = kv_per_token * max_seq_len
  total_kv          = kv_per_request * max_batch_size
  activation_buffer = ~0.5-2 GB (varies by model)
  framework_overhead= ~1-3 GB (CUDA context, kernels, buffers)

  TOTAL_VRAM_NEEDED = weight_memory + total_kv + activation_buffer + framework_overhead

=== TRAINING MEMORY BUDGET ===

Input:
  model_params      = ___B
  optimizer          = Adam (8 bytes/param) or SGD (4 bytes/param)
  precision          = mixed (BF16 + FP32 master)
  micro_batch_size   = ___

Calculations:
  fp16_weights      = model_params * 2
  fp32_master       = model_params * 4
  gradients         = model_params * 4
  optimizer_states  = model_params * 8 (Adam) or 4 (SGD)
  activations       = f(micro_batch_size, seq_len, hidden_dim, num_layers)

  TOTAL_BEFORE_ZERO = fp16_weights + fp32_master + gradients + optimizer_states + activations
  WITH_ZERO_3(N)    = (fp16_weights + fp32_master + gradients + optimizer_states) / N + activations
```

---

## Appendix A: GPU Architecture Quick Reference

| Spec | V100 | A100 | H100 SXM | H200 | B200 |
|---|---|---|---|---|---|
| Architecture | Volta | Ampere | Hopper | Hopper | Blackwell |
| Compute Capability | 7.0 | 8.0 | 9.0 | 9.0 | 10.0 |
| SMs | 80 | 108 | 132 | 132 | 148 |
| Register File/SM | 256 KB | 256 KB | 256 KB | 256 KB | 256 KB |
| Max SMEM/SM | 96 KB | 164 KB | 228 KB | 228 KB | 228 KB |
| TMEM/SM | -- | -- | -- | -- | 256 KB |
| L2 Cache | 6 MB | 40 MB | 50 MB | 50 MB | 126 MB |
| HBM Capacity | 32 GB | 80 GB | 80 GB | 141 GB | 192 GB |
| HBM Bandwidth | 900 GB/s | 2.0 TB/s | 3.35 TB/s | 4.8 TB/s | 8.0 TB/s |
| BF16 TFLOPs | 125 | 312 | 990 | 990 | 2,250 |
| FP8 TFLOPs | -- | 624 | 1,979 | 1,979 | 4,500 |
| Max Registers/Thread | 255 | 255 | 255 | 255 | 255 |
| Max Warps/SM | 64 | 64 | 64 | 64 | 64 |
| Max Threads/SM | 2048 | 2048 | 2048 | 2048 | 2048 |
| Max Thread Blocks/SM | 32 | 32 | 32 | 32 | 32 |
| NVLink BW (bidi) | 300 GB/s | 600 GB/s | 900 GB/s | 900 GB/s | 1,800 GB/s |

## Appendix B: Key Nsight Compute Metrics for Memory

```
# Memory throughput
gpu__dram_throughput.avg.pct_of_peak_sustained     # HBM utilization %
lts__throughput.avg.pct_of_peak_sustained           # L2 throughput %
l1tex__throughput.avg.pct_of_peak_sustained         # L1 throughput %

# Coalescing
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum      # Global load sectors
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum     # Global load requests
# Ratio should be 4 for float, 8 for double, 2 for half

# Bank conflicts
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum  # Shared mem load conflicts
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum  # Shared mem store conflicts

# Occupancy
sm__warps_active.avg.pct_of_peak_sustained         # Achieved occupancy %
launch__registers_per_thread                        # Registers per thread
launch__shared_memory_per_block_driver              # Shared memory per block

# Memory access patterns
memory__l1_sectors_per_request                      # L1 sectors per request
memory__l2_sectors_per_request                      # L2 sectors per request
```

---

*This document covers GPU memory optimization from hardware fundamentals through practical deployment calculations. All specifications are sourced from NVIDIA official documentation, architecture whitepapers, and verified benchmarks.*
