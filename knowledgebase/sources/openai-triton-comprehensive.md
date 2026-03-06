---
id: openai-triton-comprehensive
kind: source
title: "OpenAI Triton: Comprehensive GPU Programming Reference"
type: official-doc
category: kernel-lib
summary: "In-depth reference covering the Triton language, programming model, compiler internals, kernel patterns, autotuning, and ecosystem projects (Liger Kernel, Unsloth, ThunderKittens)."
support_level: ""
reliability: official
review_status: reviewed
url: https://triton-lang.org
tags:
  - triton
  - kernel
  - gpu-programming
  - cuda-alternative
  - mlir
  - autotuning
  - llm-inference
  - llm-training
  - flash-attention
  - quantization
  - liger-kernel
  - unsloth
  - thunderkittens
aliases: []
family: ""
market: ""
compute_capability: ""
memory_gb: 0
memory_bandwidth_gbps: 0
preferred_precisions: []
experimental_precisions: []
strengths: []
constraints: []
source_ids: []
workloads: []
operators: []
---

# OpenAI Triton: Comprehensive GPU Programming Reference

## 1. Triton Overview

### What Is Triton?

Triton is an open-source Python-native GPU programming language and compiler developed by OpenAI. It enables researchers and engineers to write highly efficient GPU kernels without requiring deep CUDA expertise. Triton code typically achieves 80-95% of expert CUDA performance with a fraction of the code complexity.

### Design Philosophy

Triton's core philosophy is **block-level programming**. Rather than exposing individual threads (as CUDA does), Triton operates on blocks of data that map naturally to how GPUs execute work. The key principle is: the compiler handles what you do not want to manage, while you retain control over algorithms, data types, and tuning parameters.

What the compiler handles automatically:
- Shared memory management and allocation
- Tensor core utilization and MMA instruction selection
- Memory access coalescing and vectorization
- Register allocation and spilling
- Software pipelining across loop iterations
- Bank conflict avoidance

What the developer controls:
- Algorithm design and data flow
- Block/tile sizes and iteration strategy
- Data types and precision choices
- Autotuning search spaces
- Memory access patterns (which pointers to load/store)

### How Triton Differs from CUDA

| Aspect | CUDA | Triton |
|--------|------|--------|
| Language | C/C++ with extensions | Python DSL |
| Abstraction level | Thread-level | Block-level (tiles) |
| Shared memory | Manual allocation and management | Automatic by compiler |
| Tensor cores | Manual via wmma/mma PTX | Automatic via `tl.dot` |
| Memory coalescing | Developer responsibility | Compiler-optimized |
| Warp synchronization | Manual `__syncwarp()` | Implicit |
| Code size | 100s-1000s of lines | 10s of lines |
| Performance ceiling | Maximum (100%) | 80-95% of CUDA expert |
| Learning curve | Steep | Moderate |
| Hardware support | NVIDIA only (native) | NVIDIA + AMD (ROCm) |

### Key Use Cases

- Custom fused kernels for LLM training and inference
- Memory-bound operations (normalization, activation fusion, embedding)
- Attention kernel variants (Flash Attention, paged attention)
- Quantization/dequantization kernels (GPTQ, AWQ)
- Rapid prototyping before potential CUDA rewrite

---

## 2. Triton Programming Model

### The @triton.jit Decorator

Every Triton kernel is a Python function decorated with `@triton.jit`. This tells the Triton compiler to trace the function and compile it to GPU machine code:

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program instance processes one block
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
```

### Program IDs and Grid Launch

Triton kernels are launched on a grid of **program instances** (analogous to CUDA thread blocks). Each instance is identified by `tl.program_id(axis)`:

```python
# 1D grid launch
grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

# 2D grid launch (e.g., for matrix multiply)
grid = lambda meta: (
    triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),
)
```

The grid can be:
- A tuple of integers: `kernel[(num_programs,)](args...)`
- A lambda that receives meta-parameters: `kernel[grid](args...)`

### Block-Level Computation

Every Triton operation works on **blocks** (1D or 2D tensors). `tl.arange(0, BLOCK_SIZE)` creates a 1D block of indices. 2D blocks are created via broadcasting:

```python
# 1D block of offsets
offs = tl.arange(0, BLOCK_SIZE)  # shape: (BLOCK_SIZE,)

# 2D block via broadcasting
offs_m = tl.arange(0, BLOCK_SIZE_M)[:, None]  # (BLOCK_SIZE_M, 1)
offs_n = tl.arange(0, BLOCK_SIZE_N)[None, :]  # (1, BLOCK_SIZE_N)
# offs_m + offs_n broadcasts to (BLOCK_SIZE_M, BLOCK_SIZE_N)
```

### Masks for Boundary Handling

Masks prevent out-of-bounds memory access at tile boundaries:

```python
offsets = block_start + tl.arange(0, BLOCK_SIZE)
mask = offsets < n_elements
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
tl.store(output_ptr + offsets, result, mask=mask)
```

For 2D operations:
```python
mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
```

### constexpr Parameters

Parameters annotated with `tl.constexpr` are compile-time constants. The compiler specializes the kernel for each unique combination of constexpr values:

```python
@triton.jit
def kernel(
    BLOCK_SIZE: tl.constexpr,    # Must be known at compile time
    ACTIVATION: tl.constexpr,     # Can be used in if-statements
    num_stages: tl.constexpr,     # Controls software pipelining
):
    if ACTIVATION == "relu":      # Compiled away - zero runtime cost
        x = tl.where(x > 0, x, 0)
```

---

## 3. Triton Memory Access Patterns

### tl.load - Loading from Global Memory

`tl.load` moves data from global (HBM) memory to on-chip SRAM of the streaming multiprocessor:

```python
triton.language.load(
    pointer,              # Pointer or block of pointers
    mask=None,            # Boolean mask (tl.int1 block)
    other=None,           # Default value where mask is False
    boundary_check=(),    # Dims for block pointer boundary check
    padding_option='',    # "", "zero", or "nan"
    cache_modifier='',    # ".ca" (all levels), ".cg" (L2+), ".cv" (no cache)
    eviction_policy='',   # Controls NVIDIA PTX eviction behavior
    volatile=False,       # PTX volatile semantics
)
```

**Usage patterns:**

```python
# Scalar pointer + offset block -> loads a block of values
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

# Block pointer (make_block_ptr) -> structured tile load
x = tl.load(block_ptr, boundary_check=(0, 1), padding_option="zero")

# Tensor descriptor load (Hopper+)
x = desc.load([offset_y, offset_x])
```

### tl.store - Writing to Global Memory

```python
triton.language.store(
    pointer,              # Pointer or block of pointers
    value,                # Data to store
    mask=None,            # Boolean mask
    boundary_check=(),    # For block pointers
    cache_modifier='',    # ".wb" (write-back), ".wt" (write-through)
    eviction_policy='',
)
```

### Block Pointers (make_block_ptr)

Block pointers provide a structured way to describe tile access patterns without manual pointer arithmetic:

```python
a_block_ptr = tl.make_block_ptr(
    base=a_ptr,                          # Base pointer
    shape=(M, K),                        # Tensor shape
    strides=(stride_am, stride_ak),      # Element strides
    offsets=(pid_m * BLOCK_M, 0),        # Starting offsets
    block_shape=(BLOCK_M, BLOCK_K),      # Tile shape
    order=(1, 0),                        # Memory layout order
)

# Load a tile
a = tl.load(a_block_ptr, boundary_check=(0, 1), padding_option="zero")

# Advance to next tile
a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
```

Block pointers create a "viewfinder" that tells the program where and how to access a tile from a larger tensor. The compiler uses shape, stride, and order information to generate optimal memory transactions.

### Strided Access Patterns

For non-contiguous access (e.g., accessing columns of a row-major matrix):

```python
# Access pattern for matrix A (M x K), row-major
offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
offs_k = tl.arange(0, BLOCK_SIZE_K)
a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
```

### Cache Hints

Fine-grained control over GPU cache behavior:

```python
# Cache at all levels (default) - good for data reused soon
x = tl.load(ptr + offs, cache_modifier=".ca")

# Cache only at L2 - good for streaming access
x = tl.load(ptr + offs, cache_modifier=".cg")

# Don't cache - good for one-time access
x = tl.load(ptr + offs, cache_modifier=".cv")
```

### Memory Coalescing

The Triton compiler automatically analyzes access patterns and thread mappings to coalesce memory transactions. It combines multiple per-thread accesses into fewer, wider transactions. The compiler determines optimal vectorization (e.g., 128-bit loads) based on data types and access strides.

---

## 4. Triton Compute Operations

### tl.dot - Matrix Multiplication (Tensor Cores)

`tl.dot` is the primary compute-intensive operation that maps to GPU Tensor Core instructions:

```python
# Standard dot product: C += A @ B
# A: (BLOCK_M, BLOCK_K), B: (BLOCK_K, BLOCK_N)
accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for k in range(0, K // BLOCK_K):
    a = tl.load(a_ptrs)
    b = tl.load(b_ptrs)
    accumulator = tl.dot(a, b, accumulator)  # FP32 accumulation
    a_ptrs += BLOCK_K * stride_ak
    b_ptrs += BLOCK_K * stride_bk
```

**Key constraints:**
- Input blocks must have compatible shapes for matmul
- The K dimension must typically be >= 16 for tensor core utilization
- Accumulator is usually float32 for numerical stability

### tl.dot_scaled - Block Scaled Matrix Multiply

For MX format (MXFP4/MXFP8) operations on Blackwell and CDNA4:

```python
# Scaled dot product with per-block scale factors
accumulator = tl.dot_scaled(
    a, scale_a, "e4m3",    # A matrix, scales, format
    b, scale_b, "e2m1",    # B matrix, scales, format
    accumulator             # FP32 accumulator
)
```

Format strings: `"e4m3"` (FP8 E4M3), `"e2m1"` (FP4/MXFP4), `"e5m2"` (FP8 E5M2)

### Reduction Operations

```python
# Sum across an axis
row_sum = tl.sum(x, axis=1)      # (BLOCK_M, BLOCK_N) -> (BLOCK_M,)
total = tl.sum(x, axis=0)        # (BLOCK_M, BLOCK_N) -> (BLOCK_N,)

# Max/Min
row_max = tl.max(x, axis=1)
row_min = tl.min(x, axis=1)

# Argmax/Argmin
idx = tl.argmax(x, axis=1)
idx = tl.argmin(x, axis=0)

# Generic reduce with custom function
result = tl.reduce(x, axis=0, combine_fn=my_reduce_fn)
```

### Scan Operations

```python
# Associative scan (parallel prefix sum)
@triton.jit
def plus_fn(a, b):
    return a + b

cumulative = tl.associative_scan(x, axis=0, combine_fn=plus_fn)

# Built-in cumulative operations
cumsum = tl.cumsum(x, axis=0)
cumprod = tl.cumprod(x, axis=0)
```

`tl.associative_scan` builds a tree structure internally and computes in parallel. The combine function must be associative: `f(f(a, b), c) == f(a, f(b, c))`.

### Element-wise Math Operations

```python
# Exponential and logarithm
y = tl.exp(x)        # e^x
y = tl.exp2(x)       # 2^x
y = tl.log(x)        # ln(x)
y = tl.log2(x)       # log2(x)

# Trigonometric
y = tl.sin(x)
y = tl.cos(x)

# Algebraic
y = tl.sqrt(x)
y = tl.rsqrt(x)      # 1/sqrt(x) - fast reciprocal square root
y = tl.abs(x)
y = tl.sigmoid(x)    # 1/(1+exp(-x))

# Rounding
y = tl.ceil(x)
y = tl.floor(x)

# Fused multiply-add
y = tl.fma(a, b, c)  # a*b + c

# Clamp
y = tl.clamp(x, min_val, max_val)

# Error function
y = tl.erf(x)

# Conditional
y = tl.where(condition, true_val, false_val)
```

### Sort and Top-K

```python
# Sort along an axis
sorted_vals = tl.sort(x, dim=0)

# Top-k values
top_values = tl.topk(x, k, dim=0)

# Gather elements by index
gathered = tl.gather(x, indices, axis=0)

# Histogram
hist = tl.histogram(x, num_bins)
```

### Atomic Operations

```python
tl.atomic_add(ptr, val, mask=mask)
tl.atomic_max(ptr, val)
tl.atomic_min(ptr, val)
tl.atomic_and(ptr, val)
tl.atomic_or(ptr, val)
tl.atomic_xor(ptr, val)
tl.atomic_xchg(ptr, val)
old = tl.atomic_cas(ptr, cmp, val)  # Compare-and-swap
```

### Random Number Generation

```python
# Generate 4 random uint32 values
r0, r1, r2, r3 = tl.randint4x(seed, offsets)

# Uniform random float in [0, 1)
r = tl.rand(seed, offsets)

# Normal distribution
r = tl.randn(seed, offsets)
```

---

## 5. Triton Type System

### Scalar Types

```python
# Integer types
tl.int1    # Boolean
tl.int8, tl.uint8
tl.int16, tl.uint16
tl.int32, tl.uint32
tl.int64, tl.uint64

# Floating-point types
tl.float16    # FP16 (half precision)
tl.bfloat16   # BF16
tl.float32    # FP32 (single precision)
tl.float64    # FP64 (double precision)

# FP8 types
tl.float8e4nv     # NVIDIA E4M3 format
tl.float8e5       # E5M2 format
tl.float8e4b8     # E4M3 B8 format
tl.float8e4b15    # E4M3 B15 format
```

### Type Promotion Rules

Triton follows a hierarchical promotion system:
1. **Kind hierarchy**: `bool < integer < floating-point`
2. **Width-based**: Within same kind, wider types win. `(float32, float16) -> float32`
3. **Float preference**: Equal width different float types prefer `float16`. `(float16, bfloat16) -> float16`
4. **Unsigned preference**: Same width, different signedness -> unsigned

**Scalar-tensor interactions**: Scalars (literals, `tl.constexpr` values) use NumPy rules. Lower/equal-kind scalars do not promote the tensor type. Higher-kind scalars promote to the lowest fitting dtype.

### constexpr

`tl.constexpr` marks compile-time constants. These values are baked into the compiled kernel:

```python
@triton.jit
def kernel(
    N: tl.constexpr,           # Used for loop bounds, mask computation
    BLOCK_SIZE: tl.constexpr,  # Tile sizes - must be power of 2
    ACTIVATION: tl.constexpr,  # String for conditional compilation
):
    # constexpr enables compile-time branching
    if ACTIVATION == "relu":
        x = tl.maximum(x, 0)
    elif ACTIVATION == "gelu":
        x = x * tl.sigmoid(1.702 * x)
```

### Type Casting

```python
# Explicit cast
x_fp16 = x.to(tl.float16)
x_fp32 = x.to(tl.float32)
x_bf16 = tl.cast(x, tl.bfloat16)

# Common mixed-precision pattern: compute in FP32, store in FP16/BF16
accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
# ... compute in FP32 ...
result = accumulator.to(tl.float16)
tl.store(output_ptr + offsets, result, mask=mask)
```

### Pointer Types

Pointers in Triton carry element type information:
```python
# x_ptr is a pointer to float16 elements
# x_ptr + offsets creates a block of pointers
# tl.load(x_ptr + offsets) returns a block of float16 values
```

### Broadcasting Rules

1. Shorter tensor shapes are left-padded with ones
2. Dimensions match if equal or one is 1; dimension-1 expands

```python
# (BLOCK_M,) and (BLOCK_N,) -> need explicit expansion
a = tl.arange(0, BLOCK_M)[:, None]  # (BLOCK_M, 1)
b = tl.arange(0, BLOCK_N)[None, :]  # (1, BLOCK_N)
c = a + b  # (BLOCK_M, BLOCK_N)
```

### Integer Division Semantics

Triton uses C-style rounding (towards zero) for `//` and `%` with tensors, not Python-style (towards negative infinity). This only matters for negative operands.

---

## 6. Triton Autotuning

### @triton.autotune Decorator

The autotune decorator systematically searches over kernel configurations to find the fastest one:

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8},
                      num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
                      num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
                      num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],  # Re-tune when these change
)
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, ...):
    ...
```

### triton.Config

Each `triton.Config` specifies:
- **kwargs**: Dictionary of meta-parameter values (BLOCK_SIZE, GROUP_SIZE, etc.)
- **num_warps**: Number of warps per program instance (affects parallelism)
- **num_stages**: Number of software pipelining stages (affects memory latency hiding)
- **num_ctas**: Number of CTAs in a CGA (cluster) for Hopper+

```python
triton.Config(
    kwargs={'BLOCK_SIZE': 128, 'GROUP_SIZE': 8},
    num_warps=4,        # 4 warps = 128 threads
    num_stages=3,       # 3-stage software pipeline
    num_ctas=1,         # Single CTA (default)
)
```

### The key Parameter

The `key` parameter is a list of kernel argument names. When any key argument changes value, all configs are re-evaluated:

```python
@triton.autotune(
    configs=[...],
    key=['M', 'N', 'K'],  # Different matrix sizes may need different tile configs
)
```

This means the autotuner maintains separate best-configs for each unique `(M, N, K)` combination.

### @triton.heuristics - Rule-Based Tuning

When exhaustive autotuning is too expensive, use heuristics to compute meta-parameters:

```python
@triton.heuristics(
    values={
        'BLOCK_SIZE': lambda args: triton.next_power_of_2(args['n_cols']),
        'num_warps': lambda args: 4 if args['n_cols'] <= 2048 else 8,
    }
)
@triton.jit
def kernel(x_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    ...
```

Heuristics can be stacked with autotune:
```python
@triton.autotune(configs=[...], key=['M', 'N'])
@triton.heuristics(values={'EVEN_K': lambda args: args['K'] % args['BLOCK_K'] == 0})
@triton.jit
def kernel(..., EVEN_K: tl.constexpr):
    if EVEN_K:
        a = tl.load(a_ptrs)  # No mask needed
    else:
        a = tl.load(a_ptrs, mask=mask, other=0.0)
```

### Configuration Pruning

The `prune_configs_by` parameter accepts a performance model that prunes the search space:

```python
@triton.autotune(
    configs=[...],
    key=['M', 'N', 'K'],
    prune_configs_by={
        'early_config_prune': my_prune_function,
        'perf_model': my_performance_model,
        'top_k': 10,  # Keep top 10 configs from perf_model
    },
)
```

### Practical Autotuning Tips

1. **Start broad**: Include configs with varying BLOCK sizes from 32 to 256
2. **Vary num_warps**: Try 2, 4, 8 (2 for small blocks, 8 for large)
3. **Vary num_stages**: Try 2-5 (more stages = more memory latency hiding but more register pressure)
4. **Use key wisely**: Only include arguments that meaningfully affect performance
5. **Cache results**: Use `triton-dejavu` (IBM) to persist tuning results across runs

---

## 7. Common Triton Kernels

### 7.1 Matrix Multiplication (GEMM)

The canonical Triton GEMM kernel with L2 cache optimization via grouped ordering:

```python
@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
        ACTIVATION: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if ACTIVATION == "leaky_relu":
        accumulator = tl.where(accumulator >= 0, accumulator, 0.01 * accumulator)

    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
```

**Wrapper function:**
```python
def matmul(a, b, activation=""):
    assert a.shape[1] == b.shape[0]
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        ACTIVATION=activation,
    )
    return c
```

**Key optimization: Grouped ordering** maps program IDs to (M, N) tiles such that nearby programs access nearby L2 cache lines, dramatically improving cache hit rates for large matrices.

### 7.2 Fused Softmax

```python
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride,
                   n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

        # Numerically stable softmax
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator

        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)
```

**Performance**: Naive PyTorch softmax reads 5MN + 2M elements and writes 3MN + 2M. The fused kernel reads MN and writes MN, yielding ~4x theoretical speedup for bandwidth-bound cases.

### 7.3 Flash Attention in Triton

The Triton Flash Attention implementation uses online softmax with block-level tiling:

```python
@triton.jit
def _attn_fwd(Q, K, V, sm_scale, M, Out,
              stride_qz, stride_qh, stride_qm, stride_qk,
              stride_kz, stride_kh, stride_kn, stride_kk,
              stride_vz, stride_vh, stride_vk, stride_vn,
              stride_oz, stride_oh, stride_om, stride_on,
              Z, H, N_CTX,
              HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr,
              BLOCK_N: tl.constexpr, STAGE: tl.constexpr):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    # Initialize pointers to Q, K, V
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset, shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset, shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0), block_shape=(HEAD_DIM, BLOCK_N), order=(0, 1))
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset, shape=(N_CTX, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0), block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0))

    # Initialize accumulators
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale * 1.44269504  # 1/log(2) for exp2

    q = tl.load(Q_block_ptr)

    # Iterate over K, V blocks
    lo, hi = 0, (start_m + 1) * BLOCK_M if STAGE == 2 else N_CTX
    for start_n in range(lo, hi, BLOCK_N):
        k = tl.load(K_block_ptr)
        # Compute QK^T
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k, qk) * qk_scale

        # Causal mask
        if STAGE == 2:
            offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = start_n + tl.arange(0, BLOCK_N)
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float("-inf"))

        # Online softmax update
        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(qk - m_new[:, None])

        # Rescale accumulator and update
        acc = acc * alpha[:, None]
        v = tl.load(V_block_ptr)
        acc = tl.dot(p.to(v.dtype), v, acc)

        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new

        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    # Final normalization
    acc = acc / l_i[:, None]
    # Store output
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset, shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))
```

**Key features**: Online softmax (never materializes full NxN attention matrix), causal masking via stage parameter, exp2 with prescaled QK for faster computation, FP8 output support on Hopper+.

### 7.4 Layer Normalization

```python
@triton.jit
def _layer_norm_fwd_fused(X, Y, W, B, Mean, Rstd, stride, N, eps,
                          BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride

    # Compute mean
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N

    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)

    # Normalize and apply affine transform
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        tl.store(Y + cols, y, mask=mask)
```

### 7.5 RMS Normalization

```python
@triton.jit
def rmsnorm_kernel(X, Y, W, stride, n_cols, eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    X_row = X + row * stride
    Y_row = Y + row * stride

    # Compute sum of squares
    sum_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
        sum_sq += x * x

    mean_sq = tl.sum(sum_sq) / n_cols
    rstd = 1.0 / tl.sqrt(mean_sq + eps)

    # Normalize and scale
    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)
        y = x * rstd * w
        tl.store(Y_row + cols, y.to(tl.float16), mask=mask)
```

**Performance**: PyTorch RMSNorm achieves ~11% of peak memory bandwidth (168 GB/s on H100). The Triton version achieves ~88% (1365 GB/s), an 8.1x speedup.

### 7.6 Fused Adam Optimizer

The fused Adam pattern combines all elementwise optimizer operations into a single kernel pass:

```python
@triton.jit
def fused_adam_kernel(
    param_ptr, grad_ptr, exp_avg_ptr, exp_avg_sq_ptr,
    lr, beta1, beta2, eps, weight_decay, step,
    n_elements, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load all tensors
    param = tl.load(param_ptr + offsets, mask=mask)
    grad = tl.load(grad_ptr + offsets, mask=mask)
    exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask)
    exp_avg_sq = tl.load(exp_avg_sq_ptr + offsets, mask=mask)

    # Weight decay (decoupled - AdamW)
    param = param * (1.0 - lr * weight_decay)

    # Update biased first moment estimate
    exp_avg = beta1 * exp_avg + (1.0 - beta1) * grad
    # Update biased second moment estimate
    exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * grad * grad

    # Bias correction
    bias_correction1 = 1.0 - tl.exp(step * tl.log(beta1))
    bias_correction2 = 1.0 - tl.exp(step * tl.log(beta2))
    step_size = lr / bias_correction1

    # Parameter update
    denom = tl.sqrt(exp_avg_sq / bias_correction2) + eps
    param = param - step_size * exp_avg / denom

    # Store updated values
    tl.store(param_ptr + offsets, param, mask=mask)
    tl.store(exp_avg_ptr + offsets, exp_avg, mask=mask)
    tl.store(exp_avg_sq_ptr + offsets, exp_avg_sq, mask=mask)
```

**Key benefit**: Reduces 4-6 separate kernel launches to 1, reading each parameter once instead of 4-6 times.

### 7.7 Quantization/Dequantization Kernels (GPTQ)

The GPTQ Triton dequantization kernel fuses dequantization with matrix multiplication:

```python
@triton.jit
def gptq_dequant_matmul_kernel(
    A_ptr, B_ptr, C_ptr, scales_ptr, zeros_ptr,
    M, N, K, group_size,
    stride_am, stride_ak, stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        # Load activation tile (FP16)
        a = tl.load(A_ptr + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak,
                     mask=(offs_m[:, None] < M) & ((k + offs_k[None, :]) < K), other=0.0)

        # Load packed INT4 weights (8 values per int32)
        b_packed = tl.load(B_ptr + (k + offs_k[:, None]) // 8 * stride_bk + offs_n[None, :],
                           mask=((k + offs_k[:, None]) < K) & (offs_n[None, :] < N), other=0)

        # Dequantize: extract 4-bit values, apply scale and zero point
        shift = ((k + offs_k[:, None]) % 8) * 4
        b_int4 = (b_packed >> shift) & 0xF

        group_idx = (k + offs_k[:, None]) // group_size
        scale = tl.load(scales_ptr + group_idx * N + offs_n[None, :])
        zero = tl.load(zeros_ptr + group_idx * N + offs_n[None, :])

        b_dequant = (b_int4.to(tl.float16) - zero) * scale

        accumulator = tl.dot(a, b_dequant, accumulator)

    c = accumulator.to(tl.float16)
    tl.store(C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, c,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

**Key optimization**: Fusing dequantization into the GEMM inner loop avoids materializing the full dequantized weight matrix in memory (W4A16 pattern).

### 7.8 Rotary Position Embedding (RoPE)

```python
@triton.jit
def rope_kernel(
    Q, K, COS, SIN, Out_Q, Out_K,
    seq_len, head_dim, stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (batch, head, seq_pos) combination
    pid = tl.program_id(0)
    batch = pid // (seq_len * num_heads)
    remainder = pid % (seq_len * num_heads)
    head = remainder // seq_len
    seq_pos = remainder % seq_len

    half_dim = head_dim // 2
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < half_dim

    # Load cos and sin for this position
    cos = tl.load(COS + seq_pos * half_dim + offs, mask=mask)
    sin = tl.load(SIN + seq_pos * half_dim + offs, mask=mask)

    # Load Q first half and second half
    q_base = Q + batch * stride_qb + head * stride_qh + seq_pos * stride_qs
    q_first = tl.load(q_base + offs * stride_qd, mask=mask)
    q_second = tl.load(q_base + (offs + half_dim) * stride_qd, mask=mask)

    # Apply rotation (GPT-NeoX style for coalesced access)
    q_out_first = q_first * cos - q_second * sin
    q_out_second = q_second * cos + q_first * sin

    out_q_base = Out_Q + batch * stride_qb + head * stride_qh + seq_pos * stride_qs
    tl.store(out_q_base + offs * stride_qd, q_out_first, mask=mask)
    tl.store(out_q_base + (offs + half_dim) * stride_qd, q_out_second, mask=mask)

    # Same for K (often fused with Q for efficiency)
    k_base = K + batch * stride_kb + head * stride_kh + seq_pos * stride_ks
    k_first = tl.load(k_base + offs * stride_kd, mask=mask)
    k_second = tl.load(k_base + (offs + half_dim) * stride_kd, mask=mask)

    k_out_first = k_first * cos - k_second * sin
    k_out_second = k_second * cos + k_first * sin

    out_k_base = Out_K + batch * stride_kb + head * stride_kh + seq_pos * stride_ks
    tl.store(out_k_base + offs * stride_kd, k_out_first, mask=mask)
    tl.store(out_k_base + (offs + half_dim) * stride_kd, k_out_second, mask=mask)
```

### 7.9 SwiGLU / GeGLU Activation Fusion

```python
@triton.jit
def swiglu_fwd_kernel(
    Gate, Up, Out,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    gate = tl.load(Gate + offsets, mask=mask).to(tl.float32)
    up = tl.load(Up + offsets, mask=mask).to(tl.float32)

    # SwiGLU: SiLU(gate) * up
    # SiLU(x) = x * sigmoid(x)
    gate_sigmoid = tl.sigmoid(gate)
    silu_gate = gate * gate_sigmoid
    out = silu_gate * up

    tl.store(Out + offsets, out.to(tl.float16), mask=mask)

@triton.jit
def swiglu_bwd_kernel(
    Gate, Up, DOut, DGate, DUp,
    n_elements, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    gate = tl.load(Gate + offsets, mask=mask).to(tl.float32)
    up = tl.load(Up + offsets, mask=mask).to(tl.float32)
    dout = tl.load(DOut + offsets, mask=mask).to(tl.float32)

    gate_sigmoid = tl.sigmoid(gate)
    silu_gate = gate * gate_sigmoid

    # d(SiLU)/dx = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    #            = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    dsilu = gate_sigmoid * (1.0 + gate * (1.0 - gate_sigmoid))

    dgate = dout * up * dsilu
    dup = dout * silu_gate

    tl.store(DGate + offsets, dgate.to(tl.float16), mask=mask)
    tl.store(DUp + offsets, dup.to(tl.float16), mask=mask)
```

**Memory savings**: By recomputing activations during backward instead of caching them, SwiGLU/GeGLU fusion achieves ~1.6x peak memory reduction (Liger Kernel approach).

### 7.10 Top-k / Sampling Kernels

The vLLM project provides Triton-based top-k and top-p sampling kernels:

```python
@triton.jit
def topk_kernel(
    logits_ptr, output_ptr, indices_ptr,
    vocab_size, K,
    stride_batch,
    BLOCK_SIZE: tl.constexpr,
):
    batch_id = tl.program_id(0)
    logits_base = logits_ptr + batch_id * stride_batch

    # Outlier search optimization:
    # First pass: compute mean and std
    # Gather logits > mean + 2.15 * std_dev
    # This typically captures the top-k candidates efficiently

    # Pivot-based search: iteratively narrow the search space
    # using pivots to partition logits into candidates
    ...
```

**Performance characteristics**: Works best with larger batch sizes and K < vocab_size * 0.03. For vocab_size ~128k, optimal when K < 4000. Each program processes one row (one batch element).

---

## 8. Triton Compiler Internals

### Compilation Pipeline

The Triton compiler progressively lowers Python DSL code through multiple MLIR-based intermediate representations:

```
Python DSL (@triton.jit)
    |
    v
[AST Walking / Frontend]
    |
    v
Triton IR (TTIR) - Hardware-agnostic tensor operations
    |
    v
[TritonGPU Passes: layout assignment, coalescing, pipelining]
    |
    v
TritonGPU IR (TTGIR) - GPU-specific layouts and scheduling
    |
    v
[TritonNVIDIAGPU / TritonAMDGPU Passes: TMA, async, warp specialization]
    |
    v
LLVM IR - Low-level with GPU intrinsics
    |
    v
[LLVM Optimization Passes]
    |
    v
PTX Assembly (NVIDIA) / AMDGCN (AMD)
    |
    v
[JIT Compilation: ptxas / lld]
    |
    v
CUBIN (NVIDIA) / HSACO (AMD) - GPU binary
```

### Triton IR (TTIR)

The first IR stage captures high-level tensor operations without hardware specifics:

- Uses operations like `tt.load`, `tt.store`, `tt.splat`, `arith.addf`
- Abstract data types and tensor shapes
- Source location metadata for debugging
- No thread/warp organization information

### TritonGPU IR (TTGIR)

Adds GPU-specific execution details:

- **Compute capability**: e.g., `compute-capability = 89` for Ada
- **Thread organization**: `num-warps`, `threads-per-warp`, `num-ctas`
- **Layout encodings**: Describes how tensor data maps to threads

Example layout annotation:
```
#blocked = #triton_gpu.blocked<{
    sizePerThread = [4],
    threadsPerWarp = [32],
    warpsPerCTA = [4],
    order = [0]
}>
```

### MLIR Dialects

The compiler uses three main MLIR dialects:

1. **Triton Dialect**: High-level Triton-specific operations
2. **TritonGPU Dialect**: GPU-level abstractions with layout information
3. **TritonNVIDIAGPU Dialect**: NVIDIA-specific optimizations (TMA, async dot, warp specialization)

### Key Optimization Passes

**General MLIR passes:**
- Common Subexpression Elimination (CSE)
- Dead Code Elimination (DCE)
- Function Inlining

**TritonGPU-specific passes:**
- **Coalesce**: Optimizes memory access patterns for coalesced transactions
- **Pipeline**: Generates multi-stage software pipelines for latency hiding
- **Prefetch**: Inserts prefetch instructions for upcoming data
- **MatmulAccelerate**: Maps `tl.dot` to tensor core MMA instructions
- **RemoveLayout**: Eliminates unnecessary layout conversions

**TritonNVIDIAGPU-specific passes:**
- **TMA Materialization**: Tensor Memory Accelerator support (Hopper+)
- **TMA Multicast**: Multi-GPU memory operations
- **Async Dot**: Asynchronous matrix operations
- **Warp Specialization**: Differentiates thread roles (producer/consumer)

### LLVM IR Stage

The LLVM IR includes:
- NVIDIA-specific annotations: `nvvm.annotations`, `llvm.nvvm.read.ptx.sreg.tid.x()`
- CUDA address spaces: `addrspace(1)` for global memory
- Vectorized memory operations: `ld.global.v4.b32`
- Inline assembly for GPU-specific instructions

### PTX and CUBIN Generation

PTX (Parallel Thread Execution) is NVIDIA's virtual ISA:
```
.version 8.2
.target sm_89
.reg .pred %p<8>
.reg .b32 %r<32>
.reg .f32 %f<16>
```

The PTX is compiled JIT via `ptxas` into CUBIN (ELF format binary). The final SASS (Shader Assembly) includes instruction scheduling information.

### Inspecting Compilation Stages

```python
# Dump all IR stages
kernel = my_kernel.warmup(args..., grid=(1,))

# Access different IR representations
print(kernel.asm['ttir'])     # Triton IR
print(kernel.asm['ttgir'])    # TritonGPU IR
print(kernel.asm['llir'])     # LLVM IR
print(kernel.asm['ptx'])      # PTX assembly
print(kernel.asm['cubin'])    # Binary (bytes)
```

---

## 9. Triton vs CUDA

### When to Use Triton

- **Memory-bound kernels**: Normalization, activations, embedding, softmax
- **Fused operations**: Combining multiple elementwise ops into one pass
- **Rapid prototyping**: Testing kernel ideas before potential CUDA implementation
- **Cross-platform**: Need kernels that work on both NVIDIA and AMD
- **Team capability**: Team familiar with Python but not CUDA
- **Attention variants**: Custom attention patterns, masking strategies

### When to Use CUDA

- **Maximum performance**: Chasing the last 5-20% of peak throughput
- **Fine-grained control**: Custom warp scheduling, register allocation, async copies
- **Complex algorithms**: Algorithms that don't map well to block-level abstraction
- **Kernel libraries**: Building reusable, production kernel libraries
- **p99 latency SLAs**: When every microsecond matters
- **Advanced features**: Custom TMA patterns, warp-specialized megakernels

### Performance Comparison

| Kernel Type | Triton vs cuBLAS/Expert CUDA |
|-------------|------------------------------|
| GEMM (large) | 90-100% |
| GEMM (small/irregular) | 80-95% |
| Fused Softmax | 95-100% (often faster due to fusion) |
| Flash Attention | 90-95% |
| RMSNorm | 85-95% |
| Elementwise fusion | 95-100% |
| Complex multi-pass | 70-85% |

### Limitations of Triton

1. **Optimization ceiling**: Some compiler decisions cannot be overridden
2. **Hardware requirements**: Requires post-Pascal GPUs (compute capability >= 7.0)
3. **Debug tooling**: Less mature profiling/debugging than CUDA ecosystem
4. **Irregular patterns**: Algorithms with irregular control flow or data-dependent access patterns are harder to express
5. **Compilation time**: JIT compilation adds startup overhead
6. **Dynamic shapes**: Less flexible than CUDA for highly dynamic tensor shapes

### Recommended Workflow

1. First optimize with existing libraries (PyTorch, cuBLAS, FlashAttention)
2. Profile to identify bottleneck kernels
3. Write Triton kernels for those bottlenecks
4. If still insufficient, rewrite critical kernels in CUDA/CUTLASS
5. Use Triton for rapid iteration and CUDA for final optimization

---

## 10. Triton for LLM Inference

### Prefill Phase Patterns

During prefill, the model processes the entire input prompt at once. Kernels are launched with `tokens_in_batch x query_heads` instances (2D grid):

- **Compute-bound**: Large matrix multiplications dominate
- **Standard GEMM**: Large M dimension, benefits from standard autotuned configs
- **Attention**: Full Q x K^T computation, standard Flash Attention applies

### Decode Phase Patterns

During decode, only one new token is generated per step. This phase is **memory-bandwidth-bound**:

- **Skinny GEMM**: M=1 (or small batch), effectively matrix-vector multiplication
- **KV Cache access**: Reading cached K, V from previous positions
- **Paged Attention**: Triton kernels for vLLM's paged KV cache

Key Triton patterns for decode:
```python
# Paged attention: each program handles one query head
# Loads K, V from non-contiguous page table entries
pid = tl.program_id(0)  # query head index
# Load block table to find physical KV cache pages
# Iterate over pages, accumulating attention output
```

### Speculative Decoding

Speculative decoding uses a small draft model to predict future tokens, verified by the target model in a single forward pass:

1. Draft model generates N candidate tokens (fast, small model)
2. Target model verifies all N tokens in one forward pass
3. Accept valid tokens, reject and resample from divergence point

Triton kernels support both the draft model's efficient decode and the target model's batched verification step.

### Common LLM Inference Kernels in Triton

| Kernel | Phase | Bound | Fusion Opportunity |
|--------|-------|-------|-------------------|
| Linear projection | Both | Compute | + bias, + activation |
| RMSNorm | Both | Memory | + residual add |
| RoPE | Both | Memory | + Q/K projection |
| Softmax | Both | Memory | Fused in attention |
| KV Cache update | Decode | Memory | + RoPE |
| Paged Attention | Decode | Memory | Standalone |
| SwiGLU | Both | Memory | Gate + Up fusion |
| Top-k sampling | Decode | Memory | + temperature |

---

## 11. Block Scaled MatMul (MX Formats)

### Overview

Block scaled matrix multiplication supports microscaling (MX) formats that provide fine-grained scaling for low-precision computation:

| Format | Description | Bits | Elements per Byte | Scale Group |
|--------|-------------|------|-------------------|-------------|
| MXFP4 | OCP MX FP4 (E2M1) | 4 | 2 | 32 |
| MXFP8 | OCP MX FP8 (E4M3) | 8 | 1 | 32 |
| NVFP4 | NVIDIA FP4 | 4 | 2 | 16 |
| Mixed | FP8 A, FP4 B | Mixed | Mixed | 32 |

### Hardware Requirements

- **NVIDIA**: Compute capability 10.0+ (Blackwell), PTX 8.7+, 5th generation tensor cores
- **AMD**: CDNA4 architecture

### Triton Kernel Implementation

```python
@triton.jit
def block_scaled_matmul_kernel(
    a_desc, a_scale_desc, b_desc, b_scale_desc, c_desc,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ELEM_PER_BYTE_A: tl.constexpr, ELEM_PER_BYTE_B: tl.constexpr,
    VEC_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_K)):
        a = a_desc.load([pid_m * BLOCK_M, k * BLOCK_K // ELEM_PER_BYTE_A])
        b = b_desc.load([pid_n * BLOCK_N, k * BLOCK_K // ELEM_PER_BYTE_B])
        scale_a = a_scale_desc.load([...])  # Block scales
        scale_b = b_scale_desc.load([...])

        # Reshuffle scales into tensor-core-friendly layout
        scale_a = scale_a.reshape(...).trans(...).reshape(BLOCK_M, BLOCK_K // VEC_SIZE)
        scale_b = scale_b.reshape(...).trans(...).reshape(BLOCK_N, BLOCK_K // VEC_SIZE)

        # Block-scaled dot product
        if ELEM_PER_BYTE_A == 2 and ELEM_PER_BYTE_B == 2:
            accumulator = tl.dot_scaled(a, scale_a, "e2m1", b.T, scale_b, "e2m1", accumulator)
        elif ELEM_PER_BYTE_A == 1 and ELEM_PER_BYTE_B == 1:
            accumulator = tl.dot_scaled(a, scale_a, "e4m3", b.T, scale_b, "e4m3", accumulator)
        else:
            accumulator = tl.dot_scaled(a, scale_a, "e4m3", b.T, scale_b, "e2m1", accumulator)

    c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], accumulator.to(output_dtype))
```

### Scale Layout Requirements

Scale factors must be stored in a contiguous memory layout matching the tensor core access pattern. The kernel reshuffles scales from a 5D packed layout `(M//32//4, K//VEC_SIZE//4, 32, 4, 4)` into the 2D layout `(M, K//VEC_SIZE)` expected by the hardware.

### Performance

- MXFP8 achieves performance similar to standard FP8 GEMMs while providing native block scaling
- MXFP4 doubles the hardware-accelerated performance of FP8/MXFP8 GEMMs
- Both formats leverage 5th generation tensor core instructions natively

---

## 12. Triton on AMD (ROCm)

### HIP Backend

Triton supports AMD GPUs through the ROCm/HIP backend. The compiler pipeline targets:

```
Triton IR -> TritonGPU IR -> TritonAMDGPU IR -> LLVM IR -> AMDGCN -> HSACO
```

### ROCm 7.0 Integration

ROCm 7.0 ships with unified Triton 3.3, enabling:
- Cross-vendor kernel portability (same Triton code on NVIDIA and AMD)
- Built-in support for vendor-specific instructions
- AMD-specific optimization passes

### AMD-Specific Optimizations

The AMD backend includes optimization passes not present in the NVIDIA backend:
- **OptimizeLDSUsage**: Optimizes Local Data Share (AMD's equivalent of shared memory)
- **BlockPingpong**: AMD-specific pipelining strategy
- **MFMA instruction selection**: Maps to AMD's Matrix Fused Multiply-Add instructions

### GEMM Configuration for AMD

AMD GPUs require different autotuning configs than NVIDIA:

```python
def get_hip_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 1,
                       'waves_per_eu': 2}, num_warps=4, num_stages=0),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 4,
                       'waves_per_eu': 2}, num_warps=8, num_stages=0),
        # Note: num_stages=0 is common for AMD (different pipeline model)
        # matrix_instr_nonkdim controls MFMA instruction shape
    ]
```

### Performance on AMD

- vLLM's Triton-only backend achieves state-of-the-art performance on both NVIDIA and AMD GPUs with a single codebase
- GEAK-OpenEvolve reports 3.42x average speedup on TritonBench and 7.02x on ROCm benchmarks
- Triton on AMD MI300X achieves competitive performance with hand-written HIP kernels for attention

### CDNA4 Block Scaled MatMul

AMD's CDNA4 architecture supports block-scaled matrix multiplication with unique scale reshuffling patterns:

```python
# MFMA 32x32 instruction
if mfma_nonkdim == 32:
    a_scales = tl.load(a_scale_ptrs).reshape(
        BLOCK_M // 32, BLOCK_K // SCALE_GROUP_SIZE // 8, 2, 32, 4, 1
    ).permute(0, 3, 1, 4, 2, 5).reshape(BLOCK_M, BLOCK_K // SCALE_GROUP_SIZE)

# MFMA 16x16 instruction
elif mfma_nonkdim == 16:
    a_scales = tl.load(a_scale_ptrs).reshape(
        BLOCK_M // 32, BLOCK_K // SCALE_GROUP_SIZE // 8, 4, 16, 2, 2, 1
    ).permute(0, 5, 3, 1, 4, 2, 6).reshape(BLOCK_M, BLOCK_K // SCALE_GROUP_SIZE)
```

---

## 13. Liger Kernel

### Overview

Liger Kernel (LinkedIn GPU Efficient Runtime) is an open-source collection of optimized Triton kernels designed for LLM training. Developed by LinkedIn, it achieves on average 20% increase in training throughput and 60% reduction in GPU memory usage compared to HuggingFace implementations.

### Available Kernels

**Normalization**: RMSNorm, LayerNorm
**Positional**: RoPE (Rotary Position Embedding)
**Activation**: SwiGLU, GeGLU
**Loss**: CrossEntropyLoss, FusedLinearCrossEntropyLoss
**Attention**: MultiTokenAttention, Softmax, Sparsemax
**Advanced**: Embedding, Hyper-Connections (mHC)

**Alignment/Post-Training** (up to 80% memory savings):
FusedLinearCPOLoss, FusedLinearDPOLoss, FusedLinearORPOLoss, FusedLinearSimPOLoss, FusedLinearKTOLoss

**Distillation**: KLDivLoss, JSD, FusedLinearJSD, TVDLoss

### Key Optimization Techniques

1. **Kernel Fusion**: Multiple operations combined into single kernels
   - RMSNorm fuses normalization + scaling: 7x speedup, 3x memory reduction
   - RoPE fuses Q and K rotation: 8x speedup, 3x memory reduction
   - CrossEntropy fuses softmax + log + loss: 3x speedup, 5x memory reduction

2. **Input Chunking** (FusedLinearCrossEntropy):
   - Avoids materializing the full logit tensor (e.g., 16.8 GB for Gemma 256k vocab, batch 8)
   - Processes vocabulary in chunks: `chunk_size = 2^ceil(log2(ceil((B*T)/ceil(V/H))))`

3. **In-Place Replacement**:
   - CE and FLCE replace gradient tensors in-place of input tensors
   - Prevents simultaneous materialization of both large tensors

4. **Recomputation Strategy**:
   - SwiGLU/GeGLU recompute activations during backward instead of caching
   - Trades small compute cost for ~1.6x memory reduction

### Performance Benchmarks (A100 80GB, 4x GPUs)

| Model | Throughput Improvement | Memory Reduction |
|-------|----------------------|------------------|
| LLaMA 3-8B | +42.8% | -54.8% |
| Qwen2 | +25.5% | -56.8% |
| Mistral | +27.0% | -21.0% |

Individual kernel results (hidden_dim=16384):
- RMSNorm: 7x faster, 3x less memory
- RoPE: 8x faster, 3x less memory
- CrossEntropy (vocab=163840): 3x faster, 5x less memory
- SwiGLU/GeGLU: Speed parity, 1.6x less memory

### Supported Models

**Text**: LLaMA 2/3/4, Mistral, Mixtral, Qwen2/2.5/3, Phi3, Gemma 1/2/3, Granite, OLMo2/3, GLM-4, GPT-OSS, Pixtral
**Multimodal**: LLaMA 3.2-Vision, Qwen2-VL, Qwen2.5-VL, PaliGemma, InternVL3

### Integration

```python
# One-line integration
from liger_kernel.transformers import AutoLigerKernelForCausalLM
model = AutoLigerKernelForCausalLM.from_pretrained("meta-llama/Llama-3-8b")

# Model-specific patching
from liger_kernel.transformers import apply_liger_kernel_to_llama
apply_liger_kernel_to_llama()  # Patches before model loading
model = transformers.AutoModelForCausalLM.from_pretrained("...")

# Direct kernel usage
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
loss_fn = LigerFusedLinearCrossEntropyLoss()
loss = loss_fn(model.lm_head.weight, hidden_states, labels)
```

### Dependencies

- `torch >= 2.1.2`
- `triton >= 2.3.0`
- `transformers >= 4.x` (optional, for model patching)

---

## 14. Unsloth

### Overview

Unsloth is an open-source framework for fast LLM fine-tuning and reinforcement learning. It uses custom Triton kernels and mathematical optimizations to deliver up to 5x faster training (typically 3x) with 30-90% less VRAM and no accuracy loss.

### Technical Approach

Unsloth's approach involves:
1. **Manual backpropagation derivation**: Deriving gradient formulas analytically rather than relying on autograd
2. **Triton kernel rewriting**: Rewriting all PyTorch modules into fused Triton kernels
3. **In-place operations**: Eliminating unnecessary memory allocations

### Key Triton Kernels

**RoPE Kernel**:
- Merged two separate kernels into one unified Triton kernel
- Fully in-place operation (no clones or contiguous transpose)
- Variable-length RoPE support for padding-free training
- 2.3x faster on long contexts, 1.9x faster on short contexts

**MLP Kernels (SwiGLU/GeGLU)**:
- Fixed int32 indexing issues for long context training
- Uses `LONG_INDEXING` as a `tl.constexpr` for compiler specialization
- Avoids naive int64 indexing performance penalty

**Cross Entropy Backward**:
- Online softmax approach: `grad = exp(x - logsumexp)` with conditional label logic
- Supports softcapping and logit scaling transformations
- Fused with linear projection for memory efficiency

**MoE (Mixture of Experts) Kernels**:
- ~12x faster MoE training with >35% less VRAM
- ~6x longer context support
- Custom mathematical optimizations for expert routing

### Smart Auto Packing

Unsloth's auto padding-free uncontaminated packing is automatically enabled for all training runs without code changes. Compatible with FlashAttention 3, xFormers, and SDPA backends.

### Performance

- Train Qwen3-4B 3x faster on just 3.9 GB VRAM
- 2x faster fine-tuning with 70% less VRAM (general claim)
- MoE models: 12x faster training

### Supported Models

LLaMA, Qwen, DeepSeek, Gemma, Phi, Mistral, and many others. Supports SFT, DPO, ORPO, and GRPO reinforcement learning methods.

---

## 15. ThunderKittens

### Overview

ThunderKittens (TK) is a lightweight C++ embedded DSL developed by Stanford's Hazy Research for writing high-performance AI kernels on GPUs. Published at ICLR 2025, it takes a fundamentally different approach from Triton by operating at a lower level while maintaining accessibility.

### Design Philosophy

Three core principles:
1. **Tensor core focus**: 94% of H100 compute is tensor cores -- design everything around them
2. **PyTorch-like API**: Familiar syntax for AI practitioners
3. **Hardware transparency**: Expose GPU mechanics rather than hiding them

### Abstraction Hierarchy

**Warp Level (innermost)**:
- 16x16 matrix tiles as fundamental data structures
- Three memory-level tile types:
  - `rt_bf<16, 64>` -- register tile (bfloat16, 16x64)
  - Shared tiles -- in SMEM, templated by type and shape
  - Global layout descriptors -- 4D tensor indexing (batch, head, length, embed)
- Operations: `mma_AB`, `mma_ABt` (tensor core multiply), `multiply`, `exp`, `copy`, `sub_row`, `div_row`, `cumsum`

**Block Level (middle)**:
- Load-Compute-Store-Finish (LCSF) template
- Developer implements 4 functions: `load()`, `compute()`, `store()`, `finish()`
- Coordinates asynchronous execution across warps
- Multi-stage pipelined buffers hide memory latency

**Grid Level (outermost)**:
- Persistent grid scheduling to reduce block launch overhead
- Controlled block ordering for L2 cache optimization

### Code Example - Attention Kernel

```cuda
using namespace kittens;
rt_bf<16, 64> k_reg, v_reg;
load(k_reg, k_smem[subtile]);
zero(att);
mma_ABt(att, q_reg, k_reg, att);      // QK^T via tensor cores
sub_row(att, att, max_vec);             // Subtract max for stability
exp(att, att);                          // Exponentiate
div_row(att, att, norm_vec);            // Normalize
copy(att_mma, att);                     // Copy for MMA layout
load(v_reg, v_smem[subtile]);
auto &v_reg_col = swap_layout_inplace(v_reg);
mma_AB(o_reg, att_mma, v_reg_col, o_reg);  // Multiply by V
```

### Memory Layout Management

TK automatically selects from three strided layouts (32, 64, 128 bytes) to minimize shared memory bank conflicts. Layout conversion between operations (e.g., row-major for compute, column-major for tensor cores) is handled transparently.

### Performance Benchmarks

| Kernel | TK Performance |
|--------|---------------|
| GEMM | Matches cuBLAS |
| Attention Forward | Matches FlashAttention-3 |
| Attention Backward | 10-40% faster than FA3 |
| Linear Attention (Based) | 14x faster than Flash Linear Attention |
| Mamba-2 | 8x faster |
| Long Convolution | 7.9x faster |

### ThunderKittens 2.0 (February 2026)

TK 2.0 adds:
- Boilerplate templates for common patterns
- Custom on-device schedulers
- Full Blackwell GPU support
- FP8 support
- Multi-GPU support
- Megakernel support (combining multiple operations)

### HipKittens (AMD Port)

HipKittens extends ThunderKittens to AMD Instinct GPUs, enabling cross-vendor kernel development.

### Comparison with Triton

| Aspect | ThunderKittens | Triton |
|--------|---------------|--------|
| Language | C++ embedded DSL | Python DSL |
| Abstraction | Tile-level (16x16) | Block-level (arbitrary) |
| Compiler | None (C++ templates) | Full compiler (MLIR) |
| Hardware control | Very fine-grained | Moderate |
| Ease of use | Moderate (C++ knowledge needed) | High (Python-native) |
| Code size | <1MB library | Large compiler |
| Performance ceiling | Higher (full CUDA access) | Slightly lower |
| Deployment | Together AI, Jump Trading, Cursor | Broad ML ecosystem |

---

## Complete API Reference Summary

### triton.language Functions by Category

**Programming Model**: `program_id`, `num_programs`, `tensor`, `tensor_descriptor`

**Creation**: `arange`, `cat`, `full`, `zeros`, `zeros_like`, `cast`

**Shape Manipulation**: `broadcast`, `broadcast_to`, `expand_dims`, `interleave`, `join`, `permute`, `ravel`, `reshape`, `split`, `trans`, `view`

**Linear Algebra**: `dot`, `dot_scaled`

**Memory**: `load`, `store`, `make_tensor_descriptor`, `load_tensor_descriptor`, `store_tensor_descriptor`, `make_block_ptr`, `advance`

**Indexing**: `flip`, `where`, `swizzle2d`

**Math**: `abs`, `cdiv`, `ceil`, `clamp`, `cos`, `div_rn`, `erf`, `exp`, `exp2`, `fdiv`, `floor`, `fma`, `log`, `log2`, `maximum`, `minimum`, `rsqrt`, `sigmoid`, `sin`, `softmax`, `sqrt`, `sqrt_rn`, `umulhi`

**Reduction**: `argmax`, `argmin`, `max`, `min`, `reduce`, `sum`, `xor_sum`

**Scan/Sort**: `associative_scan`, `cumprod`, `cumsum`, `histogram`, `sort`, `topk`, `gather`

**Atomic**: `atomic_add`, `atomic_and`, `atomic_cas`, `atomic_max`, `atomic_min`, `atomic_or`, `atomic_xchg`, `atomic_xor`

**Random**: `randint4x`, `randint`, `rand`, `randn`

**Iterators**: `range`, `static_range`

**Assembly**: `inline_asm_elementwise`

**Compiler Hints**: `assume`, `debug_barrier`, `max_constancy`, `max_contiguous`, `multiple_of`

**Debug**: `static_print`, `static_assert`, `device_print`, `device_assert`

---

## Sources

- [Introducing Triton: Open-source GPU programming for neural networks - OpenAI](https://openai.com/index/triton/)
- [Triton Language Documentation](https://triton-lang.org/main/python-api/triton.language.html)
- [Triton Semantics Documentation](https://triton-lang.org/main/python-api/triton-semantics.html)
- [Triton Matrix Multiplication Tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
- [Triton Fused Softmax Tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
- [Triton Fused Attention Tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
- [Triton Layer Normalization Tutorial](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)
- [Triton Block Scaled MatMul Tutorial](https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html)
- [triton.autotune Documentation](https://triton-lang.org/main/python-api/generated/triton.autotune.html)
- [triton.heuristics Documentation](https://triton-lang.org/main/python-api/generated/triton.heuristics.html)
- [triton.language.load Documentation](https://triton-lang.org/main/python-api/generated/triton.language.load.html)
- [triton.language.store Documentation](https://triton-lang.org/main/python-api/generated/triton.language.store.html)
- [triton.language.associative_scan Documentation](https://triton-lang.org/main/python-api/generated/triton.language.associative_scan.html)
- [Triton GitHub Repository](https://github.com/triton-lang/triton)
- [Deep Dive into Triton Internals (Part 1) - Kapil Sharma](http://www.kapilsharma.dev/posts/deep-dive-into-triton-internals/)
- [Deep Dive into Triton Internals (Part 2) - Kapil Sharma](http://www.kapilsharma.dev/posts/deep-dive-into-triton-internals-2/)
- [Deep Dive into Triton Internals (Part 3) - Kapil Sharma](http://www.kapilsharma.dev/posts/deep-dive-into-triton-internals-3/)
- [Triton Kernel Compilation Stages - PyTorch Blog](https://pytorch.org/blog/triton-kernel-compilation-stages/)
- [Triton Compiler Development Tips - Lei.Chat()](https://www.lei.chat/posts/triton-compiler-development-tips/)
- [A Deep Dive Into AMD Triton Compilation](https://medium.com/@nzhangnju/a-deep-dive-into-amd-triton-compilation-912d96e68e45)
- [Developing Triton Kernels on AMD GPUs - ROCm Blog](https://rocm.blogs.amd.com/artificial-intelligence/triton/README.html)
- [Unlock Peak Performance on AMD GPUs with Triton - ROCm Blog](https://rocm.blogs.amd.com/software-tools-optimization/kernel-development-optimizations-with-triton-on-/README.html)
- [ROCm 7.0 Announcement - AMD](https://www.amd.com/en/blogs/2025/rocm7-supercharging-ai-and-hpc-infrastructure.html)
- [OpenAI Triton on NVIDIA Blackwell - NVIDIA Blog](https://developer.nvidia.com/blog/openai-triton-on-nvidia-blackwell-boosts-ai-performance-and-programmability/)
- [Accelerating Triton Dequantization Kernels for GPTQ - PyTorch Blog](https://pytorch.org/blog/accelerating-triton/)
- [From 11% to 88% Peak Bandwidth: Custom Triton Kernels - Subhadip Mitra](https://subhadipmitra.com/blog/2025/triton-kernels-llm-inference/)
- [Understanding Flash Attention in Triton - Alex Dremov](https://alexdremov.me/understanding-flash-attention-writing-the-algorithm-from-scratch-in-triton/)
- [Triton Flash Attention Kernel Walkthrough - Nathan Chen](https://nathanchen.me/public/Triton-Flash-Attention-Kernel-Walkthrough.html)
- [The Anatomy of a Triton Attention Kernel - arXiv](https://arxiv.org/html/2511.11581v1)
- [Triton-based Top-k and Top-p Sampler - vLLM PR #25824](https://github.com/vllm-project/vllm/pull/25824)
- [GPTQ-Triton Implementation](https://github.com/fpgaminer/GPTQ-triton)
- [Fused SwiGLU Triton Kernels](https://github.com/fattorib/fusedswiglu)
- [Liger Kernel: Efficient Triton Kernels for LLM Training - arXiv](https://arxiv.org/abs/2410.10989)
- [Liger-Kernel GitHub Repository](https://github.com/linkedin/Liger-Kernel)
- [Unsloth GitHub Repository](https://github.com/unslothai/unsloth)
- [3x Faster LLM Training with Unsloth Kernels - Unsloth Docs](https://docs.unsloth.ai/new/3x-faster-training-packing)
- [ThunderKittens: Simple, Fast, and Adorable AI Kernels - arXiv](https://arxiv.org/html/2410.20399v1)
- [ThunderKittens: A Simple Embedded DSL for AI Kernels - Hazy Research](https://hazyresearch.stanford.edu/blog/2024-05-12-quick-tk)
- [ThunderKittens 2.0 - Hazy Research](https://hazyresearch.stanford.edu/blog/2026-02-19-tk-2)
- [ThunderKittens GitHub Repository](https://github.com/HazyResearch/ThunderKittens)
- [HipKittens: Fast and Furious AMD Kernels - Hazy Research](https://hazyresearch.stanford.edu/blog/2025-11-09-hk)
- [GPU MODE Lecture 14: Practitioners Guide to Triton](https://christianjmills.com/posts/cuda-mode-notes/lecture-014/)
- [Programming AI Accelerators with Triton - DigitalOcean](https://www.digitalocean.com/community/tutorials/introduction-to-triton-programming)
- [Block Pointers - Triton Exercises](https://lweitkamp.github.io/triton_exercises/introduction/block_pointers.html)
- [Triton Make Block Ptr Guide - Nathan Chen](https://nathanchen.me/public/Triton-make-block-ptr-Guide.html)
- [Memory Coalescing and Access Optimization - DeepWiki](https://deepwiki.com/triton-lang/triton/4.6-memory-coalescing-and-access-optimization)
- [IBM Triton-Dejavu: Zero Overhead Autotuning](https://github.com/IBM/triton-dejavu)
- [vLLM Triton Backend for Cross-Platform Performance - IBM Research](https://research.ibm.com/publications/vllm-triton-backend-how-to-get-state-of-the-art-performance-on-nvidia-and-amd-with-just-triton)
- [Accelerated Scan (Triton associative_scan)](https://github.com/proger/accelerated-scan)
- [Mamba: The Hard Way - Triton Scan Implementation](https://srush.github.io/annotated-mamba/hard.html)
- [Triton RMSNorm Kernel Implementation](https://github.com/gashon/rms-norm-triton-kernel)
- [Tri-RMSNorm: Efficient RMS Normalization](https://github.com/dtunai/Tri-RMSNorm)
- [GemLite: Fast Low-Bit Matmul Kernels in Triton - Dropbox](https://github.com/dropbox/gemlite)
- [Chasing 6+ TB/s: MXFP8 Quantizer on Blackwell - fal.ai](https://blog.fal.ai/chasing-6-tb-s-an-mxfp8-quantizer-on-blackwell/)
