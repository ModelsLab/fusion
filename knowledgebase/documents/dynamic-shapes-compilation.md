---
id: dynamic_shapes_compilation
kind: document
title: Dynamic Shape Handling and Kernel Compilation
category: compilation
summary: Guide to handling variable sequence lengths, dynamic batching, and shape-dependent compilation in LLM inference, covering padding strategies, bucketing, torch.compile dynamic shapes, and ONNX/TensorRT export.
tags:
  - dynamic-shapes
  - compilation
  - torch-compile
  - tensorrt
  - onnx
  - bucketing
  - padding
source_ids: []
operators:
  - attention
  - matmul
  - general
gpu_families:
  - Ampere
  - Ada
  - Hopper
  - Blackwell
---

# Dynamic Shape Handling and Kernel Compilation

## 1. The Dynamic Shape Problem

LLM inference is inherently shape-variable. Three dimensions change constantly at runtime:

- **Sequence length**: each request has a different prompt length; during decode, the KV-cache grows by one token per step.
- **Batch size**: continuous batching adds and removes requests mid-flight, so the batch dimension is never fixed.
- **Number of tokens (ragged)**: in paged-attention systems the total token count across a batch varies even when the batch size is the same.

These variable shapes create problems for compiled kernels. GPU kernels achieve peak throughput
when memory access patterns and thread-block configurations are known at compile time. When
shapes change, compilers must either:

1. **Recompile** a new kernel specialization (latency spike),
2. **Guard and dispatch** to a pre-compiled variant (lookup overhead), or
3. **Pad inputs** to a fixed shape (wasted compute).

| Dimension       | Typical Range     | Why It Varies                        |
|-----------------|-------------------|--------------------------------------|
| Batch size      | 1 - 512           | Continuous batching, request arrival  |
| Sequence length | 1 - 128k+         | Prompt length, decode position        |
| Num KV heads    | Fixed per model    | Does not change (GQA/MQA)            |
| Head dim        | 64, 128, 256      | Fixed per model                       |
| Total tokens    | 1 - ~500k         | Sum of all seqlens in ragged batch    |

The performance gap between a shape-specialized kernel and a generic one can be 2-5x,
making dynamic shape handling a first-class optimization concern.

## 2. Padding Strategies

### 2.1 Right-Padding (Standard)

The simplest approach: pad all sequences in a batch to the length of the longest sequence.

```python
import torch
import torch.nn.functional as F

def right_pad_batch(sequences: list[torch.Tensor], pad_value: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad sequences to max length in batch, padding on the right."""
    max_len = max(s.shape[-1] for s in sequences)
    padded = torch.stack([
        F.pad(s, (0, max_len - s.shape[-1]), value=pad_value)
        for s in sequences
    ])
    # attention mask: 1 for real tokens, 0 for padding
    mask = torch.stack([
        torch.cat([torch.ones(s.shape[-1]), torch.zeros(max_len - s.shape[-1])])
        for s in sequences
    ])
    return padded, mask
```

### 2.2 Left-Padding (Decoder Inference)

Left-padding is preferred for autoregressive generation because the last token position
is always valid, simplifying KV-cache indexing.

```python
def left_pad_batch(sequences: list[torch.Tensor], pad_value: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad on the left so all sequences end at the same position."""
    max_len = max(s.shape[-1] for s in sequences)
    padded = torch.stack([
        F.pad(s, (max_len - s.shape[-1], 0), value=pad_value)
        for s in sequences
    ])
    mask = torch.stack([
        torch.cat([torch.zeros(max_len - s.shape[-1]), torch.ones(s.shape[-1])])
        for s in sequences
    ])
    return padded, mask
```

### 2.3 Power-of-2 Padding

Pad to the next power of two. This limits the number of distinct shapes the compiler sees
and aligns with GPU warp/wavefront sizes.

```python
import math

def next_power_of_2(n: int) -> int:
    return 1 << math.ceil(math.log2(max(n, 1)))

def pad_to_power_of_2(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    current = tensor.shape[dim]
    target = next_power_of_2(current)
    pad_amount = target - current
    # Build pad tuple (reverse dimension order for F.pad)
    pad = [0] * (2 * (tensor.ndim - 1 - (dim % tensor.ndim))) + [0, pad_amount]
    return F.pad(tensor, pad)
```

### 2.4 Unpadded / Ragged Tensors (FlashInfer / FlashAttention)

Concatenate all tokens into a single 1-D sequence and use a `cu_seqlens` array
to demarcate boundaries. Zero wasted compute.

```python
def pack_ragged(sequences: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack variable-length sequences into a flat tensor with cumulative lengths."""
    packed = torch.cat(sequences, dim=0)  # (total_tokens, hidden)
    cu_seqlens = torch.zeros(len(sequences) + 1, dtype=torch.int32)
    for i, s in enumerate(sequences):
        cu_seqlens[i + 1] = cu_seqlens[i] + s.shape[0]
    return packed, cu_seqlens.cuda()

# Usage with FlashAttention
# flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
```

## 3. Bucketing

Bucketing is a middle ground: instead of padding to an arbitrary max or every power of 2,
you define a set of allowed lengths and pad to the nearest bucket boundary.

### 3.1 Bucket Design

```python
# TensorRT-LLM style buckets: geometrically spaced
def make_buckets(min_len: int = 1, max_len: int = 8192, num_buckets: int = 32) -> list[int]:
    """Generate geometrically-spaced sequence length buckets."""
    import numpy as np
    buckets = np.geomspace(min_len, max_len, num=num_buckets).astype(int)
    buckets = sorted(set(buckets))  # deduplicate
    if max_len not in buckets:
        buckets.append(max_len)
    return buckets

def find_bucket(seq_len: int, buckets: list[int]) -> int:
    """Find the smallest bucket that fits seq_len."""
    for b in buckets:
        if b >= seq_len:
            return b
    return buckets[-1]

# Example buckets
BUCKETS = make_buckets(1, 8192, num_buckets=24)
# [1, 2, 4, 7, 12, 21, 36, 62, 107, 185, 319, 551, 951, 1641, 2833, 4891, 8192]
```

### 3.2 How TensorRT-LLM Uses Buckets

TensorRT-LLM precompiles one TensorRT engine per bucket. At runtime, the scheduler:

1. Sorts incoming requests by sequence length.
2. Groups them into the nearest bucket.
3. Pads all sequences in a group to the bucket boundary.
4. Dispatches to the pre-compiled engine for that bucket.

This avoids runtime recompilation while keeping padding waste under ~15% on average.

| Bucket  | Actual seqlen range | Avg padding waste |
|---------|---------------------|-------------------|
| 128     | 1 - 128             | ~35%              |
| 256     | 129 - 256           | ~18%              |
| 512     | 257 - 512           | ~18%              |
| 1024    | 513 - 1024          | ~18%              |
| 2048    | 1025 - 2048         | ~18%              |
| 4096    | 2049 - 4096         | ~18%              |

More buckets reduce waste but increase compilation time and memory (one engine per bucket).

## 4. torch.compile with Dynamic Shapes

### 4.1 Static vs Dynamic Modes

```python
import torch

model = MyLLM()

# Static: recompiles on every new shape (default)
compiled_static = torch.compile(model)

# Dynamic: generates generic code with symbolic shapes
compiled_dynamic = torch.compile(model, dynamic=True)
```

With `dynamic=False` (default), torch.compile inserts shape guards. Every time an input
arrives with a new shape, the guard fails and triggers recompilation. After a threshold
(default 8 recompiles), it automatically switches to dynamic shapes for that dimension.

### 4.2 mark_dynamic and Dim Constraints

For fine-grained control, mark specific dimensions as dynamic:

```python
from torch.export import Dim

batch = Dim("batch", min=1, max=512)
seq_len = Dim("seq_len", min=1, max=8192)

# Mark dimensions as dynamic on input tensors
x = torch.randn(4, 128, 768)
torch._dynamo.mark_dynamic(x, 0)  # batch dim
torch._dynamo.mark_dynamic(x, 1)  # seq_len dim

# Or use constraints in torch.export
exported = torch.export.export(
    model,
    (x,),
    dynamic_shapes={"x": {0: batch, 1: seq_len}},
)
```

### 4.3 Guard Failures and Recompilation

Monitor guard failures to diagnose performance problems:

```python
import torch._dynamo as dynamo

# Log recompilations
dynamo.config.log_recompilations = True

# Set the recompilation limit before auto-dynamic
dynamo.config.cache_size_limit = 8  # default

# Force all integer shapes to be dynamic from the start
dynamo.config.assume_static_by_default = False
```

Common guard failure patterns:

| Guard Type            | Cause                              | Fix                             |
|-----------------------|------------------------------------|---------------------------------|
| `tensor_shape`        | Input shape changed                | Use `dynamic=True` or buckets   |
| `data_dependent`      | Shape depends on tensor values     | Refactor to avoid data-dep      |
| `nn_module_attr`      | Module attribute changed           | Mark as buffer or use config    |
| `global_variable`     | Global changed between calls       | Avoid mutable globals           |

## 5. CUDA Graphs with Dynamic Shapes

CUDA graphs capture a fixed sequence of kernel launches. Dynamic shapes break this
because kernel launch parameters (grid size, shared memory) change with shape.

### 5.1 Padding Approach

The simplest solution: always pad to max shape, capture one graph.

```python
class CUDAGraphRunner:
    def __init__(self, model, max_batch: int, max_seq: int, hidden: int):
        self.model = model
        self.device = next(model.parameters()).device
        # Pre-allocate max-size buffers
        self.static_input = torch.zeros(max_batch, max_seq, hidden, device=self.device)
        self.static_output = torch.zeros(max_batch, max_seq, hidden, device=self.device)
        # Warm up and capture
        self._capture(max_batch, max_seq, hidden)

    def _capture(self, b, s, h):
        # Warmup
        for _ in range(3):
            self.static_output.copy_(self.model(self.static_input))
        # Capture
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_output = self.model(self.static_input)

    def run(self, x: torch.Tensor) -> torch.Tensor:
        b, s, h = x.shape
        self.static_input[:b, :s, :h].copy_(x)
        self.graph.replay()
        return self.static_output[:b, :s, :h].clone()
```

### 5.2 Graph Pool Switching (Multiple Graphs)

Capture one graph per bucket and switch at runtime:

```python
class BucketedCUDAGraphRunner:
    def __init__(self, model, buckets: list[int], max_batch: int, hidden: int):
        self.graphs = {}
        self.inputs = {}
        self.outputs = {}
        for bucket in buckets:
            inp = torch.zeros(max_batch, bucket, hidden, device="cuda")
            # Warmup
            for _ in range(3):
                model(inp)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = model(inp)
            self.graphs[bucket] = g
            self.inputs[bucket] = inp
            self.outputs[bucket] = out

    def run(self, x: torch.Tensor, bucket: int) -> torch.Tensor:
        b, s, h = x.shape
        self.inputs[bucket][:b, :s, :h].copy_(x)
        self.graphs[bucket].replay()
        return self.outputs[bucket][:b, :s, :h].clone()
```

### 5.3 CUDAGraphTree (PyTorch 2.x)

PyTorch's `torch.compile` with `mode="reduce-overhead"` uses CUDAGraphTree internally.
It records a tree of CUDA graphs indexed by input shapes and reuses graphs when shapes
repeat.

```python
# Automatic CUDA graph management via torch.compile
compiled = torch.compile(model, mode="reduce-overhead")
# First call with shape (4, 128, 768): captures graph
# Second call with same shape: replays graph
# Call with new shape (8, 256, 768): captures new graph in the tree
```

## 6. Triton Kernels with Dynamic Shapes

### 6.1 constexpr BLOCK Sizes and Masking

Triton kernels use compile-time block sizes. The kernel handles shapes that are not
multiples of BLOCK_SIZE through masking.

```python
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(
    x_ptr, y_ptr, out_ptr,
    N,  # dynamic shape: total number of elements
    BLOCK_SIZE: tl.constexpr,  # compile-time constant
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N  # guard against out-of-bounds
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)

def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    N = x.numel()
    BLOCK = 1024
    grid = ((N + BLOCK - 1) // BLOCK,)
    vector_add_kernel[grid](x, y, out, N, BLOCK_SIZE=BLOCK)
    return out
```

### 6.2 Autotuning for Different Shapes

Use `triton.autotune` to select optimal parameters per shape range:

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 64}, num_warps=2),
    ],
    key=["M", "N", "K"],  # re-tune when these change
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Standard tiled matmul with masking for non-multiple shapes
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        mask_a = (offs_m[:, None] < M) & ((k + offs_k[None, :]) < K)
        mask_b = ((k + offs_k[:, None]) < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask_c)
```

The `key` parameter in `@triton.autotune` triggers re-tuning whenever M, N, or K change.
Autotuned configs are cached by shape, so repeated shapes hit the cache.

## 7. TensorRT Dynamic Shape Profiles

TensorRT requires explicit min/opt/max shape specifications per input tensor.

```python
import tensorrt as trt

def build_engine_dynamic(onnx_path: str, shapes: dict) -> trt.ICudaEngine:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        parser.parse(f.read())

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4 GB

    # Define optimization profile
    profile = builder.create_optimization_profile()
    for name, (min_s, opt_s, max_s) in shapes.items():
        profile.set_shape(name, min_s, opt_s, max_s)
    config.add_optimization_profile(profile)

    engine = builder.build_serialized_network(network, config)
    return trt.Runtime(logger).deserialize_cuda_engine(engine)

# Example: LLM decoder layer
shapes = {
    "input_ids":   ((1, 1),    (16, 512),  (64, 4096)),
    "position_ids": ((1, 1),   (16, 512),  (64, 4096)),
    "kv_cache":    ((1, 32, 1, 128), (16, 32, 512, 128), (64, 32, 4096, 128)),
}
engine = build_engine_dynamic("decoder_layer.onnx", shapes)
```

### Multi-Profile Engines

For workloads spanning very different shape regimes (e.g., prefill vs decode),
use multiple profiles:

```python
# Profile 0: decode (batch=large, seq=1)
profile_decode = builder.create_optimization_profile()
profile_decode.set_shape("input_ids", (1, 1), (256, 1), (512, 1))

# Profile 1: prefill (batch=small, seq=large)
profile_prefill = builder.create_optimization_profile()
profile_prefill.set_shape("input_ids", (1, 128), (4, 2048), (8, 8192))

config.add_optimization_profile(profile_decode)
config.add_optimization_profile(profile_prefill)
```

TensorRT optimizes each profile independently: kernel selection, tactic choices, and
memory allocation are all profile-specific.

## 8. ONNX Export with Dynamic Axes

```python
import torch

model = MyLLM().eval()
dummy = torch.randn(1, 128, 768)

torch.onnx.export(
    model,
    (dummy,),
    "model.onnx",
    input_names=["hidden_states"],
    output_names=["logits"],
    dynamic_axes={
        "hidden_states": {0: "batch_size", 1: "seq_len"},
        "logits": {0: "batch_size", 1: "seq_len"},
    },
    opset_version=17,
)
```

### Shape Inference and Validation

After export, run shape inference to propagate dynamic dimensions:

```python
import onnx
from onnx import shape_inference

model = onnx.load("model.onnx")
inferred = shape_inference.infer_shapes(model)
onnx.save(inferred, "model_inferred.onnx")

# Validate with different shapes
import onnxruntime as ort
session = ort.InferenceSession("model_inferred.onnx")
for b, s in [(1, 64), (8, 256), (32, 1024)]:
    x = torch.randn(b, s, 768).numpy()
    out = session.run(None, {"hidden_states": x})
    assert out[0].shape[:2] == (b, s)
```

**Opset considerations**: Dynamic shapes work best with opset >= 13. Opset 17+ adds
better support for dynamic Reshape and Expand operations. Avoid opset < 11 entirely
for dynamic models.

## 9. Ragged / Variable-Length Batching

### 9.1 vLLM PagedAttention

vLLM avoids padding entirely by managing KV-cache in fixed-size pages (blocks).
Each sequence occupies only the pages it needs.

```python
# Conceptual vLLM-style ragged batch construction
class RaggedBatch:
    def __init__(self):
        self.token_ids: torch.Tensor = None      # (total_tokens,)
        self.seq_lens: list[int] = []             # per-sequence lengths
        self.block_tables: torch.Tensor = None    # (num_seqs, max_blocks) page table

    def from_requests(self, requests):
        all_tokens = []
        self.seq_lens = []
        for req in requests:
            all_tokens.extend(req.token_ids)
            self.seq_lens.append(len(req.token_ids))
        self.token_ids = torch.tensor(all_tokens, dtype=torch.long, device="cuda")
        # block_tables map each sequence to physical KV-cache pages
```

### 9.2 SGLang RadixAttention

SGLang extends ragged batching with prefix caching via a radix tree. Shared prefixes
across requests reuse the same KV-cache pages, eliminating redundant computation.

Key data structures:
- `cu_seqlens`: cumulative sequence lengths for FlashAttention varlen API
- `slot_mapping`: maps each token position to a physical KV-cache slot
- `context_lens`: number of cached prefix tokens per sequence (skip recomputation)

### 9.3 FlashInfer Ragged Kernels

FlashInfer provides kernels optimized for ragged tensors with different scheduling:

```python
# FlashInfer batch-ragged prefill
import flashinfer

# Build ragged prefill handler
handler = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace_buffer)
handler.plan(
    qo_indptr=cu_seqlens_q,    # (batch+1,) cumulative query lengths
    kv_indptr=cu_seqlens_kv,   # (batch+1,) cumulative kv lengths
    num_qo_heads=32,
    num_kv_heads=8,
    head_dim=128,
)
output = handler.run(q, k, v)
```

## 10. Performance Comparison

Benchmarks on A100-80GB, Llama-2-7B decode, batch=32, measured in tokens/sec:

| Strategy               | SeqLen=128 | SeqLen=512 | SeqLen=2048 | SeqLen=8192 |
|------------------------|------------|------------|-------------|-------------|
| Padded (max=8192)      | 1,240      | 1,180      | 1,050       | 980         |
| Power-of-2 padding     | 3,850      | 3,420      | 2,100       | 980         |
| Bucketed (24 buckets)  | 4,100      | 3,680      | 2,250       | 1,020       |
| Ragged (FlashAttention)| 4,520      | 4,380      | 3,100       | 1,080       |
| Ragged (FlashInfer)    | 4,650      | 4,500      | 3,250       | 1,100       |

Compilation overhead per strategy:

| Strategy               | Compile/Capture Time | Memory Overhead | Recompilation Risk |
|------------------------|---------------------|-----------------|--------------------|
| Padded (fixed graph)   | ~2s (one graph)     | High (max alloc)| None               |
| Power-of-2 padding     | ~15s (12 graphs)    | Moderate        | None               |
| Bucketed (24)          | ~30s (24 graphs)    | Moderate        | None               |
| torch.compile dynamic  | ~10s (first call)   | Low             | On new shape class |
| Ragged (no graphs)     | 0                   | Minimal         | N/A                |

### Recommendations by Workload

| Workload             | Recommended Strategy                                    |
|----------------------|---------------------------------------------------------|
| Offline batch eval   | Bucketing with torch.compile; shapes are known ahead    |
| Online serving       | Ragged batching (vLLM/SGLang) for maximum throughput    |
| TensorRT deployment  | Multi-profile engine with prefill/decode split           |
| Edge / mobile        | Fixed shapes with padding; predictability matters        |
| Research / prototyping | torch.compile dynamic=True; minimal setup              |

### Key Takeaways

1. **Avoid max-padding whenever possible.** Padding to the global maximum wastes 50-90% of compute for typical request distributions.
2. **Ragged batching delivers the best throughput** but requires kernel support (FlashAttention varlen, FlashInfer, or custom Triton).
3. **Bucketing is the best compromise** when you need CUDA graphs or TensorRT: predictable latency with bounded waste.
4. **torch.compile dynamic=True** is the lowest-effort solution for PyTorch-native workloads but may still suffer from occasional recompilations.
5. **Always separate prefill and decode** -- they have fundamentally different shape profiles and should use different compilation strategies or TensorRT profiles.
