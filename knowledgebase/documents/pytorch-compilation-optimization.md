---
id: pytorch_compilation_optimization
kind: document
title: PyTorch Compilation and Optimization Ecosystem
category: compilation
summary: Deep guide to torch.compile, TorchInductor, CUDA graphs, TorchAO, FlexAttention, custom CUDA extensions, and memory management in PyTorch.
tags:
  - pytorch
  - torch-compile
  - inductor
  - cuda-graphs
  - torchao
  - flex-attention
  - compilation
source_ids:
  - pytorch-torch-compile
  - pytorch-cuda-graphs
  - pytorch-attention-docs
gpu_families:
  - Ampere
  - Ada
  - Hopper
  - Blackwell
---

# PyTorch Compilation and Optimization Ecosystem

## torch.compile Architecture

### The Compilation Pipeline

```
Python code → TorchDynamo (graph capture) → FX Graph → AOTAutograd (backward)
                                                            ↓
Optimized code ← Triton/C++ kernels ← TorchInductor (code generation) ← FX Graph
```

### TorchDynamo (Graph Capture)
- Intercepts Python bytecode at frame level
- Traces tensor operations to build FX graph
- Handles control flow by guarding on conditions
- Falls back to Python interpreter on "graph breaks"

### TorchInductor (Code Generation)
- Takes FX graph, generates optimized kernels
- **Triton kernels** for pointwise, reduction, and some matmul patterns
- **C++ kernels** for CPU operations
- **cuBLAS/cuDNN** calls for GEMM and convolutions
- **Fusion rules**: automatically fuses compatible operations

### Compilation Modes

```python
# Default: balanced compilation, good for most cases
model = torch.compile(model)

# reduce-overhead: uses CUDA graphs to minimize launch overhead
# Best for: decode step (fixed shapes, repeated execution)
model = torch.compile(model, mode="reduce-overhead")

# max-autotune: tries more Triton configs, slower compilation
# Best for: prefill (large shapes, compute-bound)
model = torch.compile(model, mode="max-autotune")

# max-autotune-no-cudagraphs: autotune without CUDA graphs
# Best for: dynamic shapes that can't use CUDA graphs
model = torch.compile(model, mode="max-autotune-no-cudagraphs")
```

### Graph Breaks

**What causes graph breaks** (prevents full fusion):
```python
# 1. Data-dependent control flow
if x.sum() > 0:     # Graph break: condition depends on tensor value
    y = x * 2

# 2. Unsupported operations
print(x)             # Graph break: side effect
x.numpy()            # Graph break: leaves PyTorch domain
torch.autograd.grad() # Complex autograd

# 3. Dynamic shapes (sometimes)
# If shapes change across calls, may cause recompilation

# 4. Custom Python code in forward()
for item in my_python_list:  # Graph break if list length varies
    ...
```

**How to debug graph breaks**:
```python
import torch._dynamo
torch._dynamo.config.verbose = True
# or
TORCH_COMPILE_DEBUG=1 python my_script.py
# or
torch._dynamo.explain(model)(sample_input)
```

**How to avoid graph breaks**:
```python
# Use torch operations instead of Python
# torch.where instead of if/else
# torch.tensor instead of python lists
# mark_dynamic for known-dynamic dimensions
torch._dynamo.mark_dynamic(x, dim=0)  # sequence length varies
```

### Inductor Fusion Rules

**Pointwise fusion**: any combination of element-wise ops
```python
# These all fuse into one Triton kernel:
y = x * 2 + 1
y = torch.sigmoid(y)
y = y * residual
# → single Triton kernel: y = sigmoid(x * 2 + 1) * residual
```

**Reduction fusion**: reduction + pointwise
```python
# Fuses into one kernel:
mean = x.mean(dim=-1, keepdim=True)
y = x - mean  # broadcast subtract
# → fused norm-like kernel
```

**GEMM epilogue fusion**: via cuBLASLt
```python
# Inductor detects and fuses:
y = F.linear(x, weight, bias)  # GEMM + bias
y = F.gelu(y)                   # activation
# → cuBLASLt GEMM with GELU epilogue
```

## CUDA Graphs in PyTorch

### Why CUDA Graphs Matter
```
Without CUDA graphs (eager mode):
  CPU: [launch_k1][launch_k2][launch_k3]...[launch_k200] = 1-2ms overhead
  GPU: [k1][k2][k3]...[k200]
  Per-kernel launch: ~5-10 microseconds

With CUDA graphs:
  CPU: [replay_graph] = ~10 microseconds
  GPU: [k1][k2][k3]...[k200] (same execution)
  Total overhead: ~10 microseconds (100x reduction!)
```

### Manual CUDA Graph Usage
```python
# 1. Create static tensors (sizes won't change)
static_input = torch.randn(batch, seq_len, hidden, device='cuda')
static_output = torch.empty_like(static_input)

# 2. Warmup
for _ in range(3):
    static_output = model(static_input)

# 3. Capture
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_output = model(static_input)

# 4. Replay (in inference loop)
for real_input in inputs:
    static_input.copy_(real_input)  # copy data into captured tensor
    g.replay()
    result = static_output.clone()  # or process in-place
```

### CUDAGraphTree (torch.compile reduce-overhead)
```python
# Automatic CUDA graph management with some dynamism support
# Maintains a tree of captured graphs for different shapes
model = torch.compile(model, mode="reduce-overhead")

# If input shape changes, a new graph is captured
# Previously captured graphs are reused when shapes match
# Trade-off: more memory (multiple captured graphs) for flexibility
```

## TorchAO (Architecture Optimization)

### Quantization APIs
```python
import torchao

# INT8 weight-only quantization:
torchao.quantize_(model, torchao.quantization.int8_weight_only())

# INT4 weight-only quantization (GPTQ-like):
torchao.quantize_(model, torchao.quantization.int4_weight_only(group_size=128))

# INT8 dynamic quantization (weights + activations):
torchao.quantize_(model, torchao.quantization.int8_dynamic_activation_int8_weight())

# FP8 quantization:
torchao.quantize_(model, torchao.quantization.float8_weight_only())

# Autoquantization: pick best quant per layer
torchao.quantize_(model, torchao.quantization.autoquant())
```

### Semi-Structured Sparsity (2:4)
```python
from torchao.sparsity import semi_structured_sparsify

# Apply 2:4 sparsity to a model
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        semi_structured_sparsify(module.weight)
# Uses cuSPARSELt for 2x speedup on Ampere+
```

### Float8 Training
```python
from torchao.float8 import convert_to_float8_training

# Convert model for FP8 training
convert_to_float8_training(model, module_filter_fn=lambda mod, fqn: isinstance(mod, nn.Linear))

# Training loop is standard - FP8 happens automatically
for batch in dataloader:
    output = model(batch)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

## FlexAttention

### What It Is
PyTorch's composable attention API that compiles custom attention patterns to efficient fused kernels.

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

# Define custom score modification
def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

# Create block mask (precomputed for efficiency)
block_mask = create_block_mask(causal_mask, B=batch, H=heads, Q_LEN=seq, KV_LEN=seq)

# Run attention with custom masking
output = flex_attention(query, key, value, block_mask=block_mask)
```

### Custom Score Modifications
```python
# Sliding window attention:
def sliding_window(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx).abs() <= WINDOW_SIZE

# Prefix-LM (bidirectional on prefix, causal on rest):
def prefix_lm(b, h, q_idx, kv_idx):
    return (kv_idx <= PREFIX_LEN) | (q_idx >= kv_idx)

# Document masking (different docs in batch):
def document_mask(b, h, q_idx, kv_idx):
    return document_id[q_idx] == document_id[kv_idx]

# Relative position bias:
def alibi_bias(b, h, q_idx, kv_idx):
    return -h * (q_idx - kv_idx).abs()  # ALiBi-style

# Combine multiple:
def custom_attention(b, h, q_idx, kv_idx):
    causal = q_idx >= kv_idx
    window = (q_idx - kv_idx) <= WINDOW_SIZE
    return causal & window
```

### Performance
- Compiled to fused Triton kernels (no separate mask materialization)
- Block-sparse computation: skips blocks that are entirely masked
- Competitive with FlashAttention for standard causal masking
- Significantly faster than naive masking for sparse patterns

## Custom CUDA Extensions in PyTorch

### JIT Compilation
```python
from torch.utils.cpp_extension import load

# Compile CUDA extension at runtime:
my_extension = load(
    name='my_cuda_ops',
    sources=['my_ops.cpp', 'my_ops_kernel.cu'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
)

# Use:
output = my_extension.my_op(input)
```

### Ahead-of-Time Compilation
```python
# setup.py:
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='my_cuda_ops',
    ext_modules=[
        CUDAExtension('my_cuda_ops', [
            'my_ops.cpp',
            'my_ops_kernel.cu',
        ]),
    ],
    cmdclass={'build_ext': BuildExtension}
)
```

### Custom Autograd Function
```python
class MyCustomOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        output = my_cuda_extension.forward(input, weight)
        ctx.save_for_backward(input, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input, grad_weight = my_cuda_extension.backward(grad_output, input, weight)
        return grad_input, grad_weight

# Make it work with torch.compile:
my_op = torch.library.custom_op("mylib::my_op", mutates_args=())
@my_op.register_fake
def my_op_fake(input, weight):
    return torch.empty_like(input)  # just the shape/dtype info
```

## PyTorch Memory Management

### CUDA Caching Allocator
```python
# Check memory usage:
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

# Memory snapshot for debugging:
torch.cuda.memory._record_memory_history()
# ... run code ...
snapshot = torch.cuda.memory._snapshot()
# Analyze snapshot for allocation patterns
```

### Key Environment Variables
```bash
# Expandable segments: reduce fragmentation
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Max split size: prevent small block fragmentation
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Garbage collection threshold
PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8

# Combined:
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

# Debug: log allocations
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
PYTORCH_NO_CUDA_MEMORY_CACHING=1  # Disable caching (debug only!)
```

## Distributed Inference in PyTorch

### Tensor Parallelism with DTensor
```python
from torch.distributed.tensor import DTensor, Shard, Replicate
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel

# Parallelize a transformer layer:
parallelize_module(
    model.layer,
    device_mesh,
    {
        "attention.q_proj": ColwiseParallel(),
        "attention.k_proj": ColwiseParallel(),
        "attention.v_proj": ColwiseParallel(),
        "attention.o_proj": RowwiseParallel(),
        "mlp.gate_proj": ColwiseParallel(),
        "mlp.up_proj": ColwiseParallel(),
        "mlp.down_proj": RowwiseParallel(),
    }
)
```

## Practical torch.compile Recipes

### Recipe 1: Compile a Model for Inference (3 Lines)

```python
model = AutoModelForCausalLM.from_pretrained("model_name").cuda()
model = torch.compile(model, mode="reduce-overhead")
output = model(input_ids)  # first call compiles, subsequent calls are fast
```

**Expected speedup**: 1.3x-2.5x on decoder-only LLMs during generation, with the largest gains on small batch sizes where kernel launch overhead dominates. Prefill (long input) benefits less than decode (token-by-token).

**Warmup time**: The first call triggers compilation, which takes 30 seconds to 5+ minutes depending on model size and mode. Subsequent calls with the same input shapes reuse the compiled graph. Use `TORCHINDUCTOR_FX_GRAPH_CACHE=1` to persist compiled artifacts across runs.

**When it helps**: Repeated inference with fixed or few distinct shapes (serving, benchmarks, batch processing). Models with many small ops that can be fused (attention, layer norms, activations).

**When it hurts**: One-shot scripts where compilation time exceeds total inference time. Highly dynamic models with variable control flow. Models that are already bottlenecked on a single large GEMM (compilation adds overhead with no fusion benefit).

### Recipe 2: Debug Compilation Failures

```bash
TORCH_LOGS="dynamo" python script.py        # see what Dynamo traces and captures
TORCH_LOGS="graph_breaks" python script.py   # see only graph break locations
torch._dynamo.explain(model)(input)          # explain all breaks with reasons
```

**The 5 most common graph break causes and their fixes:**

1. **`print()` or logging inside `forward()`** — Remove print statements or guard them with `if not torch.compiler.is_compiling():`. Replace with `torch._dynamo.comptime(print_fn)` if debug output is needed during tracing.

2. **Calling `.item()`, `.tolist()`, or `.numpy()` on tensors** — These force a sync and exit the tensor domain. Replace `.item()` comparisons with `torch.where()`. If you need the value for control flow, restructure to avoid data-dependent branching.

3. **Data-dependent control flow (`if tensor.sum() > 0`)** — Use `torch.where(condition, true_branch, false_branch)` or `torch.cond` for true conditional execution. If the condition is shape-dependent (not value-dependent), use `torch._dynamo.mark_dynamic`.

4. **Unsupported third-party library calls in forward** — Move non-PyTorch calls outside the compiled region. Use `torch.compiler.allow_in_graph` for functions you know are safe, or wrap non-compilable code in a separate non-compiled function.

5. **Python built-ins on tensors (e.g., `list(tensor)`, `dict` with tensor keys)** — Use `torch.unbind()` instead of `list()`, keep containers as tuples of tensors, and avoid using tensors as dictionary keys.

### Recipe 3: torch.compile for LLM Serving

**How vLLM uses torch.compile internally:**
vLLM (v0.6+) uses `torch.compile` as its default compilation backend. It compiles the model's forward pass and leverages Piecewise CUDA Graphs — the model graph is split at operations that cannot be captured in CUDA graphs (like attention with paged KV cache), and each segment is independently captured. This gives most of the CUDA graph benefit without requiring the entire forward pass to be graph-capturable.

**When to use each mode:**
- `mode="reduce-overhead"` — Best for **decode** (autoregressive generation). Fixed small batch, repeated identical shapes, kernel-launch-bound. The CUDA graph replay eliminates per-kernel launch overhead.
- `mode="max-autotune"` — Best for **prefill** (processing the prompt). Larger matrices, compute-bound. The extra Triton autotuning finds faster tile sizes and configs for the bigger GEMMs. Compilation is slower but pays off for sustained throughput workloads.
- For serving, consider compiling prefill and decode with **different modes** by using separate compiled versions of the model or by relying on vLLM's built-in mode selection.

**CUDA graphs integration with torch.compile:**
```python
# reduce-overhead mode automatically uses CUDAGraphTree internally.
# For manual control, you can combine torch.compile with explicit CUDA graphs:

compiled_model = torch.compile(model, mode="max-autotune-no-cudagraphs")

# Then manually capture CUDA graphs over the compiled model:
static_input = torch.randint(0, vocab_size, (batch, seq_len), device="cuda")
with torch.cuda.graph(cuda_graph):
    static_output = compiled_model(static_input)

# This is useful when you need precise control over graph boundaries
# (e.g., paged attention, variable cache sizes).
```

### Recipe 4: TorchAO Quantization (Post-Training)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchao.quantization import quantize_, int4_weight_only

# Load model in full precision
model_name = "meta-llama/Llama-3.1-8B"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Apply INT4 weight-only quantization (in-place, no calibration data needed)
quantize_(model, int4_weight_only(group_size=128))

# Compile for maximum performance
model = torch.compile(model, mode="max-autotune")

# Benchmark
input_ids = tokenizer("The future of AI is", return_tensors="pt").input_ids.cuda()

# Warmup (triggers compilation)
for _ in range(3):
    _ = model(input_ids)

# Timed run
import time
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(100):
    _ = model(input_ids)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start
print(f"Average latency: {elapsed / 100 * 1000:.2f} ms")
```

**Expected results (Llama-3.1-8B on A100-80GB):**
- BF16 baseline: ~35ms per forward pass
- INT4 weight-only + torch.compile: ~14ms per forward pass (~2.5x speedup)
- Memory: ~4.5 GB (down from ~16 GB for BF16)

**Key considerations:**
- `int4_weight_only` uses asymmetric quantization with group_size=128 by default, which preserves quality well for most LLMs.
- `quantize_` modifies the model in-place — the Linear layers are replaced with quantized versions that dequantize on-the-fly during the matmul.
- `torch.compile` with `max-autotune` is critical: it fuses the dequantize + matmul into efficient Triton kernels. Without compilation, INT4 can be *slower* than BF16 due to dequantization overhead.

### Recipe 5: FlexAttention Custom Mask (Sliding Window + Causal)

```python
import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

# Configuration
B, H, SEQ_LEN, D = 2, 32, 4096, 128
WINDOW_SIZE = 1024

# Define combined mask: causal AND sliding window
def sliding_window_causal(b, h, q_idx, kv_idx):
    causal = q_idx >= kv_idx
    window = (q_idx - kv_idx) <= WINDOW_SIZE
    return causal & window

# Create the block mask (precomputed sparse structure)
# BLOCK_SIZE controls granularity — 128 is a good default
block_mask = create_block_mask(
    sliding_window_causal,
    B=B, H=H, Q_LEN=SEQ_LEN, KV_LEN=SEQ_LEN,
    _compile=True,  # compile the mask creation itself for speed
)

# Create Q, K, V tensors
query = torch.randn(B, H, SEQ_LEN, D, device="cuda", dtype=torch.bfloat16)
key = torch.randn(B, H, SEQ_LEN, D, device="cuda", dtype=torch.bfloat16)
value = torch.randn(B, H, SEQ_LEN, D, device="cuda", dtype=torch.bfloat16)

# Run flex_attention (must be inside torch.compile for best performance)
compiled_flex = torch.compile(flex_attention)
output = compiled_flex(query, key, value, block_mask=block_mask)
```

**How BlockMask works:**
- The sequence is divided into blocks (default 128 tokens each).
- `create_block_mask` evaluates the mask function at block granularity to determine which blocks have any non-masked entries.
- Blocks that are entirely masked are **skipped** during computation — this is where the speedup comes from for sparse patterns like sliding window.
- The block mask is a precomputed data structure; it does not materialize a full (SEQ_LEN x SEQ_LEN) mask tensor.

**Performance comparison vs standard SDPA (seq_len=4096, window=1024, A100):**
- `F.scaled_dot_product_attention` with full causal mask: ~4.2 ms (computes full causal, wastes work outside window)
- `F.scaled_dot_product_attention` with materialized combined mask: ~5.1 ms (mask materialization adds memory + time)
- `flex_attention` with block mask: ~1.8 ms (~2.3x faster — skips blocks outside window)
- The speedup scales with sparsity: a smaller window relative to sequence length yields greater speedup.

### Troubleshooting torch.compile

| Symptom | Cause | Fix |
|---------|-------|-----|
| First inference very slow (minutes) | Compilation overhead | Use `mode="reduce-overhead"`, cache with `TORCHINDUCTOR_FX_GRAPH_CACHE=1` |
| Recompiles every call | Dynamic shapes trigger new compilations | Use `torch._dynamo.mark_dynamic(tensor, dim)` or `torch.compile(dynamic=True)` |
| Slower than eager mode | Graph breaks prevent fusion | Check `TORCH_LOGS="graph_breaks"`, eliminate breaks |
| OOM during compilation | Inductor autotuning allocates memory for many kernel variants | Reduce `max-autotune` configs: `torch._inductor.config.max_autotune_gemm_backends = "TRITON"` |
| `torch._dynamo.exc.Unsupported` error | Operation not supported by Dynamo | Wrap unsupported code in `torch._dynamo.disable()` decorated function |
| "CUDAGraphs skipped" warning | Operations incompatible with CUDA graph capture | Check for CPU-GPU syncs, in-place ops on non-static tensors, or allocations inside the graph region |
| Accuracy differs from eager | Floating-point reordering in fused kernels | Set `torch._inductor.config.fallback_random=True`; for strict numerics use `torch.compile(fullgraph=False)` and isolate the divergent op |
| `BackendCompilerFailed` with Triton error | Triton code generation bug or unsupported pattern | Update PyTorch/Triton to latest nightly; report bug with `TORCH_COMPILE_DEBUG=1` output |
| Compilation succeeds but no speedup | Model is already memory-bandwidth-bound on a single large op | Profile with `torch.profiler` — if one GEMM dominates, compile cannot help; try quantization instead |
| "shape mismatch" during CUDA graph replay | Input shape changed after graph capture | Use `dynamic=True` or bucket inputs into fixed shapes; avoid `reduce-overhead` for truly dynamic workloads |
| Excessive memory usage at runtime | CUDA graphs allocate memory pools per captured graph | Limit distinct shape buckets; call `torch._dynamo.reset()` to free old compilations |
| `torch.compile` hangs indefinitely | Infinite loop in Dynamo tracing due to complex control flow | Simplify `forward()`, extract complex logic into `@torch._dynamo.disable` functions, set `torch._dynamo.config.cache_size_limit` lower |
