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
