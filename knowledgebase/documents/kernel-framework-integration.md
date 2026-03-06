---
id: kernel_framework_integration
kind: document
title: Kernel to Framework Integration Patterns
category: engineering
summary: How to integrate custom CUDA/Triton kernels into PyTorch, vLLM, SGLang, and llama.cpp, including autograd integration, torch.compile compatibility, and serving framework extension.
tags:
  - integration
  - pytorch
  - vllm
  - sglang
  - custom-ops
  - torch-library
  - autograd
source_ids: []
operators:
  - general
gpu_families:
  - Ampere
  - Ada
  - Hopper
  - Blackwell
---

# Kernel to Framework Integration Patterns

## PyTorch Custom Op Integration

### Method 1: torch.autograd.Function
```python
import torch
import triton
import triton.language as tl

@triton.jit
def fused_rmsnorm_kernel(X, W, OUT, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(W + cols, mask=mask)
    out = x * rstd * w
    tl.store(OUT + row * stride + cols, out, mask=mask)

class FusedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps=1e-6):
        M, N = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        fused_rmsnorm_kernel[(M,)](x, weight, out, x.stride(0), N, eps, BLOCK)
        ctx.save_for_backward(x, weight, out)
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, out = ctx.saved_tensors
        # Compute gradients (simplified)
        grad_x = grad_weight = None
        if ctx.needs_input_grad[0]:
            grad_x = compute_rmsnorm_grad(grad_output, x, weight, ctx.eps)
        if ctx.needs_input_grad[1]:
            grad_weight = (grad_output * (x / rms(x))).sum(0)
        return grad_x, grad_weight, None

# Usage: drop-in replacement
def rmsnorm(x, weight, eps=1e-6):
    return FusedRMSNorm.apply(x, weight, eps)
```

### Method 2: torch.library (Modern, torch.compile compatible)
```python
import torch
from torch.library import Library, impl

# Define custom op
lib = Library("myops", "DEF")
lib.define("fused_rmsnorm(Tensor x, Tensor weight, float eps=1e-6) -> Tensor")

# CPU fallback implementation
@impl(lib, "fused_rmsnorm", "CPU")
def fused_rmsnorm_cpu(x, weight, eps=1e-6):
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    return x / rms * weight

# CUDA implementation
@impl(lib, "fused_rmsnorm", "CUDA")
def fused_rmsnorm_cuda(x, weight, eps=1e-6):
    out = torch.empty_like(x)
    M, N = x.shape
    BLOCK = triton.next_power_of_2(N)
    fused_rmsnorm_kernel[(M,)](x, weight, out, x.stride(0), N, eps, BLOCK)
    return out

# FakeTensor (meta) implementation for torch.compile shape inference
@impl(lib, "fused_rmsnorm", "Meta")
def fused_rmsnorm_meta(x, weight, eps=1e-6):
    return torch.empty_like(x)

# Usage with torch.compile
@torch.compile
def model_forward(x, w):
    return torch.ops.myops.fused_rmsnorm(x, w)  # compiles correctly
```

### Method 3: CUDA Extension (C++/CUDA)
```python
# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='my_kernels',
    ext_modules=[
        CUDAExtension('my_kernels', [
            'csrc/bindings.cpp',
            'csrc/rmsnorm_kernel.cu',
        ])
    ],
    cmdclass={'build_ext': BuildExtension}
)
```

```cpp
// csrc/bindings.cpp
#include <torch/extension.h>

torch::Tensor fused_rmsnorm_cuda(torch::Tensor x, torch::Tensor weight, float eps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_rmsnorm", &fused_rmsnorm_cuda, "Fused RMSNorm (CUDA)");
}
```

## vLLM Integration

### Adding a Custom Op to vLLM
```python
# vLLM uses custom ops registered via torch.library
# Location: vllm/model_executor/layers/

# Step 1: Create the op
# vllm/_custom_ops.py or vllm/model_executor/custom_op.py
import torch
from vllm import _custom_ops as ops

# Step 2: Register in the model
# vllm/model_executor/models/llama.py
class LlamaAttention(nn.Module):
    def forward(self, ...):
        # Use custom attention kernel
        attn_output = ops.paged_attention_v2(
            query, key_cache, value_cache,
            block_tables, seq_lens, ...
        )
        return attn_output

# Step 3: For new model architectures, register the model
# vllm/model_executor/models/__init__.py
_MODELS = {
    "LlamaForCausalLM": ("llama", "LlamaForCausalLM"),
    "MyCustomModel": ("my_model", "MyCustomModelForCausalLM"),
}
```

### vLLM Custom Quantization Method
```python
# vllm/model_executor/layers/quantization/
class MyQuantMethod(QuantizationConfig):
    @classmethod
    def get_name(cls) -> str:
        return "my_quant"

    @classmethod
    def get_supported_act_dtypes(cls):
        return [torch.float16, torch.bfloat16]

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, LinearBase):
            return MyQuantLinearMethod(self)
        return None

class MyQuantLinearMethod(LinearMethodBase):
    def create_weights(self, layer, ...):
        # Create quantized weight tensors
        layer.register_parameter("qweight", ...)
        layer.register_parameter("scales", ...)

    def apply(self, layer, x, bias=None):
        # Dequantize and compute
        return my_quant_gemm(x, layer.qweight, layer.scales)
```

## SGLang Integration

### Custom Backend Kernel
```python
# SGLang uses its own model runner
# sglang/srt/layers/

# Register custom attention backend
class MyAttentionBackend(AttentionBackend):
    def forward(self, q, k, v, ...):
        return my_custom_attention(q, k, v, ...)

# Register custom sampling kernel
class MyCustomSampler:
    def forward_batch(self, logits, sampling_params):
        # Custom top-p/top-k implementation
        return my_fused_sampling(logits, ...)
```

## llama.cpp Integration (GGML)

### Custom GGML Operator
```c
// ggml/src/ggml-cuda/my_op.cu

// Define the CUDA kernel
__global__ void my_custom_kernel(
    const float * x, float * dst,
    int ne00, int ne01
) {
    // kernel implementation
}

// Register with GGML
void ggml_cuda_op_my_custom(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;

    my_custom_kernel<<<blocks, threads, 0, stream>>>(
        src0_d, dst_d, src0->ne[0], src0->ne[1]
    );
}

// In ggml-cuda.cu, add to the dispatch:
case GGML_OP_MY_CUSTOM:
    ggml_cuda_op_my_custom(ctx, dst);
    break;
```

## torch.compile Compatibility

### Making Kernels Compile-Friendly
```python
# BAD: breaks torch.compile (graph break)
def my_op(x):
    if x.shape[0] > 128:  # data-dependent control flow
        return fast_path(x)
    else:
        return slow_path(x)

# GOOD: use torch.cond or static shapes
def my_op(x):
    # Register as custom op with Meta implementation
    return torch.ops.myops.my_kernel(x)

# GOOD: use torch._dynamo.config for dynamic shapes
torch._dynamo.config.capture_dynamic_output_shape_ops = True

# Key rules for torch.compile compatibility:
# 1. Register Meta (FakeTensor) implementation for shape inference
# 2. No Python-level data-dependent control flow
# 3. No in-place ops on inputs (use out-of-place)
# 4. No calls to .item(), .numpy(), or .tolist()
# 5. Avoid torch.Tensor construction in forward pass
```

### Inductor Custom Lowering
```python
# Register kernel as Inductor lowering for automatic fusion
from torch._inductor.lowering import register_lowering

@register_lowering(torch.ops.myops.fused_rmsnorm)
def lower_fused_rmsnorm(x, weight, eps=1e-6):
    # Tell Inductor how to lower this op
    from torch._inductor.ir import TensorBox, Pointwise
    # ... Inductor IR construction
    # This enables Inductor to fuse this op with surrounding ops
```

## Testing Custom Kernels

### Correctness Testing Pattern
```python
import torch
import pytest

@pytest.mark.parametrize("M", [1, 16, 128, 1024])
@pytest.mark.parametrize("N", [64, 128, 256, 4096])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_fused_rmsnorm(M, N, dtype):
    x = torch.randn(M, N, device='cuda', dtype=dtype)
    w = torch.randn(N, device='cuda', dtype=dtype)

    # Reference implementation
    ref = torch.nn.functional.rms_norm(x.float(), (N,), w.float())

    # Custom kernel
    out = fused_rmsnorm(x, w)

    # Tolerance depends on precision
    atol = {torch.float32: 1e-5, torch.float16: 1e-2, torch.bfloat16: 1e-1}[dtype]
    torch.testing.assert_close(out.float(), ref, atol=atol, rtol=1e-3)

@pytest.mark.parametrize("M", [1, 128])
def test_fused_rmsnorm_backward(M):
    x = torch.randn(M, 256, device='cuda', requires_grad=True)
    w = torch.randn(256, device='cuda', requires_grad=True)

    # Gradient check
    torch.autograd.gradcheck(
        lambda x, w: fused_rmsnorm(x.double(), w.double()),
        (x.double(), w.double()),
        eps=1e-6, atol=1e-4
    )
```

### Performance Testing
```python
import triton

def benchmark_kernel(fn, *args, quantiles=[0.5, 0.2, 0.8]):
    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: fn(*args), quantiles=quantiles
    )
    return ms

# Compare against reference
ref_ms = benchmark_kernel(torch.nn.functional.rms_norm, x, (N,), w)
custom_ms = benchmark_kernel(fused_rmsnorm, x, w)
speedup = ref_ms / custom_ms
print(f"Speedup: {speedup:.2f}x")
```
