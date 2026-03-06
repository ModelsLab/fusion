---
id: fp8_gemm_hopper_example
kind: example
title: FP8 GEMM on Hopper with Delayed Scaling
category: kernel
summary: Example of FP8 matrix multiplication using torch._scaled_mm and Transformer Engine patterns for Hopper GPU, demonstrating delayed scaling and per-tensor quantization.
tags:
  - fp8
  - gemm
  - hopper
  - scaled-mm
  - transformer-engine
source_ids: []
operators:
  - matmul
  - gemm
gpu_families:
  - Hopper
  - Blackwell
precision:
  - fp8
---

## FP8 GEMM with torch._scaled_mm

```python
import torch

def fp8_quantize(tensor, dtype=torch.float8_e4m3fn):
    """Quantize BF16/FP16 tensor to FP8 with per-tensor scaling."""
    # Compute scale: map tensor range to FP8 range
    amax = tensor.abs().max()
    if dtype == torch.float8_e4m3fn:
        fp8_max = 448.0  # E4M3 max representable value
    else:  # E5M2
        fp8_max = 57344.0

    scale = fp8_max / amax.clamp(min=1e-12)

    # Scale and cast
    tensor_scaled = (tensor.float() * scale).clamp(-fp8_max, fp8_max)
    tensor_fp8 = tensor_scaled.to(dtype)

    return tensor_fp8, scale

def fp8_gemm(a_bf16, b_bf16):
    """FP8 GEMM: C = A @ B^T using scaled_mm."""
    # Quantize inputs
    a_fp8, scale_a = fp8_quantize(a_bf16, torch.float8_e4m3fn)
    b_fp8, scale_b = fp8_quantize(b_bf16, torch.float8_e4m3fn)

    # FP8 matmul with FP32 accumulation
    # scale_a and scale_b are inverse scales for dequantization
    result = torch._scaled_mm(
        a_fp8,
        b_fp8.t(),
        scale_a=torch.tensor(1.0 / scale_a, device='cuda'),
        scale_b=torch.tensor(1.0 / scale_b, device='cuda'),
        out_dtype=torch.bfloat16,
    )
    return result

# Example usage
M, N, K = 4096, 4096, 4096
a = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
b = torch.randn(N, K, device='cuda', dtype=torch.bfloat16)

# FP8 GEMM
result_fp8 = fp8_gemm(a, b)

# Reference BF16 GEMM
result_ref = torch.matmul(a, b.t())

# Compare
relative_error = (result_fp8 - result_ref).abs() / result_ref.abs().clamp(min=1e-6)
print(f"Mean relative error: {relative_error.mean():.4f}")
print(f"Max relative error: {relative_error.max():.4f}")
```

## Delayed Scaling Pattern (Transformer Engine Style)

```python
class FP8LinearWithDelayedScaling(torch.nn.Module):
    """Linear layer with FP8 GEMM and delayed scaling."""

    def __init__(self, in_features, out_features, history_len=16):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(out_features, in_features, dtype=torch.bfloat16)
        )
        # Scaling state
        self.register_buffer('amax_history_input', torch.zeros(history_len))
        self.register_buffer('amax_history_weight', torch.zeros(history_len))
        self.register_buffer('scale_input', torch.tensor(1.0))
        self.register_buffer('scale_weight', torch.tensor(1.0))
        self.history_idx = 0
        self.history_len = history_len

    def update_scales(self, input_amax, weight_amax):
        """Update scales based on amax history (delayed by 1 step)."""
        FP8_MAX = 448.0

        # Record current amax
        idx = self.history_idx % self.history_len
        self.amax_history_input[idx] = input_amax
        self.amax_history_weight[idx] = weight_amax
        self.history_idx += 1

        # Compute scale from history (delayed)
        self.scale_input = FP8_MAX / self.amax_history_input.max().clamp(min=1e-12)
        self.scale_weight = FP8_MAX / self.amax_history_weight.max().clamp(min=1e-12)

    def forward(self, x):
        # Track amax for next iteration's scaling
        input_amax = x.abs().max().detach()
        weight_amax = self.weight.abs().max().detach()

        # Quantize with CURRENT scales (computed from PAST amax)
        x_fp8 = (x.float() * self.scale_input).clamp(-448, 448).to(torch.float8_e4m3fn)
        w_fp8 = (self.weight.float() * self.scale_weight).clamp(-448, 448).to(torch.float8_e4m3fn)

        # FP8 GEMM
        out = torch._scaled_mm(
            x_fp8, w_fp8.t(),
            scale_a=torch.tensor(1.0 / self.scale_input, device=x.device),
            scale_b=torch.tensor(1.0 / self.scale_weight, device=x.device),
            out_dtype=torch.bfloat16,
        )

        # Update scales for next iteration
        self.update_scales(input_amax, weight_amax)

        return out

# Benchmark
import triton
layer = FP8LinearWithDelayedScaling(4096, 4096).cuda()
x = torch.randn(128, 4096, device='cuda', dtype=torch.bfloat16)

# Warmup scales
for _ in range(20):
    _ = layer(x)

ms_fp8 = triton.testing.do_bench(lambda: layer(x))

layer_ref = torch.nn.Linear(4096, 4096, bias=False, dtype=torch.bfloat16, device='cuda')
ms_bf16 = triton.testing.do_bench(lambda: layer_ref(x))

print(f"FP8: {ms_fp8:.3f}ms, BF16: {ms_bf16:.3f}ms, Speedup: {ms_bf16/ms_fp8:.2f}x")
```
