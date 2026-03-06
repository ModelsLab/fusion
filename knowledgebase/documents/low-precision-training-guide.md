---
id: low_precision_training_guide
kind: document
title: Low-Precision Training Techniques
category: training
summary: Complete guide to training and fine-tuning in reduced precision including FP8 training, BF16 strategies, mixed-precision patterns, loss scaling, and Transformer Engine integration.
tags:
  - fp8-training
  - mixed-precision
  - bf16
  - loss-scaling
  - transformer-engine
  - gradient-scaling
source_ids: []
operators:
  - matmul
  - layernorm
  - softmax
gpu_families:
  - Ampere
  - Ada
  - Hopper
  - Blackwell
---

# Low-Precision Training Techniques

## Mixed-Precision Training Fundamentals

### The Standard Recipe (AMP)
```python
# PyTorch Automatic Mixed Precision
scaler = torch.amp.GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        output = model(batch)
        loss = criterion(output, targets)

    # BF16: no scaler needed (8 exponent bits = same range as FP32)
    # FP16: scaler required (5 exponent bits = limited range)
    if using_fp16:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:  # BF16
        loss.backward()
        optimizer.step()
```

### Precision Hierarchy
| Format | Exponent | Mantissa | Range | Use Case |
|--------|----------|----------|-------|----------|
| FP32 | 8 bits | 23 bits | +/-3.4e38 | Master weights, accumulation |
| TF32 | 8 bits | 10 bits | +/-3.4e38 | Tensor core compute (Ampere+) |
| BF16 | 8 bits | 7 bits | +/-3.4e38 | Training activations/gradients |
| FP16 | 5 bits | 10 bits | +/-65504 | Inference, legacy training |
| FP8 E4M3 | 4 bits | 3 bits | +/-448 | Forward pass (Hopper+) |
| FP8 E5M2 | 5 bits | 2 bits | +/-57344 | Backward pass (Hopper+) |

## BF16 Training

### Why BF16 > FP16 for Training
- Same exponent range as FP32: no loss scaling needed
- Gradient underflow virtually eliminated
- Simpler training loop (no GradScaler)
- Native support on Ampere+ (A100, H100, RTX 30xx+)

### BF16 Pitfalls
```python
# Problem: BF16 accumulation loses precision
# 1.0 + 0.0001 in BF16 = 1.0 (mantissa too small)

# Solution: Always accumulate in FP32
# PyTorch does this automatically for matmul on tensor cores
# But custom kernels must handle it explicitly:

@triton.jit
def fused_linear_kernel(X, W, OUT, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # Accumulate in FP32 even with BF16 inputs
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)  # FP32 accumulator
    for k in range(0, K, BLOCK_K):
        a = tl.load(X + ...).to(tl.float32)  # upcast not needed for tl.dot
        b = tl.load(W + ...)
        acc += tl.dot(a, b)  # Triton auto-accumulates in FP32
    # Store as BF16
    tl.store(OUT + ..., acc.to(tl.bfloat16))
```

### Layer-Specific Precision
```python
# Some layers need higher precision:
# - Final LayerNorm: FP32 (small variance values)
# - Loss computation: FP32 (log/exp precision)
# - Embedding lookup: BF16 OK
# - Attention softmax: FP32 accumulation required
# - GEMM: BF16 inputs, FP32 accumulation (automatic on tensor cores)
```

## FP8 Training (Hopper/Blackwell)

### NVIDIA Transformer Engine
```python
import transformer_engine.pytorch as te

# Replace standard layers with TE layers
model = nn.Sequential(
    te.Linear(4096, 4096, bias=False),  # FP8 GEMM
    te.LayerNorm(4096),                  # FP8-aware normalization
    te.Linear(4096, 11008, bias=False),
)

# Training with FP8
with te.fp8_autocast(enabled=True):
    output = model(input_bf16)
    loss = criterion(output, target)

loss.backward()
optimizer.step()
```

### Delayed Scaling Algorithm
```
For each FP8 tensor:
  1. Track amax (absolute maximum) history over last N steps
  2. Compute scale = FP8_MAX / amax_history.max()
  3. Apply scale BEFORE casting to FP8
  4. Store inverse scale for dequantization

Why "delayed"?
  - Scale is computed from PREVIOUS iterations' amax
  - Current iteration's amax updates the history
  - Avoids two-pass (compute amax then quantize) overhead
  - Works because amax changes slowly between steps
```

```python
# Delayed scaling internals (simplified)
class FP8TensorMeta:
    scale: torch.Tensor          # current scale factor
    scale_inv: torch.Tensor      # 1/scale for dequant
    amax_history: torch.Tensor   # (history_len,) ring buffer

def update_scaling(meta, current_amax):
    # Update history
    meta.amax_history = torch.roll(meta.amax_history, -1)
    meta.amax_history[-1] = current_amax

    # Compute new scale from history max
    amax = meta.amax_history.max()
    FP8_MAX = 448.0  # E4M3 max
    meta.scale = FP8_MAX / amax.clamp(min=1e-12)
    meta.scale_inv = 1.0 / meta.scale
```

### FP8 GEMM Pattern
```
Forward:  Y = (X_fp8_e4m3) @ (W_fp8_e4m3)^T  → accumulate FP32 → output BF16
Backward: dX = (dY_fp8_e5m2) @ (W_fp8_e4m3)   → accumulate FP32 → output BF16
          dW = (X_fp8_e4m3)^T @ (dY_fp8_e5m2)  → accumulate FP32 → output FP32

Why E4M3 for forward, E5M2 for backward?
  - Forward: activations/weights have narrow range, need more precision (3 mantissa bits)
  - Backward: gradients have wide range, need more dynamic range (5 exponent bits)
```

### FP8 Training Challenges
```
1. Outlier activations: Some channels have 100x larger values
   → Per-tensor scaling can lose small values
   → Solution: per-channel or block-wise scaling

2. Gradient distribution: Gradients span many orders of magnitude
   → E5M2 helps but still limited
   → Solution: gradient clipping before FP8 cast

3. Optimizer states: MUST remain in FP32
   → Adam moments (m, v) need full precision
   → Only GEMM operands go to FP8

4. Residual connections: Accumulated residuals need higher precision
   → Keep residual stream in BF16
   → Only quantize GEMM inputs
```

## Loss Scaling (FP16 Training)

### Static vs Dynamic Loss Scaling
```python
# Static: fixed scale factor
LOSS_SCALE = 2**16  # 65536
scaled_loss = loss * LOSS_SCALE
scaled_loss.backward()
# Unscale gradients before optimizer step
for p in model.parameters():
    p.grad /= LOSS_SCALE

# Dynamic (PyTorch GradScaler): adapts scale automatically
# - Start with large scale (2^16)
# - If inf/nan in gradients: skip step, halve scale
# - If N consecutive good steps: double scale
# - Typical: scale oscillates around optimal value
```

### Gradient Underflow Analysis
```
FP16 smallest normal: 2^-14 ≈ 6.1e-5
FP16 smallest subnormal: 2^-24 ≈ 5.96e-8

Typical gradient magnitudes:
- Early layers: 1e-5 to 1e-7 (UNDERFLOW RISK)
- Late layers: 1e-3 to 1e-5 (safe)

With loss scale 2^16:
- Early layers: 1e-5 * 65536 = 0.65 (safe in FP16)
- Late layers: 1e-3 * 65536 = 65.5 (safe in FP16)
```

## Stochastic Rounding

### Why It Matters for Low-Precision Training
```
Standard rounding: 1.0 + 0.1 in INT8 → 1 (always rounds down)
  After 10 additions: still 1 (information lost!)

Stochastic rounding: 1.0 + 0.1 → 2 with prob 0.1, 1 with prob 0.9
  After 10 additions: E[result] = 2.0 (correct in expectation!)

# Critical for: weight updates in low precision, gradient accumulation
# Supported in hardware on Blackwell (5th gen tensor cores)
```

### Implementation
```python
@triton.jit
def stochastic_round_kernel(x_ptr, out_ptr, N, seed, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(x_ptr + offs, mask=mask)

    # Generate random uniform [0, 1)
    random = tl.rand(seed, offs)

    # Stochastic round to nearest representable value
    # For FP8: round based on truncated bits
    x_floor = x.to(tl.float8e4m3fn).to(tl.float32)  # round-toward-zero
    x_ceil = x_floor + tl.where(x >= 0, get_ulp(x_floor), -get_ulp(x_floor))

    prob = (x - x_floor) / (x_ceil - x_floor + 1e-10)
    result = tl.where(random < prob, x_ceil, x_floor)

    tl.store(out_ptr + offs, result.to(tl.float8e4m3fn), mask=mask)
```

## Practical Training Configurations

### Configuration by GPU
| GPU | Recommended | Notes |
|-----|------------|-------|
| V100 | FP16 + loss scaling | No BF16 support |
| A100 | BF16 (no scaler) | TF32 for matmul automatically |
| RTX 3090 | BF16 | Same as A100 for precision |
| RTX 4090 | BF16 or FP8 (limited) | FP8 via Transformer Engine (limited support) |
| H100 | FP8 forward + BF16 backward | Full FP8 with Transformer Engine |
| B200 | FP4 forward + FP8 backward | 5th gen tensor cores |

### Fine-Tuning Precision Guide
```
Full Fine-Tuning:
  - Weights: BF16 (master copy FP32)
  - Activations: BF16
  - Gradients: BF16
  - Optimizer: FP32
  - Memory: ~16 bytes/param (4 FP32 weight + 4 FP32 momentum + 4 FP32 variance + 2 BF16 grad + 2 BF16 weight)

LoRA Fine-Tuning:
  - Base weights: frozen, can be INT4/INT8
  - LoRA A, B: BF16
  - Gradients: BF16 (only for A, B)
  - Optimizer: FP32 (only for A, B)
  - Memory: ~0.5-1 bytes/param (base) + 16 bytes/LoRA param

QLoRA:
  - Base weights: NF4 (0.5 bytes/param)
  - LoRA A, B: BF16
  - Dequantize on-the-fly for forward pass
  - Gradient flows through dequantization
  - Memory: ~0.75 bytes/base param + 16 bytes/LoRA param
```

## Numerical Stability Patterns

### Softmax in Low Precision
```python
# WRONG: direct softmax in FP16
exp_x = torch.exp(x.half())  # overflow for x > 11.09 (FP16 max = 65504)

# CORRECT: subtract max first
x_max = x.max(dim=-1, keepdim=True).values
exp_x = torch.exp((x - x_max).half())  # always <= 1.0, safe in FP16

# FlashAttention: online softmax tracks running max
# Rescales partial sums when max changes
```

### Cross-Entropy in Low Precision
```python
# WRONG: separate softmax then log
probs = softmax(logits.half())  # loss of precision for small probs
loss = -log(probs[target])       # log(0) = -inf

# CORRECT: fused log-softmax (logsumexp trick)
# log(softmax(x_i)) = x_i - log(sum(exp(x_j)))
# Compute logsumexp in FP32, subtract in FP32, then cast
log_probs = torch.log_softmax(logits.float(), dim=-1)
loss = F.nll_loss(log_probs, target)
```

### Gradient Norm Clipping with Mixed Precision
```python
# Compute gradient norm in FP32 to avoid overflow
total_norm = 0.0
for p in model.parameters():
    if p.grad is not None:
        param_norm = p.grad.data.float().norm(2)  # upcast to FP32
        total_norm += param_norm.item() ** 2
total_norm = total_norm ** 0.5

# Clip
max_norm = 1.0
clip_coef = max_norm / (total_norm + 1e-6)
if clip_coef < 1:
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data.mul_(clip_coef)
```
