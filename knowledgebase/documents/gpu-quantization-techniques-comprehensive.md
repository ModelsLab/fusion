# GPU Kernel Quantization Techniques: Comprehensive Reference

> A deeply technical reference covering every major quantization technique for LLM inference and training, including mathematical formulations, kernel implementation strategies, memory layouts, and performance characteristics.

---

## Table of Contents

1. [Weight Quantization Fundamentals](#1-weight-quantization-fundamentals)
2. [AWQ (Activation-Aware Weight Quantization)](#2-awq-activation-aware-weight-quantization)
3. [GPTQ (Generalized Post-Training Quantization)](#3-gptq-generalized-post-training-quantization)
4. [SmoothQuant](#4-smoothquant)
5. [FP8 Quantization (E4M3 and E5M2)](#5-fp8-quantization-e4m3-and-e5m2)
6. [FP4/NF4 Quantization](#6-fp4nf4-quantization)
7. [INT4 Quantization Kernels](#7-int4-quantization-kernels)
8. [MX Formats (Microscaling)](#8-mx-formats-microscaling)
9. [KV Cache Quantization](#9-kv-cache-quantization)
10. [Activation Quantization](#10-activation-quantization)
11. [Sparsity](#11-sparsity)
12. [GGUF/GGML Quantization (llama.cpp)](#12-ggufggml-quantization-llamacpp)
13. [Quantization-Aware Training (QAT)](#13-quantization-aware-training-qat)
14. [Recent Advances](#14-recent-advances)
15. [Decision Matrix](#15-decision-matrix)

---

## 1. Weight Quantization Fundamentals

### 1.1 The Quantization Equation

All quantization techniques build on the fundamental mapping between floating-point values and integer representations:

```
Quantize:    q = clamp(round(x / scale) + zero_point, qmin, qmax)
Dequantize:  x_hat = (q - zero_point) * scale
```

Where:
- `scale = (xmax - xmin) / (qmax - qmin)` (asymmetric)
- `scale = max(|xmax|, |xmin|) / ((qmax - qmin) / 2)` (symmetric)
- `zero_point = round(-xmin / scale) + qmin` (asymmetric) or `zero_point = 0` (symmetric)
- `qmin, qmax` define the integer range (e.g., [-128, 127] for INT8 signed)

### 1.2 Symmetric vs. Asymmetric Quantization

**Symmetric Quantization:**
- Zero point is fixed at 0
- Scale: `s = max(|x|) / ((2^(b-1)) - 1)`
- Range is symmetric around zero: `[-s * qmax, s * qmax]`
- Simpler kernel implementation: dequant is just `x_hat = q * scale` (no zero_point subtraction)
- Preferred for weights (typically centered around zero)

**Asymmetric Quantization:**
- Non-zero zero_point allows representing skewed distributions
- Tighter fit when data is not centered at zero
- Additional compute cost: requires subtracting zero_point before multiplication
- Better for activations (often have non-zero mean, e.g., after ReLU)

### 1.3 Quantization Granularity

**Per-Tensor Quantization:**
- Single `(scale, zero_point)` pair for the entire tensor
- Lowest overhead (2 extra values per tensor)
- Poorest accuracy: outlier channels force wide dynamic range
- Used in: basic INT8 inference, FP8 per-tensor mode

**Per-Channel Quantization:**
- One `(scale, zero_point)` per output channel (row of weight matrix)
- Standard for weight quantization in production
- Handles varying weight distributions across channels
- Overhead: N_channels * sizeof(scale) additional storage

**Per-Group Quantization:**
- One `(scale, zero_point)` per group of G consecutive elements
- Typical group sizes: 32, 64, 128, 256
- Fine-grained control: smaller groups = better accuracy, more overhead
- Used in: GPTQ (group_size=128), AWQ (group_size=128), most W4A16 kernels
- Memory overhead per weight: `sizeof(fp16) / group_size` additional bits

**Per-Block Quantization:**
- 2D blocking (e.g., 128x128 tiles) with one scale per block
- Used in: FP8 block-wise quantization, MX formats
- Enables efficient tiled GEMM implementation

### 1.4 Calibration Methods

All PTQ methods require determining the quantization range from calibration data:

**MinMax Calibration:**
```
xmin = min(X_calibration)
xmax = max(X_calibration)
scale = (xmax - xmin) / (qmax - qmin)
```
- Simplest approach
- Highly sensitive to outliers: one extreme value expands range for all values
- Fast: single pass through data

**Percentile Calibration:**
```
xmin = percentile(X_calibration, p)      # e.g., p = 0.01%
xmax = percentile(X_calibration, 100-p)  # e.g., 99.99%
```
- Clips extreme outliers
- Typical percentiles: 99.99% or 99.999%
- Trades clipping error for reduced range (better precision for majority of values)

**MSE (Mean Squared Error) Calibration:**
```
scale* = argmin_scale E[(X - Dequant(Quant(X, scale)))^2]
```
- Exhaustive search over candidate thresholds
- Directly minimizes the reconstruction error
- More expensive but often gives best results for weight quantization

**KL-Divergence (Entropy) Calibration:**
```
scale* = argmin_scale KL(P_original || P_quantized)
```
- Minimizes information loss between original and quantized distributions
- Default in NVIDIA TensorRT
- Bins the histogram of values, finds threshold that minimizes KL divergence
- Good for activations with complex distributions

---

## 2. AWQ (Activation-Aware Weight Quantization)

### 2.1 Core Insight

Not all weights are equally important. AWQ finds that protecting only ~1% of salient weights dramatically reduces quantization error. Critically, salient weights are identified by **activation magnitudes**, not weight magnitudes -- weights corresponding to large activations matter more.

### 2.2 Mathematical Formulation

**Error Analysis for a Single Weight:**

The quantization error for a weight-activation product is:

```
Err(Q(w) * x) = Delta * RoundErr(w/Delta) * x
```

Where `Delta` is the quantization step size and `RoundErr` has expected value 0.25 under uniform distribution assumptions. The error is directly proportional to the activation magnitude `x`.

**The Scaling Transformation:**

AWQ applies an equivalent transformation: scale weights up by `s` and activations down by `s`:

```
y = (X * diag(s)^{-1}) * (diag(s) * W) = X_hat * W_hat
```

This is mathematically equivalent (same output) but changes the quantization error:

```
Err(Q(w*s) * x/s) / Err(Q(w) * x) = Delta' / (Delta * s)
```

For salient channels (large activations), scaling up the weights (s > 1) reduces quantization error because the new step size `Delta'` grows sub-linearly with `s`.

**Optimization Objective:**

```
s* = argmin_s E_x[ || Q(W * diag(s)) * (diag(s)^{-1} * x) - W*x ||^2 ]
```

Since `Q()` is non-differentiable, this cannot be solved with gradient descent. Instead, AWQ parameterizes:

```
s = s_x^alpha,  where  s_x = E_x[|x|]  (mean activation magnitude per channel)
```

And finds optimal alpha via **grid search** over `alpha in [0, 1]`.

### 2.3 Complete AWQ Algorithm

```
Input: Pretrained model, calibration data (small set, ~128 samples)
Output: Quantized INT4 model

For each linear layer:
  1. Collect activation statistics: s_x = E[|X|] per input channel
  2. Grid search for optimal alpha:
     For alpha in {0, 0.05, 0.1, ..., 1.0}:
       s = s_x^alpha
       W_scaled = W * diag(s)
       W_q = GroupQuantize(W_scaled, group_size=128, bits=4)
       loss = || W_q * diag(s)^{-1} * X - W * X ||^2
     alpha* = argmin loss
  3. Apply optimal scaling:
     s* = s_x^{alpha*}
     W_final = GroupQuantize(W * diag(s*), group_size=128, bits=4)
  4. Store: quantized weights, scales, zero_points, and s* (for activation rescaling)
```

### 2.4 Kernel Implementation: W4A16 Fused Dequant+GEMM

AWQ's inference kernel performs **dequantization fused with GEMM** to avoid materializing full FP16 weights:

**Weight Packing (Offline):**
- 8 INT4 weights packed into one INT32
- Group scales and zero_points stored as FP16
- Layout permuted for coalesced memory access per warp

**Kernel Architecture:**
```
// Pseudocode for AWQ W4A16 kernel
__global__ void awq_gemm_kernel(
    half* C,           // output [M, N]
    half* A,           // activations [M, K], FP16
    uint32_t* B_q,     // quantized weights [N, K/8], packed INT4
    half* scales,      // [N, K/group_size]
    half* zeros        // [N, K/group_size]
) {
    // Each warp handles a tile of the output
    // Load packed INT4 weights from global memory
    // Dequantize: w_fp16 = (unpack_int4(B_q) - zeros) * scales
    // Accumulate: C_tile += A_tile * W_tile  (using FP16 tensor cores)
}
```

**Two kernel paths:**
- `gemv_forward_cuda`: Matrix-vector for autoregressive decoding (batch_size=1)
- `gemm_forward_cuda`: Matrix-matrix for prefill/batched inference

### 2.5 Performance Characteristics

- **Quantization speed**: ~minutes for 70B model (no backprop, no reconstruction)
- **Accuracy**: Minimal degradation at 4-bit, competitive with GPTQ
- **Inference**: ~3-4x memory reduction, 1.5-3x speedup depending on batch size
- **When to use**: General-purpose W4A16 deployment; fast quantization needed; no calibration data constraints

---

## 3. GPTQ (Generalized Post-Training Quantization)

### 3.1 Foundation: Optimal Brain Quantization (OBQ)

GPTQ builds on OBQ, which frames weight quantization as an optimization problem using second-order information:

**Objective:** For each layer, minimize the squared error of the layer output:

```
argmin_W_q || WX - W_q X ||_F^2
```

This is equivalent to minimizing `sum_i (w_i - w_q_i)^T * H * (w_i - w_q_i)` where `H = 2XX^T` is the Hessian of the layer reconstruction error and `w_i` are rows of W.

**OBQ Approach:**
1. Quantize one weight at a time
2. After quantizing weight `w_p`, update all remaining weights to compensate:
   ```
   delta_F = -((w_p - quant(w_p)) / [H^{-1}]_{pp}) * (H^{-1})_{:,p}
   ```
3. Update the inverse Hessian by removing row/column p
4. Complexity: O(d_row * d_col^3) -- too slow for large models

### 3.2 GPTQ Improvements Over OBQ

**Key Insight 1: Fixed Column Ordering**
- OBQ dynamically chooses which weight to quantize next (greedy by error)
- GPTQ quantizes columns in a fixed order (left to right)
- Observation: the final quantization quality depends primarily on the cumulative Hessian-weighted error, not the order
- Enables batched processing

**Key Insight 2: Lazy Batch Updates**
```
For columns i = 0, B, 2B, ... (batch size B, typically 128):
  1. Quantize columns i to i+B using current Hessian inverse
  2. Accumulate weight updates for columns > i+B
  3. Apply all accumulated updates at once to remaining columns
```
This dramatically improves GPU utilization by converting many small updates into few large matrix operations.

**Key Insight 3: Cholesky Decomposition**
- Instead of explicitly computing and updating `H^{-1}`, use Cholesky decomposition
- `H^{-1} = (L L^T)^{-1}` where L is lower triangular
- Numerically stable; avoids accumulating floating-point errors
- Add small diagonal dampening: `H += lambda * I` to ensure positive definiteness

### 3.3 Complete GPTQ Algorithm

```
Input: Weight matrix W [d_row x d_col], Hessian H = 2XX^T, bit-width b
Output: Quantized weights W_q, scales, zero_points

1. H^{-1} = Cholesky(H + lambda*I)^{-1}
2. Optionally reorder columns by descending Hessian diagonal ("act-order")
3. For i = 0, B, 2B, ..., d_col-B:
     E = zeros(d_row, B)    // block error accumulator
     For j = i, i+1, ..., i+B-1:
       q_j = Quantize(W[:,j])                   // quantize column j
       E[:,j-i] = (W[:,j] - q_j) / [H^{-1}]_{jj}  // scaled error
       W[:,j] = q_j
       W[:,j+1:i+B] -= E[:,j-i] * H^{-1}_{j,j+1:i+B}  // update within block
     W[:,i+B:] -= E * H^{-1}_{i:i+B, i+B:}     // lazy batch update
4. Return W_q (now fully quantized)
```

### 3.4 Geometric Interpretation

Recent work (2025) shows GPTQ is mathematically identical to **Babai's Nearest Plane Algorithm** for the Closest Vector Problem (CVP) on a lattice defined by the Hessian. When executed back-to-front:
- The error propagation step corresponds to projecting onto lattice hyperplanes
- The "act-order" heuristic corresponds to choosing the Gram-Schmidt basis order
- The "min-pivot" order (minimum diagonal entry at each LDL step) provides a theoretically motivated improvement

### 3.5 Performance Characteristics

- **Quantization speed**: ~2-4 GPU-hours for 70B model
- **Accuracy**: Generally best-in-class for 4-bit PTQ (especially with act-order)
- **Inference**: Uses Marlin or exllama kernels (see Section 7)
- **When to use**: When quantization quality matters most; willing to spend more time calibrating

---

## 4. SmoothQuant

### 4.1 The Activation Outlier Problem

LLMs exhibit systematic outlier channels in activations -- specific channels consistently have magnitudes 100x larger than others. This makes activation quantization to INT8 extremely difficult: the outlier channels force a wide quantization range, destroying precision for all other channels.

### 4.2 The Smoothing Transformation

SmoothQuant migrates quantization difficulty from activations to weights via a mathematically equivalent per-channel scaling:

```
Y = X * W = (X * diag(s)^{-1}) * (diag(s) * W) = X_hat * W_hat
```

**Computing the smoothing factor:**

```
s_j = max(|X_j|)^alpha / max(|W_j|)^{1-alpha}
```

Where:
- `j` indexes input channels
- `alpha` controls the migration strength (0 = all difficulty on activations, 1 = all on weights)
- `alpha = 0.5` is generally robust (evenly splits difficulty)
- Models with severe outliers (e.g., GLM-130B) benefit from `alpha = 0.75`

### 4.3 W8A8 Quantization Pipeline

```
Offline (once):
  1. Collect activation statistics from 512 calibration sentences
  2. Compute per-channel smoothing factors s_j
  3. Apply smoothing: W_hat = diag(s) * W
  4. Quantize weights to INT8: W_q = round(W_hat / scale_W)
  5. Compute static activation quantization scales

Online (every forward pass):
  1. Smooth activations: X_hat = X * diag(s)^{-1}
  2. Quantize activations to INT8: X_q = round(X_hat / scale_X)
  3. INT8 GEMM: Y_int32 = X_q @ W_q^T
  4. Dequantize: Y = Y_int32 * scale_X * scale_W
```

### 4.4 INT8 GEMM Kernel Implementation

```
// SmoothQuant uses CUTLASS INT8 GEMM kernels
// Key: per-tensor quantization enables standard INT8 matmul
// No custom hardware needed -- uses existing INT8 tensor cores

// Accumulation in INT32:
// C_int32[m][n] = sum_k A_int8[m][k] * B_int8[k][n]

// Dequantization fused with output:
// C_fp16[m][n] = C_int32[m][n] * scale_A * scale_B
```

**Hardware support:**
- NVIDIA: INT8 tensor cores on Turing+ (A100: 624 TOPS INT8 vs 312 TFLOPS FP16)
- CUTLASS INT8 GEMM kernels
- TensorRT-LLM native SmoothQuant support
- Intel: MKL-DT INT8 kernels

### 4.5 Performance Characteristics

- **Memory**: 2x reduction vs FP16 (both weights and activations in INT8)
- **Compute**: Up to 2x speedup (INT8 tensor cores have 2x throughput)
- **Accuracy**: Negligible degradation on OPT-175B, BLOOM-176B, MT-NLG 530B
- **When to use**: When both weight and activation quantization is needed; serving large batch sizes where activation memory matters

---

## 5. FP8 Quantization (E4M3 and E5M2)

### 5.1 Format Specifications

**E4M3 (1-4-3 layout):**
```
[S | EEEE | MMM]  (8 bits total)
- Sign: 1 bit
- Exponent: 4 bits (bias = 7)
- Mantissa: 3 bits
- Dynamic range: +-448
- Special values: NaN (all exponent and mantissa bits = 1)
- No infinity representation
- Precision: ~3-4 decimal digits
```

**E5M2 (1-5-2 layout):**
```
[S | EEEEE | MM]  (8 bits total)
- Sign: 1 bit
- Exponent: 5 bits (bias = 15)
- Mantissa: 2 bits
- Dynamic range: +-57344
- Special values: NaN, +-inf (like IEEE FP formats)
- Precision: ~2-3 decimal digits
```

**When to use each:**
- E4M3: Forward pass (weights, activations) -- needs precision
- E5M2: Backward pass (gradients) -- needs dynamic range
- For inference-only: E4M3 exclusively

### 5.2 Scaling Strategies

FP8's narrow dynamic range (448 for E4M3 vs 65504 for FP16) necessitates explicit scaling:

**Scale computation:**
```
scale = max_representable_fp8 / amax(tensor)
     = 448.0 / max(|tensor|)          # for E4M3
tensor_fp8 = cast_to_fp8(tensor * scale)
```

**Per-Tensor Scaling:**
- Single scale per tensor (weight, activation, or gradient)
- Lowest overhead
- Sufficient for many models when combined with delayed scaling

**Delayed Scaling (Transformer Engine):**
```python
# Maintains history of amax values across iterations
amax_history = ring_buffer(length=1024)  # configurable

# Each iteration:
amax_history.append(max(|current_tensor|))

# Scale computation:
if algo == "max":
    amax = max(amax_history)        # most conservative
elif algo == "most_recent":
    amax = amax_history[-1]         # most responsive

scale = fp8_max / amax              # fp8_max = 448 for E4M3
```

- Uses the scale from the **previous iteration** for the current one (avoids synchronization overhead)
- Maintains amax history window (default 1024 steps)
- Smooths out transient spikes

**Current Scaling:**
- Computes scale from the current tensor's statistics in real-time
- More accurate but requires a synchronization point
- Preferred when distribution changes rapidly

**Per-Block Scaling (MXFP8):**
- Scale per block of 32 elements (see MX Formats section)
- E8M0 (power-of-2) scale format
- Native on Blackwell tensor cores

### 5.3 FP8 GEMM on Hopper (H100)

Hopper introduces native FP8 tensor cores:

```
// H100 FP8 Tensor Core operation:
// D = A_fp8 @ B_fp8 + C
// Accumulation in FP32, output in FP16/BF16/FP32

// Performance:
// H100 SXM: 1979 TFLOPS FP8 (vs 990 TFLOPS FP16)
// = 2x throughput improvement over FP16
```

**Kernel structure (CUTLASS):**
```
// SM90 (Hopper) FP8 GEMM with per-tensor scaling:
// 1. Load A_fp8, B_fp8 tiles via TMA (Tensor Memory Accelerator)
// 2. Warp-group MMA on fp8 tensor cores, accumulate in FP32
// 3. Apply dequantization scales: output = acc * scale_A * scale_B
// 4. Store result in FP16/BF16

// Block-wise variant:
// Each 128x128 tile has its own scale pair
// Scales loaded alongside data tiles
```

**Shape constraints:** Both dimensions must be divisible by 16 for FP8 tensor core operations.

### 5.4 NVIDIA Transformer Engine

Transformer Engine automates FP8 training/inference:

```python
import transformer_engine.pytorch as te

# Replace nn.Linear with TE layer
layer = te.Linear(in_features, out_features)

# FP8 forward pass with delayed scaling
with te.fp8_autocast(enabled=True, fp8_recipe=te.recipe.DelayedScaling(
    amax_history_len=16,
    amax_compute_algo="max",
    fp8_format=te.recipe.Format.HYBRID  # E4M3 forward, E5M2 backward
)):
    output = layer(input)
```

**Key implementation detail:** For MXFP8 training, Transformer Engine creates **both regular and transposed copies** of each tensor from the high-precision input, because Blackwell tensor cores require data to be consecutive over the reduction dimension. This avoids double-quantization errors from quantizing an already-quantized transposed tensor.

### 5.5 Performance Characteristics

- **Memory**: 2x reduction vs FP16
- **Compute**: 2x throughput on Hopper (FP8 vs FP16 tensor cores)
- **Accuracy**: Within ~1% of BF16 with proper scaling
- **When to use**: Hopper/Blackwell GPUs; large batch inference; FP8-native models (increasingly common)

---

## 6. FP4/NF4 Quantization

### 6.1 NF4 (4-bit NormalFloat) -- QLoRA Format

**Key Insight:** Neural network weights are approximately normally distributed. NF4 creates 16 quantization levels that are information-theoretically optimal for the normal distribution.

**The 16 NF4 Quantization Levels:**
```
[-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
 0.0796,  0.1609,  0.2461,  0.3379,  0.4407,  0.5626,  0.7230, 1.0]
```

These are computed as:
```python
# Compute quantile boundaries for N(0,1)
# 2^4 = 16 levels, but 0 must be exactly representable
# Solution: 8 negative levels + 0 + 7 positive levels (asymmetric around zero)
quantiles = scipy.stats.norm.ppf(np.linspace(0, 1, 17))  # 17 boundaries -> 16 bins
nf4_levels = (quantiles[:-1] + quantiles[1:]) / 2         # bin centers
nf4_levels = nf4_levels / max(abs(nf4_levels))             # normalize to [-1, 1]
# Manually set exact zero
nf4_levels[7] = 0.0
```

**Quantization Process:**
```
For each block of 64 elements:
  1. absmax = max(|block|)                          # scaling factor
  2. normalized = block / absmax                     # normalize to [-1, 1]
  3. indices = nearest_nf4_level(normalized)          # 4-bit index (0-15)
  4. Store: packed indices (2 per byte) + absmax (FP16)
```

**Dequantization:**
```
x_hat = nf4_levels[index] * absmax
```

### 6.2 Double Quantization

QLoRA's additional compression: quantize the quantization constants (absmax values) themselves.

```
First quantization: 64-element blocks, each with FP16 absmax (16 bits per 64 weights = 0.25 bpw overhead)

Double quantization:
  - Group 256 absmax values together
  - Quantize them to INT8 with their own FP32 scale
  - Overhead: 8 bits per 64 weights + (32 bits / 256) per 64 weights
  - = 0.127 bpw vs 0.25 bpw original
  - Saves ~0.37 bits per parameter (~3GB for a 65B model)
```

### 6.3 bitsandbytes Implementation

```python
import bitsandbytes as bnb

# Create 4-bit linear layer
linear4bit = bnb.nn.Linear4bit(
    in_features, out_features,
    bias=False,
    compute_dtype=torch.bfloat16,    # compute in BF16
    quant_type="nf4",                # NF4 vs "fp4"
    compress_statistics=True          # enable double quantization
)
```

**Runtime behavior:**
- Storage: INT4 (NF4 encoded)
- Compute: BF16 (dequantize on-the-fly before matmul)
- No fused dequant+GEMM kernel -- dequantizes to BF16, then uses standard GEMM
- This means NF4 is **memory-bound**, not compute-bound

### 6.4 NVIDIA Blackwell NVFP4

Blackwell introduces hardware-native FP4:

**NVFP4 Format:**
```
[S | EE | M]  (4 bits)
- Sign: 1 bit
- Exponent: 2 bits
- Mantissa: 1 bit
- ~8 representable values per sign
- Two-level scaling:
  - Micro-block scale: FP8 (E4M3) per 16 elements
  - Tensor-level scale: FP32 global
```

**Performance on Blackwell:**
- 20 PFLOPS FP4 on B200 (vs 10 PFLOPS FP8)
- 2x throughput over FP8 tensor cores
- ~1.8x memory reduction vs FP8, ~3.5x vs FP16
- Accuracy: within ~1% of FP8 in many benchmarks

**Kernel in vLLM:**
```python
# W4A8 or W4A4 GEMM
# scaled_fp4_quant: FP16/BF16 -> FP4 with block scales
# cutlass_scaled_fp4_mm: FP4 x FP4 or FP4 x FP8 GEMM
# Block scales at 128x128 dimensions in FP8 format
```

### 6.5 When to Use

- **NF4/QLoRA**: Fine-tuning large models on limited VRAM; inference with bitsandbytes when kernel performance is secondary to memory savings
- **NVFP4**: Blackwell-native inference/training; maximum throughput with minimal accuracy loss

---

## 7. INT4 Quantization Kernels

### 7.1 W4A16 Kernel Architecture

All W4A16 kernels share the same fundamental approach:
1. Weights stored as packed INT4 (8 weights per INT32)
2. During GEMM, weights are dequantized to FP16 **on the fly** in registers
3. FP16 tensor core MMA (matrix multiply-accumulate) computes the result
4. Never materialize full FP16 weight matrix in memory

This is memory-bandwidth optimal because you load 4 bits instead of 16 bits per weight, achieving up to 4x effective bandwidth amplification.

### 7.2 Marlin Kernel (NVIDIA, IST-DASLab)

The gold standard for W4A16 GEMM on Ampere GPUs.

**Key Design Decisions:**

*Weight Packing and Layout:*
```
- INT4 weights packed: 8 elements per INT32
- Layout permuted for coalesced warp-level access
- Group scales (FP16) stored separately
- Permutation ensures each warp loads contiguous memory
```

*Dequantization via LOP3:*
```
// Direct INT4->FP16 conversion is slow (requires shifts, masks, float conversion)
// Marlin uses binary bit manipulation via lop3 instruction:
// 1. Extract two INT4 values from an INT32
// 2. Use lop3 (3-input logic operation) to construct FP16 bit patterns directly
// 3. Single instruction per pair of dequantized values
// This avoids the multi-instruction INT->FP conversion path
```

*Asynchronous Memory Loading:*
```
// Uses Ampere's cp.async instruction:
// - Loads directly from global/L2 to shared memory
// - No temporary registers needed (unlike regular loads)
// - Enables overlapping loads with computation
// Pipeline stages: Load tile N+2 | Load tile N+1 | Compute tile N
```

*Thread Block Configuration:*
```
- Thread block: 256 threads
- Each thread block handles a tile of output
- Warp-level GEMM using tensor core MMA instructions
- Multiple pipeline stages to hide memory latency
- Shared memory double-buffering for A and B tiles
```

**Performance:**
- Near-ideal memory bandwidth utilization on Ampere (A100, RTX 3090/4090)
- ~3.9x speedup over FP16 GEMM for large models at batch_size=1
- Struggles on Hopper (H100) due to architecture differences

### 7.3 Marlin-24 (2:4 Sparsity + INT4)

Combines structured sparsity with INT4 quantization:

```
Original: W [N x K], FP16
After 2:4 pruning: W_sparse [N x K], 50% zeros in 2:4 pattern
After INT4 quant: W_compressed [N x K/2], packed INT4 nonzeros + metadata

Effective compression: 4 bits * 0.5 density = 2 bits per original weight
                       + metadata overhead (~0.5 bits)
                       = ~2.5 bits per weight
```

**Kernel Implementation:**
```
// Sparse-Marlin applies Marlin's dense compression on top of 2:4 compression:
// 1. 2:4 sparsity halves K dimension (store only nonzeros)
// 2. INT4 packing further compresses 8 values per INT32
// 3. 2-bit metadata indicates which 2 of 4 positions are nonzero
// 4. During GEMM: load compressed weights + metadata,
//    use metadata to gather corresponding activation values,
//    dequantize INT4 to FP16, compute with tensor cores

// Offline preprocessing:
// Weights, metadata, and group scales are reshuffled into
// layout that gives ideal memory access patterns and
// Sparse Tensor Core data organization
```

**Performance:**
- ~5.3x effective speedup over dense FP16 GEMM
- More robust to larger batch sizes than dense Marlin
- Supported in vLLM on Ampere+ (compute capability >= 8.0)

### 7.4 ExLlama v2 Kernel

Optimized for consumer GPU inference (RTX 3090, 4090):

**EXL2 Format:**
- Variable bit-width: supports 2, 3, 4, 5, 6, and 8-bit quantization
- **Mixed precision within a single model**: different layers (or even different parts of a layer) use different bit-widths
- Importance-aware: allocates more bits to sensitive layers
- Based on GPTQ quantization with custom packing

**Kernel Characteristics:**
- CUDA kernels compiled on first run, cached to `~/.cache/torch_extensions/`
- Optimized for single-GPU inference (not multi-GPU tensor parallel)
- Fused dequantization + GEMM + activation in single kernel where possible
- ~13% faster than ExLlama v1 on equivalent models

**Performance:** ~56 tokens/sec on T4 GPU for 7B models; fastest option for consumer GPU inference.

### 7.5 EETQ (Easy & Efficient Quantization for Transformers)

**Approach:**
- INT8 per-channel weight-only quantization (W8A16)
- No calibration data required
- Kernels from FasterTransformer and TensorRT-LLM
- Attention optimized with FlashAttention2

**Implementation:**
```python
from transformers import EetqConfig
config = EetqConfig(bits=8)  # INT8 per-channel weight quantization
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=config)
```

- Also supports W8A16 GEMV with 10-30% performance improvement
- Negligible accuracy degradation due to per-channel granularity
- When to use: Quick deployment without calibration; INT8 is sufficient

### 7.6 Machete Kernel (Red Hat / vLLM)

Successor to Marlin for Hopper (H100) GPUs:

- Uses Hopper's TMA (Tensor Memory Accelerator) for efficient data loading
- Supports various weight precisions (INT4, INT8, FP8)
- Dynamically generates kernels for input/output type combinations
- Better performance than Marlin on H100 (Marlin suffers up to 20.3% degradation on Hopper)

---

## 8. MX Formats (Microscaling)

### 8.1 OCP Microscaling Standard

The Open Compute Project (OCP) published the MX Specification v1.0 (September 2023) defining four formats:

| Format | Element Type | Element Bits | Scale Format | Block Size | Total Bits/Element |
|--------|-------------|-------------|-------------|-----------|-------------------|
| MXFP8  | FP8 (E4M3 or E5M2) | 8 | E8M0 | 32 | 8.25 |
| MXFP6  | FP6 (E3M2 or E2M3) | 6 | E8M0 | 32 | 6.25 |
| MXFP4  | FP4 (E2M1) | 4 | E8M0 | 32 | 4.25 |
| MXINT8 | INT8 | 8 | E8M0 | 32 | 8.25 |

**Key properties:**
- Block size: 32 elements share one E8M0 scale (8-bit exponent, power-of-2 only)
- Scale overhead: 8 bits / 32 elements = 0.25 bits per element
- E8M0 format: 8 exponent bits, 0 mantissa bits = pure power-of-2 multiplier (2^(e-127))

### 8.2 Block-Scaled Matmul

```
Given: A [M x K] and B [K x N]
Block size: 32 along K dimension

For each 32-element block along K:
  A_block_i has scale s_A_i (E8M0) and elements a_i (FP4/FP8)
  B_block_j has scale s_B_j (E8M0) and elements b_j (FP4/FP8)

  Partial product contribution:
  C += (s_A_i * s_B_j) * sum(a_i * b_j)

  where a_i * b_j is computed in low precision
  and s_A_i * s_B_j is a power-of-2 multiplication (free in hardware)
```

### 8.3 Hardware Support on Blackwell

Blackwell Tensor Cores natively support MXFP4, MXFP6, and MXFP8:

- **Fused block-scaled loading**: Scale factors loaded and applied during tensor core operation
- **On-the-fly dequantization**: No separate dequant step needed
- **MXFP4 performance**: 2x throughput vs FP8/MXFP8 GEMMs
- **tcgen05 packed layout**: Scales stored in hardware-specific format for direct tensor core consumption

### 8.4 NVIDIA NVFP4 vs OCP MXFP4

| Property | NVFP4 | MXFP4 |
|---------|-------|-------|
| Block size | 16 elements | 32 elements |
| Scale format | FP8 (E4M3) + FP32 global | E8M0 |
| Two-level scaling | Yes (micro + tensor) | No (block only) |
| Hardware | Blackwell native | Blackwell native |
| VEC_SIZE in Triton | 16 | 32 |

### 8.5 Triton Implementation

```python
@triton.jit
def block_scaled_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    scale_A_ptr, scale_B_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    VEC_SIZE: tl.constexpr,  # 16 for NVFP4, 32 for MXFP4
):
    # Grid: (M // BLOCK_M) x (N // BLOCK_N) thread blocks
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in tl.range(0, tl.cdiv(K, BLOCK_K)):
        # Load A and B tiles (FP4 or FP8 elements)
        a = tl.load(A_ptr + ...)   # [BLOCK_M, BLOCK_K]
        b = tl.load(B_ptr + ...)   # [BLOCK_K, BLOCK_N]

        # Load scales -- stored in 5D packed layout for hardware:
        # (M//32//4, K//VEC_SIZE//4, 32, 4, 4)
        # Reshaped and transposed to match 2D interface:
        # scale_a.reshape(rep_m, rep_k, 32, 4, 4).trans(0,3,2,1,4)
        #        .reshape(BLOCK_M, BLOCK_K // VEC_SIZE)
        scale_a = load_and_reshape_scales(scale_A_ptr, ...)
        scale_b = load_and_reshape_scales(scale_B_ptr, ...)

        # Block-scaled dot product (hardware-accelerated on Blackwell)
        acc = tl.dot_scaled(a, scale_a, "e2m1", b.T, scale_b, "e2m1", acc)

    # Store result
    tl.store(C_ptr + ..., acc)

# Typical block sizes:
# FP4: BLOCK_M=128, BLOCK_N=256, BLOCK_K=256
# FP8: BLOCK_M=128, BLOCK_N=256, BLOCK_K=128
```

### 8.6 MXFP8 Quantizer on Blackwell (fal.ai)

A high-performance MXFP8 quantizer achieving 6+ TB/s on B200:

**Quantization process per 32-element block:**
```
1. Find max absolute value: amax = max(|block|)
2. Compute E8M0 scale: S = ceil_pow2(amax / 448)  # 448 = FP8 E4M3 max
3. Quantize: block_fp8 = saturate_fp8(block / S)
```

**Key optimization: K-dimension tiling**
```
// Instead of one CTA per row:
// Split K dimension across CTAs
// M=16384, K=16384, rows_per_cta=8, k_tile=256
// = 131,072 CTAs instead of 2,048 (64x more parallel work)
// Performance: 1.3 TB/s -> 3.3 TB/s (2.5x improvement from this alone)
```

**Additional optimizations:**
- TMA bulk loads: single transaction per CTA tile from HBM to shared memory
- Scale bytes packed into 32-bit aligned stores (avoids scattered byte writes)
- Writes scales directly in tcgen05 packed layout (eliminates separate packing step)

### 8.7 When to Use

- **MXFP8**: Blackwell training (matches BF16 convergence); high-throughput inference
- **MXFP4**: Maximum throughput on Blackwell (2x over MXFP8) with minimal accuracy loss
- **MXFP6**: Balance between MXFP4 and MXFP8 when available

---

## 9. KV Cache Quantization

### 9.1 Why Quantize the KV Cache

For long-context LLM serving, the KV cache dominates memory:
```
KV cache size = 2 * num_layers * num_heads * head_dim * seq_len * batch_size * sizeof(dtype)

Example: Llama-70B, seq_len=8192, batch_size=32
= 2 * 80 * 8 * 128 * 8192 * 32 * 2 bytes (FP16)
= ~85 GB

With FP8: ~42 GB (2x reduction)
With INT4: ~21 GB (4x reduction)
```

### 9.2 FP8 KV Cache

**Quantization strategy in vLLM:**

*Per-tensor quantization:*
```
- Single scale for each Q, K, V tensor per layer
- Simplest approach, supported by all backends
- Scale computed once during calibration or dynamically
```

*Per-attention-head quantization:*
```
- One scale per attention head
- Each head has different value distributions
- Only available with FlashAttention backend
- Requires calibration via llm-compressor
```

**Implementation:**
```python
# vLLM: Enable FP8 KV cache
# In model config or via CLI:
# --kv-cache-dtype fp8_e4m3

# Quantize at store time: BF16 -> FP8 when writing to KV cache
# Dequantize at read time: FP8 -> BF16 for attention computation
```

### 9.3 INT4 KV Cache

More aggressive compression but higher accuracy impact:
```
- 4x memory reduction vs FP16
- Per-head or per-token quantization with group scaling
- Higher quantization error impacts attention scores
- Most useful for very long contexts where memory is critical
```

### 9.4 SGLang KV Cache Quantization

```bash
# SGLang supports FP8, INT8, and experimental FP4 KV cache
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3-8B \
    --kv-cache-dtype fp8_e4m3    # or fp8_e5m2, fp4_e2m1 (experimental)

# FP8 blockwise quantization:
# - Benchmarks FP8 GEMM kernels with different block shapes
# - Accumulation in BF16 for numerical stability
# - INT8 scaled matmul fused with W8A8 also supported
```

### 9.5 Impact on Attention Accuracy

```
Attention: softmax(Q @ K^T / sqrt(d)) @ V

Quantization errors in K directly affect attention logits:
- Error in K^T -> error in dot products -> error after softmax
- Softmax amplifies errors: small logit changes can shift attention mass

Quantization errors in V directly affect output:
- Output is weighted sum of V vectors
- Errors in V are linearly propagated

Rule of thumb:
- FP8 KV cache: negligible accuracy loss for most models
- INT4 KV cache: measurable perplexity increase (~0.1-0.5 PPL)
- Per-head quantization significantly better than per-tensor for KV cache
```

### 9.6 When to Use

- **FP8 KV cache**: Default recommendation for long-context serving on Hopper+
- **INT4 KV cache**: When memory is severely constrained; 128K+ context lengths
- **No KV quantization**: Short contexts, accuracy-critical applications

---

## 10. Activation Quantization

### 10.1 The Challenge

Unlike weights (which are static and can be carefully calibrated), activations are:
- **Dynamic**: Change with every input
- **Outlier-prone**: LLMs exhibit systematic outlier channels (100x larger than typical)
- **Distribution-varying**: Different layers, different tokens have different distributions

### 10.2 W8A8 (SmoothQuant Approach)

See Section 4 for full details. Key points:
- Migrate outlier difficulty from activations to weights
- Per-tensor INT8 quantization for both weights and activations
- 2x compute speedup on INT8 tensor cores
- CUTLASS INT8 GEMM kernels

### 10.3 W4A8 Approaches

Combine INT4 weight quantization with INT8 activation quantization:
```
// Weight: INT4 with group quantization (like GPTQ/AWQ)
// Activation: INT8 with per-token or per-tensor scaling
// GEMM: Dequantize INT4 -> INT8, then INT8 GEMM
// Or: Dequantize both to FP16, then FP16 GEMM
```

Challenges:
- INT4 * INT8 mixed-precision GEMM not natively supported on most hardware
- Typically requires dequantizing one operand
- Less common than W4A16 or W8A8

### 10.4 Dynamic Activation Quantization

```
Per-Token Quantization:
  For each token t in the sequence:
    scale_t = max(|activation_t|) / 127  # for INT8
    activation_q_t = round(activation_t / scale_t)

  // Better accuracy: each token has optimal range
  // Higher overhead: scale computation per token per layer

Per-Tensor Quantization:
  scale = max(|activation_batch|) / 127
  activation_q = round(activation_batch / scale)

  // Lower overhead: one scale per tensor
  // Worse accuracy: outlier tokens penalize all tokens
```

### 10.5 When to Use

- **W8A8**: Large batch serving where activation memory is significant; compute-bound scenarios
- **W4A16**: Memory-bound scenarios (small batches, large models)
- **W4A8**: Niche; when both maximum weight compression and activation quantization are needed

---

## 11. Sparsity

### 11.1 2:4 Structured Sparsity (NVIDIA Ampere+)

**Pattern:** In every contiguous block of 4 elements, exactly 2 must be zero.

```
Dense:    [a, b, c, d, e, f, g, h]
2:4:      [a, 0, c, 0, 0, f, 0, h]  (many valid patterns per block)
```

**Sparse Tensor Core Operation:**
```
// Dense:  C = A @ B          (A is sparse, B is dense)
// Sparse: C = A_compressed @ B_selected

// Hardware flow:
// 1. Load compressed A (only non-zeros, 50% of elements)
// 2. Load 2-bit metadata per element (indicates position of non-zeros)
// 3. Use metadata to select corresponding rows/columns from B
// 4. Tensor core MMA on the reduced problem
// Effective: 2x throughput (half the multiplications)
```

**Metadata Format:**
```
// 2 bits per non-zero element position within a 4-element tile
// For each tile of 4 elements, store indices of 2 non-zero positions
// C(4,2) = 6 combinations, encoded in 2 bits per element
// Storage overhead: 2 bits * N_nonzeros = 2 bits per 4 original elements
//                 = 0.5 bits per element = 12.5% overhead for FP16, 25% for INT8
```

**Compressed Matrix Layout:**
```
Original: [N x K] with FP16 values
Compressed: [N x K/2] non-zero values + [N x K/4] metadata (2-bit indices)
Total: ~50% of original size + small metadata overhead
```

### 11.2 Pruning Methods

**Magnitude Pruning:**
```
// Simplest approach: remove smallest weights per 4-element block
For each block of 4 weights [w1, w2, w3, w4]:
  sorted = sort_by_magnitude(block)
  zero out the 2 smallest
// Fast but suboptimal: ignores weight importance
```

**SparseGPT:**
```
// Extends GPTQ framework to pruning
// Uses Hessian information to decide which weights to prune
// After pruning, updates remaining weights to compensate

For each row of W:
  1. Compute H = X @ X^T (Hessian)
  2. For each column:
     - Compute pruning metric using H^{-1}
     - If pruning this weight, set to 0 and update remaining weights:
       delta_W = -(w_p / [H^{-1}]_pp) * H^{-1}_{:,p}
  3. Enforce 2:4 pattern constraints

// Much better accuracy than magnitude pruning
// ~4 GPU-hours for 70B model
```

**Wanda (Pruning by Weights and Activations):**
```python
# Metric: |W| * ||X||_2 (weight magnitude * activation norm)
# Key insight: consider both weight and activation importance

metric = W.abs() * X.norm(p=2, dim=0)  # per-output basis

# For each output row, for each group of 4:
#   Keep the 2 elements with highest metric values
#   Zero the other 2

# Advantages:
# - Single forward pass (no weight updates, no Hessian)
# - Nearly matches SparseGPT accuracy
# - Orders of magnitude faster
```

### 11.3 Combining Sparsity + Quantization

**Marlin-24 (2:4 Sparsity + INT4):**
```
Compression stack:
  1. Prune to 2:4 pattern (50% sparsity) -> 0.5x elements
  2. Quantize remaining to INT4 -> 0.25x bits per element
  3. Pack 8 INT4 values per INT32

Effective compression:
  Original: 16 bits/weight (FP16)
  After: ~2.5 bits/weight (including metadata)
  = ~6.4x compression ratio

Performance: ~5.3x speedup over dense FP16
```

**cuSPARSELt API Workflow:**
```cpp
// 1. Initialize
cusparseLtInit(&handle);

// 2. Create matrix descriptors
cusparseLtStructuredDescriptorInit(&matA, ...);  // sparse
cusparseLtDenseDescriptorInit(&matB, ...);       // dense

// 3. Create matmul descriptor
cusparseLtMatmulDescriptorInit(&matmulDesc, opA, opB, &matA, &matB, &matC, &matC, computeType);

// 4. Prune to 2:4
cusparseLtSpMMAPrune(&handle, &matmulDesc, d_A, d_A_pruned, CUSPARSELT_PRUNE_SPMMA_STRIP);

// 5. Compress
cusparseLtSpMMACompress(&handle, &plan, d_A_pruned, d_A_compressed, d_A_meta);

// 6. Execute sparse GEMM
cusparseLtMatmul(&handle, &plan, &alpha, d_A_compressed, d_B, &beta, d_C, d_D, workspace);
```

**Performance Numbers (BERT-Large, A100):**
```
Layer          Dense (ms)   Sparse (ms)   Speedup
QKV            0.42         0.30          1.4x
FC2            0.38         0.24          1.6x
Overall        -            -             1.3-1.6x
```

### 11.4 When to Use

- **2:4 sparsity alone**: When model can tolerate 50% weight removal; Ampere+ GPUs
- **Sparsity + INT4**: Maximum compression (~6x) with good accuracy; serving cost-sensitive
- **Wanda**: Quick pruning, minimal infrastructure
- **SparseGPT**: When accuracy matters most; willing to invest calibration time

---

## 12. GGUF/GGML Quantization (llama.cpp)

### 12.1 Legacy Formats

**Type-0 formats (symmetric, scale-only):**

| Format | Bits | Block Size | Storage | Description |
|--------|------|-----------|---------|-------------|
| Q4_0   | 4    | 32        | 4.5 bpw | 32 x 4-bit weights + 1 FP16 scale |
| Q5_0   | 5    | 32        | 5.5 bpw | 32 x 5-bit weights + 1 FP16 scale |
| Q8_0   | 8    | 32        | 8.5 bpw | 32 x 8-bit weights + 1 FP16 scale |

**Type-1 formats (asymmetric, scale + min):**

| Format | Bits | Block Size | Storage | Description |
|--------|------|-----------|---------|-------------|
| Q4_1   | 4    | 32        | 5.0 bpw | 32 x 4-bit weights + FP16 scale + FP16 min |
| Q5_1   | 5    | 32        | 5.5 bpw | 32 x 5-bit weights + FP16 scale + FP16 min |

**Dequantization (Type-0):**
```
x = q * scale          // symmetric
```

**Dequantization (Type-1):**
```
x = q * scale + min    // asymmetric, min is the offset
```

Legacy formats are simple and fast to decode but leave accuracy on the table at low bit-widths.

### 12.2 K-Quants (Super-Block Quantization)

K-quants improve accuracy by introducing **super-blocks** of 256 weights, where sub-block scales are themselves quantized:

| Format | Bits/Weight | Structure | Quality |
|--------|------------|-----------|---------|
| Q2_K   | 2.9-3.1    | 256-weight super-blocks, 4-bit sub-block scales | Low quality, extreme compression |
| Q3_K_S | 3.6        | Small variant of Q3_K | Moderate-low quality |
| Q3_K_M | 3.9        | Medium Q3_K | Moderate quality |
| Q3_K_L | 4.3        | Large Q3_K | Better quality |
| Q4_K_S | 4.6        | Small Q4_K | Good quality |
| Q4_K_M | 4.9        | Medium Q4_K | **Recommended default** |
| Q5_K_S | 5.5        | Small Q5_K | Very good quality |
| Q5_K_M | 5.7        | Medium Q5_K | Near-FP16 quality |
| Q6_K   | 6.5        | 6-bit K-quant | Excellent quality |

**K-Quant Structure (example Q4_K):**
```
Super-block: 256 weights
  - FP16 d (super-block scale)
  - FP16 dmin (super-block minimum)
  - 8 sub-blocks of 32 weights each
    - Each sub-block has 6-bit scale and 6-bit minimum
    - These sub-block scales are quantized relative to d and dmin
  - 256 x 4-bit weight values

Dequant: x = d * sub_scale * q + dmin * sub_min
```

The key insight: by quantizing the quantization parameters themselves (sub-block scales stored as 6-bit), K-quants achieve better accuracy at the same bit rate than legacy formats.

### 12.3 IQ Quants (Importance Matrix Quantization)

State-of-the-art for extreme compression using non-linear codebooks:

| Format | Bits/Weight | Technique |
|--------|------------|-----------|
| IQ1_S  | 2.0        | Ternary-like, importance-weighted |
| IQ1_M  | 2.1        | Slightly higher precision |
| IQ2_XXS| 2.3        | Importance-matrix codebook |
| IQ2_XS | 2.5        | |
| IQ2_S  | 2.6        | |
| IQ2_M  | 2.9        | |
| IQ3_XXS| 3.2        | |
| IQ3_XS | 3.4        | |
| IQ3_S  | 3.6        | |
| IQ4_XS | 4.4        | |
| IQ4_NL | 4.7        | Non-linear quant levels |

**How IQ Quants Work:**
```
1. Compute importance matrix from calibration data:
   importance[i] = sum(|activation[i]|^2)  per weight position

2. Use importance to weight the quantization error:
   loss = sum(importance[i] * (w[i] - w_q[i])^2)

3. Non-linear codebooks optimized for the weighted error:
   - Unlike K-quants (linear scale per block)
   - IQ quants use lookup tables of non-uniform levels
   - Levels concentrated where importance is highest

4. Per-tensor-row scales (newer IQ variants):
   - IQ4_KS, IQ2_KS, IQ1_TN use per-row scaling
   - Departing from strict block-wise quantization
```

### 12.4 Kernel Implementation on Different Hardware

**CPU (x86 AVX/AVX-512):**
```
// Optimized for SIMD dequantization + dot product
// Load 32 packed INT4 values (16 bytes)
// Use shuffle/blend instructions to unpack
// Multiply with FP32 activation values
// Horizontal reduction for dot product
```

**Apple Metal (M1/M2/M3/M4):**
```
// Metal compute shaders for GPU-accelerated inference
// Matrix multiplication with on-the-fly dequantization
// Leverages unified memory (no CPU<->GPU copy)
// IQ quants are slower on Apple Silicon due to complex codebook lookups
```

**CUDA:**
```
// GGML CUDA kernels for NVIDIA GPUs
// Dequantize INT4/INT8 in shared memory
// Use tensor cores where possible for the FP16 GEMM
// Performance varies significantly by quant type:
//   Q4_0: simple, fast dequant
//   IQ2_XXS: complex codebook, slower dequant
```

**Vulkan/OpenCL:**
```
// Cross-platform GPU acceleration
// Simpler kernels (no tensor core equivalent)
// Useful for AMD GPUs, Intel iGPUs
```

### 12.5 When to Use

- **Q4_K_M**: Best default -- good balance of size, quality, and speed
- **Q5_K_M**: When accuracy is more important than size
- **IQ4_XS**: When you need to fit a larger model in limited memory
- **IQ2_XXS/IQ2_XS**: Extreme compression for exploration, expect quality loss
- **Q8_0**: When you want near-FP16 quality with 2x compression

---

## 13. Quantization-Aware Training (QAT)

### 13.1 Straight-Through Estimator (STE)

The fundamental challenge: quantization is a step function with zero gradient almost everywhere.

**Solution:** During backpropagation, approximate the quantization gradient as 1 (identity):

```
Forward:  q = Quantize(x)    // discrete, non-differentiable
Backward: dL/dx ≈ dL/dq      // pretend quantize was identity

Formally:
  STE: d/dx Quantize(x) ≈ 1   (within clipping range)
                          ≈ 0   (outside clipping range)
```

**Implementation in PyTorch:**
```python
class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, zero_point, qmin, qmax):
        q = torch.clamp(torch.round(x / scale) + zero_point, qmin, qmax)
        x_hat = (q - zero_point) * scale
        return x_hat

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through: gradient passes through unchanged
        return grad_output, None, None, None, None
```

### 13.2 LSQ (Learned Step Size Quantization)

**Key innovation:** Learn the quantization step size (scale) as a model parameter with a proper gradient.

**Quantization:**
```
x_q = clamp(round(x / s), -Q_N, Q_P)    # s is learnable step size
x_hat = x_q * s                           # dequantized output
```

**Gradient for step size s:**
```
dL/ds = dL/dx_hat * dx_hat/ds

where dx_hat/ds depends on the regime:
  If x/s < -Q_N:  dx_hat/ds = -Q_N        (clipped low)
  If x/s > Q_P:   dx_hat/ds = Q_P         (clipped high)
  Otherwise:       dx_hat/ds = x_q - x/s   (quantization error gradient)
```

**Critical difference from STE:** LSQ gradient is sensitive to the distance between values and transition points, providing finer-grained optimization signal for the step size.

**Gradient scaling:** To ensure step size gradient is properly scaled:
```
gradient_scale = 1.0 / sqrt(Q_P * num_elements)
dL/ds_scaled = gradient_scale * dL/ds
```

### 13.3 PACT (Parameterized Clipping Activation)

**Key innovation:** Learn the clipping range alpha as a model parameter:

```
Forward:
  y = 0                    if x < 0
  y = x                    if 0 <= x < alpha
  y = alpha                if x >= alpha
  y_q = Quantize(y, alpha)  // quantize within [0, alpha]

Gradient for alpha:
  dL/dalpha = dL/dy * 1    if x >= alpha  (clipping boundary)
  dL/dalpha = 0             otherwise
```

**Limitation:** PACT gradient is zero below the clip point, providing no optimization signal for the step size when values are within range. LSQ provides a gradient everywhere, leading to better convergence.

### 13.4 QAT Training Recipe

```python
# Standard QAT workflow:
# 1. Start with a pre-trained FP32/FP16 model
# 2. Insert fake quantization nodes
# 3. Fine-tune with quantization in the loop

model = load_pretrained_model()

# Insert fake quantization
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        # Weight quantizer with learnable step size
        module.weight_quantizer = LSQQuantizer(bits=4)
        # Activation quantizer (optional)
        module.act_quantizer = LSQQuantizer(bits=8)

# Training loop
for batch in dataloader:
    # Forward pass with fake quantization
    output = model(batch)  # uses quantized weights/activations
    loss = criterion(output, target)

    # Backward pass (STE through quantization)
    loss.backward()  # gradients flow through fake quantize nodes

    # Update both model weights AND quantization parameters (scales)
    optimizer.step()

# After training: export with real quantization
model_quantized = export_quantized(model)
```

### 13.5 When to Use

- **QAT**: When PTQ accuracy is insufficient (typically below 4 bits)
- **LSQ**: Default choice for QAT -- better than PACT
- **Full QAT vs LoRA+QAT**: Full QAT is expensive; QLoRA-style fine-tuning is more practical for LLMs

---

## 14. Recent Advances

### 14.1 QuIP# (Quantized Incoherent Processing with Lattice Codebooks)

**State-of-the-art for extreme compression (2-3 bits per weight).**

**Three key innovations:**

**1. Randomized Hadamard Transform (RHT) for Incoherence:**
```
// Problem: weight matrices have "coherent" structure (outlier rows/columns)
// Solution: multiply by random Hadamard matrices to spread information uniformly

W_hat = (D_U * H) @ W @ (H * D_V)
H_hat = (D_U * H) @ H @ (H * D_V)

where:
  H = Hadamard matrix (entries ±1, no floating-point mult needed)
  D_U, D_V = diagonal sign matrices (random ±1)

// Result: O(1)-incoherent matrices with high probability
// Computation: O(n log n) via Fast Walsh-Hadamard Transform
// Inference: apply inverse transform to input/output (cheap, ±1 multiplications)
```

**2. E8 Lattice Codebook (E8P):**
```
// E8 lattice achieves optimal 8-dimensional sphere packing
// 16-bit codebook (65536 entries) compressed to 1KB using E8 symmetries:
//   8 bits: source codebook lookup (227 base entries)
//   7 bits: sign flip indicators
//   1 bit: shift by ±1/4

// Fits in L1 cache of any modern GPU!
// Compare: AQLM uses 1MB codebooks that don't fit in L1

// Vector quantization: groups of 8 weights -> single 16-bit code
// At 2 bits/weight: 8 weights encoded in 16 bits
```

**3. Fine-Tuning:**
```
Stage 1: Within each transformer block, fine-tune unquantized layers
         to compensate for already-quantized layers
Stage 2: After all linear layers quantized, fine-tune remaining
         parameters (including sign vectors) to minimize model-wide error

// Sign vectors optimized as real-valued (continuous relaxation)
// Cost: <0.01 bits per weight for large models
```

**Performance:**
- 2-bit QuIP# achieves quality comparable to other methods' 3-bit models
- 106 tokens/sec on 2-7B (vs AQLM's 20.6 tok/s) -- 5x faster inference
- Over 50% of peak memory bandwidth (1TB/s) during generation with 2-bit models

### 14.2 AQLM (Additive Quantization of Language Models)

**Multi-codebook vector quantization for extreme compression:**

```
// Represent groups of 8-16 weights as a sum of vector codes:
W_group ≈ C_1[i_1] + C_2[i_2] + ... + C_M[i_M]

where:
  C_m = codebook m (learned)
  i_m = index into codebook m (stored)
  M = number of codebooks (typically 1-2 for 2-bit, 2-4 for 3-4 bit)

// Each codebook: 2^B entries of dimension d (group size)
// Storage per group: M * B bits for indices + codebook overhead
```

**Optimization:**
1. Initialize codebooks via beam search on calibration data
2. Jointly optimize codebook entries across entire layer blocks
3. Optional: PV-Tuning (beyond straight-through estimation) for fine-tuning

**Performance:**
- Pareto-optimal at < 3 bits per parameter
- Matches FP16 speed with 8x memory reduction at higher bit rates
- Slower than QuIP# at 2-bit due to larger codebook (doesn't fit L1 cache)

### 14.3 HQQ (Half-Quadratic Quantization)

**Calibration-free quantization using robust optimization:**

```
// Problem: min_q ||W - Dequant(q)||^2  (standard MSE)
// HQQ: min_q ||W - Dequant(q)||_p^p  where p < 1 (non-convex sparsity-promoting norm)

// Solved via Half-Quadratic splitting:
// Introduce auxiliary variable Z:
// min_{q,Z} ||W - Z||_p^p + beta * ||Z - Dequant(q)||^2

// Alternating optimization:
// 1. Fix q, solve for Z (closed-form proximal operator)
// 2. Fix Z, solve for q (closed-form rounding)

// Parameters: p=0.7, beta=1, kappa=1.01, iterations=20
```

**Key advantages:**
- **No calibration data needed** -- works directly on weights
- Quantizes on GPU in half-precision
- 50x faster than GPTQ (minutes for 70B model)
- Supports 8, 4, 3, 2, and 1-bit quantization
- Compatible with Marlin kernels for inference (up to 200 tok/s on 4090)

**Recommended config:** `nbits=4, group_size=64, axis=1`

### 14.4 BitNet (1-bit/1.58-bit LLMs)

**Not post-training quantization -- models trained natively with ternary weights.**

**BitNet b1.58:**
```
// Weights constrained to {-1, 0, +1} during training
// Quantization: absmean quantization
W_ternary = Round(W / mean(|W|))  // maps to {-1, 0, 1}

// Matrix multiplication becomes:
Y = X @ W_ternary
  = sum of additions and subtractions (no multiplications!)

// Effectively: 1.58 bits per weight (log2(3) = 1.58)
```

**Kernel Implementation (bitnet.cpp):**
```
// Based on llama.cpp
// Optimized kernels for ternary matmul on CPU:
// - Replace multiply-accumulate with add/subtract
// - Parallel kernel with configurable tiling
// - Embedding quantization support

// Performance on x86 CPU:
// - 2.37x to 6.17x speedup vs FP16
// - 71.9% to 82.2% energy reduction
// - 100B model at 5-7 tokens/sec on single CPU (human reading speed)
```

**Limitations:**
- Requires training from scratch (cannot post-train quantize existing models)
- Currently CPU-focused (GPU/NPU support planned)
- 2B-4T model released (competitive with similar-size FP16 models)

### 14.5 SpinQuant (Rotation-Based Quantization)

**Insert learned rotation matrices at rotationally-invariant points in LLM architecture:**

```
// Identify 4 invariant points in LLaMA/OPT architectures:
// R1, R2: offline rotations (applied to weights, no runtime cost)
// R3, R4: online Hadamard rotations (for activations/KV cache, ~8% overhead)

// Rotation removes outliers from weights AND activations
// Then standard quantization works much better

// Key finding: random rotations vary by up to 13 PPL points!
// SpinQuant learns optimal rotations via Cayley SGD on Stiefel manifold

// Cayley SGD: gradient descent constrained to orthogonal matrices
// R_{t+1} = Cayley(R_t, -lr * grad)
// Preserves orthogonality exactly (no projection step)
```

**Performance:**
- W4A4KV4 (4-bit everything): only 2.9 point gap vs full precision on LLaMA-2-7B
- Beats LLM-QAT by 19.1 points, SmoothQuant by 25.0 points
- Online Hadamard rotation: ~8% latency overhead (using fast Hadamard kernel)

---

## 15. Decision Matrix

### When to Use What

| Scenario | Recommended Technique | Bits/Weight | Memory Savings | Speed |
|----------|----------------------|------------|---------------|-------|
| **Production serving, Ampere GPU** | AWQ or GPTQ + Marlin kernel | 4-bit (W4A16) | ~4x | ~3-4x |
| **Production serving, Hopper GPU** | FP8 W8A8 or AWQ + Machete | 8-bit or 4-bit | 2-4x | 2-4x |
| **Production serving, Blackwell** | NVFP4 or MXFP8 | 4-8 bit | 2-4x | 2-5x |
| **Maximum compression, quality matters** | QuIP# or AQLM | 2-3 bit | 5-8x | ~2-3x |
| **Maximum compression, speed matters** | GPTQ + Marlin-24 (sparse+quant) | ~2.5 effective | ~6x | ~5x |
| **Consumer GPU inference** | ExLlama v2 (EXL2) | 2-8 bit mixed | 3-8x | Best |
| **CPU inference** | llama.cpp Q4_K_M or IQ4_XS | 4-5 bit | ~4x | Good |
| **Fine-tuning on limited VRAM** | QLoRA (NF4 + LoRA) | 4-bit storage | ~4x | N/A |
| **Long context serving** | + FP8 KV cache | KV: 8-bit | 2x KV | Minimal loss |
| **Training** | FP8 (Transformer Engine) or MXFP8 | 8-bit | 2x | 2x |
| **No calibration data available** | HQQ or EETQ | 4-8 bit | 2-4x | ~2-3x |
| **From-scratch training** | BitNet b1.58 | 1.58-bit | ~10x | CPU-native |
| **W8A8 full quantization** | SmoothQuant | 8-bit | 2x | 2x |

### Accuracy vs. Compression Frontier (approximate ordering, best to worst at each bit level)

```
16-bit: FP16/BF16 (baseline)
8-bit:  FP8 E4M3 ≈ SmoothQuant W8A8 > EETQ W8A16
6-bit:  Q6_K > MXFP6
5-bit:  Q5_K_M > EXL2-5bit
4-bit:  GPTQ (act-order) ≈ AWQ ≈ Q4_K_M > HQQ > NF4(bitsandbytes)
3-bit:  QuIP# 3-bit > AQLM 3-bit > GPTQ 3-bit > Q3_K_M
2-bit:  QuIP# 2-bit > AQLM 2-bit >> others (mostly unusable)
1.58-bit: BitNet b1.58 (trained, not PTQ)
```

### Hardware Compatibility Matrix

| Technique | Ampere (A100) | Ada (4090) | Hopper (H100) | Blackwell (B200) | CPU |
|-----------|:---:|:---:|:---:|:---:|:---:|
| Marlin W4A16 | Best | Good | Degraded | - | - |
| Machete W4A16 | - | - | Best | Best | - |
| FP8 Tensor Core | - | - | Native | Native | - |
| MXFP4/MXFP8 | - | - | - | Native | - |
| 2:4 Sparsity | Native | Native | Native | Native | - |
| INT8 GEMM | Native | Native | Native | Native | AVX-512 |
| GGUF/llama.cpp | CUDA | CUDA | CUDA | CUDA | Native |
| BitNet | - | - | - | - | Native |

---

## Sources

- [AWQ: Activation-aware Weight Quantization (arXiv 2306.00978)](https://arxiv.org/abs/2306.00978)
- [AWQ GitHub (mit-han-lab)](https://github.com/mit-han-lab/llm-awq)
- [AWQ Algorithm Deep Dive (Lei Mao)](https://leimao.github.io/blog/AWQ-Activation-Aware-Weight-Quantization/)
- [GPTQ Algorithm Mechanics](https://apxml.com/courses/practical-llm-quantization/chapter-3-advanced-ptq-techniques/gptq-mechanics)
- [GPTQ Paper (arXiv 2210.17323)](https://arxiv.org/pdf/2210.17323)
- [GPTQ as Babai's Nearest Plane Algorithm](https://arxiv.org/pdf/2507.18553)
- [GPTQ HuggingFace Integration](https://huggingface.co/blog/gptq-integration)
- [GPTQModel (ModelCloud)](https://github.com/ModelCloud/GPTQModel)
- [SmoothQuant Paper (arXiv 2211.10438)](https://arxiv.org/abs/2211.10438)
- [SmoothQuant GitHub (mit-han-lab)](https://github.com/mit-han-lab/smoothquant)
- [FP8 Introduction (NVIDIA Blog)](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/)
- [FP8 Primer (Transformer Engine)](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
- [Per-Tensor and Per-Block FP8 Scaling (NVIDIA Blog)](https://developer.nvidia.com/blog/per-tensor-and-per-block-scaling-strategies-for-effective-fp8-training/)
- [MARLIN Kernel Paper (arXiv 2408.11743)](https://arxiv.org/pdf/2408.11743)
- [Machete: Mixed-Input GEMM for Hopper (Red Hat)](https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel)
- [Sparse-Marlin GitHub (IST-DASLab)](https://github.com/IST-DASLab/Sparse-Marlin)
- [MX Formats Paper (arXiv 2310.10537)](https://arxiv.org/pdf/2310.10537)
- [Block Scaled Matmul (Triton Tutorial)](https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html)
- [MXFP8 Quantizer on Blackwell (fal.ai)](https://blog.fal.ai/chasing-6-tb-s-an-mxfp8-quantizer-on-blackwell/)
- [Triton on Blackwell (NVIDIA Blog)](https://developer.nvidia.com/blog/openai-triton-on-nvidia-blackwell-boosts-ai-performance-and-programmability/)
- [vLLM Quantized KV Cache](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/)
- [SGLang Quantized KV Cache](https://docs.sglang.io/advanced_features/quantized_kv_cache.html)
- [vLLM FP8 W8A8](https://docs.vllm.ai/en/latest/features/quantization/fp8/)
- [vLLM Quantization Kernels (DeepWiki)](https://deepwiki.com/bytedance-iaas/vllm/11.4-quantization-kernels)
- [QLoRA/NF4 (HuggingFace Blog)](https://huggingface.co/blog/4bit-transformers-bitsandbytes)
- [NF4 Deep Dive](https://manalelaidouni.github.io/4Bit-Quantization-Models-QLoRa.html)
- [llama.cpp Quantization README](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)
- [GGUF Quantization Guide](https://kaitchup.substack.com/p/choosing-a-gguf-model-k-quants-i)
- [2:4 Sparsity with TensorRT (NVIDIA Blog)](https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/)
- [cuSPARSELt (NVIDIA Blog)](https://developer.nvidia.com/blog/exploiting-ampere-structured-sparsity-with-cusparselt/)
- [2:4 Sparsity in PyTorch](https://pytorch.org/blog/accelerating-neural-network-training/)
- [Wanda Pruning (arXiv 2306.11695)](https://arxiv.org/abs/2306.11695)
- [QuIP# Paper (arXiv 2402.04396)](https://arxiv.org/abs/2402.04396)
- [QuIP# GitHub (Cornell)](https://github.com/Cornell-RelaxML/quip-sharp)
- [AQLM Paper (arXiv 2401.06118)](https://arxiv.org/abs/2401.06118)
- [HQQ (Dropbox)](https://dropbox.tech/machine-learning/halfquadratic-quantization-of-large-machine-learning-models)
- [HQQ GitHub](https://github.com/mobiusml/hqq)
- [BitNet GitHub (Microsoft)](https://github.com/microsoft/BitNet)
- [BitNet b1.58 Technical Report](https://arxiv.org/pdf/2504.12285)
- [bitnet.cpp Paper](https://arxiv.org/pdf/2502.11880)
- [SpinQuant Paper (arXiv 2405.16406)](https://arxiv.org/abs/2405.16406)
- [SpinQuant GitHub (Facebook Research)](https://github.com/facebookresearch/SpinQuant)
- [EETQ GitHub (NetEase-FuXi)](https://github.com/NetEase-FuXi/EETQ)
- [NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)
- [ExLlamaV2 GitHub](https://github.com/turboderp-org/exllamav2)
- [LSQ Paper (arXiv 1902.08153)](https://arxiv.org/pdf/1902.08153)
- [QAT with PyTorch](https://pytorch.org/blog/quantization-aware-training/)
- [Quantization White Paper (arXiv 2106.08295)](https://arxiv.org/pdf/2106.08295)
