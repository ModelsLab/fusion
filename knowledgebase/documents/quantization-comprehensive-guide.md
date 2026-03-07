---
id: quantization_comprehensive_guide
kind: document
title: Comprehensive Quantization Guide - Every Technique Explained
category: quantization
summary: Deep technical guide covering every quantization technique for LLM inference - AWQ, GPTQ, FP8, FP4, GGUF, MX formats, sparsity, and their kernel implementations.
tags:
  - quantization
  - awq
  - gptq
  - fp8
  - fp4
  - int8
  - int4
  - gguf
  - marlin
  - sparsity
source_ids:
  - awq-activation-aware-weight-quantization
gpu_families:
  - Ampere
  - Ada
  - Hopper
  - Blackwell
precision:
  - fp8
  - fp4
  - int8
  - int4
  - nf4
---

# Comprehensive Quantization Guide

## Quantization Fundamentals

### The Quantization Equation

**Uniform Affine (Asymmetric) Quantization:**
```
q = clamp(round(x / scale + zero_point), qmin, qmax)
x_dequant = (q - zero_point) * scale

scale = (max_val - min_val) / (qmax - qmin)
zero_point = round(qmin - min_val / scale)
```

**Symmetric Quantization:**
```
q = clamp(round(x / scale), -qmax, qmax)
x_dequant = q * scale

scale = max(abs(max_val), abs(min_val)) / qmax
```

### Granularity Levels

| Granularity | Scale Sharing | Memory Overhead | Accuracy |
|------------|---------------|-----------------|----------|
| Per-tensor | 1 scale for entire tensor | Minimal | Lowest |
| Per-channel | 1 scale per output channel | Low | Good |
| Per-group (g=128) | 1 scale per 128 elements | Moderate | Very good |
| Per-group (g=32) | 1 scale per 32 elements | Higher | Excellent |
| Per-element | 1 scale per element | 100% overhead | Perfect (but pointless) |

Standard: **Per-group with group_size=128** is the sweet spot for INT4 weight quantization.

### Calibration Methods

1. **MinMax**: Use observed min/max of tensor → sensitive to outliers
2. **Percentile (99.9%)**: Clip extreme values → more robust
3. **MSE (Mean Squared Error)**: Find scale that minimizes reconstruction error
4. **KL-Divergence**: Match quantized distribution to original → used in TensorRT
5. **OMSE (Optimal MSE)**: Grid search for optimal clipping threshold

## Weight-Only Quantization Techniques

### AWQ (Activation-Aware Weight Quantization)

**Key Insight**: Not all weights are equally important. Weights corresponding to large activation magnitudes are salient and should be preserved more carefully.

**Algorithm**:
1. Run calibration data through model, collect activation statistics per channel
2. For each linear layer, compute importance of each weight channel based on activation magnitudes
3. Find optimal per-channel scaling factor `s` that minimizes quantization error:
   ```
   argmin_s || Q(W * diag(s)) * diag(s)^{-1} * X - W * X ||
   ```
4. Scale salient channels up before quantization (they get more precision), scale back in activation
5. Quantize the scaled weights to INT4

**Why It Works**: By scaling salient weight channels up, they occupy more of the INT4 range and suffer less rounding error. The inverse scaling is absorbed into the activation (or previous layer's output).

**Kernel Implementation (W4A16)**:
```python
# Weights stored as INT4 (packed, 8 values per int32)
# Scales stored as FP16 per group (group_size=128)
# During GEMM:
#   1. Load packed INT4 weights from global memory
#   2. Unpack to FP16 in registers: w_fp16 = (w_int4 - zero) * scale
#   3. MMA with FP16 activations
#   4. Accumulate in FP32
# Dequantization overlaps with memory loads (effectively "free")
```

**Performance**:
- 4-bit AWQ achieves <0.5 perplexity degradation on most models
- Near-lossless for 7B+ models
- Marlin kernel: 3.5-4x speedup over FP16 for decode (memory-bound)

### GPTQ (Post-Training Quantization with Hessian)

**Key Insight**: Use approximate second-order information (Hessian of the layer's reconstruction loss) to make better quantization decisions.

**Algorithm** (per layer):
1. Collect calibration data, compute `H = 2 * X^T * X` (Hessian of squared error)
2. Process columns in order (or with optimal ordering):
   ```
   for each column j:
       q_j = quantize(W[:, j])               # quantize column
       error_j = (W[:, j] - q_j) / H[j,j]    # quantization error
       W[:, j+1:] -= error_j * H[j, j+1:]     # update remaining weights
   ```
3. This compensates for quantization error of each column in subsequent columns

**Lazy Batch Updates**: Process columns in blocks of 128 for better cache behavior:
```
for each block of 128 columns:
    quantize within the block using Hessian updates
    apply accumulated error to remaining weight matrix
```

**Performance**:
- Similar accuracy to AWQ for most models
- Quantization takes longer (minutes per layer) due to Hessian computation
- Some models (with outliers) work better with AWQ

### Marlin Kernel (Fastest W4A16)

The Marlin kernel from IST Austria / NVIDIA achieves near-optimal W4A16 GEMM:

**Key Design Choices**:
1. **Asynchronous global→shared memory**: Use cp.async to load INT4 weights
2. **Register-level dequantization**: Unpack INT4→FP16 in registers using bit manipulation
3. **Full tensor core utilization**: Dequant happens during memory latency
4. **Specialized for batch=1 and small batch**: Optimized for decode
5. **Column-major weight layout**: Better memory access pattern for decode

**Performance Numbers** (H100):
- Batch=1, LLaMA-7B: ~3.8x faster than FP16 cuBLAS
- Batch=16: ~2.5x faster than FP16
- Batch=64: ~1.5x faster than FP16 (approaching compute-bound)
- Achieves 85-90% of theoretical HBM bandwidth

### Marlin-24 (2:4 Sparsity + INT4)

Combines structured sparsity with quantization:
- 2:4 sparsity reduces effective weights by 2x
- INT4 quantization reduces each weight by 4x
- Combined: 8x compression, ~6x speedup for decode

### ExLlamaV2 Kernel

Custom CUDA kernel optimized for GPTQ inference:
- Mixed precision dequantization
- Supports various quantization configs (2-bit, 3-bit, 4-bit, 5-bit, 6-bit, 8-bit)
- Efficient bit packing and unpacking
- Optimized for consumer GPUs (RTX 3090, 4090)

## FP8 Quantization

### Format Details

**E4M3 (4-bit exponent, 3-bit mantissa)**:
- Range: [-448, 448]
- Smallest subnormal: 2^{-9} = 1/512
- Precision: ~3.6 decimal digits
- Use for: forward pass, weights, activations

**E5M2 (5-bit exponent, 2-bit mantissa)**:
- Range: [-57344, 57344]
- Much larger range but less precision
- Use for: backward pass (gradients), or when dynamic range matters more

### Per-Tensor vs Per-Token Scaling

**Per-Tensor (Static)**:
```python
# During calibration, find max activation magnitude
amax = max(abs(tensor))
scale = amax / 448.0  # fp8_max for E4M3
tensor_fp8 = cast_to_fp8(tensor / scale)
```

**Per-Token (Dynamic)**:
```python
# Compute scale per row/token - better accuracy
amax_per_row = tensor.abs().amax(dim=-1, keepdim=True)
scale_per_row = amax_per_row / 448.0
tensor_fp8 = cast_to_fp8(tensor / scale_per_row)
```

### NVIDIA Transformer Engine

Handles FP8 training and inference automatically:

**Delayed Scaling Algorithm**:
```python
# Maintain amax history (window of recent maximums)
amax_history = deque(maxlen=1024)

# For each forward pass:
current_amax = tensor.abs().amax()
amax_history.append(current_amax)

# Scale factor based on historical amax (delayed by 1 step)
scale = fp8_max / max(amax_history)
tensor_fp8 = cast_to_fp8(tensor * scale)
```

**Why Delayed?** Using the previous step's amax avoids a synchronization point in the current step. The assumption is amax doesn't change drastically between steps.

**Usage**:
```python
import transformer_engine.pytorch as te

# Drop-in replacement for nn.Linear
layer = te.Linear(in_features, out_features, bias=True)
# Automatically uses FP8 GEMM when recipe is enabled

with te.fp8_autocast(enabled=True):
    output = layer(input)
```

### FP8 GEMM on Ada/Hopper/Blackwell

Native FP8 tensor cores provide 2x throughput vs FP16:
```
H100 SXM:
- FP16 tensor core: 990 TFLOPS
- FP8 tensor core: 1,979 TFLOPS (2x)
- With sparsity: 3,958 TFLOPS (4x vs FP16 dense)
```

The GEMM: `C_fp32 = scale_A * scale_B * (A_fp8 @ B_fp8)`
- A and B are E4M3 format
- Accumulation in FP32
- Scale factors applied in epilogue
- Result can be output as FP16, BF16, FP32, or FP8

## FP4 Quantization (Blackwell)

### NVIDIA FP4 (E2M1)
- 4-bit floating point: 1 sign + 2 exponent + 1 mantissa
- Range: [-6, 6]
- Values: {0, 0.5, 1, 1.5, 2, 3, 4, 6} (and negatives)
- Requires block scaling for useful dynamic range

### NF4 (NormalFloat4) - QLoRA
- 4-bit format optimized for normally-distributed weights
- Values are quantiles of N(0,1): {-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0, 0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0}
- Better than uniform INT4 for weight quantization

### Block Scaling (MX Format)
OCP Microscaling standard:
```
# Block of 32 elements shares one 8-bit exponent
# Each element has its own mantissa
block_size = 32
shared_exponent = max_exponent(block)  # 8-bit
elements = mantissa_only(each_element)  # MXFP4: 3-bit mantissa

# Dequantize:
value = shared_scale * element_mantissa
```

MX Formats:
| Format | Element bits | Block exponent | Block size | Total bits/element |
|--------|-------------|----------------|------------|-------------------|
| MXFP8 | 8 (E4M3/E5M2) | 8-bit shared | 32 | 8.25 |
| MXFP6 | 6 | 8-bit shared | 32 | 6.25 |
| MXFP4 | 4 | 8-bit shared | 32 | 4.25 |
| MXINT8 | 8 (INT8) | 8-bit shared | 32 | 8.25 |

Blackwell provides hardware support for MX format GEMM.

## GGUF Quantization (llama.cpp)

### Format Types

**Legacy Formats**:
| Type | Bits | Block Size | Description |
|------|------|-----------|-------------|
| Q4_0 | 4.5 | 32 | 4-bit, 1 FP16 scale per block |
| Q4_1 | 5.0 | 32 | 4-bit, 1 FP16 scale + 1 FP16 min |
| Q5_0 | 5.5 | 32 | 5-bit, 1 FP16 scale |
| Q5_1 | 6.0 | 32 | 5-bit, scale + min |
| Q8_0 | 8.5 | 32 | 8-bit, 1 FP16 scale |

**K-Quants (Improved)**:
| Type | Avg BPW | Super Block | Description |
|------|---------|-------------|-------------|
| Q2_K | 2.5625 | 256 | 2-bit with 4-bit scales |
| Q3_K | 3.4375 | 256 | 3-bit with 6-bit scales |
| Q4_K | 4.5 | 256 | 4-bit with 6-bit scales, super-block structure |
| Q5_K | 5.5 | 256 | 5-bit with 6-bit scales |
| Q6_K | 6.5625 | 256 | 6-bit with 8-bit scales |

**IQ-Quants (Importance Matrix)**:
| Type | BPW | Description |
|------|-----|-------------|
| IQ1_S | 1.5625 | 1-bit with importance weighting |
| IQ1_M | 1.75 | 1-bit medium quality |
| IQ2_XXS | 2.0625 | 2-bit extra extra small |
| IQ2_XS | 2.3125 | 2-bit extra small |
| IQ2_S | 2.5 | 2-bit small |
| IQ3_XXS | 3.0625 | 3-bit extra extra small |
| IQ3_S | 3.4375 | 3-bit small |
| IQ4_NL | 4.5 | 4-bit non-linear (lookup table) |
| IQ4_XS | 4.25 | 4-bit extra small |

**IQ-Quant Algorithm**: Uses importance matrix (computed from calibration data) to weight the quantization error. More important weights get more precise quantization.

### K-Quant Super Block Structure
```
// Q4_K super block (256 elements):
struct block_q4_K {
    ggml_fp16_t d;           // super-block scale (FP16)
    ggml_fp16_t dmin;        // super-block minimum (FP16)
    uint8_t scales[12];      // sub-block scales and mins (6-bit each, packed)
    uint8_t qs[128];         // 4-bit quantized values (256 values packed into 128 bytes)
};
// Total: 2 + 2 + 12 + 128 = 144 bytes for 256 values = 4.5 BPW
```

### GGUF Kernel Implementation

**CPU (AVX2/AVX-512)**:
- SIMD vectorized dequantization + dot product
- Use 256-bit/512-bit operations to process multiple elements
- Dequant-as-you-go: unpack, scale, multiply-accumulate

**CUDA**:
- Similar to W4A16 GEMM but with GGML-specific block layouts
- Custom dequantization kernels for each quant type
- Optimized for consumer GPUs (warp-level reduction for small M)

**Metal (Apple Silicon)**:
- SIMD group functions for warp-level operations
- Threadgroup shared memory for tile-based GEMM
- Optimized for unified memory architecture

## SmoothQuant (W8A8)

### Algorithm
**Problem**: Activations have outlier channels (magnitudes 100x larger than median), making activation quantization hard.

**Solution**: Migrate quantization difficulty from activations to weights:
```python
# Per-channel smoothing factor
s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha)  # alpha typically 0.5

# Smooth the model:
W_smooth = W * diag(s)        # weights get harder to quantize (absorb activation range)
X_smooth = X * diag(s)^{-1}   # activations become easier to quantize

# Now both W_smooth and X_smooth can be quantized to INT8 with per-tensor scaling
```

### INT8 GEMM Kernel
```
C_int32 = A_int8 @ B_int8    # INT8 tensor core GEMM, INT32 accumulation
C_fp32 = scale_A * scale_B * C_int32  # dequantize in epilogue
```

**Performance**:
- 2x tensor core throughput vs FP16
- Near-lossless for most models
- Works best for models without extreme outliers

## KV Cache Quantization

### FP8 KV Cache
```python
# Per-head quantization
for each attention head:
    k_fp8 = quantize_to_fp8(k_fp16, scale=amax_k / 448)
    v_fp8 = quantize_to_fp8(v_fp16, scale=amax_v / 448)
# Saves 50% memory, minimal accuracy loss
```

### INT4 KV Cache (KIVI)
```python
# Per-channel quantization for Key, per-token for Value
k_int4 = quantize_per_channel(k, num_bits=4, group_size=128)
v_int4 = quantize_per_token(v, num_bits=4, group_size=128)
# Key: channel-wise is better because attention pattern is per-head
# Value: token-wise because values are aggregated across tokens
```

### KV Cache Memory Impact

For LLaMA-70B, seq_len=4096, batch=1:
```
KV cache per layer = 2 * num_kv_heads * head_dim * seq_len * dtype_size
                   = 2 * 8 * 128 * 4096 * sizeof(dtype)

FP16: 2 * 8 * 128 * 4096 * 2 = 16 MB per layer * 80 layers = 1.25 GB
FP8:  2 * 8 * 128 * 4096 * 1 = 8 MB per layer * 80 layers = 0.625 GB
INT4: 2 * 8 * 128 * 4096 * 0.5 = 4 MB per layer * 80 layers = 0.3125 GB
```

At batch=64, seq_len=4096:
```
FP16: 80 GB (!!!) - more than model weights
FP8:  40 GB
INT4: 20 GB
```
KV cache quantization is critical for high-throughput serving.

## 2:4 Structured Sparsity

### How It Works
For every group of 4 consecutive elements, exactly 2 must be zero:
```
Original:  [0.5, 0.3, -0.1, 0.8,  0.2, -0.4, 0.6, 0.1]
Sparse:    [0.5, 0,   0,    0.8,  0,   -0.4, 0.6, 0   ]
Metadata:  [1,0, 0,1,  0,1,  1,0]  // 2-bit index per group of 4
```

### Tensor Core Support
- Ampere+ tensor cores have native 2:4 sparsity support
- 2x throughput for supported operations
- Metadata stored alongside the sparse matrix (2 bits per 4 elements = 50% overhead on indices but 50% fewer values)

### Compression Format
```
// For each group of 4 FP16 values, store:
// - 2 non-zero FP16 values (4 bytes)
// - 2-bit metadata indicating which 2 of 4 are non-zero (0.5 bytes amortized)
// Compression ratio: ~2x (excluding metadata)
```

### Training with 2:4 Sparsity (ASP - Automatic SParsity)
```python
from apex.contrib.sparsity import ASP

# After training, apply 2:4 sparsity
ASP.prune_trained_model(model, optimizer)

# Fine-tune with sparsity constraint
for epoch in range(fine_tune_epochs):
    train_one_epoch(model)
    ASP.compute_sparse_masks(model)  # re-enforce 2:4 pattern
```

## Advanced Quantization Techniques

### QuIP# (Lattice Quantization)
- Uses lattice codebooks (E8 lattice) instead of uniform quantization
- Applies random orthogonal rotation (incoherence processing) to weights before quantization
- Achieves 2-bit quantization with acceptable quality
- More compute-intensive dequantization kernel (lattice lookup)

### AQLM (Additive Quantization)
- Represents each weight vector as sum of multiple codebook entries
- Vector quantization approach
- Very high compression (2-bit effective) with good quality
- Codebook lookup kernel needed for dequantization

### HQQ (Half-Quadratic Quantization)
- No calibration data needed (data-free)
- Optimizes quantization parameters using half-quadratic splitting
- Fast quantization process
- Comparable quality to GPTQ/AWQ

### SpinQuant (Rotation-Based)
- Learns rotation matrices applied before quantization
- Rotated weights are more quantization-friendly
- Rotation can be fused into adjacent layers

### BitNet / 1-bit LLMs
- Ternary weights: {-1, 0, 1}
- Replace matrix multiplication with addition/subtraction
- Dramatically different kernel design:
  ```
  // Instead of: C += A * W
  // Do: C += A * sign(W)  // just add or subtract activation
  // Kernel is pure memory-bound addition with sign selection
  ```
- BitNet b1.58: ternary {-1, 0, 1} with 1.58 bits per weight

## Quantization Decision Guide

### By GPU Type

| GPU | Recommended Weight Quant | Recommended Activation | KV Cache |
|-----|-------------------------|----------------------|----------|
| RTX 3090 (24GB) | AWQ INT4 (g128) | FP16 | FP16 |
| RTX 4090 (24GB) | AWQ INT4 (g128) | FP16 | FP8 |
| RTX 5090 (32GB) | AWQ INT4 or FP8 | FP16 or FP8 | FP8 |
| A100 80GB | FP8 or AWQ INT4 | FP8 (SmoothQuant) | FP16/FP8 |
| H100 | FP8 (native) | FP8 (native) | FP8 |
| B200 | FP4 (native) | FP8 | FP8/FP4 |

### By Model Size (single GPU)

| Model Size | RTX 4090 (24GB) | A100 (80GB) | H100 (80GB) |
|-----------|-----------------|-------------|-------------|
| 7B | FP16 or FP8 | FP16 | FP8 |
| 13B | INT4 (AWQ/GPTQ) | FP16 | FP8 |
| 34B | INT4 (AWQ) | FP8 | FP8 |
| 70B | Won't fit | INT4 (AWQ) | FP8 (barely) |
| 70B | - | FP8 (2xGPU) | FP8 |

### Quality vs Speed Tradeoff

Ranked by quality (best to worst):
1. FP16/BF16 (baseline, 1x speed)
2. FP8 E4M3 (~0.1 PPL degradation, 1.5-2x speed)
3. AWQ INT4 g128 (~0.3 PPL degradation, 2-3x speed for decode)
4. GPTQ INT4 g128 (~0.3-0.5 PPL degradation, 2-3x speed)
5. AWQ INT4 g32 (~0.2 PPL degradation, 2.5x speed, more scale overhead)
6. GGUF Q4_K_M (~0.3-0.5 PPL degradation, good CPU performance)
7. GGUF Q3_K_M (~1.0 PPL degradation)
8. INT4 per-tensor (~2.0 PPL degradation, not recommended)
9. 2-bit (QuIP#, AQLM) (~1.5-3.0 PPL degradation)

## Practical Quantization Quick Reference

### Installation Commands (All Tools)
```bash
# AWQ
pip install autoawq

# GPTQ
pip install auto-gptq

# FP8 (vLLM handles this natively, OR use llm-compressor)
pip install llmcompressor

# GGUF (need llama.cpp)
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make

# bitsandbytes (NF4/INT8 for QLoRA)
pip install bitsandbytes

# TorchAO (PyTorch-native quantization)
pip install torchao
```

### One-Liner Conversions
```python
# AWQ INT4 (10 min, needs ~32GB VRAM for 8B model)
from awq import AutoAWQForCausalLM
model = AutoAWQForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model.quantize(tokenizer, quant_config={"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "gemm"})
model.save_quantized("./model-awq")

# FP8 (instant, no calibration, Ada/Hopper/Blackwell only)
vllm serve meta-llama/Llama-3.1-8B-Instruct --quantization fp8

# GGUF Q4_K_M
python convert_hf_to_gguf.py ./model --outfile model.gguf --outtype f16
./llama-quantize model.gguf model-q4km.gguf Q4_K_M

# TorchAO INT4 (PyTorch native)
from torchao.quantization import quantize_, int4_weight_only
quantize_(model, int4_weight_only(group_size=128))
```

### Speed & Quality Quick Reference
| Method | Quality Loss | Memory | Decode Speedup | GPU Required | Calibration |
|--------|-------------|--------|---------------|-------------|-------------|
| FP8 | <0.1 PPL | 2x smaller | 1.5-2x | Ada/Hopper/Blackwell | No |
| AWQ INT4 | 0.1-0.5 PPL | 4x smaller | 3-4x | Any NVIDIA | Yes (15min) |
| GPTQ INT4 | 0.1-0.5 PPL | 4x smaller | 2-3x | Any NVIDIA | Yes (1-4hr) |
| GGUF Q4_K_M | 0.2-0.5 PPL | 4x smaller | 2-3x (CPU/GPU) | Any | No |
| INT8 SmoothQuant | <0.1 PPL | 2x smaller | 1.5x | Ampere+ | Yes (15min) |
| NF4 (bnb) | 0.2-0.5 PPL | 4x smaller | 1x (not for serving) | Any NVIDIA | No |

### "Which Quantization Should I Use?" Decision Tree
```
Is quality the #1 priority?
├─ YES → FP8 (if Ada/Hopper/Blackwell) or INT8 SmoothQuant
└─ NO → continue

Need to fit on consumer GPU (24GB)?
├─ YES → Is model > 13B? → AWQ INT4 or GGUF Q4_K_M
│        Is model < 13B? → FP16 might fit, try first
└─ NO → continue

Deploying with vLLM/SGLang?
├─ YES → AWQ INT4 (Marlin kernel = fastest)
└─ NO → GGUF Q4_K_M for llama.cpp/Ollama

Fine-tuning (not inference)?
├─ YES → QLoRA NF4 (bitsandbytes)
└─ NO → AWQ for inference
```

### Common Quantization Mistakes
| Mistake | Consequence | Fix |
|---------|------------|-----|
| Quantizing a small model (<3B) to INT4 | Noticeable quality degradation | Use INT8 or FP8 for small models |
| Using random calibration data | Poor AWQ/GPTQ quality | Use representative task data (128+ samples) |
| Quantizing embedding/LM head layers | Breaks model output quality | Add to ignore list: `ignore=["lm_head"]` |
| INT4 for math/coding tasks | Math accuracy drops significantly | Use FP8 or INT8 for precision-sensitive tasks |
| Deploying GPTQ without Marlin | 2x slower than necessary | Use vLLM with `--quantization gptq_marlin` |
| Using bitsandbytes NF4 for serving | Very slow inference | NF4 is for training only; convert to AWQ for serving |
