# Training Optimization Techniques for GPU Kernel Design

> A comprehensive reference covering mixed precision, distributed training, communication optimization, memory efficiency, fine-tuning, optimizer kernels, data pipelines, and compilation strategies -- all from the perspective of GPU kernel design and optimization.

---

## Table of Contents

1. [Mixed Precision Training](#1-mixed-precision-training)
2. [Distributed Training Deep Dive](#2-distributed-training-deep-dive)
3. [Communication Optimization](#3-communication-optimization)
4. [Memory-Efficient Training](#4-memory-efficient-training)
5. [Fine-Tuning Optimization](#5-fine-tuning-optimization)
6. [Optimizer Kernels](#6-optimizer-kernels)
7. [Data Pipeline Optimization](#7-data-pipeline-optimization)
8. [Compilation for Training](#8-compilation-for-training)

---

## 1. Mixed Precision Training

### 1.1 Floating-Point Format Landscape

| Format | Bits | Sign | Exponent | Mantissa | Dynamic Range | Precision | Hardware |
|--------|------|------|----------|----------|---------------|-----------|----------|
| FP32 | 32 | 1 | 8 | 23 | ~1e-38 to 3e38 | ~7 decimal digits | All GPUs |
| TF32 | 19 | 1 | 8 | 10 | Same as FP32 | ~3.5 digits | Ampere+ |
| FP16 | 16 | 1 | 5 | 10 | ~6e-8 to 65504 | ~3.5 digits | Volta+ |
| BF16 | 16 | 1 | 8 | 7 | Same as FP32 | ~2 digits | Ampere+ |
| FP8 E4M3 | 8 | 1 | 4 | 3 | ~0.002 to 448 | ~1 digit | Hopper+ |
| FP8 E5M2 | 8 | 1 | 5 | 2 | ~1e-7 to 57344 | <1 digit | Hopper+ |
| FP4 E2M1 (NVFP4) | 4 | 1 | 2 | 1 | 0 to 6 | Very limited | Blackwell |
| MXFP8 | 8+shared | 1 | 4 | 3 | Block-scaled | Block-enhanced | Blackwell |

### 1.2 FP16 Training with Loss Scaling

#### Why Loss Scaling Is Necessary

FP16 has a limited dynamic range (6e-8 to 65504). During backpropagation, gradient values frequently fall below the minimum representable value (underflow). Loss scaling multiplies the loss by a large factor before `backward()`, shifting gradients into representable range, then divides by the same factor before the optimizer step.

#### Static Loss Scaling

A fixed scaling factor is chosen (e.g., 1024, 8192, 65536) and remains constant throughout training:

```python
loss = model(input)
scaled_loss = loss * SCALE_FACTOR
scaled_loss.backward()
for param in model.parameters():
    param.grad.data /= SCALE_FACTOR
optimizer.step()
```

**Kernel implications**: Static scaling requires an additional elementwise division kernel on all gradients. This can be fused with gradient clipping or the optimizer step itself.

#### Dynamic Loss Scaling (GradScaler)

PyTorch's `GradScaler` dynamically adjusts the scale factor during training:

```python
scaler = torch.amp.GradScaler()
for input, target in data:
    optimizer.zero_grad()
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        output = model(input)
        loss = loss_fn(output, target)
    scaler.scale(loss).backward()    # Gradients are scaled
    scaler.step(optimizer)            # Unscales, checks for inf/NaN, conditionally steps
    scaler.update()                   # Adjusts scale factor
```

**GradScaler internals**:

1. **Scale phase**: Multiplies loss by current scale factor before `backward()`. All resulting gradients carry this scale.
2. **Unscale phase**: `scaler.step()` first calls `scaler.unscale_(optimizer)`, dividing all gradients by the scale factor. This is a per-parameter elementwise kernel.
3. **Inf/NaN check**: After unscaling, each gradient tensor is checked for `inf` or `NaN` values. This requires a reduction kernel across all gradient tensors.
4. **Conditional step**: If no infs/NaNs found, `optimizer.step()` proceeds normally. Otherwise, the step is **skipped entirely**.
5. **Scale update**: If no overflows occurred for `growth_interval` consecutive steps (default: 2000), the scale factor is multiplied by `growth_factor` (default: 2.0). If an overflow was detected, the scale factor is multiplied by `backoff_factor` (default: 0.5).

**Kernel design considerations for GradScaler**:
- The unscale + inf check can be fused into a single kernel that divides by scale and simultaneously reduces a boolean "found_inf" flag
- The conditional skip requires CPU-GPU synchronization (reading the found_inf flag), creating a pipeline bubble
- Multi-tensor variants (`_amp_foreach_non_finite_check_and_unscale_`) batch the operation across all parameter groups

### 1.3 BF16 Training: Why Preferred on Ampere+

#### Advantages Over FP16

BF16 uses the same 8-bit exponent as FP32, providing identical dynamic range (~1e-38 to ~3e38). This eliminates the need for loss scaling entirely because gradients almost never underflow or overflow:

```python
# BF16 training -- no GradScaler needed
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    output = model(input)
    loss = loss_fn(output, target)
loss.backward()
optimizer.step()
```

**Why it is preferred**:
- No loss scaling overhead (no extra kernels for scale/unscale/inf-check)
- No skipped optimizer steps (no CPU-GPU sync for inf checking)
- No risk of divergence from aggressive dynamic scaling
- Same tensor core throughput as FP16 on Ampere and later architectures
- Simpler training loops, fewer hyperparameters

**Trade-off**: BF16 has only 7 mantissa bits vs FP16's 10, giving roughly half the precision. For most neural network training, this reduced precision does not affect convergence. However, certain numerically sensitive operations (e.g., small accumulations, precise loss values) may still need FP32 master copies.

#### Kernel Implications

- BF16 Tensor Core operations on A100: 312 TFLOPS (same as FP16)
- BF16 memory bandwidth benefit: 2x data per byte vs FP32
- Accumulation should be in FP32 for GEMMs (hardware does this automatically on Tensor Cores)
- Elementwise ops (activations, normalization) should maintain FP32 accumulators where precision matters

### 1.4 FP8 Training (Transformer Engine)

#### FP8 Format Details

Two complementary FP8 formats are used for different phases:

- **E4M3** (4 exponent, 3 mantissa): Higher precision, max value 448. Used for **forward pass** (weights and activations) where precision of the computation matters most.
- **E5M2** (5 exponent, 2 mantissa): Wider dynamic range, max value 57344, supports infinity. Used for **backward pass** (gradients) where the dynamic range of gradient distributions is wider.

This "hybrid" format is the default: `Format.HYBRID` in Transformer Engine.

#### Scaling Strategies

Because FP8 has an extremely narrow representable range, scaling is mandatory. Different strategies trade off accuracy vs performance:

##### Delayed Scaling (History-Based)

The original FP8 training approach. Uses a rolling window of absolute maximum (amax) values from previous iterations to compute the scaling factor for the next iteration:

```python
from transformer_engine.common.recipe import DelayedScaling, Format

recipe = DelayedScaling(
    fp8_format=Format.HYBRID,       # E4M3 forward, E5M2 backward
    amax_history_len=16,            # Track amax over 16 iterations
    amax_compute_algo="max"         # Use max of history window
)

with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
    output = te_model(input)
```

**How it works**:
1. Each FP8 operation maintains an amax history buffer of length `amax_history_len`
2. After each iteration, the current amax (max absolute value of the tensor) is recorded
3. The scaling factor for the next iteration = `FP8_MAX / amax_from_history`
4. The `amax_compute_algo` determines how the history is reduced: `"max"` takes the maximum across history, `"most_recent"` uses only the last value

**Kernel implications**:
- Requires amax reduction kernels after each FP8 operation
- Scaling factor computation is lightweight (elementwise)
- The "delay" means the first few iterations may have suboptimal scaling
- Amax history buffers consume additional memory per FP8 tensor

##### Per-Tensor Current Scaling (Just-in-Time)

Computes the scaling factor based on the current tensor's statistics, not historical data:

```python
from transformer_engine.common.recipe import Float8CurrentScaling
recipe = Float8CurrentScaling()
```

**Advantages**: Instantaneous adaptation, no history buffers, simpler implementation, robust against distribution shifts.

**Kernel implications**: Requires an additional pass over the data to compute amax before quantization, or fused amax-and-quantize kernels.

##### Per-Block Scaling (DeepSeek-v3 Style)

Divides each tensor into smaller contiguous blocks and assigns a separate scaling factor to each block:

```python
from transformer_engine.common.recipe import Float8BlockScaling
recipe = Float8BlockScaling()  # Configurable block dimensions (e.g., 1x128, 128x128)
```

**Advantages**: Different regions of a tensor can have vastly different value distributions. Per-block scaling captures this variation, improving quantization fidelity.

**Kernel implications**: Block-level amax computation, per-block quantization kernels, scaling factors stored in FP32 alongside FP8 data.

##### MXFP8 Block Scaling (Blackwell)

Hardware-native block scaling on Blackwell GPUs with blocks of 32 consecutive values:

```python
from transformer_engine.common.recipe import MXFP8BlockScaling
recipe = MXFP8BlockScaling()
```

- Scaling factors use E8M0 format (8-bit power-of-2 representation)
- All values use E4M3 format (both forward and backward)
- Hardware handles requantization for transposes automatically
- Convergence matches BF16 on Nemotron 2B and 8B models with no significant accuracy differences

**Constraint**: Both dimensions of Linear layer inputs must be divisible by 16.

#### NeMo Framework Configuration

```yaml
# In NeMo config
fp8_recipe: "delayed"      # or "tensorwise", "mxfp8", "blockwise"
```

### 1.5 TF32 for Accelerated FP32

TensorFloat-32 is not a storage format but a **compute mode** for Tensor Cores on Ampere+ GPUs.

**How it works**:
1. Input operands are in FP32 format (32 bits in memory)
2. Before Tensor Core computation, mantissa is truncated from 23 bits to 10 bits (matching FP16 precision)
3. The 8-bit exponent is preserved (maintaining FP32 dynamic range)
4. Accumulation is done in full FP32 precision
5. Result is stored as FP32

**Performance**: Up to 8x speedup over FP32 V100 Tensor Core operations on A100. Up to 10x speedup compared to FP32 on V100 (non-Tensor-Core).

**PyTorch configuration**:
```python
# TF32 for matmul (disabled by default since PyTorch 1.12)
torch.backends.cuda.matmul.allow_tf32 = True

# TF32 for cuDNN convolutions (enabled by default)
torch.backends.cudnn.allow_tf32 = True
```

**Kernel implications**:
- No code changes needed -- cuBLAS and cuDNN automatically select TF32 kernels when enabled
- Provides "free" speedup for FP32 workloads without any precision management
- The 10-bit mantissa provides ~3.5 decimal digits of precision (same as FP16/BF16)
- Accumulation in FP32 prevents error accumulation in large reductions

### 1.6 When to Use Each Precision

| Scenario | Recommended | Rationale |
|----------|------------|-----------|
| Quick prototyping on Ampere+ | BF16 | No loss scaling needed, simple code |
| Production training on Volta/Turing | FP16 + GradScaler | Only FP16 Tensor Cores available |
| Production training on Ampere+ | BF16 | Best simplicity-performance trade-off |
| Maximum throughput on Hopper | FP8 (delayed or current scaling) | 2x Tensor Core throughput over BF16 |
| Maximum throughput on Blackwell | MXFP8 or NVFP4 | Hardware-native block scaling |
| Legacy FP32 code, quick speedup | Enable TF32 | Zero code changes |
| Numerically sensitive training | FP32 with TF32 enabled | Full FP32 accumulation |
| Fine-tuning with limited memory | QLoRA (NF4) + BF16 compute | 4-bit weights, BF16 compute |
| Inference | FP8 E4M3 or INT8/INT4 | No backward pass needed |

### 1.7 GradScaler Implementation Deep Dive

**Key internal state**:
- `_scale`: Current loss scale factor (tensor on GPU, default: 2^16 = 65536)
- `_growth_tracker`: Count of consecutive non-inf steps
- `_growth_factor`: Multiplier when growing scale (default: 2.0)
- `_backoff_factor`: Multiplier when backing off (default: 0.5)
- `_growth_interval`: Steps between growth attempts (default: 2000)
- `_per_optimizer_states`: Per-optimizer inf/NaN tracking

**Multi-tensor operations** (kernel-level):
- `_amp_foreach_non_finite_check_and_unscale_`: Fused kernel that iterates over a list of tensors, divides each by the scale factor, and sets a `found_inf` flag if any non-finite value is encountered
- This avoids launching N separate kernels for N parameter groups
- The found_inf tensor lives on GPU; reading it requires a D2H sync

**Performance impact of dynamic scaling**:
- Typical overhead: 2-5% of training step time
- Main cost: CPU-GPU synchronization for inf check result
- BF16 eliminates this entirely, which is why it is preferred when hardware supports it

---

## 2. Distributed Training Deep Dive

### 2.1 Data Parallelism

#### DDP (DistributedDataParallel)

Each GPU maintains a **complete copy** of model parameters, gradients, and optimizer states. Training proceeds as:

1. Each worker receives a different mini-batch
2. Forward pass produces local loss
3. Backward pass computes local gradients
4. **All-reduce** synchronizes gradients across all workers (averaging)
5. Each worker applies identical optimizer step

**Memory per GPU**: Full model parameters + full gradients + full optimizer states (e.g., Adam: 2x model size for m and v)

**Communication**: Single all-reduce per backward pass, overlapped with gradient computation through **bucketing** -- gradients are all-reduced as soon as a bucket fills, without waiting for the full backward pass.

**Bucket strategy**: Default bucket size is 25MB. Gradients are added to buckets in reverse parameter order (matching backward computation order). When a bucket is full, its all-reduce begins asynchronously while backward continues computing gradients for the next bucket.

**Kernel implications**:
- All-reduce uses NCCL ring or tree algorithms
- Gradient bucketing requires memory copies into contiguous buffers
- The bucket-fill and all-reduce launch are on separate CUDA streams, enabling true overlap

#### FSDP (Fully Sharded Data Parallel)

Implements ZeRO Stage 3 in PyTorch. Each GPU holds only a **shard** (1/N) of parameters, gradients, and optimizer states:

1. **Before forward**: `all_gather` to reconstruct full parameters for the current layer
2. **Forward computation**: Uses full parameters
3. **After forward**: Discard non-local parameter shards (free memory)
4. **Before backward**: `all_gather` again to reconstruct parameters
5. **Backward computation**: Compute gradients on full parameters
6. **After backward**: `reduce_scatter` gradients so each GPU gets its gradient shard
7. **Optimizer step**: Each GPU updates only its parameter shard

**Memory per GPU**: (1/N) * (parameters + gradients + optimizer states) + transient full-layer parameters

**Communication volume**: 3x the parameter size per step (vs 1x for DDP), but memory savings are dramatic.

**Trade-off**: FSDP increases communication volume but reduces memory by ~N-fold, enabling much larger models or batch sizes (2-3x larger batches than DDP).

#### FSDP2

PyTorch's redesigned FSDP with key improvements:

- **Per-parameter sharding**: Parameters are individually sharded (no flattening/concatenation into FlatParameters), improving debuggability and model compatibility
- **DTensor-based**: Sharded parameters are represented as `DTensor` with `Shard(dim=0)` placement
- **Composable**: Works with `torch.compile`, tensor parallelism, and other PyTorch native features
- **Mixed precision**: Via `MixedPrecisionPolicy` for parameter, reduction, and output dtypes independently

```python
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy

mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
for layer in model.layers:
    fully_shard(layer, mp_policy=mp_policy)
fully_shard(model, mp_policy=mp_policy)  # Root wrap
```

#### HSDP (Hybrid Sharded Data Parallel)

Combines FSDP within a node and DDP across nodes. Parameters are fully sharded across GPUs within a node (using fast NVLink), but replicated across nodes (using slower inter-node network). This reduces inter-node communication to a simple all-reduce of gradients rather than the more communication-heavy all-gather/reduce-scatter pattern of full FSDP.

### 2.2 ZeRO Redundancy Optimizer

DeepSpeed's ZeRO partitions training state across data-parallel processes in three progressive stages:

#### Stage 1: Optimizer State Partitioning

Each GPU stores only 1/N of the optimizer states (for Adam: first and second moments). Parameters and gradients are still replicated.

**Memory savings**: Optimizer states typically consume 2x model size (Adam m + v) + 1x FP32 master copy = ~12 bytes/param for mixed precision. Stage 1 reduces this to 12/N bytes/param for optimizer states.

**Communication**: Same as DDP (one all-reduce for gradients).

#### Stage 2: + Gradient Partitioning

Each GPU stores only 1/N of the gradients in addition to Stage 1.

**Memory savings**: Gradients consume 2 bytes/param (FP16) or 4 bytes/param (FP32). Stage 2 reduces this to 2/N or 4/N bytes/param.

**Communication**: Replace all-reduce with **reduce-scatter** (each GPU receives only its gradient shard). Communication volume is the same as all-reduce but memory is freed immediately after scatter.

#### Stage 3: + Parameter Partitioning

Each GPU stores only 1/N of the parameters. Before each forward/backward computation, parameters are gathered via all-gather.

**Memory savings**: Near-optimal -- each GPU stores approximately `(16 * params) / N` bytes for mixed-precision Adam (FP16 param + FP16 grad + FP32 master + FP32 m + FP32 v = 16 bytes/param total, divided by N).

**Communication**: 1.5x the volume of DDP (additional all-gather for parameters in forward).

**Kernel implications**:
- Stage 3 requires careful prefetching: the all-gather for layer L+1 should overlap with compute for layer L
- Parameter reconstruction kernels must handle the gather-compute-release lifecycle efficiently
- Gradient reduce-scatter can be fused with gradient computation

### 2.3 Tensor Parallelism (Megatron-LM Style)

Tensor parallelism (TP) splits individual layers across GPUs. The canonical approach from Megatron-LM splits the weight matrices of transformer layers:

#### Column-Parallel Linear

The weight matrix W is split column-wise: W = [W1 | W2 | ... | Wn], one chunk per GPU.

```
Input X (replicated on all GPUs)
  |
  +---> GPU 0: Y0 = X @ W0   (partial output)
  +---> GPU 1: Y1 = X @ W1
  +---> GPU n: Yn = X @ Wn
  |
  All-gather: Y = [Y0 | Y1 | ... | Yn]
```

Used for the **first linear layer** in MLP (before activation) and the **QKV projection** in attention.

**Key property**: The activation function can be applied locally before the all-gather if the activation is element-wise (e.g., GeLU on MLP).

#### Row-Parallel Linear

The weight matrix W is split row-wise, and input is split correspondingly:

```
Input X (split across GPUs: X = [X0 | X1 | ... | Xn])
  |
  +---> GPU 0: Y0 = X0 @ W0   (partial sum)
  +---> GPU 1: Y1 = X1 @ W1
  +---> GPU n: Yn = Xn @ Wn
  |
  All-reduce: Y = Y0 + Y1 + ... + Yn
```

Used for the **second linear layer** in MLP (after activation) and the **output projection** in attention.

#### Communication Pattern

In a transformer block with TP:
- **Forward**: 2 all-reduces per transformer layer (one for attention, one for MLP). With sequence parallelism, these become all-gather and reduce-scatter.
- **Backward**: 2 all-reduces (conjugate of forward communication)
- Total: 4 all-reduce (or equivalent) operations per layer per step

**Kernel implications**:
- TP reduces per-GPU GEMM size, increasing arithmetic intensity requirements
- For small TP sizes (2-4), the reduced GEMM size may fall below Tensor Core efficiency thresholds
- TP is best used within a node (NVLink) due to high communication frequency
- Custom kernels must handle the split/gather patterns at layer boundaries

### 2.4 Pipeline Parallelism

#### GPipe Schedule

The simplest pipeline schedule. All microbatches flow through forward, then all flow through backward ("fill-drain"):

```
GPU 0: F0 F1 F2 F3 ---- B3 B2 B1 B0
GPU 1:    F0 F1 F2 F3 ---- B3 B2 B1 B0
GPU 2:       F0 F1 F2 F3 ---- B3 B2 B1 B0
GPU 3:          F0 F1 F2 F3 ---- B3 B2 B1 B0
```

**Bubble ratio**: `(p-1) / m` where p = pipeline stages, m = microbatches. With 4 stages and 16 microbatches: 18.75% bubble.

**Memory**: Must store activations for all m microbatches simultaneously.

#### 1F1B (One-Forward-One-Backward) Schedule

After a warmup phase, alternates one forward and one backward microbatch:

```
GPU 0: F0 F1 F2 F3 B0 F4 B1 F5 B2 F6 B3  B4 B5 B6
GPU 1:    F0 F1 F2 B0 F3 B1 F4 B2 F5 B3 F6  B4 B5 B6
...
```

**Bubble ratio**: Same as GPipe: `(p-1) / m`

**Memory advantage**: Only p microbatch activations need to be stored (vs m for GPipe), because backward passes free activations early.

#### Interleaved 1F1B Schedule

Each GPU handles multiple non-contiguous stages (virtual pipeline stages). Model is divided into more chunks than physical stages:

```
# With 4 GPUs and virtual_pipeline_size=2:
GPU 0: Stages {0, 4}
GPU 1: Stages {1, 5}
GPU 2: Stages {2, 6}
GPU 3: Stages {3, 7}
```

**Bubble ratio**: `(p-1) / (v * m)` where v = virtual pipeline size. With v=2: half the bubble of standard 1F1B.

**Trade-off**: More communication (each microbatch passes through each GPU v times) but smaller bubbles.

#### Zero-Bubble Pipeline Parallelism

The key innovation: split the backward pass into two phases:
- **B phase**: Compute gradients with respect to **inputs** (needed for upstream stages)
- **W phase**: Compute gradients with respect to **weights** (needed only for local optimizer step)

Since W does not depend on upstream stages, it can be freely scheduled to fill bubble slots.

**ZB-H1 (ZB-1p)**:
- Bubble = `(p-1)(T_F + T_B - T_W)` -- roughly 1/3 of 1F1B bubble
- Same memory as 1F1B
- 8-20% throughput improvement over 1F1B

**ZB-H2 (ZB-2p)**:
- Bubble = `(p-1)(T_F + T_B - 2*T_W)` -- near-zero bubble
- 2x memory of 1F1B (must store activations for deferred W)
- 23-31% throughput improvement over 1F1B

**ZB-V**:
- Divides model into 2p chunks in a "V" shape
- Achieves zero bubble under equal execution time assumptions
- Same memory as 1F1B
- 15-25% throughput improvement

**Optimizer synchronization**: Since W phases are deferred, the optimizer step timing differs across stages. The solution uses a **post-validation strategy**: partial state propagation forward, then full validation backward with reversible optimizer rollback (demonstrated for AdamW).

**Implementation**: Based on Megatron-LM, requires profiling iterations to measure T_F, T_B, T_W for the automatic scheduling algorithm.

### 2.5 Sequence Parallelism

Distributes the computation along the **sequence dimension** for operations that are not parallelized by tensor parallelism (e.g., LayerNorm, Dropout):

- In a standard TP setup, LayerNorm operates on the full sequence on each GPU (redundantly)
- With SP, the sequence is split across TP GPUs, so each GPU processes 1/N of the sequence for these operations
- Communication: converts all-reduce to reduce-scatter (forward) and all-gather (backward)

**Memory savings**: Activation memory for LayerNorm and Dropout is reduced by TP_size factor.

**Requirement**: `tensor_model_parallel_size > 1` (SP is an extension of TP, not independent).

### 2.6 Context Parallelism

Designed for **long-context training** where the sequence length makes attention computation memory-prohibitive:

- Input tensor is split along the sequence dimension across **all** layers (unlike SP which only splits non-TP operations)
- Each GPU processes a chunk of the sequence
- **Ring attention** implementation: KV sub-blocks are passed between GPUs in a ring topology via peer-to-peer communication

**How Ring Attention works**:
1. Split the input sequence into P chunks (one per GPU)
2. Each GPU computes local attention for its Q chunk against its local KV chunk
3. KV chunks are rotated around the ring (GPU i sends to GPU (i+1) % P)
4. Each GPU computes attention against the received KV chunk and accumulates
5. After P-1 rotations, each GPU has computed full attention

**Memory**: O(S/P) per GPU instead of O(S) -- enables training with 128K+ context lengths.

**Kernel implications**: Attention kernels must support **chunked/incremental attention** where partial softmax results are accumulated across chunks using the log-sum-exp trick for numerical stability.

### 2.7 Expert Parallelism (MoE)

For Mixture-of-Experts models where routing selects a subset of experts per token:

- Experts are distributed across GPUs: each GPU holds a subset of experts
- **All-to-All communication**: Tokens are routed from the GPU that processed them to the GPU holding the selected expert
- After expert computation, results are sent back via another All-to-All

**Key configurations** (Megatron-Core):
```python
num_moe_experts = 64          # Total number of experts
moe_router_topk = 2           # Experts activated per token
expert_model_parallel_size = 8 # GPUs across which experts are distributed
expert_tensor_parallel_size = 1 # TP within each expert
```

**Load balancing**: Token dropping and auxiliary loss terms ensure experts receive roughly equal numbers of tokens. Imbalanced routing leads to GPU idle time.

**Communication pattern**: All-to-All is fundamentally different from all-reduce. It requires GPU i to send unique data to each GPU j, making it bandwidth-intensive and latency-sensitive.

**Kernel implications**:
- Routing kernel must compute top-k expert selection efficiently
- Token permutation/unpermutation kernels rearrange tokens for expert processing
- Grouped GEMM kernels process variable-sized batches per expert efficiently
- DeepEP provides optimized expert parallelism primitives

### 2.8 3D Parallelism Combinations

Modern large-scale training uses combinations of all parallelism dimensions:

```
Total GPUs = DP * TP * PP * CP * EP
```

**Typical hierarchy**:
- **TP** (2-8): Within a node (NVLink, highest bandwidth)
- **PP** (2-16): Across nodes within a rack (low-latency network)
- **DP/FSDP** (remaining): Across racks (tolerates higher latency)
- **CP** (2-8): Overlaid on DP for long sequences
- **EP** (8-64): For MoE expert distribution

**Example**: 512 GPUs, TP=8, PP=4, DP=16 trains a model with 8-way tensor parallelism within each node (8 GPUs/node), 4-stage pipeline across 4 nodes, and 16-way data parallelism.

**Constraint**: All dimensions must evenly divide the total GPU count.

### 2.9 Activation Recomputation Strategies

Activation recomputation (gradient checkpointing) trades compute for memory by recomputing activations during the backward pass instead of storing them.

#### Full Layer Recomputation

Checkpoints the input to each transformer layer. During backward, recomputes the entire layer's forward pass.

- **Memory savings**: Reduces activation memory from O(L * per_layer_activations) to O(L * input_size + 1 * per_layer_activations)
- **Compute overhead**: ~30% increase (re-executing the entire forward for each layer)

#### Selective Activation Checkpointing (SAC)

Fine-grained control over which operations to recompute:

```python
from torch.utils.checkpoint import checkpoint

# Checkpoint only attention blocks, keep MLP activations
for layer in model.layers:
    layer.attention = checkpoint(layer.attention)
```

**Key heuristic**: Recompute **cheap operations** (pointwise ops, activations), cache **expensive operations** (matmuls, attention).

- Attention is ~10x more expensive than feedforward per FLOP
- Some activations are 100MB, others are 1KB
- Optimal policy: checkpoint largest activations of cheapest operations

#### SAC in PyTorch

```python
# Context-function-based SAC: control which ops to recompute
from torch.utils.checkpoint import create_selective_checkpoint_contexts

def policy_fn(ctx, op, *args, **kwargs):
    # Recompute everything except matmuls
    if op in (torch.ops.aten.mm.default, torch.ops.aten.bmm.default):
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE
```

#### Memory-Budget-Based Checkpointing

PyTorch's `torch.compile` includes automatic activation checkpointing that uses a min-cut algorithm (via AOTAutograd) to find the optimal set of activations to save vs recompute, given a memory budget.

**Self-Attention Recomputation** (Megatron-specific):
- Checkpoints inputs of each self-attention block only
- Recomputes intermediate attention activations (QKV projections, softmax)
- High memory savings with minimal recomputation cost (attention intermediates are small relative to savings)

---

## 3. Communication Optimization

### 3.1 Gradient Compression

#### PowerSGD

A practical low-rank gradient compression algorithm that approximates gradient matrices using generalized power iteration:

1. For each gradient matrix G of shape (m, n), compute a low-rank approximation G ≈ P @ Q^T where P is (m, r) and Q is (n, r)
2. Communicate P and Q instead of G, reducing communication from m*n to (m+n)*r
3. Uses **error feedback**: the compression error (G - P @ Q^T) is accumulated and added to the next iteration's gradient

**Implementation via PyTorch DDP hooks**:
```python
from torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook import powerSGD_hook, PowerSGDState

state = PowerSGDState(
    process_group=process_group,
    matrix_approximation_rank=1,     # Rank of approximation
    start_powerSGD_iter=1000,        # Warmup with vanilla allreduce
    min_compression_rate=2,          # Only compress if beneficial
    warm_start=True                  # Reuse Q from previous step
)
model.register_comm_hook(state, powerSGD_hook)
```

**Tuning**: Start `start_powerSGD_iter` at 10% of total training steps. Increase until satisfactory accuracy.

**Kernel implications**:
- Requires per-bucket matrix decomposition kernels (power iteration)
- The approximation must be **allreduce-compatible** (associative): PowerSGD satisfies this
- Compression is most beneficial at small batch sizes and slow networks

#### 1-bit Adam

Exploits the observation that Adam's variance term stabilizes after a warmup phase:

1. **Warmup phase** (first `freeze_step` iterations): Standard Adam with full-precision communication
2. **Compression phase**: Freeze the variance term. Compress the momentum (first moment) to 1-bit per element using error-compensated compression

**Communication savings**: Up to 5x reduction in communication volume.

**Performance**: Up to 3.5x faster training on BERT-Large, 2.7x on SQuAD fine-tuning.

**0/1 Adam** (successor): Provides better efficiency by adaptively choosing between 0-bit (no communication) and 1-bit compression based on gradient similarity across workers.

#### 1-bit LAMB

Combines LAMB's layer-wise adaptive learning rates with 1-bit gradient compression for large-batch distributed training. Uses the same warmup-then-compress strategy as 1-bit Adam.

### 3.2 Communication-Computation Overlap

#### Gradient Bucketing (DDP)

DDP's primary overlap mechanism:
1. Gradients are assigned to buckets (default 25MB) in reverse computation order
2. As the backward pass computes gradients for a bucket, the all-reduce for the **previous** bucket runs concurrently
3. The all-reduce runs on a separate CUDA stream from the compute stream

**Tuning**: Smaller buckets enable earlier overlap start but increase launch overhead. Larger buckets are more communication-efficient. The 25MB default is a good balance.

#### Micro-Pipeline Overlap

FSDP and ZeRO Stage 3 overlap all-gather with forward computation:
1. Before computing layer L, prefetch (all-gather) parameters for layer L
2. While computing layer L, prefetch parameters for layer L+1
3. After computing layer L, release gathered parameters for layer L

This creates a rolling pipeline where communication and computation overlap across layers.

#### Tensor Parallelism Overlap

Communication in TP (all-reduce, reduce-scatter) can be overlapped with computation:
- Split the GEMM into chunks
- After each chunk completes, start communicating its partial result
- Continue computing the next chunk concurrently

### 3.3 NCCL Tuning for Large Clusters

#### Algorithms

NCCL selects from multiple collective algorithms:

- **Ring**: All GPUs form a ring. Data flows around the ring. Optimal bandwidth utilization for large messages. Communication time: `2(N-1)/N * S / BW` where S = message size.
- **Tree**: Hierarchical reduction/broadcast. Lower latency for small messages. Better for non-uniform topologies.
- **CollNet** (network-assisted): Uses in-network compute (e.g., InfiniBand SHARP) for collectives. Can halve communication time for all-reduce.

#### Protocols

- **LL (Low Latency)**: Optimized for small messages, higher overhead per byte
- **LL128**: Balanced approach, uses 128-byte chunks
- **Simple**: Optimized for large messages, maximum bandwidth utilization

NCCL's cost model automatically selects the optimal algorithm-protocol combination based on message size, topology, and GPU count.

#### Key Environment Variables

```bash
# Algorithm override (for benchmarking)
NCCL_ALGO=Ring,Tree,CollNet

# Protocol override
NCCL_PROTO=LL,LL128,Simple

# Buffer size per CTA (default varies)
NCCL_BUFFSIZE=4194304

# Network device assignment policy
NCCL_NETDEVS_POLICY=AUTO

# Cross-NIC communication for multi-NIC systems
NCCL_CROSS_NIC=1    # Use different NICs for cross-node comm

# GPU Direct RDMA control
NCCL_NET_GDR_LEVEL=SYS  # Maximum distance for GDR

# Cross-datacenter optimization
NCCL_SCATTER_XDC=1  # Enable scatter for cross-DC rings

# Channel count tuning
NCCL_NCHANNELS_PER_NET_PEER=4
NCCL_MIN_CTAS=4
```

**General recommendation**: NCCL's automatic tuning is well-optimized and should not need manual overrides in most cases. Use **tuner plugins** for cluster-specific overrides without modifying application code.

### 3.4 Network Topology-Aware Placement

#### Automatic Topology Detection

NCCL automatically detects:
- NVLink topology between GPUs within a node
- PCIe topology and switch hierarchy
- Network device (NIC) proximity to each GPU
- InfiniBand/RoCE switch topology

The detected topology informs algorithm selection, channel assignment, and data routing.

#### Fabric ID and Cross-DC Communication

NCCL supports cross-datacenter communication with topology-aware routing:
- **Fabric ID**: Captures topology information and connectivity between devices
- **Scatter mode**: When `NCCL_SCATTER_XDC=1`, each ring uses two different nodes to cross the DC connection, improving bandwidth utilization

#### GPU-NIC Affinity

For multi-NIC systems, NCCL assigns GPUs to the closest NIC based on PCIe topology:
- `NCCL_NETDEVS_POLICY=AUTO`: Automatic assignment considering bandwidth and topology
- **Rail-optimized networks**: GPUs on the same PCIe rail use the same NIC for inter-node communication

#### Placement Recommendations

- **TP groups**: Same node (NVLink)
- **PP groups**: Adjacent nodes (low latency)
- **DP groups**: Any topology (tolerant of higher latency)
- **EP All-to-All**: Benefits from full-bisection-bandwidth networks

---

## 4. Memory-Efficient Training

### 4.1 Gradient Accumulation

Simulates larger batch sizes by accumulating gradients over multiple micro-batches before applying an optimizer step:

```python
accumulation_steps = 4
for i, (input, target) in enumerate(dataloader):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        output = model(input)
        loss = loss_fn(output, target) / accumulation_steps  # Normalize loss
    loss.backward()  # Gradients accumulate in .grad buffers

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Memory implications**:
- Peak activation memory = single micro-batch (not the full accumulated batch)
- Gradient buffers are the same size regardless of accumulation steps
- Effective batch size = micro_batch_size * accumulation_steps * num_gpus

**Kernel implications**:
- `loss.backward()` **adds** to existing `.grad` tensors (in-place accumulation)
- `optimizer.zero_grad(set_to_none=True)` is preferred: sets `.grad = None` instead of zeroing, avoiding the zero-fill kernel and reducing memory fragmentation
- The `/ accumulation_steps` normalization should be fused with the loss computation kernel

**Synergy with mixed precision**: BF16 reduces activation memory, allowing larger micro-batches, which in turn reduces the number of accumulation steps needed.

**Interaction with DDP**: Gradients should only be synchronized on the accumulation boundary:
```python
if (i + 1) % accumulation_steps != 0:
    with model.no_sync():  # Skip allreduce
        loss.backward()
else:
    loss.backward()  # Allreduce happens here
    optimizer.step()
    optimizer.zero_grad()
```

### 4.2 Activation Checkpointing / Recomputation

See [Section 2.9](#29-activation-recomputation-strategies) for detailed coverage.

**Memory savings summary**:
| Strategy | Memory Reduction | Compute Overhead |
|----------|-----------------|-----------------|
| No checkpointing | 0% | 0% |
| Full layer checkpoint | ~60-70% | ~30-33% |
| Selective (SAC) | ~40-60% | ~10-20% |
| Self-attention only | ~30-50% | ~5-10% |
| torch.compile min-cut | Optimal for budget | Variable |

### 4.3 CPU Offloading

#### ZeRO-Offload (Stage 2)

Offloads **optimizer states and computation** to CPU:
- FP32 master parameters and Adam states reside in CPU memory
- Gradient reduction happens on GPU
- After reduce-scatter, gradients are transferred to CPU
- CPU computes the optimizer step (using DeepSpeedCPUAdam for 5-7x speedup over PyTorch CPU Adam)
- Updated FP16 parameters are sent back to GPU

**Enables**: Training 13B parameter models on a single GPU

**Bandwidth constraint**: PCIe 4.0 x16 provides ~32 GB/s bidirectional. For a 10B model with 40GB of optimizer state, the transfer takes ~1.25 seconds per step.

#### ZeRO-Infinity (Stage 3)

Extends offloading to **NVMe SSDs** in addition to CPU:
- Parameters, gradients, and optimizer states can all be offloaded
- NVMe provides ~3-5 GB/s per drive (can use multiple drives)
- Enables training **trillion-parameter** models on limited GPU clusters

**Key optimizations**:
- **Bandwidth-centric partitioning**: Data placement is optimized for available PCIe and NVMe bandwidth
- **Overlap-centric design**: CPU computation overlaps with GPU-to-CPU and NVMe-to-CPU data transfers
- **Infinity offload engine**: Manages the memory hierarchy (GPU HBM > CPU DRAM > NVMe)

### 4.4 Memory-Efficient Optimizers

#### Standard Adam Memory Footprint

For mixed-precision training of a model with P parameters:
- FP16 parameters: 2P bytes
- FP16 gradients: 2P bytes
- FP32 master parameters: 4P bytes
- FP32 first moment (m): 4P bytes
- FP32 second moment (v): 4P bytes
- **Total: 16P bytes** (e.g., 48GB for a 3B model)

#### 8-bit Adam (bitsandbytes)

Quantizes optimizer states to 8-bit:

```python
import bitsandbytes as bnb
optimizer = bnb.optim.Adam8bit(model.parameters(), lr=1e-3)
```

**How it works**:
1. First and second moments are stored in 8-bit using **block-wise quantization**
2. Each block of values shares a single FP32 scaling factor
3. Dynamic quantization maps the full range of each block to 8-bit values
4. Dequantization happens on-the-fly during the optimizer step

**Memory savings**: 8-bit m + 8-bit v + FP32 master = ~6P bytes (vs 12P for Adam states alone).

**Kernel implications**:
- Block-wise quantize/dequantize kernels for each optimizer state update
- The quantization error is small enough not to affect convergence in practice
- Custom CUDA kernels handle the 8-bit arithmetic

#### Adafactor

Factorizes the second moment matrix into row and column factors:

Instead of storing the full second moment matrix v of shape (m, n), stores:
- Row factor r of shape (m,): `r_i = (1/n) * sum_j(v_ij)`
- Column factor c of shape (n,): `c_j = (1/m) * sum_i(v_ij)`
- Reconstructed: `v_ij ≈ r_i * c_j / mean(r)`

**Memory**: O(m + n) instead of O(m * n) -- dramatic savings for large embedding and vocabulary matrices.

**Trade-off**: Slower convergence than Adam in some cases. No first moment by default (uses sign of gradient).

#### CAME (Confidence-guided Adaptive Memory Efficient Optimization)

Combines Adafactor's memory efficiency with Adam's convergence speed:

- Uses the same low-rank decomposition as Adafactor for the second moment
- Adds a **confidence matrix** that guides updates based on the reliability of the low-rank approximation
- Achieves Adam-level convergence with Adafactor-level memory usage

**Memory**: Same as Adafactor (~O(m + n) for second moments).

### 4.5 Flash Attention for Memory Savings in Training

FlashAttention dramatically reduces attention memory from O(N^2) to O(N):

**Standard attention memory**: Must store S = Q @ K^T (N x N) and P = softmax(S) (N x N), consuming O(N^2) memory.

**FlashAttention memory**: Uses tiling and recomputation:
1. **Forward**: Processes Q, K, V in tiles, never materializing the full N x N matrices. Saves only the output O and softmax normalization statistics (logsumexp per row, O(N)).
2. **Backward**: Recomputes S and P from the saved statistics and original Q, K, V. The recomputation is faster than reading from HBM because data is loaded in tiles that fit in SRAM.

**Memory savings**: 10x at sequence length 2K, 20x at 4K (proportional to sequence length).

**Speed improvement**: Up to 7.6x on GPT-2 due to reduced HBM reads/writes, despite the extra recomputation FLOPs.

**Kernel design**:
- Forward: Fused tiled GEMM + online softmax + output accumulation in one kernel
- Backward: Fused tiled GEMM + softmax recomputation + gradient computation
- Key optimization: IO-awareness -- minimize HBM reads/writes by keeping all intermediates in SRAM
- FlashAttention-2: Better work partitioning between warps within thread blocks
- FlashAttention-3: Asynchronous pipelining and FP8 support on Hopper
- FlashAttention-4: Algorithm-kernel pipelining co-design for asymmetric hardware

---

## 5. Fine-Tuning Optimization

### 5.1 LoRA (Low-Rank Adaptation)

#### Core Concept

Freezes pretrained weights W and adds trainable low-rank decomposition:

```
W_new = W + alpha * (B @ A)
```

Where:
- W: Original weight matrix (frozen), shape (d, k)
- A: Down-projection, shape (r, k), initialized with Kaiming uniform
- B: Up-projection, shape (d, r), initialized with zeros
- r: Rank (typically 4-64, much smaller than d, k)
- alpha: Scaling factor (alpha/r)

**Trainable parameters**: 2 * r * (d + k) per adapted layer vs d * k for full fine-tuning. With r=8 on a 4096x4096 weight: 65K params vs 16.7M (0.39%).

#### Kernel Implications

- **Forward pass**: `y = W @ x + (alpha/r) * B @ (A @ x)`. The LoRA path is two small GEMMs.
- **Fused kernel opportunity**: The LoRA computation can be fused with the base model's linear operation: compute `W @ x` and `B @ (A @ x)` simultaneously, adding results.
- **Batch efficiency**: LoRA's small matrices (r << d) can underutilize Tensor Cores. Batching multiple LoRA computations or using specialized small-GEMM kernels helps.
- **Memory**: Only A, B matrices and their gradients are in GPU memory. W remains frozen (can be quantized).

### 5.2 QLoRA

Combines LoRA with 4-bit quantization of base model weights:

1. Base model weights quantized to NF4 (Normal Float 4-bit) with double quantization
2. LoRA adapters added in BF16
3. Computation: dequantize W to BF16, compute forward, LoRA path in BF16
4. Only LoRA parameters receive gradients

**Memory breakdown** for a 70B model:
- NF4 weights: ~35GB (0.5 bytes/param)
- LoRA adapters (r=64): ~0.5GB
- Gradients + optimizer states for LoRA: ~2GB
- Activations: ~10-15GB
- **Total**: ~50-55GB (fits on single A100 80GB)

**Kernel implications**:
- **Dequantization kernel**: NF4 -> BF16 conversion must be fast. Uses lookup table (16 entries for 4-bit) with block-wise scaling factors.
- **Fused dequant-GEMM**: Ideally dequantize on-the-fly during the GEMM, avoiding materialization of the full BF16 weight matrix.
- **Double quantization**: The FP32 scaling factors are themselves quantized to FP8, adding a two-level dequantization path.

### 5.3 Adapter Methods

Insert small trainable modules between frozen transformer layers:

```
                    +--------+
Input -> FrozenLayer -> | Adapter | -> Output
                    +--------+
                    |  Down   |  (d -> r)
                    |  Act    |  (nonlinearity)
                    |  Up     |  (r -> d)
                    +--------+
```

- Down-projection: d -> bottleneck_size (small linear)
- Nonlinearity: ReLU or similar
- Up-projection: bottleneck_size -> d

**Trainable parameters**: ~3.6% of full model for BERT-equivalent performance.

**Kernel implications**: Two small linear layers with activation. Can be fused into a single kernel. The serial dependency (down -> act -> up) limits parallelism within the adapter.

### 5.4 Prefix Tuning

Prepends learned continuous vectors ("prefixes") to the key and value sequences at each transformer layer:

```
K_new = concat(K_prefix, K)   # K_prefix is learnable, shape (prefix_len, d)
V_new = concat(V_prefix, V)   # V_prefix is learnable
```

**Trainable parameters**: `2 * num_layers * prefix_len * d_model`. With prefix_len=20 on a 24-layer model with d=1024: ~1M params.

**Kernel implications**:
- Attention kernel must handle prepended prefix tokens (increases effective sequence length)
- Memory cost is small (prefix is shared across batch)
- Gradient only flows through prefix parameters

### 5.5 Unsloth Optimizations

Unsloth achieves 2-2.7x faster fine-tuning with 74% less memory through:

1. **Hand-written Triton kernels**: Manually derives backpropagation steps and rewrites PyTorch modules into optimized Triton kernels
2. **Custom backward pass**: Instead of relying on autograd, manually implements backward passes for key operations, eliminating intermediate tensor storage
3. **Weight tying optimization**: Shares computation between tied weights
4. **Ultra-low precision**: Dynamic quantization down to 1.58 bits, intelligently choosing which parameters to keep at higher precision

**Specific kernel optimizations**:
- Fused RoPE + attention kernel
- Fused cross-entropy with label smoothing
- In-place operations to avoid tensor copies
- Custom memory layout for adapter weights

**Integration**: Compatible with HuggingFace PEFT, TRL, and Transformers.

### 5.6 PEFT Memory and Compute Analysis

| Method | Trainable Params | Memory vs Full FT | Training Speed | Accuracy vs Full FT |
|--------|-----------------|-------------------|----------------|---------------------|
| Full Fine-Tuning | 100% | 1.0x | 1.0x | Baseline |
| LoRA (r=8) | 0.1-2% | 0.3-0.5x | 0.5-0.8x | 95-100% |
| QLoRA (r=8, 4-bit) | 0.1-2% | 0.15-0.25x | 0.4-0.7x | 93-99% |
| Adapters | 2-5% | 0.4-0.6x | 0.6-0.85x | 95-100% |
| Prefix Tuning | <1% | 0.3-0.5x | 0.7-0.9x | 90-97% |
| LoRA + Unsloth | 0.1-2% | 0.08-0.15x | 1.5-2.7x vs LoRA | 95-100% |

**UniPELT**: Integrates LoRA, prefix tuning, and adapters with a learned gating mechanism per method per layer. The gating mechanism uses three small FFNs that produce scalar gates, allowing the model to learn which PEFT method is most useful for each layer.

---

## 6. Optimizer Kernels

### 6.1 Fused Adam Optimizer

#### NVIDIA Apex FusedAdam

Two key fusions:

1. **Elementwise fusion**: All Adam update operations (momentum update, variance update, bias correction, parameter update) combined into a single kernel instead of 5-8 separate kernels.

2. **Multi-tensor apply**: Batches the elementwise updates for **all model parameters** into one or a few kernel launches, eliminating per-parameter kernel launch overhead.

```python
from apex.optimizers import FusedAdam
optimizer = FusedAdam(model.parameters(), lr=1e-3, adam_w_mode=True)
```

**Standard Adam** (unfused) for a single parameter:
```
# 7 separate kernels:
m = beta1 * m + (1 - beta1) * grad
v = beta2 * v + (1 - beta2) * grad^2
m_hat = m / (1 - beta1^t)
v_hat = v / (1 - beta2^t)
update = m_hat / (sqrt(v_hat) + eps)
param = param - lr * update
param = param - lr * weight_decay * param  # AdamW
```

**FusedAdam**: All 7 operations in a single kernel per parameter, or batched across all parameters with multi-tensor apply.

#### DeepSpeed FusedAdam

Similar fusion to Apex but with additional features:
- Direct integration with ZeRO stages
- Mixed-precision support (FP16/BF16 parameters, FP32 states)
- CPU Adam variant (DeepSpeedCPUAdam) for ZeRO-Offload with 5-7x speedup over PyTorch CPU Adam

### 6.2 8-bit Optimizers (bitsandbytes)

#### Block-wise Quantization

The core technique for 8-bit optimizer states:

1. **Block partitioning**: Divide optimizer state tensor into blocks of B elements (typically B=2048)
2. **Per-block scaling**: Compute the absolute maximum of each block
3. **Quantization**: Map each block's values to 8-bit using the block max as scale factor
4. **Dynamic data type**: Uses non-uniform quantization bins optimized for the distribution of optimizer states

```python
import bitsandbytes as bnb

# Drop-in replacement for Adam
optimizer = bnb.optim.Adam8bit(model.parameters(), lr=1e-3)

# Also available: AdamW, LAMB, Lars, SGD in 8-bit
optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-3)
```

**Kernel implementation**:
- Custom CUDA kernels for block-wise quantize/dequantize
- Fused optimizer step: dequantize -> update -> requantize in single kernel
- The quantization error is stochastic and averages out over training

**Memory**: Adam states go from 8 bytes/param (FP32 m + FP32 v) to 2 bytes/param (8-bit m + 8-bit v) + small overhead for block scaling factors.

### 6.3 LAMB Optimizer

LAMB (Layer-wise Adaptive Moments optimizer for Batch training) enables training with very large batch sizes (up to 64K) by using layer-wise learning rate scaling:

```
# Standard Adam update
m = beta1 * m + (1 - beta1) * grad
v = beta2 * v + (1 - beta2) * grad^2
update = m / (sqrt(v) + eps) + weight_decay * param

# LAMB scaling
r1 = norm(param)
r2 = norm(update)
trust_ratio = r1 / r2 if r1 > 0 and r2 > 0 else 1.0
param = param - lr * trust_ratio * update
```

#### DeepSpeed FusedLamb

```python
from deepspeed.ops.lamb import FusedLamb
optimizer = FusedLamb(model.parameters(), lr=1e-3)
```

**Kernel fusion**: Same two-level fusion as FusedAdam (elementwise + multi-tensor apply), plus the additional trust ratio computation (requires per-layer norm reduction kernels).

**Key parameters**:
- `max_grad_norm`: Global gradient clipping (requires a separate all-reduce for global norm)
- `max_coeff` / `min_coeff`: Bounds on the trust ratio (10.0 / 0.01 defaults)

### 6.4 Kernel Fusion for Optimizer Step

#### Liger Kernel Approach

Liger Kernel extends fusion beyond just the optimizer to the entire training loop:

**Available fused kernels for training**:
- **FusedLinearCrossEntropy**: Combines the final linear projection and cross-entropy loss into a single kernel. Processes in chunks to avoid materializing the full logit tensor.
  - 3x faster, 5x less memory for vocab_size=163840
- **RMSNorm**: Fuses normalization and scaling, caches RMS for backward
- **SwiGLU**: Recomputes SiLU/GELU during backward to save 1.6x memory
- **RoPE**: Fused rotary position embedding application
- **Multi-Token Prediction**: Fused loss for speculative decoding training

**Overall impact**: 20% faster multi-GPU training, 60% memory reduction, 80% memory savings for alignment/post-training losses.

**Framework compatibility**: Works with FlashAttention, FSDP, DeepSpeed, HuggingFace Trainer.

#### General Fusion Strategies for Training

1. **Operator fusion**: Combine adjacent elementwise ops (e.g., bias + activation + dropout)
2. **Epilogue fusion**: Fuse operations after GEMM (bias add, activation, residual add)
3. **Prologue fusion**: Fuse operations before GEMM (dequantization, scaling)
4. **Cross-layer fusion**: FusedLinearCrossEntropy spans two layers
5. **Backward pass fusion**: Compute gradients for multiple operations in a single kernel
6. **In-place operations**: Overwrite input tensors with gradients to save memory

---

## 7. Data Pipeline Optimization

### 7.1 GPU-Accelerated Data Loading (NVIDIA DALI)

DALI offloads data preprocessing from CPU to GPU, eliminating the CPU bottleneck:

**Architecture**:
- Defines a data pipeline as a graph of operations
- Operations execute on GPU using optimized CUDA kernels
- Built-in prefetching, parallel execution, and batch processing
- Asynchronous execution: batches are prepared before they are requested

**Key capabilities**:
- Image decode (JPEG, PNG) on GPU using nvJPEG
- Augmentation operations (crop, resize, color jitter) on GPU
- Normalization and format conversion on GPU
- Video decode and audio processing

**Integration**:
```python
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator

@pipeline_def
def training_pipeline():
    images, labels = fn.readers.file(file_root=data_dir)
    images = fn.decoders.image(images, device="mixed")   # Decode on GPU
    images = fn.resize(images, size=(224, 224))            # Resize on GPU
    images = fn.crop_mirror_normalize(images, ...)         # Normalize on GPU
    return images, labels
```

**Performance impact**: Eliminates CPU data loading bottleneck. Particularly important when using fast GPUs (H100, B200) where the GPU compute is so fast that CPU-based data loading cannot keep up.

### 7.2 CPU-GPU Prefetching

#### PyTorch DataLoader Prefetching

```python
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=8,          # CPU workers for parallel data loading
    pin_memory=True,        # Pin CPU memory for faster D2H transfer
    prefetch_factor=2,      # Number of batches to prefetch per worker
    persistent_workers=True  # Keep workers alive between epochs
)
```

**Pinned memory**: Allocates page-locked (pinned) host memory, enabling DMA transfers to GPU without staging through pageable memory. Provides 2-3x faster H2D transfer.

**Non-blocking transfers**:
```python
for batch in dataloader:
    # Transfer to GPU asynchronously
    input = batch[0].to(device, non_blocking=True)
    target = batch[1].to(device, non_blocking=True)
    # Compute overlaps with transfer of next batch
    output = model(input)
```

#### Custom Prefetcher

```python
class CUDAPrefetcher:
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return
        with torch.cuda.stream(self.stream):
            for k in self.next_batch:
                self.next_batch[k] = self.next_batch[k].to(self.device, non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch
```

### 7.3 Data Format Optimization

#### WebDataset

Stores training data in standard TAR archives:
- Sequential access pattern (ideal for spinning disks and network storage)
- No random access overhead
- Sharding is natural (one TAR per shard)
- DALI has native WebDataset reader support

```python
import webdataset as wds

dataset = (
    wds.WebDataset("data-{000..099}.tar")
    .shuffle(1000)
    .decode("pil")
    .to_tuple("input.jpg", "target.cls")
    .map_tuple(transform, identity)
)
```

#### Mosaic StreamingDataset (MDS Format)

Purpose-built for distributed large-scale training:

**MDS format characteristics**:
- Row-based storage with shard files + index metadata
- Fast random-access via header-based sample lookup
- Ultra-low sample retrieval latency
- Deterministic sample ordering across any number of GPUs/nodes

**Key features**:
- **Deterministic shuffling**: Same order regardless of GPU count, enabling reproducible training
- **Elastic resumption**: Resume training exactly where it left off without replaying data
- **No duplicate data**: Deterministic shard assignment avoids workers processing the same data
- **Cloud-native**: Streams efficiently from S3, GCS, Azure Blob

```python
from streaming import StreamingDataset, StreamingDataLoader

dataset = StreamingDataset(
    remote='s3://bucket/dataset',
    local='/tmp/cache',
    shuffle=True,
    batch_size=32
)
dataloader = StreamingDataLoader(dataset, batch_size=32)
```

**Performance**: Model convergence matches local-disk training thanks to high-quality shuffling within and across shards.

### 7.4 Tokenizer Optimization

For language model training, tokenization can become a bottleneck:

**Strategies**:
- **Pre-tokenization**: Tokenize the entire dataset offline and store token IDs directly. Eliminates runtime tokenization cost.
- **Parallel tokenization**: Use multiple CPU workers for on-the-fly tokenization with `num_workers` in DataLoader.
- **Rust-based tokenizers**: HuggingFace `tokenizers` library uses Rust for 10-100x speedup over pure Python.
- **Batched tokenization**: Process multiple sequences simultaneously to amortize overhead.

**For MDS format**: Pre-tokenize and store as integer arrays in MDS shards. This eliminates tokenization entirely during training.

---

## 8. Compilation for Training

### 8.1 torch.compile for Training

`torch.compile` is PyTorch's JIT compiler that captures and optimizes computation graphs:

```python
model = torch.compile(model, mode="reduce-overhead")  # or "max-autotune"
```

**Modes for training**:
- `"default"`: Balanced compilation time and performance
- `"reduce-overhead"`: Minimizes kernel launch overhead via CUDA graphs
- `"max-autotune"`: Tries all available kernels (cuBLAS, CUTLASS, Triton) and selects the fastest

**Typical speedup**: 1.5-2x over eager mode for training.

**How it works for training**:
1. **Dynamo**: Captures the forward pass as a graph by intercepting Python bytecode
2. **AOTAutograd**: Traces the backward pass ahead-of-time, producing a backward graph
3. **Inductor**: Compiles both forward and backward graphs into optimized GPU code

### 8.2 Graph Breaks in Training

Graph breaks interrupt compilation, forcing a fallback to eager mode for the offending operation. Common causes in training:

1. **Data-dependent control flow**: `if tensor.item() > 0` forces materialization
2. **Unsupported operations**: Some custom CUDA extensions, certain autograd functions
3. **Dynamic shapes**: Tensors changing shape between iterations (e.g., variable-length sequences)
4. **In-place mutations**: Certain in-place operations on tensors that require gradient tracking
5. **Print statements**: `print(tensor)` forces evaluation
6. **Distributed collectives**: Most NCCL operations (all-reduce, all-gather) are not captured in the graph by default
7. **Custom autograd functions**: Must use `torch.autograd.Function` with `setup_context` for compatibility

**Impact**: Each graph break creates a separate compiled region. The regions before and after the break are still compiled, but the break itself and any inter-region transitions run in eager mode with kernel launch overhead.

**Detection**:
```python
# Log graph breaks
torch._dynamo.config.log_level = logging.DEBUG
# Or force full-graph compilation (error on break)
model = torch.compile(model, fullgraph=True)
```

### 8.3 Compiled Autograd

Compiled Autograd (introduced PyTorch 2.4) captures the **full backward graph at runtime**, addressing limitations of AOTAutograd:

**Problem with AOTAutograd**:
- Graph breaks in forward lead to graph breaks in backward
- Backward hooks are not captured
- DDP/FSDP communication is outside the compiled region

**How Compiled Autograd solves this**:
1. When `loss.backward()` is called, Compiled Autograd intercepts the autograd engine
2. Records the complete backward pass as a single graph
3. Compiles the backward graph with `torch.compile` in inference mode
4. The full backward graph can be optimized holistically

**Usage**:
```python
torch._dynamo.config.compiled_autograd = True

# Or with separate backward options:
with torch._dynamo.compiled_autograd.enable(
    torch.compile(backend="inductor", fullgraph=True)
):
    loss.backward()
```

**Advantages**:
- Forward graph breaks no longer fragment the backward graph
- Backward hooks are captured and compiled
- Larger optimization scope for the compiler

**Disadvantages**:
- Runtime overhead for cache lookup at backward start
- More prone to recompilation (larger graph = more cache miss triggers)
- Not all autograd operations are supported yet

**Recompilation triggers**:
- Autograd structure changes (different ops produce different gradient histories)
- Shape variations (triggers dynamic shape marking)

### 8.4 Training-Specific Inductor Optimizations

The Inductor backend provides several training-relevant optimizations:

#### Pointwise and Reduction Fusions

Inductor automatically fuses chains of elementwise operations (activations, normalization, dropout) and reduction operations (mean, sum, softmax) into single Triton kernels, eliminating intermediate memory allocations.

#### Matmul Backend Autotuning

In `max-autotune` mode, Inductor benchmarks multiple GEMM implementations:
- cuBLAS (NVIDIA's hand-tuned library)
- CUTLASS (template-based GEMM library)
- Triton (auto-generated kernels)

The fastest implementation is cached and used for subsequent calls.

#### CUDA Graph Integration

In `reduce-overhead` mode, Inductor captures the compiled graph as a CUDA graph, reducing kernel launch overhead. The latest versions include improved "soundness guarantees" ensuring correctness.

#### Automatic Activation Checkpointing

Inductor can automatically determine the optimal set of activations to checkpoint using a min-cut algorithm, considering:
- Tensor sizes (save large tensors)
- Operation costs (recompute cheap operations)
- Global memory budget

This provides theoretically optimal memory-compute trade-off without manual checkpoint placement.

#### DTensor Support

The DTensor abstraction supports SPMD (Single Program Multiple Data) operations, enabling Inductor to optimize distributed computations within the compiled graph.

#### Regional Compilation

For large models, compile only the hot path (e.g., individual transformer blocks) rather than the entire model:

```python
for layer in model.layers:
    layer = torch.compile(layer)  # Compile each block separately
```

**Benefits**: Faster compilation time, fewer graph breaks, more predictable performance.

### 8.5 Known Limitations for Training Compilation

| Limitation | Impact | Workaround |
|-----------|--------|-----------|
| NOT bitwise equivalent with eager | Fused ops have different precision | Accept for training, verify for eval |
| Double backward unsupported | Cannot compute second-order gradients | Use eager for Hessian computation |
| Dynamic shapes | Caching overhead, recompilation | Use static shapes where possible |
| Distributed collectives | Not optimized by default | Use functional collectives (limited autograd) |
| Tensor subclasses | Require specific support | Check compatibility per subclass |
| Compilation time | Minutes for large models | Use regional compilation |
| Memory during compilation | Tracing requires significant RAM | Compile on CPU-rich nodes |

---

## Sources

### Mixed Precision Training
- [How can using FP16, BF16, or FP8 mixed precision speed up model training?](https://www.runpod.io/articles/guides/fp16-bf16-fp8-mixed-precision-speed-up-my-model-training)
- [Floating-Point 8: An Introduction to Efficient, Lower-Precision AI Training](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/)
- [Mixed Precision Training in LLMs: FP16, BF16, FP8, and Beyond](https://medium.com/@dpratishraj7991/mixed-precision-training-in-llms-fp16-bf16-fp8-and-beyond-b4af13ca846f)
- [What Every User Should Know About Mixed Precision Training in PyTorch](https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/)
- [Using FP8 and FP4 with Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
- [Per-Tensor and Per-Block Scaling Strategies for Effective FP8 Training](https://developer.nvidia.com/blog/per-tensor-and-per-block-scaling-strategies-for-effective-fp8-training/)
- [Accelerating AI Training with NVIDIA TF32 Tensor Cores](https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/)
- [Automatic Mixed Precision examples (PyTorch)](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)
- [PyTorch AMP Documentation](https://docs.pytorch.org/docs/stable/amp.html)

### Distributed Training
- [Getting Started with FSDP2 (PyTorch)](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [Parallelisms Guide -- Megatron Bridge](https://docs.nvidia.com/nemo/megatron-bridge/0.2.0/parallelisms.html)
- [ZeRO Optimization Strategies for Large-Scale Model Training](https://huggingface.co/blog/josh-a/zero-optimization-strategies)
- [ZeRO Redundancy Optimizer Tutorial (DeepSpeed)](https://www.deepspeed.ai/tutorials/zero/)
- [ZeRO-Offload Tutorial (DeepSpeed)](https://www.deepspeed.ai/tutorials/zero-offload/)
- [Zero Bubble Pipeline Parallelism](https://arxiv.org/html/2401.10241v1)
- [Pipeline Parallelism (PyTorch)](https://docs.pytorch.org/docs/stable/distributed.pipelining.html)
- [Scaling LLM Inference: Innovations in TP, CP, and EP (Meta)](https://engineering.fb.com/2025/10/17/ai-research/scaling-llm-inference-innovations-tensor-parallelism-context-parallelism-expert-parallelism/)
- [Paradigms of Parallelism (Colossal-AI)](https://colossalai.org/docs/concepts/paradigms_of_parallelism/)
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://people.eecs.berkeley.edu/~matei/papers/2021/sc_megatron_lm.pdf)

### Communication Optimization
- [PowerSGD: Practical Low-Rank Gradient Compression](https://ar5iv.labs.arxiv.org/html/1905.13727)
- [DDP Communication Hooks (PyTorch)](https://docs.pytorch.org/docs/stable/ddp_comm_hooks.html)
- [1-bit Adam: Communication Efficient Large-Scale Training (DeepSpeed)](https://www.deepspeed.ai/tutorials/onebit-adam/)
- [0/1 Adam (DeepSpeed)](https://www.deepspeed.ai/tutorials/zero-one-adam/)
- [Understanding NCCL Tuning to Accelerate GPU-to-GPU Communication](https://developer.nvidia.com/blog/understanding-nccl-tuning-to-accelerate-gpu-to-gpu-communication)
- [NCCL Deep Dive: Cross Data Center Communication and Network Topology Awareness](https://developer.nvidia.com/blog/nccl-deep-dive-cross-data-center-communication-and-network-topology-awareness)
- [NCCL Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)

### Memory-Efficient Training
- [Current and New Activation Checkpointing Techniques in PyTorch](https://pytorch.org/blog/activation-checkpointing-techniques/)
- [Activation Recomputation (Megatron Bridge)](https://docs.nvidia.com/nemo/megatron-bridge/0.2.0/training/activation-recomputation.html)
- [ZeRO-Infinity: Breaking the GPU Memory Wall](https://arxiv.org/pdf/2104.07857)
- [8-bit Optimizers via Block-wise Quantization](https://arxiv.org/pdf/2110.02861)
- [CAME: Confidence-guided Adaptive Memory Efficient Optimization](https://arxiv.org/pdf/2307.02047)
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/pdf/2205.14135)
- [FlashAttention-2](https://tridao.me/publications/flash2/flash2.pdf)
- [FlashAttention-3](https://tridao.me/publications/flash3/flash3.pdf)
- [FlashAttention-4](https://research.colfax-intl.com/flashattention-4-algorithm-and-kernel-pipelining-co-design-for-asymmetric-hardware-scaling/)
- [Flash Attention GitHub](https://github.com/Dao-AILab/flash-attention)

### Fine-Tuning Optimization
- [Fine-Tuning Infrastructure: LoRA, QLoRA, and PEFT at Scale](https://introl.com/blog/fine-tuning-infrastructure-lora-qlora-peft-scale-guide-2025)
- [Make LLM Fine-tuning 2x faster with Unsloth and TRL](https://huggingface.co/blog/unsloth-trl)
- [PEFT: State-of-the-art Parameter-Efficient Fine-Tuning (GitHub)](https://github.com/huggingface/peft)
- [Understanding Parameter-Efficient Finetuning: From Prefix Tuning to LLaMA-Adapters](https://lightning.ai/pages/community/article/understanding-llama-adapters/)

### Optimizer Kernels
- [DeepSpeed Optimizers Documentation](https://deepspeed.readthedocs.io/en/latest/optimizers.html)
- [NVIDIA Apex Optimizers](https://nvidia.github.io/apex/optimizers.html)
- [Liger Kernel: Efficient Triton Kernels for LLM Training (GitHub)](https://github.com/linkedin/Liger-Kernel)
- [Liger Kernel Paper](https://arxiv.org/pdf/2410.10989)
- [bitsandbytes 8-bit Optimizers](https://huggingface.co/docs/bitsandbytes/optimizers)

### Data Pipeline Optimization
- [NVIDIA DALI Documentation](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/index.html)
- [NVIDIA DALI GitHub](https://github.com/NVIDIA/DALI)
- [Mosaic StreamingDataset (GitHub)](https://github.com/mosaicml/streaming)
- [MosaicML StreamingDataset Blog](https://www.databricks.com/blog/mosaicml-streamingdataset)

### Compilation for Training
- [State of torch.compile for Training (August 2025)](https://blog.ezyang.com/2025/08/state-of-torch-compile-august-2025/)
- [Compiled Autograd Tutorial (PyTorch)](https://docs.pytorch.org/tutorials/intermediate/compiled_autograd_tutorial.html)
- [torch.compile Documentation](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
- [torch.compile FAQ](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_faq.html)
