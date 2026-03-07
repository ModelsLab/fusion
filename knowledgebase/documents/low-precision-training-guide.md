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
| FP8 E4M3 | 4 bits | 3 bits | +/-448 | Forward pass (Ada/Hopper/Blackwell) |
| FP8 E5M2 | 5 bits | 2 bits | +/-57344 | Backward pass (Ada/Hopper/Blackwell) |

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

## FP8 Training (Ada/Hopper/Blackwell)

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

## Practical Training Recipes

### Recipe 1: BF16 Fine-Tuning (Simplest)

The most straightforward approach for Ampere+ GPUs. No quantization libraries needed.

```python
# bf16_finetune.py — copy-paste ready
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="bfloat16", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
dataset = load_dataset("tatsu-lab/alpaca", split="train[:5000]")

def tokenize(example):
    return tokenizer(example["text"], truncation=True, max_length=512, padding="max_length")

dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

trainer = Trainer(
    model=model,
    train_dataset=dataset,
    args=TrainingArguments(
        output_dir="./bf16-ft", bf16=True, per_device_train_batch_size=4,
        gradient_accumulation_steps=4, num_train_epochs=1, learning_rate=2e-5,
        logging_steps=10, save_steps=500, optim="adamw_torch",
    ),
)
trainer.train()
```

**Expected resource usage (7B model, A100 80GB):**
- Peak memory: ~58 GB (full fine-tune with AdamW optimizer states)
- Training speed: ~3200 tokens/sec on A100-80GB
- BF16 halves activation/gradient memory vs FP32 but optimizer states remain FP32

**Expected resource usage (7B model, RTX 3090 24GB):**
- Will OOM on full fine-tune — use gradient checkpointing + batch size 1, or switch to LoRA/QLoRA

### Recipe 2: QLoRA Fine-Tuning (Most Memory Efficient)

Run a 70B model on a single 24GB GPU using NF4 quantization + LoRA adapters.

```bash
pip install transformers accelerate peft bitsandbytes datasets trl
```

```python
# qlora_finetune.py — 70B on a single 24GB GPU
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer

model_name = "meta-llama/Llama-2-70b-hf"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,  # nested quantization saves ~0.4 bits/param
)

model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=bnb_config, device_map="auto",
)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=64, lora_alpha=16, lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # expect ~0.1-0.2% of total params

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
dataset = load_dataset("tatsu-lab/alpaca", split="train")

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=TrainingArguments(
        output_dir="./qlora-70b", bf16=True, per_device_train_batch_size=1,
        gradient_accumulation_steps=16, num_train_epochs=1, learning_rate=2e-4,
        logging_steps=10, save_steps=200, optim="paged_adamw_8bit",
        max_grad_norm=0.3, warmup_ratio=0.03, lr_scheduler_type="cosine",
        gradient_checkpointing=True,
    ),
    max_seq_length=512,
)
trainer.train()
```

**Expected resource usage (70B model, RTX 3090/4090 24GB):**
- Model weights: ~35 GB in NF4 with double quantization (~0.55 bytes/param)
- Actual peak VRAM: ~21-23 GB (with paged optimizer offload, gradient checkpointing, batch size 1, seq_len 512)
- Training speed: ~150-250 tokens/sec on RTX 4090
- For 80B+ models: use `max_memory={0: "22GiB", "cpu": "60GiB"}` in `from_pretrained`

**Common errors and fixes:**

| Error | Cause | Fix |
|-------|-------|-----|
| `CUDA out of memory` | Batch size too large or seq_len too long | Set `per_device_train_batch_size=1`, reduce `max_seq_length`, enable `gradient_checkpointing=True` |
| `ValueError: Target modules not found` | Model architecture uses different layer names | Inspect `model.named_modules()` and update `target_modules` list |
| `RuntimeError: expected scalar type BFloat16` | Compute dtype mismatch | Set `bnb_4bit_compute_dtype=torch.bfloat16` in `BitsAndBytesConfig` |
| `bitsandbytes not compiled with GPU support` | Missing CUDA toolkit or wrong bitsandbytes version | `pip install bitsandbytes --upgrade` or install CUDA toolkit matching your driver |
| `Tokenizer has no pad_token` | Missing padding token | Add `tokenizer.pad_token = tokenizer.eos_token` |
| `NaN loss at step 1` | Learning rate too high for QLoRA | Use `2e-4` or lower; add warmup with `warmup_ratio=0.03` |
| `Training runs but loss does not decrease` | LoRA rank too low or wrong target modules | Increase `r` to 64-128 and target all linear layers |
| `CUDA error: device-side assert triggered` | Label values out of vocabulary range | Check dataset for labels >= `vocab_size`; ensure `eos_token` is set correctly |

### Recipe 3: FP8 Training with Transformer Engine (Ada/Hopper/Blackwell GPUs)

Native FP8 training for H100/H200/B200 GPUs using NVIDIA Transformer Engine.

```bash
pip install transformer-engine[pytorch]
# Requires: CUDA 12.1+, PyTorch 2.1+, Hopper or newer GPU
```

```python
# fp8_training.py — FP8 training loop with Transformer Engine
import torch
import torch.nn as nn
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling

# Define model with TE layers (drop-in replacements for nn.Linear / nn.LayerNorm)
class FP8TransformerBlock(nn.Module):
    def __init__(self, hidden_size=4096, ffn_size=11008):
        super().__init__()
        self.ln1 = te.LayerNorm(hidden_size)
        self.attn_qkv = te.Linear(hidden_size, hidden_size * 3, bias=False)
        self.attn_out = te.Linear(hidden_size, hidden_size, bias=False)
        self.ln2 = te.LayerNorm(hidden_size)
        self.ffn_up = te.Linear(hidden_size, ffn_size, bias=False)
        self.ffn_down = te.Linear(ffn_size, hidden_size, bias=False)

    def forward(self, x):
        # Self-attention (simplified, no masking shown)
        h = self.ln1(x)
        qkv = self.attn_qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = x + self.attn_out(attn)
        # FFN
        h = self.ln2(x)
        x = x + self.ffn_down(torch.nn.functional.silu(self.ffn_up(h)))
        return x

model = FP8TransformerBlock().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# FP8 recipe: delayed scaling with amax history of 16 steps
fp8_recipe = DelayedScaling(
    fp8_format=Format.HYBRID,  # E4M3 forward, E5M2 backward
    amax_history_len=16,
    amax_compute_algo="max",
)

# Training loop
for step, (inputs, targets) in enumerate(dataloader):
    inputs, targets = inputs.cuda(), targets.cuda()
    optimizer.zero_grad()

    # fp8_autocast handles quantization/dequantization automatically
    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        output = model(inputs)

    loss = torch.nn.functional.mse_loss(output, targets)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")
```

**Expected speedup vs BF16 (H100 SXM):**
- GEMM throughput: 1.6-2x faster (FP8 tensor cores: 1979 TFLOPS vs 989 TFLOPS for BF16)
- End-to-end training: 1.3-1.5x faster (non-GEMM ops remain in BF16)
- Memory savings: ~10-15% reduction in activation memory
- Accuracy: within 0.1% of BF16 baseline for most LLM training runs

**Key caveats:**
- Only GEMM operations run in FP8; LayerNorm, softmax, residual adds stay in higher precision
- First ~100 steps may show slightly higher loss as delayed scaling calibrates
- Not all model architectures benefit equally — gains are largest for GEMM-bound models

### Troubleshooting Precision Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| Loss is NaN | Gradient overflow in FP16 | Switch to BF16 or increase GradScaler initial scale |
| Loss plateaus early | Gradient underflow — small gradients round to zero | Increase loss scale factor or switch from FP16 to BF16 |
| Training diverges after N steps | Accumulation precision degrades over time | Enable FP32 gradient accumulation: `args.fp16_full_eval=True` or use BF16 |
| Model outputs garbage after quantized training | Wrong compute dtype during NF4/INT4 forward pass | Set `bnb_4bit_compute_dtype=torch.bfloat16` explicitly |
| Validation loss spikes periodically | GradScaler halving scale too aggressively (FP16) | Increase `growth_interval` or switch to BF16 (no scaler needed) |
| Fine-tuned model worse than base | Learning rate too high for low-precision training | Reduce LR by 2-5x; low precision amplifies effective noise |
| Checkpoints produce different results on reload | FP16 weights saved without master FP32 copy | Save optimizer state dict; use `model.half()` only at inference |
| Slow training despite using FP16/BF16 | Tensors not aligned to 8 — tensor cores not engaged | Ensure hidden dims are multiples of 8 (ideally 64 or 128) |
| Loss NaN only on long sequences | Attention scores overflow before softmax | Use FlashAttention or force FP32 softmax in attention |
| Gradient norm is always zero | All gradients underflowed to zero in FP16 | Switch to BF16 or apply loss scaling before backward pass |
| Weights become all zeros after training | Weight decay too aggressive at low precision | Reduce `weight_decay` (try 0.01 instead of 0.1) and use FP32 master weights |
| Inf values in activations | Intermediate values exceed dtype range | Add activation clipping or use BF16 which has FP32-equivalent range |

### Memory Comparison Table

Approximate peak GPU memory (VRAM) for training, including model weights, gradients, optimizer states, and activations (batch size 1, sequence length 512).

| Setup | 7B Model | 13B Model | 70B Model |
|-------|----------|-----------|-----------|
| Full Fine-Tune FP32 | ~112 GB | ~208 GB | ~1120 GB |
| Full Fine-Tune BF16 (mixed precision) | ~58 GB | ~108 GB | ~580 GB |
| LoRA BF16 (r=16, all linear) | ~16 GB | ~28 GB | ~140 GB |
| LoRA BF16 (r=64, all linear) | ~18 GB | ~32 GB | ~155 GB |
| QLoRA NF4 (r=16, all linear) | ~6 GB | ~10 GB | ~38 GB |
| QLoRA NF4 (r=64, all linear) | ~8 GB | ~13 GB | ~42 GB |
| QLoRA NF4 + double quant (r=64) | ~7 GB | ~11 GB | ~38 GB |
| Inference only FP32 | ~28 GB | ~52 GB | ~280 GB |
| Inference only BF16 | ~14 GB | ~26 GB | ~140 GB |
| Inference only INT4 (GPTQ/AWQ) | ~4 GB | ~7.5 GB | ~35 GB |

**Notes on memory estimates:**
- FP32 full fine-tune: ~16 bytes/param (4B weights + 4B gradients + 8B AdamW states)
- BF16 mixed precision: ~18 bytes/param in theory (2B weights + 4B master weights + 2B gradients + 8B optimizer + ~2B activations), but in practice ~8-8.5 bytes/param effective with activation checkpointing
- LoRA: base model frozen in BF16 (2 bytes/param) + 16 bytes per trainable LoRA param (typically 0.1-1% of total)
- QLoRA: base model in NF4 (~0.55 bytes/param with double quant) + 16 bytes per LoRA param + dequantization overhead
- Activation memory scales with batch size and sequence length; estimates above assume batch size 1, seq_len 512, gradient checkpointing enabled for LoRA/QLoRA
- Actual memory varies by model architecture, framework version, and CUDA memory allocator behavior
