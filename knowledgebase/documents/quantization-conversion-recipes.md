---
id: quantization_conversion_recipes
kind: document
title: Quantization Conversion Recipes - Practical Step-by-Step Guide
category: quantization
summary: Complete practical recipes for converting LLM models to AWQ INT4, GPTQ INT4, FP8, GGUF, and other quantized formats, with CLI commands, tool installation, calibration data selection, quality validation, and troubleshooting.
tags:
  - quantization
  - awq
  - gptq
  - fp8
  - gguf
  - conversion
  - recipe
  - autoawq
  - auto-gptq
  - llama-cpp
  - calibration
source_ids:
  - awq-activation-aware-weight-quantization
operators:
  - matmul
  - gemm
gpu_families:
  - Ampere
  - Ada
  - Hopper
  - Blackwell
precision:
  - fp8
  - int8
  - int4
  - fp4
  - nf4
---

# Quantization Conversion Recipes

## Quick Decision: Which Format Do I Need?

```
What is your deployment target?
│
├─ vLLM / SGLang (GPU serving)
│  ├─ Hopper/Blackwell GPU? → FP8 (best quality, ~2x memory savings)
│  ├─ Ampere/Ada GPU? → AWQ INT4 (best speed with Marlin kernel)
│  └─ Multi-GPU? → FP8 if Hopper, else AWQ INT4
│
├─ llama.cpp / Ollama (CPU or consumer GPU)
│  └─ GGUF Q4_K_M (best quality/size tradeoff)
│
├─ TensorRT-LLM
│  ├─ Hopper? → FP8 (native TRT-LLM support)
│  └─ Ampere/Ada? → INT4 AWQ or INT8 SmoothQuant
│
├─ PyTorch / HuggingFace (research/dev)
│  ├─ Fine-tuning? → QLoRA (NF4 via bitsandbytes)
│  └─ Inference? → AWQ or GPTQ
│
└─ Edge / Mobile
   └─ GGUF Q4_0 or Q3_K_S (smallest)
```

## Memory-Bound vs Compute-Bound: Why It Matters for Quantization

### The Core Insight
```
LLM inference has TWO phases with DIFFERENT bottlenecks:

PREFILL (processing input prompt):
  - Compute-bound (large matrix multiplications)
  - Arithmetic Intensity = 2*N (high, where N = batch * seq_len)
  - GPU tensor cores are the bottleneck
  - Quantization helps: fewer FLOPS (INT4 GEMM = 2x faster than FP16)

DECODE (generating tokens one-by-one):
  - Memory-bound (loading weights for each token)
  - Arithmetic Intensity ≈ 1-2 (very low at batch=1)
  - HBM bandwidth is the bottleneck
  - Quantization helps MORE: 4x less data to load (INT4 vs FP16)

This is why quantization gives bigger speedups for decode:
  FP16 → INT4: weights are 4x smaller → 4x less memory to load → ~3-4x faster decode
  FP16 → FP8:  weights are 2x smaller → 2x less memory to load → ~1.5-2x faster decode
```

### Roofline Analysis for Quantization Decisions
```
                  FLOPS
                    │
  Compute ceiling ──┤━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    │                   ╱
                    │                 ╱   ← FP16
                    │               ╱
  INT4 TC ceiling ──┤─────────────╱────────────────────
                    │           ╱╱
                    │         ╱╱   ← INT4 (2x compute, 4x less memory)
                    │       ╱╱
                    │     ╱╱
                    │   ╱╱
                    │ ╱╱
                    ╱╱
                    ├──────────┬──────────────────── Arithmetic Intensity
                    0    Decode(~1)   Prefill(~100+)
                       (memory-     (compute-
                        bound)       bound)

Key insight: At low AI (decode), the gap between FP16 and INT4
is determined by MEMORY BANDWIDTH, not compute.
INT4 shifts the roofline knee-point LEFT, making more operations compute-bound.
```

### Memory-Bound Analysis by GPU
```
Model: LLaMA 3 8B

Weight loading time per token (batch=1):
┌────────────┬──────────┬──────────┬──────────┬──────────┐
│ GPU        │ FP16     │ FP8      │ INT4     │ INT4+    │
│            │ (16 GB)  │ (8 GB)   │ (4 GB)   │ Marlin   │
├────────────┼──────────┼──────────┼──────────┼──────────┤
│ RTX 3090   │ 16.5 ms  │ 8.3 ms   │ 4.1 ms   │ 3.8 ms  │
│ (936 GB/s) │ 60 t/s   │ 120 t/s  │ 240 t/s  │ 260 t/s │
├────────────┼──────────┼──────────┼──────────┼──────────┤
│ RTX 4090   │ 15.4 ms  │ 7.7 ms   │ 3.8 ms   │ 3.2 ms  │
│ (1008 GB/s)│ 65 t/s   │ 130 t/s  │ 260 t/s  │ 310 t/s │
├────────────┼──────────┼──────────┼──────────┼──────────┤
│ A100 80GB  │ 7.8 ms   │ 3.9 ms   │ 2.0 ms   │ 1.8 ms  │
│ (2039 GB/s)│ 128 t/s  │ 256 t/s  │ 500 t/s  │ 555 t/s │
├────────────┼──────────┼──────────┼──────────┼──────────┤
│ H100 SXM   │ 4.8 ms   │ 2.4 ms   │ 1.2 ms   │ 1.0 ms  │
│ (3350 GB/s)│ 208 t/s  │ 416 t/s  │ 833 t/s  │ 1000 t/s│
└────────────┴──────────┴──────────┴──────────┴──────────┘

Formula: time_per_token = model_size_bytes / bandwidth
Tokens/sec ≈ 1 / time_per_token (simplified, ignores KV cache + activations)

At batch > 1, arithmetic intensity increases:
  batch=1:  AI ≈ 1   (memory-bound, quantization helps a lot)
  batch=16: AI ≈ 16  (still memory-bound for most ops)
  batch=64: AI ≈ 64  (approaching compute-bound, quantization helps less)
  batch=256: AI ≈ 256 (compute-bound, quantization helps via faster tensor cores)
```

### When Quantization Helps Most vs Least
```
MOST BENEFIT (memory-bound scenarios):
  ✓ Single-user inference (batch=1)
  ✓ Long context generation (large KV cache competes for bandwidth)
  ✓ Consumer GPUs (limited bandwidth: 500-1000 GB/s)
  ✓ Decode phase (always memory-bound)
  ✓ Models that barely fit in memory

LEAST BENEFIT (compute-bound scenarios):
  ✗ Large batch prefill (batch=64+, already compute-bound)
  ✗ High-end GPUs with high batch (H100 at batch=128)
  ✗ Short prompts with batched processing
  ✗ Training (forward pass is batched, backward is compute-heavy)

NEGATIVE IMPACT (when quantization hurts):
  ✗ Quality-sensitive tasks (math, reasoning, code) with aggressive INT4
  ✗ Small models (<3B) where quantization error is proportionally larger
  ✗ Languages/domains far from calibration data
```

---

## Recipe 1: AWQ INT4 Quantization

### What is AWQ?
```
Activation-Aware Weight Quantization:
- Finds "salient" weight channels by looking at activation magnitudes
- Scales those channels UP before quantization (protects important weights)
- Scales activations DOWN equivalently (mathematically lossless)
- Then applies standard INT4 group quantization (group_size=128)
- Result: INT4 weights with minimal quality loss
```

### Installation
```bash
# AutoAWQ (recommended, easiest)
pip install autoawq

# Or from source for latest features
pip install autoawq@git+https://github.com/casper-hansen/AutoAWQ.git
```

### Step-by-Step Conversion
```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# Step 1: Load model in FP16
model_id = "meta-llama/Llama-3.1-8B-Instruct"
model = AutoAWQForCausalLM.from_pretrained(model_id, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Step 2: Configure quantization
quant_config = {
    "zero_point": True,      # asymmetric quantization (better quality)
    "q_group_size": 128,     # per-group quantization (128 weights share a scale)
    "w_bit": 4,              # 4-bit weights
    "version": "gemm",       # "gemm" for general, "gemv" for batch=1 only
}
# version options:
#   "gemm"  → works with all batch sizes, uses Marlin kernel in vLLM
#   "gemv"  → optimized for batch=1 only (slightly faster for single-user)
#   "marlin" → directly uses Marlin-compatible layout

# Step 3: Quantize (takes 10-30 min depending on model size)
model.quantize(tokenizer, quant_config=quant_config)

# Step 4: Save
model.save_quantized("Llama-3.1-8B-Instruct-AWQ")
tokenizer.save_pretrained("Llama-3.1-8B-Instruct-AWQ")

# Step 5: Verify - quick generation test
model = AutoAWQForCausalLM.from_quantized("Llama-3.1-8B-Instruct-AWQ")
tokens = tokenizer("The capital of France is", return_tensors="pt").to("cuda")
output = model.generate(**tokens, max_new_tokens=20)
print(tokenizer.decode(output[0]))
```

### Custom Calibration Data
```python
# Default: uses built-in "pileval" dataset (general English text)
# For domain-specific models, use your own calibration data:

calibration_data = [
    "Your domain-specific text sample 1...",
    "Your domain-specific text sample 2...",
    # Use 128-512 samples, each 512-2048 tokens
    # Samples should be REPRESENTATIVE of your actual workload
]

model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data=calibration_data,  # custom calibration
    # OR use a HuggingFace dataset:
    # calib_data="wikitext",
    # split="train",
)
```

### Serving AWQ Models
```bash
# vLLM (recommended for production)
vllm serve Llama-3.1-8B-Instruct-AWQ \
  --quantization awq \
  --dtype float16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9

# vLLM with Marlin kernel (faster, auto-detected for AWQ)
# Marlin is used automatically when available - no extra flag needed

# SGLang
python -m sglang.launch_server \
  --model Llama-3.1-8B-Instruct-AWQ \
  --quantization awq

# HuggingFace Transformers (dev/testing)
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "Llama-3.1-8B-Instruct-AWQ",
    device_map="auto"
)
```

### AWQ Quality Validation
```python
# Perplexity evaluation
from lm_eval import evaluator
results = evaluator.simple_evaluate(
    model="hf",
    model_args=f"pretrained=Llama-3.1-8B-Instruct-AWQ",
    tasks=["wikitext"],
    batch_size=8,
)
ppl = results['results']['wikitext_2']['word_perplexity']
print(f"AWQ INT4 perplexity: {ppl:.2f}")
# Expected: < 0.5 PPL increase over FP16 baseline

# Quick sanity check prompts
test_prompts = [
    "Explain quantum computing in simple terms:",
    "Write a Python function to sort a list:",
    "What is 15 * 23 + 7?",  # math is most sensitive to quantization
]
```

### Troubleshooting AWQ
```
Problem: "CUDA out of memory" during quantization
  → Quantization needs the full FP16 model + calibration activations
  → Need ~2x model size in GPU memory (e.g., 32GB for 8B model)
  → Solution: Use --device_map "cpu" then move layers to GPU one at a time
  → Or use a machine with more VRAM just for quantization

Problem: Poor quality on specific tasks
  → Try group_size=64 (more scales, better quality, slightly larger)
  → Use domain-specific calibration data
  → Compare with GPTQ (sometimes GPTQ is better for specific models)

Problem: Slow inference with AWQ
  → Ensure Marlin kernel is being used (check vLLM logs for "marlin")
  → Use version="gemm" not "gemv" if batch > 1
  → Check: pip install autoawq[kernels] for compiled CUDA kernels

Problem: Model architecture not supported
  → Check AutoAWQ supported models: most LLaMA, Mistral, Qwen, Phi, Gemma
  → For unsupported: try GPTQ instead (wider architecture support)
```

---

## Recipe 2: GPTQ INT4 Quantization

### When to Use GPTQ vs AWQ
```
AWQ: faster quantization, slightly better at very low bits, Marlin kernel support
GPTQ: wider model support, sometimes better quality, more configuration options
Rule of thumb: Try AWQ first, fall back to GPTQ if AWQ doesn't support the model
```

### Installation
```bash
pip install auto-gptq
# Or with CUDA extensions pre-built:
pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu121/
```

### Step-by-Step Conversion
```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer

model_id = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Step 1: Configure
quantize_config = BaseQuantizeConfig(
    bits=4,              # 4-bit quantization
    group_size=128,      # per-group (128 weights share scale/zero)
    damp_percent=0.01,   # Hessian dampening (higher = more stable, less accurate)
    desc_act=False,      # activation ordering (True = better quality, slower quant)
    sym=True,            # symmetric quantization
    model_seqlen=2048,   # calibration sequence length
)

# Step 2: Load model
model = AutoGPTQForCausalLM.from_pretrained(model_id, quantize_config)

# Step 3: Prepare calibration data
from datasets import load_dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
calibration_data = [
    tokenizer(text, return_tensors="pt", max_length=2048, truncation=True)
    for text in dataset["text"][:256]  # 256 samples
    if len(text.strip()) > 100  # skip short samples
]

# Step 4: Quantize (can take 1-4 hours for large models)
model.quantize(calibration_data)

# Step 5: Save
model.save_quantized("Llama-3.1-8B-Instruct-GPTQ")
tokenizer.save_pretrained("Llama-3.1-8B-Instruct-GPTQ")
```

### GPTQ with Activation Ordering (desc_act=True)
```python
# desc_act=True processes columns in descending activation magnitude order
# Better quality but:
#   - Slower quantization (2-3x)
#   - Requires act-order-aware kernels (ExLlamaV2, Marlin supports it)
#   - Slightly slower inference (reordering overhead)

quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=True,  # Enable activation ordering
)
```

### Serving GPTQ Models
```bash
# vLLM
vllm serve Llama-3.1-8B-Instruct-GPTQ \
  --quantization gptq \
  --dtype float16

# vLLM with Marlin backend (faster, auto-detected)
vllm serve Llama-3.1-8B-Instruct-GPTQ \
  --quantization gptq_marlin  # explicit Marlin backend
```

---

## Recipe 3: FP8 Quantization (Hopper/Blackwell)

### Why FP8?
```
FP8 E4M3: 4 exponent bits, 3 mantissa bits, range [-448, 448]
  - 2x memory reduction vs FP16 (8 GB instead of 16 GB for 8B model)
  - <0.1 perplexity degradation (nearly lossless)
  - Native tensor core support on Hopper (H100/H200) and Blackwell (B100/B200)
  - No calibration data needed for simple per-tensor quantization
  - cuBLAS/cuBLASLt support FP8 GEMM natively
```

### Method A: vLLM FP8 (Simplest - No Pre-Conversion Needed)
```bash
# vLLM can quantize on-the-fly at load time (dynamic quantization)
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --quantization fp8 \
  --dtype float16 \
  --max-model-len 8192

# This applies per-tensor FP8 quantization to all linear layers
# No separate quantization step needed!
# Works only on Hopper+ GPUs (H100, H200, B100, B200)
```

### Method B: Offline FP8 Quantization with llm-compressor
```bash
pip install llmcompressor
```

```python
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor import oneshot
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# FP8 weight-only (simple, no calibration)
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8",            # FP8 E4M3 weights
    ignore=["lm_head"],      # keep output layer in FP16
)

oneshot(
    model=model,
    recipe=recipe,
    output_dir="Llama-3.1-8B-Instruct-FP8",
)
tokenizer.save_pretrained("Llama-3.1-8B-Instruct-FP8")
```

### Method C: FP8 with Calibration (Better Quality)
```python
from datasets import load_dataset

# Calibrated FP8: uses activation statistics for better scale selection
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",    # FP8 with dynamic per-tensor activation scaling
    ignore=["lm_head"],
)

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
calibration_data = dataset["text"][:512]

oneshot(
    model=model,
    recipe=recipe,
    dataset=calibration_data,
    output_dir="Llama-3.1-8B-Instruct-FP8-Calibrated",
)
```

### Method D: FP8 KV Cache (Separate from Weight Quantization)
```bash
# vLLM: FP8 weights + FP8 KV cache (maximum memory savings)
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --quantization fp8 \
  --kv-cache-dtype fp8_e4m3 \
  --dtype float16

# This gives:
# - 2x weight memory reduction (FP8 weights)
# - 2x KV cache memory reduction (FP8 KV)
# - Enables much longer contexts or larger batches
```

### FP8 vs INT4: When to Choose Which
```
┌──────────────────┬──────────────┬──────────────┐
│ Metric           │ FP8          │ INT4 (AWQ)   │
├──────────────────┼──────────────┼──────────────┤
│ Memory savings   │ 2x           │ 4x           │
│ Quality loss     │ <0.1 PPL     │ 0.1-0.5 PPL  │
│ Calibration      │ Optional     │ Required     │
│ GPU requirement  │ Hopper+      │ Any NVIDIA   │
│ Decode speedup   │ ~1.5-2x      │ ~3-4x        │
│ Prefill speedup  │ ~1.5x        │ ~1.5-2x      │
│ Best for         │ Quality-first│ Memory-first │
└──────────────────┴──────────────┴──────────────┘

Decision: If you have a Hopper GPU AND quality matters → FP8
          If you need max memory savings OR have Ampere/Ada → INT4 AWQ
          If you want both → FP8 weights + INT4 KV cache
```

---

## Recipe 4: GGUF Conversion (llama.cpp / Ollama)

### When to Use GGUF
```
- CPU inference or mixed CPU+GPU
- Consumer GPUs with limited VRAM
- Ollama / llama.cpp deployment
- Edge devices, Apple Silicon
- When you need a single self-contained file
```

### Step-by-Step Conversion
```bash
# Step 1: Install llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make -j$(nproc)  # or: cmake -B build && cmake --build build

# Step 2: Install Python dependencies
pip install -r requirements/requirements-convert_hf_to_gguf.txt

# Step 3: Convert HuggingFace model to GGUF (FP16 base)
python convert_hf_to_gguf.py \
  /path/to/Llama-3.1-8B-Instruct \
  --outfile Llama-3.1-8B-Instruct-F16.gguf \
  --outtype f16

# Step 4: Quantize to desired format
./llama-quantize \
  Llama-3.1-8B-Instruct-F16.gguf \
  Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  Q4_K_M

# Step 5: Test
./llama-cli -m Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  -p "The capital of France is" -n 50
```

### GGUF Format Selection Guide
```
┌────────────┬──────────┬──────────┬────────────┬──────────────────────────┐
│ Format     │ Bits/W   │ 8B Size  │ Quality    │ Best For                 │
├────────────┼──────────┼──────────┼────────────┼──────────────────────────┤
│ F16        │ 16.0     │ 16.0 GB  │ Baseline   │ Base for quantization    │
│ Q8_0       │ 8.5      │ 8.5 GB   │ Excellent  │ When memory allows       │
│ Q6_K       │ 6.6      │ 6.6 GB   │ Very good  │ Quality-first budget     │
│ Q5_K_M     │ 5.7      │ 5.7 GB   │ Good       │ Balanced                 │
│ Q4_K_M ★   │ 4.8      │ 4.8 GB   │ Good       │ RECOMMENDED DEFAULT      │
│ Q4_K_S     │ 4.6      │ 4.6 GB   │ Acceptable │ Slightly smaller         │
│ Q3_K_M     │ 3.9      │ 3.9 GB   │ Fair       │ Tight memory budget      │
│ Q2_K       │ 3.4      │ 3.4 GB   │ Poor       │ Extreme compression      │
│ IQ4_XS     │ 4.3      │ 4.3 GB   │ Good       │ Better quality than Q4_K │
│ IQ3_XXS    │ 3.1      │ 3.1 GB   │ Fair       │ Smallest usable          │
└────────────┴──────────┴──────────┴────────────┴──────────────────────────┘

★ Q4_K_M = recommended starting point for most use cases
  K = "K-quant" (uses super-blocks with mixed precision)
  M = "Medium" (balance between quality and size)

IQ = "Importance-based Quantization" (uses codebooks, better quality per bit)
  Requires more CPU at inference time
```

### Importance Matrix Quantization (Best Quality)
```bash
# Step 1: Generate importance matrix from calibration data
./llama-imatrix \
  -m Llama-3.1-8B-Instruct-F16.gguf \
  -f calibration_data.txt \
  -o imatrix.dat \
  --chunks 200

# Step 2: Quantize with importance matrix
./llama-quantize \
  --imatrix imatrix.dat \
  Llama-3.1-8B-Instruct-F16.gguf \
  Llama-3.1-8B-Instruct-IQ4_XS.gguf \
  IQ4_XS

# IQ formats with imatrix: significantly better quality than standard quants
# IQ4_XS + imatrix ≈ Q5_K_M quality at Q4_K_M size
```

### Using with Ollama
```bash
# Create Modelfile
cat > Modelfile << 'EOF'
FROM ./Llama-3.1-8B-Instruct-Q4_K_M.gguf

TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>
{{ .System }}<|eot_id|>{{ end }}<|start_header_id|>user<|end_header_id|>
{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 8192
EOF

# Import into Ollama
ollama create my-llama -f Modelfile
ollama run my-llama
```

---

## Recipe 5: INT8 SmoothQuant (W8A8)

### When to Use
```
- When INT4 quality is not acceptable
- When FP8 is not available (pre-Hopper GPUs)
- When you need both weight AND activation quantization
- Server workloads with large batches (INT8 tensor cores are fast)
```

### Conversion with llm-compressor
```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

model_id = "meta-llama/Llama-3.1-8B-Instruct"

recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),  # balance activation outliers
    QuantizationModifier(
        targets="Linear",
        scheme="W8A8",           # INT8 weights + INT8 activations
        ignore=["lm_head"],
    ),
]

oneshot(
    model=model_id,
    recipe=recipe,
    dataset="wikitext",
    output_dir="Llama-3.1-8B-Instruct-W8A8",
    max_seq_length=2048,
    num_calibration_samples=512,
)
```

---

## Recipe 6: QLoRA / NF4 (For Fine-Tuning)

### Setup
```bash
pip install bitsandbytes peft transformers accelerate
```

### Loading in NF4 for Fine-Tuning
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# NF4 quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NormalFloat4 (better than INT4)
    bnb_4bit_compute_dtype=torch.bfloat16,  # compute in BF16
    bnb_4bit_use_double_quant=True,      # quantize the quantization constants
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)

# Memory: ~5 GB for 8B model (vs 16 GB FP16)
# NOT for production inference - use AWQ/GPTQ instead
# USE for: fine-tuning with LoRA adapters
```

---

## Quality Comparison Across Methods

```
Model: LLaMA 3.1 8B Instruct
Eval: WikiText-2 Perplexity (lower is better)

┌──────────────┬─────────┬──────────┬──────────────┐
│ Method       │ PPL     │ Δ PPL    │ Model Size   │
├──────────────┼─────────┼──────────┼──────────────┤
│ FP16 (base)  │ 6.14    │ 0.00     │ 16.0 GB      │
│ FP8          │ 6.16    │ +0.02    │ 8.0 GB       │
│ INT8 W8A8    │ 6.20    │ +0.06    │ 8.5 GB       │
│ AWQ INT4     │ 6.38    │ +0.24    │ 4.5 GB       │
│ GPTQ INT4    │ 6.42    │ +0.28    │ 4.5 GB       │
│ GGUF Q4_K_M  │ 6.45    │ +0.31    │ 4.8 GB       │
│ GGUF IQ4_XS  │ 6.35    │ +0.21    │ 4.3 GB       │
│ GGUF Q3_K_M  │ 6.85    │ +0.71    │ 3.9 GB       │
│ GGUF Q2_K    │ 8.20    │ +2.06    │ 3.4 GB       │
│ NF4 (bnb)    │ 6.50    │ +0.36    │ 5.0 GB       │
└──────────────┴─────────┴──────────┴──────────────┘

Quality ranking: FP8 > IQ4_XS ≈ AWQ > GPTQ ≈ Q4_K_M > NF4 > Q3_K_M >> Q2_K
Speed ranking:   AWQ+Marlin > FP8 > INT8 > GPTQ > GGUF (on GPU)
```

## End-to-End Conversion Checklist

```
[ ] 1. Decide format based on deployment target (see decision tree above)
[ ] 2. Check GPU compatibility (FP8 needs Hopper+, INT4 works everywhere)
[ ] 3. Prepare calibration data (128-512 representative samples)
[ ] 4. Run quantization (AWQ: ~15min, GPTQ: ~1-4hr, FP8: ~5min, GGUF: ~10min)
[ ] 5. Validate quality:
      - Perplexity on wikitext (should be <0.5 PPL increase for INT4)
      - Run 10-20 representative prompts manually
      - Check math/code/reasoning tasks specifically
[ ] 6. Benchmark inference speed:
      - tokens/sec at batch=1 (decode)
      - tokens/sec at batch=16 (throughput)
      - time-to-first-token (TTFT)
      - peak GPU memory
[ ] 7. Deploy with appropriate serving framework
[ ] 8. Monitor quality in production (sample outputs periodically)
```
