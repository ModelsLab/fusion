---
id: training_free_acceleration_techniques
kind: document
title: Training-Free Model Acceleration Techniques
category: optimization
summary: Comprehensive guide to training-free methods for accelerating LLM and diffusion model inference, including distillation-free speedups, cache-based acceleration, token pruning, layer skipping, and step reduction techniques with practical recipes.
tags:
  - training-free
  - acceleration
  - distillation
  - step-reduction
  - cache-reuse
  - token-pruning
  - layer-skipping
  - speculative-decoding
  - diffusion
source_ids: []
operators:
  - attention
  - matmul
  - general
gpu_families:
  - Ampere
  - Ada
  - Hopper
  - Blackwell
---

# Training-Free Model Acceleration Techniques

This document covers practical, training-free (or minimal-training) methods to accelerate LLM and
diffusion model inference. Each technique includes what it does, how it works, where to find it,
performance numbers, and when to use it.

---

## Part 1: LLM Acceleration

### 1. Speculative Decoding

Generate multiple candidate tokens per forward pass, then verify them in parallel against the
target model. The target model's output distribution is preserved exactly -- there is zero quality
loss.

#### 1a. Draft-Model Speculative Decoding

A small "draft" model proposes N candidate tokens. The large target model verifies all N in a
single forward pass, accepting tokens that match. Rejected tokens trigger re-sampling from the
target distribution.

- **Paper**: [Accelerating LLM Inference with Staged Speculative Decoding](https://arxiv.org/abs/2302.01318)
- **Speedup**: 1.5-3x depending on acceptance rate and draft model quality
- **Quality Impact**: Lossless (mathematically equivalent output distribution)

**Usage with vLLM:**

```bash
pip install vllm
```

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    speculative_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    num_speculative_tokens=5,
)
output = llm.generate("Explain quantum computing", SamplingParams(temperature=0.7))
```

#### 1b. EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency)

Uses a lightweight autoregressive head attached to the target model's internal layers to predict
draft tokens by extrapolating second-top-layer feature vectors, eliminating the need for a
separate draft model.

- **Paper**: [EAGLE-1 (ICML 2024)](https://arxiv.org/abs/2401.15077), [EAGLE-3 (NeurIPS 2025)](https://github.com/SafeAILab/EAGLE)
- **GitHub**: https://github.com/SafeAILab/EAGLE
- **Speedup**: 2-3x (EAGLE-1), 2-6x (EAGLE-3 depending on batch size)
- **Quality Impact**: Lossless
- **Training Required**: Yes -- the EAGLE head must be trained (~hours on 1 GPU), but base model is frozen

**Usage with vLLM:**

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    speculative_config={
        "method": "eagle",
        "model": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
        "num_speculative_tokens": 5,
    },
)
```

**Usage with SGLang:**

```bash
python -m sglang.launch_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --speculative-algo EAGLE \
    --speculative-draft yuhuili/EAGLE-LLaMA3.1-Instruct-8B \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 8 \
    --speculative-num-draft-tokens 64
```

#### 1c. Medusa

Adds K extra language model heads to the base model. Each head predicts a different future token
position. A tree-based attention mechanism constructs and verifies multiple candidate continuations
simultaneously.

- **Paper**: https://arxiv.org/abs/2401.10774
- **GitHub**: https://github.com/FasterDecoding/Medusa
- **Speedup**: 2-3x
- **Quality Impact**: Lossless (with typical acceptance)
- **Training Required**: Yes -- Medusa heads need fine-tuning, but base model is frozen

```bash
pip install medusa-llm
```

#### 1d. LayerSkip (Self-Speculative Decoding)

Uses the model itself for speculation: early layers generate draft tokens via early exit, then the
full model verifies. No separate draft model needed. Requires training with progressive layer
dropout.

- **Paper**: https://arxiv.org/abs/2404.16710 (ACL 2024)
- **GitHub**: https://github.com/facebookresearch/LayerSkip
- **Speedup**: 1.8-2.2x
- **Quality Impact**: Lossless (via self-speculative verification)

**When to use**: Best when you cannot host a separate draft model. Requires LayerSkip-trained
checkpoints. **When NOT to use**: If you already have a good draft model, standard speculative
decoding or EAGLE will be faster.

---

### 2. Token Pruning / Early Exit

Skip computation for tokens that contribute little to the final output, reducing FLOPs per layer.

#### 2a. FastGen

Exploits attention sparsity by retaining only the most influential tokens' KV cache entries
during the decoding stage. Uses attention-score-based profiling to decide which tokens to keep.

- **Paper**: https://arxiv.org/abs/2310.01801
- **Speedup**: 1.5-2x with long contexts
- **Quality Impact**: Minimal degradation on generation tasks

#### 2b. CALM (Confident Adaptive Language Modeling)

Dynamically exits at an intermediate transformer layer when the model is confident enough in its
prediction. Uses softmax confidence, cosine similarity between layers, or learned classifiers as
early-exit criteria.

- **Paper**: https://arxiv.org/abs/2207.07061 (NeurIPS 2022)
- **GitHub**: https://github.com/google-research/t5x (contrib/calm)
- **Speedup**: 2-3x (uses only 1/3 to 1/2 of layers on average)
- **Quality Impact**: Maintains full-model performance on CNN/DM, WMT, SQuAD
- **Training Required**: Needs calibration of confidence thresholds

**When to use**: Tasks with variable-difficulty tokens (summarization, translation). **When NOT
to use**: Tasks requiring maximum precision on every token (code generation, math).

#### 2c. Mixture of Depths (MoD)

Uses a learned top-k router at each layer to select which tokens participate in self-attention and
MLP computation. Tokens not selected skip the layer entirely via a residual connection.

- **Paper**: https://arxiv.org/abs/2404.02258
- **Speedup**: Up to 50% FLOP reduction per forward pass
- **Quality Impact**: Matches or exceeds vanilla transformer at equivalent FLOPs
- **Training Required**: Yes -- must be trained with the routing mechanism from scratch

#### 2d. LazyLLM

Selectively computes KV for tokens important to the next prediction using attention scores from
the prior layer, deferring computation of remaining tokens to later steps.

- **Paper**: https://arxiv.org/abs/2407.14057
- **Speedup**: 1.5-2.5x on long-context inputs
- **Quality Impact**: Minimal -- deferred tokens are computed if needed later

**When to use**: Long-context inference with many low-importance tokens. **When NOT to use**:
Short prompts where all tokens matter equally.

---

### 3. Layer Skipping

Remove or skip entire transformer layers at inference time. Works because many LLM layers produce
highly similar outputs (high cosine similarity between input and output).

#### 3a. ShortGPT

Defines a Block Influence (BI) metric measuring each layer's contribution. Layers with lowest BI
scores are permanently removed. No retraining required.

- **Paper**: https://arxiv.org/abs/2403.03853
- **GitHub**: https://github.com/sramshetty/ShortGPT
- **Speedup**: ~25% parameter and compute reduction
- **Quality Impact**: Retains 92-95% of performance (e.g., MMLU drops from 55.0 to 52.2 removing
  25% of layers from LLaMA-2-13B)

```bash
pip install transformers torch
# Compute BI scores and prune
git clone https://github.com/sramshetty/ShortGPT.git
cd ShortGPT
python prune.py --model meta-llama/Llama-2-13b-hf --remove_layers 10
```

#### 3b. SLEB (Streamlining LLMs through Redundancy Verification and Elimination)

Iteratively prunes transformer blocks by measuring perplexity degradation on a calibration set.
Removes blocks whose elimination causes the least perplexity increase.

- **Paper**: https://arxiv.org/abs/2402.09025
- **GitHub**: https://github.com/jiwonsong-dev/SLEB
- **Speedup**: 20-30% compute reduction
- **Quality Impact**: Better calibrated than ShortGPT due to perplexity-guided selection

**When to use**: You need a permanently smaller model and can tolerate a small quality drop.
**When NOT to use**: You need lossless inference -- use speculative decoding instead.

---

### 4. KV Cache Optimization

Reduce memory footprint and improve throughput by intelligently managing the KV cache, which
grows linearly with sequence length and can dominate GPU memory.

#### 4a. StreamingLLM

Maintains attention sinks (first few tokens) plus a sliding window of recent tokens. Enables
infinite-length generation with fixed memory by exploiting the observation that initial tokens
accumulate disproportionate attention mass.

- **Paper**: https://arxiv.org/abs/2309.17453
- **GitHub**: https://github.com/mit-han-lab/streaming-llm
- **Speedup**: Enables infinite context with fixed memory (not a direct speed gain)
- **Quality Impact**: Good for streaming/dialogue; degrades on tasks requiring distant context

```bash
pip install streaming-llm
```

#### 4b. H2O (Heavy-Hitter Oracle)

Dynamically retains a balanced mix of "heavy hitter" tokens (those with high cumulative attention
scores) and recent tokens. Maintains a fixed-size KV cache by evicting low-scoring entries.

- **Paper**: https://arxiv.org/abs/2306.14048
- **GitHub**: https://github.com/FMInference/H2O
- **Speedup**: 5-10x memory reduction; enables larger batch sizes
- **Quality Impact**: Minimal on generation tasks; can degrade on needle-in-haystack retrieval

#### 4c. SnapKV

During the prefill stage, uses a small "observation window" at the end of the prompt to predict
which KV entries will be important during generation. Compresses the prompt KV cache before
decoding begins.

- **Paper**: https://arxiv.org/abs/2404.14469 (NeurIPS 2024)
- **Speedup**: 3.6x decoding speedup; 8.2x memory reduction
- **Quality Impact**: Outperforms H2O -- matches full KV cache quality with 1024 entries on
  Mistral-7B where H2O degrades at 4096

```bash
pip install snapkv  # or integrate via transformers
```

#### 4d. xKV (Personalized KV Cache Reduction)

Assigns per-layer cache budgets using a hierarchical framework that adapts to each layer's
attention pattern. Layers with sparser attention get smaller caches.

- **Paper**: https://arxiv.org/abs/2412.05896
- **Speedup**: Better memory-quality tradeoff than uniform eviction
- **Quality Impact**: Consistently outperforms H2O and SnapKV on long-context tasks

**When to use**: Long-context inference, high-throughput serving. **When NOT to use**: Short
sequences where KV cache is not the bottleneck.

---

### 5. Quantization (Post-Training)

Reduce weight precision from FP16/BF16 to INT4/INT8/FP8 without retraining. Reduces memory
bandwidth requirements and enables larger batch sizes.

#### 5a. AWQ (Activation-Aware Weight Quantization)

Identifies salient weight channels based on activation magnitudes and protects them during
quantization. Non-salient weights are aggressively quantized to INT4.

- **Paper**: https://arxiv.org/abs/2306.00978 (MLSys 2024 Best Paper)
- **GitHub**: https://github.com/mit-han-lab/llm-awq
- **Speedup**: 2-3x throughput improvement; ~3x memory reduction
- **Quality Impact**: <1% degradation on most benchmarks at INT4

```bash
pip install autoawq
```

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model = AutoAWQForCausalLM.from_quantized("TheBloke/Llama-2-7B-AWQ", fuse_layers=True)
tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-7B-AWQ")
output = model.generate(**tokenizer("Hello", return_tensors="pt").to("cuda"), max_new_tokens=64)
```

#### 5b. GPTQ

Layer-wise quantization using Hessian-based optimization (OBQ/OBS). Minimizes output error at
each layer by optimally rounding weights based on second-order information.

- **Paper**: https://arxiv.org/abs/2210.17323
- **GitHub**: https://github.com/ModelCloud/GPTQModel
- **Speedup**: 2-3x with Marlin kernels (712 tok/s vs 276 tok/s baseline)
- **Quality Impact**: Comparable to AWQ; slightly better on some models

```bash
pip install gptqmodel  # or: pip install auto-gptq
```

#### 5c. GGUF (llama.cpp format)

CPU-friendly quantization format supporting mixed-precision (e.g., Q4_K_M keeps some layers at
higher precision). Excellent for CPU and Apple Silicon deployment.

- **GitHub**: https://github.com/ggerganov/llama.cpp
- **Speedup**: Enables CPU inference at usable speeds; 2-4x memory reduction
- **Quality Impact**: Q4_K_M retains ~98% of FP16 quality

```bash
# Install llama.cpp
brew install llama.cpp  # macOS
# or build from source
# Download GGUF model from HuggingFace and run:
llama-cli -m model.Q4_K_M.gguf -p "Hello world"
```

#### 5d. FP8 Quantization

Native 8-bit floating point on Hopper+ GPUs (H100, H200, B100). Supported natively by
TensorRT-LLM, vLLM, and SGLang with minimal quality loss.

- **Speedup**: 1.5-2x over FP16 on Hopper GPUs
- **Quality Impact**: Near-lossless (<0.5% degradation)

```python
# vLLM with FP8
from vllm import LLM
llm = LLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct", quantization="fp8")
```

**When to use**: Almost always -- quantization is the easiest win. **When NOT to use**: Tasks
requiring maximum numerical precision (rare); models already at target size.

---

### 6. Sparse Attention

Reduce the O(n^2) attention computation by skipping or approximating attention over distant or
unimportant token pairs.

#### 6a. SageAttention

Quantizes attention computation to INT8 (keys/queries) and FP8 (values), achieving 2-5x speedup
over FlashAttention with no end-to-end metric degradation. Drop-in replacement.

- **Paper**: https://arxiv.org/abs/2410.02367 (ICLR 2025)
- **GitHub**: https://github.com/thu-ml/SageAttention
- **Speedup**: 2-5x over FlashAttention; 560 TOPS on RTX 5090
- **Quality Impact**: No measurable degradation across language, image, and video models

```bash
pip install sageattention
```

```python
from sageattention import sageattn
# Replace F.scaled_dot_product_attention with sageattn
attn_output = sageattn(q, k, v)  # drop-in replacement
```

#### 6b. SpargeAttn

Combines SageAttention's quantization with dynamic sparsity -- skips attention blocks where the
attention matrix is predicted to be near-zero based on a lightweight predictor.

- **Paper**: https://arxiv.org/abs/2502.18137 (ICML 2025)
- **Speedup**: 2.5-5x over FlashAttention
- **Quality Impact**: Robust end-to-end performance preservation

#### 6c. NSA (Native Sparse Attention)

Hardware-aligned sparse attention designed for training and inference. Combines compressed
coarse-grained tokens, selected important tokens, and a sliding window in a single attention pass.

- **Paper**: https://arxiv.org/abs/2502.11089
- **Speedup**: Significant on long contexts (>8K tokens)
- **Quality Impact**: Natively trainable; matches dense attention when trained from scratch

**When to use**: Long-context models (>4K tokens). SageAttention for drop-in speedup; NSA for
models trained from scratch. **When NOT to use**: Short sequences where attention is not the
bottleneck.

---

### 7. Prompt Compression

Reduce the number of input tokens to decrease prefill cost and KV cache size.

#### 7a. LLMLingua / LLMLingua-2

Uses a small language model (GPT-2 or LLaMA-7B) to identify and remove unimportant tokens from
prompts. Achieves up to 20x compression with minimal performance loss.

- **Paper**: https://arxiv.org/abs/2310.05736 (EMNLP 2023)
- **GitHub**: https://github.com/microsoft/LLMLingua
- **Speedup**: 1.7-5.7x end-to-end inference acceleration
- **Quality Impact**: <1.5% loss at 20x compression on GSM8K

```bash
pip install llmlingua
```

```python
from llmlingua import PromptCompressor

compressor = PromptCompressor(model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank")
result = compressor.compress_prompt(
    original_prompt,
    rate=0.5,  # Keep 50% of tokens
    force_tokens=["\\n", "?"],
)
compressed_prompt = result["compressed_prompt"]
```

**When to use**: RAG pipelines with long retrieved contexts. **When NOT to use**: Short prompts;
tasks where every input token is critical (code completion with full file context).

---

### 8. Lookahead Decoding

Applies Jacobi iteration to autoregressive decoding: instead of generating tokens one at a time,
maintain a window of future token positions and iteratively refine them in parallel. N-grams that
converge are accepted.

- **Paper**: https://arxiv.org/abs/2402.02057 (ICML 2024)
- **GitHub**: https://github.com/hao-ai-lab/LookaheadDecoding
- **Speedup**: 1.5-2.3x (depends on n-gram hit rate)
- **Quality Impact**: Lossless (exact same output distribution)
- **Key Advantage**: No draft model, no training, no data store required

```bash
pip install lookahead-decoding  # or clone from GitHub
```

**When to use**: When you cannot train a draft model and need lossless acceleration.
**When NOT to use**: When EAGLE or Medusa draft heads are available (typically faster).

---

### 9. Dynamic Inference / Model Routing

Route queries to different-sized models or different compute paths based on input difficulty.
Simple queries use a small model; complex queries use the full model.

- **Survey**: https://arxiv.org/abs/2603.04445
- **Approaches**: Confidence-based routing, learned difficulty classifiers, cascading (try small
  model first, escalate on low confidence)
- **Speedup**: 2-5x average cost reduction across mixed workloads
- **Quality Impact**: Matches large-model quality on average if routing is well-calibrated

**Practical Implementation:**

```python
# Simple cascading example
from vllm import LLM, SamplingParams

small_llm = LLM("meta-llama/Meta-Llama-3.1-8B-Instruct")
large_llm = LLM("meta-llama/Meta-Llama-3.1-70B-Instruct")

def route_and_generate(prompt):
    result = small_llm.generate(prompt, SamplingParams(temperature=0.0, max_tokens=256))
    confidence = estimate_confidence(result)  # your confidence metric
    if confidence < 0.8:
        result = large_llm.generate(prompt, SamplingParams(temperature=0.0, max_tokens=256))
    return result
```

**When to use**: Production serving with mixed-difficulty workloads. **When NOT to use**: All
queries are equally difficult; latency-sensitive single-request scenarios.

---

### 10. Knowledge Distillation (Offline -- For Contrast)

Unlike the training-free methods above, knowledge distillation trains a smaller "student" model to
mimic a larger "teacher." Included here for comparison.

| Model | Teacher | Parameters | Performance Retained | Training Cost |
|-------|---------|------------|---------------------|---------------|
| DistilBERT | BERT-base | 66M (vs 110M) | 97% | Moderate |
| TinyLlama-1.1B | LLaMA-2 architecture | 1.1B | Good for size | 3T tokens |
| Minitron-8B | Nemotron-15B | 8B | ~95% | Pruning + distillation |

- **DistilBERT**: https://huggingface.co/docs/transformers/model_doc/distilbert
- **TinyLlama**: https://github.com/jzhang38/TinyLlama
- **Minitron**: https://github.com/NVlabs/Minitron

**Key Difference**: Distillation requires significant training compute but produces a permanently
smaller, faster model. Training-free methods accelerate any existing model immediately.

---

## Part 2: Diffusion Model Acceleration

### 11. Step Reduction (Efficient Solvers)

Use higher-order ODE solvers to achieve high-quality images in fewer denoising steps.

#### 11a. DDIM (Denoising Diffusion Implicit Models)

First-order deterministic solver that enables skipping steps in the diffusion process. Baseline
for all faster solvers.

- **Paper**: https://arxiv.org/abs/2010.02502
- **Steps**: 50-250 (vs 1000 for DDPM)
- **Quality**: Acceptable at 50 steps; degrades below 20

#### 11b. DPM-Solver / DPM-Solver++

Higher-order ODE solver for diffusion models. Uses Taylor expansion to predict multiple steps at
once. DDIM is the first-order special case.

- **Paper**: https://arxiv.org/abs/2211.01095 (NeurIPS 2022 Oral)
- **GitHub**: https://github.com/LuChengTHU/dpm-solver
- **Steps**: 10-25 steps for high quality
- **Quality**: Near-converged at 20 steps; good at 10

```python
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
image = pipe("a photo of a cat", num_inference_steps=20).images[0]
```

#### 11c. LCM (Latent Consistency Models)

Distilled models that generate in 2-8 steps. LCM-LoRA adapters can be applied to any fine-tuned
SD/SDXL model without per-model distillation.

- **Paper**: https://arxiv.org/abs/2310.04378
- **Steps**: 4-8 steps
- **Quality**: Good; slight softness compared to 20-step DPM-Solver++
- **Training Required**: LCM-LoRA adapters are pre-trained; applying them is training-free

```python
from diffusers import StableDiffusionXLPipeline, LCMScheduler

pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
image = pipe("a photo of a cat", num_inference_steps=4, guidance_scale=1.0).images[0]
```

#### 11d. TCD (Trajectory Consistency Distillation)

Improves upon LCM by better preserving the original diffusion trajectory. Generates higher-quality
images in 2-4 steps than LCM.

- **Paper**: https://arxiv.org/abs/2402.19159
- **Steps**: 2-4 steps
- **Quality**: Surpasses LCM and numerical solvers at 4 steps

**When to use**: DPM-Solver++ is the default choice; LCM/TCD for real-time applications needing
<8 steps. **When NOT to use**: If quality at 4 steps is insufficient, use more steps with
DPM-Solver++ rather than forcing fewer steps.

---

### 12. Feature Caching / Reuse

Exploit temporal redundancy between consecutive denoising steps by caching and reusing intermediate
features instead of recomputing them.

#### 12a. DeepCache

Caches high-level U-Net features and reuses them across adjacent steps while only updating
low-level features cheaply.

- **Paper**: https://arxiv.org/abs/2312.00858 (CVPR 2024)
- **GitHub**: https://github.com/horseee/DeepCache
- **Speedup**: 2.3x on SD v1.5 with only 0.05 CLIP Score decline
- **Quality Impact**: Near-lossless

```bash
pip install DeepCache
```

```python
from DeepCache import DeepCacheSDHelper
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
helper = DeepCacheSDHelper(pipe=pipe)
helper.set_params(cache_interval=3, cache_branch_id=0)
helper.enable()
image = pipe("a photo of a castle", num_inference_steps=50).images[0]
```

#### 12b. T-GATE (Temporal Gating Attention)

Caches and reuses cross-attention outputs after a scheduled timestep, since cross-attention
(text conditioning) stabilizes early in the denoising process.

- **Paper**: https://arxiv.org/abs/2404.02747
- **GitHub**: https://github.com/HaozheLiu-ST/T-GATE
- **Speedup**: 10-50% speedup
- **Quality Impact**: Minimal; compatible with DeepCache for stacking

```python
# Integrated in diffusers
from diffusers import StableDiffusionXLPipeline
# See: https://huggingface.co/docs/diffusers/optimization/tgate
```

#### 12c. Spectrum (Adaptive Spectral Feature Forecasting)

Models the evolution of diffusion features using Chebyshev polynomials. Fits coefficients online
via ridge regression and forecasts features at multiple future steps.

- **Paper**: https://arxiv.org/abs/2603.01623
- **Website**: https://hanjq17.github.io/Spectrum/
- **Speedup**: 3.5x on FLUX.1 with only 14 network evaluations
- **Quality Impact**: No measurable quality degradation

#### 12d. FasterDiffusion

Caches and reuses encoder features to parallelize decoder computation across multiple timesteps.

- **Speedup**: 1.5-2x
- **Quality Impact**: Minimal

**When to use**: Always worth trying -- DeepCache is a reliable first choice. Stack with
step-reduction solvers for compound speedup. **When NOT to use**: Very few steps (<8) where
there is no redundancy to exploit.

---

### 13. Token Merging for Diffusion (ToMe)

Merges redundant image tokens (patches) during the forward pass based on cosine similarity,
reducing the number of tokens processed by attention and MLP layers.

- **Paper**: https://arxiv.org/abs/2303.17604 (CVPR 2023 Workshop)
- **GitHub**: https://github.com/facebookresearch/ToMe
- **Speedup**: Up to 2x; up to 5.6x memory reduction with 60% token reduction
- **Quality Impact**: Minimal at moderate merging ratios (20-40%)

```bash
pip install tomesd
```

```python
import tomesd
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
tomesd.apply_patch(pipe, ratio=0.5)  # Merge 50% of tokens
image = pipe("a photo of a dog").images[0]
```

**When to use**: High-resolution generation where attention is the bottleneck. **When NOT to
use**: Low-resolution generation; tasks requiring fine spatial detail.

---

### 14. Distillation for Diffusion (For Contrast)

Like LLM distillation, these require training but produce models that generate in 1-4 steps.

| Model | Base | Steps | Method | Quality |
|-------|------|-------|--------|---------|
| SDXL-Turbo | SDXL | 1-4 | Adversarial Diffusion Distillation | Good |
| SDXL-Lightning | SDXL | 1-4 | Progressive + Adversarial Distillation | Better |
| LCM | SD/SDXL | 4-8 | Consistency Distillation | Good |
| Hyper-SD | SD/SDXL | 1-4 | Score distillation | Very good |

```python
# SDXL-Lightning example
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download

pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
pipe.load_lora_weights(hf_hub_download("ByteDance/SDXL-Lightning", "sdxl_lightning_4step_lora.safetensors"))
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
image = pipe("a cat", num_inference_steps=4, guidance_scale=0.0).images[0]
```

---

### 15. Guidance Distillation

Classifier-free guidance (CFG) requires two forward passes per step (conditional + unconditional).
Guidance distillation trains a single model to approximate the combined output, halving the
compute per step.

- **Paper**: https://arxiv.org/abs/2210.03142
- **Adapter approach (AGD)**: https://arxiv.org/abs/2503.07274 -- trains only ~2% extra parameters
- **Speedup**: 2x per step (eliminates the unconditional pass)
- **Quality Impact**: Matches or slightly exceeds CFG quality
- **Training Required**: Yes -- but AGD requires minimal compute

**When to use**: If guidance scale > 1.0 is required and you control the model. **When NOT to
use**: Models already using guidance_scale=0 or 1 (e.g., distilled models).

---

### 16. Architecture Search / Pruning for Diffusion

Structurally prune the U-Net or DiT to create smaller architectures.

- **DiP-GO**: Few-shot gradient-based pruning for DiT; 4.4x speedup on SD 1.5 without retraining
  (https://arxiv.org/abs/2412.XXXXX, NeurIPS 2024)
- **LAPTOP-Diff**: Layer pruning + distillation for diffusion U-Nets
  (https://arxiv.org/abs/2404.11098)
- **DiTFastAttn**: Identifies and removes redundant attention computations in DiT models
- **DiffNAS**: GPT-4-driven architecture search for optimal U-Net designs
  (https://arxiv.org/abs/2310.04750)

**When to use**: Building production inference pipelines where permanent model modification is
acceptable. **When NOT to use**: Rapid prototyping; you need compatibility with community
fine-tunes.

---

## Comparison Table

| Method | Type | Models Supported | Speedup | Quality Impact | Training Required | Difficulty |
|--------|------|-----------------|---------|----------------|-------------------|------------|
| Speculative Decoding (draft) | LLM | Any causal LM | 1.5-3x | Lossless | No (need draft model) | Easy |
| EAGLE | LLM | LLaMA, Vicuna, Mixtral | 2-6x | Lossless | Yes (head training) | Medium |
| Medusa | LLM | Any causal LM | 2-3x | Lossless | Yes (head training) | Medium |
| LayerSkip | LLM | Supported models | 1.8-2.2x | Lossless | Yes (layer dropout) | Medium |
| CALM (Early Exit) | LLM | T5, encoder-decoder | 2-3x | Minimal | Calibration only | Medium |
| LazyLLM (Token Pruning) | LLM | Any transformer | 1.5-2.5x | Minimal | No | Easy |
| ShortGPT (Layer Skip) | LLM | Any LLM | 1.25x | ~5% drop | No | Easy |
| SLEB (Layer Skip) | LLM | Any LLM | 1.2-1.3x | ~3-5% drop | No | Easy |
| StreamingLLM | LLM | Any causal LM | Memory only | Degrades on long-range | No | Easy |
| H2O (KV Cache) | LLM | Any causal LM | Memory 5-10x | Minimal | No | Easy |
| SnapKV | LLM | Any causal LM | 3.6x decode | Minimal | No | Easy |
| xKV | LLM | Any causal LM | Best tradeoff | Minimal | No | Medium |
| AWQ (INT4) | LLM | Any LLM | 2-3x | <1% | No (PTQ) | Easy |
| GPTQ (INT4) | LLM | Any LLM | 2-3x | <1% | No (PTQ) | Easy |
| GGUF | LLM | Any LLM | CPU viable | <2% | No | Easy |
| FP8 | LLM | Any LLM (Hopper+) | 1.5-2x | <0.5% | No | Easy |
| SageAttention | LLM/Diffusion | Any transformer | 2-5x attn | None | No | Easy |
| LLMLingua | LLM | Any LLM | 1.7-5.7x | <1.5% | No | Easy |
| Lookahead Decoding | LLM | Any causal LM | 1.5-2.3x | Lossless | No | Medium |
| Dynamic Routing | LLM | Any LLM | 2-5x avg | Minimal (avg) | Optional | Medium |
| DPM-Solver++ | Diffusion | Any diffusion | 2-5x steps | Good at 20 steps | No | Easy |
| LCM-LoRA | Diffusion | SD/SDXL | 5-10x steps | Slight softness | No (apply LoRA) | Easy |
| TCD | Diffusion | SD/SDXL | 10-25x steps | Better than LCM | No (apply LoRA) | Easy |
| DeepCache | Diffusion | U-Net models | 2.3x | Near-lossless | No | Easy |
| T-GATE | Diffusion | SD/SDXL/DiT | 1.1-1.5x | Minimal | No | Easy |
| Spectrum | Diffusion | FLUX, DiT | 3.5x | None | No | Medium |
| ToMe | Diffusion | SD/SDXL | Up to 2x | Minimal | No | Easy |
| SDXL-Turbo/Lightning | Diffusion | SDXL | 10-25x steps | Good | Yes (distilled) | Easy |
| Guidance Distillation | Diffusion | Any CFG model | 2x/step | Minimal | Yes (small) | Medium |

---

## Decision Tree

```
I want to speed up my model. What should I try?

[LLM or Diffusion Model?]
 |
 +-- LLM
 |    |
 |    +-- [What's your constraint?]
 |    |    |
 |    |    +-- Memory (can't fit model)
 |    |    |    --> Quantization (AWQ/GPTQ for GPU, GGUF for CPU)
 |    |    |    --> Then try KV cache compression (SnapKV, H2O)
 |    |    |
 |    |    +-- Latency (tokens/sec too slow)
 |    |    |    --> Quantization first (easiest win)
 |    |    |    --> Then speculative decoding (EAGLE > draft model > Lookahead)
 |    |    |    --> Then SageAttention (drop-in)
 |    |    |
 |    |    +-- Throughput (requests/sec too low)
 |    |    |    --> Quantization (smaller model = larger batches)
 |    |    |    --> KV cache compression (SnapKV/xKV for more concurrent requests)
 |    |    |    --> SageAttention
 |    |    |
 |    |    +-- Long context (>8K tokens)
 |    |    |    --> KV cache compression (SnapKV, xKV)
 |    |    |    --> Sparse attention (NSA, SageAttention)
 |    |    |    --> Prompt compression (LLMLingua) for RAG
 |    |    |    --> StreamingLLM for infinite streaming
 |    |    |
 |    |    +-- Willing to accept quality loss
 |    |         --> Layer skipping (ShortGPT/SLEB): permanent 25% speedup
 |    |         --> Token pruning (LazyLLM): dynamic per-input savings
 |    |
 +-- Diffusion Model
      |
      +-- [What's your constraint?]
           |
           +-- Need fewer steps
           |    --> DPM-Solver++ (training-free, try 20 steps first)
           |    --> LCM-LoRA (4-8 steps, apply adapter)
           |    --> TCD (2-4 steps, best few-step quality)
           |    --> SDXL-Lightning/Turbo (1-4 steps, distilled models)
           |
           +-- Same steps, faster per step
           |    --> DeepCache (cache U-Net features, ~2.3x)
           |    --> ToMe (merge tokens, up to 2x)
           |    --> SageAttention (faster attention, 2-5x on attn)
           |    --> T-GATE (cache cross-attention, 10-50%)
           |
           +-- Maximum speed
                --> Combine: LCM/TCD (4 steps) + DeepCache + ToMe + SageAttention
                --> Or: SDXL-Lightning (4 steps) + DeepCache
```

---

## Stacking Guide: Combining Techniques

Many techniques are orthogonal and can be combined for compound speedup:

### LLM Stacking Recipes

| Recipe | Components | Expected Speedup | Notes |
|--------|-----------|-----------------|-------|
| **Basic** | AWQ + vLLM | 2-3x | Easiest setup |
| **Standard** | AWQ + EAGLE + vLLM | 4-6x | Best latency reduction |
| **Long-Context** | AWQ + SnapKV + SageAttention | 3-5x | For >8K contexts |
| **Maximum** | FP8 + EAGLE + SnapKV + SageAttention | 5-10x | Hopper GPU required |
| **Budget** | GGUF Q4_K_M + llama.cpp | 2-3x (CPU viable) | No GPU needed |

### Diffusion Stacking Recipes

| Recipe | Components | Expected Speedup | Notes |
|--------|-----------|-----------------|-------|
| **Basic** | DPM-Solver++ (20 steps) | 2.5x vs DDPM | No quality loss |
| **Fast** | LCM-LoRA (4 steps) + DeepCache | 5-8x | Good quality |
| **Faster** | LCM-LoRA + DeepCache + ToMe (0.5) | 8-12x | Slight quality loss |
| **Fastest** | SDXL-Lightning (4 steps) + SageAttention | 10-15x | Needs distilled model |
| **DiT** | Spectrum + SageAttention | 5-8x on FLUX | Best for DiT models |

### Compatibility Matrix

| | Quantization | Spec. Decoding | KV Cache Compress | Sparse Attn | Token Pruning | Layer Skip |
|---|---|---|---|---|---|---|
| **Quantization** | -- | Yes | Yes | Yes | Yes | Yes |
| **Spec. Decoding** | Yes | -- | Yes | Partial | No | Partial |
| **KV Cache Compress** | Yes | Yes | -- | Yes | Yes | Yes |
| **Sparse Attention** | Yes | Partial | Yes | -- | Yes | Yes |
| **Token Pruning** | Yes | No | Yes | Yes | -- | Partial |
| **Layer Skipping** | Yes | Partial | Yes | Yes | Partial | -- |

**Notes on compatibility:**
- Quantization stacks with everything -- always apply first
- Speculative decoding + token pruning conflict (spec. decoding needs full token set for verification)
- Layer skipping + speculative decoding: LayerSkip is designed for this; ShortGPT + external spec. decoding needs testing
- DeepCache + T-GATE are explicitly compatible and tested together
- ToMe + DeepCache stack well for diffusion models
- LCM-LoRA + DeepCache is a tested, reliable combination

---

## Quick Reference: pip install Commands

```bash
# LLM Serving
pip install vllm                    # vLLM with spec decoding, FP8, AWQ, GPTQ
pip install sglang                  # SGLang with EAGLE support
pip install autoawq                 # AWQ quantization
pip install gptqmodel               # GPTQ quantization
pip install sageattention           # SageAttention drop-in
pip install llmlingua               # Prompt compression
pip install streaming-llm           # StreamingLLM

# Diffusion
pip install diffusers               # HuggingFace diffusers (DPM-Solver++, LCM, TCD built-in)
pip install DeepCache               # DeepCache for U-Net caching
pip install tomesd                  # Token merging for Stable Diffusion

# CPU / Edge
brew install llama.cpp              # GGUF inference on macOS
pip install llama-cpp-python        # Python bindings for llama.cpp
```
