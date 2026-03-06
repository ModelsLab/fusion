---
id: cpu_gpu_coexecution
kind: document
title: CPU-GPU Co-Execution and Offloading Patterns
category: optimization
summary: Strategies for utilizing CPU alongside GPU for LLM inference including weight offloading, KV cache offloading, CPU-based preprocessing, hybrid CPU-GPU inference, and pipeline overlap.
tags:
  - cpu-offloading
  - heterogeneous
  - weight-offloading
  - kv-cache-offloading
  - pinned-memory
  - pcie
source_ids: []
operators:
  - memory
  - general
gpu_families:
  - Ampere
  - Ada
  - Hopper
  - Blackwell
---

# CPU-GPU Co-Execution and Offloading

## When to Use CPU-GPU Co-Execution
```
Model doesn't fit in GPU memory even after quantization
  → Weight offloading (keep some layers on CPU)

KV cache exceeds GPU memory for long contexts
  → KV cache offloading (spill to CPU)

Preprocessing is CPU-intensive
  → Overlap CPU tokenization with GPU computation

Small batch, GPU underutilized
  → CPU handles tokenization/sampling while GPU does compute
```

## Weight Offloading

### Layer-Level Offloading
```python
# HuggingFace Accelerate device_map
from accelerate import infer_auto_device_map, dispatch_model

# Automatically split model between GPU and CPU
device_map = infer_auto_device_map(
    model,
    max_memory={0: "20GiB", "cpu": "64GiB"},
    no_split_module_classes=["LlamaDecoderLayer"]  # keep layers intact
)
# Example result: {layer_0: 'cuda:0', ..., layer_28: 'cuda:0', layer_29: 'cpu', ...}

model = dispatch_model(model, device_map=device_map)
```

### Prefetching Strategy
```python
class OffloadedModel:
    def __init__(self, model, gpu_layers, cpu_layers):
        self.gpu_layers = gpu_layers  # permanently on GPU
        self.cpu_layers = cpu_layers  # on CPU, prefetched
        self.prefetch_stream = torch.cuda.Stream()

    def forward(self, x):
        # Process GPU-resident layers
        for layer in self.gpu_layers:
            x = layer(x)

        # Process offloaded layers with prefetching
        for i, layer in enumerate(self.cpu_layers):
            # Prefetch next layer while current executes
            if i + 1 < len(self.cpu_layers):
                with torch.cuda.stream(self.prefetch_stream):
                    self.cpu_layers[i+1].to('cuda', non_blocking=True)

            # Execute current layer on GPU
            x = layer(x)  # layer was prefetched to GPU

            # Move back to CPU after use
            if i > 0:
                self.cpu_layers[i-1].to('cpu', non_blocking=True)

        return x
```

### Performance Analysis
```
PCIe 4.0 x16: 32 GB/s bidirectional
PCIe 5.0 x16: 64 GB/s bidirectional

Time to transfer one LLaMA 7B layer (BF16):
  Layer size: ~400MB (QKV + O + gate + up + down + norms)
  PCIe 4.0: 400MB / 32 GB/s = 12.5ms
  PCIe 5.0: 400MB / 64 GB/s = 6.25ms

GPU compute for one layer at batch=1:
  ~2ms (memory-bound, small batch)

Conclusion: offloading adds 3-6x latency per offloaded layer
Best case: overlap transfer with compute of previous layer
```

## KV Cache Offloading

### vLLM CPU Offloading
```python
# vLLM supports KV cache offloading to CPU
# When GPU KV blocks are exhausted:
# 1. Evict least-recently-used KV blocks to CPU
# 2. Bring back when sequence needs them

# Configuration
engine_args = EngineArgs(
    model="meta-llama/Llama-3-70B",
    swap_space=32,  # GB of CPU swap space for KV cache
    # vLLM manages block-level swapping
)
```

### Pinned Memory for Fast Transfers
```python
# Pinned (page-locked) memory: ~2x faster CPU↔GPU transfer
# Regular: CPU memory → staging buffer → GPU (two copies)
# Pinned: CPU memory → GPU (direct DMA, one copy)

# Allocate pinned memory for KV cache
kv_cache_cpu = torch.empty(
    (num_blocks, block_size, num_heads, head_dim),
    dtype=torch.float16,
    pin_memory=True  # pinned memory
)

# Async transfer
def offload_kv_block(gpu_block, cpu_slot, stream):
    with torch.cuda.stream(stream):
        cpu_slot.copy_(gpu_block, non_blocking=True)

def load_kv_block(cpu_slot, gpu_block, stream):
    with torch.cuda.stream(stream):
        gpu_block.copy_(cpu_slot, non_blocking=True)
```

## CPU Preprocessing Pipeline

### Tokenization Overlap
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class OverlappedPipeline:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.cpu_executor = ThreadPoolExecutor(max_workers=4)

    async def generate(self, prompts):
        # Stage 1: CPU tokenization (parallel with previous GPU work)
        loop = asyncio.get_event_loop()
        tokens = await loop.run_in_executor(
            self.cpu_executor,
            lambda: self.tokenizer(prompts, padding=True, return_tensors="pt")
        )

        # Stage 2: GPU forward pass
        with torch.no_grad():
            outputs = self.model.generate(**tokens.to('cuda'))

        # Stage 3: CPU detokenization (parallel with next GPU work)
        text = await loop.run_in_executor(
            self.cpu_executor,
            lambda: self.tokenizer.batch_decode(outputs.cpu())
        )
        return text
```

## llama.cpp Hybrid Inference

### GPU Layer Splitting
```bash
# llama.cpp: specify how many layers on GPU
./llama-cli -m model.gguf \
  -ngl 28 \  # 28 of 32 layers on GPU, rest on CPU
  --threads 8  # CPU threads for remaining layers

# Optimal split depends on:
# - GPU memory available
# - Model size per layer
# - PCIe bandwidth
# - CPU compute capability
```

### Metal (Apple Silicon) Co-Execution
```
Apple Silicon: unified memory architecture
  - No PCIe bottleneck (CPU and GPU share same memory)
  - GPU accesses CPU memory at full bandwidth
  - Ideal for weight offloading (zero-copy)

M1 Pro/Max: 200-400 GB/s memory bandwidth shared
M2 Ultra: 800 GB/s
M4 Max: 546 GB/s

llama.cpp on Apple Silicon:
  - All layers "on GPU" = Metal compute
  - CPU layers use NEON SIMD
  - No transfer overhead for shared memory
```

## Optimization Strategies

### Decision Matrix
```
| Scenario | Strategy | Expected Overhead |
|----------|----------|-------------------|
| Model barely doesn't fit | Offload 1-2 layers | 5-15% slower |
| Model 2x GPU memory | Offload 50% layers | 3-5x slower |
| Long context OOM | KV cache swap | 10-30% slower |
| Batch preprocessing | CPU pipeline overlap | Minimal |
| Apple Silicon | Unified memory | <5% overhead |
| Multi-GPU available | Tensor parallel | Much better than offload |
```

### Key Optimizations
```
1. Use pinned memory for all CPU↔GPU transfers
2. Overlap transfers with computation using CUDA streams
3. Prefetch next layer while computing current
4. Keep embedding and LM head on GPU (used every token)
5. Offload middle layers first (less impact on pipeline)
6. For KV cache: block-level granularity, not sequence-level
7. Consider NVMe offloading for very large models (FlexGen)
```

## Practical Offloading Recipes

### Recipe 1: HuggingFace Accelerate Auto Device Map
```bash
pip install accelerate
```
```python
# Automatically split model between GPU and CPU
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-70B-Instruct",
    device_map="auto",           # auto-split across GPU + CPU
    torch_dtype="bfloat16",
    max_memory={0: "22GiB", "cpu": "100GiB"},  # limit per device
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B-Instruct")

# Check where each layer landed
print(model.hf_device_map)
# → {'model.embed_tokens': 0, 'model.layers.0': 0, ..., 'model.layers.60': 'cpu', ...}

# Generate (slow for offloaded layers, but works)
inputs = tokenizer("Hello, world!", return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(output[0]))
```

### Recipe 2: llama.cpp Partial GPU Offloading
```bash
# Run 70B model on 24GB GPU: put 20 of 80 layers on GPU
./llama-cli -m Llama-3.1-70B-Q4_K_M.gguf \
  -ngl 20 \           # 20 layers on GPU (rest on CPU)
  --threads 16 \       # CPU threads for remaining layers
  -p "Explain quantum computing" \
  -n 200

# Finding optimal -ngl:
# Start with -ngl 0 (all CPU), increase until GPU memory is ~90% full
# Monitor with: nvidia-smi -l 1
# Each layer for 70B Q4_K_M ≈ 0.5 GB → 20 layers ≈ 10 GB on GPU

# Performance expectation (RTX 4090, 70B Q4_K_M):
#   -ngl 0:  ~5 tokens/sec (pure CPU, i9-13900K)
#   -ngl 20: ~12 tokens/sec (hybrid)
#   -ngl 35: ~18 tokens/sec (most on GPU)
#   -ngl 80: won't fit (needs ~35 GB, have 24 GB)
```

### Recipe 3: Check If Your Model Fits
```python
# Quick memory check before loading
def will_it_fit(model_name, gpu_memory_gb, dtype="float16"):
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name)

    # Estimate parameter count
    h = config.hidden_size
    L = config.num_hidden_layers
    V = config.vocab_size
    I = getattr(config, 'intermediate_size', h * 4)
    params_B = (L * (4*h*h + 3*h*I + 2*h) + 2*V*h) / 1e9

    bytes_per_param = {"float32": 4, "float16": 2, "bfloat16": 2, "int8": 1, "int4": 0.5}
    model_gb = params_B * bytes_per_param.get(dtype, 2)

    fits = model_gb < gpu_memory_gb * 0.85  # leave 15% for KV cache
    print(f"Model: {model_name}")
    print(f"  Params: {params_B:.1f}B")
    print(f"  Size ({dtype}): {model_gb:.1f} GB")
    print(f"  GPU: {gpu_memory_gb} GB → {'FITS ✓' if fits else 'DOES NOT FIT ✗'}")
    if not fits:
        for q in ["int8", "int4"]:
            q_gb = params_B * bytes_per_param[q]
            if q_gb < gpu_memory_gb * 0.85:
                print(f"  → Would fit with {q}: {q_gb:.1f} GB")
                break
        else:
            print(f"  → Needs offloading or multi-GPU")
    return fits

will_it_fit("meta-llama/Llama-3.1-8B-Instruct", 24)   # FITS
will_it_fit("meta-llama/Llama-3.1-70B-Instruct", 24)   # DOES NOT FIT → try int4
will_it_fit("meta-llama/Llama-3.1-70B-Instruct", 80)   # FITS on H100
```
