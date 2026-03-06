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
