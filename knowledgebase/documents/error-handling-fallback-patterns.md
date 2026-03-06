---
id: error_handling_fallback_patterns
kind: document
title: Error Handling and Fallback Patterns for GPU Kernels
category: engineering
summary: Patterns for handling GPU kernel failures, OOM recovery, numerical instability detection, graceful degradation, and fallback strategies in production inference systems.
tags:
  - error-handling
  - oom
  - fallback
  - numerical-stability
  - production
  - reliability
source_ids: []
operators:
  - general
gpu_families:
  - Ampere
  - Ada
  - Hopper
  - Blackwell
---

# Error Handling and Fallback Patterns

## GPU Error Categories

### 1. Out of Memory (OOM)
```python
# Pattern: try optimized path, fall back on OOM
def run_with_oom_fallback(model, input_ids, **kwargs):
    try:
        return model.forward(input_ids, **kwargs)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        gc.collect()

        # Fallback strategies (in order):
        # 1. Reduce batch size
        if input_ids.shape[0] > 1:
            return run_in_chunks(model, input_ids, chunk_size=1, **kwargs)

        # 2. Enable activation checkpointing
        model.gradient_checkpointing_enable()
        return model.forward(input_ids, **kwargs)

        # 3. Use quantized KV cache
        # 4. Reduce max_seq_len
        # 5. Offload to CPU
```

### 2. Numerical Errors (NaN/Inf)
```python
# Detection in forward pass
def checked_forward(model, x):
    output = model(x)

    if torch.isnan(output.logits).any():
        # Log diagnostic info
        log_tensor_stats(x, "input")
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                logger.error(f"NaN in parameter: {name}")

        # Fallback: recompute in FP32
        with torch.amp.autocast(device_type='cuda', enabled=False):
            output = model(x.float())

    return output

# Common NaN sources:
# - Softmax overflow (logits > 88 in FP16)
# - Log of zero (cross-entropy with hard labels)
# - Division by zero (LayerNorm with zero variance)
# - Gradient explosion (learning rate too high)
```

### 3. CUDA Errors
```python
# CUDA illegal memory access, device-side assert
# These are usually unrecoverable without process restart

# Prevention: bounds checking in custom kernels
@triton.jit
def safe_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N  # ALWAYS mask to prevent OOB access
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x, mask=mask)

# Recovery pattern for serving:
def serve_with_recovery(request):
    try:
        return generate(request)
    except RuntimeError as e:
        if "CUDA" in str(e):
            logger.error(f"CUDA error: {e}")
            # Reset CUDA state
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            # Retry once
            try:
                return generate(request)
            except:
                return error_response("GPU error, request failed")
```

## Kernel Fallback Chains

### Attention Fallback
```python
def attention_with_fallback(q, k, v, causal=True):
    # Try FlashAttention first (fastest)
    try:
        from flash_attn import flash_attn_func
        return flash_attn_func(q, k, v, causal=causal)
    except (ImportError, RuntimeError):
        pass

    # Try xformers memory-efficient attention
    try:
        from xformers.ops import memory_efficient_attention
        return memory_efficient_attention(q, k, v, attn_bias=...)
    except (ImportError, RuntimeError):
        pass

    # Try PyTorch SDPA (built-in, multiple backends)
    try:
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    except RuntimeError:
        pass

    # Final fallback: naive attention (slow but always works)
    scale = q.shape[-1] ** -0.5
    attn = torch.matmul(q * scale, k.transpose(-2, -1))
    if causal:
        mask = torch.triu(torch.ones_like(attn), diagonal=1).bool()
        attn.masked_fill_(mask, float('-inf'))
    attn = torch.softmax(attn, dim=-1)
    return torch.matmul(attn, v)
```

### GEMM Fallback
```python
def gemm_with_fallback(a, b, dtype=None):
    # Try FP8 GEMM (Hopper+)
    if dtype == torch.float8_e4m3fn and hasattr(torch, '_scaled_mm'):
        try:
            return torch._scaled_mm(a, b, ...)
        except RuntimeError:
            pass

    # Try cuBLAS with tensor cores
    try:
        return torch.matmul(a, b)
    except RuntimeError:
        pass

    # CPU fallback (last resort)
    return torch.matmul(a.cpu(), b.cpu()).cuda()
```

### Quantization Fallback
```
Quantization fallback chain:
1. FP8 (best quality, Hopper+ only)
   → if not supported: fall back to...
2. AWQ INT4 with Marlin kernel (fast, good quality)
   → if CUDA error or quality too low: fall back to...
3. GPTQ INT4 (widely compatible)
   → if still issues: fall back to...
4. INT8 SmoothQuant (minimal quality loss)
   → if memory still insufficient: fall back to...
5. BF16 with model parallelism
```

## Production Reliability Patterns

### Health Check Kernel
```python
def gpu_health_check():
    """Run before serving to verify GPU is functional."""
    try:
        # Basic compute test
        a = torch.randn(1024, 1024, device='cuda')
        b = torch.matmul(a, a.t())
        assert not torch.isnan(b).any()

        # Memory test
        free_mem = torch.cuda.mem_get_info()[0]
        test_tensor = torch.empty(free_mem // 2, dtype=torch.uint8, device='cuda')
        del test_tensor
        torch.cuda.empty_cache()

        return True
    except Exception as e:
        logger.error(f"GPU health check failed: {e}")
        return False
```

### Graceful Degradation for Serving
```python
class AdaptiveServer:
    def __init__(self, model):
        self.model = model
        self.max_batch = 64
        self.max_seq_len = 8192
        self.oom_count = 0

    def handle_request(self, request):
        try:
            result = self.generate(request)
            self.oom_count = max(0, self.oom_count - 1)  # recover
            return result
        except torch.cuda.OutOfMemoryError:
            self.oom_count += 1
            torch.cuda.empty_cache()

            # Progressive degradation
            if self.oom_count > 3:
                self.max_batch = max(1, self.max_batch // 2)
                logger.warning(f"Reducing max_batch to {self.max_batch}")
            if self.oom_count > 5:
                self.max_seq_len = max(512, self.max_seq_len // 2)
                logger.warning(f"Reducing max_seq_len to {self.max_seq_len}")

            # Retry with reduced parameters
            return self.generate(request)
```

### CUDA Graph Error Recovery
```python
def capture_cuda_graph_safe(model, example_input):
    """Capture CUDA graph with fallback to eager mode."""
    try:
        # Warmup
        for _ in range(3):
            model(example_input)
        torch.cuda.synchronize()

        # Capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = model(example_input)

        return graph, output
    except Exception as e:
        logger.warning(f"CUDA graph capture failed: {e}, using eager mode")
        return None, None  # caller uses eager mode

def run_with_graph(graph, model, input_tensor, static_input):
    if graph is not None:
        static_input.copy_(input_tensor)
        graph.replay()
        return static_output
    else:
        return model(input_tensor)  # eager fallback
```

## Debugging GPU Issues

### Common Error Messages and Solutions
```
"CUDA out of memory":
  → Reduce batch size, enable gradient checkpointing, use quantization

"CUDA error: device-side assert triggered":
  → Index out of bounds in kernel. Set CUDA_LAUNCH_BLOCKING=1 to get exact location

"CUDA error: an illegal memory access was encountered":
  → Buffer overflow in custom kernel. Check mask bounds, shared memory size

"cuDNN error: CUDNN_STATUS_NOT_SUPPORTED":
  → Unsupported combination of dtype/dimensions. Check cuDNN docs for supported configs

"NCCL error: unhandled system error":
  → Network issue in multi-GPU. Check NCCL_DEBUG=INFO, verify GPU interconnect

"Triton compilation failed":
  → Check constexpr values, block sizes must be power of 2, verify BLOCK_SIZE <= problem size
```
