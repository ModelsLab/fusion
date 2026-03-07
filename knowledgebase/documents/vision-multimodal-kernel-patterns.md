---
id: vision_multimodal_kernel_patterns
kind: document
title: Vision Transformer and Multimodal Kernel Patterns
category: kernel
summary: Kernel optimization patterns for vision transformers (ViT), diffusion models (DiT/Stable Diffusion), vision-language models (LLaVA, Qwen-VL), and multimodal architectures including patch embedding, 2D attention, and cross-attention kernels.
tags:
  - vision-transformer
  - vit
  - diffusion
  - dit
  - multimodal
  - llava
  - cross-attention
  - patch-embedding
source_ids: []
operators:
  - attention
  - convolution
  - matmul
  - layernorm
gpu_families:
  - Ampere
  - Ada
  - Hopper
  - Blackwell
---

## 1. Vision Transformer (ViT) Kernels

### Patch Embedding: Conv2d vs Linear Projection

ViT converts an image into a sequence of patch tokens. The two dominant implementations
have different kernel profiles.

**Conv2d approach** uses a single strided convolution with kernel_size=patch_size and
stride=patch_size. On GPU this maps to an implicit GEMM via cuDNN. For typical ViT-B/16
on 224x224 input the output is (B, 768, 14, 14) which is then flattened to (B, 196, 768).

**Linear projection approach** reshapes the image into explicit patches first, then
applies a batched matmul. This avoids cuDNN dispatch overhead for small spatial sizes
but loses hardware-optimized im2col paths.

```
Conv2d path (preferred for GPU):
  Input: (B, 3, 224, 224)
    -> nn.Conv2d(3, 768, kernel_size=16, stride=16)
    -> (B, 768, 14, 14)
    -> flatten + transpose -> (B, 196, 768)

Linear path (preferred for compile/export):
  Input: (B, 3, 224, 224)
    -> unfold to (B, 196, 3*16*16)
    -> nn.Linear(768, 768)
    -> (B, 196, 768)
```

Decision rule: use Conv2d on Ampere+ when batch size > 1. Use linear projection when
targeting torch.compile or graph capture since strided convolutions can cause graph breaks.

### 2D Position Embeddings

Absolute learned embeddings are a simple (1, N+1, D) parameter added to patch tokens.
The kernel cost is negligible. Rotary 2D embeddings (used in EVA, InternVL) require
computing sin/cos tables for (row, col) pairs and applying them per-head. Fuse the 2D
RoPE computation into the attention kernel to avoid a separate pass over Q and K.

### CLS Token Handling

Prepending a CLS token increases sequence length by 1 (196 -> 197). This creates an
odd sequence length that misaligns tensor core tiles (multiples of 8 or 16). Two options:

1. Pad to 200 tokens (next multiple of 8) and mask the padding in attention.
2. Remove CLS and use global average pooling over patch tokens instead (DINOv2 style).

Option 2 is strictly faster and avoids padding waste.

## 2. Diffusion Model Kernels

### DiT Architecture: adaLN-Zero

DiT replaces standard LayerNorm with adaptive LayerNorm-Zero (adaLN-Zero). Each block
produces six modulation parameters (gamma1, beta1, alpha1, gamma2, beta2, alpha2) from
the timestep embedding via a single linear projection, then applies scale/shift/gate:

```
Architecture per DiT block:
  t_emb -> SiLU -> Linear(D, 6*D) -> chunk into (gamma1, beta1, alpha1, gamma2, beta2, alpha2)

  x = LayerNorm(x)
  x = gamma1 * x + beta1          # scale and shift
  x = Attention(x)
  x = alpha1 * x                  # gate
  x = x + residual

  # repeat pattern for MLP with gamma2, beta2, alpha2
```

Fuse the scale/shift/gate into a single Triton kernel that reads the modulation vector
once and applies all three operations in a single pass over the hidden state.

```python
@triton.jit
def adaln_zero_fwd(X, Gamma, Beta, Alpha, Out,
                   N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask)
    g = tl.load(Gamma + cols, mask=mask)
    b = tl.load(Beta + cols, mask=mask)
    a = tl.load(Alpha + cols, mask=mask)

    # fused layernorm + scale + shift (norm omitted for brevity)
    x_norm = (x - tl.sum(x, axis=0) / N)  # simplified
    out = a * (g * x_norm + b)
    tl.store(Out + row * N + cols, out, mask=mask)
```

### U-Net Attention in Stable Diffusion

Stable Diffusion U-Net uses both self-attention and cross-attention at multiple
resolutions. The attention maps have these typical shapes:

```
Resolution   Seq Len   Heads   Head Dim   Memory (fp16)
64x64        4096      8       40         ~100 MB
32x32        1024      8       80         ~13 MB
16x16        256       8       160        ~1.6 MB
8x8          64        8       160        ~0.1 MB
```

The 64x64 resolution is the bottleneck. FlashAttention-2 eliminates the O(N^2) memory
for the 4096-length attention. Without it, a single attention layer at 64x64 with
batch=2 and classifier-free guidance (effective batch=4) requires ~400 MB just for
attention scores.

### Timestep Conditioning

Timestep embeddings use sinusoidal encoding followed by a two-layer MLP. This is
computed once per forward pass and broadcast to every block. Cache the timestep
embedding computation and avoid recomputing it in each block.

## 3. FlashAttention for Images

### 2D Tiling Strategies

Standard FlashAttention tiles along the sequence dimension. For images, the sequence
has inherent 2D spatial structure. Two tiling approaches:

```
Standard 1D tiling:
  Tiles of size BLOCK_M along flattened sequence
  [patch_0, patch_1, ..., patch_15 | patch_16, ... | ...]
  No spatial locality guarantee within a tile

2D-aware tiling:
  Tiles correspond to spatial blocks in the image
  Row-major: consecutive patches in a row are in the same tile
  Hilbert curve: space-filling curve preserves 2D locality better
```

2D-aware tiling improves cache hit rates for local attention patterns. For global
attention (standard ViT) the difference is minimal. For windowed or local attention
patterns the difference is significant.

### Windowed Attention for High-Resolution Images

High-resolution vision models (1024x1024 = 4096 patches at patch_size=16) use windowed
attention to reduce cost from O(N^2) to O(N*W) where W is the window size.

```
Swin-style shifted window attention:
  Window size: 7x7 = 49 tokens per window
  Image 224x224 with patch 4: 56x56 = 3136 tokens -> 64 windows
  Each window: standard attention on 49 tokens (trivially fast)
  Shifted windows in alternating layers for cross-window communication

  Cost: 64 * O(49^2) vs O(3136^2)
  Speedup: ~64x fewer attention FLOPs
```

For FlashAttention integration, each window maps to a single FlashAttention call with
short sequence length. Batch all windows together as a single batched attention call:

```python
# Reshape (B, H*W, D) -> (B * num_windows, window_size, D)
# Run FlashAttention on the reshaped tensor
# Reshape back
```

## 4. Vision-Language Models

### LLaVA Architecture

```
LLaVA pipeline:
  Image -> ViT Encoder -> Image Tokens (576 tokens for 336x336)
        -> MLP Projector (2-layer) -> Projected Tokens (576 x LLM_dim)

  Text  -> Tokenizer -> Text Tokens

  [Image Tokens | Text Tokens] -> LLM (e.g., Llama-2/3) -> Output

  Total sequence to LLM: 576 (image) + N (text) tokens
```

Key kernel considerations:

1. **ViT encoder** runs once per image. Optimize independently from the LLM.
2. **MLP projector** is two linear layers with GELU. Trivial cost but must match
   LLM precision (e.g., FP16/BF16/FP8).
3. **Prefill** processes 576 image tokens + text tokens together. The image portion
   is fixed-length and benefits from static shape optimization.
4. **KV cache** for image tokens is computed once and reused across all generated tokens.

### Cross-Attention Patterns

Models like Flamingo and Qwen-VL use cross-attention instead of concatenation:

```
Cross-attention pattern:
  Q = LLM hidden states (text)
  K, V = Vision encoder output (image tokens)

  Attention: (B, H, N_text, D) x (B, H, N_image, D)^T -> (B, H, N_text, N_image)

  N_text varies (autoregressive), N_image is fixed per image
```

Cross-attention kernels have asymmetric sequence lengths. FlashAttention handles this
natively. The key optimization is to precompute K and V projections for image tokens
once and cache them across all decoder layers that use cross-attention.

### Image Token Handling in Mixed Sequences

When images and text are interleaved (multi-image conversations), the attention mask
must correctly handle which text tokens can attend to which image tokens. Use a block-
diagonal attention mask:

```
Sequence: [IMG1_tok0 ... IMG1_tok575 | TEXT0 ... TEXTn | IMG2_tok0 ... IMG2_tok575 | ...]

Attention mask (causal with image blocks):
  - Image tokens attend to all tokens in their own image block
  - Text tokens attend to all preceding image and text tokens
  - Second image block is fully visible to subsequent text
```

## 5. Stable Diffusion Optimization

### VAE Encoder/Decoder

The VAE is often the second-largest latency contributor after the U-Net. Key patterns:

1. **Channel-last memory format**: VAE convolutions benefit from NHWC layout on
   Ampere+. Use `model.to(memory_format=torch.channels_last)`.
2. **Tiled VAE**: For high-resolution outputs (1024x1024+), decode the latent in
   overlapping tiles to avoid OOM. Each tile is a standard VAE decode.
3. **FP16 VAE**: The VAE is sensitive to FP16 numerical issues in the decoder.
   Use BF16 on Ampere+ or keep the decoder in FP32 with only the encoder in FP16.

### CLIP Text Encoder

The CLIP text encoder processes 77 tokens through a 12-layer transformer. It is
lightweight relative to the U-Net but runs on every denoising step if using
prompt-embedding caching incorrectly.

Optimization: compute text embeddings once before the denoising loop, not inside it.

### U-Net Optimization Summary

```
Component          % Latency    Optimization
-----------        ---------    ------------
ResNet blocks      ~30%         Conv2d fusion, channels_last, torch.compile
Self-attention     ~25%         FlashAttention-2, xformers
Cross-attention    ~20%         FlashAttention-2, cached K/V from CLIP
GroupNorm          ~10%         Fused GroupNorm+SiLU kernel
Timestep MLP       ~5%         Cache across blocks
Upsampling         ~10%         Fused interpolate+conv
```

### Classifier-Free Guidance Batching

CFG requires two forward passes (conditional + unconditional). Batch them together:

```python
# Naive: two sequential forward passes
cond_out = unet(latent, t, cond_embedding)
uncond_out = unet(latent, t, uncond_embedding)
out = uncond_out + guidance_scale * (cond_out - uncond_out)

# Optimized: single batched forward pass
latent_input = torch.cat([latent, latent], dim=0)
embedding_input = torch.cat([cond_embedding, uncond_embedding], dim=0)
both_out = unet(latent_input, t, embedding_input)
cond_out, uncond_out = both_out.chunk(2)
out = uncond_out + guidance_scale * (cond_out - uncond_out)
```

The batched approach doubles the effective batch size through the U-Net, improving
GPU utilization at the cost of 2x memory. On high-VRAM GPUs (24 GB+) this is always
the correct choice.

## 6. Image Token Compression

### Token Merging (ToMe)

ToMe reduces token count by merging similar tokens between attention layers:

```
ToMe algorithm per layer:
  1. Partition tokens into two sets A and B (alternating)
  2. Compute cosine similarity between all (a, b) pairs using keys
  3. For each token in A, find most similar token in B
  4. Merge top-r pairs by averaging their values
  5. Token count reduces by r per layer

  Example: ViT-B with 196 tokens, r=8 per layer, 12 layers
  Final tokens: 196 - (8 * 12) = 100 tokens
  Speedup: ~1.5-2x with <0.5% accuracy loss on ImageNet
```

The similarity computation is the bottleneck. Use a fused kernel that computes
pairwise cosine similarity and top-r selection in a single pass.

### Adaptive Token Pruning

Instead of fixed-r merging, prune tokens based on attention scores:

```
After each attention layer:
  cls_attention = attn_weights[:, :, 0, 1:]  # CLS attention to all patches
  importance = cls_attention.mean(dim=1)      # average across heads
  keep_idx = importance.topk(k).indices       # keep top-k tokens
  x = x[:, keep_idx]                         # prune
```

This is more aggressive than ToMe but requires careful tuning of the keep ratio
per layer. Early layers should keep more tokens; later layers can be more aggressive.

### Dynamic Resolution

Models like NaViT and Qwen-VL support variable input resolutions by packing
multiple images of different sizes into a single batch using sequence packing:

```
Image A: 224x224 -> 196 tokens
Image B: 448x224 -> 392 tokens
Image C: 224x448 -> 392 tokens

Packed sequence: [A_tokens | B_tokens | C_tokens] = 980 tokens
Attention mask: block-diagonal (each image attends only to itself)

Benefits: no padding waste, native multi-resolution support
Requires: FlashAttention with variable-length sequences (varlen API)
```

## 7. Video Model Kernels

### Temporal Attention

Video transformers add temporal attention across frames. Two patterns:

```
Factored attention (TimeSformer, ViViT):
  1. Spatial attention: each frame independently, (B*T, H*W, D)
  2. Temporal attention: each spatial position across frames, (B*H*W, T, D)

  Cost: O(T * (H*W)^2) + O(H*W * T^2)
  vs joint: O((T*H*W)^2)

  For T=16, H*W=196: factored is ~196x cheaper

Joint space-time attention (VideoMAE):
  Full attention over (T*H*W) tokens
  Only feasible for short clips or with windowed attention
```

### 3D Convolutions

Video models using 3D convolutions (C3D, SlowFast) have kernels of shape
(C_out, C_in, T, H, W). cuDNN supports 3D convolutions but they are significantly
slower than 2D equivalents.

Optimization: factorize 3D conv into (1xHxW) spatial + (Tx1x1) temporal convolutions.
This is the (2+1)D decomposition from R(2+1)D and reduces parameters and FLOPs while
often improving accuracy.

### Frame-Level Parallelism

For inference, process frames in parallel when possible:

```
Embarrassingly parallel (per-frame encoder):
  frames = [f0, f1, ..., f15]  # 16 frames
  # Batch all frames as a single ViT forward pass
  tokens = vit_encoder(torch.stack(frames))  # (16, 196, 768)

Temporal fusion (requires sequential):
  # Temporal attention must process all frames together
  # Use FlashAttention with T as the sequence dimension
```

## 8. Quantization for Vision Models

### INT8 Conv2d

INT8 convolutions on Ampere+ use the INT8 tensor cores. The challenge for vision
models is the activation distribution:

```
Problem: Vision model activations have outlier channels (similar to LLMs)
  - ViT attention projections: ~5% channels have 10x larger magnitude
  - Conv2d early layers: input is normalized [0,1] but intermediate features vary

Solutions:
  1. Per-channel quantization for weights (standard)
  2. Per-token quantization for activations (SmoothQuant-style)
  3. Dynamic quantization: compute scale factors at runtime

INT8 Conv2d performance on RTX 4090:
  Layer              FP16      INT8      Speedup
  Conv2d 3->64       0.12ms    0.08ms    1.5x
  Conv2d 64->128     0.45ms    0.22ms    2.0x
  Conv2d 256->256    1.20ms    0.55ms    2.2x
```

### FP8 Attention in Vision

FP8 (E4M3 for forward, E5M2 for backward) is available on Ada, Hopper, and Blackwell.
Vision attention benefits less than LLM attention from FP8 because sequence lengths
are shorter (196-1024 vs 2048-128K), making attention less memory-bound.

FP8 is most impactful for:
- DiT models with 4096-token sequences (64x64 latent attention)
- High-resolution ViTs (1024x1024 = 4096 patches)
- Video models with long temporal sequences

### Post-Training Quantization Challenges

Vision models are harder to quantize post-training than LLMs:

1. **Calibration data sensitivity**: image distributions vary more than text.
   Use diverse calibration sets (1000+ images from varied categories).
2. **First and last layer sensitivity**: patch embedding and final classifier
   should remain in FP16/BF16.
3. **Normalization interaction**: LayerNorm/GroupNorm after quantized layers
   can amplify quantization error. Keep normalization in higher precision.

## 9. Multi-Resolution Handling

### Dynamic Image Sizes

Fixed-resolution ViTs waste compute on small images and lose detail on large ones.
Dynamic resolution strategies:

```
Strategy 1: Resize + Pad (simple, wasteful)
  Input: 640x480
  Resize to: 224x224 (loses aspect ratio) or pad to 224x224 (wastes tokens)

Strategy 2: Adaptive patch count (NaViT)
  Input: 640x480 with patch_size=16
  Patches: 40x30 = 1200 tokens (no resize needed)
  Position: 2D interpolated position embeddings

Strategy 3: Multi-crop (InternVL, LLaVA-NeXT)
  Input: 640x480
  Split into: 2x2 grid of 320x240 crops, each resized to 336x336
  Total tokens: 4 * 576 = 2304 image tokens + 1 thumbnail (576) = 2880
```

### Aspect Ratio Preservation

```
Aspect-ratio-aware binning:
  Predefined aspect ratios: [1:1, 4:3, 3:4, 16:9, 9:16, 2:1, 1:2]
  Predefined resolutions per ratio:
    1:1  -> 336x336, 672x672
    4:3  -> 448x336
    16:9 -> 672x384

  For input image:
    1. Compute aspect ratio
    2. Find closest predefined ratio
    3. Resize to matching resolution
    4. Compute patches (variable token count)

  Kernel implication: variable sequence lengths require
  FlashAttention varlen or padding
```

### Padding Strategies

When batch processing images of different sizes, padding is necessary:

```
Right-padding with attention mask:
  Image A: 196 tokens -> [tok0 ... tok195 | PAD ... PAD]  (pad to 400)
  Image B: 392 tokens -> [tok0 ... tok391 | PAD ... PAD]  (pad to 400)

  Wasted compute: (400-196)/400 = 51% for Image A

Sequence packing (preferred):
  Pack A and B into a single sequence: [A_tokens | B_tokens] = 588 tokens
  Use varlen FlashAttention with cu_seqlens = [0, 196, 588]
  Zero wasted compute
```

## 10. Performance Comparison and Decision Trees

### Kernel Selection by Model Type

```
Model Type          Best Attention        Best Precision    Key Kernel
-----------         ---------------       --------------    ----------
ViT-B/16 (224)      FlashAttention-2      INT8 (PTQ)        Fused patch embed
ViT-L/14 (336)      FlashAttention-2      FP8 (Ada/Hopper+)  Fused patch embed
DiT-XL (256)        FlashAttention-2      BF16/FP8          Fused adaLN-Zero
SD 1.5 U-Net        xformers/Flash-2      FP16              Fused GroupNorm+SiLU
SDXL U-Net          FlashAttention-2      BF16              CFG batching
LLaVA-1.5           FlashAttention-2      AWQ INT4 (LLM)    Image token caching
Qwen-VL             FlashAttention-2      GPTQ INT4 (LLM)   Cross-attn K/V cache
Video-ViT (16f)     FlashAttention-2      BF16              Factored temporal attn
```

### Optimization Decision Tree

```
START: Vision/Multimodal Model Optimization
  |
  +-> Is model a diffusion model (SD, DiT)?
  |     YES -> Apply CFG batching
  |          -> Fuse adaLN-Zero or GroupNorm+SiLU
  |          -> Use FlashAttention for 64x64 resolution attention
  |          -> Cache timestep embeddings across blocks
  |          -> Cache CLIP text embeddings outside denoising loop
  |          -> Try channels_last memory format for convolutions
  |          -> Try torch.compile on U-Net/DiT
  |
  +-> Is model a vision encoder (ViT, DINOv2)?
  |     YES -> Remove CLS token, use mean pooling
  |          -> Pad sequence to multiple of 8
  |          -> Use FlashAttention (even for 196 tokens, less memory)
  |          -> INT8 quantization for deployment
  |          -> Consider ToMe for 1.5-2x speedup with accuracy trade-off
  |
  +-> Is model a VLM (LLaVA, Qwen-VL)?
  |     YES -> Optimize vision encoder and LLM independently
  |          -> Cache image KV in the LLM KV cache
  |          -> Quantize LLM backbone (AWQ/GPTQ INT4)
  |          -> Keep vision encoder in FP16/BF16
  |          -> Use multi-crop for high-res instead of upscaling ViT
  |
  +-> Is model a video model?
        YES -> Use factored spatial+temporal attention
             -> Batch frames through spatial encoder
             -> FlashAttention for temporal dimension
             -> Consider frame subsampling for inference speed
```

### GPU Family Recommendations

```
GPU Family    VRAM     Best Diffusion Config          Best VLM Config
----------    ----     ---------------------          ---------------
Ampere        24 GB    FP16, Flash-2, CFG batch       INT4 LLM + FP16 ViT
Ada           24 GB    BF16/FP8, Flash-2, CFG batch   FP8 LLM + FP16 ViT
Hopper        80 GB    FP8, Flash-2, torch.compile    FP8 LLM + FP16 ViT
Blackwell     96 GB    FP8/FP4, Flash-3, compile      FP8 LLM + FP8 ViT
```

### Memory Budget Planning

```
Component               ViT-L/14     SD 1.5       LLaVA-13B    DiT-XL
---------               --------     ------       ---------    ------
Model weights (FP16)    0.6 GB       1.7 GB       26 GB        1.2 GB
KV cache                0.01 GB      N/A          2-8 GB       N/A
Activations (peak)      0.2 GB       3.5 GB       1.5 GB       2.0 GB
Attention scores*       0.01 GB      0.8 GB       0.5 GB       0.3 GB
Total (no Flash)        0.8 GB       6.0 GB       30 GB        3.5 GB
Total (with Flash)      0.8 GB       4.5 GB       29.5 GB      3.2 GB

* Attention score memory eliminated by FlashAttention (shown for reference)
```

## Summary of Key Patterns

1. **Fuse modulation kernels**: adaLN-Zero scale/shift/gate in one pass.
2. **Batch CFG**: always batch conditional and unconditional passes together.
3. **Cache embeddings**: timestep embeddings, CLIP text embeddings, image KV.
4. **Use FlashAttention everywhere**: even short sequences benefit from reduced memory.
5. **Channels-last for convolutions**: free speedup on Ampere+.
6. **Sequence packing over padding**: eliminates wasted compute for variable sizes.
7. **Factored attention for video**: spatial and temporal separately.
8. **Quantize asymmetrically in VLMs**: aggressive quantization on LLM, conservative on vision encoder.
9. **Token compression**: ToMe or pruning for deployment with accuracy budget.
10. **Profile before optimizing**: the bottleneck differs by model type (attention for DiT, convolutions for U-Net, LLM decode for VLMs).
