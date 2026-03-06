---
id: torchao
kind: source
title: TorchAO - PyTorch Architecture Optimization
type: official-doc
category: quantization
summary: PyTorch-native library for quantization (INT4, INT8, FP8, NF4), sparsity (2:4), and float8 training with composable APIs.
reliability: official
review_status: reviewed
url: https://github.com/pytorch/ao
tags:
  - torchao
  - pytorch
  - quantization
  - sparsity
  - fp8
  - int4
---

## Key Features
- int4_weight_only(), int8_weight_only(), float8_weight_only() APIs
- Semi-structured 2:4 sparsity via cuSPARSELt
- Float8 training support
- Autoquantization (picks best quant per layer)
- Composable with torch.compile for maximum performance
