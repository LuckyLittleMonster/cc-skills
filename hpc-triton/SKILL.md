---
name: hpc-triton
description: Triton kernel development — block-level GPU programming, autotuning, torch.compile as reference code. Use when writing Triton kernels or optimizing fused GPU operations.
user-invocable: true
---

# HPC Triton

Triton kernel development and optimization.

## Extension Files

| File | Content |
|------|---------|
| `triton-kernels.md` | Basics, autotuning configs, fused softmax/attention/reduction patterns, torch.compile output as reference, debugging, Triton vs CUDA decision |

## Key Technique: torch.compile as Reference

```python
# Step 1: Write operation in PyTorch
compiled = torch.compile(my_op)
out = compiled(x, y)

# Step 2: Find generated Triton code
# TORCH_COMPILE_DEBUG=1 python script.py
# Files at: ~/.cache/torch_inductor/<hash>/

# Step 3: Study the generated kernel as starting point
# Learn: block sizes, reduction tiling, numeric stability tricks, autotune configs
```

## When Triton vs CUDA

| Scenario | Use Triton | Use CUDA |
|----------|-----------|----------|
| Fused elementwise + reduction | Yes | Overkill |
| Custom attention variant | Yes | If Triton too slow |
| Warp-level intrinsics | No | Yes |
| Quick prototype | Yes (Python!) | No |
| Maximum absolute performance | Maybe | Yes (diminishing returns) |
| PyTorch integration | Seamless | Via C++ extensions |

## Review Checklist (Triton)

```
□ Block sizes are powers of 2?
□ Autotune configs cover relevant block size range?
□ Masked loads for boundary handling?
□ Accumulation in float32, store in float16/bf16?
□ Reduction within block before atomic_add?
□ Benchmarked against torch.compile baseline?
□ num_stages tuned for memory latency hiding?
```
