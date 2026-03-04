---
name: hpc-cuda
description: CUDA kernel development, GPU memory patterns, cuBLAS/cuDNN/Thrust, and NVIDIA profiling (nsys/ncu). Use when writing CUDA kernels, optimizing GPU memory access, or profiling GPU performance.
user-invocable: true
---

# HPC CUDA

CUDA kernel development, GPU memory optimization, NVIDIA libraries, and profiling.

**Authoritative Reference:** https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html

## Extension Files

| File | Content |
|------|---------|
| `cuda-kernels.md` | Architecture-aware kernel dev, CCCL/CuTe/TMA, warp ops, cuBLAS/cuDNN/Thrust |
| `cuda-memory.md` | Coalesced access, bank conflicts, shared memory, pinned memory, unified memory |
| `profiling.md` | nsys (system-wide), ncu (kernel-level), torch.profiler, roofline analysis |
| `gpu-memory-management.md` | CUDA caching allocator, OOM diagnosis, memory profiling, fragmentation fixes, using spare memory to increase throughput |

## Key Principle: Use Libraries Before Custom Kernels

```
GEMM/BLAS     → cuBLAS (ALWAYS)
Conv/Pool/RNN → cuDNN (ALWAYS)
Sort/Reduce   → Thrust/CUB (ALWAYS)
Fused custom  → Triton first, CUDA if Triton too slow
Novel algo    → Custom CUDA kernel
Custom GEMM   → CUTLASS/CuTe
```

## Profiling Workflow (MANDATORY before optimizing)

```
Step 1: nvidia-smi dmon -s u          → Is GPU even utilized?
Step 2: nsys profile ... python x.py  → System-wide: CPU/GPU balance, gaps
Step 3: ncu --kernel-name "X" ...     → Kernel deep-dive: throughput, stalls
Step 4: ncu --set roofline ...        → Where on roofline? Correct optimization?
Step 5: Fix → re-profile → compare
```

## Review Checklist (CUDA)

```
□ Memory accesses coalesced? (consecutive threads → consecutive addrs)
□ No bank conflicts in shared memory? (pad arrays: [N][33] not [N][32])
□ Register pressure acceptable? (check with --ptxas-options=-v)
□ Using cuBLAS/cuDNN/Thrust where applicable? (don't reinvent)
□ Correct arch flag? (-arch=sm_90a for Hopper)
□ Profiled with nsys/ncu before optimizing?
□ Kernel launch overhead acceptable? (no tiny kernels in Python loops)
```
