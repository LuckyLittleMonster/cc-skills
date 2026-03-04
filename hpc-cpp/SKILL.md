---
name: hpc-cpp
description: High-performance C++ — compiler optimization, OpenMP, Boost, memory layout, cache blocking. Use when writing or reviewing C++ code for HPC, PyTorch C++ extensions, or system-level performance code.
user-invocable: true
---

# HPC C++

High-performance C++ patterns for HPC. Focus on what the compiler CANNOT optimize.

## Extension Files

| File | Content |
|------|---------|
| `cpp-performance.md` | Compiler flags, OpenMP patterns, Boost libraries, memory layout (SoA/AoSoA), cache blocking, false sharing, lock-free, PyTorch C++ extensions |
| `python-extensions.md` | Compile C++/CUDA into .so for Python: torch cpp_extension (JIT/AOT), pybind11, ctypes, cffi, build recipes, debugging |

## Core Principle

`g++ -O3` handles: loop unrolling, vectorization, inlining, constant folding, branch prediction.

**You focus on what `-O3` CANNOT do:**
- Algorithm/data structure choices
- Memory layout (AoS → SoA → AoSoA)
- Parallelism (OpenMP, std::thread)
- Cache-aware blocking
- Lock-free concurrency
- False sharing prevention

## Quick Reference

```bash
# Development
g++ -O0 -g -fsanitize=address,undefined -Wall -Wextra

# Production
g++ -O3 -march=native -flto=auto -fopenmp -DNDEBUG

# GH200 ARM
g++ -O3 -march=armv9-a+sve2 -mtune=neoverse-v2 -fopenmp

# Check vectorization failures (fix these!)
g++ -O3 -march=native -fopt-info-vec-missed 2>&1 | head -50
```

## Review Checklist (C++)

```
□ Using -O3 -march=native -flto? (don't manually optimize what compiler does)
□ Memory layout SoA for vectorizable loops?
□ OpenMP for CPU-parallel sections? (with correct schedule type)
□ No false sharing? (alignas(64) on per-thread data)
□ Cache blocking for nested loops over large arrays?
□ Using Boost lockfree / pool where applicable?
□ Checked -fopt-info-vec-missed for vectorization failures?
```
