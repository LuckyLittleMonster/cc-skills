---
name: hpc-guide
description: HPC core mental models and hardware-first reasoning. The root skill that corrects LLM biases in HPC contexts. Slim framework — language-specific details are in hpc-python, hpc-cuda, hpc-cpp, hpc-triton, hpc-infra skills.
user-invocable: true
---

# HPC Guide — Core Mental Models

Corrects systematic LLM biases when working on HPC code. Load this BEFORE any performance-sensitive work.

## Extension Files

| File | Content |
|------|---------|
| `optimization-decision.md` | Decision framework: library vs GPU tensor vs CPU parallel vs C++ extension, with escalation paths and real examples |

## 1. Hardware-First Rule (MANDATORY)

**NEVER debug software or suggest optimizations without first understanding the hardware.**

```
Step 1: READ CACHED HW INFO  → .cache/hw_info/ (see hpc-infra skill)
Step 2: CHECK HW CAPABILITIES → Does hardware ALREADY implement this?
Step 3: CHECK KNOWN LIMITS    → NIC:GPU ratio? PCIe bandwidth? NUMA?
Step 4: ONLY THEN             → Software debug/optimization
```

**Anti-patterns (MUST AVOID):**
- Debugging NCCL without knowing GPU:NIC ratio
- Porting IB optimizations to Slingshot without checking Cassini HW capabilities
- Assuming login node hardware == compute node hardware

## 2. Roofline Thinking

**For ANY code, first ask: compute-bound or memory-bound?**

```
Arithmetic Intensity (AI) = FLOPs / Bytes accessed

GH200:  HBM3 ~4 TB/s, FP32 ~60 TFLOPS → ridge ~15 FLOPs/byte
A100:   HBM2e ~2 TB/s, FP32 ~20 TFLOPS → ridge ~10 FLOPs/byte
H100:   HBM3 ~3.35 TB/s, FP32 ~67 TFLOPS → ridge ~20 FLOPs/byte

Below ridge → memory-bound → optimize data access, reduce transfers
Above ridge → compute-bound → optimize arithmetic, use tensor cores
Optimizing the wrong dimension = wasted effort
```

## 3. Memory Hierarchy

```
Level          Bandwidth        Latency      Size
─────────────────────────────────────────────────
Registers      ~20 TB/s         1 cycle      ~256 KB/SM
Shared Mem     ~19 TB/s         ~20 cycles   64-228 KB/SM
L2 Cache       ~6 TB/s          ~200 cycles  40-50 MB
HBM (DRAM)     2-4 TB/s         ~400 cycles  40-96 GB
CPU DRAM       ~0.5 TB/s        ~μs          128-512 GB
NVLink         600-900 GB/s     ~μs          peer GPU
PCIe 5.0       ~64 GB/s         ~μs          CPU↔GPU
Network (IB)   25-50 GB/s       ~1-10 μs     cross-node
```

**Golden Rule:** Keep data at the highest level. Every level crossing costs 2-10x.

## 4. Parallelism ≠ Concurrency

| General Concurrency | HPC Parallelism |
|---------------------|-----------------|
| Time-sharing, async I/O | Spatial data decomposition |
| Responsiveness, throughput | Raw FLOPS utilization |
| GIL, I/O wait | Communication, sync, load imbalance |
| — | Redundant compute can beat communication |

## 5. Inverted Defaults for HPC Code

| General Programming | HPC Programming |
|--------------------|-----------------|
| Encapsulation > Performance | **Performance > Encapsulation** |
| DRY > Repetition | **Loop unrolling/inlining may be faster** |
| Abstraction layers good | **Reduce indirection, vtable calls** |
| Bounds checking everywhere | **Remove checks from hot paths** |
| `.item()` for debug | **NEVER .item()/.cpu() in loops** |
| Clean function boundaries | **Kernel fusion across boundaries** |
| Dynamic allocation | **Pre-allocate, reuse buffers** |

## 6. Performance Killers — Quick Reference

```
CRITICAL (order-of-magnitude):
├── Uncoalesced GPU memory access       → hpc-cuda
├── CPU↔GPU sync in loops (.item())     → hpc-python
├── Small kernel launch overhead        → hpc-cuda, hpc-triton
├── Python GIL in compute path          → hpc-python
└── Wrong NIC:GPU ratio for RDMA        → hpc-infra

SIGNIFICANT (2-10x):
├── Warp divergence / bank conflicts    → hpc-cuda
├── Non-overlapped communication        → hpc-python (DDP), hpc-infra
├── Unnecessary gradient sync           → hpc-python (DDP)
└── Suboptimal AllReduce algorithm      → hpc-infra

MODERATE (10-50%):
├── Suboptimal batch size               → profile to find sweet spot
├── Unnecessary CPU↔GPU copies          → hpc-python
├── Untuned NCCL env vars              → hpc-infra
└── Missing mixed precision             → ⚠️ SEE WARNING BELOW
```

## ⚠️ Mixed Precision Policy

**Do NOT enable mixed precision (AMP/bf16/fp16) without explicit user approval.**

Mixed precision changes numerical results. While it can provide 2-4× speedup, it alters model behavior:
- Gradient accumulation errors differ
- Loss landscape changes
- Convergence behavior may shift
- Reproducibility is affected

**When you identify mixed precision as a potential optimization:**
1. Inform the user that mixed precision would help and by how much
2. Explain the trade-off (speed vs numerical fidelity)
3. Wait for explicit approval before implementing
4. Never silently add `torch.amp.autocast` or change dtype to fp16/bf16

## 7. Language-Specific Skills

| Skill | Use When |
|-------|----------|
| **hpc-python** | Python parallelism, PyTorch DDP, CUDA streams, latency hiding |
| **hpc-cuda** | CUDA kernels, memory patterns, cuBLAS/cuDNN, nsys/ncu profiling |
| **hpc-cpp** | C++ performance, OpenMP, Boost, compiler flags, cache blocking |
| **hpc-triton** | Triton kernels, autotuning, torch.compile reference |
| **hpc-infra** | NCCL, network fabrics, GPU topology, hardware capabilities, hw cache |

## 8. Diagnosis Flowchart

```
Problem reported
    │
    ├─ Performance issue?
    │   ├─ Query hardware (hpc-infra)
    │   ├─ Profile first (hpc-cuda: nsys/ncu)
    │   ├─ Identify bound (roofline, section 2)
    │   └─ Optimize in correct language skill
    │
    ├─ Communication failure?
    │   ├─ Query network fabric (hpc-infra)
    │   ├─ Check NIC:GPU ratio (hpc-infra)
    │   ├─ Check NCCL config (hpc-infra)
    │   └─ LAST: look at application code
    │
    └─ Porting optimization?
        ├─ Query TARGET hardware capabilities (hpc-infra)
        ├─ Does target HW already implement this?
        │   ├─ YES → do not port
        │   └─ NO → proceed with implementation
        └─ Check known limitations of target platform
```
