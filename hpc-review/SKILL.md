---
name: hpc-review
description: HPC-specific code performance review. Applies hardware-first analysis, roofline thinking, and parallelization checks. Use when reviewing GPU code, distributed training, NCCL communication, or any performance-sensitive HPC code.
user-invocable: true
---

# HPC Performance Review

Review code through an HPC lens — hardware-aware, performance-focused, with mandatory hardware-first diagnosis.

**This skill overrides general code review defaults.** In HPC context, performance > encapsulation, and hardware limits > software fixes.

## Pre-Review: Hardware Context (MANDATORY)

Before reviewing ANY code, gather hardware context using **hpc-infra** skill's cached hardware info at `.cache/hw_info/`. Do NOT skip this step. See `hpc-infra/SKILL.md` for query and cache details.

**If cache is empty or stale:** Run the hw query script from hpc-infra. If on login node, query compute node separately.

## Review Checklist

For each file/function under review, evaluate ALL 7 dimensions:

### 1. Memory Access Pattern
```
□ GPU memory accesses coalesced? (consecutive threads → consecutive addresses)
□ No bank conflicts in shared memory?
□ Data layout: SoA preferred over AoS for GPU?
□ Unnecessary .contiguous() calls?
□ Tensor strides correct after reshape/permute/transpose?
```
**Reference:** `hpc-cuda/cuda-memory.md`

### 2. Compute/Memory Bound Analysis
```
□ Is this code compute-bound or memory-bound?
□ Arithmetic intensity calculated/estimated?
□ Optimization targeting the correct bottleneck?
   - Memory-bound → optimize access patterns, reduce transfers
   - Compute-bound → use tensor cores, fuse operations
□ NOT optimizing the wrong dimension?
```
**Reference:** `hpc-guide/SKILL.md` (Roofline section)

### 3. CPU↔GPU Transfer
```
□ No .cpu() / .item() / .numpy() in hot loops?
□ Accumulate results on GPU, transfer final result only?
□ Pin memory for DataLoader? (pin_memory=True)
□ Async transfers where possible? (non_blocking=True)
□ Minimize synchronization points?
```
**Common anti-pattern:**
```python
# BAD: transfers every iteration
for batch in dataloader:
    loss = model(batch)
    total_loss += loss.item()  # ← CPU sync every step!

# GOOD: accumulate on GPU
losses = []
for batch in dataloader:
    losses.append(model(batch))
total_loss = torch.stack(losses).sum().item()  # ← one transfer
```

### 4. Parallelization Opportunity
```
□ Python for-loops over large datasets → vectorize/batch?
□ Sequential model() calls → batch inference?
□ CPU-bound preprocessing → mp.Pool or ThreadPoolExecutor?
□ list.index() for repeated lookups → dict?
□ Independent operations → parallel execution?
```
**Reference:** CLAUDE.md parallelization guidelines, `hpc-python/parallel-python.md`

### 5. Communication Overhead (Multi-GPU/Node)
```
□ DDP gradient sync overlapping with backward pass?
□ Using model.no_sync() for gradient accumulation?
□ AllReduce algorithm appropriate for message size?
□ NCCL env vars tuned for this fabric?
□ NIC:GPU ratio adequate for workload?
```
**Reference:** `hpc-infra/nccl-comms.md`, `hpc-python/pytorch-ddp.md`

### 6. Kernel Launch Overhead
```
□ No small CUDA kernels in Python loops?
□ Operations fused where possible? (torch.compile, CUDAGraphs)
□ Avoiding unnecessary tensor creation in hot path?
□ Pre-allocated buffers reused?
```
**Common anti-pattern:**
```python
# BAD: N kernel launches
for i in range(N):
    result[i] = torch.matmul(a[i], b[i])

# GOOD: 1 kernel launch (batched)
result = torch.bmm(a, b)
```

### 7. Hardware Awareness
```
□ Code accounts for actual GPU topology?
□ NIC:GPU ratio considered for communication patterns?
□ NUMA affinity respected for CPU operations?
□ No assumptions about PCIe/NVLink topology?
□ Software optimizations verified not redundant with hardware?
```
**Reference:** `hpc-infra/gpu-topology.md`, `hpc-infra/network-fabrics.md`

## Report Format

For each reviewed file, report findings:

```
## File: <path>

### Hardware Context
- GPU: <model>, <count>
- Topology: <NVLink/PCIe>, NIC:GPU = <ratio>
- Fabric: <IB/Slingshot/RoCE>

### Performance Analysis
- Bound: [COMPUTE / MEMORY / COMMUNICATION / CPU]
- Evidence: <profiling data or estimation>

### Issues Found

#### [CRITICAL] <title>
- Dimension: <which of the 7 dimensions>
- Impact: <estimated perf impact>
- Current: <code snippet>
- Fix: <suggested fix>

#### [WARNING] <title>
- Dimension: ...
- Impact: ...
- Suggestion: ...

### OK Dimensions
- <dimensions with no issues>

### Recommendations
1. <prioritized list of improvements>
2. <with estimated impact>
```

## Anti-Patterns to Actively Flag

These are patterns that general code review would APPROVE but are BAD for HPC:

| Pattern | Why It's Wrong in HPC |
|---------|----------------------|
| Clean abstractions with virtual dispatch | vtable overhead in hot path |
| DRY with helper functions | Function call overhead, prevents inlining |
| Defensive `.clone()` before operations | Unnecessary memory allocation |
| Exception handling in compute loops | Branch overhead, prevents vectorization |
| `logging.debug()` in hot paths | String formatting overhead even when disabled |
| `isinstance()` checks in loops | Dynamic dispatch overhead |
| `dict[key]` in tight GPU loops | Hash overhead, use tensors |
| `if not tensor.is_contiguous(): tensor = tensor.contiguous()` | Check is often unnecessary if you control input |

## Integration with Other Skills

- **hpc-guide**: Load core mental models before review
- **hpc-infra**: Hardware context, NIC/GPU topology, fabric detection
- **hpc-cuda**: CUDA-specific review (memory, kernels, profiling)
- **hpc-python**: Python-specific review (parallelism, DDP, latency hiding)
- **hpc-cpp**: C++ review (compiler flags, OpenMP, memory layout)
- **hpc-triton**: Triton kernel review (block sizes, autotuning)
- **code-review**: Use for non-performance dimensions (usage, comments)
- **maple-guide**: Reference for SLURM-specific optimizations
