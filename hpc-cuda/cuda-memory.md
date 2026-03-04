# CUDA Memory Hierarchy & Access Patterns

## Coalesced Memory Access

The single most important GPU optimization. A warp (32 threads) accesses memory in 128-byte transactions.

```
GOOD (coalesced): threads access consecutive addresses
  Thread 0 → addr[0], Thread 1 → addr[1], ... Thread 31 → addr[31]
  → 1 memory transaction (128 bytes)

BAD (strided): threads access every Nth element
  Thread 0 → addr[0], Thread 1 → addr[N], ... Thread 31 → addr[31*N]
  → Up to 32 memory transactions (32× slower)

BAD (random): threads access scattered addresses
  → Up to 32 transactions, no spatial locality
```

**Common anti-patterns in Python/PyTorch:**
- Transposing tensors then accessing columns → strided access
- AoS (Array of Structures) layout → use SoA (Structure of Arrays)
- `tensor[indices]` with random indices → scatter/gather

**Fix:** Restructure data layout so consecutive threads access consecutive memory.

## Shared Memory & Bank Conflicts

Shared memory has 32 banks. If multiple threads in a warp access the same bank (different addresses), accesses serialize.

```
32 banks, each 4 bytes wide (32-bit mode)
Bank assignment: addr % 32

NO conflict:  each thread hits different bank
CONFLICT:     threads 0 and 16 both hit bank 0 (stride-16 access)
BROADCAST:    all threads read SAME address → no conflict (broadcast)

Common conflict pattern: stride = 2^N where N ≥ 5
  stride-32 → every thread hits same bank → 32-way conflict!
Fix: pad shared memory arrays → smem[N][33] instead of smem[N][32]
```

## Memory Alignment

- Allocations should be 256-byte aligned for optimal coalescing
- `torch.empty()` handles this automatically
- Custom CUDA kernels: use `__align__(16)` for shared memory

## L2 Cache & Persistence

- A100/H100: 40-50 MB L2
- Set L2 persistence for frequently reused data:
  ```c
  cudaStreamAttrValue attr;
  attr.accessPolicyWindow.base_ptr = ptr;
  attr.accessPolicyWindow.num_bytes = size;
  attr.accessPolicyWindow.hitRatio = 1.0;
  attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
  ```
- PyTorch: not directly exposed, but affects kernel fusion decisions

## Pinned (Page-Locked) Memory

```python
# BAD: pageable memory → CPU↔GPU copy goes through staging buffer
tensor = torch.randn(N, device='cpu')

# GOOD: pinned memory → direct DMA transfer
tensor = torch.randn(N, device='cpu', pin_memory=True)

# DataLoader with pinned memory
DataLoader(dataset, pin_memory=True, num_workers=4)
```

Pinned memory is 2-3× faster for CPU→GPU transfers but reduces available system RAM.

## Unified Memory (GH200 specific)

GH200 has NVLink-C2C between Grace CPU and Hopper GPU:
- 900 GB/s bidirectional bandwidth (vs 64 GB/s PCIe)
- Unified memory can leverage this for oversubscription
- `cudaMallocManaged()` with hints: `cudaMemAdviseSetPreferredLocation`

## Query Commands

```bash
# GPU memory info
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv

# Detailed memory breakdown
nvidia-smi -q -d MEMORY

# L2 cache size
nvidia-smi -q | grep "L2 Cache"

# Check if pinned memory is being used (in nsys profile)
nsys profile --trace=cuda python script.py
```

## Anti-Patterns LLMs Commonly Suggest

1. **"Use `.contiguous()` everywhere"** → Only needed before ops that require contiguous memory. Unnecessary `.contiguous()` copies waste bandwidth.
2. **"Move to GPU one tensor at a time"** → Batch transfers, or restructure to minimize transfers.
3. **"Use shared memory for everything"** → Only helps if data is reused. Single-use data should stay in registers or use L2.
4. **"Larger batch = always faster"** → Beyond a point, diminishing returns due to memory pressure on L2/HBM.
