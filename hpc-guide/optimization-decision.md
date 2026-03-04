# Optimization Decision Framework

For any function that needs optimization: **where** should it run and **how** should it be optimized?

---

## The Decision Tree

```
Function to optimize
│
├─ 1. Does a well-tested library already do this?
│   ├─ YES → USE THE LIBRARY (always first choice)
│   │   ├─ Linear algebra       → cuBLAS / LAPACK / numpy
│   │   ├─ Convolution/Pooling  → cuDNN (PyTorch auto-uses)
│   │   ├─ Sort/Scan/Reduce     → Thrust / CUB / torch ops
│   │   ├─ FFT                  → cuFFT / numpy.fft
│   │   ├─ Sparse ops           → cuSPARSE / scipy.sparse
│   │   ├─ Graph algorithms     → cuGraph / networkx / torch_geometric
│   │   ├─ Molecular fingerprint → RDKit / nvmolkit (GPU)
│   │   ├─ Similarity search    → FAISS (GPU) / nvmolkit
│   │   ├─ String/regex         → re2 / hyperscan
│   │   └─ Compression          → lz4 / zstd (C libraries)
│   └─ NO → continue to step 2
│
├─ 2. Can it be expressed as tensor operations?
│   ├─ YES → GPU TENSOR OPS (PyTorch / NumPy)
│   │   ├─ Elementwise          → torch arithmetic (auto-vectorized)
│   │   ├─ Matrix multiply      → torch.mm / torch.bmm
│   │   ├─ Reduction            → torch.sum / mean / max
│   │   ├─ Broadcasting         → torch shape rules
│   │   ├─ Masking/Selection    → torch.where / boolean indexing
│   │   ├─ Scatter/Gather       → torch.scatter / torch.gather
│   │   └─ Top-k / Sort         → torch.topk / torch.sort
│   │
│   │   If multiple ops → torch.compile() to fuse kernels
│   │   If still too slow → Triton custom kernel
│   │   If Triton insufficient → CUDA C++ kernel
│   │
│   └─ NO → continue to step 3
│
├─ 3. What is the bottleneck?
│   ├─ I/O bound (file, network, database)
│   │   └─ CPU THREADING (ThreadPoolExecutor / asyncio)
│   │
│   ├─ CPU compute bound (Python logic, parsing, string ops)
│   │   ├─ Vectorizable loop → NumPy vectorization first
│   │   ├─ Complex logic      → C++ extension with OpenMP
│   │   └─ Simple parallel    → multiprocessing.Pool
│   │
│   ├─ GPU compute bound (already on GPU, need more FLOPS)
│   │   ├─ Enable tensor cores (AMP / bf16)
│   │   ├─ Fuse kernels (torch.compile / Triton)
│   │   └─ Custom CUDA kernel (last resort)
│   │
│   └─ Memory bound (bandwidth limited)
│       ├─ GPU memory → optimize access patterns (coalescing, SoA)
│       ├─ CPU↔GPU transfer → minimize transfers, async pipeline
│       └─ CPU memory → cache blocking, SoA layout
│
└─ 4. Hybrid: pipeline stages on different devices
    └─ CPU preprocessing → GPU inference → CPU postprocessing
        Use CUDA streams + threading to overlap stages
```

---

## Quick Reference Table

| Operation Type | Best Approach | Example |
|---|---|---|
| Matrix math (GEMM, dot, norm) | **torch/cuBLAS** | `torch.mm(A, B)` |
| Elementwise (add, mul, exp) | **torch tensor ops** | `torch.exp(x)` |
| Reduction (sum, mean, max) | **torch tensor ops** | `x.sum(dim=0)` |
| Sort / top-k | **torch.sort / topk** | `torch.topk(x, k)` |
| Tanimoto similarity (bulk) | **nvmolkit GPU** | GPU fingerprint + cross-similarity |
| kNN search | **FAISS GPU / nvmolkit** | `faiss.IndexFlatIP` |
| Convolution / RNN | **cuDNN (via PyTorch)** | `nn.Conv2d` |
| Sparse matrix ops | **cuSPARSE / torch.sparse** | `torch.sparse.mm` |
| SMILES parsing / Mol construction | **RDKit (C++)** | `Chem.MolFromSmiles()` |
| SMILES→Mol batch (large scale) | **mp.Pool + RDKit** | Parallelize across cores |
| File I/O (read many files) | **ThreadPoolExecutor** | `concurrent.futures` |
| HTTP requests (many URLs) | **asyncio + aiohttp** | `async with session.get()` |
| Custom fused GPU op | **Triton kernel** | Softmax + mask + scale fused |
| Warp-level intrinsics | **CUDA C++ extension** | `__shfl_down_sync` |
| Complex C++ algorithm | **pybind11 / torch ext** | Tree search, graph traversal |

---

## Detailed Decision: Three Optimization Paths

### Path A: GPU Tensor Operations

**When:** Data is numeric, operations are regular (same op on all elements), batch size is large.

```python
# BAD: Python loop
similarities = []
for i in range(len(fps)):
    sim = tanimoto(query_fp, fps[i])
    similarities.append(sim)

# GOOD: Tensor operation (1000x faster)
# fps_tensor: (N, D) on GPU, query: (1, D) on GPU
intersection = (fps_tensor & query).sum(dim=1).float()
union = (fps_tensor | query).sum(dim=1).float()
similarities = intersection / union
```

**Escalation path:**
```
torch ops → torch.compile → Triton → CUDA C++
  (easy)     (auto-fuse)    (custom)  (maximum control)
```

**Signs you need to escalate:**
- torch.compile output still has many small kernels → write Triton
- Triton kernel can't express your algorithm (warp intrinsics, shared memory tricks) → CUDA
- Need to call cuBLAS/cuDNN inside custom kernel → CUDA

### Path B: CPU Parallelism

**When:** Logic is inherently serial per item but items are independent, or operations involve Python objects (strings, Mol objects, graphs).

```python
# BAD: Sequential
results = [process_smiles(smi) for smi in smiles_list]  # 100s

# GOOD: Multiprocessing (for CPU-bound)
from multiprocessing import Pool
with Pool(32) as p:
    results = p.map(process_smiles, smiles_list)  # 5s

# GOOD: Threading (for I/O-bound)
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(16) as ex:
    results = list(ex.map(fetch_url, urls))
```

**CPU parallel decision:**
```
CPU-bound task
├─ Pure numeric, array-like    → NumPy vectorization (no parallelism needed)
├─ Complex logic, per-item     → multiprocessing.Pool
├─ Needs shared state          → Threading (with GIL-releasing C code)
└─ Mixed: some serial, some parallel → Pipeline with queues
```

**CRITICAL:** `mp.Pool` must be created BEFORE any `torch.cuda` call. Fork + CUDA = deadlock.

### Path C: C++ Extension

**When:** Python overhead is the bottleneck (not I/O, not GPU), AND the logic cannot be expressed as tensor ops.

```
Indicators you need C++:
├─ Python for-loop is the bottleneck (not the ops inside it)
├─ Complex branching/control flow per element
├─ Tree/graph traversal with irregular access patterns
├─ String processing at scale (SMILES manipulation)
├─ Need to call C libraries (RDKit, OpenBabel) efficiently
└─ Custom data structures (priority queues, hash maps, tries)
```

**Build options** (see `hpc-cpp/python-extensions.md`):
```
Simplest:   torch.utils.cpp_extension.load()  (JIT, for prototyping)
Production: setup.py + CUDAExtension          (ahead-of-time)
No PyTorch: pybind11                           (pure C++)
Minimal:    ctypes                             (call any .so)
```

---

## Real-World Examples from This Project

### Example 1: Tanimoto Similarity (655K molecules)

```
Attempt 1: Python loop + RDKit         → 2.5 hours ✗
Attempt 2: NumPy vectorized            → 15 minutes (better but still slow)
Attempt 3: nvmolkit GPU kernels        → 3.2 minutes ✓ (47x speedup)
Decision: Library (nvmolkit) > GPU tensor > CPU parallel
```

### Example 2: Batch Preparation (index lookup)

```
Attempt 1: list.index() in Python loop → 26x slower
Fix: Precomputed dict lookup           → 26x faster ✓
Decision: Algorithmic fix (L2 cache) > any hardware parallelism
```

### Example 3: Negative Sampling (2.6M edges)

```
Attempt 1: Single-core Python          → 90 minutes
Fix: multiprocessing.Pool(32)          → ~3 minutes ✓
Decision: CPU parallel (mp.Pool) — items are independent, involve Python set operations
```

### Example 4: Graph Data Loading

```
Attempt 1: Build graph per sample      → 27ms/batch
Fix: Pre-tensorized cache + torch.cat  → 2.3ms/batch ✓ (12x)
Decision: Precompute + cache (L1 file cache) > runtime optimization
```

### Example 5: Product Prediction (ReactionT5v2)

```
Attempt 1: Single-sample inference     → 0.4s/call
Fix: LRU cache + batch inference       → 0.02s/call (cache hit), 0.1s/call (batch)
Decision: L2 function cache + GPU batch inference
```

---

## Anti-Patterns

| Anti-Pattern | Why It's Wrong | Correct Approach |
|---|---|---|
| Writing CUDA kernel for matrix multiply | cuBLAS is faster, tested, maintained | Use `torch.mm` or `cublasSgemm` |
| Using mp.Pool for GPU inference | GPU is already parallel; mp adds overhead | Batch inputs, single GPU call |
| C++ extension for simple array math | PyTorch tensor ops are already C++/CUDA | Use `torch` operations |
| Threading for CPU-bound Python | GIL prevents true parallelism | Use multiprocessing |
| GPU for 100-element array | Kernel launch overhead > computation | Keep on CPU with NumPy |
| Custom sort in CUDA | Thrust/CUB radix sort is near-optimal | `torch.sort` or `thrust::sort` |
| Optimizing code before profiling | May optimize the wrong thing | Profile first (nsys/ncu) |
| Parallelizing already-fast code | Overhead may exceed savings | Only parallelize bottlenecks |

---

## Checklist: Before Optimizing a Function

```
□ Profiled to confirm this function IS the bottleneck?
□ Checked if a library already does this?
□ Determined: I/O bound vs CPU bound vs GPU bound?
□ If numeric + regular → tried tensor ops first?
□ If CPU bound → tried mp.Pool or NumPy vectorization?
□ If nothing works → considered C++ extension?
□ Measured the speedup to confirm it was worth it?
```
