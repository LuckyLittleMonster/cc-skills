# DataLoader Optimization

PyTorch DataLoader is often the hidden bottleneck. A slow data pipeline starves the GPU.

---

## Diagnosis: Is DataLoader the Bottleneck?

```python
# Quick check: profile one epoch
import time

data_time = 0
compute_time = 0

end = time.perf_counter()
for batch in dataloader:
    data_time += time.perf_counter() - end  # time to get batch

    start = time.perf_counter()
    output = model(batch)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    compute_time += time.perf_counter() - start

    end = time.perf_counter()

ratio = data_time / (data_time + compute_time)
print(f"Data: {data_time:.1f}s ({ratio:.0%}), Compute: {compute_time:.1f}s")
# If data > 10%: DataLoader is the bottleneck
```

**In nsys profile:** Look for gaps between CUDA kernels — those gaps are CPU waiting for data.

---

## Key Parameters

```python
DataLoader(
    dataset,
    batch_size=64,             # see gpu-memory-management.md for sizing
    num_workers=8,             # CPU workers for data loading
    pin_memory=True,           # enable DMA transfer to GPU
    prefetch_factor=2,         # batches prefetched per worker
    persistent_workers=True,   # keep workers alive between epochs
    drop_last=True,            # avoid small last batch (DDP requires)
    multiprocessing_context="forkserver",  # safer than fork with CUDA
)
```

### num_workers

```
num_workers=0:  Main process loads data (slowest, debug only)
num_workers=1:  One worker, still serialized
num_workers=N:  N parallel workers prefetching data

Rule of thumb:  start at 4, increase until data_time < 10% of total
Upper bound:    min(cpu_count, 16)
Diminishing:    beyond 8-16 workers, OS scheduling overhead increases
```

```python
# Find optimal num_workers empirically
import time
for nw in [0, 1, 2, 4, 8, 16]:
    loader = DataLoader(dataset, batch_size=64, num_workers=nw, pin_memory=True)
    start = time.perf_counter()
    for i, batch in enumerate(loader):
        if i >= 50:
            break
    elapsed = time.perf_counter() - start
    print(f"num_workers={nw:2d}: {elapsed:.2f}s for 50 batches")
```

### pin_memory

```
pin_memory=True:
  - DataLoader output is in page-locked (pinned) memory
  - Enables async CPU→GPU transfer with non_blocking=True
  - 2-3× faster transfers via DMA engine
  - Costs: slightly more CPU memory, slower CPU access to that memory

ALWAYS use pin_memory=True unless CPU memory is critically low.
```

```python
# Combined with non_blocking transfer:
for batch in dataloader:  # pin_memory=True
    batch = batch.to("cuda", non_blocking=True)  # async DMA
    # ... GPU compute overlaps with next batch prefetch ...
```

### prefetch_factor

```
prefetch_factor=N:
  Each worker prefetches N batches ahead.
  Default: 2 (good for most cases)
  Increase to 3-4 if: per-sample processing is highly variable
  Decrease to 1 if: memory constrained (each prefetched batch uses RAM)

  Total prefetched = num_workers × prefetch_factor
  Example: 8 workers × 2 = 16 batches in memory at once
```

### persistent_workers

```
persistent_workers=True:
  Workers stay alive between epochs (no re-fork overhead)
  Default: False
  ALWAYS set True if num_workers > 0

  Without: ~2-5s worker startup at each epoch
  With:    0s startup (workers already running)

  Caveat: workers keep their memory across epochs
```

---

## Dataset Design Patterns

### Pattern 1: Pre-Tensorized Dataset (Fastest)

Avoid per-sample processing in __getitem__. Do it once, save tensors.

```python
class PreTensorizedDataset(Dataset):
    def __init__(self, cache_path):
        # Load pre-processed tensors (see preload-caching.md L1)
        self.data = torch.load(cache_path)  # dict of tensors
        self.x = self.data["x"]             # (N, D) tensor
        self.y = self.data["y"]             # (N,) tensor

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]     # just indexing, no processing

    def __len__(self):
        return len(self.x)
```

**Rule:** If `__getitem__` does more than tensor indexing, consider pre-processing.

### Pattern 2: Memory-Mapped Dataset (Large Datasets)

```python
class MemmapDataset(Dataset):
    def __init__(self, path, shape, dtype=np.float32):
        self.data = np.memmap(path, dtype=dtype, mode="r", shape=shape)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx].copy())  # copy: memmap is read-only

    def __len__(self):
        return self.data.shape[0]
```

### Pattern 3: LMDB Dataset (Random Access on Large Datasets)

```python
import lmdb, pickle

class LMDBDataset(Dataset):
    def __init__(self, db_path):
        self.env = lmdb.open(db_path, readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin() as txn:
            self.length = txn.stat()["entries"]

    def __getitem__(self, idx):
        with self.env.begin() as txn:
            data = pickle.loads(txn.get(str(idx).encode()))
        return data

    def __len__(self):
        return self.length
```

### Pattern 4: In-Memory GPU Dataset (Small Datasets)

If the entire dataset fits in GPU memory, skip the DataLoader entirely.

```python
# Load everything to GPU once
x_gpu = torch.load("data.pt").cuda()  # (N, D) on GPU
y_gpu = torch.load("labels.pt").cuda()

# Random batch sampling — zero transfer overhead
for epoch in range(num_epochs):
    perm = torch.randperm(len(x_gpu), device="cuda")
    for i in range(0, len(x_gpu), batch_size):
        idx = perm[i:i+batch_size]
        batch_x = x_gpu[idx]  # GPU→GPU indexing, very fast
        batch_y = y_gpu[idx]
        # ... train ...
```

---

## Custom Collation

Default collate stacks tensors. Custom collate for variable-length data.

```python
def collate_graphs(batch):
    """Collate variable-size graphs into batched format."""
    xs = [item["x"] for item in batch]
    edges = [item["edge_index"] for item in batch]

    # Concatenate with batch offset for edges
    x = torch.cat(xs, dim=0)
    batch_vec = torch.cat([torch.full((len(x_i),), i) for i, x_i in enumerate(xs)])

    offset = 0
    shifted_edges = []
    for e in edges:
        shifted_edges.append(e + offset)
        offset += len(xs[shifted_edges[-1:]])  # node count
    edge_index = torch.cat(shifted_edges, dim=1)

    return {"x": x, "edge_index": edge_index, "batch": batch_vec}

dataloader = DataLoader(dataset, collate_fn=collate_graphs, ...)
```

**Performance tip:** If collation is slow, pre-collate batches and save (see preload-caching.md).

---

## Worker Issues

### Fork + CUDA Deadlock

```python
# SAFE: DataLoader workers don't touch CUDA (they prepare CPU tensors)
# pin_memory_thread handles CPU→pinned transfer in main process

# UNSAFE: Dataset.__getitem__ calls CUDA ops
class BadDataset(Dataset):
    def __getitem__(self, idx):
        x = self.data[idx].cuda()  # ← CUDA in forked worker = deadlock
        return x

# FIX: Keep workers CPU-only. Transfer to GPU in training loop.
```

### Worker Memory Growth

```python
# Each worker copies full dataset in fork. For large datasets:
# 1. Use memory-mapped files (np.memmap) — shared across workers
# 2. Use LMDB — each worker opens its own read-only handle
# 3. Use shared_memory — explicitly share data across workers
```

### Slow First Batch

```python
# First batch includes worker startup. Use persistent_workers=True.
# Also: iterate one batch as warmup before timing.
```

---

## Data Format Selection

| Format | Random Access | Compression | Multi-Process | Best For |
|--------|:---:|:---:|:---:|---|
| torch .pt | Full | No | ✓ (load once) | Small-medium tensors |
| np.memmap | Full | No | ✓ (shared pages) | Large numeric arrays |
| LMDB | Full | No | ✓ (per-worker txn) | Large item count, variable size |
| HDF5 | Full | Yes | ✗ (GIL issues) | Compressed, hierarchical |
| safetensors | Full | No | ✓ (mmap) | Model weights, zero-copy |
| SQLite | Query | No | ✓ (WAL mode) | When you need SQL queries |
| Parquet | Column | Yes | ✓ | Tabular data, analytics |

## Checklist

```
□ Measured data loading time vs compute time?
□ num_workers > 0? (start at 4, tune up)
□ pin_memory=True?
□ persistent_workers=True?
□ __getitem__ does minimal work? (no heavy processing, no CUDA)
□ If data fits in memory: considered pre-tensorized or in-memory GPU?
□ If data doesn't fit: using memmap or LMDB?
□ Custom collate_fn efficient? (no Python loops over large lists)
□ drop_last=True for DDP?
```
