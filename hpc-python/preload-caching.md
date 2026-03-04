# Preload & Caching Strategies

Three levels of caching, from coarsest to finest granularity. Each level trades storage/memory for compute/IO time.

**Core principle:** Never compute the same thing twice. Never load the same data twice. Never transfer the same bytes twice.

---

## Level 1: File-Level (Persistent / Disk)

Avoid repeated I/O, parsing, and data transformation across runs and sessions.

### 1.1 Serialized Cache (pickle / torch.save)

Cache expensive preprocessing results to disk. Reload on subsequent runs.

```python
import pickle, hashlib, os

CACHE_DIR = ".cache/precomputed"

def cached_preprocess(raw_path, process_fn, version="v1"):
    """Cache processed data to disk. Invalidate on source change or version bump."""
    raw_stat = os.stat(raw_path)
    cache_key = hashlib.md5(
        f"{raw_path}:{raw_stat.st_mtime}:{version}".encode()
    ).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.pkl")

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    result = process_fn(raw_path)
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    return result
```

**When to use:**
- Dataset parsing/cleaning (SMILES→Mol, tokenization, graph construction)
- Negative sampling, compatibility matrix building
- Any preprocessing > 10s that doesn't change between runs

**Cache invalidation:** Hash source file mtime + version string. Bump version when processing logic changes.

**Real example:** Pre-tensorized graph cache (12x speedup), reaction_db_cache.pkl (325s→instant), kNN precomputed cache (1.98GB, 3.2min→instant)

### 1.2 Memory-Mapped Files (np.memmap / torch.from_file)

Random access into large datasets without loading everything into RAM.

```python
import numpy as np

# Write once
data = compute_large_array()  # e.g., 655K × 2048 fingerprints
fp = np.memmap("fingerprints.dat", dtype="float32", mode="w+",
               shape=data.shape)
fp[:] = data[:]
fp.flush()

# Read many times — OS manages page cache, only loads accessed pages
fp = np.memmap("fingerprints.dat", dtype="float32", mode="r",
               shape=(655000, 2048))
batch = fp[indices]  # Only these pages loaded from disk
```

**When to use:**
- Datasets larger than available RAM
- Random access patterns (not sequential scan)
- Multiple processes reading the same data (OS page cache shared)

**Alternatives:**
- **LMDB**: Key-value store, excellent for random read (image datasets, molecular databases)
- **HDF5** (h5py): Hierarchical, supports compression, chunked access
- **SQLite**: When you need queries, not just key-value
- **torch.save / safetensors**: For tensor collections, safetensors supports zero-copy mmap

```python
# LMDB example — fast random read
import lmdb
env = lmdb.open("mol_db", map_size=10 * 1024**3, readonly=True, lock=False)
with env.begin() as txn:
    data = txn.get(smiles.encode())  # O(1) lookup
```

### 1.3 Pre-Tensorized Datasets

Convert raw data to tensors ONCE, save as .pt files, load directly to GPU.

```python
import torch

# Preprocess once
def build_tensor_cache(raw_data, cache_path):
    tensors = {}
    for key, mol_graph in raw_data.items():
        tensors[key] = {
            "x": torch.tensor(mol_graph.node_features, dtype=torch.float32),
            "edge_index": torch.tensor(mol_graph.edges, dtype=torch.long),
        }
    torch.save(tensors, cache_path)

# Load every run — skips all parsing, conversion, graph construction
cache = torch.load(cache_path, weights_only=False)  # instant
batch_x = torch.stack([cache[k]["x"] for k in keys]).cuda()
```

**Key insight:** The bottleneck is often not disk I/O but Python object construction (Mol objects, networkx graphs, tokenization). Pre-tensorizing skips ALL of that.

### 1.4 Shared /dev/shm for Multi-Process

When multiple workers need the same read-only data, put it in shared memory once.

```python
import multiprocessing.shared_memory as shm
import numpy as np

# Parent: create shared block
data = np.array(big_dataset, dtype=np.float32)
sm = shm.SharedMemory(name="dataset", create=True, size=data.nbytes)
shared_arr = np.ndarray(data.shape, dtype=data.dtype, buffer=sm.buf)
shared_arr[:] = data[:]

# Workers: attach without copy
sm = shm.SharedMemory(name="dataset", create=False)
shared_arr = np.ndarray(shape, dtype=np.float32, buffer=sm.buf)
# Each worker reads shared_arr — zero copy, zero pickle
```

**When to use:** mp.Pool workers all need the same large lookup table (building blocks, fingerprint DB, compatibility matrix).

**Cleanup:** Always `sm.close()` in workers, `sm.unlink()` in parent.

---

## Level 2: Function-Level (Runtime / In-Memory)

Avoid repeated computation within a single run.

### 2.1 functools.lru_cache / cache

Memoize pure function results by arguments.

```python
from functools import lru_cache

@lru_cache(maxsize=65536)
def smiles_to_fingerprint(smiles: str) -> tuple:
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    return tuple(fp)  # must be hashable for cache key

# First call: 0.5ms (compute). Subsequent calls: <1μs (lookup).
```

**Rules:**
- Arguments MUST be hashable (no lists, dicts, tensors — use tuples, frozensets, str)
- `maxsize=None` for unlimited cache (careful with memory)
- `maxsize=2**16` typical for bounded cache
- Use `.cache_info()` to monitor hit rate
- `@cache` (Python 3.9+) = `@lru_cache(maxsize=None)`

**When to use:**
- SMILES→Mol, SMILES→fingerprint conversions
- Expensive property calculations called repeatedly with same inputs
- Template matching, reaction validation

### 2.2 Dictionary Lookup Cache

For non-hashable inputs or custom eviction, use plain dicts.

```python
class ProductPredictor:
    def __init__(self):
        self._cache = {}  # (reactant_smi, coreactant_smi) → product_smi

    def predict(self, reactant, coreactant):
        key = (reactant, coreactant)
        if key in self._cache:
            return self._cache[key]

        product = self._expensive_t5_predict(reactant, coreactant)
        self._cache[key] = product
        return product

    def clear_cache(self):
        self._cache.clear()
```

**Real example:** ReactionT5v2 LRU cache — hit rate ~60-80% in RL episodes (0.4s→0.02s per call).

**Bounded variant with LRU eviction:**
```python
from collections import OrderedDict

class BoundedCache:
    def __init__(self, maxsize=100000):
        self._cache = OrderedDict()
        self._maxsize = maxsize

    def get(self, key):
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key, value):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)
```

### 2.3 Deduplication Cache

Skip redundant computation in batch processing.

```python
def extend_batch_dedup(molecules, predict_fn):
    """Process unique molecules only, map results back."""
    unique = list(set(molecules))
    unique_results = {mol: predict_fn(mol) for mol in unique}
    return [unique_results[mol] for mol in molecules]

# If batch has 64 molecules but only 40 unique → 37.5% compute saved
```

**When to use:**
- RL action generation (many molecules overlap across episodes)
- Batch inference where inputs repeat
- Any map operation over a list with duplicates

### 2.4 Index / Lookup Table Precomputation

Replace repeated O(n) scans with O(1) lookups.

```python
# BAD: O(n) per lookup — called thousands of times
for mol in batch:
    idx = molecule_list.index(mol)  # O(n) linear scan

# GOOD: O(1) after one-time O(n) construction
mol_to_idx = {mol: i for i, mol in enumerate(molecule_list)}
for mol in batch:
    idx = mol_to_idx[mol]  # O(1) hash lookup
```

**Real example:** Precomputed index tables for batch preparation (26x speedup). The bottleneck was `list.index()` called per sample in every batch.

**Extends to:**
- Compatibility matrices: `compat[template_id][bb_id] → bool` (precompute once)
- Neighbor lists: `neighbors[node_id] → List[node_id]` (precompute from edge list)
- Reverse mappings: `product_to_reactions[product_smi] → List[reaction]`

### 2.5 Precomputed Masks and Filters

Avoid recomputing boolean masks that don't change.

```python
# Precompute template compatibility mask ONCE at init
# Shape: (num_templates, num_building_blocks) — sparse bool
compat_mask = build_compatibility_mask(templates, building_blocks)

# At runtime: instant mask lookup instead of re-running RDKit matching
valid_bbs = building_blocks[compat_mask[template_id]]
```

**Real example:** PaRoutes 4.67M compatibility pairs precomputed at init (6.1s) → runtime mask lookup O(1).

---

## Level 3: Variable-Level (GPU / Tensor)

Avoid redundant memory allocation, transfer, and computation at the tensor level.

### 3.1 Pre-Allocated Output Buffers

Reuse tensor storage instead of allocating new tensors per iteration.

```python
# BAD: allocates new tensor every iteration
for i in range(num_steps):
    output = model(input_batch)  # new tensor each time
    results.append(output.cpu())

# GOOD: pre-allocate and fill
output_buffer = torch.empty(num_steps, batch_size, dim, device="cuda")
for i in range(num_steps):
    model(input_batch, out=output_buffer[i])  # write into pre-allocated
results = output_buffer.cpu()  # one transfer at the end
```

**Where this matters:**
- Training loops (loss accumulation, metric collection)
- RL environments (state/action/reward buffers)
- Inference pipelines (output collection)

**PyTorch ops that support `out=` parameter:**
`torch.matmul`, `torch.add`, `torch.bmm`, `torch.mm`, `torch.cat`, `torch.stack`, and most arithmetic ops.

### 3.2 GPU Tensor Caching (Avoid Repeated CPU→GPU Transfer)

Keep frequently-used tensors on GPU. Never re-transfer constants.

```python
class PolicyNetwork(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # Register as buffer — moves with model.to(device), saved in state_dict
        self.register_buffer("action_mask", torch.zeros(num_actions, dtype=torch.bool))
        self.register_buffer("position_encoding", build_pos_encoding(...))

    def forward(self, x):
        # self.action_mask is already on correct device — no transfer
        logits = self.head(x)
        logits.masked_fill_(~self.action_mask, float("-inf"))
        return logits
```

**Common mistakes:**
```python
# BAD: re-creates tensor on GPU every call
mask = torch.tensor([1,0,1,0], device="cuda")  # allocation + transfer each time

# GOOD: create once, reuse
# In __init__:  self.mask = torch.tensor([1,0,1,0], device="cuda")
# In forward:   ... self.mask ...
```

### 3.3 Pinned Memory + Async Transfer Pipeline

Pre-pin CPU tensors for DMA transfer, overlap with compute.

```python
# Pre-allocate pinned buffer ONCE
pin_buffer = torch.empty(batch_size, channels, H, W, pin_memory=True)

# In data loading loop:
pin_buffer.copy_(cpu_data)                          # fast memcpy to pinned
gpu_data = pin_buffer.to("cuda", non_blocking=True) # DMA, returns immediately
# ... compute on previous batch while this transfer happens ...
torch.cuda.current_stream().synchronize()           # ensure transfer done before use
```

**Key rule:** `non_blocking=True` is ONLY async if the source tensor is pinned. Without pinned memory, it silently falls back to synchronous.

### 3.4 Model Warmup

First inference/training step is slow due to lazy initialization. Warm up before timing or production.

```python
def warmup_model(model, example_input, warmup_steps=3):
    """Run dummy forward passes to trigger all lazy inits."""
    model.eval()
    with torch.no_grad():
        for _ in range(warmup_steps):
            _ = model(example_input)
    torch.cuda.synchronize()
    # Now: cuDNN benchmarking done, memory pools allocated,
    # JIT caches warm, torch.compile graphs traced

# MUST warm up before:
# - torch.backends.cudnn.benchmark (selects fastest conv algorithm on first run)
# - torch.compile (traces and compiles on first call)
# - CUDAGraphs capture (requires static shapes, warm state)
# - Any timing measurement
```

### 3.5 Gradient Checkpointing (Trade Compute for Memory)

Not caching in the traditional sense — intentionally DISCARDS intermediate activations to save memory, recomputes during backward.

```python
from torch.utils.checkpoint import checkpoint

class LargeModel(nn.Module):
    def forward(self, x):
        # Recomputes block1 during backward instead of storing activations
        x = checkpoint(self.block1, x, use_reentrant=False)
        x = checkpoint(self.block2, x, use_reentrant=False)
        return self.head(x)
```

**When to use:** OOM during training with large models. Trades ~30% more compute for ~50-70% less activation memory.

### 3.6 KV Cache (Transformer Inference)

Cache key/value projections from previous tokens to avoid recomputation in autoregressive generation.

```python
# Conceptual — most frameworks handle this internally
class CachedAttention(nn.Module):
    def forward(self, x, past_kv=None):
        k, v = self.k_proj(x), self.v_proj(x)
        if past_kv is not None:
            prev_k, prev_v = past_kv
            k = torch.cat([prev_k, k], dim=-2)
            v = torch.cat([prev_v, v], dim=-2)
        # ... attention computation ...
        return output, (k, v)  # return updated cache

# Generation loop:
past_kv = None
for step in range(max_len):
    logits, past_kv = model(next_token, past_kv=past_kv)
    # Only computes attention for new token, reuses cached K/V
```

**Real relevance:** ReactionT5v2 uses HuggingFace generate() which manages KV cache internally. Ensure `use_cache=True` (default).

---

## Decision Guide

```
Need to cache something?
│
├─ Does it survive across runs/restarts?
│   └─ YES → Level 1 (File)
│       ├─ Structured data, random access → LMDB / memmap
│       ├─ Tensor data → torch.save / safetensors
│       ├─ General Python objects → pickle
│       └─ Multi-process shared read → /dev/shm or memmap
│
├─ Does it live for the duration of one run?
│   └─ YES → Level 2 (Function)
│       ├─ Pure function, hashable args → @lru_cache
│       ├─ Complex key or eviction logic → dict cache
│       ├─ Batch with duplicates → dedup before compute
│       ├─ Repeated O(n) lookup → precompute index dict
│       └─ Static boolean filter → precompute mask
│
└─ Does it live within a loop/kernel?
    └─ YES → Level 3 (Variable)
        ├─ Repeated allocation → pre-allocate buffer
        ├─ Repeated CPU→GPU transfer → keep on GPU / register_buffer
        ├─ Slow first iteration → model warmup
        ├─ OOM on activations → gradient checkpointing
        └─ Autoregressive decoding → KV cache
```

## Anti-Patterns

| Anti-Pattern | Fix | Level |
|---|---|---|
| Re-parsing SMILES every epoch | Serialize parsed Mol objects / fingerprints to disk | L1 |
| torch.load() per batch | Load entire cache at init, index in memory | L1→L2 |
| `list.index()` in hot loop | Precompute `{item: idx}` dict | L2 |
| Same model input computed twice in batch | Dedup inputs, map results back | L2 |
| `torch.tensor(const, device="cuda")` in forward() | `register_buffer()` in `__init__()` | L3 |
| `torch.zeros(...)` per iteration | Pre-allocate once, `.zero_()` to reset | L3 |
| Timing includes first-call JIT/compile warmup | Warmup 3 steps before timing | L3 |
| generate() with `use_cache=False` | Enable KV cache | L3 |
