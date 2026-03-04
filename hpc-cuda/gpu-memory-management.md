# GPU Memory Management

Managing GPU memory: avoid OOM, diagnose leaks, and **use spare memory to increase throughput**.

---

## Core Principle

GPU memory is a finite resource. The goal is NOT to minimize usage — it's to **maximize utilization**.

```
Memory underutilized → increase parallelism (bigger batch, more workers, larger buffers)
Memory nearly full   → optimize allocation, reduce fragmentation
Memory OOM           → diagnose and fix root cause, then right-size
```

---

## 1. Understanding PyTorch's Memory Allocator

PyTorch uses a **caching allocator** — it does NOT return memory to CUDA after `del tensor`.

```python
import torch

x = torch.randn(1000, 1000, device="cuda")  # allocates from CUDA
del x                                         # returns to CACHE, not to CUDA
# Memory still "allocated" in nvidia-smi, but available for PyTorch reuse

torch.cuda.empty_cache()  # returns cached memory to CUDA
# Only needed when sharing GPU with other processes
```

### Memory Categories

```
nvidia-smi shows:
├─ "Used"  = everything PyTorch + CUDA runtime + other processes
│
torch.cuda.memory_stats() shows:
├─ allocated   = actively used by tensors
├─ reserved    = held by caching allocator (allocated + cached free blocks)
├─ inactive    = reserved but not currently allocated (available for reuse)
└─ fragmented  = reserved blocks too small for requested allocation
```

```python
# Quick memory check
def print_gpu_memory():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_mem / 1024**3
    free = total - reserved
    print(f"Allocated: {allocated:.1f} GB, Reserved: {reserved:.1f} GB, "
          f"Free: {free:.1f} GB, Total: {total:.1f} GB")
```

## 2. OOM Diagnosis Workflow

```
torch.cuda.OutOfMemoryError
│
├─ Step 1: Where did it happen?
│   └─ Read the traceback — which line / which tensor
│
├─ Step 2: What's the memory state?
│   ├─ torch.cuda.memory_summary() — full breakdown
│   ├─ Check "allocated" vs "reserved" gap (fragmentation?)
│   └─ Check peak memory: torch.cuda.max_memory_allocated()
│
├─ Step 3: Is it a leak or a spike?
│   ├─ Leak: memory grows over iterations → find retained references
│   └─ Spike: single allocation too large → reduce size or offload
│
├─ Step 4: Common fixes
│   ├─ Reduce batch size
│   ├─ Gradient checkpointing (trade compute for memory)
│   ├─ Delete intermediate tensors + empty_cache()
│   ├─ Move non-critical data to CPU
│   └─ Fix fragmentation (see Section 4)
│
└─ Step 5: Still OOM?
    ├─ Memory snapshot (see Section 3)
    └─ Consider FSDP / activation offloading for large models
```

## 3. Memory Profiling

### Quick: memory_summary()

```python
# After OOM or at peak usage
print(torch.cuda.memory_summary(abbreviated=True))
# Shows: allocated, reserved, num_alloc_retries, num_ooms
```

### Detailed: Memory Snapshot

```python
# Record memory history
torch.cuda.memory._record_memory_history(max_entries=100000)

# ... run your code ...

# Save snapshot
torch.cuda.memory._dump_snapshot("memory_snapshot.pkl")
torch.cuda.memory._record_memory_history(enabled=None)  # stop recording

# Visualize: open in browser
# python -m torch.utils.viz._memory_viz trace_plot memory_snapshot.pkl -o memory.html
```

### Per-Operation Tracking

```python
# Track which operations allocate most memory
with torch.profiler.profile(
    profile_memory=True,
    record_shapes=True,
    with_stack=True,
) as prof:
    output = model(input_batch)
    loss = criterion(output, target)
    loss.backward()

# Sort by GPU memory allocated
print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=20))
```

## 4. Memory Fragmentation

The caching allocator can have enough total free memory but fail because free blocks are too small.

**Symptoms:**
- OOM despite `reserved - allocated` showing free space
- `num_alloc_retries` > 0 in memory_summary()
- Memory keeps growing across epochs even with constant batch size

**Diagnosis:**
```python
stats = torch.cuda.memory_stats()
print(f"Alloc retries: {stats['num_alloc_retries']}")      # > 0 = fragmentation
print(f"OOM count: {stats['num_ooms']}")
print(f"Active blocks: {stats['active_bytes.all.current'] / 1024**3:.2f} GB")
print(f"Reserved: {stats['reserved_bytes.all.current'] / 1024**3:.2f} GB")
# Large gap between reserved and active = fragmentation
```

**Fixes:**
```python
# 1. Limit max split size — forces larger contiguous blocks
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# 2. Expandable segments (PyTorch 2.1+) — reduces fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 3. Periodic cache clear between phases
torch.cuda.empty_cache()  # between train and eval, or between epochs
```

## 5. Using Spare Memory (Maximize Throughput)

**If GPU memory is underutilized, you're leaving performance on the table.**

### 5.1 Increase Batch Size

The most direct way to use spare memory. Larger batches = better GPU utilization.

```python
# Find maximum batch size empirically
def find_max_batch_size(model, input_shape, start=32, max_tries=10):
    batch_size = start
    for _ in range(max_tries):
        try:
            torch.cuda.empty_cache()
            x = torch.randn(batch_size * 2, *input_shape, device="cuda")
            with torch.no_grad():
                _ = model(x)
            del x
            batch_size *= 2
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            break
    return batch_size  # last successful size

# Or use automatic batch size finder (less precise but faster)
# Start at batch_size, increase until 80% memory used
def auto_batch_size(model, input_fn, target_util=0.8):
    total = torch.cuda.get_device_properties(0).total_mem
    batch_size = 32
    while True:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            x = input_fn(batch_size)
            _ = model(x)
            peak = torch.cuda.max_memory_allocated()
            util = peak / total
            if util > target_util:
                return batch_size
            batch_size = int(batch_size * target_util / util)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return batch_size // 2
```

### 5.2 Increase DataLoader Workers

More workers = more CPU preprocessing overlap. Each worker uses some GPU memory for pinned buffers.

```python
# Check memory headroom, then increase workers
free_mem = torch.cuda.get_device_properties(0).total_mem - torch.cuda.memory_reserved()
if free_mem > 4 * 1024**3:  # > 4 GB free
    num_workers = min(16, os.cpu_count())
else:
    num_workers = 4
```

### 5.3 Larger Replay Buffer / Cache (RL)

In RL, spare GPU memory can hold more experience or precomputed data.

```python
# Instead of fixed-size buffer on CPU:
# Put replay buffer (or part of it) on GPU for faster sampling
free_gb = (torch.cuda.get_device_properties(0).total_mem
           - torch.cuda.memory_reserved()) / 1024**3

# Estimate: each transition ~1KB → 1GB holds ~1M transitions
buffer_size = int(free_gb * 0.5 * 1e6)  # use 50% of free memory
gpu_buffer = torch.empty(buffer_size, state_dim, device="cuda")
```

### 5.4 Preload Lookup Tables to GPU

Move frequently-accessed lookup tables from CPU to GPU if memory allows.

```python
# Fingerprint database, embedding tables, compatibility masks
# Check if it fits
table_size = fp_database.nelement() * fp_database.element_size()
free = torch.cuda.get_device_properties(0).total_mem - torch.cuda.memory_reserved()
if table_size < free * 0.3:  # use at most 30% of free memory
    fp_database = fp_database.cuda()  # all lookups are now GPU-side, no transfer
```

## 6. Common Patterns

### Gradient Checkpointing (Trade Compute for Memory)

```python
from torch.utils.checkpoint import checkpoint

class BigModel(nn.Module):
    def forward(self, x):
        # Recomputes block during backward instead of storing activations
        x = checkpoint(self.block1, x, use_reentrant=False)
        x = checkpoint(self.block2, x, use_reentrant=False)
        return self.head(x)
# Saves ~50-70% activation memory, costs ~30% more compute
```

### Inference Memory Optimization

```python
@torch.no_grad()  # disables gradient storage — saves significant memory
def inference(model, data):
    return model(data)

# Or use inference_mode (stricter, slightly faster)
with torch.inference_mode():
    output = model(data)
```

### Avoid Hidden Memory Leaks

```python
# BAD: accumulating tensors with grad history
losses = []
for batch in dataloader:
    loss = model(batch)
    losses.append(loss)      # ← retains entire computation graph!

# GOOD: detach before storing
losses = []
for batch in dataloader:
    loss = model(batch)
    losses.append(loss.detach())  # ← no grad history retained

# Also BAD: default tensor accumulation
total_loss += loss              # ← graph grows every iteration
# GOOD:
total_loss += loss.item()       # ← but only OUTSIDE hot loop (causes sync)
# BETTER:
total_loss += loss.detach()     # ← no sync, no graph accumulation
```

## 7. Memory Budget Planning

Before running, estimate memory requirements:

```
Model parameters:       sum(p.numel() * p.element_size() for p in model.parameters())
Gradients (training):   same as parameters (1x)
Optimizer state:        Adam = 2x parameters (momentum + variance)
Activations:            depends on model, batch_size, sequence_length
Input data:             batch_size × input_dims × element_size
─────────────────────────────────────────────────
Total training ≈        params × 4 (param + grad + 2× optimizer) + activations

Example (100M param fp32 model):
  Parameters:    400 MB
  Gradients:     400 MB
  Adam state:    800 MB
  Activations:   ~1-4 GB (batch-dependent)
  ──────────────
  Total:         ~2.6-5.6 GB → plenty of room on 96 GB GH200
  → Increase batch size or preload data to GPU
```

## Checklist

```
□ Know current memory usage? (memory_summary or print_gpu_memory)
□ If OOM: identified leak vs spike?
□ If fragmentation: tried expandable_segments or max_split_size_mb?
□ If memory underutilized: increased batch size / workers / buffers?
□ Training: using @torch.no_grad() for validation?
□ Training: .detach() before accumulating losses?
□ Not calling empty_cache() in hot loop? (expensive, only between phases)
□ Estimated memory budget before running?
```
