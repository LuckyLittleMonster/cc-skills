# CPU/GPU Latency Hiding

## Core Concept

Latency hiding = overlap independent operations so one's execution hides another's wait time.

```
WITHOUT latency hiding (serial):
  CPU prep ████░░░░░░░░░░░░░░░░░░░░░░
  GPU comp ░░░░████░░░░░░░░░░░░░░░░░░
  CPU post ░░░░░░░░████░░░░░░░░░░░░░░
  Total: 3T

WITH latency hiding (pipelined):
  CPU prep ████████████░░░░░░░░░░░░░░  (preparing batch N+1 while GPU runs N)
  GPU comp ░░░░████████████░░░░░░░░░░  (computing batch N while CPU preps N+1)
  CPU post ░░░░░░░░████████████░░░░░░  (postprocessing N while GPU runs N+1)
  Total: T + small overlap overhead
```

## CUDA Streams

CUDA streams enable concurrent GPU operations. Default stream serializes everything.

```python
import torch

# Create streams
compute_stream = torch.cuda.Stream()
transfer_stream = torch.cuda.Stream()

# Operations on different streams can overlap
with torch.cuda.stream(transfer_stream):
    # Async H2D transfer on transfer stream
    data_gpu = data_cpu.to('cuda', non_blocking=True)

with torch.cuda.stream(compute_stream):
    # Compute on compute stream (can run while transfer happens)
    result = model(previous_data_gpu)

# Synchronize when needed
torch.cuda.current_stream().wait_stream(transfer_stream)
torch.cuda.current_stream().wait_stream(compute_stream)
```

### Stream Semantics
```
Key rules:
1. Operations on the SAME stream execute in order (FIFO)
2. Operations on DIFFERENT streams MAY execute concurrently
3. Default stream (stream 0) synchronizes with ALL other streams
   → Avoid default stream in latency-hiding code!
4. wait_stream() creates a dependency: current stream waits for target
5. Events can synchronize specific points between streams
```

### Stream Events (Fine-Grained Sync)

```python
# Events allow precise synchronization between streams
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# Record event on one stream
with torch.cuda.stream(stream_a):
    stream_a_work()
    event = torch.cuda.Event()
    event.record()  # Mark this point in stream_a

# Wait for event on another stream
with torch.cuda.stream(stream_b):
    stream_b.wait_event(event)  # Wait until stream_a reaches the event
    stream_b_work_that_depends_on_a()
```

## Pattern 1: Double Buffering (H2D Transfer + Compute Overlap)

The most common and impactful latency hiding pattern.

```python
import torch
from torch.utils.data import DataLoader

def train_with_prefetch(model, dataloader, optimizer):
    """Overlap data transfer with computation using double buffering."""
    transfer_stream = torch.cuda.Stream()

    # Prefetch first batch
    data_iter = iter(dataloader)
    batch = next(data_iter)
    with torch.cuda.stream(transfer_stream):
        batch = {k: v.to('cuda', non_blocking=True) for k, v in batch.items()}

    for next_batch_cpu in data_iter:
        # Wait for current batch transfer to complete
        torch.cuda.current_stream().wait_stream(transfer_stream)
        current_batch = batch

        # Start transferring next batch (overlaps with compute below)
        with torch.cuda.stream(transfer_stream):
            batch = {k: v.to('cuda', non_blocking=True) for k, v in next_batch_cpu.items()}

        # Compute on current batch (overlaps with next batch transfer)
        loss = model(**current_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

**Why this works:**
- H2D transfer of batch N+1 uses DMA engine (independent of CUDA cores)
- Compute of batch N uses CUDA cores
- Both happen simultaneously → transfer latency hidden

## Pattern 2: Compute + Communication Overlap (DDP)

PyTorch DDP does this automatically via gradient buckets, but you can optimize it.

```python
# DDP automatic overlap:
# Backward pass computes gradients layer by layer (back to front)
# When a gradient bucket fills, AllReduce starts immediately
# → AllReduce of early layers overlaps with backward of later layers

# Tune bucket size for better overlap:
model = DDP(model, bucket_cap_mb=25)  # Default 25MB
# Smaller = more overlap opportunity, more launch overhead
# Larger = less overhead, less overlap
# Rule of thumb: match to your allreduce latency

# Manual gradient accumulation with overlap:
for i, batch in enumerate(dataloader):
    # Disable sync during accumulation steps
    ctx = model.no_sync() if (i + 1) % accum_steps != 0 else nullcontext()
    with ctx:
        loss = model(batch) / accum_steps
        loss.backward()

    if (i + 1) % accum_steps == 0:
        # AllReduce only happens here — overlaps with nothing
        # Consider: start next batch prefetch during AllReduce
        optimizer.step()
        optimizer.zero_grad()
```

## Pattern 3: CPU Preprocessing Pipeline

Overlap CPU data preprocessing with GPU computation.

```python
import threading
import queue

class PrefetchPipeline:
    """CPU preprocessing → GPU computation pipeline with prefetch buffer."""

    def __init__(self, data_source, preprocess_fn, buffer_size=3):
        self.queue = queue.Queue(maxsize=buffer_size)
        self.data_source = data_source
        self.preprocess_fn = preprocess_fn
        self._stop = threading.Event()

        # Start background preprocessing thread
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        """Background thread: preprocess and enqueue."""
        transfer_stream = torch.cuda.Stream()
        for item in self.data_source:
            if self._stop.is_set():
                break
            # CPU preprocessing (released GIL for C extensions)
            processed = self.preprocess_fn(item)
            # Async GPU transfer
            with torch.cuda.stream(transfer_stream):
                gpu_data = processed.to('cuda', non_blocking=True)
            transfer_stream.synchronize()
            self.queue.put(gpu_data)
        self.queue.put(None)  # Sentinel

    def __iter__(self):
        while True:
            item = self.queue.get()
            if item is None:
                break
            yield item

    def stop(self):
        self._stop.set()
```

## Pattern 4: Multi-Stream Kernel Execution

Run independent GPU kernels concurrently on different streams.

```python
# Useful when: multiple independent small operations
# NOT useful when: single large operation (already uses all SMs)

streams = [torch.cuda.Stream() for _ in range(4)]

# Launch independent operations on different streams
results = []
for i, (data_chunk, stream) in enumerate(zip(data_chunks, streams)):
    with torch.cuda.stream(stream):
        results.append(model_part(data_chunk))

# Synchronize all streams
torch.cuda.synchronize()
# Or selectively wait:
# torch.cuda.current_stream().wait_stream(streams[0])
```

**When this helps:**
- Small kernels that don't saturate GPU → multiple can run in parallel
- Independent operations on different data

**When this does NOT help:**
- Large kernels (matmul on large tensors) → already uses all SMs
- Stream overhead > parallelism benefit for tiny operations

## Pattern 5: Async Checkpointing

Save model checkpoints without blocking training.

```python
import threading
import io

def async_save_checkpoint(model, optimizer, path):
    """Save checkpoint in background without blocking GPU."""
    # Snapshot state to CPU (quick, one GPU→CPU transfer)
    state = {
        'model': {k: v.cpu() for k, v in model.state_dict().items()},
        'optimizer': optimizer.state_dict(),
    }

    def _save():
        torch.save(state, path)

    thread = threading.Thread(target=_save)
    thread.start()
    return thread  # Caller can join() before exit

# In training loop:
for step in range(total_steps):
    train_step()
    if step % 1000 == 0:
        save_thread = async_save_checkpoint(model, optimizer, f"ckpt_{step}.pt")
        # Training continues while save happens in background
```

## Pattern 6: RL Environment Parallelism

Overlap CPU environment steps with GPU model inference.

```python
import multiprocessing as mp
import torch

# Architecture:
# Main process: GPU model inference
# Worker processes: CPU environment steps (mp.Pool, created BEFORE CUDA)

class ParallelRLRunner:
    def __init__(self, env_fn, model, num_envs=64):
        # CRITICAL: Pool BEFORE CUDA
        self.pool = mp.Pool(num_envs)
        self.model = model.cuda()
        self.transfer_stream = torch.cuda.Stream()

    def collect_rollout(self, steps):
        obs = self.pool.map(reset_env, range(self.num_envs))

        for step in range(steps):
            # CPU→GPU transfer (async)
            with torch.cuda.stream(self.transfer_stream):
                obs_tensor = torch.stack(obs).to('cuda', non_blocking=True)

            # Wait for transfer, then compute actions
            torch.cuda.current_stream().wait_stream(self.transfer_stream)
            with torch.no_grad():
                actions = self.model(obs_tensor).cpu().numpy()

            # Environment steps on CPU (parallel, overlaps with... nothing yet)
            # But if we have TWO batches, we can overlap:
            # - GPU computes actions for batch B
            # - CPU steps environments for batch A
            results = self.pool.starmap(step_env,
                zip(range(self.num_envs), actions))
            obs = [r[0] for r in results]
```

## Pattern 7: CUDAGraphs (Eliminate Launch Overhead)

For repeated identical sequences of small kernels.

```python
import torch

# Warmup: run the sequence once to capture
model.eval()
static_input = torch.randn(batch_size, input_dim, device='cuda')
static_output = torch.empty(batch_size, output_dim, device='cuda')

# Capture graph
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    static_output = model(static_input)

# Replay: near-zero launch overhead
for batch in dataloader:
    static_input.copy_(batch)  # Copy new data into static buffer
    graph.replay()             # Replay captured sequence
    result = static_output     # Read output from static buffer

# LIMITATIONS:
# - Input/output shapes must be FIXED (no dynamic shapes)
# - No CPU-GPU sync inside the graph
# - No control flow that depends on tensor values
# - Must use static allocated tensors (no allocation inside graph)
```

## Pattern 8: torch.compile Overlap

`torch.compile` can automatically fuse and overlap operations.

```python
# torch.compile with reduce-overhead mode
# Internally uses CUDAGraphs + kernel fusion
model = torch.compile(model, mode='reduce-overhead')

# Modes:
# 'default'          - balance compile time and performance
# 'reduce-overhead'  - aggressive fusion, CUDAGraphs, minimal Python overhead
# 'max-autotune'     - tries all kernel variants, slowest compile, fastest run
```

## Decision Matrix: Which Latency Hiding Technique?

| Situation | Technique | Complexity |
|-----------|-----------|-----------|
| Data loading bottleneck | DataLoader `num_workers` + `pin_memory` | Low |
| H2D transfer visible in profile | Double buffering (Pattern 1) | Medium |
| Gradient AllReduce dominant | DDP bucket tuning, `no_sync()` | Low |
| CPU preprocessing slow | Prefetch pipeline (Pattern 3) | Medium |
| Many small independent kernels | Multi-stream (Pattern 4) | Low |
| Checkpoint saving blocks training | Async save (Pattern 5) | Low |
| RL env steps slow | mp.Pool + Pipeline (Pattern 6) | High |
| Repeated fixed-shape inference | CUDAGraphs (Pattern 7) | Medium |
| General kernel fusion | torch.compile (Pattern 8) | Low |

## Profiling Latency Hiding

To verify your overlap actually works:

```python
# Use nsys to visualize stream concurrency
# Look for: parallel bars on different CUDA streams in the timeline
nsys profile -t cuda,nvtx python train.py

# Use torch.profiler to see stream activity
with torch.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    with_stack=True
) as prof:
    train_step()

# Check: do H2D copy and compute overlap in the timeline?
# If they're sequential → your non_blocking/stream setup is wrong
# Common mistake: forgetting non_blocking=True in .to('cuda')
```

## Common Mistakes

```python
# MISTAKE 1: non_blocking without pinned memory → no actual overlap
data = torch.randn(N)  # ← pageable memory
data_gpu = data.to('cuda', non_blocking=True)  # ← STILL SYNCHRONOUS!
# FIX: data = torch.randn(N).pin_memory()

# MISTAKE 2: Implicit sync via default stream
with torch.cuda.stream(my_stream):
    a = func_a(x)    # On my_stream
b = func_b(y)        # On DEFAULT stream → waits for my_stream!
# FIX: Use explicit streams for all concurrent work

# MISTAKE 3: .item() or print() in pipeline → forces sync
for batch in dataloader:
    loss = model(batch)
    print(f"loss: {loss.item()}")  # ← GPU sync every step!
# FIX: Log every N steps, or accumulate losses

# MISTAKE 4: Overlapping dependent operations
with torch.cuda.stream(stream_a):
    intermediate = layer1(x)
with torch.cuda.stream(stream_b):
    result = layer2(intermediate)  # ← RACE CONDITION! stream_b doesn't wait for stream_a
# FIX: stream_b.wait_stream(stream_a) or use events

# MISTAKE 5: Too many streams → overhead exceeds benefit
streams = [torch.cuda.Stream() for _ in range(1000)]  # ← Overkill!
# FIX: 2-4 streams is usually optimal. More = diminishing returns.
```
