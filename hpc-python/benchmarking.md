# Benchmarking Methodology

How to correctly measure performance of Python/PyTorch code. Wrong benchmarks lead to wrong optimization decisions.

---

## The Fundamental Rule

**GPU operations are asynchronous.** Python returns immediately after launching a kernel. You MUST synchronize before measuring.

```python
# WRONG: measures kernel launch time, not execution time
start = time.time()
output = model(input)
elapsed = time.time() - start  # ← meaningless, GPU still computing

# CORRECT: synchronize before measuring
torch.cuda.synchronize()
start = time.time()
output = model(input)
torch.cuda.synchronize()
elapsed = time.time() - start  # ← actual GPU execution time
```

---

## 1. CUDA Events (Most Accurate for GPU)

```python
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# Warmup (MANDATORY)
for _ in range(3):
    _ = fn(x)

torch.cuda.synchronize()
start_event.record()
output = fn(x)
end_event.record()
torch.cuda.synchronize()

elapsed_ms = start_event.elapsed_time(end_event)
```

**Why CUDA events > time.time():**
- Events measure GPU time directly (not CPU wall clock)
- No CPU→GPU sync overhead in the measurement
- Sub-millisecond precision

## 2. torch.utils.benchmark.Timer (Recommended)

Handles warmup, statistics, and GPU sync automatically.

```python
import torch.utils.benchmark as benchmark

t = benchmark.Timer(
    stmt="model(x)",
    globals={"model": model, "x": input_tensor},
    num_threads=1,
)

# Automatic warmup + multiple runs + statistics
result = t.blocked_autorange(min_run_time=2.0)
print(result)
# <torch.utils.benchmark.utils.common.Measurement object at 0x...>
#   model(x)
#   2.35 ms
#   1 measurement, 100 runs, 1 thread
```

### Compare Multiple Implementations

```python
results = []
for name, fn in [("baseline", baseline_fn), ("optimized", optimized_fn)]:
    t = benchmark.Timer(
        stmt="fn(x)",
        globals={"fn": fn, "x": input_tensor},
        label="forward",
        sub_label=name,
        description=f"{name} implementation",
    )
    results.append(t.blocked_autorange(min_run_time=2.0))

compare = benchmark.Compare(results)
compare.print()
```

## 3. Warmup Requirements

**Why:** First execution includes one-time costs that don't reflect steady-state performance.

| What | First-Call Cost | Steady-State |
|------|----------------|-------------|
| CUDA context creation | ~200ms | 0 |
| cuDNN benchmark | ~1-10s (tests algorithms) | 0 |
| torch.compile tracing | seconds-minutes | 0 |
| JIT cache miss | varies | 0 |
| Memory pool allocation | varies | 0 (reuse) |
| CUDAGraphs capture | ~100ms | 0 |

```python
# Minimum warmup
WARMUP_STEPS = 5  # 3-10 depending on what you're measuring

for _ in range(WARMUP_STEPS):
    _ = fn(x)
torch.cuda.synchronize()

# Now measure
```

**If using `torch.backends.cudnn.benchmark = True`:** First forward pass per input shape triggers algorithm search. Warm up with representative shapes.

## 4. Common Timing Pitfalls

### Pitfall 1: Measuring Python Overhead Instead of GPU

```python
# This measures Python dict lookup + kernel launch, not kernel execution
for op_name, op in ops.items():
    start = time.time()
    op(x)  # async launch
    times[op_name] = time.time() - start  # ~0.1ms regardless of kernel size
```

### Pitfall 2: Including Data Transfer

```python
# BAD: measures H2D transfer + compute
x_cpu = torch.randn(N, D)
start = time.time()
output = model(x_cpu.cuda())  # includes CPU→GPU transfer!
torch.cuda.synchronize()
elapsed = time.time() - start

# GOOD: separate transfer from compute
x_gpu = x_cpu.cuda()
torch.cuda.synchronize()
start = time.time()
output = model(x_gpu)
torch.cuda.synchronize()
elapsed = time.time() - start
```

### Pitfall 3: Measuring Allocation Not Compute

```python
# First call allocates output tensor. Second reuses cache.
# Measure after caching allocator has warmed up.
```

### Pitfall 4: time.time() Resolution

```python
# time.time() resolution is ~1μs on Linux, ~15ms on some systems
# For sub-ms measurements, use CUDA events or perf_counter_ns

import time
start = time.perf_counter_ns()  # nanosecond resolution
# ... CPU code ...
elapsed_ns = time.perf_counter_ns() - start
```

### Pitfall 5: Power State

```python
# GPU may be in low-power state. First operation wakes it up (~100ms).
# Always run warmup iterations to bring GPU to full clock speed.
# Use nvidia-smi -pm 1 to enable persistence mode (avoids context teardown).
```

## 5. What to Measure

### Throughput (samples/second)

```python
batch_size = 64
num_batches = 100

torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(num_batches):
    _ = model(batch)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start

throughput = (batch_size * num_batches) / elapsed
print(f"{throughput:.0f} samples/sec")
```

### Latency (ms/sample)

```python
# For interactive / real-time applications
latency_ms = elapsed * 1000 / (batch_size * num_batches)
```

### Memory

```python
torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()

_ = fn(x)
torch.cuda.synchronize()

peak_mb = torch.cuda.max_memory_allocated() / 1024**2
print(f"Peak memory: {peak_mb:.0f} MB")
```

### Scaling

```python
# How does performance change with batch size?
for bs in [1, 4, 16, 64, 256, 1024]:
    x = torch.randn(bs, D, device="cuda")
    t = benchmark.Timer(stmt="fn(x)", globals={"fn": fn, "x": x})
    result = t.blocked_autorange()
    throughput = bs / result.mean
    print(f"bs={bs:4d}: {result.mean*1000:8.2f} ms, {throughput:8.0f} samples/s")
```

## 6. Reporting Benchmarks

Always include:
```
Hardware:    GPU model, driver, CUDA version
Software:    PyTorch version, Python version
Config:      batch size, input shape, dtype, compile mode
Warmup:      N steps discarded
Measurement: N runs, statistical summary (mean ± std)
Memory:      peak GPU memory allocated
```

## Checklist

```
□ GPU synchronized before AND after timed region?
□ Warmup iterations run (≥ 3)?
□ Data already on GPU before timing? (not measuring transfer)
□ Using CUDA events or torch.utils.benchmark.Timer? (not time.time())
□ Measuring steady-state, not first-call compilation?
□ If comparing: same input data, same device, same conditions?
□ Reported hardware/software versions?
□ Measured memory alongside latency?
```
