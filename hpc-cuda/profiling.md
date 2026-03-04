# HPC Profiling — Tools & Roofline Analysis

## Profiling Priority

**Profile before optimizing. Always.**

```
Common mistake: "This loop looks slow, let me optimize it"
Correct: "Let me profile to find the actual bottleneck"

Profiling order:
1. High-level timing (time.perf_counter, torch.cuda.Event)
2. PyTorch Profiler (kernel-level, memory, NCCL)
3. nsys (system-wide: CPU, GPU, network, memory)
4. ncu (single kernel deep-dive)
```

## Quick Timing

```python
import time
import torch

# CPU timing
start = time.perf_counter()
result = expensive_operation()
torch.cuda.synchronize()  # ← CRITICAL: GPU ops are async!
elapsed = time.perf_counter() - start

# GPU timing (more accurate)
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
result = expensive_operation()
end_event.record()
torch.cuda.synchronize()
elapsed_ms = start_event.elapsed_time(end_event)
```

**Common mistake:** Timing GPU ops without `torch.cuda.synchronize()` → measures only launch time, not execution time.

## PyTorch Profiler

```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    schedule=torch.profiler.schedule(
        wait=1, warmup=1, active=3, repeat=1
    ),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler'),
) as prof:
    for step, batch in enumerate(dataloader):
        with record_function("forward"):
            output = model(batch)
        with record_function("backward"):
            loss.backward()
        with record_function("optimizer"):
            optimizer.step()
        prof.step()

# Print summary
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# Export Chrome trace (open in chrome://tracing)
prof.export_chrome_trace("trace.json")
```

### Key Profiler Output Columns

| Column | Meaning |
|--------|---------|
| `Self CPU` | CPU time excluding child calls |
| `CPU total` | CPU time including children |
| `Self CUDA` | GPU kernel time (actual compute) |
| `CUDA total` | GPU time including children |
| `CPU Mem` | CPU memory allocated |
| `Self CUDA Mem` | GPU memory allocated by this op |

**What to look for:**
- Large gap between CPU total and CUDA total → CPU bottleneck (data loading, preprocessing)
- Many small CUDA kernels → kernel launch overhead → fuse operations
- `aten::copy_` dominating → too many CPU↔GPU transfers
- `nccl:allReduce` taking > 30% → communication bottleneck

## NVIDIA Nsight Systems (nsys) — System-Wide Profiler

**Purpose:** See the BIG PICTURE — CPU/GPU interaction, kernel launch gaps, NCCL overlap, memory copies, data loading stalls. Always start here before ncu.

**Docs:** https://docs.nvidia.com/nsight-systems/

### Basic Usage

```bash
# Minimal (quick overview)
nsys profile -o report python train.py

# Recommended (full trace with NCCL, cuBLAS, cuDNN)
nsys profile \
    -t cuda,nvtx,osrt,cudnn,cublas \
    -o report \
    --force-overwrite true \
    python -u train.py

# With Python backtrace (identify which Python line launched each kernel)
nsys profile \
    -t cuda,nvtx,osrt,cudnn,cublas \
    --python-backtrace=cuda \
    --python-sampling=true \
    -o report python train.py
```

### Profiling a Specific Region (Skip Warmup)

```python
# In your code: only profile the interesting part
import torch.cuda.profiler as profiler
import torch.cuda.nvtx as nvtx

# Warmup (not profiled)
for i in range(5):
    train_step()

# Start profiling
profiler.start()
nvtx.range_push("training_loop")
for i in range(10):
    nvtx.range_push(f"step_{i}")
    train_step()
    nvtx.range_pop()
nvtx.range_pop()
profiler.stop()
```

```bash
# Launch with capture range set to cudaProfilerApi
nsys profile --capture-range=cudaProfilerApi \
    -t cuda,nvtx,osrt,cudnn,cublas \
    -o report python train.py
```

### SLURM / Remote Node Profiling

```bash
# Via srun
srun -n1 --gres=gpu:1 -p maple \
    nsys profile -o /path/to/report python -u train.py

# Via SSH to allocated node
ssh maple-n05 "cd $(pwd) && source ~/.bashrc_maple 2>/dev/null && \
    conda activate rl4 && \
    nsys profile -o report -t cuda,nvtx,osrt python -u train.py"

# DDP profiling: profile each rank separately
srun -n2 --gres=gpu:1 \
    nsys profile -o report_rank%q{SLURM_PROCID} python -u train_ddp.py
```

### CLI Analysis (No GUI Needed)

```bash
# Summary statistics (no GUI required)
nsys stats report.nsys-rep

# CUDA kernel summary (top kernels by time)
nsys stats --report cuda_gpu_kern_sum report.nsys-rep

# CUDA memory operation summary
nsys stats --report cuda_gpu_mem_size_sum report.nsys-rep

# NCCL operation summary
nsys stats --report nvtx_sum report.nsys-rep

# Export to SQLite for custom queries
nsys export --type=sqlite -o report.sqlite report.nsys-rep
# Then: sqlite3 report.sqlite "SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY end - start DESC LIMIT 20"
```

### What to Look For in nsys

```
1. GPU IDLE GAPS (most common issue)
   Timeline shows: ████░░░░████░░░░████
   Cause: CPU is the bottleneck (data loading, Python overhead, preprocessing)
   Fix: Increase DataLoader num_workers, prefetch, or move preprocessing to GPU

2. NO COMPUTE/NCCL OVERLAP (DDP issue)
   Timeline shows: ██compute██ ██nccl██ ██compute██ ██nccl██
   Should be:      ██compute████████████
                        ██nccl████████
   Fix: Tune DDP bucket_cap_mb, ensure backward is producing gradients continuously

3. MANY SMALL KERNELS WITH GAPS
   Timeline shows: █░█░█░█░█░█░█░  (tiny kernels with launch gaps)
   Cause: Python loop launching individual CUDA ops
   Fix: torch.compile, CUDAGraphs, or manual kernel fusion

4. LARGE D2H/H2D COPIES IN CRITICAL PATH
   Timeline shows: ████copy████ blocking compute
   Cause: .cpu(), .item(), .numpy() in hot loop
   Fix: Accumulate on GPU, transfer only final results

5. CUDA API SERIALIZATION
   Timeline shows: cudaMalloc/cudaFree during training
   Cause: Dynamic tensor allocation in hot path
   Fix: Pre-allocate buffers, use memory pool (torch.cuda.memory)
```

### NVTX Annotations (Custom Markers)

```python
import torch.cuda.nvtx as nvtx

# Range markers (show up as colored bars in nsys timeline)
nvtx.range_push("data_loading")
batch = next(dataloader)
nvtx.range_pop()

nvtx.range_push("forward")
output = model(batch)
nvtx.range_pop()

# Or use decorator
@nvtx.range("my_function")
def my_function():
    ...

# Or context manager
with nvtx.range("critical_section"):
    result = compute()
```

## NVIDIA Nsight Compute (ncu) — Kernel-Level Profiler

**Purpose:** Deep-dive into a SINGLE kernel's performance. Answers: why is this kernel slow? Memory-bound? Compute-bound? What's stalling the warps?

**Docs:** https://docs.nvidia.com/nsight-compute/

**IMPORTANT:** ncu replays each kernel multiple times for accurate metrics. This makes profiling MUCH slower than nsys. Profile only the kernels you care about.

### Basic Usage

```bash
# Profile ALL kernels (slow! use only for short runs)
ncu --set full -o report python kernel_test.py

# Profile only specific kernel by name
ncu --kernel-name "ampere_sgemm" --set full -o report python test.py

# Profile only kernel #5-#10 (skip warmup kernels)
ncu --launch-skip 4 --launch-count 6 --set full -o report python test.py

# Profile with roofline data
ncu --set roofline -o roofline_report python test.py

# Lightweight: just occupancy and memory throughput
ncu --set basic -o report python test.py
```

### Metric Sets

```bash
--set basic       # Occupancy, compute/memory throughput (fast)
--set full        # All metrics: stall reasons, instruction mix, memory chart (slow)
--set roofline    # Roofline chart data (medium)
--set source      # Source-level metrics with line correlation (needs -lineinfo)

# Custom metrics (pick only what you need — faster)
ncu --metrics \
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    dram__throughput.avg.pct_of_peak_sustained_elapsed,\
    launch__occupancy,\
    sm__warps_active.avg.pct_of_peak_sustained_elapsed \
    -o report python test.py
```

### Key Metrics to Understand

```
THROUGHPUT METRICS (% of peak):
  Compute (SM) Throughput   → % of peak FLOPS being used
  Memory (DRAM) Throughput  → % of peak HBM bandwidth being used

  If Compute high, Memory low → compute-bound → optimize arithmetic
  If Memory high, Compute low → memory-bound → optimize access patterns
  If BOTH low → latency-bound → increase occupancy or reduce stalls

OCCUPANCY:
  Achieved Occupancy        → actual active warps / max warps per SM
  Theoretical Occupancy     → limited by registers, shared memory, block size

  Low occupancy causes:
    Too many registers per thread → reduce or use --maxrregcount
    Too much shared memory per block → reduce or partition
    Block size too large/small → tune block dimensions

STALL REASONS (why warps are waiting):
  Stall Long Scoreboard     → waiting for global memory (HBM) load
  Stall Short Scoreboard    → waiting for shared memory or L2
  Stall Wait                → waiting for barrier (__syncthreads)
  Stall Not Selected        → scheduler chose another warp (OK if occupancy high)
  Stall Math Pipe Throttle  → compute units busy (compute-bound, actually good!)
  Stall MIO Throttle        → memory instruction queue full

INSTRUCTION MIX:
  FP32/FP16/INT instructions → actual compute work
  Load/Store instructions    → memory operations
  Control instructions       → branches, jumps

  High Load/Store ratio → memory-bound
  High Control ratio → divergence or complex control flow
```

### SLURM / Remote ncu

```bash
# ncu needs elevated permissions on some systems
# Check: ncu --query-metrics (if this fails, ask admin for nvidia-modprobe)

# Via srun
srun -n1 --gres=gpu:1 ncu --set full \
    --kernel-name "my_kernel" \
    -o /path/report python test.py

# ncu generates .ncu-rep files → transfer to local machine for GUI
scp maple-n05:/path/report.ncu-rep ./
ncu-ui report.ncu-rep    # Open in Nsight Compute GUI
```

### ncu + Source Correlation

```bash
# Compile CUDA with line info (no performance impact)
nvcc -O3 -lineinfo -arch=sm_90a kernel.cu -o kernel

# Profile with source metrics
ncu --set source -o report ./kernel

# In GUI: see per-line metrics (which line causes stalls, memory traffic)
```

### Comparing Kernel Versions

```bash
# Profile baseline
ncu --set full -o baseline python v1.py

# Profile optimized version
ncu --set full -o optimized python v2.py

# Compare in GUI: File → Open → Add Baseline
# Or CLI diff:
ncu --import baseline.ncu-rep --import optimized.ncu-rep --page diff
```

## Tool Selection Decision Tree

```
What do I need to know?
│
├─ "Where is time being spent overall?"
│  → nsys (system-wide timeline)
│  → Look at: CPU/GPU balance, kernel gaps, NCCL overlap
│
├─ "Why is this specific kernel slow?"
│  → ncu (kernel deep-dive)
│  → Look at: throughput %, stall reasons, occupancy, roofline position
│
├─ "Is my PyTorch code efficient?"
│  → torch.profiler (op-level, Python-integrated)
│  → Look at: CUDA time per op, memory allocation, op counts
│
├─ "Where is GPU memory going?"
│  → torch.cuda.memory._record_memory_history + memory_viz
│  → Or: ncu with memory chart
│
├─ "Is my DDP communication overlapping?"
│  → nsys with NCCL trace
│  → Look at: NCCL bars overlapping backward compute bars
│
└─ "Quick: is GPU even being used?"
   → nvidia-smi dmon -s u -d 1  (live GPU utilization)
   → If < 50%: definitely CPU-bound, start with nsys
```

## Profiling Workflow (Recommended Order)

```
Step 1: Quick check
  nvidia-smi dmon -s u     → Is GPU utilized at all?

Step 2: System overview (nsys)
  nsys profile -t cuda,nvtx,osrt -o overview python train.py
  nsys stats overview.nsys-rep
  → Identify: CPU bottleneck? kernel gaps? NCCL issues?

Step 3: Drill into slow kernels (ncu)
  ncu --kernel-name "the_slow_kernel" --set full -o detail python train.py
  → Identify: memory-bound? compute-bound? occupancy issue?

Step 4: Roofline positioning
  ncu --set roofline -o roofline python train.py
  → Where does my kernel sit? How far from peak?

Step 5: Fix and re-profile
  Make changes → re-run Step 2-4 → compare
  → Did throughput improve? Did stalls reduce?
```

## Roofline Analysis

```
Steps to build a roofline:
1. Get peak FLOPS for your GPU:
   nvidia-smi -q | grep "Max Clock"
   # A100: 19.5 TFLOPS (FP32), 312 TFLOPS (TF32 tensor)
   # H100: 67 TFLOPS (FP32), 989 TFLOPS (TF32 tensor)
   # GH200: ~60 TFLOPS (FP32)

2. Get peak memory bandwidth:
   # A100: 2039 GB/s (HBM2e)
   # H100: 3350 GB/s (HBM3)
   # GH200: ~4000 GB/s (HBM3)

3. Compute ridge point:
   ridge = peak_flops / peak_bandwidth
   # A100: 19.5T / 2039G ≈ 9.6 FLOPs/byte (FP32)
   # H100: 67T / 3350G ≈ 20 FLOPs/byte (FP32)

4. For your kernel, compute:
   AI = FLOPs_executed / Bytes_transferred

5. Plot on roofline:
   - If AI < ridge → memory-bound → optimize memory access
   - If AI > ridge → compute-bound → optimize arithmetic/use tensor cores
```

**Using ncu for roofline:**
```bash
ncu --set roofline -o roofline_report python kernel_test.py
# Opens interactive roofline chart in ncu GUI
```

## Memory Profiling

```python
# PyTorch memory snapshot
torch.cuda.memory._record_memory_history(max_entries=100000)
# ... run code ...
torch.cuda.memory._dump_snapshot("mem_snapshot.pickle")
# Visualize at: https://pytorch.org/memory_viz

# Quick memory check
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
print(f"Reserved:  {torch.cuda.memory_reserved() / 1e9:.1f} GB")
print(f"Max alloc: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
```

## Common Bottleneck Patterns

| Symptom | Likely Cause | How to Confirm |
|---------|-------------|---------------|
| GPU util < 50% | CPU bottleneck (data loading) | nsys: GPU idle gaps |
| GPU util ~100%, slow | Memory-bound kernel | ncu: memory throughput near peak |
| NCCL > 30% of step time | Communication bottleneck | profiler: nccl:allReduce time |
| Many small kernels | Launch overhead | nsys: kernel gaps in timeline |
| OOM despite small model | Activation memory | memory snapshot: gradient checkpointing |
