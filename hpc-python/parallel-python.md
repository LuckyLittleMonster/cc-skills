# Python Parallelism — Threading, Multiprocessing, Async

## Decision Tree: Which Parallelism Model?

```
What is the bottleneck?
│
├─ I/O bound (network, disk, API calls)
│  ├─ Many concurrent connections → asyncio
│  ├─ Few connections, simple logic → threading (ThreadPoolExecutor)
│  └─ Mixed I/O + light CPU → asyncio + thread executor
│
├─ CPU bound (parsing, preprocessing, numerics)
│  ├─ Pure Python computation → multiprocessing (mp.Pool)
│  ├─ NumPy/Torch operations → already parallel (MKL/CUDA), don't wrap
│  └─ Many independent tasks → mp.Pool or ProcessPoolExecutor
│
├─ GPU bound (model inference, tensor ops)
│  ├─ Single model, multiple inputs → batch the inputs, don't parallelize
│  ├─ Multiple independent models → CUDA streams (see latency-hiding.md)
│  └─ Data preprocessing feeding GPU → mp.Pool for prep, single GPU thread
│
└─ Mixed (CPU prep → GPU compute → CPU postprocess)
   ├─ Pipeline pattern → producer/consumer with mp.Queue or asyncio
   ├─ Overlap CPU/GPU → CUDA streams + async copies (see latency-hiding.md)
   └─ RL environments → mp.Pool for env steps, single GPU for model
```

## Threading (threading / concurrent.futures.ThreadPoolExecutor)

### When to Use
- I/O-bound tasks: file reads, HTTP requests, SSH commands, database queries
- Waiting on external processes (subprocess, network)
- GUI event handling

### When NOT to Use
- CPU-bound computation (GIL prevents true parallelism)
- CUDA operations (thread-safe but no speedup — GPU serializes anyway)

### Patterns

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

# Pattern 1: Map over I/O tasks
def fetch_url(url):
    return requests.get(url).text

with ThreadPoolExecutor(max_workers=16) as pool:
    results = list(pool.map(fetch_url, urls))

# Pattern 2: Fire-and-forget with futures
with ThreadPoolExecutor(max_workers=8) as pool:
    futures = {pool.submit(fetch_url, url): url for url in urls}
    for future in as_completed(futures):
        url = futures[future]
        try:
            data = future.result()
        except Exception as e:
            print(f"{url} failed: {e}")

# Pattern 3: Background I/O while GPU computes
import threading

def save_checkpoint_async(state_dict, path):
    """Save in background thread — doesn't block GPU training."""
    t = threading.Thread(target=torch.save, args=(state_dict, path))
    t.start()
    return t  # caller can join() if needed

# In training loop:
if step % save_interval == 0:
    # .cpu() needed to avoid GPU memory pinning during async save
    sd = {k: v.cpu() for k, v in model.state_dict().items()}
    save_thread = save_checkpoint_async(sd, f"ckpt_{step}.pt")
```

### GIL Facts
```
Python GIL (Global Interpreter Lock):
- Only ONE thread executes Python bytecode at a time
- Released during I/O operations (file, network, sleep)
- Released during C extensions (NumPy, torch ops)
- NOT released during pure Python computation

Implication:
  threading + I/O      → real parallelism ✓
  threading + NumPy    → real parallelism ✓ (C extension releases GIL)
  threading + Python   → NO parallelism ✗ (GIL serializes)
  threading + torch    → safe but GPU serializes anyway
```

## Multiprocessing (multiprocessing / concurrent.futures.ProcessPoolExecutor)

### When to Use
- CPU-bound Python computation (GIL bypass via separate processes)
- Data preprocessing pipelines
- Parallel environment stepping in RL
- Any pure-Python loop over large datasets

### CRITICAL: CUDA + Fork Interaction

```python
# ⚠️ DANGER: fork() after CUDA init = DEADLOCK or CORRUPTION
# CUDA runtime is not fork-safe!

# RULE: Create mp.Pool BEFORE any CUDA operation
import multiprocessing as mp
pool = mp.Pool(32)            # ← BEFORE torch.cuda.anything()
model = Model().cuda()         # ← AFTER pool creation

# Or use spawn/forkserver (safer but slower startup):
mp.set_start_method('spawn')   # Safe with CUDA, but pickles args
pool = mp.Pool(32)

# Or use fork but ensure workers don't touch CUDA:
def cpu_only_worker(smiles):
    """This function must NOT import torch.cuda or use GPU."""
    from rdkit import Chem
    return Chem.MolFromSmiles(smiles) is not None
pool = mp.Pool(32)
results = pool.map(cpu_only_worker, smiles_list)
```

### Patterns

```python
# Pattern 1: Simple parallel map
from multiprocessing import Pool

def process_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.MolWt(mol) if mol else None

with Pool(32) as pool:
    weights = pool.map(process_molecule, smiles_list)

# Pattern 2: Chunked processing (reduce IPC overhead)
with Pool(32) as pool:
    weights = pool.map(process_molecule, smiles_list, chunksize=100)
    # chunksize=100: send 100 items per IPC call instead of 1

# Pattern 3: imap_unordered for streaming results
with Pool(32) as pool:
    for result in pool.imap_unordered(process_molecule, smiles_list, chunksize=100):
        if result is not None:
            results.append(result)

# Pattern 4: Shared memory (avoid pickling large data)
import multiprocessing.shared_memory as shm
import numpy as np

# Create shared array
arr = np.zeros((1000000, 128), dtype=np.float32)
shared = shm.SharedMemory(create=True, size=arr.nbytes)
shared_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shared.buf)
np.copyto(shared_arr, arr)

def worker_with_shared(idx_range, shm_name, shape, dtype):
    existing = shm.SharedMemory(name=shm_name)
    arr = np.ndarray(shape, dtype=dtype, buffer=existing.buf)
    # Read arr[idx_range[0]:idx_range[1]] — zero-copy!
    return arr[idx_range[0]:idx_range[1]].sum()

# Pattern 5: ProcessPoolExecutor (simpler API)
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(max_workers=32) as pool:
    futures = [pool.submit(process_molecule, s) for s in smiles_list]
    results = [f.result() for f in futures]
```

### Performance Tips
```
- chunksize matters: default=1 has high IPC overhead for small tasks
  Rule of thumb: chunksize = len(data) // (num_workers * 4)
- Pickling overhead: large args are pickled/unpickled per call
  Fix: use shared memory, memory-mapped files, or global state with fork
- Process startup: spawn/forkserver is slower than fork
  Fix: reuse Pool, don't create/destroy per iteration
- Too many workers: context switching overhead, memory pressure
  Rule: num_workers ≤ num_cpu_cores, typically 0.5-1× core count
```

## Asyncio (asyncio / aiohttp / aiofiles)

### When to Use
- High-concurrency I/O: 100s-1000s of concurrent connections
- Non-blocking event loops
- Coordinating multiple I/O-bound coroutines
- Web scraping, API calls, streaming data

### When NOT to Use
- CPU-bound work (asyncio is single-threaded!)
- Simple sequential I/O (threading is simpler)
- Code that calls blocking libraries without async support

### Patterns

```python
import asyncio
import aiohttp

# Pattern 1: Concurrent HTTP requests
async def fetch(session, url):
    async with session.get(url) as response:
        return await response.text()

async def fetch_all(urls, max_concurrent=50):
    semaphore = asyncio.Semaphore(max_concurrent)  # Rate limit
    async with aiohttp.ClientSession() as session:
        async def limited_fetch(url):
            async with semaphore:
                return await fetch(session, url)
        return await asyncio.gather(*[limited_fetch(url) for url in urls])

results = asyncio.run(fetch_all(urls))

# Pattern 2: Producer-consumer pipeline
async def producer(queue, data_source):
    for item in data_source:
        processed = preprocess(item)  # CPU work
        await queue.put(processed)
    await queue.put(None)  # Sentinel

async def consumer(queue, model):
    while True:
        item = await queue.get()
        if item is None:
            break
        result = model(item)  # GPU work
        yield result

# Pattern 3: Async + threading (for blocking calls)
async def run_blocking_in_thread(func, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, func, *args)

# Pattern 4: Async file I/O
import aiofiles

async def read_files_async(paths):
    async def read_one(path):
        async with aiofiles.open(path, 'r') as f:
            return await f.read()
    return await asyncio.gather(*[read_one(p) for p in paths])

# Pattern 5: Timeout handling
async def fetch_with_timeout(url, timeout=10):
    try:
        async with asyncio.timeout(timeout):
            return await fetch(session, url)
    except asyncio.TimeoutError:
        return None
```

## Choosing the Right Model: Quick Reference

| Scenario | Best Choice | Why |
|----------|------------|-----|
| 100 HTTP API calls | `asyncio + aiohttp` | High concurrency, I/O bound |
| 10 SSH commands | `ThreadPoolExecutor` | Simple, few connections |
| Parse 1M SMILES | `mp.Pool(32)` | CPU bound, GIL bypass |
| Save checkpoint during training | `threading.Thread` | Background I/O, don't block GPU |
| Batch GPU inference | **Don't parallelize** — batch inputs | GPU parallelism is internal |
| RL env.step() × 64 | `mp.Pool` (fork before CUDA) | CPU-bound env, parallel steps |
| DataLoader preprocessing | `DataLoader(num_workers=N)` | Built-in mp, handles CUDA safety |
| Mixed CPU prep + GPU compute | Pipeline + CUDA streams | See `latency-hiding.md` |
| Parallel file I/O + GPU | `ThreadPoolExecutor` + CUDA stream | I/O in threads, GPU async |

## Common Anti-Patterns

```python
# BAD: threading for CPU-bound work (GIL bottleneck)
with ThreadPoolExecutor(32) as pool:
    pool.map(heavy_python_computation, data)  # ← NO speedup!
# FIX: Use mp.Pool(32) instead

# BAD: multiprocessing for tiny tasks (IPC > computation)
with Pool(32) as pool:
    pool.map(lambda x: x+1, range(100))  # ← overhead > work!
# FIX: Vectorize with numpy: np.array(range(100)) + 1

# BAD: creating Pool inside training loop
for epoch in range(100):
    pool = Pool(32)           # ← startup cost every epoch!
    pool.map(preprocess, data)
    pool.close()
# FIX: Create pool once, reuse across epochs

# BAD: asyncio.run() inside Jupyter notebook (event loop conflict)
asyncio.run(main())  # ← RuntimeError: event loop already running
# FIX: await main() or use nest_asyncio

# BAD: Mixing CUDA operations in forked workers
pool = Pool(32)  # fork()
def gpu_worker(x):
    return model(x.cuda())  # ← DEADLOCK or SEGFAULT!
# FIX: Keep GPU in main process, use pool only for CPU work

# BAD: Sending large tensors through mp.Queue
queue.put(huge_tensor)  # ← Pickles entire tensor!
# FIX: Use shared memory or torch.multiprocessing.Queue (shares CUDA tensors)
```

## torch.multiprocessing (Special)

PyTorch's `torch.multiprocessing` extends stdlib `multiprocessing` with CUDA tensor sharing:

```python
import torch.multiprocessing as mp

# Shares CUDA tensors between processes via shared memory
# No pickling, no copying — processes access same GPU memory

# Pattern: Shared model for parallel RL environments
model = Model().cuda()
model.share_memory()  # Move to shared memory

def worker(rank, model, queue):
    """Worker process can READ model without copy."""
    with torch.no_grad():
        obs = get_observation()
        action = model(obs.cuda())
        queue.put(action.cpu())

processes = []
queue = mp.Queue()
for i in range(num_workers):
    p = mp.Process(target=worker, args=(i, model, queue))
    p.start()
    processes.append(p)
```
