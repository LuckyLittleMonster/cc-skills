# PyTorch Distributed Data Parallel (DDP) Internals

## How DDP Works

```
Forward pass:
  Each rank computes loss independently with its data shard

Backward pass (overlapped with gradient computation):
  1. Gradients are computed layer by layer (back to front)
  2. Gradients are grouped into "buckets" (default 25MB)
  3. When a bucket is full, AllReduce starts IMMEDIATELY
     (overlaps with computing gradients for earlier layers)
  4. AllReduce averages gradients across all ranks

Optimizer step:
  Each rank applies identical averaged gradients → models stay in sync
```

**Key insight:** Communication overlaps with computation. Bucket size controls the trade-off:
- Smaller buckets → more overlap, more launch overhead
- Larger buckets → less overhead, less overlap opportunity

## DDP Configuration

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
dist.init_process_group(
    backend='nccl',        # NCCL for GPU, gloo for CPU
    init_method='env://',  # Or 'tcp://host:port'
)

model = Model().cuda()
model = DDP(
    model,
    device_ids=[local_rank],
    output_device=local_rank,
    find_unused_parameters=False,  # True only if needed (slower!)
    gradient_as_bucket_view=True,  # Memory optimization
    static_graph=True,             # If graph doesn't change (faster)
    bucket_cap_mb=25,              # Bucket size in MB
)
```

### Critical Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `find_unused_parameters` | False | Scans for unused params each iteration. **Costly.** Only enable if some params don't get gradients. |
| `gradient_as_bucket_view` | False | Avoids copying gradients into buckets. **Always enable.** |
| `static_graph` | False | Optimizes for static computation graphs. **Enable when possible.** |
| `bucket_cap_mb` | 25 | Gradient bucket size. Tune based on model size and network bandwidth. |

## Common DDP Bugs

### 1. Unused Parameters
```python
# BUG: property_predictor exists but isn't used in forward()
# → DDP hangs waiting for its gradient AllReduce
class Model(nn.Module):
    def __init__(self):
        self.encoder = Encoder()
        self.property_predictor = PropertyHead()  # ← unused in loss
    def forward(self, x):
        return self.encoder(x)  # property_predictor never called

# FIX: find_unused_parameters=True (slower) or remove the unused module
model = DDP(model, find_unused_parameters=True)
```

### 2. Non-Deterministic Operations
```python
# BUG: different ranks take different code paths → gradient desync
if random.random() > 0.5:  # ← different on each rank!
    output = self.branch_a(x)
else:
    output = self.branch_b(x)

# FIX: use rank-consistent control flow, or seed consistently
```

### 3. Gradient Accumulation
```python
# WRONG: AllReduce happens every backward() call
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()  # ← AllReduce here, wasted communication!
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# RIGHT: disable sync during accumulation
for i, batch in enumerate(dataloader):
    context = model.no_sync() if (i + 1) % accumulation_steps != 0 else nullcontext()
    with context:
        loss = model(batch) / accumulation_steps
        loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 4. Model Saving/Loading
```python
# SAVE: only on rank 0, save the .module (unwrapped model)
if dist.get_rank() == 0:
    torch.save(model.module.state_dict(), 'checkpoint.pt')
dist.barrier()  # Wait for rank 0 to finish saving

# LOAD: load on all ranks consistently
map_location = f'cuda:{local_rank}'
state_dict = torch.load('checkpoint.pt', map_location=map_location)
model.module.load_state_dict(state_dict)
```

## DistributedSampler

```python
from torch.utils.data import DataLoader, DistributedSampler

sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
    drop_last=True,  # Ensures all ranks get same number of batches
)

dataloader = DataLoader(
    dataset,
    batch_size=per_gpu_batch_size,
    sampler=sampler,
    num_workers=4,
    pin_memory=True,
)

# CRITICAL: set epoch for proper shuffling each epoch
for epoch in range(epochs):
    sampler.set_epoch(epoch)  # ← Don't forget this!
    for batch in dataloader:
        ...
```

## Mixed Precision with DDP

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    with autocast(device_type='cuda', dtype=torch.bfloat16):
        loss = model(batch)
    scaler.scale(loss).backward()  # AllReduce in bf16 → less bandwidth
    scaler.step(optimizer)
    scaler.update()
```

**DDP + AMP benefits:**
- Gradient AllReduce in fp16/bf16 → 2× less communication
- Forward/backward in mixed precision → faster compute
- GH200/H100: prefer bf16 over fp16 (no loss scaling needed)

## Debugging DDP

```bash
# Enable NCCL debug logging
NCCL_DEBUG=INFO torchrun --nproc_per_node=2 train.py

# Check for hanging: add timeout
dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=5))

# Verify all ranks have same model
for name, param in model.named_parameters():
    tensor = param.data.clone()
    dist.all_reduce(tensor)
    tensor /= dist.get_world_size()
    if not torch.allclose(param.data, tensor, atol=1e-5):
        print(f"Rank {rank}: {name} is out of sync!")
```

## Launch Methods

```bash
# torchrun (recommended)
torchrun --nproc_per_node=4 --nnodes=2 \
    --node_rank=0 --master_addr=node0 --master_port=29500 \
    train.py

# SLURM + srun
srun -N 2 --ntasks-per-node=1 --gres=gpu:1 \
    python -u train.py

# SLURM env vars automatically set by srun:
#   SLURM_PROCID → global rank
#   SLURM_LOCALID → local rank
#   SLURM_NTASKS → world size
```
