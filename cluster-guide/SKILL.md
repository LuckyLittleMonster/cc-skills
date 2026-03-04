---
name: cluster-guide
description: Unified reference guide for Maple GH200 and Delta A100 clusters - hardware specs, partitions, NCCL config, execution tiers, sbatch templates. Auto-detects cluster from hostname.
user-invocable: true
---

# Cluster Reference Guide

## Step 0: Detect Cluster

```bash
_HOSTNAME=$(hostname -f 2>/dev/null || hostname)
if [[ "$_HOSTNAME" == *delta* || "$_HOSTNAME" == dt-* || "$_HOSTNAME" == gpua* || "$_HOSTNAME" == gpuc* || "$_HOSTNAME" == gpue* ]]; then
    CLUSTER="delta"
elif [[ "$_HOSTNAME" == *maple* ]]; then
    CLUSTER="maple"
elif [[ "$(uname -m)" == "aarch64" ]]; then
    CLUSTER="maple"
else
    CLUSTER="unknown"
fi
```

Report the detected cluster. If unknown, ask the user which cluster they are targeting.

## Cluster Comparison

| Feature | Maple | Delta |
|---------|-------|-------|
| GPU | 1x GH200 96GB HBM3 | A100-SXM4-40GB (x4 or x8) |
| CPU | 72 ARM (aarch64) | 64 x86_64 (x4), 128 (x8) |
| GPU/node | 1 | 4 or 8 |
| Login GPU | Yes (shared) | No |
| Account | `dpp` | `bcpt-delta-gpu` |
| Partition | `maple` | `gpuA100x4`, `gpuA100x8` |
| Network | InfiniBand | Slingshot CXI |
| Module loads | None | `ml cudatoolkit && ml libfabric && ml aws-ofi-nccl` |
| Conda | `source ~/.bashrc_maple 2>/dev/null && conda activate rl4` | `conda activate DistNNP` |
| srun flags | `srun -u` | `srun --cpu_bind=cores --gpu-bind=none` |

## Execution Tier Priority (Both Clusters)

**Always prefer the highest tier that fits the task:**

| Tier | Method | When to Use | Buffering |
|------|--------|-------------|-----------|
| 1 | **Login node** (direct) | Maple only: CPU<32, GPU<=1, <1h, login GPU<=70% | Line buffer |
| 2 | **SSH existing node** | Have allocation with free resources | Line buffer |
| 3 | **salloc** (interactive) | Need fresh allocation, interactive | Line buffer |
| 4 | **sbatch** (batch) | Multi-node / long unattended / array | Block buffer |

**Delta note**: No login GPU, so effective starting tier is 3 (salloc) or 4 (sbatch).

Debug and newly-written code: always use **line buffer** (`python -u` or `PYTHONUNBUFFERED=1`).
Production training (proven, >12h): use **block buffer** (default, higher I/O throughput).

## Cluster-Specific Details

### Maple GH200

See `maple.md` for:
- Hardware, nodes table, QOS tiers (3 GPU default, night QOS for burst)
- salloc/sbatch templates, array jobs, DDP multi-node
- Conda environments (`rl4`, `nvmolkit`)
- Gotchas (--mem=0 context, squeue time format, login node etiquette)

### Delta A100

See `delta.md` for:
- Hardware (A100x4 NV4, A100x8 NV12/NVSwitch), partitions
- Account, conda, salloc/sbatch templates (single-node, multi-node)
- Partition selection logic, srun pattern

See `delta-nccl.md` for:
- **MANDATORY** module loads (20-50x penalty if missing)
- CXI/Slingshot provider settings
- NCCL network config (asymmetric GDR, AWS Libfabric)
- PyTorch CUDA allocator fix
- Complete copy-paste env block

## Quick Command Reference

### Both Clusters

```bash
# My jobs
squeue -u $USER

# Job history
sacct -u $USER --format=JobID,JobName,State,Elapsed -S today

# Release allocation
scancel <jobid>
```

### Maple-Specific

```bash
# Login node GPU
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free --format=csv

# My maple jobs
squeue -u $USER -p maple

# Partition status
sinfo -p maple

# Allocate node
salloc -p maple -A dpp --cpus-per-task=72 --gres=gpu:1 --mem=0 -t 4:00:00 --no-shell
```

### Delta-Specific

```bash
# My delta jobs
squeue -u $USER -A bcpt-delta-gpu

# Partition status
sinfo -p gpuA100x4,gpuA100x8

# Allocate x4 node
salloc -p gpuA100x4 -A bcpt-delta-gpu --nodes=1 --ntasks-per-node=4 --gpus-per-task=1 --mem=120g -t 1:00:00 --no-shell

# Allocate x8 node
salloc -p gpuA100x8 -A bcpt-delta-gpu --nodes=1 --ntasks-per-node=1 --gpus-per-task=8 --cpus-per-task=64 --mem=180g -t 1:00:00 --no-shell
```
