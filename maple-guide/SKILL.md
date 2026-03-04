---
name: maple-guide
description: Reference guide for maple GH200 cluster - hardware specs, QOS limits, execution tiers, gotchas, sbatch templates. Consult when any SLURM or maple-related question arises.
user-invocable: true
---

# Maple GH200 Cluster Reference Guide

## Hardware

| Component | Spec |
|-----------|------|
| GPU | NVIDIA GH200 (Grace Hopper), 96GB HBM3 |
| CPU | 72 ARM cores per node (Grace, aarch64) |
| GPU/node | 1 |
| CPU-GPU link | NVLink-C2C (900 GB/s) |
| Inter-node | InfiniBand |
| Login node | **Also has full GH200** (shared, for quick tasks) |

## Execution Tier Priority

**Always prefer the highest tier that fits the task:**

| Tier | Method | When to Use | Buffering |
|------|--------|-------------|-----------|
| 1 | **Login node** (direct) | CPU<32, GPU≤1, <1h, login GPU≤70% | Line buffer |
| 2 | **SSH existing node** | Have allocation with free resources | Line buffer |
| 3 | **salloc** (interactive) | Need fresh allocation, interactive | Line buffer |
| 4 | **sbatch** (batch) | Multi-node / long unattended / array | Block buffer |

Debug and newly-written code: always use **line buffer** (`python -u` or `PYTHONUNBUFFERED=1`).
Production training (proven, >12h): use **block buffer** (default, higher I/O throughput).

## Nodes

| Node | CPUs | GPU | Notes |
|------|------|-----|-------|
| maple-n01 ~ n09 | 72 | 1× GH200 | Standard compute nodes |
| maple-n101 ~ n104 | 144 | 1× GH200 | Dual-CPU nodes (2×72) |

## SLURM Configuration

| Parameter | Value |
|-----------|-------|
| Partition | `maple` |
| Account | `dpp` (primary), `app` (alternate) |
| Default QOS | `part_maple` |
| AllowQos | ALL |
| PreemptMode | REQUEUE |
| MaxTime | 1 day |
| Node naming | `maple-n01` ~ `maple-n09`, `maple-n101` ~ `maple-n104` |

## QOS Tiers (GPU Limits)

| QOS | GPU Limit | CPU Limit | Memory Limit | Flags | Use Case |
|-----|-----------|-----------|--------------|-------|----------|
| `part_maple` (default) | **3 GPUs/user** | 216 | 1.7TB | DenyOnLimit | Normal jobs |
| `part_maple_night` | **No explicit limit** | — | — | DenyOnLimit | Burst/overflow jobs |
| `maple_reserve` | **6 GPUs/user** | 432 | 3.4TB | OverPartQOS | Reserved access |

**Key insight**: The default limit is **3 GPUs** (not 2). Use `--qos=part_maple_night` when you need more than 3.

## salloc Pattern (Tier 3)

```bash
# Allocate entire node without opening a shell (72 CPU, all memory, 1 GPU)
salloc --partition=maple --account=dpp \
  --nodes=1 --cpus-per-task=72 --gres=gpu:1 \
  --mem=0 --time=4:00:00 --no-shell

# Find allocated node
squeue -u $USER -p maple -h -o "%i %N %T" | grep RUNNING

# SSH and run (line-buffered for debug)
ssh <node> "cd /your/project && \
  source ~/.bashrc_maple 2>/dev/null && \
  conda activate rl4 && \
  export CUDA_VISIBLE_DEVICES=0 && \
  python -u your_script.py [args]"

# Release when done
scancel <jobid>
```

## sbatch Template (Tier 4)

```bash
#!/bin/bash
#SBATCH --job-name=<NAME>
#SBATCH --output=Experiments/logs/<NAME>_%j.out
#SBATCH --error=Experiments/logs/<NAME>_%j.err
#SBATCH --partition=maple
#SBATCH --account=dpp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# NOTE: No PYTHONUNBUFFERED for production runs (block buffer = better I/O)
# Add `export PYTHONUNBUFFERED=1` only for debug runs

source ~/.bashrc_maple 2>/dev/null
conda activate rl4
cd /path/to/project

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv

python your_training_script.py [args]

echo "Done at $(date)"
```

### Array Job (multiple experiments)

```bash
#SBATCH --array=1-N%3    # %3 = max 3 concurrent (default QOS GPU limit)

case ${SLURM_ARRAY_TASK_ID} in
  1) ARGS="--lr 1e-4" ;;
  2) ARGS="--lr 5e-5" ;;
esac
python your_script.py $ARGS
```

### Burst Array Job (using night QOS)

```bash
#SBATCH --qos=part_maple_night
#SBATCH --array=1-8       # No %N throttle needed — no explicit GPU limit

case ${SLURM_ARRAY_TASK_ID} in
  1) ARGS="--lr 1e-4" ;;
  2) ARGS="--lr 5e-5" ;;
  # ...
esac
python your_script.py $ARGS
```

### DDP Multi-Node (2-3 GPUs)

```bash
#SBATCH --nodes=2          # or 3 (within part_maple limit)
#SBATCH --ntasks-per-node=1

export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500
srun -u python ddp_launcher.py [args]
```

## SSH Piggyback Pattern (Tier 2)

```bash
# Check existing node resources
ssh <node> nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader

# Run foreground (interactive debug)
ssh <node> "cd $(pwd) && source ~/.bashrc_maple 2>/dev/null && \
  conda activate rl4 && python -u script.py"

# Run background (nohup, survives disconnect)
ssh <node> "cd $(pwd) && source ~/.bashrc_maple 2>/dev/null && \
  conda activate rl4 && \
  nohup python -u script.py > logs/out.log 2>&1 &"
```

## Conda Environments

| Env | Python | PyTorch | Use Case |
|-----|--------|---------|----------|
| `rl4` | 3.13 | 2.10.0+cu126 | Default: training, inference, RL |
| `nvmolkit` | 3.x | N/A | GPU molecular toolkit (kNN, Tanimoto) |

Always: `source ~/.bashrc_maple 2>/dev/null && conda activate <env>`

## Gotchas

### 1. `--mem=0` Context Matters
- **salloc**: Use `--mem=0` — you WANT the entire node (72 CPU + all memory + GPU). Exclusive access.
- **sbatch**: Use `--mem=64G` if you plan to run multiple concurrent jobs. `--mem=0` claims all memory → `QOSMaxMemoryPerUser` → blocks your other jobs.

### 2. squeue TIME Is Not HH:MM
`45:23` = 45 minutes 23 seconds, NOT 45 hours. Format: `MM:SS` | `H:MM:SS` | `D-HH:MM:SS`.

### 3. Login Node Is Shared
Has full GH200 but shared with all users. Quick tasks only (<1h, <32 CPU). Check `nvidia-smi` first.

### 4. QOS GPU Limits & Night QOS
Default `part_maple` allows **3 GPUs/user** (not 2). For more, use `--qos=part_maple_night` (no explicit GPU limit) or `--qos=maple_reserve` (6 GPUs). Night QOS is ideal for burst experiments on idle nodes.

### 5. salloc Holds Resources While Idle
Release allocations promptly: `scancel <jobid>`. Don't leave idle salloc running overnight.

### 6. conda stderr Noise
`source ~/.bashrc_maple` emits module messages. Suppress with `2>/dev/null`.

### 7. SSH Processes Die With SLURM Job
SSH-piggybacked processes are killed when the host SLURM job ends. For critical long tasks, use sbatch.

### 8. Batch Size Needs LR Scaling
Linear rule: `lr *= batch_size / base_batch_size`. Missing this drops ~0.01 AUC.

### 9. RCNS Negative Sampling Takes 30-90 min
On 655K-mol datasets, this CPU-bound preprocessing is normal. Don't kill it.

## Quick Command Reference

```bash
# Login node GPU check
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free --format=csv

# My jobs
squeue -u $USER -p maple

# Partition status
sinfo -p maple

# Job history
sacct -u $USER --format=JobID,JobName,State,Elapsed -S today

# Allocate entire node interactively (72 CPU, all mem, 1 GPU)
salloc -p maple -A dpp --cpus-per-task=72 --gres=gpu:1 --mem=0 -t 4:00:00 --no-shell

# Release allocation
scancel <jobid>

# GPU on remote node
ssh <node> nvidia-smi

# Kill piggybacked process
ssh <node> kill <PID>
```
