# Maple GH200 Cluster Reference

## Hardware

| Component | Spec |
|-----------|------|
| GPU | NVIDIA GH200 (Grace Hopper), 96GB HBM3 |
| CPU | 72 ARM cores per node (Grace, aarch64) |
| GPU/node | 1 |
| CPU-GPU link | NVLink-C2C (900 GB/s) |
| Inter-node | InfiniBand |
| Login node | **Also has full GH200** (shared, for quick tasks) |

## Nodes

| Node | CPUs | GPU | Notes |
|------|------|-----|-------|
| maple-n01 ~ n09 | 72 | 1x GH200 | Standard compute nodes |
| maple-n101 ~ n104 | 144 | 1x GH200 | Dual-CPU nodes (2x72) |

## SLURM Configuration

| Parameter | Value |
|-----------|-------|
| Partition | `maple` |
| Account | `dpp` (primary), `app` (alternate) |
| Default QOS | `part_maple` |
| MaxTime | 1 day |

## QOS Tiers (GPU Limits)

| QOS | GPU Limit | CPU Limit | Memory Limit | Use Case |
|-----|-----------|-----------|--------------|----------|
| `part_maple` (default) | **3 GPUs/user** | 216 | 1.7TB | Normal jobs |
| `part_maple_night` | **No explicit limit** | -- | -- | Burst/overflow jobs |
| `maple_reserve` | **6 GPUs/user** | 432 | 3.4TB | Reserved access |

**Key insight**: Default limit is **3 GPUs** (not 2). Use `--qos=part_maple_night` for more.

## salloc Pattern

```bash
# Allocate entire node (72 CPU, all memory, 1 GPU)
salloc --partition=maple --account=dpp \
  --nodes=1 --cpus-per-task=72 --gres=gpu:1 \
  --mem=0 --time=4:00:00 --no-shell

# Find allocated node
squeue -u $USER -p maple -h -o "%i %N %T" | grep RUNNING

# SSH and run
ssh <node> "cd /your/project && \
  source ~/.bashrc_maple 2>/dev/null && \
  conda activate rl4 && \
  export CUDA_VISIBLE_DEVICES=0 && \
  python -u your_script.py [args]"

# Release when done
scancel <jobid>
```

## sbatch Template

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

source ~/.bashrc_maple 2>/dev/null
conda activate rl4
cd /path/to/project

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv

python your_training_script.py [args]

echo "Done at $(date)"
```

### Array Job

```bash
#SBATCH --array=1-N%3    # %3 = max 3 concurrent (default QOS GPU limit)

case ${SLURM_ARRAY_TASK_ID} in
  1) ARGS="--lr 1e-4" ;;
  2) ARGS="--lr 5e-5" ;;
esac
python your_script.py $ARGS
```

### Burst Array (night QOS)

```bash
#SBATCH --qos=part_maple_night
#SBATCH --array=1-8       # No throttle needed
```

### DDP Multi-Node

```bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1

export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500
srun -u python ddp_launcher.py [args]
```

## SSH Piggyback Pattern

```bash
# Check resources
ssh <node> nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader

# Run foreground
ssh <node> "cd $(pwd) && source ~/.bashrc_maple 2>/dev/null && \
  conda activate rl4 && python -u script.py"

# Run background (survives disconnect)
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

1. **`--mem=0` context**: salloc = claim entire node (good). sbatch with multiple jobs = blocks others (`QOSMaxMemoryPerUser`). Use `--mem=64G` for sbatch.
2. **squeue TIME format**: `45:23` = 45 min 23 sec, NOT 45 hours. Format: `MM:SS` | `H:MM:SS` | `D-HH:MM:SS`.
3. **Login node is shared**: Full GH200 but shared. Quick tasks only (<1h, <32 CPU). Check `nvidia-smi` first.
4. **salloc holds resources while idle**: Release promptly with `scancel`.
5. **conda stderr noise**: `source ~/.bashrc_maple` emits module messages. Suppress with `2>/dev/null`.
6. **SSH processes die with SLURM job**: SSH-piggybacked processes are killed when the host job ends. Use sbatch for critical long tasks.
