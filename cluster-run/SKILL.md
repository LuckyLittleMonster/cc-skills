---
name: cluster-run
description: Smart execution dispatcher for Maple GH200 and Delta A100 clusters. Evaluates task requirements and picks the most efficient execution method. Auto-detects cluster from hostname.
argument-hint: "[task description or command]"
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
---

# Cluster Smart Task Runner

Run tasks on Maple GH200 or Delta A100 nodes using the most efficient method available.

## Step 0: Detect Cluster & Set Defaults

```bash
_HOSTNAME=$(hostname -f 2>/dev/null || hostname)
if [[ "$_HOSTNAME" == *delta* || "$_HOSTNAME" == dt-* || "$_HOSTNAME" == gpua* || "$_HOSTNAME" == gpuc* || "$_HOSTNAME" == gpue* ]]; then
    CLUSTER="delta"
    ACCOUNT="bcpt-delta-gpu"
    PARTITION="gpuA100x4"      # default; use gpuA100x8 for 5-8 GPUs
    CONDA_INIT="conda activate DistNNP"
    SRUN_PREFIX="srun --cpu_bind=cores --gpu-bind=none"
    HAS_LOGIN_GPU=false
    GPU_PER_NODE=4              # default x4; 8 for x8 partition
elif [[ "$_HOSTNAME" == *maple* || "$(uname -m)" == "aarch64" ]]; then
    CLUSTER="maple"
    ACCOUNT="dpp"
    PARTITION="maple"
    CONDA_INIT="source ~/.bashrc_maple 2>/dev/null && conda activate rl4"
    SRUN_PREFIX="srun -u"
    HAS_LOGIN_GPU=true
    GPU_PER_NODE=1
else
    CLUSTER="unknown"
fi
```

Report detected cluster. If unknown, ask user.

## Decision Flowchart

```
Task requirements assessment
         |
         v
  Maple login available? (CPU<32, GPU<=1, <1h)
     +-- YES --> Check login GPU <= 70%
     |              +-- YES -> RUN ON LOGIN NODE (Tier 1)
     |              +-- NO  -> fall through v
     |
     +-- NO (Delta or heavy task)
         |
         v
  Any existing salloc/sbatch allocation?
     +-- YES --> SSH check: enough free CPU/GPU?
     |              +-- YES -> SSH + RUN ON EXISTING NODE (Tier 2)
     |              +-- NO  -> fall through v
     v
  Need <=1 node?
     +-- YES --> salloc interactive (Tier 3)
     v
  Multi-node or long unattended?
     +--------> sbatch (Tier 4)
```

**Delta note**: Tier 1 (login node) is never available on Delta — skip to Tier 2+.

## Step-by-Step Execution

### Phase 1: Assess Task Requirements

Evaluate the command/task to estimate:
- **CPU cores needed**: data loading workers, preprocessing -> typically 4-16
- **GPU needed**: 0 (CPU-only) or 1+ (training/inference/distributed)
- **Duration**: <1h (quick), 1-12h (medium), >12h (long)
- **Buffering**: debug/new code -> line buffer; production training -> block buffer
- **GPUs total**: determines partition (Delta) or node count

### Phase 2: Check Resources & Pick Tier

Execute checks IN ORDER and use the first tier that fits.

#### Tier 1: Login Node (Maple Only)

**Conditions**: CPU<32, GPU<=1, estimated time < 1 hour, Maple cluster.

```bash
# Check login node GPU utilization
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,memory.free --format=csv,noheader
```

If GPU utilization <= 70% AND enough free memory -> run directly:

```bash
# Line-buffered (debug / new code)
python -u your_script.py [args]

# Background on login node
nohup python -u your_script.py [args] > output.log 2>&1 &
```

**IMPORTANT**: Login node is shared. Don't run >1h, >32 CPU. Check `nvidia-smi` first.

#### Tier 2: SSH to Existing Allocation

```bash
# Maple:
squeue -u $USER -p maple -o "%.10i %.12j %.8T %.6C %.7m %.20R %.10l" -h
# Delta:
squeue -u $USER -A bcpt-delta-gpu -o "%.10i %.12j %.8T %.6C %.7m %.20R %.10l" -h
```

If you have a RUNNING job, SSH to its node and check resources:

```bash
ssh <node> "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free --format=csv,noheader; echo '---'; nproc; echo '---'; uptime"
```

If enough free GPU memory and CPU -> piggyback:

**Maple:**
```bash
ssh <node> "cd $(pwd) && \
  source ~/.bashrc_maple 2>/dev/null && \
  conda activate rl4 && \
  export CUDA_VISIBLE_DEVICES=0 && \
  python -u your_script.py [args]"
```

**Delta:**
```bash
ssh <node> "cd $(pwd) && \
  conda activate DistNNP && \
  python -u your_script.py [args]"
```

For background tasks (survives SSH disconnect):
```bash
ssh <node> "cd $(pwd) && \
  $CONDA_INIT && \
  nohup python -u your_script.py [args] > logs/output.log 2>&1 &"
```

#### Tier 3: salloc (Interactive Allocation)

When no existing node is available or resources are insufficient.

**Maple:**
```bash
salloc --partition=maple --account=dpp \
  --nodes=1 --cpus-per-task=72 \
  --gres=gpu:1 --mem=0 \
  --time=<TIME> --no-shell
```

**Delta x4 (1-4 GPUs):**
```bash
salloc --partition=gpuA100x4 --account=bcpt-delta-gpu \
  --nodes=1 --ntasks-per-node=4 --gpus-per-task=1 \
  --mem=120g --time=<TIME> --no-shell
```

**Delta x8 (5-8 GPUs):**
```bash
salloc --partition=gpuA100x8 --account=bcpt-delta-gpu \
  --nodes=1 --ntasks-per-node=1 --gpus-per-task=8 --cpus-per-task=64 \
  --mem=180g --time=<TIME> --no-shell
```

Then find and use the allocated node:

```bash
# Maple:
squeue -u $USER -p maple -h -o "%i %N %T" | grep RUNNING
# Delta:
squeue -u $USER -A bcpt-delta-gpu -h -o "%i %N %T" | grep RUNNING

# SSH and run
ssh <node> "cd $(pwd) && $CONDA_INIT && python -u your_script.py [args]"

# Release when done
scancel <jobid>
```

#### Tier 4: sbatch (Batch Submission)

For multi-node jobs, long unattended runs (>12h), or array jobs.

**Maple sbatch**: See `cluster-guide/maple.md` for templates.

**Delta sbatch**: Write a script with the full NCCL/CXI configuration from `cluster-guide/delta-nccl.md`.

**Delta partition selection:**

| GPUs Needed | Partition | SBATCH Config |
|-------------|-----------|---------------|
| 1-4 | `gpuA100x4` | `-N 1 --ntasks-per-node=<N> --gpus-per-task=1` |
| 5-8 | `gpuA100x8` | `-N 1 --ntasks-per-node=1 --gpus-per-task=<N> --cpus-per-task=64` |
| 9-16 | `gpuA100x4` multi-node | `-N <ceil(N/4)> --ntasks-per-node=4 --gpus-per-task=1` |

**Delta single-node x4 sbatch template:**

```bash
#!/bin/bash
#SBATCH --job-name=<NAME>
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH -A bcpt-delta-gpu
#SBATCH -t <TIME>
#SBATCH --partition=gpuA100x4
#SBATCH --mem=120g
#SBATCH -o logs/<NAME>_%j.out

mkdir -p logs

# === Delta NCCL/CXI Configuration (MANDATORY) ===
ml cudatoolkit
ml libfabric
ml aws-ofi-nccl

export FI_PROVIDER=cxi
export CXI_FORK_SAFE=0
export CXI_FORK_SAFE_HP=0
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_EAGER_SIZE=0
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DEFAULT_CQ_SIZE=131072

export NCCL_NET_GDR_LEVEL=SYS
export NCCL_NET_GDR_READ=0
export NCCL_CROSS_NIC=1
export NCCL_NET="AWS Libfabric"
export NCCL_PROTO=^LL128

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
# === End NCCL/CXI Configuration ===

conda activate DistNNP
cd /path/to/project

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"

srun --cpu_bind=cores --gpu-bind=none \
     --output=logs/<NAME>_%j_rank%t.out \
     --error=logs/<NAME>_%j_rank%t.err \
     python your_script.py [args]

echo "Done at $(date)"
```

**Maple sbatch template:**

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
python your_script.py [args]
echo "Done at $(date)"
```

## Buffering Rules

| Scenario | Buffering | How |
|----------|-----------|-----|
| Debug / testing new code | **Line buffer** | `python -u` or `export PYTHONUNBUFFERED=1` |
| Interactive development | **Line buffer** | Same as above |
| Short runs (<1h) | **Line buffer** | Same as above |
| Production training (>12h) | **Block buffer** (default) | Don't set PYTHONUNBUFFERED |
| sbatch jobs | **Block buffer** (default) | SLURM default is fine |

To force line buffer in sbatch:
```bash
export PYTHONUNBUFFERED=1
# Or: stdbuf -oL python your_script.py [args]
```

## Conda Environment Selection

| Cluster | Environment | Activation | Use Case |
|---------|-------------|-----------|----------|
| Maple | `rl4` | `source ~/.bashrc_maple 2>/dev/null && conda activate rl4` | Default: RL, training |
| Maple | `nvmolkit` | `source ~/.bashrc_maple 2>/dev/null && conda activate nvmolkit` | GPU molecular toolkit |
| Delta | `DistNNP` | `conda activate DistNNP` | Default: distributed NNP |

## Post-Run

After launching, report to the user:
1. Which tier was used and why
2. Node name and job ID (if applicable)
3. How to monitor: `tail -f <logfile>` or `ssh <node> nvidia-smi`
4. How to stop: `kill <PID>` / `scancel <jobid>` / `Ctrl+C`
5. **Delta reminder**: Confirm NCCL modules were loaded if using distributed training
