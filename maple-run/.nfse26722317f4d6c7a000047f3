---
name: maple-run
description: Smart execution dispatcher for maple GH200 nodes. Evaluates task requirements (CPU, GPU, duration) and picks the most efficient execution method: login node → SSH existing allocation → salloc → sbatch. Use whenever running GPU/CPU tasks on the maple cluster.
argument-hint: "[task description or command]"
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
---

# Maple GH200 Smart Task Runner

Run tasks on maple GH200 nodes using the most efficient method available.

## Decision Flowchart

```
Task requirements assessment
         │
         ▼
  CPU<32 & GPU≤1 & time<1h?
     ├── YES ──► Check login node GPU
     │              ├── usage ≤70% → RUN ON LOGIN NODE (Tier 1)
     │              └── usage >70% → fall through ▼
     ▼
  Any existing salloc/sbatch allocation?
     ├── YES ──► SSH check: enough free CPU/GPU?
     │              ├── YES → SSH + RUN ON EXISTING NODE (Tier 2)
     │              └── NO  → fall through ▼
     ▼
  Need ≤1 node?
     ├── YES ──► salloc interactive allocation (Tier 3)
     ▼
  Need full node(s) or long unattended run?
     └──────────► sbatch (Tier 4)
```

## Step-by-Step Execution

### Phase 1: Assess Task Requirements

Evaluate the command/task to estimate:
- **CPU cores needed**: data loading workers, preprocessing → typically 4-16
- **GPU needed**: 0 (CPU-only) or 1 (training/inference)
- **Duration**: <1h (quick), 1-12h (medium), >12h (long)
- **Buffering**: debug/new code → line buffer; production training → block buffer

### Phase 2: Check Resources & Pick Tier

Execute these checks IN ORDER and use the first tier that fits:

#### Tier 1: Login Node (Direct)

**Conditions**: CPU<32, GPU≤1, estimated time < 1 hour.

```bash
# Check login node GPU utilization
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,memory.free --format=csv,noheader
```

If GPU utilization ≤ 70% AND enough free memory → run directly on login node.

**How to run**:
```bash
# Line-buffered (debug / new code)
python -u your_script.py [args]

# Or for background tasks on login node
nohup python -u your_script.py [args] > output.log 2>&1 &
```

**IMPORTANT**: Login node is shared. Be courteous:
- Don't run jobs longer than ~1 hour
- Don't use more than 32 CPU cores
- Monitor GPU memory; leave headroom for others

#### Tier 2: SSH to Existing Allocation

```bash
# Check existing allocations
squeue -u $USER -p maple -o "%.10i %.12j %.8T %.6C %.7m %.20R %.10l" -h
```

If you have a RUNNING job, SSH to its node and check resources:

```bash
# Get node name from squeue output, then:
ssh <node> "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free --format=csv,noheader; echo '---'; nproc; echo '---'; uptime"
```

If there's enough free GPU memory and CPU → piggyback:

```bash
ssh <node> "cd $(pwd) && \
  source ~/.bashrc_maple 2>/dev/null && \
  conda activate rl4 && \
  export CUDA_VISIBLE_DEVICES=0 && \
  python -u your_script.py [args]"
```

For background tasks (survives SSH disconnect):
```bash
ssh <node> "cd $(pwd) && \
  source ~/.bashrc_maple 2>/dev/null && \
  conda activate rl4 && \
  export CUDA_VISIBLE_DEVICES=0 && \
  nohup python -u your_script.py [args] > logs/output.log 2>&1 &"
```

#### Tier 3: salloc (Interactive Allocation)

When no existing node is available or resources are insufficient:

```bash
# Request interactive allocation — claim the ENTIRE node (72 CPU, all memory, 1 GPU)
salloc --partition=maple --account=dpp \
  --nodes=1 --cpus-per-task=72 \
  --gres=gpu:1 --mem=0 \
  --time=<TIME> --no-shell
```

**NOTE**: salloc claims the full node (`--cpus-per-task=72 --mem=0`) so you have
exclusive access to all 72 CPU cores, all memory, and the GPU. This prevents
resource contention with other users' jobs on the same node.

Then find and use the allocated node:

```bash
# Get the node name
squeue -u $USER -p maple -h -o "%i %N %T" | grep RUNNING

# SSH and run
ssh <node> "cd $(pwd) && \
  source ~/.bashrc_maple 2>/dev/null && \
  conda activate rl4 && \
  export CUDA_VISIBLE_DEVICES=0 && \
  python -u your_script.py [args]"
```

When done, release the allocation:
```bash
scancel <jobid>
```

**salloc tips**:
- Use `--time` wisely — allocation holds resources even when idle
- `--no-shell` prevents salloc from opening an interactive shell, letting you script around it
- If `--no-shell` fails on this SLURM version, use: `salloc ... bash -c "sleep infinity"` in background, then SSH

#### Tier 4: sbatch (Batch Submission)

Only for:
- Multi-node jobs (DDP training, ≥2 GPUs)
- Long unattended production runs (>12h)
- Array jobs (systematic experiment matrices)

Write a script and submit:

```bash
sbatch your_script.sh
```

See `/maple-guide` for sbatch template and best practices.

## Buffering Rules

| Scenario | Buffering | How |
|----------|-----------|-----|
| Debug / testing new code | **Line buffer** | `python -u` or `export PYTHONUNBUFFERED=1` |
| Interactive development | **Line buffer** | Same as above |
| Short runs (<1h) | **Line buffer** | Same as above |
| Production training (>12h) | **Block buffer** (default) | Don't set PYTHONUNBUFFERED |
| sbatch jobs | **Block buffer** (default) | SLURM default is fine |

Line buffer rationale: You can see output in real-time for debugging.
Block buffer rationale: Higher I/O throughput for long training runs; logs are flushed periodically anyway.

To explicitly force line buffer in sbatch if needed:
```bash
# In the sbatch script, add before the python command:
export PYTHONUNBUFFERED=1
# Or use stdbuf:
stdbuf -oL python your_script.py [args]
```

## Conda Environment Selection

| Environment | Activation | Use Case |
|-------------|-----------|----------|
| `rl4` | `conda activate rl4` | Default: RL, link predictor, T5v2, most training |
| `nvmolkit` | `conda activate nvmolkit` | GPU molecular toolkit (kNN, fingerprints, Tanimoto) |

Always prefix with: `source ~/.bashrc_maple 2>/dev/null`

## Resource Limits & QOS Strategy

| QOS | GPU Limit | When to Use |
|-----|-----------|-------------|
| `part_maple` (default) | **3 GPUs/user** | Normal jobs, up to 3 concurrent |
| `part_maple_night` | **No explicit limit** | Burst experiments, filling idle nodes |
| `maple_reserve` | **6 GPUs/user** | Reserved access, large-scale runs |

- **Login node**: Shared — limit yourself to <32 CPU, ≤1 GPU, <1h
- **Memory**: salloc uses `--mem=0` (claim entire node). sbatch uses `--mem=64G` if running multiple concurrent jobs
- **Burst strategy**: When idle nodes available, use `--qos=part_maple_night` to submit beyond the 3-GPU default limit

To use alternative QOS in sbatch/salloc, add: `--qos=part_maple_night`

## Post-Run

After launching, report to the user:
1. Which tier was used and why
2. Node name and job ID (if applicable)
3. How to monitor: `tail -f <logfile>` or `ssh <node> nvidia-smi`
4. How to stop: `kill <PID>` / `scancel <jobid>` / `Ctrl+C`
