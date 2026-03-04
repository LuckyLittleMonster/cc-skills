---
name: cluster-status
description: Check cluster resource status - login node GPU, running jobs, allocated nodes, queue state. Auto-detects Maple or Delta. Use before running tasks, or when user asks about cluster/job status.
argument-hint: "[job-id or empty for full status]"
allowed-tools: Bash, Read, Grep, Glob
---

# Cluster Status Check

Perform a comprehensive resource check on the detected cluster.

## Step 0: Detect Cluster

```bash
_HOSTNAME=$(hostname -f 2>/dev/null || hostname)
if [[ "$_HOSTNAME" == *delta* || "$_HOSTNAME" == dt-* || "$_HOSTNAME" == gpua* || "$_HOSTNAME" == gpuc* || "$_HOSTNAME" == gpue* ]]; then
    CLUSTER="delta"
    PARTITIONS="gpuA100x4,gpuA100x8"
    ACCOUNT="bcpt-delta-gpu"
    HAS_LOGIN_GPU=false
elif [[ "$_HOSTNAME" == *maple* ]]; then
    CLUSTER="maple"
    PARTITIONS="maple"
    ACCOUNT="dpp"
    HAS_LOGIN_GPU=true
elif [[ "$(uname -m)" == "aarch64" ]]; then
    CLUSTER="maple"
    PARTITIONS="maple"
    ACCOUNT="dpp"
    HAS_LOGIN_GPU=true
else
    CLUSTER="unknown"
fi
```

Report detected cluster. If unknown, ask user.

## Full Status Check (default)

Run ALL of the following checks and present a clear summary.

### 1. Login Node GPU Status

**Maple only** — Delta has no login GPU, report "No login GPU on Delta".

```bash
# Maple only:
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,memory.free --format=csv,noheader
```

Report: GPU utilization %, memory used/total, whether login node is available for quick tasks (<=70% GPU utilization).

### 2. My Running Jobs

```bash
# Maple:
squeue -u $USER -p maple -o "%.10i %.16j %.8T %.10M %.6C %.7m %.20R %.12l" --sort=-t

# Delta:
squeue -u $USER -A bcpt-delta-gpu -o "%.10i %.16j %.8T %.10M %.6C %.7m %.20R %.12l" --sort=-t
```

Columns: JobID, Name, State, Elapsed, CPUs, Memory, Node, TimeLimit.

**Time format reminder**:
- `MM:SS` = minutes:seconds (NOT hours:minutes)
- `H:MM:SS` = hours:min:sec
- `D-HH:MM:SS` = days-hours:min:sec

For each RUNNING job, also check GPU usage on its node:

```bash
ssh <node> "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free --format=csv,noheader" 2>/dev/null
```

### 3. Pending Jobs

```bash
# Maple:
squeue -u $USER -p maple -t PENDING -o "%.10i %.16j %.8T %.30R"

# Delta:
squeue -u $USER -A bcpt-delta-gpu -t PENDING -o "%.10i %.16j %.8T %.30R"
```

Common REASON codes:
- `Resources` — waiting for free nodes
- `Priority` — lower priority, queued
- `QOSMaxGRESPerUser` — hit GPU limit
- `QOSMaxMemoryPerUser` — memory limit hit (likely `--mem=0` in some job)
- `QOSMaxCpuPerUserLimit` — CPU limit hit

### 4. QOS/Resource Usage

**Maple:**
```bash
RUNNING_GPUS=$(squeue -u $USER -p maple -t RUNNING -h -o "%b" | grep -c "gpu")
echo "GPU jobs running: $RUNNING_GPUS / 3 (part_maple default QOS limit)"
echo "Tip: Use --qos=part_maple_night for no explicit GPU limit"
echo "Tip: Use --qos=maple_reserve for up to 6 GPUs"
```

**Delta:**
```bash
RUNNING_X4=$(squeue -u $USER -A bcpt-delta-gpu -p gpuA100x4 -t RUNNING -h | wc -l)
RUNNING_X8=$(squeue -u $USER -A bcpt-delta-gpu -p gpuA100x8 -t RUNNING -h | wc -l)
echo "Running jobs: x4=$RUNNING_X4, x8=$RUNNING_X8"
```

### 5. Partition Overview

```bash
# Maple:
sinfo -p maple -o "%P %a %D %T %N %C %G" --noheader

# Delta:
sinfo -p gpuA100x4,gpuA100x8 -o "%P %a %D %T %N %C %G" --noheader
```

### 6. Background Processes on Nodes (SSH piggyback)

**IMPORTANT**: Only check nodes that appear in step 2's squeue output as RUNNING with a real node name.
Do NOT hardcode node names. If there are no running jobs, SKIP this section entirely.

```bash
# Extract node names from squeue RUNNING jobs (dynamic, not hardcoded)
# Maple:
MY_NODES=$(squeue -u $USER -p $PARTITIONS -t RUNNING -h -o "%N" | sort -u)
# Delta:
MY_NODES=$(squeue -u $USER -A $ACCOUNT -t RUNNING -h -o "%N" | sort -u)

# For each node, check for python processes
for node in $MY_NODES; do
  echo "=== $node ==="
  ssh $node "ps aux --sort=-%mem | grep python | grep -v grep | head -10" 2>/dev/null
done
```

If `MY_NODES` is empty (no running jobs), report "No allocated nodes -- skipping piggyback check."

## Summary Format

### Maple Summary

```
=== Maple GH200 Status ===

Login Node:
  GPU: 45% utilization, 18GB/96GB used, 78GB free
  Status: AVAILABLE for quick tasks

My Allocations (1/3 GPU slots used, default QOS):
  JobID 12345 | v3_full_A | RUNNING 2:34:15 | maple-n05 | GPU: 62%, 24GB/96GB

Piggybacked Processes (only nodes with running jobs):
  maple-n05: python train_v3_full_D.py (PID 28831, 8.2GB GPU)

Pending: none

Recommendation: 2 GPU slots available (default QOS). Or use --qos=part_maple_night for more.
```

### Delta Summary

```
=== Delta A100 Status ===

Login Node: No login GPU

My Allocations:
  JobID 15751150 | dist_2n8g | RUNNING 5:23 | gpua042,gpua043 | x4 partition
  JobID 15751200 | uma_bench  | PENDING (Resources)

Piggybacked Processes:
  gpua042: python dist_md.py (PID 12345)

Partition Status:
  gpuA100x4: 12 idle / 99 total
  gpuA100x8: 2 idle / 6 total

Recommendation: x4 nodes available. Remember to load NCCL modules (ml cudatoolkit && ml libfabric && ml aws-ofi-nccl).
```

**KEY RULE**: The node list in "Piggybacked Processes" MUST come from the squeue query in step 2.
If the user has 0 running jobs, this section should say "No allocated nodes" and NOT list any nodes.

## Specific Job Query

If $ARGUMENTS contains a job ID, focus on that job:

```bash
# Job details
sacct -j $ARGUMENTS --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,Start,End,NodeList

# If running, check its node
squeue -j $ARGUMENTS -h -o "%N %T"
```

Then tail its logs:
```bash
# Maple:
ls -t Experiments/logs/*_${ARGUMENTS}*.out 2>/dev/null | head -1 | xargs tail -50

# Delta:
ls -t logs/*_${ARGUMENTS}*.out prof_logs/*_${ARGUMENTS}*.out 2>/dev/null | head -1 | xargs tail -50
```

## Node-Specific Query

If $ARGUMENTS looks like a node name (e.g., `maple-n05`, `gpua042`):

```bash
# What's running on this node
squeue -w $ARGUMENTS -o "%.10i %.10u %.16j %.8T %.10M"

# GPU status
ssh $ARGUMENTS nvidia-smi 2>/dev/null

# Process list
ssh $ARGUMENTS "ps aux --sort=-%mem | grep python | grep -v grep | head -10" 2>/dev/null
```
