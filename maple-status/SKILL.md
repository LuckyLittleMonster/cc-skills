---
name: maple-status
description: Check maple cluster resource status - login node GPU, running jobs, allocated nodes, queue state. Use before running tasks, or when user asks about cluster/job status.
argument-hint: "[job-id or empty for full status]"
allowed-tools: Bash, Read, Grep, Glob
---

# Maple Cluster Status Check

Perform a comprehensive resource check on the maple GH200 cluster.

## Full Status Check (default)

Run ALL of the following checks and present a clear summary:

### 1. Login Node GPU Status

```bash
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,memory.free --format=csv,noheader
```

Report: GPU utilization %, memory used/total, whether login node is available for quick tasks (≤70% GPU utilization).

### 2. My Running Jobs

```bash
squeue -u $USER -p maple -o "%.10i %.16j %.8T %.10M %.6C %.7m %.20R %.12l" --sort=-t
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
squeue -u $USER -p maple -t PENDING -o "%.10i %.16j %.8T %.30R"
```

Common REASON codes:
- `Resources` — waiting for free nodes
- `Priority` — lower priority, queued
- `QOSMaxGRESPerUser` — hit GPU limit (3 for part_maple, 6 for maple_reserve)
- `QOSMaxMemoryPerUser` — memory limit hit (likely `--mem=0` in some job)
- `QOSMaxCpuPerUserLimit` — hit CPU limit (216 for part_maple)

### 4. QOS Usage

```bash
RUNNING_GPUS=$(squeue -u $USER -p maple -t RUNNING -h -o "%b" | grep -c "gpu")
echo "GPU jobs running: $RUNNING_GPUS / 3 (part_maple default QOS limit)"
echo "Tip: Use --qos=part_maple_night for no explicit GPU limit"
echo "Tip: Use --qos=maple_reserve for up to 6 GPUs"
```

### 5. Partition Overview

```bash
sinfo -p maple -o "%P %a %D %T %N %C %G" --noheader
```

### 6. Background Processes on Nodes (SSH piggyback)

**IMPORTANT**: Only check nodes that appear in step 2's squeue output as RUNNING with a real node name.
Do NOT hardcode node names. If there are no running jobs, SKIP this section entirely.

For each node from step 2 that has a RUNNING job assigned to it:

```bash
# Extract node names from squeue RUNNING jobs (dynamic, not hardcoded)
MY_NODES=$(squeue -u $USER -p maple -t RUNNING -h -o "%N" | sort -u)

# For each node, check for python processes
for node in $MY_NODES; do
  echo "=== $node ==="
  ssh $node "ps aux --sort=-%mem | grep python | grep -v grep | head -10" 2>/dev/null
done
```

If `MY_NODES` is empty (no running jobs), report "No allocated nodes — skipping piggyback check."

## Summary Format

Present results as a clear status dashboard:

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
ls -t Experiments/logs/*_${ARGUMENTS}*.out 2>/dev/null | head -1 | xargs tail -50
```

## Node-Specific Query

If $ARGUMENTS looks like a node name (e.g., `maple-n05`):

```bash
# What's running on this node
squeue -w $ARGUMENTS -o "%.10i %.10u %.16j %.8T %.10M"

# GPU status
ssh $ARGUMENTS nvidia-smi 2>/dev/null

# Process list
ssh $ARGUMENTS "ps aux --sort=-%mem | grep python | grep -v grep | head -10" 2>/dev/null
```
