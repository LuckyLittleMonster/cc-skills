# Delta A100 Cluster Reference

## Hardware

| Component | A100x4 Partition | A100x8 Partition |
|-----------|-----------------|-----------------|
| GPU | 4x A100-SXM4-40GB | 8x A100-SXM4-40GB |
| NVLink | NV4 (4 links/pair) | NV12 (12 links/pair, NVSwitch) |
| CPU | 64 x86_64 cores | 128 x86_64 cores |
| Nodes | 99 nodes | 6 nodes |
| NIC | 1x Slingshot CXI | 1x Slingshot CXI |
| GPU-NIC distance | 8-9 (PHB/SYS) | 8-9 (PHB/SYS) |

## SLURM Configuration

| Parameter | Value |
|-----------|-------|
| Account | `bcpt-delta-gpu` |
| Partitions | `gpuA100x4`, `gpuA100x8` |
| Login GPU | **No** (effective starting tier is 3/4) |
| Conda | `conda activate DistNNP` |
| srun pattern | `srun --cpu_bind=cores --gpu-bind=none` |

## Partition Selection Logic

| GPUs Needed | Partition | Config |
|-------------|-----------|--------|
| 1-4 | `gpuA100x4` | 1 node, ntasks=N, gpus-per-task=1 |
| 5-8 | `gpuA100x8` | 1 node, ntasks=1, gpus-per-task=N |
| >8 | `gpuA100x4` multi-node | N nodes, ntasks-per-node=4 |

## NCCL Configuration (CRITICAL)

**ALL scripts using NCCL MUST load modules and set env vars.** Missing these causes 20-50x performance degradation.

See `delta-nccl.md` for the complete configuration. Minimum required:

```bash
ml cudatoolkit && ml libfabric && ml aws-ofi-nccl
export FI_PROVIDER=cxi
export NCCL_CROSS_NIC=1
export NCCL_NET="AWS Libfabric"
```

## salloc Patterns

### Single Node x4

```bash
salloc -p gpuA100x4 -A bcpt-delta-gpu \
  --nodes=1 --ntasks-per-node=4 --gpus-per-task=1 \
  --mem=120g -t 1:00:00 --no-shell

# Find node and SSH
squeue -u $USER -A bcpt-delta-gpu -h -o "%i %N %T" | grep RUNNING
ssh <node> "conda activate DistNNP && python -u script.py"
```

### Single Node x8

```bash
salloc -p gpuA100x8 -A bcpt-delta-gpu \
  --nodes=1 --ntasks-per-node=1 --gpus-per-task=8 --cpus-per-task=64 \
  --mem=180g -t 1:00:00 --no-shell
```

## sbatch Templates

### Single Node x4 (Most Common)

```bash
#!/bin/bash
#SBATCH --job-name=<NAME>
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH -A bcpt-delta-gpu
#SBATCH -t 0:30:00
#SBATCH --partition=gpuA100x4
#SBATCH --mem=120g
#SBATCH -o logs/<NAME>_%j.out

mkdir -p logs

# --- NCCL/CXI setup (MANDATORY) ---
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
# --- end NCCL/CXI setup ---

conda activate DistNNP
cd /path/to/project

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"

srun --cpu_bind=cores --gpu-bind=none \
     --output=logs/<NAME>_%j_rank%t.out \
     --error=logs/<NAME>_%j_rank%t.err \
     python your_script.py [args]

echo "Done at $(date)"
```

### Single Node x8

```bash
#!/bin/bash
#SBATCH --job-name=<NAME>
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=64
#SBATCH -A bcpt-delta-gpu
#SBATCH -t 0:30:00
#SBATCH --partition=gpuA100x8
#SBATCH --mem=180g
#SBATCH -o logs/<NAME>_%j.out

# ... same NCCL/CXI setup block as above ...
```

### Multi-Node (e.g., 2 nodes x 4 GPUs)

```bash
#!/bin/bash
#SBATCH --job-name=<NAME>
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH -A bcpt-delta-gpu
#SBATCH -t 0:30:00
#SBATCH --partition=gpuA100x4
#SBATCH --mem=120g
#SBATCH -o logs/<NAME>_%j.out

# ... same NCCL/CXI setup block as above ...

srun --cpu_bind=cores --gpu-bind=none \
     --output=logs/<NAME>_%j_rank%t.out \
     --error=logs/<NAME>_%j_rank%t.err \
     python your_dist_script.py [args]
```

## Gotchas

1. **Module loads are MANDATORY**: Missing `ml cudatoolkit && ml libfabric && ml aws-ofi-nccl` causes 20-50x NCCL degradation (e.g., 0.11 vs 4.96 steps/s).
2. **CXI_FORK_SAFE**: Set to `0` — MADV_DONTFORK can cause segfaults with RDMA.
3. **expandable_segments**: Must be `False` — conflicts with libfabric CXI memory registration.
4. **GDR distance 8-9**: Asymmetric GDR required (`GDR_LEVEL=SYS` + `GDR_READ=0`). Full GDR causes segfaults on ranks 2,3.
5. **No login GPU**: Cannot run GPU tasks on login node. Use salloc or sbatch.
6. **srun flags**: Always `--cpu_bind=cores --gpu-bind=none` for Ray-based scripts.
7. **LL128 protocol**: Disable with `NCCL_PROTO=^LL128` — worse on Slingshot (NCCL 2.27+).
8. **Slingshot eager buffer overflow**: Set `FI_CXI_RDZV_*=0` to prevent all_to_all timeouts on multi-node.
