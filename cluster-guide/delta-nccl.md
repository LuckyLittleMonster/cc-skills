# Delta NCCL/CXI Complete Configuration Reference

Delta uses HPE Slingshot (CXI provider) with the AWS OFI NCCL plugin. This requires specific module loads and environment variables for correct NCCL operation.

**Missing this configuration causes 20-50x performance degradation** (e.g., 0.11 vs 4.96 steps/s).

## Mandatory Module Loads

```bash
ml cudatoolkit     # CUDA runtime + driver
ml libfabric       # CXI transport for Slingshot
ml aws-ofi-nccl    # NCCL plugin for libfabric
```

These must be loaded **before any NCCL operation** (PyTorch distributed, NCCL collectives, etc.).

## CXI Provider Settings

```bash
# Select CXI as the libfabric provider (Slingshot network)
export FI_PROVIDER=cxi

# Disable fork safety — MADV_DONTFORK causes segfaults with RDMA
# See: https://ofiwg.github.io/libfabric/v2.1.0/man/fi_cxi.7.html
export CXI_FORK_SAFE=0
export CXI_FORK_SAFE_HP=0
```

## Slingshot Eager Buffer Overflow Fix

Prevents NCCL all_to_all timeouts on multi-node jobs.

```bash
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_EAGER_SIZE=0
```

## CXI Tuning

Recommended by CSCS/Alps and ALCF/Polaris documentation.

```bash
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DEFAULT_CQ_SIZE=131072
```

## NCCL Network Configuration

```bash
export NCCL_CROSS_NIC=1             # Allow cross-NIC communication
export NCCL_NET="AWS Libfabric"     # Use AWS OFI NCCL plugin
export NCCL_PROTO=^LL128            # Disable LL128 (worse on Slingshot, default in NCCL 2.27+)
```

## Asymmetric GDR (Critical for Delta)

Delta nodes have 4 GPUs sharing 1 NIC with GPU-NIC distance 8-9 for GPUs 2,3 (NUMA crossing). Full GDR causes segfaults. Asymmetric mode: RDMA write (NIC->GPU) enabled, RDMA read (GPU->NIC) uses CPU bounce.

```bash
export NCCL_NET_GDR_LEVEL=SYS      # Enable GDR at max distance
export NCCL_NET_GDR_READ=0          # Disable GDR read (CPU bounce for send)
```

| Direction | Operation | Path |
|-----------|-----------|------|
| Send (GPU->NIC) | RDMA Read | **Disabled** -> GPU -> CPU buffer -> NIC |
| Receive (NIC->GPU) | RDMA Write | **Enabled** -> NIC -> GPU (direct) |

Performance: ~10-20% send overhead vs full GDR, but stable across all ranks.

## PyTorch CUDA Allocator Fix

```bash
# Disable expandable_segments — conflicts with libfabric CXI memory registration
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
```

## Debug Settings (Enable When Troubleshooting)

```bash
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT,NET
export TORCH_NCCL_TRACE_BUFFER_SIZE=10000    # Stack traces on failures
export PYTHONFAULTHANDLER=1                   # Python crash stack traces
```

## Complete Copy-Paste Environment Block

Use this in all Delta sbatch/salloc scripts:

```bash
# === Delta NCCL/CXI Configuration ===
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
# === End Delta NCCL/CXI Configuration ===
```

## References

- NCCL Environment Variables: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
- AWS OFI NCCL Plugin: https://github.com/aws/aws-ofi-nccl
- CSCS NCCL on Alps: https://docs.cscs.ch/software/communication/nccl/
- ALCF Polaris NCCL: https://docs.alcf.anl.gov/polaris/applications-and-libraries/libraries/nccl/
- Delta GDR analysis: `docs/delta_gdr_configuration.md` in DistNNP repo
