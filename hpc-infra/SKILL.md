---
name: hpc-infra
description: HPC infrastructure — NCCL, network fabrics (IB/Slingshot/RoCE), GPU topology, hardware capabilities, hardware info caching. Use when debugging communication issues, checking hardware, or working with multi-node systems.
user-invocable: true
---

# HPC Infrastructure

Network communication, GPU topology, hardware capabilities, and hardware info caching.

## Extension Files

| File | Content |
|------|---------|
| `nccl-comms.md` | AllReduce algorithms, NCCL env vars, error diagnosis, GPU Direct RDMA checklist |
| `network-fabrics.md` | IB/Slingshot/RoCE capability matrix, fabric detection, UCCL lesson |
| `gpu-topology.md` | Topology patterns (DGX, shared-NIC, GH200), NIC:GPU analysis, Delta lesson |
| `hardware-capabilities.md` | A100/H100/GH200/B200 specs, NIC generations, NVSwitch, system configs |

## Hardware Info Cache

Hardware info is cached in `.cache/hw_info/` to avoid repeated queries. **Always check cache first.**

```bash
HW_CACHE=".cache/hw_info"

if [ -d "$HW_CACHE" ] && [ -f "$HW_CACHE/hostname.txt" ] && \
   find "$HW_CACHE/hostname.txt" -mtime -7 -print -quit | grep -q .; then
    echo "=== Using cached hardware info ==="
    cat "$HW_CACHE/hostname.txt"
    cat "$HW_CACHE/gpu_topo.txt"
    cat "$HW_CACHE/nic_info.txt"
    cat "$HW_CACHE/nic_gpu_ratio.txt"
    cat "$HW_CACHE/numa.txt"
else
    echo "=== Querying and caching hardware info ==="
    mkdir -p "$HW_CACHE"
    { hostname; echo "SLURM_JOB_ID=${SLURM_JOB_ID:-not_in_slurm}"; } > "$HW_CACHE/hostname.txt"
    nvidia-smi topo -m 2>/dev/null > "$HW_CACHE/gpu_topo.txt"
    nvidia-smi -q 2>/dev/null > "$HW_CACHE/gpu_details.txt"
    { ibstat 2>/dev/null | head -20 || fi_info -p cxi 2>/dev/null | head -10 || echo "No IB/Slingshot"; } > "$HW_CACHE/nic_info.txt"
    echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l), NICs: $(ls /sys/class/infiniband/ 2>/dev/null | wc -l)" > "$HW_CACHE/nic_gpu_ratio.txt"
    numactl -H 2>/dev/null > "$HW_CACHE/numa.txt"
    lspci -tv 2>/dev/null | grep -A2 -i "nvidia\|mellanox" > "$HW_CACHE/pcie_topo.txt"
fi
```

**Cache conventions:**
- Location: `.cache/hw_info/` (local), `.cache/hw_info/<nodename>/` (compute nodes)
- Expiry: 7 days. Force refresh: `rm -rf .cache/hw_info/`

**SLURM-aware (login ≠ compute):**
```bash
NODE=$(squeue -u $USER -h -o %N | head -1)
HW_CACHE=".cache/hw_info/$NODE"
if [ ! -d "$HW_CACHE" ] || ! find "$HW_CACHE/gpu_topo.txt" -mtime -7 2>/dev/null | grep -q .; then
    mkdir -p "$HW_CACHE"
    ssh $NODE "nvidia-smi topo -m" > "$HW_CACHE/gpu_topo.txt"
    ssh $NODE "ibstat 2>/dev/null || fi_info -p cxi 2>/dev/null" > "$HW_CACHE/nic_info.txt"
    ssh $NODE "numactl -H" > "$HW_CACHE/numa.txt"
fi
```

## Key Lessons (Real Failures)

**UCCL/Slingshot:** UCCL optimized adaptive routing + congestion control for IB. Slingshot Cassini NIC implements BOTH in hardware. Porting was unnecessary. → Always check fabric capability matrix before porting.

**NCSA Delta 4xA100:1NIC:** GPU Direct RDMA contention (4 GPUs sharing 1 NIC). Symptom: NCCL timeouts. Root cause: hardware topology, not NCCL software. → Always check NIC:GPU ratio before debugging NCCL.

## Review Checklist (Infrastructure)

```
□ Hardware info cached and current? (.cache/hw_info/)
□ NIC:GPU ratio adequate? (< 1:1 = contention risk)
□ GPU Direct RDMA working? (nvidia-peermem loaded?)
□ Network fabric identified? (IB vs Slingshot vs RoCE)
□ Hardware already implements needed optimization?
□ NCCL env vars tuned for this fabric?
□ Login node vs compute node hardware verified?
```
