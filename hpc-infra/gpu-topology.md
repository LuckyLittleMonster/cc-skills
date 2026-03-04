# GPU Topology — Patterns, NIC Sharing & PCIe Layout

## CRITICAL: Know Your Topology BEFORE Debugging

**Lesson learned (NCSA Delta incident):** 4× A100 sharing 1 NIC caused GPU Direct RDMA contention. Symptom looked like NCCL bug (timeouts, low bandwidth). Root cause was hardware topology — not software.

**Rule:** ALWAYS run `nvidia-smi topo -m` and count NICs before debugging any multi-GPU communication issue.

## Query Commands

```bash
# GPU topology matrix (MOST IMPORTANT COMMAND)
nvidia-smi topo -m
# Output key:
#   X   = self
#   SYS = cross NUMA, through CPU socket
#   NODE = same NUMA node
#   PHB = same PCIe Host Bridge
#   PIX = same PCIe switch
#   PXB = cross PCIe switch (same root complex)
#   NV# = NVLink (# = number of links)

# Count GPUs and NICs
nvidia-smi -L | wc -l                    # GPU count
ibstat 2>/dev/null | grep -c "Port 1"    # IB NIC count
ls /sys/class/infiniband/ 2>/dev/null | wc -l  # Alternative
fi_info -p cxi 2>/dev/null | grep -c "domain"  # Slingshot NIC count

# NIC:GPU ratio
echo "GPUs: $(nvidia-smi -L | wc -l), NICs: $(ls /sys/class/infiniband/ 2>/dev/null | wc -l)"

# PCIe topology (visual tree)
lspci -tv | grep -A2 -i "nvidia\|mellanox"

# NUMA layout
numactl -H
# Shows: which CPUs on which NUMA node, memory per node

# CPU-GPU-NIC affinity
nvidia-smi topo -m
# Look for NIC rows (mlx5_*, cxi*) — should be PIX/PHB with their GPU
```

## Common Topology Patterns

### Pattern A: DGX A100 / DGX H100 (Optimal)
```
8× GPU + 8× NIC + NVSwitch
├── NIC:GPU = 1:1 → full RDMA bandwidth per GPU
├── NVSwitch → all GPUs fully connected (600-900 GB/s)
├── Each NIC co-located with its GPU (PIX/PHB)
├── GPU Direct RDMA: full bandwidth, no contention
└── NCCL: NVSwitch for intra-node, IB for inter-node

nvidia-smi topo -m shows:
  GPU0-GPU7: all NV12 (12 NVLink lanes)
  mlx5_0: PIX with GPU0
  mlx5_1: PIX with GPU1
  ...
```

### Pattern B: Budget Multi-GPU (Shared NIC) — DANGER
```
4× GPU + 1× NIC (e.g., NCSA Delta, many university clusters)
├── NIC:GPU = 1:4 → RDMA bandwidth contention!
├── Per-GPU effective bandwidth: peak_nic_bw / 4
├── GPU Direct RDMA: works but saturates quickly
├── Symptom: NCCL timeouts, low allreduce bandwidth
├── NCCL reports no errors — just slow or timeout
│
├── Mitigations:
│   ├── NCCL_NET_GDR_LEVEL=PHB (restrict to local GPU-NIC pair)
│   ├── Node-local AllReduce first, then cross-node (hierarchical)
│   ├── Reduce cross-node communication (larger batch, grad accumulation)
│   └── Disable GPU Direct: NCCL_NET_GDR_LEVEL=LOC (use host staging)
│
└── DO NOT: spend time debugging NCCL code — it's a hardware limit

nvidia-smi topo -m shows:
  GPU0-GPU3: PIX or PHB (PCIe switch)
  mlx5_0: PHB with GPU0, SYS with GPU2/GPU3 ← problem!
```

### Pattern C: PCIe Switch Topology (Asymmetric)
```
4× GPU on 2× PCIe switches (2 GPU per switch)
├── GPU0-GPU1: PIX (same switch, fast P2P)
├── GPU2-GPU3: PIX (same switch, fast P2P)
├── GPU0-GPU2: PHB or PXB (cross switch, slower P2P)
│
├── Impact on pipeline parallel:
│   ├── Stage boundaries should align with PCIe switch boundaries
│   ├── GPU0→GPU1 (fast), GPU1→GPU2 (slow!)
│   └── Better: Stage0=[GPU0,GPU1], Stage1=[GPU2,GPU3]
│
└── Impact on tensor parallel:
    ├── TP groups should be within same PCIe switch
    └── Cross-switch TP = reduced bandwidth
```

### Pattern D: GH200 (Grace Hopper Superchip)
```
1× Grace CPU + 1× Hopper GPU (per node)
├── CPU-GPU: NVLink-C2C (900 GB/s, not PCIe!)
├── Unified memory architecture
├── No PCIe bottleneck for CPU↔GPU
├── But: 1 GPU per node → multi-node = always network
│
├── Key differences from x86+PCIe:
│   ├── CPU↔GPU transfer is ~14× faster than PCIe 5.0
│   ├── Unified memory with coherent access
│   ├── Data loading from CPU RAM almost as fast as GPU local
│   └── Less need for pinned memory tricks
│
└── SLURM implication: 1 GPU/node → need more nodes for same GPU count
```

## NIC:GPU Ratio Analysis

```
Ratio    Impact                           Action
──────────────────────────────────────────────────────────
1:1      Optimal. Full RDMA per GPU.      Default NCCL settings work.
1:2      50% bandwidth per GPU.           Tune NCCL_NET_GDR_LEVEL.
                                          Consider grad accumulation.
1:4      25% bandwidth. Contention.       Hierarchical AllReduce.
                                          Larger batches.
                                          May need to disable GDR.
1:8      Severe bottleneck.               Minimize cross-node comm.
                                          Use model/pipeline parallel
                                          instead of data parallel.
0:N      No RDMA NIC.                     TCP only (very slow).
         (e.g., consumer GPUs)            Single-node only practical.
```

## NVLink Topology

```bash
# Check NVLink status
nvidia-smi nvlink --status

# NVLink versions and bandwidth (bidirectional per link):
# NVLink 3.0 (A100):  50 GB/s per link, 12 links → 600 GB/s total
# NVLink 4.0 (H100):  50 GB/s per link, 18 links → 900 GB/s total
# NVLink-C2C (GH200): 900 GB/s CPU↔GPU

# Check NVSwitch (if present)
nvidia-smi nvlink --status -i 0
# All GPUs showing NV12/NV18 = NVSwitch (fully connected)
# Only some GPUs showing NV links = point-to-point NVLink
```

## Diagnostic Checklist for Multi-GPU Issues

```
□ nvidia-smi topo -m → understand interconnect type
□ Count GPUs and NICs → compute NIC:GPU ratio
□ Check NIC-GPU affinity → PIX/PHB (good) vs SYS (bad)
□ Check NVLink presence → NV# entries in topo matrix
□ Check NUMA → GPU and NIC on same NUMA node?
□ Check GPU Direct RDMA → nvidia-peermem loaded?
□ Run nccl-tests → baseline AllReduce bandwidth

Only after ALL above → investigate NCCL config / application code
```
