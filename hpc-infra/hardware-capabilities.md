# Hardware Capabilities Quick Reference

## NVIDIA GPU Generations

### A100 (Ampere, 2020)
```
Compute:  FP32 19.5 TFLOPS, TF32 156 TFLOPS, FP16 312 TFLOPS
Memory:   40 GB or 80 GB HBM2e, 2039 GB/s bandwidth
NVLink:   3.0, 12 links, 600 GB/s total
PCIe:     Gen 4 x16 (31.5 GB/s)
Features: TF32 tensor cores, sparsity (2:4), MIG (up to 7 instances)
L2 Cache: 40 MB
SM count: 108
```

### H100 (Hopper, 2022)
```
Compute:  FP32 67 TFLOPS, TF32 989 TFLOPS, FP16 1979 TFLOPS
Memory:   80 GB HBM3, 3350 GB/s bandwidth
NVLink:   4.0, 18 links, 900 GB/s total
PCIe:     Gen 5 x16 (64 GB/s)
Features: FP8 tensor cores, transformer engine, DPX (dynamic programming)
L2 Cache: 50 MB
SM count: 132
Notable:  ~3.4× A100 in FP32, ~6× in FP16 tensor
```

### GH200 (Grace Hopper Superchip, 2023)
```
GPU (Hopper):
  Same as H100 GPU die
  Compute: FP32 ~60 TFLOPS, TF32 ~989 TFLOPS
  Memory:  96 GB HBM3, ~4000 GB/s bandwidth

CPU (Grace, ARM):
  72 Neoverse V2 cores (aarch64)
  Up to 512 GB LPDDR5X, ~500 GB/s bandwidth

Interconnect:
  NVLink-C2C: 900 GB/s CPU↔GPU (not PCIe!)
  Coherent unified memory (CPU can access GPU memory and vice versa)

Key difference: No PCIe bottleneck. CPU↔GPU is NVLink speed.
This changes optimization strategies — less need to minimize transfers.
```

### B100/B200 (Blackwell, 2024-2025)
```
Compute:  FP32 ~80 TFLOPS, FP4 ~18000 TFLOPS (2nd gen transformer engine)
Memory:   192 GB HBM3e, ~8000 GB/s bandwidth
NVLink:   5.0, 1.8 TB/s total
Features: FP4 tensor cores, 2nd gen transformer engine, confidential compute
Notable:  NVLink Switch for 576-GPU domains, decompression engine
```

## Network Interface Cards (NICs)

### Mellanox/NVIDIA ConnectX Series (InfiniBand / RoCE)
```
ConnectX-6:  HDR (200 Gbps), PCIe Gen4
ConnectX-6 Dx: 200 Gbps Ethernet/RoCE, no IB
ConnectX-7:  NDR (400 Gbps), PCIe Gen5
  Features: Crypto offload, in-NIC computing
  NOTE: Adaptive routing and congestion control are SOFTWARE (managed by subnet manager)

BlueField DPU:
  ConnectX + ARM cores
  Can offload networking to DPU, freeing host CPU
```

### HPE Slingshot (Cassini NIC)
```
Slingshot-10: 100 Gbps, based on modified Ethernet
Slingshot-11: 200 Gbps, Cassini ASIC
  Features IMPLEMENTED IN HARDWARE:
    ✓ Adaptive routing (packet-level, hardware-managed)
    ✓ Congestion control (hardware-level, not ECN-based)
    ✓ Multi-rail (hardware-native load balancing)
    ✓ Traffic class isolation (8 classes, hardware-enforced)

  Uses libfabric (CXI provider), NOT libibverbs
  NCCL plugin: aws-ofi-nccl (bridges NCCL → libfabric)

  Systems: Frontier, LUMI, Perlmutter, Alps

  CRITICAL: Do NOT port IB software optimizations (adaptive routing,
  congestion control) — Cassini does these in hardware!
```

### Intel/Cornelis OPX (Omni-Path Express)
```
Omni-Path: 100 Gbps (Gen1, discontinued by Intel)
OPX (Cornelis): 400 Gbps
  Uses libfabric (OPX provider)
  Limited GPU Direct support
  Primarily on older HPC systems
```

## NVSwitch

```
NVSwitch v1 (V100 DGX-2):  300 GB/s per GPU port
NVSwitch v2 (A100 DGX):    600 GB/s per GPU port
NVSwitch v3 (H100 DGX):    900 GB/s per GPU port
NVSwitch v4 (B200):        1.8 TB/s per GPU port, 576-GPU domain

What NVSwitch does:
  - Provides full bisection bandwidth between all GPUs in a node
  - Any GPU can talk to any other at full bandwidth simultaneously
  - NCCL auto-detects and uses NVSwitch for intra-node AllReduce

Without NVSwitch (point-to-point NVLink):
  - Each GPU has NVLink to specific neighbors only
  - Ring topology is common (A→B→C→D→A)
  - Cross-ring communication goes through intermediate GPUs
```

## Common System Configurations

| System | GPUs | NIC:GPU | Network | NVLink | Notes |
|--------|------|---------|---------|--------|-------|
| DGX A100 | 8× A100 | 1:1 | 8× HDR IB | NVSwitch | Gold standard |
| DGX H100 | 8× H100 | 1:1 | 8× NDR IB | NVSwitch | Compute powerhouse |
| NCSA Delta | 4× A100 | **1:4** | 1× HDR IB | NVLink pairs | RDMA contention! |
| Alps (CSCS) | 1× GH200 | 1:1 | 4× SS-11 | C2C only | 1 GPU/node |
| Frontier | 4× MI250X | 1:1 | 4× SS-11 | Infinity Fabric | AMD, 8 GCDs |
| Perlmutter | 1× A100 | 1:1 | 4× SS-11 | None | 1 GPU/node |
| Polaris (ALCF) | 4× A100 | 1:2 | 2× HDR IB | NVLink pairs | Partial contention |
| Summit | 6× V100 | 1:3 | 2× EDR IB | NVLink 2.0 | Legacy |

## Query Commands Summary

```bash
# Full hardware survey (run on compute node!)
echo "=== GPU ==="
nvidia-smi -L
nvidia-smi topo -m

echo "=== NVLink ==="
nvidia-smi nvlink --status 2>/dev/null || echo "No NVLink"

echo "=== Network ==="
ibstat 2>/dev/null || echo "No IB"
fi_info -p cxi 2>/dev/null || echo "No Slingshot"

echo "=== NIC:GPU Ratio ==="
GPU_COUNT=$(nvidia-smi -L | wc -l)
NIC_COUNT=$(ls /sys/class/infiniband/ 2>/dev/null | wc -l)
echo "GPUs: $GPU_COUNT, NICs: $NIC_COUNT, Ratio: 1:$((GPU_COUNT/NIC_COUNT))"

echo "=== NUMA ==="
numactl -H

echo "=== PCIe ==="
lspci | grep -i -c nvidia
lspci | grep -i -c mellanox

echo "=== Drivers ==="
nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1
ofed_info -s 2>/dev/null || echo "No OFED"

echo "=== GPU Direct RDMA ==="
lsmod | grep -E "nvidia_peermem|nv_peer_mem" || echo "NOT loaded!"
```
