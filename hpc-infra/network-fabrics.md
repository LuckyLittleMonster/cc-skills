# Network Fabrics — IB, Slingshot, RoCE

## CRITICAL: Check Hardware Capabilities BEFORE Software Optimization

**Lesson learned (UCCL incident):** UCCL developed NCCL plugin optimizations (adaptive routing, congestion control) for InfiniBand. When attempting to port to Slingshot, these optimizations were unnecessary — Slingshot's Cassini NIC implements both IN HARDWARE.

**Rule:** Before porting any network optimization, ALWAYS query the target fabric's hardware capabilities.

## Fabric Detection

```bash
# Detect fabric type
if ibstat &>/dev/null && ibstat | grep -q "State: Active"; then
    echo "InfiniBand detected"
    ibstat                     # HCA details
    ibv_devinfo               # Detailed device info
    ibv_devices               # List IB devices
elif fi_info -p cxi &>/dev/null; then
    echo "Slingshot (CXI) detected"
    fi_info -p cxi            # Fabric interface info
    slingshot-diag             # Slingshot diagnostics (if available)
elif ibstat &>/dev/null && ibstat | grep -q "RoCE"; then
    echo "RoCE detected"
    ibstat
    show_gids                  # RoCE GID table
fi

# Check NIC model
lspci | grep -i -E "mellanox|nvidia.*connect|cray|hpe"

# Check driver version
ofed_info 2>/dev/null         # OFED version (IB/RoCE)
modinfo cxi 2>/dev/null       # CXI driver version (Slingshot)
```

## Hardware Capability Matrix

```
┌─────────────────────────┬────────────────┬──────────────────┬──────────────┐
│ Feature                 │ InfiniBand     │ Slingshot-11     │ RoCE v2      │
│                         │ (ConnectX-7)   │ (Cassini NIC)    │ (ConnectX-7) │
├─────────────────────────┼────────────────┼──────────────────┼──────────────┤
│ Adaptive routing        │ SOFTWARE       │ HARDWARE ✓       │ N/A          │
│ Congestion control      │ SOFTWARE (ECN) │ HARDWARE ✓       │ ECN-based    │
│ In-network reduce       │ SHARP (HW) ✓   │ No (planned)     │ No           │
│ GPU Direct RDMA         │ Yes ✓          │ Yes (CXI) ✓      │ Yes ✓        │
│ Multi-rail              │ Software       │ HARDWARE ✓       │ Software     │
│ Traffic class isolation │ VL/SL          │ HW traffic class │ DSCP/PFC     │
│ Ordering guarantees     │ Per-QP ordered │ Relaxed ordering │ Per-QP       │
│ Max bandwidth (per NIC) │ 400 Gbps       │ 200 Gbps         │ 400 Gbps     │
│ Typical latency         │ ~1 μs          │ ~1-2 μs          │ ~2-5 μs      │
│ Lossless fabric         │ Credit-based   │ Credit-based     │ PFC required │
└─────────────────────────┴────────────────┴──────────────────┴──────────────┘

KEY IMPLICATIONS:
- IB: Need UCCL/software for adaptive routing + congestion control
- Slingshot: Do NOT port IB adaptive routing/CC — hardware already does it
- RoCE: MUST configure PFC (Priority Flow Control) or packets WILL be lost
- SHARP: Only on IB with compatible switches — gives ~2× AllReduce speedup
```

## InfiniBand Specifics

```bash
# Check link speed and state
ibstat
# Look for: Rate: 400 (NDR), 200 (HDR), 100 (EDR)
# Look for: State: Active, Physical state: LinkUp

# Check SHARP availability
sharp_hello                    # Tests SHARP daemon
ibv_devinfo | grep -i sharp   # Check device capabilities

# Check NIC-GPU affinity
nvidia-smi topo -m
# mlx5_0 should be PIX or PHB with its GPU (not SYS!)

# Subnet manager
sminfo                         # Show subnet manager info
smpquery -D portinfo 0        # Port info
```

## Slingshot Specifics

```bash
# Slingshot uses libfabric (not libibverbs)
fi_info -p cxi                 # CXI provider info
fi_pingpong -p cxi             # Basic connectivity test

# Slingshot topology
slingshot-topology              # If available

# NCCL on Slingshot uses aws-ofi-nccl plugin
# Key env vars:
FI_CXI_DISABLE_HOST_REGISTER=1  # Sometimes needed for GPU Direct
FI_CXI_DEFAULT_CQ_SIZE=131072   # Completion queue size
NCCL_NET_GDR_LEVEL=PHB          # May need tuning on Slingshot

# Slingshot systems (examples):
# - LUMI (AMD MI250X + Slingshot-11)
# - Frontier (AMD MI250X + Slingshot-11)
# - Perlmutter (A100 + Slingshot-11)
# - Alps (GH200 + Slingshot-11)
```

## RoCE v2 Specifics

```bash
# RoCE REQUIRES lossless Ethernet (PFC or ECN)
# Without PFC, packets are SILENTLY DROPPED → NCCL hangs/errors

# Check PFC configuration
mlnx_qos -i eth0               # Show PFC settings
ethtool -S eth0 | grep pfc     # PFC frame counters

# Check RoCE GIDs
show_gids                       # List GID table entries
ibv_devinfo -v | grep GID      # Alternative

# Common RoCE issues:
# 1. PFC not enabled → packet loss → NCCL timeout
# 2. Wrong GID index → connection failure
# 3. MTU mismatch → performance degradation
# 4. DSCP not configured → no traffic differentiation
```

## NCCL Network Plugin Architecture

```
Application (PyTorch DDP)
    │
    ├── NCCL Core
    │   ├── Built-in: IB verbs (libibverbs)
    │   ├── Plugin: aws-ofi-nccl (libfabric → Slingshot CXI)
    │   └── Plugin: UCCL (custom IB optimizations)
    │
    ├── Transport Selection (automatic)
    │   ├── Intra-node: NVLink/NVSwitch > PCIe P2P > SHM
    │   └── Inter-node: RDMA > TCP
    │
    └── GPU Direct RDMA (if available)
        ├── nvidia-peermem kernel module
        └── NIC must be on same PCIe root as GPU
```

## Decision Tree: Network Optimization

```
Is it a communication bottleneck? (profiler shows NCCL > 30%)
│
├── YES → What fabric?
│   │
│   ├── InfiniBand
│   │   ├── Check SHARP availability → enable if supported
│   │   ├── Check UCCL plugin → adaptive routing + CC
│   │   └── Tune: NCCL_IB_HCA, NCCL_NET_GDR_LEVEL
│   │
│   ├── Slingshot
│   │   ├── DO NOT port IB-specific optimizations blindly
│   │   ├── Cassini HW handles: adaptive routing, CC, multi-rail
│   │   ├── Check aws-ofi-nccl plugin version
│   │   └── Tune: FI_CXI_* vars, NCCL_NET_GDR_LEVEL
│   │
│   └── RoCE
│       ├── FIRST: verify PFC is enabled (silent drops otherwise)
│       ├── Check DSCP/ECN configuration
│       └── Tune: NCCL_SOCKET_IFNAME, GID index
│
└── NO → Bottleneck is elsewhere (compute/memory/data loading)
```
