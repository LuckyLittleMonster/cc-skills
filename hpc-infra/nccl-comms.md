# NCCL Communication — Topology, Algorithms & Debugging

## AllReduce Algorithms

NCCL selects algorithm automatically based on topology and message size:

```
Ring AllReduce:
  - Default for most topologies
  - Bandwidth-optimal: each GPU sends/receives N/P data
  - Latency: O(P) steps → bad for many GPUs with small messages
  - Best for: few GPUs, large messages

Tree AllReduce:
  - Latency-optimal: O(log P) steps
  - Uses less bandwidth than ring for large messages
  - Best for: many GPUs, small messages

SHARP (In-Network Reduce):
  - InfiniBand SHARP: reduction happens IN the switch
  - Requires Mellanox switches with SHARP support
  - Not available on Slingshot or RoCE
  - ~2× AllReduce speedup for supported ops
```

## Key NCCL Environment Variables

```bash
# Debugging
NCCL_DEBUG=INFO              # Basic info (topology, algorithm selection)
NCCL_DEBUG=TRACE             # Detailed traces (per-op timing)
NCCL_DEBUG_SUBSYS=ALL        # All subsystems (INIT,NET,GRAPH,TUNING,...)
NCCL_DEBUG_FILE=/tmp/nccl_%h_%p.log  # Per-rank log files

# Algorithm selection
NCCL_ALGO=Ring,Tree,CollNetDirect,CollNetChain  # Force algorithm
NCCL_PROTO=Simple,LL,LL128   # Protocol selection

# Network
NCCL_NET_GDR_LEVEL=SYS       # GPU Direct RDMA level (PHB,NODE,SYS)
NCCL_NET_GDR_READ=1           # Enable GPU Direct RDMA for reads
NCCL_SOCKET_IFNAME=eth0       # Network interface to use
NCCL_IB_HCA=mlx5              # InfiniBand HCA to use
NCCL_P2P_LEVEL=NVL            # P2P transport level (LOC,NVL,PIX,PHB,SYS)
NCCL_P2P_DISABLE=0            # Don't disable P2P (default)

# Topology
NCCL_TOPO_FILE=/path/to.xml   # Custom topology file
NCCL_TOPO_DUMP_FILE=/tmp/topo.xml  # Dump detected topology

# Performance tuning
NCCL_BUFFSIZE=8388608          # Communication buffer size (8MB default)
NCCL_NTHREADS=512              # Threads per NCCL kernel
NCCL_MAX_NCHANNELS=32          # Max channels (parallel streams)
NCCL_MIN_NCHANNELS=1           # Min channels
NCCL_CROSS_NIC=1               # Allow cross-NIC communication

# Timeout
NCCL_TIMEOUT=1800000           # Timeout in ms (default varies)
```

## Common NCCL Errors & Root Causes

```
"NCCL WARN Cuda failure 'peer access is not supported'"
  → GPUs not on same PCIe switch, or NVLink not available
  → Check: nvidia-smi topo -m

"NCCL WARN NET/IB : Got completion with error"
  → Network issue: bad cable, switch error, or NIC:GPU mismatch
  → Check: ibstat, dmesg | grep mlx5

"NCCL WARN Timeout" / "watchdog timeout"
  → Rank desync, one rank stuck, or network partition
  → Check: Are all ranks reaching the same collective?
  → Check: NIC:GPU ratio — contention causes timeouts

"NCCL WARN Call to ibv_modify_qp failed"
  → RDMA setup failure, often firmware/driver mismatch
  → Check: ofed_info, nvidia-smi (driver version)

"Unhandled system error, NCCL version X.Y.Z"
  → Often: GPU Direct RDMA not working
  → Check: Is nvidia-peermem module loaded?
  →   lsmod | grep nvidia_peermem
```

## GPU Direct RDMA Checklist

```
For GPU Direct RDMA to work, ALL of these must be true:
1. nvidia-peermem (or nv_peer_mem) kernel module loaded
2. NIC and GPU on same PCIe root complex (or NVSwitch)
3. NIC:GPU ratio ≥ 1:1 for full bandwidth (< 1:1 = contention)
4. IOMMU disabled or configured for passthrough
5. Compatible driver versions (GPU driver + OFED/CXI)

Query commands:
  lsmod | grep -E "nvidia_peermem|nv_peer_mem"
  nvidia-smi topo -m    # Check NIC-GPU affinity (look for PIX/PHB/SYS)
  cat /proc/driver/nvidia/params | grep peer  # Peer mappings
```

## NCCL Topology Detection

NCCL auto-detects topology from PCIe/NVLink info. Dump it:

```bash
NCCL_TOPO_DUMP_FILE=/tmp/topo.xml NCCL_DEBUG=INFO python -c "
import torch.distributed as dist
dist.init_process_group('nccl')
"
cat /tmp/topo.xml
```

## Known Hardware-Specific Issues

### NIC:GPU Ratio Problem (e.g., NCSA Delta)
```
4× A100 sharing 1× ConnectX-6 NIC
├── GPU Direct RDMA: works but 4 GPUs contend for 1 NIC
├── Effective per-GPU bandwidth: ~50 GB/s ÷ 4 = ~12.5 GB/s
├── Symptom: NCCL timeouts, not NCCL bugs
├── Fix: NCCL_NET_GDR_LEVEL=PHB or disable GDR
└── Fix: Use node-local AllReduce first, then cross-node
```

### Multi-NIC Systems
```
DGX A100: 8× A100 + 8× ConnectX-6 (1:1 ratio)
├── Each GPU has dedicated NIC → full RDMA bandwidth
├── NCCL automatically detects and uses NIC affinity
└── Optimal: no contention

DGX H100: 8× H100 + 8× ConnectX-7 (1:1 ratio) + NVSwitch
├── Intra-node: NVSwitch (900 GB/s per GPU)
├── Inter-node: 8× 400 Gbps IB
└── NCCL uses NVSwitch for intra, IB for inter
```

## NCCL Performance Testing

```bash
# Build and run nccl-tests
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests && make MPI=1

# Single node, all GPUs
./build/all_reduce_perf -b 1M -e 1G -f 2 -g <num_gpus>

# Multi-node (via MPI)
mpirun -np 4 -H node1:2,node2:2 ./build/all_reduce_perf -b 1M -e 1G -f 2

# Key output columns:
#   algbw = algorithm bandwidth (actual throughput)
#   busbw = bus bandwidth (normalized, comparable across GPU counts)
```
