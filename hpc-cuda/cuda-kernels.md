# CUDA Kernel Development — Architecture-Aware Programming

## Authoritative Reference

**ALWAYS consult the CUDA C++ Programming Guide before writing or optimizing CUDA code:**
https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html

Key chapters to reference:
- Ch. 3: Programming Model (grids, blocks, threads, memory hierarchy)
- Ch. 5: Performance Guidelines (memory coalescing, occupancy, instruction throughput)
- Ch. 7: Cooperative Groups
- Ch. 15: Mathematical Functions (precision vs speed trade-offs)
- Appendix K: Compute Capabilities (per-architecture limits)

## NVIDIA Accelerated Libraries — Use Before Writing Custom Kernels

**Before writing ANY custom kernel, check if an NVIDIA library already solves it.**
These are hand-tuned by NVIDIA engineers for each architecture — you will NOT beat them.

### cuBLAS — Dense Linear Algebra
```
GEMM, GEMV, batched GEMM, strided batched GEMM
Docs: https://docs.nvidia.com/cuda/cublas/

When to use:
- Matrix multiply of any size → cublasGemmEx / cublasSgemm
- Batched small matrix multiply → cublasGemmBatchedEx
- Mixed precision GEMM → cublasGemmEx with CUBLAS_COMPUTE_32F

When NOT to use:
- Sparse matrices → use cuSPARSE
- Element-wise ops → custom kernel or Triton
- Fused GEMM + activation → CUTLASS or custom kernel

PyTorch integration:
- torch.mm, torch.matmul, torch.bmm → already use cuBLAS
- torch.linalg.* → cuBLAS/cuSOLVER backend
- No need to call cuBLAS directly unless bypassing PyTorch

Key tuning:
- cublasSetMathMode(CUBLAS_TENSOR_OP_MATH)  → enable tensor cores
- CUBLAS_WORKSPACE_CONFIG=:4096:8           → deterministic mode
- cublas*Ex variants allow mixed precision (FP16 input, FP32 compute)
```

### cuDNN — Deep Learning Primitives
```
Convolution, pooling, normalization, activation, RNN, attention
Docs: https://docs.nvidia.com/deeplearning/cudnn/

When to use:
- Standard DL layers → cuDNN is THE backend (PyTorch uses it automatically)
- Convolution algorithm selection → cudnnFindConvolutionForwardAlgorithm
- Fused operations → cudnnFusedOps (conv+bias+activation in one kernel)

Key tuning:
- torch.backends.cudnn.benchmark = True   → auto-select fastest conv algorithm
- torch.backends.cudnn.deterministic = True → reproducible but may be slower
- cuDNN 9+: graph API for fused multi-op patterns

When NOT to use:
- Non-standard layer designs → write custom kernel
- Attention variants not in cuDNN → Flash Attention / Triton
```

### Thrust — GPU STL (Part of CCCL)
```
Sort, reduce, scan, transform, gather, scatter
Docs: https://nvidia.github.io/cccl/thrust/

When to use:
- Device-wide sort → thrust::sort (radix sort, fastest on GPU)
- Reductions → thrust::reduce, thrust::transform_reduce
- Prefix sum → thrust::inclusive_scan, thrust::exclusive_scan
- Remove/filter → thrust::remove_if, thrust::copy_if
- Key-value sort → thrust::sort_by_key

Performance notes:
- Thrust uses CUB internally → close to hand-tuned performance
- Fancy iterators (counting, zip, transform) avoid materializing temp arrays
- thrust::device_vector manages GPU memory (but has alloc overhead per op)
- For repeated ops: use raw pointers + thrust::device_ptr to avoid realloc

PyTorch equivalent:
- torch.sort() → uses CUB/Thrust internally
- torch.cumsum() → uses CUB scan
- Generally no need to call Thrust from Python — but useful in C++ extensions
```

### Other Key Libraries
```
cuSPARSE  — Sparse matrix operations (SpMV, SpMM, SpGEMM)
cuSOLVER  — Dense/sparse factorizations (LU, QR, SVD, eigensolvers)
cuFFT     — Fast Fourier Transform
cuRAND    — Random number generation on GPU
NCCL      — Multi-GPU/node collective communication (see nccl-comms.md)
CUTLASS   — Template library for custom GEMM variants (see CuTe section below)

Decision: Library vs Custom Kernel
├── Standard BLAS/DL op → cuBLAS/cuDNN (ALWAYS)
├── Sort/scan/reduce → Thrust/CUB (ALWAYS)
├── Fused custom ops → Triton first, CUDA if Triton is too slow
├── Novel algorithm → Custom CUDA kernel
└── Custom GEMM variant → CUTLASS/CuTe
```

## Core Principle: CUDA Compiler (nvcc/nvrtc) Optimization is Weak

Unlike g++ `-O3` which aggressively optimizes, nvcc:
- Does NOT auto-vectorize across threads (you manage warps explicitly)
- Does NOT auto-tile or auto-block (you set grid/block dims)
- Limited register/shared memory allocation optimization
- No cross-kernel fusion (you must fuse manually or use CUDAGraphs)

**You must explicitly manage:**
- Thread/block/grid dimensions
- Memory hierarchy (registers → shared → L2 → global)
- Warp-level operations (shuffle, vote, cooperative groups)
- Memory coalescing and bank conflict avoidance
- Occupancy and register pressure

## GPU Architecture Quick Reference

```
Streaming Multiprocessor (SM):
├── Warp Schedulers (4 per SM on Hopper)
│   └── Each schedules 1 warp (32 threads) per cycle
├── CUDA Cores (FP32/INT32)
├── Tensor Cores (matrix multiply: FP16/BF16/TF32/FP8)
├── Register File (256 KB per SM)
├── Shared Memory / L1 Cache (configurable, up to 228 KB on Hopper)
└── Special Function Units (SFU: sin, cos, exp, rsqrt)

Hopper (H100/GH200):
  132 SMs, 16896 CUDA cores, 528 Tensor Cores
  Max 2048 threads per SM, 64 warps per SM
  Register file: 256 KB per SM → 255 registers per thread max
  Shared memory: up to 228 KB per SM
  L2: 50 MB

Occupancy = active_warps / max_warps_per_SM
  Higher occupancy → better latency hiding (more warps to schedule)
  But: more warps = fewer registers per thread = potential spills
  Sweet spot: often 50-75% occupancy, not 100%
```

## CUDA 12/13 Features to Use

### CUDA Cooperative Groups (since CUDA 9, mature in 12+)

```cuda
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void kernel(float* data, int n) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    // Warp-level reduction (replaces __shfl_down_sync)
    float val = data[threadIdx.x];
    for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
        val += warp.shfl_down(val, offset);
    }

    // Grid-level sync (requires cooperative launch)
    cg::grid_group grid = cg::this_grid();
    grid.sync();  // All blocks synchronize
}

// Launch cooperative kernel
void* args[] = {&data, &n};
cudaLaunchCooperativeKernel((void*)kernel, grid, block, args);
```

### CCCL — CUDA C++ Core Libraries (CUDA 12+)

```cuda
// CCCL includes: libcu++ (std-like), Thrust, CUB
// Prefer CCCL over hand-written primitives

// --- libcu++ (device-side C++ standard library) ---
#include <cuda/std/atomic>
#include <cuda/std/barrier>
#include <cuda/std/semaphore>

// Atomic operations (better than legacy atomicAdd)
cuda::std::atomic<int> counter;
counter.fetch_add(1, cuda::std::memory_order_relaxed);

// Barrier (inter-thread sync within block)
__shared__ cuda::barrier<cuda::thread_scope_block> bar;
// Initialize: init(&bar, blockDim.x);
bar.arrive_and_wait();

// --- CUB (block/warp/device primitives) ---
#include <cub/cub.cuh>

// Block-level reduce (fastest possible)
__global__ void reduce_kernel(float* input, float* output, int n) {
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;

    float val = (threadIdx.x < n) ? input[blockIdx.x * blockDim.x + threadIdx.x] : 0;
    float block_sum = BlockReduce(temp).Sum(val);

    if (threadIdx.x == 0) output[blockIdx.x] = block_sum;
}

// Device-level sort (replaces thrust::sort for more control)
#include <cub/device/device_radix_sort.cuh>
void* temp = nullptr;
size_t temp_bytes = 0;
cub::DeviceRadixSort::SortKeys(temp, temp_bytes, d_in, d_out, n);
cudaMalloc(&temp, temp_bytes);
cub::DeviceRadixSort::SortKeys(temp, temp_bytes, d_in, d_out, n);

// Warp-level scan
#include <cub/warp/warp_scan.cuh>
typedef cub::WarpScan<int> WarpScan;
__shared__ typename WarpScan::TempStorage temp[NUM_WARPS];
int warp_id = threadIdx.x / 32;
int val = input[threadIdx.x];
WarpScan(temp[warp_id]).InclusiveSum(val, val);

// --- Thrust (high-level, STL-like) ---
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>

thrust::device_vector<float> d_vec(h_vec.begin(), h_vec.end());
thrust::sort(d_vec.begin(), d_vec.end());
float sum = thrust::reduce(d_vec.begin(), d_vec.end());
```

### CUDA Tile (CUDA 13 — CuTe / CUTLASS 3.x)

```cuda
// CuTe (CUDA Tensor) — from CUTLASS 3.x, available standalone in CUDA 13
// Provides tensor abstractions that map to hardware (registers, shared mem, global)

#include <cute/tensor.hpp>
using namespace cute;

// Define tensor layouts that match hardware
auto layout = make_layout(make_shape(Int<128>{}, Int<64>{}),   // 128 x 64 tile
                          make_stride(Int<64>{}, Int<1>{}));    // Row-major

// Partition across threads (automatic warp-level tiling)
auto thr_layout = make_layout(make_shape(Int<32>{}, Int<8>{}));

// Copy from global to shared memory (async, hardware-accelerated)
// Uses TMA (Tensor Memory Accelerator) on Hopper
copy(gmem_tensor, smem_tensor);  // Generates optimal ldmatrix/TMA instructions

// Matrix multiply using tensor cores
// CuTe generates mma instructions matching your data types
gemm(smem_A, smem_B, acc_C);  // FP16→FP32 accumulate on tensor cores
```

### Tensor Memory Accelerator (TMA) — Hopper Only

```cuda
// TMA: hardware unit for bulk async memory copies
// Eliminates address calculation overhead, handles tiling automatically

#include <cuda/barrier>
#include <cute/tensor.hpp>

// Create TMA descriptor (once, on host)
CUtensorMap tma_desc;
cuTensorMapEncodeTiled(&tma_desc, ...);

// In kernel: initiate async TMA copy
__shared__ cute::uint128_t smem_buf[SIZE];
cute::cp_async_bulk_tensor_2d_global_to_shared(
    &smem_buf, &tma_desc, coord_x, coord_y);

// Wait for completion
cute::cp_async_bulk_commit_group();
cute::cp_async_bulk_wait_group_read<0>();
```

## Optimization Checklist (Beyond Compiler)

### 1. Memory Coalescing

```cuda
// GOOD: consecutive threads access consecutive addresses
int idx = blockIdx.x * blockDim.x + threadIdx.x;
float val = data[idx];  // 1 transaction for 32 threads

// BAD: strided access
float val = data[idx * stride];  // Up to 32 transactions!

// BAD: AoS access pattern
struct Particle { float x, y, z; };
float x = particles[idx].x;  // Stride-3 access

// GOOD: SoA
float x = particles_x[idx];  // Contiguous
```

### 2. Shared Memory Bank Conflicts

```cuda
// 32 banks, 4 bytes each
// Conflict: two threads in same warp access different addresses in same bank

// BAD: stride-32 access → all threads hit same bank
__shared__ float smem[32][32];
float val = smem[threadIdx.x][0];  // Column access → 32-way conflict!

// GOOD: pad to avoid conflicts
__shared__ float smem[32][33];  // +1 padding
float val = smem[threadIdx.x][0];  // No conflict (bank = threadIdx.x % 32)
```

### 3. Register Pressure Management

```cuda
// Check register usage:
// nvcc --ptxas-options=-v kernel.cu
// "Used 64 registers" → 64 regs × 2048 threads/SM ÷ 32 threads/warp = 64 warps
// But SM max = 64 warps, so 64 regs → 100% occupancy still possible
// At 128 regs → max 32 warps → 50% occupancy

// Reduce register pressure:
// 1. Limit max registers per thread
__launch_bounds__(256, 4)  // 256 threads/block, min 4 blocks/SM
__global__ void kernel() { ... }

// 2. Use __ldg() for read-only data (uses texture cache, frees registers)
float val = __ldg(&input[idx]);

// 3. Move intermediate values to shared memory
__shared__ float temp[256];
temp[threadIdx.x] = expensive_computation();
__syncthreads();
// Reuse temp[] instead of keeping in registers
```

### 4. Warp-Level Operations

```cuda
// Warp shuffle: exchange data between threads WITHOUT shared memory
float val = some_computation();

// Broadcast from lane 0 to all
float broadcast = __shfl_sync(0xffffffff, val, 0);

// Shift down (for reductions)
float sum = val;
for (int offset = 16; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
}
// Lane 0 now has the sum of all 32 values

// Warp vote
bool any_positive = __any_sync(0xffffffff, val > 0);
bool all_positive = __all_sync(0xffffffff, val > 0);
uint32_t mask = __ballot_sync(0xffffffff, val > 0);
```

### 5. Async Copy (Ampere+)

```cuda
// Hardware-accelerated global → shared memory copy
// Bypasses registers → saves register pressure and bandwidth

#include <cuda_pipeline.h>

__shared__ float smem[256];

// Async copy: global → shared (no register intermediate)
__pipeline_memcpy_async(&smem[threadIdx.x],
                        &global_data[idx],
                        sizeof(float));
__pipeline_commit();
__pipeline_wait_prior(0);  // Wait for all copies
__syncthreads();
```

## nvcc Compilation

```bash
# Target architecture
nvcc -arch=sm_90a        # Hopper (H100/GH200) with async features
nvcc -arch=sm_80         # Ampere (A100)
nvcc -arch=sm_89         # Ada Lovelace (L40/RTX 4090)

# Optimization flags
nvcc -O3                 # Basic optimization
nvcc --use_fast_math     # Fast math (less accurate, faster)
nvcc -maxrregcount=128   # Limit registers per thread
nvcc --ptxas-options=-v  # Show register/shared memory usage

# Generate for multiple architectures
nvcc -gencode arch=compute_80,code=sm_80 \
     -gencode arch=compute_90,code=sm_90 \
     -gencode arch=compute_90a,code=sm_90a

# CCCL/CuTe (CUDA 13)
nvcc -std=c++17 -I/path/to/cutlass/include kernel.cu

# Profiling-ready build
nvcc -O3 -lineinfo -arch=sm_90a kernel.cu  # Keep line info for ncu
```

## Common Mistakes

```cuda
// MISTAKE 1: Unnecessary __syncthreads()
// Only needed when threads share data through shared memory
__shared__ float smem[N];
smem[tid] = compute();
__syncthreads();   // ← NEEDED: other threads read smem
result = smem[other_tid];

// Without shared memory → no sync needed:
float local = compute();  // Each thread has its own copy
result = local;           // No sync needed!

// MISTAKE 2: Divergent warps in hot path
if (threadIdx.x < 16) {
    path_a();  // Half the warp executes this
} else {
    path_b();  // Other half executes this
}
// → Both paths execute serially! 2× slower

// MISTAKE 3: Using atomicAdd when reduction is possible
atomicAdd(&global_sum, local_val);  // ← SLOW: global atomic from every thread
// FIX: warp reduce → block reduce → single atomic per block

// MISTAKE 4: Launching tiny kernels in loop
for (int i = 0; i < 1000; i++) {
    kernel<<<1, 32>>>(data, i);  // ← Launch overhead dominates!
}
// FIX: Single kernel with grid processing all iterations
// Or: CUDAGraphs to capture and replay the sequence

// MISTAKE 5: Forgetting non_blocking in PyTorch→CUDA bridge
tensor_gpu = tensor_cpu.to('cuda')  // ← SYNCHRONOUS!
// FIX: tensor_gpu = tensor_cpu.to('cuda', non_blocking=True)
```
