# High-Performance C++ for HPC

## Core Principle: Focus on What the Compiler Cannot Optimize

Modern `g++` with `-O2`/`-O3` handles:
- Loop unrolling, vectorization (auto-SIMD), inlining, constant folding
- Dead code elimination, branch prediction hints
- Basic memory access reordering

**You should focus on:**
- Algorithm and data structure choices (compiler can't change your algorithm)
- Memory layout and access patterns (compiler can't restructure your AoS→SoA)
- Parallelism (compiler doesn't auto-parallelize complex code)
- Cache-aware blocking (compiler tiling is limited)
- Lock-free concurrency (compiler can't reason about your threading model)

## Compiler Flags Reference

```bash
# Standard optimization levels
g++ -O0        # No optimization (debug)
g++ -O1        # Basic optimization
g++ -O2        # Recommended: good balance (default for production)
g++ -O3        # Aggressive: auto-vectorize, loop unrolling, function cloning
g++ -Ofast     # -O3 + fast-math (breaks IEEE float compliance!)

# Architecture-specific (CRITICAL for HPC)
g++ -march=native          # Use all instructions available on THIS CPU
g++ -march=armv9-a         # ARM (GH200 Grace)
g++ -march=x86-64-v3       # x86 with AVX2+FMA (A100/H100 host)
g++ -march=x86-64-v4       # x86 with AVX-512

# Vectorization
g++ -ftree-vectorize        # Auto-vectorize (included in -O3)
g++ -fopt-info-vec-optimized  # Report what was vectorized
g++ -fopt-info-vec-missed     # Report what was NOT vectorized (fix these!)
g++ -fopenmp-simd            # Enable OpenMP SIMD directives

# Link-time optimization (cross-file inlining)
g++ -flto                   # Link-time optimization
g++ -flto=auto              # Parallel LTO (use all cores)

# Profile-guided optimization (PGO)
g++ -fprofile-generate -o prog prog.cpp  # Step 1: instrument
./prog                                    # Step 2: run representative workload
g++ -fprofile-use -o prog prog.cpp       # Step 3: optimize using profile data

# Useful diagnostics
g++ -fopt-info-all           # All optimization reports
g++ -S -o output.s prog.cpp # Emit assembly (inspect vectorization)
```

### What `-O3` Does That You Don't Need To

```
✓ Loop unrolling           → Don't manually unroll
✓ Function inlining        → Don't inline manually (use `inline` hint sparingly)
✓ SIMD vectorization       → Don't write intrinsics unless -fopt-info-vec-missed shows failure
✓ Constant propagation     → Don't precompute constants manually
✓ Dead store elimination   → Don't worry about unused intermediate vars
✓ Branch prediction        → Don't reorder branches for prediction (use PGO instead)

What -O3 CANNOT do:
✗ Change your data structure       → You must choose SoA vs AoS
✗ Change your algorithm            → O(n²) stays O(n²)
✗ Parallelize across threads       → You must use OpenMP/threads
✗ Optimize cache blocking          → You must tile loops manually
✗ Eliminate unnecessary allocations → You must manage memory lifetime
✗ Fix false sharing                → You must align/pad shared data
```

## Boost Libraries for HPC

```cpp
// Boost.Compute — GPU computing (OpenCL backend)
#include <boost/compute.hpp>
// Use for portable GPU code when CUDA is not available

// Boost.Fiber — lightweight cooperative threads
#include <boost/fiber/all.hpp>
// Lower overhead than std::thread for fine-grained tasks

// Boost.Lockfree — lock-free data structures
#include <boost/lockfree/queue.hpp>
boost::lockfree::queue<int> queue(1024);
// For high-throughput producer-consumer without mutexes

// Boost.Pool — fast memory allocation
#include <boost/pool/pool_alloc.hpp>
std::vector<int, boost::pool_allocator<int>> vec;
// Avoids malloc overhead for frequent small allocations

// Boost.SIMD (now in std::experimental or use xsimd)
// Portable SIMD intrinsics wrapper

// Boost.Serialization — fast binary serialization
// For MPI message packing / checkpoint saving

// Boost.Interprocess — shared memory between processes
#include <boost/interprocess/shared_memory_object.hpp>
// For mp.Pool-like patterns in C++
```

## OpenMP Patterns

```cpp
// Basic parallel for
#pragma omp parallel for schedule(dynamic, 64)
for (int i = 0; i < N; i++) {
    process(data[i]);
}

// Reduction (thread-safe accumulation)
double total = 0.0;
#pragma omp parallel for reduction(+:total)
for (int i = 0; i < N; i++) {
    total += compute(data[i]);
}

// SIMD vectorization hint (when compiler misses it)
#pragma omp simd
for (int i = 0; i < N; i++) {
    a[i] = b[i] * c[i] + d[i];  // FMA candidate
}

// Nested parallelism: outer threads + inner SIMD
#pragma omp parallel for
for (int i = 0; i < M; i++) {
    #pragma omp simd
    for (int j = 0; j < N; j++) {
        result[i][j] = compute(i, j);
    }
}

// Task-based parallelism (irregular workloads)
#pragma omp parallel
{
    #pragma omp single
    {
        for (auto& item : worklist) {
            #pragma omp task firstprivate(item)
            process(item);
        }
    }
}

// Thread affinity (CRITICAL for NUMA)
export OMP_PLACES=cores
export OMP_PROC_BIND=close      # Bind threads to nearby cores
export OMP_NUM_THREADS=36       # Half of GH200 cores (one NUMA node)

// Schedule types:
// static  → equal chunks, good when work per iteration is uniform
// dynamic → grab next chunk when done, good for variable work
// guided  → decreasing chunk sizes, balance between static/dynamic
```

## std::thread and Thread Pool Patterns

```cpp
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>

// Simple thread pool (C++17)
class ThreadPool {
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex mtx;
    std::condition_variable cv;
    bool stop = false;

public:
    ThreadPool(size_t n) {
        for (size_t i = 0; i < n; i++) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock lock(mtx);
                        cv.wait(lock, [this] { return stop || !tasks.empty(); });
                        if (stop && tasks.empty()) return;
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    template<class F>
    void enqueue(F&& f) {
        {
            std::lock_guard lock(mtx);
            tasks.push(std::forward<F>(f));
        }
        cv.notify_one();
    }

    ~ThreadPool() {
        { std::lock_guard lock(mtx); stop = true; }
        cv.notify_all();
        for (auto& w : workers) w.join();
    }
};
```

## Memory Layout Optimization (Compiler Cannot Do This)

```cpp
// BAD: Array of Structures (AoS) — cache-unfriendly for column access
struct Particle { float x, y, z, mass; };
std::vector<Particle> particles(N);
// Accessing all x values: particles[0].x, particles[1].x, ...
// → stride-4 access, wastes 75% of cache lines

// GOOD: Structure of Arrays (SoA) — cache-friendly
struct Particles {
    std::vector<float> x, y, z, mass;
};
// Accessing all x values: x[0], x[1], x[2], ...
// → contiguous access, full cache line utilization

// BEST for mixed access: Hybrid AoSoA (Array of Structure of Arrays)
// Pack into SIMD-width groups
struct alignas(64) ParticleBlock {  // 64-byte aligned = cache line
    float x[16], y[16], z[16], mass[16];  // 16 = AVX-512 width
};
std::vector<ParticleBlock> blocks(N / 16);
```

## Cache-Aware Blocking (Compiler Tiling is Limited)

```cpp
// BAD: naive matrix multiply (cache-hostile for large N)
for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
        for (int k = 0; k < N; k++)
            C[i][j] += A[i][k] * B[k][j];  // B column access = cache miss

// GOOD: cache-blocked (tiles fit in L1/L2)
constexpr int BLOCK = 64;  // Tune to L1 cache size
for (int ii = 0; ii < N; ii += BLOCK)
    for (int jj = 0; jj < N; jj += BLOCK)
        for (int kk = 0; kk < N; kk += BLOCK)
            for (int i = ii; i < std::min(ii+BLOCK, N); i++)
                for (int j = jj; j < std::min(jj+BLOCK, N); j++)
                    for (int k = kk; k < std::min(kk+BLOCK, N); k++)
                        C[i][j] += A[i][k] * B[k][j];
// Block size: L1=32KB → BLOCK ≈ sqrt(32K/3/sizeof(float)) ≈ 52 → round to 64
```

## False Sharing Prevention

```cpp
// BAD: threads write to adjacent cache lines
int counters[NUM_THREADS];  // All on same or adjacent cache lines!
// Thread 0 writes counters[0], thread 1 writes counters[1]
// → cache line bouncing between cores = 10-100× slower

// GOOD: pad to cache line boundary
struct alignas(64) PaddedCounter {
    int value;
    // Padding to 64 bytes (cache line size) is implicit from alignas
};
PaddedCounter counters[NUM_THREADS];
```

## Lock-Free Patterns

```cpp
#include <atomic>

// Lock-free counter (no mutex overhead)
std::atomic<int64_t> global_counter{0};
global_counter.fetch_add(1, std::memory_order_relaxed);

// Lock-free SPSC queue (single producer, single consumer)
// Use boost::lockfree::spsc_queue or write with atomics

// Relaxed ordering (highest performance, weakest guarantees)
// Use when: independent counters, statistics
counter.store(val, std::memory_order_relaxed);

// Release-acquire (producer-consumer pattern)
// Producer: store with release → Consumer: load with acquire
data_ready.store(true, std::memory_order_release);   // producer
while (!data_ready.load(std::memory_order_acquire));  // consumer
```

## PyTorch C++ Extensions

```cpp
// torch/extension.h — write custom ops callable from Python
#include <torch/extension.h>

torch::Tensor my_fast_op(torch::Tensor input) {
    // Use ATen tensor operations (GPU-accelerated)
    auto output = torch::empty_like(input);

    // Or drop to raw pointers for custom logic
    float* in_ptr = input.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();

    #pragma omp parallel for
    for (int i = 0; i < input.numel(); i++) {
        out_ptr[i] = custom_compute(in_ptr[i]);
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_fast_op", &my_fast_op, "My fast operation");
}

// Build with: python setup.py install
// Or JIT: torch.utils.cpp_extension.load(name="my_op", sources=["my_op.cpp"])
```

## Compilation Cheat Sheet

```bash
# Development (fast compile, debuggable)
g++ -O0 -g -fsanitize=address,undefined -Wall -Wextra

# Testing (optimized but with debug info)
g++ -O2 -g -march=native -fopenmp

# Production (maximum performance)
g++ -O3 -march=native -flto=auto -fopenmp -DNDEBUG

# Production + PGO (best possible)
g++ -O3 -march=native -flto=auto -fprofile-generate -o prog prog.cpp
./prog < representative_input
g++ -O3 -march=native -flto=auto -fprofile-use -o prog prog.cpp

# Check what was vectorized
g++ -O3 -march=native -fopt-info-vec-optimized 2>&1 | head -50

# Check what WASN'T vectorized (fix these)
g++ -O3 -march=native -fopt-info-vec-missed 2>&1 | head -50

# GH200 ARM specific
g++ -O3 -march=armv9-a+sve2 -mtune=neoverse-v2 -fopenmp
```
