# Triton Kernel Development

## What is Triton

Triton is a Python-based GPU kernel language. You write block-level programs (not thread-level like CUDA). The Triton compiler handles:
- Thread/warp mapping within blocks
- Memory coalescing
- Shared memory management
- Some instruction scheduling

**You focus on:**
- Block-level algorithm design
- Tile sizes and grid dimensions
- Memory access patterns at tile granularity
- Autotuning configurations

## Using torch.compile Output as Reference

**Key technique:** Let `torch.compile` generate Triton code first, then use it as a starting point.

```python
import torch

# Step 1: Write the operation in PyTorch
def my_op(x, y):
    return torch.softmax(x + y, dim=-1)

# Step 2: Compile and inspect generated Triton code
compiled = torch.compile(my_op)
out = compiled(torch.randn(1024, 1024, device='cuda'),
               torch.randn(1024, 1024, device='cuda'))

# Step 3: Find the generated Triton code
# Location: ~/.cache/torch_inductor/<hash>/
# Or set: TORCH_COMPILE_DEBUG=1 to get the output directory
# Or set: torch._dynamo.config.log_level = "DEBUG"

# Step 4: Read the generated .py file
# It contains a Triton kernel with:
# - @triton.jit decorator
# - Autotune configs
# - The actual kernel logic
```

### Finding torch.compile Output

```bash
# Method 1: Environment variable
TORCH_COMPILE_DEBUG=1 python my_script.py
# Prints: torch._inductor.debug: Output code written to: /tmp/torchinductor_user/...

# Method 2: Inductor config
import torch._inductor.config
torch._inductor.config.debug = True
# Files saved to: ~/.cache/torch_inductor/

# Method 3: Direct inspection
from torch._inductor.codecache import PyCodeCache
# After compile, check PyCodeCache for generated code

# The output .py file contains:
# 1. Triton kernel with @triton.jit
# 2. Grid/block launch configuration
# 3. Autotune decorator with tested configs
# 4. A Python wrapper function that launches the kernel
```

### What to Learn from torch.compile Output

```
Study these patterns in generated code:
1. BLOCK_SIZE choices and why (powers of 2, matching hardware)
2. How reductions are tiled (multi-pass for large reductions)
3. How softmax numerics are handled (subtract max for stability)
4. Load/store patterns (masked loads for boundary handling)
5. Autotune config space (what block sizes the compiler tries)
6. Memory layout assumptions (contiguous, strided)
```

## Triton Kernel Basics

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,  # Compile-time constant
):
    # Each program instance processes one BLOCK_SIZE chunk
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask for boundary handling
    mask = offsets < n_elements

    # Load (masked, with boundary check)
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Compute
    output = x + y

    # Store
    tl.store(output_ptr + offsets, output, mask=mask)

# Launch
grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
add_kernel[grid](x, y, output, n, BLOCK_SIZE=1024)
```

## Autotuning

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],  # Retune when these change
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Accumulator in registers (FP32 for precision)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over K dimension in tiles
    for k in range(0, K, BLOCK_K):
        # Load tiles
        a_offsets = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] * stride_am + \
                    (k + tl.arange(0, BLOCK_K))[None, :] * stride_ak
        b_offsets = (k + tl.arange(0, BLOCK_K))[:, None] * stride_bk + \
                    (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :] * stride_bn

        a = tl.load(a_ptr + a_offsets, mask=..., other=0.0)
        b = tl.load(b_ptr + b_offsets, mask=..., other=0.0)

        # Matrix multiply (uses tensor cores if types allow)
        acc += tl.dot(a, b)

    # Store result
    c_offsets = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] * stride_cm + \
                (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :] * stride_cn
    tl.store(c_ptr + c_offsets, acc.to(tl.float16), mask=...)
```

### Autotune Config Guidelines

```
BLOCK_SIZE selection:
- Must be power of 2
- Minimum: 32 (one warp)
- Maximum: limited by shared memory and registers
- Typical: 64, 128, 256 for 1D; 32-128 per dim for 2D

num_warps:
- 4 is safe default
- 8 for large blocks (≥ 256 threads)
- 2 for small blocks or register-heavy kernels

num_stages (software pipelining):
- 2-4 typical (more stages = more shared memory usage)
- Higher stages hide memory latency better
- Hopper: up to 8 stages with TMA

Key for autotune:
- List dimensions that affect optimal config
- When input size changes significantly, optimal config may change
```

## Common Patterns

### Fused Softmax

```python
@triton.jit
def softmax_kernel(
    output_ptr, input_ptr,
    input_row_stride, output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * input_row_stride

    # Load row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    row = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=-float('inf'))

    # Numerically stable softmax
    row_max = tl.max(row, axis=0)
    row = row - row_max          # Subtract max for stability
    numerator = tl.exp(row)
    denominator = tl.sum(numerator, axis=0)
    softmax_out = numerator / denominator

    # Store
    tl.store(output_ptr + row_idx * output_row_stride + col_offsets,
             softmax_out, mask=mask)
```

### Fused Attention (Flash Attention Pattern)

```python
@triton.jit
def flash_attention_kernel(
    Q, K, V, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Tiled attention: process BLOCK_M queries at a time, iterate over keys."""
    # For each BLOCK_M chunk of queries:
    # 1. Load Q tile (BLOCK_M × BLOCK_K) into registers
    # 2. Iterate over K,V in BLOCK_N chunks:
    #    a. Load K tile → compute QK^T (BLOCK_M × BLOCK_N) in registers
    #    b. Online softmax: track running max and normalizer
    #    c. Load V tile → accumulate attention output
    # 3. Write final output (already normalized)
    pass  # See Triton tutorials for full implementation
```

### Reduction

```python
@triton.jit
def reduce_kernel(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    block_sum = tl.sum(x, axis=0)

    # Atomic add to output (one per block)
    tl.atomic_add(output_ptr, block_sum)
```

## Triton vs CUDA: When to Use What

| Scenario | Use Triton | Use CUDA |
|----------|-----------|----------|
| Fused elementwise + reduction | Yes | Overkill |
| Custom attention variant | Yes | If Triton is too slow |
| Warp-level intrinsics | No | Yes (shuffle, vote) |
| Shared memory bank conflict tuning | No (compiler handles) | Yes (manual control) |
| Multi-kernel pipeline | Maybe | Yes (CUDA streams/graphs) |
| Quick prototype | Yes (Python!) | No (compile cycle slow) |
| Maximum absolute performance | Maybe | Yes (but diminishing returns) |
| Integration with PyTorch | Yes (seamless) | Yes (via extensions) |

## Performance Tips

```python
# 1. Use tl.constexpr for anything that should be compile-time
# This enables the compiler to optimize memory layout and tiling
BLOCK_SIZE: tl.constexpr  # Good
# block_size (runtime) → bad, prevents many optimizations

# 2. Coalesce memory accesses
# Triton programs are BLOCK-level: tl.arange(0, BLOCK) accesses must be contiguous
offsets = pid * BLOCK + tl.arange(0, BLOCK)  # Contiguous → coalesced

# 3. Use appropriate dtypes
# Load as float16/bfloat16, accumulate as float32
x = tl.load(ptr + offsets).to(tl.float32)  # Upcast for accumulation
result = compute(x)
tl.store(out + offsets, result.to(tl.float16))  # Downcast for storage

# 4. Minimize tl.atomic_add calls
# Reduce within block first, then one atomic per block (not per thread)

# 5. Use num_stages for memory latency hiding
# @triton.Config({...}, num_stages=4)  # More stages = more prefetch

# 6. Profile with Triton's built-in tools
# triton.testing.do_bench(lambda: kernel[grid](...))
# Returns milliseconds, handles warmup automatically
```

## Debugging Triton

```python
# Print from kernel (for debugging only, kills performance)
tl.device_print("val =", val)

# Check generated PTX/SASS
# Set: TRITON_PRINT_AUTOTUNING=1 for config selection info
# Set: MLIR_ENABLE_DUMP=1 for intermediate IR

# Use triton.testing for benchmarks
import triton.testing

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[2**i for i in range(10, 25)],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'PyTorch'],
        ylabel='GB/s',
        plot_name='add-performance',
    )
)
def benchmark(N, provider):
    x = torch.randn(N, device='cuda', dtype=torch.float32)
    y = torch.randn(N, device='cuda', dtype=torch.float32)
    if provider == 'triton':
        return triton.testing.do_bench(lambda: add_kernel[...](x, y, output, N, BLOCK_SIZE=1024))
    else:
        return triton.testing.do_bench(lambda: x + y)
```
