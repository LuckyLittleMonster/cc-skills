# C++/CUDA → Python Extensions

Compile C++ and CUDA code into shared libraries (.so) and call from Python.

---

## Method Selection

```
Need to call C++/CUDA from Python?
│
├─ PyTorch tensor in/out?
│   ├─ YES, simple op → torch.utils.cpp_extension (easiest)
│   └─ YES, complex build → setup.py with CUDAExtension
│
├─ Pure C/C++, no PyTorch?
│   ├─ Simple interface → ctypes (no build dependency)
│   ├─ Complex C++ API → pybind11 (clean bindings)
│   └─ Need Cython perf → Cython .pyx
│
└─ Quick prototype?
    └─ torch.utils.cpp_extension.load() (JIT compile)
```

---

## 1. PyTorch C++ Extensions (Recommended for ML)

### 1.1 JIT Compilation (Development / Prototyping)

Zero setup — compiles on first call, caches .so automatically.

```python
from torch.utils.cpp_extension import load

# JIT compile C++ → .so (cached at ~/.cache/torch_extensions/)
my_op = load(
    name="my_op",
    sources=["csrc/my_op.cpp"],
    extra_cflags=["-O3", "-march=native"],
    verbose=True,
)

# Use like any Python function
output = my_op.forward(input_tensor)
```

**C++ side** (`csrc/my_op.cpp`):
```cpp
#include <torch/extension.h>

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Expected float32");

    auto output = torch::empty_like(input);

    // ... computation ...

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "My forward op");
}
```

### 1.2 JIT with CUDA Kernels

```python
my_cuda_op = load(
    name="my_cuda_op",
    sources=["csrc/my_op.cpp", "csrc/my_kernel.cu"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_90a"],
    verbose=True,
)
```

**CUDA kernel** (`csrc/my_kernel.cu`):
```cuda
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void my_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * 2.0f;  // example
    }
}

torch::Tensor my_cuda_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int n = input.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    my_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    return output;
}
```

**C++ binding** (`csrc/my_op.cpp`):
```cpp
#include <torch/extension.h>

// Declare CUDA function (defined in .cu file)
torch::Tensor my_cuda_forward(torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &my_cuda_forward, "CUDA forward");
}
```

### 1.3 Ahead-of-Time Build (Production / pip install)

```python
# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="my_ops",
    ext_modules=[
        CUDAExtension(
            name="my_ops",
            sources=[
                "csrc/my_op.cpp",
                "csrc/my_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-march=native"],
                "nvcc": ["-O3", "--use_fast_math", "-arch=sm_90a",
                         "--expt-relaxed-constexpr"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

```bash
# Build
pip install -e .
# or
python setup.py build_ext --inplace  # produces my_ops.cpython-313-*.so

# Use
import my_ops
output = my_ops.forward(input_tensor)
```

### 1.4 torch::Tensor API Quick Reference

```cpp
// Creation
auto x = torch::empty({N, D}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
auto y = torch::zeros_like(x);

// Access raw pointer (for CUDA kernels)
float* ptr = x.data_ptr<float>();

// Properties
x.size(0);          // first dimension
x.numel();          // total elements
x.is_cuda();        // device check
x.is_contiguous();  // memory layout
x.stride(0);        // stride of dim 0

// Operations (same as Python torch API)
auto z = torch::matmul(x, w) + b;
auto [values, indices] = torch::topk(z, k);

// Device management
auto x_gpu = x.to(torch::kCUDA);
auto x_cpu = x.to(torch::kCPU);

// Tensor options
auto opts = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, 0);
auto t = torch::empty({N}, opts);
```

### 1.5 Error Handling

```cpp
// Input validation (throws Python exception on failure)
TORCH_CHECK(input.is_cuda(), "Expected CUDA tensor, got CPU");
TORCH_CHECK(input.dim() == 2, "Expected 2D tensor, got ", input.dim(), "D");
TORCH_CHECK(input.dtype() == torch::kFloat32, "Expected float32");
TORCH_CHECK(input.is_contiguous(), "Expected contiguous tensor");

// CUDA error checking
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err)); \
} while(0)
```

---

## 2. pybind11 (Pure C++ Without PyTorch)

For non-tensor C++ code (algorithms, data structures, system calls).

```cpp
// my_lib.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>       // auto-convert STL containers
#include <pybind11/numpy.h>     // NumPy array support
namespace py = pybind11;

// Example: process molecular SMILES in C++ for speed
std::vector<double> compute_properties(const std::vector<std::string>& smiles_list) {
    std::vector<double> results(smiles_list.size());
    #pragma omp parallel for schedule(dynamic, 64)
    for (size_t i = 0; i < smiles_list.size(); i++) {
        results[i] = expensive_computation(smiles_list[i]);
    }
    return results;
}

// NumPy array interface (zero-copy when possible)
py::array_t<float> process_array(py::array_t<float> input) {
    auto buf = input.request();  // buffer info: ptr, shape, strides
    float* ptr = static_cast<float*>(buf.ptr);
    int n = buf.shape[0];

    auto result = py::array_t<float>(n);
    auto res_buf = result.request();
    float* res_ptr = static_cast<float*>(res_buf.ptr);

    for (int i = 0; i < n; i++) {
        res_ptr[i] = ptr[i] * 2.0f;
    }
    return result;
}

PYBIND11_MODULE(my_lib, m) {
    m.doc() = "My C++ library";
    m.def("compute_properties", &compute_properties,
          "Compute molecular properties in parallel",
          py::arg("smiles_list"));
    m.def("process_array", &process_array,
          "Process NumPy array");
}
```

**Build** (`setup.py`):
```python
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

setup(
    ext_modules=[
        Pybind11Extension(
            "my_lib",
            ["my_lib.cpp"],
            extra_compile_args=["-O3", "-march=native", "-fopenmp"],
            extra_link_args=["-fopenmp"],
        ),
    ],
    cmdclass={"build_ext": build_ext},
)
```

---

## 3. ctypes (Simplest — No Build Dependencies)

Call any .so directly from Python. No compilation step in Python.

```bash
# Compile standalone .so
g++ -O3 -march=native -shared -fPIC -o libmy_func.so my_func.cpp
# or with CUDA:
nvcc -O3 -shared -Xcompiler -fPIC -arch=sm_90a -o libmy_cuda.so my_cuda.cu
```

```python
import ctypes
import numpy as np

lib = ctypes.CDLL("./libmy_func.so")

# Declare function signature
lib.compute.argtypes = [
    ctypes.c_int,                               # int n
    np.ctypeslib.ndpointer(dtype=np.float32),   # float* input
    np.ctypeslib.ndpointer(dtype=np.float32),   # float* output
]
lib.compute.restype = ctypes.c_int  # return type

# Call
input_arr = np.random.randn(1000).astype(np.float32)
output_arr = np.empty(1000, dtype=np.float32)
lib.compute(1000, input_arr, output_arr)
```

**When to use:** Simple numerical functions, legacy C libraries, minimal dependencies.

**Limitation:** Manual type marshaling, no C++ class support (use `extern "C"`), no automatic STL/tensor conversion.

---

## 4. cffi (Alternative to ctypes)

```python
from cffi import FFI
ffi = FFI()

# Declare C interface
ffi.cdef("""
    void compute(int n, float* input, float* output);
""")

lib = ffi.dlopen("./libmy_func.so")

# Use with NumPy
input_ptr = ffi.cast("float*", input_arr.ctypes.data)
output_ptr = ffi.cast("float*", output_arr.ctypes.data)
lib.compute(1000, input_ptr, output_ptr)
```

---

## Build Recipes

### Multi-file CUDA Extension with cuBLAS

```python
# setup.py
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools import setup

setup(
    ext_modules=[
        CUDAExtension(
            "fast_ops",
            sources=[
                "csrc/bindings.cpp",
                "csrc/attention.cu",
                "csrc/reduction.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-arch=sm_90a", "--use_fast_math",
                         "--expt-relaxed-constexpr"],
            },
            libraries=["cublas", "curand"],  # link NVIDIA libraries
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

### Cross-Platform Architecture Flags

```python
import platform

nvcc_flags = ["-O3", "--use_fast_math"]
cxx_flags = ["-O3"]

if platform.machine() == "aarch64":
    # GH200 Grace ARM
    cxx_flags += ["-march=armv9-a+sve2", "-mtune=neoverse-v2"]
    nvcc_flags += ["-arch=sm_90a"]  # Hopper
elif platform.machine() == "x86_64":
    cxx_flags += ["-march=native"]
    nvcc_flags += ["-arch=sm_80"]   # A100 default
```

### Ninja Build (Faster Parallel Compilation)

```bash
pip install ninja  # parallel make replacement
```

torch cpp_extension auto-detects ninja. JIT `load()` uses it by default. For setup.py:
```python
cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)}
```

---

## Debugging Extensions

```bash
# Compile with debug symbols
python setup.py build_ext --inplace --debug

# Or JIT:
load(..., extra_cflags=["-g", "-O0"], extra_cuda_cflags=["-g", "-G", "-O0"])

# Debug with gdb
gdb -ex run --args python my_script.py

# CUDA debugging
compute-sanitizer python my_script.py       # memory errors
compute-sanitizer --tool racecheck python my_script.py  # race conditions
```

## Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| `undefined symbol` at import | Check all sources listed, libraries linked |
| JIT recompile every run | Check `~/.cache/torch_extensions/` permissions |
| Crash without error message | Add `TORCH_CHECK` / `CUDA_CHECK` to all inputs |
| Wrong results with `-O3 --use_fast_math` | `--use_fast_math` changes NaN/denorm handling — test numerics |
| `.so` built on x86 fails on ARM | Must rebuild on target architecture (or cross-compile) |
| CUDA version mismatch | `nvcc --version` must match PyTorch CUDA version |
| ABI compatibility | `torch.utils.cpp_extension.COMMON_NVCC_FLAGS` handles this |
| Slow first import (JIT) | Use ahead-of-time build for production |
