# torch.compile Optimization

torch.compile is the first tool to try when a PyTorch function is a performance bottleneck. It fuses kernels, eliminates Python overhead, and its generated Triton code serves as the starting point for hand-written Triton kernels.

---

## When to Use

```
Performance bottleneck identified (via profiling)
│
├─ Many small CUDA kernels? → torch.compile (fuses them)
├─ Python overhead between ops? → torch.compile (eliminates it)
├─ Custom loss / attention / activation? → torch.compile (auto-optimizes)
├─ Repeated identical computation? → torch.compile + CUDAGraphs
│
└─ After compile: still not fast enough?
    ├─ Read the generated Triton code (see Section 4)
    ├─ Use it as starting point for hand-tuned Triton kernel
    └─ Only then consider CUDA C++ if Triton is insufficient
```

**Escalation path:**
```
torch ops (baseline) → torch.compile → read generated Triton → hand-tune Triton → CUDA C++
```

## 1. Compilation Modes

```python
# Default: good balance of compile time and speedup
compiled_fn = torch.compile(fn)

# reduce-overhead: wraps in CUDAGraphs for minimal launch overhead
# Best for: fixed-shape training loops, repeated inference
compiled_fn = torch.compile(fn, mode="reduce-overhead")

# max-autotune: tries many Triton configs, picks fastest
# Best for: production deployment (slow compile, best runtime)
compiled_fn = torch.compile(fn, mode="max-autotune")
```

| Mode | Compile Time | Runtime Speed | Best For |
|------|-------------|--------------|----------|
| `default` | Fast | Good | Development, first try |
| `reduce-overhead` | Medium | Better | Fixed-shape training loops |
| `max-autotune` | Slow (minutes) | Best | Production, inference |

### Applying to Models

```python
# Compile entire model
model = torch.compile(model)

# Compile specific methods
model.forward = torch.compile(model.forward)

# Compile with options
model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
```

## 2. Graph Breaks (The Main Problem)

Graph breaks force torch.compile to split the computation into multiple subgraphs, reducing optimization opportunity.

### Common Causes and Fixes

```python
# BAD: data-dependent control flow → graph break
def forward(self, x):
    if x.sum() > 0:       # ← value not known at compile time
        return self.path_a(x)
    return self.path_b(x)

# GOOD: use torch.where (no graph break)
def forward(self, x):
    a = self.path_a(x)
    b = self.path_b(x)
    return torch.where(x.sum() > 0, a, b)
```

```python
# BAD: .item() / .tolist() → graph break (CPU sync)
loss_val = loss.item()
print(f"Loss: {loss_val}")

# GOOD: log outside compiled region
# Don't put print/logging inside compiled functions
```

```python
# BAD: Python built-in on tensor → graph break
n = int(tensor.shape[0])   # OK (shape is static)
n = int(tensor.sum())      # BAD (value is dynamic)

# BAD: unsupported Python constructs
for item in tensor:         # iterating over tensor
    ...

# GOOD: vectorized operations
result = tensor.sum(dim=0)
```

```python
# BAD: calling non-compilable external library
from rdkit import Chem
mol = Chem.MolFromSmiles(smi)  # ← not a torch op, graph break

# GOOD: keep non-torch code outside compiled function
# Compile only the tensor-heavy parts
```

### Detecting Graph Breaks

```bash
# Method 1: TORCH_LOGS
TORCH_LOGS="graph_breaks" python train.py

# Method 2: fullgraph=True (raises error on any break)
model = torch.compile(model, fullgraph=True)  # crash if graph break exists

# Method 3: Explain mode (dry run, report all issues)
torch._dynamo.explain(fn)(example_input)
```

### Graph Break Diagnosis Workflow

```
1. Compile with fullgraph=True → see where it fails
2. Read the error message → identifies the breaking line
3. Fix the break (see patterns above)
4. Repeat until fullgraph=True works (or accept partial compilation)
```

## 3. Dynamic Shapes

```python
# Static shapes: compile once, reuse (fastest)
# Dynamic shapes: recompile on new shapes (overhead)

# Tell compiler which dims are dynamic
compiled_fn = torch.compile(fn, dynamic=True)   # all dims dynamic
# or use torch._dynamo.mark_dynamic(tensor, dim) for specific dims

# For training with fixed batch size: dynamic=False (default) is fine
# For inference with variable input: dynamic=True needed
```

**Recompilation warning signs:**
```
# If you see repeated "Recompiling because ..." in logs:
TORCH_LOGS="recompiles" python script.py

# Common cause: changing tensor shapes triggers recompilation
# Fix: pad inputs to fixed size, or use dynamic=True
```

## 4. Reading Generated Triton Code

This is the bridge from torch.compile to hand-tuned Triton.

```bash
# Step 1: Generate code with debug output
TORCH_COMPILE_DEBUG=1 python script.py

# Output directory: /tmp/torchinductor_<user>/<hash>/
# Key file: output_code.py — contains generated Triton kernels
```

```bash
# Step 2: Find the generated files
ls /tmp/torchinductor_$(whoami)/

# Or set a known output directory:
import torch._inductor.config
torch._inductor.config.debug = True
torch._inductor.config.trace.enabled = True
```

**What to learn from generated code:**
- Block sizes chosen by autotuner → use as starting point
- How operations are fused → understand fusion opportunities
- Reduction tiling strategy → apply to your Triton kernel
- Numeric stability tricks (e.g., max subtraction before exp)
- Memory access patterns → coalescing strategy

```python
# Step 3: Extract the kernel, modify for your needs
# Generated code looks like:
@triton.jit
def triton_fused_softmax_0(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK, RBLOCK):
    # ... auto-generated kernel code ...
    # Study this, then write your optimized version
```

## 5. Common Patterns

### Compile Training Loop

```python
model = torch.compile(model)
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(num_epochs):
    for batch in dataloader:
        loss = model(batch)          # compiled forward
        loss.backward()              # compiled backward (auto)
        optimizer.step()
        optimizer.zero_grad()
        # NOTE: first iteration is slow (compilation)
        # Subsequent iterations are fast
```

### Compile Only the Bottleneck

```python
# Don't compile everything — compile the hot path
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ...
        self.decoder = ...
        # Compile only the expensive part
        self.attention = torch.compile(CustomAttention())

    def forward(self, x):
        x = self.encoder(x)          # not compiled (has graph breaks)
        x = self.attention(x)         # compiled (pure tensor ops)
        return self.decoder(x)        # not compiled
```

### Compile + Profiling

```python
# Profile to verify compilation helped
with torch.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
) as prof:
    for i, batch in enumerate(dataloader):
        if i >= 5:  # skip warmup, profile steady state
            prof.step()
        output = compiled_model(batch)

# Check: fewer kernel launches? Fused kernels? Less Python overhead?
prof.export_chrome_trace("compiled_trace.json")
```

## 6. Limitations

| Limitation | Workaround |
|---|---|
| First call slow (compilation) | Warmup before timing/production |
| Data-dependent control flow | torch.where, torch.cond |
| Non-torch Python code | Keep outside compiled region |
| Dynamic shapes = recompilation | Pad to fixed sizes or dynamic=True |
| Not all ops supported | Check torch._dynamo.explain() |
| Hard to debug compiled code | TORCH_LOGS="graph_breaks,recompiles" |
| Memory overhead from CUDAGraphs (reduce-overhead mode) | Use default mode if OOM |

## Checklist

```
□ Profiled first to confirm this IS the bottleneck?
□ Started with default mode before trying max-autotune?
□ Checked for graph breaks with fullgraph=True or TORCH_LOGS?
□ Fixed data-dependent control flow (if → torch.where)?
□ No .item()/.cpu()/.numpy() inside compiled function?
□ No external library calls (RDKit, etc.) inside compiled function?
□ Verified speedup with profiler (not just wall clock)?
□ If still slow → read generated Triton code for next step?
```
