# cc-skills

Claude Code skills collection for HPC development.

## Setup (new machine)

```bash
# Back up existing skills if any
[ -d ~/.claude/skills ] && mv ~/.claude/skills ~/.claude/skills.bak

# Clone directly as ~/.claude/skills
git clone https://github.com/LuckyLittleMonster/cc-skills.git ~/.claude/skills
```

## Update

```bash
cd ~/.claude/skills && git pull
```

## Skills

| Skill | Description |
|-------|-------------|
| `hpc-guide` | HPC core mental models and hardware-first reasoning |
| `hpc-python` | Python HPC patterns (threading, CUDA streams, DDP) |
| `hpc-cuda` | CUDA kernel development and GPU profiling |
| `hpc-cpp` | High-performance C++ and compiler optimization |
| `hpc-triton` | Triton kernel development |
| `hpc-infra` | NCCL, network fabrics, GPU topology |
| `hpc-review` | HPC-specific code performance review |
| `maple-guide` | Maple GH200 cluster reference |
| `maple-status` | Maple cluster resource status |
| `maple-run` | Smart SLURM task execution |
| `code-review` | General code review |
