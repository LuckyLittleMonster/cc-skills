# cc-skills

Claude Code skills collection for HPC development.

## Usage

Clone and symlink to `~/.claude/skills/`:

```bash
git clone https://github.com/LuckyLittleMonster/cc-skills.git ~/cc-skills

# Back up existing skills if any
mv ~/.claude/skills ~/.claude/skills.bak

# Symlink
ln -sf ~/cc-skills ~/.claude/skills
```

## Update

```bash
cd ~/cc-skills && git pull
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
