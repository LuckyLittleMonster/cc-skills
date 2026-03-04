---
name: code-review
description: Code review a pull request or recent changes using git diff and agent teams. Reviews code usage, efficiency, bugs, and comments. Backs up unused files to bk/.
user-invocable: true
---

# Code Review Skill

Review recent code changes using git diff and parallel agent teams. Focuses on code quality, efficiency, bug detection, and comment hygiene.

## Workflow

### Step 1: Identify Changes

Use git to find all modified files between the last commit and the current working tree:

```bash
# Show changed files (staged + unstaged) vs last commit
git diff HEAD --name-only

# If no uncommitted changes, compare last two commits
git diff HEAD~1 HEAD --name-only
```

Group files by directory/module for organized review. Present the file list to the user before proceeding.

### Step 2: Launch Review Agent Team

Create a team of review agents to work in parallel. Each agent reviews one or more files.

For each file under review, the agent MUST check all four dimensions:

#### a. Code Usage
- Is the code still referenced/imported elsewhere in the project?
- Are there dead functions, unused imports, or orphaned classes?
- Use `Grep` to verify references across the codebase
- Flag anything that appears unused with evidence (no callers found)

#### b. Efficiency
- Follow the project's parallelization guidelines (see CLAUDE.md):
  - Python for-loops over large datasets → vectorize or batch
  - `tensor.cpu().numpy()` inside loops → accumulate on GPU, transfer once
  - `list.index(x)` for repeated lookups → use dict
  - Sequential `model(single_input)` → batch `model(batch_input)`
  - Single-threaded file processing → `mp.Pool` or `ThreadPoolExecutor`
- Check for unnecessary data copies, redundant computations, or O(n²) patterns
- Suggest concrete improvements with code snippets

#### c. Bug Detection
- Look for logic errors, off-by-one, race conditions, resource leaks
- Check edge cases: empty inputs, None values, division by zero
- Verify correct tensor shapes and device placement (CPU vs GPU)
- Check for common Python pitfalls (mutable default args, late binding closures)
- Flag security issues (command injection, path traversal) if applicable

#### d. Comment Quality
- Are comments accurate and up-to-date with the code?
- Are complex algorithms or non-obvious logic explained?
- Remove redundant comments that just restate the code (e.g., `x += 1  # increment x`)
- Ensure docstrings exist for public APIs and are not misleading
- Flag TODO/FIXME/HACK comments that should be resolved

### Step 3: File Cleanup and Backup

For files identified as unused or obsolete:

1. **Confirm with user** before moving any file
2. Create backup directory structure mirroring the original:
   ```bash
   # Example: backing up scripts/old_script.py
   mkdir -p bk/scripts/
   mv scripts/old_script.py bk/scripts/old_script.py
   ```
3. Never delete files outright — always move to `bk/` first
4. Update any imports or references if a moved file was partially used

### Step 4: Report

Each review agent reports findings in this format:

```
## File: <path>
### Usage: [USED / PARTIALLY USED / UNUSED]
- Evidence: <grep results or caller list>

### Efficiency: [OK / NEEDS IMPROVEMENT]
- Issue: <description>
- Suggestion: <concrete fix>

### Bugs: [NONE FOUND / <count> ISSUES]
- [SEVERITY: HIGH/MEDIUM/LOW] <description>

### Comments: [OK / NEEDS CLEANUP]
- <specific comment issues>
```

## Important Rules

1. **Do NOT commit or push** — the user handles git operations themselves
2. **Ask when uncertain** — if a file's usage or correctness is ambiguous, report it to the user rather than guessing
3. **Preserve, don't destroy** — always backup to `bk/` before removing anything
4. **Evidence-based** — every "unused" or "bug" claim must include grep/search evidence
5. **Respect scope** — only review files that appear in the git diff, unless the user requests a broader review
6. **Be conservative** — when in doubt, leave code as-is and flag for human review
