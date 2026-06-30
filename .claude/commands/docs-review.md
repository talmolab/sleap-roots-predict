---
description: Review and update project documentation for accuracy, completeness, and consistency
---

# Review & Update Documentation

Systematic workflow for reviewing and updating project documentation to ensure accuracy
and completeness.

## Quick Commands

Prefer the Claude Code Grep/Glob tools over shell `find`/`grep` on this (often Windows) repo.

```bash
# Find all documentation files
git ls-files '*.md'

# Search for TODO/FIXME/TBD in docs (or use Grep)
git grep -n "TODO\|FIXME\|TBD" -- '*.md'
```

## Documentation Review Checklist

### 1. Core Documentation Files

This repo's docs (check each that the change touches):

- [ ] **README.md** — project overview, install extras, usage, architecture
- [ ] **CLAUDE.md** — Claude-specific project instructions (and the OpenSpec managed block)
- [ ] **API.md** — public API reference
- [ ] **CHANGELOG.md** — release history, if/once present (see `/update-changelog`)

### 2. Module Documentation

- [ ] Google-style docstrings on public functions in `predict.py`, `video_utils.py`,
      `plates_timelapse_experiment.py` (ruff `D` enforces these)
- [ ] The public-API list in `sleap_roots_predict/__init__.py` matches what the docs claim

### 3. OpenSpec Documentation

- [ ] **openspec/project.md** — project context and decisions (incl. the A3-predict scope note)
- [ ] **openspec/changes/\*/proposal.md** — all active proposals
- [ ] **openspec/changes/\*/design.md** — implementation documentation

## Documentation Update Workflow

### Step 1: Identify What Changed

```bash
git log --oneline -10
git diff main...HEAD --stat
# Find docs mentioning a changed symbol (or use Grep)
git grep -n "process_timelapse_experiment" -- '*.md'
```

### Step 2: Update Affected Documentation

1. **README.md** — if install extras, architecture, or public features changed
2. **API.md** — if the public API or usage changed
3. **CLAUDE.md** — if the dev workflow or toolchain changed
4. **openspec/project.md** — if conventions or decisions changed

### Step 3: Check for Accuracy

- [ ] Commands still work — copy-paste and run `uv sync --extra dev --extra cpu`,
      `uv run pytest -m "not gpu" tests/`, `uv build`
- [ ] Code examples run against the current API
- [ ] File paths and the function lists in README/API.md are correct
- [ ] Hardware-extra install commands (`cpu` / `windows_cuda` / `linux_cuda` / `macos`) accurate
- [ ] Links work (no 404s)

### Step 4: Check for Completeness

- [ ] Installation/setup instructions (with the right hardware extra)
- [ ] Common workflows (timelapse experiment processing, batch prediction)
- [ ] Output artifact formats documented (labels/.slp, metadata CSV, H5)
- [ ] Troubleshooting common issues (device selection, missing models)

### Step 5: Verify Consistency

- [ ] Terminology uniform across README / API.md / CLAUDE.md / project.md
- [ ] The same fact (version, function list, artifact format) is not duplicated and drifting
- [ ] Tone consistent (technical, concise)

## Common Documentation Issues

### Outdated Setup Instructions

Re-run setup on a clean env and update steps to match `uv sync --extra dev --extra cpu`
and the correct hardware extra. Update prerequisites and Python version (>= 3.11).

### Missing New Features

Add new public functions to README/API.md with a usage example; update architecture notes
if the processing flow changed.

### Broken Code Examples

Test each example via `uv run pytest -m "not gpu" tests/` / a quick `uv run python` snippet;
update to the current API.

### Dead Links

Find broken links (Grep), update or remove them, prefer relative paths for internal links.

## What to Document / What Not To

**Do document:** install + hardware extras, common workflows, public API / artifact-format
changes, breaking changes with migration paths, troubleshooting.

**Do not document:** implementation details (use code comments/docstrings), temporary
workarounds, self-evident behavior.

## Completeness Criteria

- [ ] A new contributor can get started using only the docs
- [ ] All code examples work when copy-pasted
- [ ] Output artifact formats are documented
- [ ] Breaking changes noted with migration paths
- [ ] All links work; no `TODO`/`TBD`/`FIXME` remain in docs

## Related Commands

- `/lint` — docstring checks (ruff `D`) + spelling (codespell)
- `/review-pr` — PR review includes a documentation check
- `/openspec:proposal` — create formal documentation for new capabilities
