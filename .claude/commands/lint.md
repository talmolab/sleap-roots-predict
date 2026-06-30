---
description: Run linting checks (check-only — use /fix-formatting to auto-fix style)
---

# Lint

Run code quality checks. This command is **check-only** — it reports issues without
modifying files. To auto-fix formatting, use `/fix-formatting`.

This repo has **no type checker** (no mypy/pyright). Lint = ruff (docstrings) + codespell,
plus a black format check.

## Commands

```bash
# Lint — ruff enforces google-style docstrings (select = ["D"])
uv run ruff check sleap_roots_predict/

# Spelling
uv run codespell

# Formatting check — reports violations without fixing (auto-fix with /fix-formatting)
uv run black --check sleap_roots_predict tests
```

Run all three before pushing; all must pass for CI to go green (they mirror the `lint`
job in `.github/workflows/ci.yml`).

## What Each Check Catches

### ruff — docstring quality

The repo configures only the pydocstyle (`D`) rule set with the google convention, so
ruff here primarily flags missing/malformed docstrings on public modules, classes, and
functions. Check `[tool.ruff.lint]` in `pyproject.toml`.

### codespell — spelling

Catches common misspellings in code and docs. Config (skip globs) lives under
`[tool.codespell]` in `pyproject.toml`.

### black --check — style consistency

Reports files whose formatting does not match black (line length 88). **Does not
auto-fix** — run `/fix-formatting` for that.

## Fixing Issues

### Auto-fixable style violations

```bash
/fix-formatting
```

### Lint errors that can be auto-fixed

Some ruff issues are auto-fixable: `uv run ruff check --fix sleap_roots_predict/`.
Missing docstrings must be written by hand.

## Common Patterns

### "Cannot find module" while running checks

Dependencies are not synced:

```bash
uv sync --extra dev --extra cpu
```

### ruff reports an unfamiliar rule

Check `[tool.ruff.lint]` in `pyproject.toml` (the `D###` codes are pydocstyle rules; the
google convention is configured under `[tool.ruff.lint.pydocstyle]`).

## Workflow Recommendations

```bash
# 1. Auto-fix style
/fix-formatting

# 2. Run quality checks
uv run ruff check sleap_roots_predict/
uv run codespell
uv run black --check sleap_roots_predict tests

# 3. Fix any remaining errors, then commit
```

## Related Commands

- `/fix-formatting` — Auto-fix style issues reported by black
- `/pre-merge` — Full gate: format-check + lint + test + build
- `/ci-debug` — Debug a failing CI run
- `/coverage` — Run tests with coverage
