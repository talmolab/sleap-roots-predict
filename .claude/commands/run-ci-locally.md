---
description: Reproduce the CI gate locally by reading the repo's actual workflows
---

# Run CI Locally

Reproduce the CI gate locally before pushing. **This command derives its job list from
`.github/workflows/` — confirm against the actual files before relying on the sweep below.**

## Step 0: Read the repo's CI workflows

```bash
ls .github/workflows/
cat .github/workflows/ci.yml
```

`ci.yml` (as of writing) defines two jobs, neither with `continue-on-error`:

- **lint** (ubuntu): `black --check`, `ruff check`, `codespell`
- **tests** (matrix: ubuntu, windows, mac — Python 3.12, all CPU): `uv sync` with the platform
  extra, an import/device smoke check, then `pytest` with coverage. GPU-marked tests are **not**
  run in CI (no GPU runner) — they are a required local step in `/pre-merge`.

`publish.yml` (PyPI) and `docker-build.yml` (GHCR image) run on release / push to `main`,
not on PRs — they are not part of the PR green gate.

## Standard local sweep (mirrors the two CI jobs)

Run in order, stopping on the first failure:

```bash
# --- lint job ---
# 1. Format check
uv run black --check sleap_roots_predict tests

# 2. Lint (docstrings)
uv run ruff check sleap_roots_predict/

# 3. Spelling
uv run codespell

# --- tests job (CPU equivalent) ---
# 4. Tests, skipping GPU-marked tests (as non-GPU runners do)
uv run pytest --cov=sleap_roots_predict --cov-report=term-missing -m "not gpu" tests/
```

### Annotated run (recommended output format)

```
[1/4] Format check...   PASSED
[2/4] Ruff...           PASSED
[3/4] Codespell...      PASSED
[4/4] Tests...          PASSED

ALL CI CHECKS PASSED
```

On failure (hard stop), print the failing step's output and the fix.

## Quick fixes

| Step | Fix |
|------|-----|
| Format check | Run `/fix-formatting` (or `uv run black sleap_roots_predict tests`) |
| Ruff | Run `uv run ruff check sleap_roots_predict/`; auto-fix with `--fix`, write missing docstrings by hand |
| Codespell | Fix the flagged spelling, or extend `[tool.codespell] skip` if a false positive |
| Tests | Read pytest output and fix; use `/tdd` for a structured loop |

## Dependencies and environment

If steps fail with import errors:

```bash
uv sync --extra dev --extra cpu
```

CI pins **Python 3.11** (lint job) / **3.12** (tests). Verify locally with `uv python list`.

## Platform matrix

CI runs on ubuntu, windows, and macOS (all CPU; no GPU runner). A local CPU run only
covers your current platform. Common cross-platform pitfalls:

- Path separators — use `pathlib` / `os.path.join`, not string literals (image-directory
  globbing in `video_utils.py` is the main risk surface)
- File-system case sensitivity — Linux is case-sensitive; macOS/Windows are not
- Device availability — GPU/MPS-only code paths won't execute on a CPU runner

## When to use

- Before every `git push`
- Before creating a PR
- Whenever you want confidence the PR will pass CI

## Related commands

- `/lint` — lint checks only
- `/coverage` — tests with coverage detail
- `/pre-merge` — full pre-merge gate (wraps this command)
- `/tdd` — structured red-green-refactor loop
- `/ci-debug` — diagnose a CI run that already failed on GitHub

## OpenSpec check

If a branch has an active OpenSpec change, validate it before pushing:

```bash
openspec validate --all --strict
```

A validation failure here means a reviewer running `/review-openspec` will also flag it.
