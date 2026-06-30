---
description: Debug a failing GitHub Actions CI run for this repo
---

# CI Debug

Diagnose and fix a failing CI run. Resolve the repo dynamically first so nothing is
hardcoded:

```bash
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
```

## Quick diagnosis

### Step 1: Identify the failing run and job

```bash
# List recent runs on the current branch
gh run list --repo "$REPO" --branch $(git branch --show-current) --limit 5

# View a specific run (job summary + step status)
gh run view <run-id> --repo "$REPO"

# View only the failed steps (most useful first pass)
gh run view <run-id> --repo "$REPO" --log-failed
```

### Step 2: Map the failure to a job

CI (`ci.yml`) has two jobs: **lint** (black/ruff/codespell) and **tests** (matrix:
ubuntu/windows/mac/self-hosted-gpu). A failure on `self-hosted-gpu` or `mac` may be a
real GPU/MPS issue you can't reproduce on a CPU machine — note that before chasing it.
The **docker-build** workflow (GHCR image) and **publish** workflow (PyPI) run on push to
`main` / release, not on PRs.

If the error is truncated in the CLI output, open the run URL and expand the failed step
in the browser.

### Step 3: Reproduce locally

Run `/run-ci-locally`, which mirrors the lint + tests jobs. Or run the specific failing step:

| Job / step | Local repro |
|------------|-------------|
| Format check | `uv run black --check sleap_roots_predict tests` |
| Ruff | `uv run ruff check sleap_roots_predict/` |
| Codespell | `uv run codespell` |
| Tests | `uv run pytest -m "not gpu" tests/` |
| Docker build | `docker build -t sleap-roots-predict .` |

## Common failure patterns and fixes

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Format check failed | black violations | Run `/fix-formatting` |
| Ruff failed | missing/bad docstrings | Run `uv run ruff check sleap_roots_predict/`; write docstrings |
| Codespell failed | misspelling | Fix it, or extend `[tool.codespell] skip` for a false positive |
| Tests failed | unit/integration failures | Run `uv run pytest -m "not gpu" tests/`; use `/tdd` |
| GPU test failed | device-only path | Reproduce on a CUDA/MPS machine; consider `@pytest.mark.gpu` if it shouldn't run on CPU |
| Install / lock out of sync | stale `uv.lock` | `uv sync --extra dev --extra cpu` and commit `uv.lock` |
| Docker build failed | Dockerfile / dep issue | `docker build -t sleap-roots-predict .` locally; scroll up past "exit code 1" |
| Platform-specific failure | path separators, case sensitivity | Test on the failing platform; use `pathlib` |
| Timeout | job exceeds limit | Identify slow tests; mock heavy I/O; shrink fixtures |
| Missing env / runner offline | self-hosted GPU runner down | Check the runner is online before assuming a code bug |

## Advanced: download logs

```bash
gh run download <run-id> --repo "$REPO" --dir ./ci-logs-<run-id>
ls ./ci-logs-<run-id>/
```

## Re-run a failed job

```bash
gh run rerun <run-id> --repo "$REPO" --failed
gh run watch --repo "$REPO"
```

## Cross-cutting issues

### Lock file out of sync

```bash
uv sync --extra dev --extra cpu
git add uv.lock
git commit -m "chore: update lock file"
```

### Passes locally but fails in CI

1. Runtime version: CI pins Python 3.11 (lint) / 3.12 (tests). Verify with `uv python list`.
2. Dependencies: run `uv sync` with the matching hardware extra (CI uses `cpu` on
   ubuntu/windows, `macos` on mac, `linux_cuda` on the GPU runner).
3. Env vars: CI does not read `.env`.
4. Platform: Linux is case-sensitive; macOS/Windows are not.

## GitHub Actions status

```bash
# Is main currently green?
gh run list --repo "$REPO" --branch main --limit 3
```

If CI fails in a way unrelated to your code, check https://www.githubstatus.com/.

## Related commands

- `/run-ci-locally` — reproduce the full CI gate before pushing
- `/lint` — lint checks only
- `/coverage` — tests with coverage detail
- `/pre-merge` — full pre-merge gate
- `/fix-formatting` — auto-fix format violations
