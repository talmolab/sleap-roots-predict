---
description: Run the pytest suite (GPU tests are marked and deselected by default)
---

# Run Tests

Run the repo's pytest suite. GPU-dependent tests are marked `@pytest.mark.gpu` and skipped
on machines without a GPU.

## Commands

```bash
# Default — skip GPU tests (matches CI on non-GPU runners)
uv run pytest -m "not gpu" tests/

# Full suite including GPU tests (run on a CUDA machine / Apple MPS)
uv run pytest tests/

# Only the GPU tests
uv run pytest -m gpu tests/

# A single module / test
uv run pytest tests/test_video_utils.py -v
uv run pytest tests/test_predict.py::test_make_predictor -v
```

> This repo has no watch mode (pytest-watch is not a dependency). Re-run the command above,
> or scope it to one file for a fast loop.

## Test Layout

Tests are **centralized** under `tests/`, mirroring the package:

- `tests/test_predict.py` — prediction interface
- `tests/test_video_utils.py` — image I/O utilities
- `tests/conftest.py` — shared fixtures (RGB / greyscale / RGBA / large-image, unicode paths)

When adding a new source module `sleap_roots_predict/foo.py`, add `tests/test_foo.py`.

## Test Environment

- Config lives in `[tool.pytest.ini_options]` in `pyproject.toml` (`testpaths = ["tests"]`,
  the `gpu` marker).
- Tests must be deterministic — mock wall-clock time, randomness, and external filesystem
  writes; use the tmp-path/fixture patterns already in `conftest.py`.

## What to Do After Running

1. **Fix failing tests** — investigate rather than weakening assertions.
2. **Add tests for new code** — new functions in `predict.py` / `video_utils.py` /
   `plates_timelapse_experiment.py` should have at least a happy-path test, and mark any
   that need a GPU with `@pytest.mark.gpu`.
3. **Watch for flakiness** — image-loading tests should use the in-repo fixtures, not
   external files.

## Common Issues

### A test needs a GPU but ran on CPU

Mark it `@pytest.mark.gpu` so it is deselected by `-m "not gpu"`; CI runs the full suite
only on the self-hosted-gpu and macOS (MPS) runners.

### Test file not picked up

Files must live under `tests/` and match pytest's default `test_*.py` discovery.

## TDD Workflow

For new features and bug fixes, prefer the test-first cycle — see `/tdd`. The
`superpowers:test-driven-development` skill formalizes it.

## Related Commands

- `/lint` — docstrings + spelling + format check
- `/coverage` — tests with coverage reporting
- `/tdd` — structured red-green-refactor loop
- `/pre-merge` — full gate before opening a PR
