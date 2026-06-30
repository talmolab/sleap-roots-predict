---
description: Verify the development environment is correctly set up
---

# Validate Development Environment

Check that the dev environment is correctly set up. Run after cloning, after dependency
changes, when imports or tests fail unexpectedly, or after switching machines.

## Checks

```bash
# 1. uv is installed
uv --version

# 2. Python matches the project requirement (requires-python >= 3.11; CI uses 3.11/3.12)
uv run python --version

# 3. Sync dependencies from the lockfile (choose the extra for your hardware)
uv sync --extra dev --extra cpu          # CPU-only
# uv sync --extra dev --extra windows_cuda   # Windows + CUDA
# uv sync --extra dev --extra linux_cuda     # Linux + CUDA
# uv sync --extra dev --extra macos          # macOS (MPS)

# 4. Dependency tree resolves cleanly
uv tree | head -30

# 5. Import smoke test — the package and its inference stack import
uv run python -c "import sleap_roots_predict; print('OK', sleap_roots_predict.__version__)"
uv run python -c "import torch, sleap_nn, sleap_io; print('torch', torch.__version__)"

# 6. Tests run (CPU subset)
uv run pytest -m "not gpu" tests/ -q
```

## Device check (optional)

To confirm GPU/MPS acceleration is wired up on this machine:

```bash
uv run python -c "import torch; print('cuda', torch.cuda.is_available(), '| mps', torch.backends.mps.is_available())"
```

## Common fixes

| Symptom | Fix |
|---------|-----|
| `uv` not found | Install uv (`https://docs.astral.sh/uv/`) |
| Wrong Python version | `uv python install 3.12` (or use pyenv/mise) |
| Import errors after pull | `uv sync --extra dev --extra cpu` |
| Lock file conflict | Delete `.venv/` and re-run `uv sync --extra dev --extra cpu` |
| `VIRTUAL_ENV` mismatch warning | Harmless if tests still pass — uv uses the project `.venv` |
| `torch`/`sleap_nn` import fails | You likely synced the wrong hardware extra — re-sync with the one matching your machine |

## Related commands

- `/run-ci-locally` — run the full CI gate after the environment is confirmed healthy
- `/test` — run the test suite
