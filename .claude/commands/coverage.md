---
description: Run pytest with coverage analysis to identify untested code
---

# Test Coverage Analysis

Run the test suite with coverage instrumentation to identify untested code.

## Command

```bash
uv run pytest --cov=sleap_roots_predict --cov-report=term-missing -m "not gpu" tests/
```

For an HTML report:

```bash
uv run pytest --cov=sleap_roots_predict --cov-report=html -m "not gpu" tests/
# open htmlcov/index.html
```

CI runs `--cov-report=xml` on CPU only (no GPU runner). The `gpu`-marked tests are excluded
from CI coverage; run them locally on a CUDA/MPS machine (`uv run pytest -m gpu`, a required
`/pre-merge` step) for coverage of the device paths.

## Understanding the Output

`--cov-report=term-missing` prints a per-file table plus the specific line numbers that
were never executed. The standard dimensions are statements, branches (if enabled),
and lines.

> This repo does **not** configure a hard coverage threshold (`--cov-fail-under`), so a
> coverage run does not fail the build on a low number. Treat the report as guidance: keep
> or improve coverage in the module you are touching.

## Interpreting Gaps

### Uncovered lines in a touched module

Open the `term-missing` output (or `htmlcov/`) and write a test exercising the missing
lines — prioritize completely-uncovered functions first.

### GPU-only paths

Code reachable only with a GPU will show as uncovered in a CPU run. Verify those paths
with `-m gpu` on a CUDA/MPS machine rather than mocking the whole device stack.

### Genuinely untestable lines

For lines that cannot be unit-tested (e.g. a hardware-device branch), prefer a focused
`# pragma: no cover` comment over a fragile mock.

## Related Commands

- `/test` — Run tests without coverage (faster local iteration)
- `/lint` — Check docstrings/spelling before committing
- `/pre-merge` — Full gate: format-check + lint + test + build
