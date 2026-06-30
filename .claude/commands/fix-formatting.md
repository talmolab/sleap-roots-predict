---
description: Automatically fix formatting issues across the codebase
---

# Fix Formatting

Automatically reformat code to match black (line length 88). This mutates files in place;
run `/lint` separately for docstring/spelling issues.

## Command

```bash
uv run black sleap_roots_predict tests
```

## What Gets Fixed

black rewrites files to match its configured style. Typical changes include:

- Line wrapping / collapsing (88-char limit)
- Quote style normalization
- Trailing commas
- Indentation and bracket spacing
- Blank-line conventions

## What Is NOT Auto-Fixed

- Docstring / spelling issues — use `/lint` (ruff + codespell)
- Code structure or correctness

## After Running

### Review the diff

```bash
git diff
```

Confirm the changes are formatting-only. If something unexpected changed, check for a
pre-existing syntax error in the file — black rarely corrupts valid code.

### Verify tests still pass

```bash
uv run pytest -m "not gpu" tests/
```

### Commit formatting separately

```bash
git add -u
git commit -m "style: apply black"
```

## Common Scenarios

### CI reports a formatting failure

```bash
# Fix locally
uv run black sleap_roots_predict tests

# Verify the check now passes
uv run black --check sleap_roots_predict tests

# Commit and push
git add -u
git commit -m "style: apply black"
git push
```

### Format a subset of files

```bash
uv run black sleap_roots_predict/predict.py
```

## Comparison with /lint

| Command | Purpose | Modifies files? |
|---|---|---|
| `/fix-formatting` | Auto-fix style (black) | Yes |
| `/lint` | Check docstrings + spelling + format | No |
| `uv run black --check sleap_roots_predict tests` | Verify formatting without fixing | No |

**Recommended order:** `/fix-formatting` → `/lint` → commit.

## IDE Integration

Configure your editor to run black on save so files are formatted before you even run the
command above.

## Related Commands

- `/lint` — Check docstrings, spelling, and formatting
- `/pre-merge` — Full gate including format check
- `/ci-debug` — Debug a CI formatting failure
