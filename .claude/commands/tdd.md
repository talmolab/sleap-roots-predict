---
description: Test-driven development workflow — red → green → refactor
---

# Test-Driven Development (TDD)

Structured TDD workflow: write tests first, then implement the minimum code to pass them,
then refactor.

## Purpose

Writing tests before code ensures:

1. Requirements are captured as executable tests before implementation
2. Edge cases are considered upfront (empty input, malformed filenames, RGBA/greyscale, large images)
3. Image/array computations have known-answer fixtures that can be verified independently
4. Regressions are caught immediately

## TDD Cycle

### Phase 1: Red (Write Failing Tests)

Write tests that define the expected behavior before any implementation. Place them in
`tests/test_<module>.py`:

```python
# tests/test_video_utils.py
def test_natural_sort_orders_numerically():
    files = ["img_10.tif", "img_2.tif", "img_1.tif"]
    assert natural_sort(files) == ["img_1.tif", "img_2.tif", "img_10.tif"]

def test_convert_to_greyscale_empty_input():
    with pytest.raises(ValueError):
        convert_to_greyscale(np.empty((0, 0, 3)))
```

### Phase 2: Confirm Red

```bash
uv run pytest tests/test_video_utils.py -v
```

New tests should fail with an import/attribute/name error or an assertion — not an
unexpected crash. If they fail for the wrong reason, fix the test setup first.

### Phase 3: Green (Implement)

Write the minimum code to make all tests pass, then:

```bash
uv run pytest tests/test_video_utils.py -v
```

### Phase 4: Refactor

Improve structure while keeping tests green (extract helpers, add google-style docstrings,
clarify names). Re-run after each step:

```bash
uv run pytest tests/test_video_utils.py -v
```

### Phase 5: Verify Quality

```bash
# Format check
uv run black --check sleap_roots_predict tests

# Lint (docstrings + spelling)
uv run ruff check sleap_roots_predict/
uv run codespell

# Full suite (CPU)
uv run pytest -m "not gpu" tests/

# Coverage for the touched module
uv run pytest --cov=sleap_roots_predict --cov-report=term-missing -m "not gpu" tests/
```

### Phase 6: Commit

```bash
git add sleap_roots_predict/<module>.py tests/test_<module>.py
git commit -m "feat: add <feature description>

- Tests define expected behavior including edge cases
- Implementation satisfies all test cases"
```

## Testing Patterns

### Known-Answer Tests

For any computed array/value, assert against a hand-verified fixture, not just "it ran":

```python
def test_greyscale_uses_standard_weights():
    rgb = np.array([[[255, 0, 0]]], dtype=np.uint8)   # pure red
    grey = convert_to_greyscale(rgb)
    assert grey[0, 0] == pytest.approx(0.299 * 255, abs=1.0)
```

### Boundary Conditions

```python
def test_load_images_single_file()      # smallest valid input
def test_find_image_directories_empty() # no images → defined behavior
```

### Fixtures (shared setup)

Reuse the RGB/greyscale/RGBA/large-image fixtures already in `tests/conftest.py` rather
than constructing arrays inline.

### GPU-marked tests

Inference paths that need a device must be marked so they are deselected on CPU:

```python
@pytest.mark.gpu
def test_make_predictor_runs_on_cuda():
    ...
```

## Integration

- Run `/lint` during Phase 5
- Run `/coverage` to check the touched module
- Run `/run-ci-locally` before committing
- Run `/pre-merge` before opening a PR

## OpenSpec alignment

If this feature implements or changes a contract captured in an OpenSpec proposal, verify:

```bash
openspec validate <id> --strict
```

All tasks in `openspec/changes/<id>/tasks.md` that this feature closes should be checked
off before the commit.
