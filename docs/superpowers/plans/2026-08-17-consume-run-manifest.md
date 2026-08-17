# Consume RunManifest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `sleap-roots-predict` consume `sleap-roots-contracts`' new `RunManifest` to scope
`discover_scans` to exactly a run's `scan_keys`, and upgrade `run_batch`'s skip-if-done from a
plain `Path.exists()` check to a real idempotency-key comparison recomputed from artifacts
already on disk.

**Architecture:** All changes are confined to `sleap_roots_predict/batch.py` (plus its test file
and a version bump). `discover_scans` gains an optional manifest-read step ahead of its existing
`rglob`; `run_batch`'s per-scan loop gains two small private helpers
(`_identity_key`/`_previous_identity_key`) and is reordered so the existing `scan.error`
short-circuit and per-scan `try/except` still isolate every new failure mode.

**Tech Stack:** Python 3.11+, pydantic (via `sleap-roots-contracts`), pytest, `uv` for
dependency management.

**Spec:** `openspec/changes/consume-run-manifest/` (proposal.md, tasks.md,
`specs/predict-container/spec.md`) and `docs/superpowers/specs/2026-08-14-consume-run-manifest-design.md`
(design rationale — read this for *why*, this plan is the *how*). Both have already been through
a critical multi-lens review round; every fix from that round is folded into the tasks below.

## Global Constraints

- `sleap-roots-contracts` pin is exact (`==X.Y.Z`), not a range — `pyproject.toml:24`.
- The Dockerfile uses `uv sync --frozen` — a pin bump without a matching `uv.lock` relock in the
  **same commit** breaks the Docker build-validation job.
- Path strings that leave this process go through `Path.as_posix()` (lab convention) — not
  applicable to anything in this plan (no new path strings are emitted), noted for awareness only.
- Task Group 3 below (identity-key skip-if-done) **must land as a single commit** — its tests
  assert behavior that doesn't exist until the loop restructuring lands.
- Every new/changed test must run under `pytest -m "not gpu and not acceptance and not wandb and
  not parity"` (this repo's CI marker filter) with no network access and no GPU.

---

## Task 1: Bump sleap-roots-contracts to 0.1.0a7 (regression baseline)

**Files:**
- Modify: `pyproject.toml:24`
- Modify: `uv.lock` (regenerated, not hand-edited)

**Interfaces:**
- Produces: `RunManifest`, `RUN_MANIFEST_FILENAME` (from `sleap_roots_contracts`),
  `compute_idempotency_key` (from `sleap_roots_contracts.identity`) — all consumed by Tasks 2–6.

This task has no test to write (it's a dependency bump); its "test" is the existing full suite
staying green.

- [ ] **Step 1: Bump the pin**

In `pyproject.toml:24`, change:
```toml
    "sleap-roots-contracts==0.1.0a6",
```
to:
```toml
    "sleap-roots-contracts==0.1.0a7",
```

- [ ] **Step 2: Relock scoped to this dependency**

Run: `uv lock -P sleap-roots-contracts`

Confirm the `uv.lock` diff (`git diff uv.lock`) touches only the `sleap-roots-contracts` package
entry (version, hashes, and its own `pydantic`/`pyyaml` sub-dependencies if their versions moved)
— not an unrelated dependency. If anything else changed, stop and investigate before continuing.

- [ ] **Step 3: Verify the new symbols import cleanly**

Run:
```bash
uv run python -c "from sleap_roots_contracts import RunManifest, RUN_MANIFEST_FILENAME; from sleap_roots_contracts.identity import compute_idempotency_key; print('ok')"
```
Expected: prints `ok` with no error.

- [ ] **Step 4: Run the full existing suite as the regression baseline**

Run: `uv run pytest -m "not gpu and not acceptance and not wandb and not parity" tests/`

Expected: every test passes, identical to before the bump (no code changed yet — this proves the
version bump alone doesn't break anything before any new logic is introduced).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: bump sleap-roots-contracts to 0.1.0a7"
```

---

## Task 2: Scope discover_scans to a present run_manifest.json

**Files:**
- Modify: `sleap_roots_predict/batch.py` (imports, `discover_scans`)
- Test: `tests/test_batch.py`

**Interfaces:**
- Consumes: `RunManifest`, `RUN_MANIFEST_FILENAME` (Task 1)
- Produces: `discover_scans(input_dir)` now scopes to a manifest's `scan_keys` when
  `input_dir / RUN_MANIFEST_FILENAME` exists; unchanged (full unscoped `rglob`) otherwise. This
  is what Tasks 3–4 extend and what Task 6's `run_batch` continues to call unmodified.

- [ ] **Step 1: Write the failing test (scoped discovery)**

Add to `tests/test_batch.py`:
```python
def test_discover_scans_scopes_to_run_manifest(tmp_path: Path):
    _write_scan(tmp_path, "scan_1009", _RICE)
    _write_scan(tmp_path, "scan_1010", _RICE)  # leftover from a prior run, not in scope
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"pipeline_run_id": "run-1", "scan_keys": ["scan_1009"]})
    )
    scans = discover_scans(tmp_path)
    assert [s.scan_key for s in scans] == ["scan_1009"]
```
(`_write_scan` and `_RICE` already exist in this file — `_write_scan` at line ~14, `_RICE` at
line ~168. Python resolves them at call time, so referencing them from an earlier-defined test is
fine.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_batch.py::test_discover_scans_scopes_to_run_manifest -v`

Expected: FAIL — both `scan_1009` and `scan_1010` are returned (today's unscoped behavior).

- [ ] **Step 3: Write the minimal implementation**

In `sleap_roots_predict/batch.py`, add to the imports (near the existing
`from sleap_roots_contracts import ResolvedParams`):
```python
from sleap_roots_contracts import RUN_MANIFEST_FILENAME, ResolvedParams, RunManifest
```

Replace `discover_scans` (currently `batch.py:66-102`) with:
```python
def discover_scans(input_dir: str | Path) -> list[ScanInput]:
    """Discover scans under ``input_dir`` by their scan-metadata sidecars.

    Recursively finds ``*.scan_metadata.json`` files; each sidecar's parent
    directory holds that scan's frames. ``scan_key`` is the sidecar's filename
    stem and must equal the sidecar's internal ``scan_key``. Invalid scans are
    returned with ``.error`` set (isolated failure); a duplicate ``scan_key``
    anywhere in the tree raises.

    If ``input_dir / RUN_MANIFEST_FILENAME`` exists, discovery is scoped to
    exactly its ``scan_keys``: a discovered sidecar outside that set is
    silently excluded, and a listed ``scan_key`` with no matching sidecar is
    returned as an isolated error entry. Absent a manifest, every sidecar found
    is returned (unscoped), unchanged from before manifest-awareness existed.

    Args:
        input_dir: Directory of staged scans (must exist).

    Returns:
        One :class:`ScanInput` per discovered (or manifest-expected-but-missing)
        sidecar, sorted by path.

    Raises:
        FileNotFoundError: If ``input_dir`` does not exist (a mis-configured mount,
            distinct from an empty-but-present directory which is a no-op).
        ValueError: If two in-scope sidecars share a ``scan_key``.
        pydantic.ValidationError: If a present ``run_manifest.json`` fails to parse
            or validate as a :class:`RunManifest`.
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(
            f"input scan directory does not exist: {input_dir.as_posix()}"
        )

    manifest_path = input_dir / RUN_MANIFEST_FILENAME
    scoped_keys: set[str] | None = None
    if manifest_path.exists():
        manifest = RunManifest.model_validate_json(manifest_path.read_text())
        scoped_keys = set(manifest.scan_keys)

    scans: list[ScanInput] = []
    seen: dict[str, Path] = {}
    for sidecar in sorted(input_dir.rglob("*" + _SIDECAR_SUFFIX)):
        scan_key = sidecar.name[: -len(_SIDECAR_SUFFIX)]
        if scoped_keys is not None and scan_key not in scoped_keys:
            continue
        if scan_key in seen:
            raise ValueError(
                f"duplicate scan_key {scan_key!r}: "
                f"{seen[scan_key].as_posix()} and {sidecar.as_posix()}"
            )
        seen[scan_key] = sidecar
        scans.append(_load_scan(sidecar, scan_key))

    if scoped_keys is not None:
        for key in sorted(scoped_keys - set(seen)):
            scans.append(
                ScanInput(
                    key,
                    input_dir / f"{key}{_SIDECAR_SUFFIX}",
                    error=f"no sidecar found for manifest scan_key {key!r}",
                )
            )

    return scans
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_batch.py::test_discover_scans_scopes_to_run_manifest -v`

Expected: PASS.

- [ ] **Step 5: Add the no-manifest regression guard**

This isn't a new-behavior RED/GREEN cycle — it documents that the fallback path is intentional,
not incidental. Add:
```python
def test_no_manifest_falls_back_to_unscoped_discovery(tmp_path: Path):
    _write_scan(tmp_path, "scanA", _RICE)
    _write_scan(tmp_path, "scanB", _RICE)
    scans = discover_scans(tmp_path)
    assert sorted(s.scan_key for s in scans) == ["scanA", "scanB"]
```
Run: `uv run pytest tests/test_batch.py::test_no_manifest_falls_back_to_unscoped_discovery -v`

Expected: PASS immediately (this path was never changed — confirms no regression).

- [ ] **Step 6: Run the full existing discovery test suite**

Run: `uv run pytest tests/test_batch.py -k discover -v`

Expected: all pass, including every pre-existing discovery test (`test_discover_scans_reads_sidecar_and_frames`, `test_duplicate_scan_key_raises`, `test_stem_scan_key_mismatch_is_error`, etc.) with no changes needed to any of them.

- [ ] **Step 7: Commit**

```bash
git add sleap_roots_predict/batch.py tests/test_batch.py
git commit -m "feat: scope discover_scans to run_manifest.json when present"
```

---

## Task 3: A manifest scan_key with no sidecar becomes an isolated failed scan

**Files:**
- Modify: `tests/test_batch.py` (test only — Task 2's implementation already produces this)

**Interfaces:**
- Consumes: `discover_scans` (Task 2) — its synthetic-error-`ScanInput` behavior for a missing
  sidecar is already implemented; this task adds the test proving it end-to-end via `run_batch`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_batch.py`:
```python
def test_manifest_scan_key_with_no_sidecar_is_failed(all_roots_source, tmp_path: Path):
    inp = tmp_path / "in"
    _real_scan(inp, "scanGOOD", _RICE)
    (inp / "run_manifest.json").write_text(
        json.dumps(
            {"pipeline_run_id": "run-1", "scan_keys": ["scanGOOD", "scanMISSING"]}
        )
    )
    out = tmp_path / "out"
    result = run_batch(inp, out, source=all_roots_source)
    statuses = {s.scan_key: s.status for s in result.scans}
    assert statuses["scanGOOD"] == "ok"
    assert statuses["scanMISSING"] == "failed"
```
(`_real_scan` already exists at line ~171, writes a real 8-frame scan + sidecar.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_batch.py::test_manifest_scan_key_with_no_sidecar_is_failed -v`

Expected: FAIL — at this point `run_batch` hasn't been touched yet (Task 6 does that), so trace
the failure carefully. `discover_scans` already returns the synthetic error `ScanInput` for
`scanMISSING` (Task 2), and `run_batch`'s existing loop already checks `scan.error is not None`
first and records `failed` — so this test may in fact **already pass** at this point in the
plan, since Task 2 alone is sufficient to satisfy it. If it passes, that's expected (not a bug in
this plan) — proceed to Step 3 as a confirmation run instead of a RED step, and note this in the
commit message.

- [ ] **Step 3: Confirm passing and commit**

Run: `uv run pytest tests/test_batch.py::test_manifest_scan_key_with_no_sidecar_is_failed -v`

Expected: PASS.

```bash
git add tests/test_batch.py
git commit -m "test: manifest scan_key with no sidecar is an isolated failure"
```

---

## Task 4: A malformed run_manifest.json raises

**Files:**
- Modify: `tests/test_batch.py` (test only — Task 2's implementation already raises)

- [ ] **Step 1: Write the tests**

Add to `tests/test_batch.py`:
```python
def test_malformed_manifest_json_raises(tmp_path: Path):
    _write_scan(tmp_path, "scanA", _RICE)
    (tmp_path / "run_manifest.json").write_text("{not valid json")
    with pytest.raises(Exception):
        discover_scans(tmp_path)


def test_manifest_with_empty_scan_keys_raises(tmp_path: Path):
    _write_scan(tmp_path, "scanA", _RICE)
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"pipeline_run_id": "run-1", "scan_keys": []})
    )
    with pytest.raises(Exception):
        discover_scans(tmp_path)
```

- [ ] **Step 2: Run and confirm both pass**

Run: `uv run pytest tests/test_batch.py -k malformed_manifest or empty_scan_keys -v`

Expected: PASS — `RunManifest.model_validate_json` already raises `pydantic.ValidationError` on
invalid JSON, and `RunManifest`'s own `_check_scan_keys` validator already rejects an empty list;
Task 2's implementation doesn't catch either, so they propagate.

- [ ] **Step 3: Commit**

```bash
git add tests/test_batch.py
git commit -m "test: malformed run_manifest.json raises before any scan is processed"
```

---

## Task 5: images_checksum field + identity-key helpers

**Files:**
- Modify: `sleap_roots_predict/batch.py` (`ScanInput`, `_load_scan`, new helpers)
- Test: `tests/test_batch.py`

**Interfaces:**
- Consumes: `compute_idempotency_key` (`sleap_roots_contracts.identity`, Task 1),
  `compute_param_hash` (already exported from `sleap_roots_contracts` top-level),
  `PredictionManifest` (already imported in `output_contract.py`; import it into `batch.py` too).
- Produces:
  - `ScanInput.images_checksum: str` (new field, default `""`)
  - `_identity_key(*, scan_key: str, images_checksum: str, params_dict: dict, model_refs, predict_code_sha: str, predict_output_params: dict) -> str`
  - `_previous_identity_key(out_scan_dir: Path, scan_key: str) -> str | None`
  Both consumed by Task 6's `run_batch` restructuring.

- [ ] **Step 1: Add the field (no test needed standalone)**

In `sleap_roots_predict/batch.py`, modify the `ScanInput` dataclass (currently `batch.py:30-42`):
```python
@dataclass(frozen=True)
class ScanInput:
    """A discovered scan.

    ``error`` is set (and ``params``/``frames`` may be empty) when the sidecar is
    invalid — an isolated per-scan failure, not a batch abort.
    """

    scan_key: str
    sidecar_path: Path
    frames: list[Path] = field(default_factory=list)
    params: ResolvedParams | None = None
    images_checksum: str = ""
    error: str | None = None
```

In `_load_scan` (currently `batch.py:105-142`), after building `resolved`, read the checksum and
pass it through:
```python
    resolved = ResolvedParams(values={k: params[k] for k in _REQUIRED_PARAM_KEYS})
    images_checksum = meta.get("images_checksum", "")
    return ScanInput(
        scan_key, sidecar, frames=frames, params=resolved, images_checksum=images_checksum
    )
```

- [ ] **Step 2: Write the failing tests for the helpers**

Add to `tests/test_batch.py` (needs `import sleap_roots_predict.batch as batch_mod` — some
existing tests already do this, e.g. `test_run_batch_constructs_single_worker`; add the import
inside each test function that needs it, matching that existing style):
```python
def test_identity_key_changes_with_images_checksum():
    import sleap_roots_predict.batch as batch_mod
    from sleap_roots_contracts import ModelRef

    ref = ModelRef(registry_id="reg/x", version="v1", sleap_nn_version="0.3.0")
    base_kwargs = dict(
        scan_key="scanA",
        params_dict={"species": "rice", "mode": "cylinder", "age": 3},
        model_refs={"primary": ref},
        predict_code_sha="sha1",
        predict_output_params={"peak_threshold": 0.2},
    )
    key_a = batch_mod._identity_key(images_checksum="sha256:a", **base_kwargs)
    key_b = batch_mod._identity_key(images_checksum="sha256:b", **base_kwargs)
    assert key_a != key_b


def test_previous_identity_key_none_when_nothing_on_disk(tmp_path: Path):
    import sleap_roots_predict.batch as batch_mod

    assert batch_mod._previous_identity_key(tmp_path, "scanA") is None


def test_previous_identity_key_none_when_predictions_json_corrupt(tmp_path: Path):
    import sleap_roots_predict.batch as batch_mod

    (tmp_path / "scanA.scan_metadata.json").write_text(
        json.dumps(
            {"scan_key": "scanA", "images_checksum": "sha256:x", "params": _RICE}
        )
    )
    (tmp_path / "scanA.predictions.json").write_text('{"not": "a valid manifest"}')
    assert batch_mod._previous_identity_key(tmp_path, "scanA") is None
```

- [ ] **Step 3: Run to verify they fail**

Run: `uv run pytest tests/test_batch.py -k identity_key -v`

Expected: FAIL with `AttributeError: module 'sleap_roots_predict.batch' has no attribute
'_identity_key'` (or `_previous_identity_key`) — the helpers don't exist yet.

- [ ] **Step 4: Write the minimal implementation**

In `sleap_roots_predict/batch.py`, update the contracts import to also pull in
`compute_param_hash` and `PredictionManifest`, and add the submodule import:
```python
from sleap_roots_contracts import (
    RUN_MANIFEST_FILENAME,
    PredictionManifest,
    ResolvedParams,
    RunManifest,
    compute_param_hash,
)
from sleap_roots_contracts.identity import compute_idempotency_key
```

Add these two functions (near `_load_scan`, before `run_batch`):
```python
def _identity_key(
    *,
    scan_key: str,
    images_checksum: str,
    params_dict: dict,
    model_refs,
    predict_code_sha: str,
    predict_output_params: dict,
) -> str:
    """Derive the predict-scoped identity key for one scan's current or prior state.

    ``model_refs`` may be a ``dict[RootType, ModelRef]`` (as returned by
    ``worker.resolve``) or any iterable of ``ModelRef`` (as read back from a
    ``PredictionManifest``'s ``artifacts``) — both are reduced to the same
    order-independent ``(registry_id, version, weights_checksum)`` tuples
    ``compute_idempotency_key`` expects. ``traits_code_sha`` is a fixed empty-string
    placeholder: predict never owns that value and only ever compares this key
    against its own previously-derived one, never against a traits-computed key.
    """
    refs = model_refs.values() if isinstance(model_refs, dict) else model_refs
    models = [(ref.registry_id, ref.version, ref.weights_checksum) for ref in refs]
    return compute_idempotency_key(
        scan_key=scan_key,
        images_checksum=images_checksum,
        models=models,
        param_hash=compute_param_hash(params_dict),
        predict_code_sha=predict_code_sha,
        traits_code_sha="",
        predict_output_params=predict_output_params,
    )


def _previous_identity_key(out_scan_dir: Path, scan_key: str) -> str | None:
    """Recompute the previous run's identity key from artifacts already on disk.

    Reads back the already-copied sidecar and already-written prediction manifest
    from ``out_scan_dir`` — no new storage is needed. Returns ``None`` if either
    file is missing, unreadable, or present but fails to parse/validate (a
    :class:`pydantic.ValidationError`, which subclasses ``ValueError`` and is
    caught here without importing pydantic directly) — a corrupt or absent
    *previous* state is treated as "changed," never as a failure.
    """
    sidecar_path = out_scan_dir / f"{scan_key}{_SIDECAR_SUFFIX}"
    manifest_path = out_scan_dir / f"{scan_key}.predictions.json"
    try:
        meta = json.loads(sidecar_path.read_text())
        manifest = PredictionManifest.model_validate_json(manifest_path.read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    params = meta.get("params")
    if not isinstance(params, dict):
        return None
    return _identity_key(
        scan_key=scan_key,
        images_checksum=meta.get("images_checksum", ""),
        params_dict={k: params[k] for k in _REQUIRED_PARAM_KEYS if k in params},
        model_refs=[artifact.model for artifact in manifest.artifacts],
        predict_code_sha=manifest.predict_code_sha,
        predict_output_params=manifest.predict_output_params,
    )
```

- [ ] **Step 5: Run to verify they pass**

Run: `uv run pytest tests/test_batch.py -k identity_key -v`

Expected: PASS, all three.

- [ ] **Step 6: Commit**

```bash
git add sleap_roots_predict/batch.py tests/test_batch.py
git commit -m "feat: add images_checksum field and identity-key helpers"
```

---

## Task 6: Idempotency-key skip-if-done in run_batch

**Land this entire task as ONE commit** — the tests in Steps 1–5 assert behavior that doesn't
exist until Step 6's implementation lands; splitting would leave an intermediate commit red.

**Files:**
- Modify: `sleap_roots_predict/batch.py` (`run_batch`, `_predict_one`)
- Test: `tests/test_batch.py`

**Interfaces:**
- Consumes: `_identity_key`, `_previous_identity_key` (Task 5), `discover_scans` (Task 2,
  unmodified by this task)
- Produces: `run_batch`'s skip decision is now identity-key-based; `_predict_one`'s signature
  changes to accept a precomputed `refs` argument instead of resolving internally.

- [ ] **Step 1: Write the failing tests — changed inputs cause a re-predict**

Add to `tests/test_batch.py`:
```python
def test_changed_params_causes_repredict(all_roots_source, tmp_path: Path):
    inp = tmp_path / "in"
    _real_scan(inp, "scanA", _RICE)
    out = tmp_path / "out"
    run_batch(inp, out, source=all_roots_source)
    manifest = out / "scanA" / "scanA.predictions.json"
    mtime1 = manifest.stat().st_mtime_ns

    sidecar = inp / "scanA" / "scanA.scan_metadata.json"
    body = json.loads(sidecar.read_text())
    body["params"] = {"species": "rice", "mode": "cylinder", "age": 4}
    sidecar.write_text(json.dumps(body))

    result2 = run_batch(inp, out, source=all_roots_source)
    assert [s.status for s in result2.scans] == ["ok"]
    assert manifest.stat().st_mtime_ns != mtime1


def test_changed_images_checksum_causes_repredict(all_roots_source, tmp_path: Path):
    inp = tmp_path / "in"
    _real_scan(inp, "scanA", _RICE)
    out = tmp_path / "out"
    run_batch(inp, out, source=all_roots_source)
    manifest = out / "scanA" / "scanA.predictions.json"
    mtime1 = manifest.stat().st_mtime_ns

    sidecar = inp / "scanA" / "scanA.scan_metadata.json"
    body = json.loads(sidecar.read_text())
    body["images_checksum"] = "sha256:changed"
    sidecar.write_text(json.dumps(body))

    result2 = run_batch(inp, out, source=all_roots_source)
    assert [s.status for s in result2.scans] == ["ok"]
    assert manifest.stat().st_mtime_ns != mtime1


def test_changed_predict_code_sha_causes_repredict(all_roots_source, tmp_path, monkeypatch):
    inp = tmp_path / "in"
    _real_scan(inp, "scanA", _RICE)
    out = tmp_path / "out"
    monkeypatch.setenv("SRP_PREDICT_CODE_SHA", "sha-one")
    run_batch(inp, out, source=all_roots_source)
    manifest = out / "scanA" / "scanA.predictions.json"
    mtime1 = manifest.stat().st_mtime_ns

    monkeypatch.setenv("SRP_PREDICT_CODE_SHA", "sha-two")
    result2 = run_batch(inp, out, source=all_roots_source)
    assert [s.status for s in result2.scans] == ["ok"]
    assert manifest.stat().st_mtime_ns != mtime1


def test_changed_model_ref_causes_repredict(tmp_path: Path, native_model_dir):
    from sleap_roots_contracts import ModelCard
    from sleap_roots_predict.model_registry import LocalCardSource

    def _source(version):
        card = ModelCard(
            species="rice",
            mode="cylinder",
            age_min=2,
            age_max=5,
            root_type="primary",
            registry_id="reg/rice-primary",
            version=version,
        )
        return LocalCardSource([(card, native_model_dir)])

    inp = tmp_path / "in"
    _real_scan(inp, "scanA", _RICE)
    out = tmp_path / "out"
    run_batch(inp, out, source=_source("v1"))
    manifest = out / "scanA" / "scanA.predictions.json"
    mtime1 = manifest.stat().st_mtime_ns

    result2 = run_batch(inp, out, source=_source("v2"))
    assert [s.status for s in result2.scans] == ["ok"]
    assert manifest.stat().st_mtime_ns != mtime1


def test_corrupt_previous_manifest_causes_repredict_not_failure(all_roots_source, tmp_path):
    inp = tmp_path / "in"
    _real_scan(inp, "scanA", _RICE)
    out = tmp_path / "out"
    run_batch(inp, out, source=all_roots_source)
    (out / "scanA" / "scanA.predictions.json").write_text('{"not": "a valid manifest"}')

    result2 = run_batch(inp, out, source=all_roots_source)
    assert [s.status for s in result2.scans] == ["ok"]
```

- [ ] **Step 2: Write the failing test — batch isolation survives the loop restructuring**

```python
def test_manifest_missing_sidecar_does_not_abort_other_scans(all_roots_source, tmp_path):
    inp = tmp_path / "in"
    _real_scan(inp, "scanGOOD", _RICE)
    (inp / "run_manifest.json").write_text(
        json.dumps({"pipeline_run_id": "run-1", "scan_keys": ["scanGOOD", "scanMISSING"]})
    )
    out = tmp_path / "out"
    result = run_batch(inp, out, source=all_roots_source)
    statuses = {s.scan_key: s.status for s in result.scans}
    assert statuses == {"scanGOOD": "ok", "scanMISSING": "failed"}
```
This proves the `scan.error` short-circuit still runs before `resolve()` — a naive
"move `resolve()` up" implementation would call `resolve(None)` on `scanMISSING`'s
`params=None` and raise `AttributeError` inside `choose_models`, which (without the ordering
fix below) would abort the whole batch instead of isolating one scan.

- [ ] **Step 3: Run all new tests to verify they fail**

Run:
```bash
uv run pytest tests/test_batch.py -k "changed_params_causes_repredict or changed_images_checksum_causes_repredict or changed_predict_code_sha_causes_repredict or changed_model_ref_causes_repredict or corrupt_previous_manifest or manifest_missing_sidecar_does_not_abort" -v
```
Expected: every one FAILS — today's `run_batch` still skips on `Path.exists()` alone (the
`test_changed_*`/`test_corrupt_previous_manifest*` tests fail because the second run is
`skipped` instead of `ok`), and `test_manifest_missing_sidecar_does_not_abort_other_scans` may
already pass (Task 2/3 cover it) or may already fail depending on `resolve()`'s current position
— either is fine at this checkpoint; Step 6 makes all of them pass together.

- [ ] **Step 4: Also run the existing skip test to see its current (still-passing) baseline**

Run: `uv run pytest tests/test_batch.py::test_rerun_skips_completed_scan -v`

Expected: PASS (unchanged by this task — recorded here as a baseline to re-check in Step 8, not
because it needs editing).

- [ ] **Step 5: Verify `_predict_one`'s current behavior before changing its signature**

Read `sleap_roots_predict/batch.py:207-245` (`_predict_one`) — note it currently calls
`worker.resolve(scan.params)` internally at its top. Step 6 removes that internal call and takes
`refs` as a parameter instead.

- [ ] **Step 6: Implement the restructured run_batch loop**

Replace `run_batch` (currently `batch.py:145-205`, post-Task-2/5 line numbers will have shifted —
locate by function name) with:
```python
def run_batch(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    source: ModelCardSource | None = None,
    predict_code_sha: str | None = None,
    predict_container_digest: str | None = None,
) -> BatchResult:
    """Predict every scan under ``input_dir``, writing outputs under ``output_dir``.

    Loads models once via a single resident worker. Per scan: a scan already
    recorded as invalid (``scan.error`` set) is isolated as ``failed`` immediately,
    before any model resolution runs. Otherwise, the currently-resolved models plus
    the scan's current inputs are compared, via a recomputed idempotency key,
    against the key recomputed from that scan's own previously-written artifacts
    (no new storage — see :func:`_previous_identity_key`); an exact match skips,
    anything else (including no previous artifacts at all) predicts and overwrites.
    A per-scan error is isolated (recorded ``failed``, batch continues). An empty
    (but present) input directory is a no-op.

    Args:
        input_dir: Directory of staged scans.
        output_dir: Directory to write per-scan outputs into.
        source: Model-card source; ``None`` uses the production WandbRegistrySource.
        predict_code_sha: Provenance sha (falls back to ``SRP_PREDICT_CODE_SHA``).
        predict_container_digest: Provenance digest (env fallback).

    Returns:
        A :class:`BatchResult` with one :class:`ScanResult` per scan.

    Raises:
        FileNotFoundError: If ``input_dir`` does not exist.
        ValueError: If two sidecars share a ``scan_key`` (a batch-level staging error,
            surfaced before any prediction).
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    scans = discover_scans(input_dir)
    result = BatchResult()
    if not scans:
        logger.warning("No scans discovered under %s", input_dir.as_posix())
        return result

    resolved_code_sha = (
        predict_code_sha
        if predict_code_sha is not None
        else os.environ.get("SRP_PREDICT_CODE_SHA", "")
    )
    worker = WarmModelWorker(source=source)
    for scan in scans:
        if scan.error is not None:
            logger.error("Scan %s failed: %s", scan.scan_key, scan.error)
            result.scans.append(ScanResult(scan.scan_key, "failed", scan.error))
            continue

        out_scan_dir = output_dir / scan.scan_key
        try:
            refs = worker.resolve(scan.params)
            current_key = _identity_key(
                scan_key=scan.scan_key,
                images_checksum=scan.images_checksum,
                params_dict=scan.params.values,
                model_refs=refs,
                predict_code_sha=resolved_code_sha,
                predict_output_params=worker.output_params(),
            )
            previous_key = _previous_identity_key(out_scan_dir, scan.scan_key)
            if previous_key is not None and previous_key == current_key:
                logger.info("Skipping %s (idempotency key unchanged)", scan.scan_key)
                result.scans.append(ScanResult(scan.scan_key, "skipped"))
                continue
            _predict_one(
                worker, scan, out_scan_dir, refs, predict_code_sha, predict_container_digest
            )
            result.scans.append(ScanResult(scan.scan_key, "ok"))
        except Exception as exc:  # noqa: BLE001 - isolate per-scan failures
            logger.exception("Scan %s failed", scan.scan_key)
            result.scans.append(ScanResult(scan.scan_key, "failed", str(exc)))
    return result
```
This requires `import os` at the top of `batch.py` if not already present — check; if absent,
add it alongside the existing `import json`/`import logging`/`import shutil` block.

Replace `_predict_one` (currently `batch.py:207-245`) with:
```python
def _predict_one(
    worker: WarmModelWorker,
    scan: ScanInput,
    out_scan_dir: Path,
    refs: dict,
    predict_code_sha: str | None,
    predict_container_digest: str | None,
) -> None:
    """Predict one scan and write its outputs + copied sidecar. Raises on failure."""
    if not scan.frames:
        raise ValueError(
            f"no image frames co-located with sidecar {scan.sidecar_path.as_posix()}"
        )
    assert scan.params is not None  # run_batch filters error scans (params-None) first
    if not refs:
        # A scan matching no model for any root type is a hard per-scan failure rather
        # than an empty-artifacts manifest: write_prediction_outputs permits an empty
        # manifest, but the downstream trait-extractor rejects one, so surface it here.
        raise ValueError(f"no models resolved for params {scan.params.values!r}")
    video = make_video_from_images(scan.frames, greyscale=True)
    labels = worker.predict(scan.params, video)
    out_scan_dir.mkdir(parents=True, exist_ok=True)
    # Copy the sidecar BEFORE the manifest: write_prediction_outputs writes the manifest
    # last as the resume commit-marker, so the sidecar must already be present when it
    # lands — else a crash in between leaves a manifest with no sidecar that resume skips
    # forever and the trait-extractor then rejects (an incomplete input tree).
    shutil.copyfile(
        scan.sidecar_path, out_scan_dir / f"{scan.scan_key}{_SIDECAR_SUFFIX}"
    )
    write_prediction_outputs(
        labels,
        refs,
        out_scan_dir,
        scan_key=scan.scan_key,
        inference_config=worker.inference_config(),
        output_params=worker.output_params(),
        predict_code_sha=predict_code_sha,
        predict_container_digest=predict_container_digest,
    )
```
Note `refs` is passed straight through to `write_prediction_outputs` unchanged from before;
`worker.predict(scan.params, video)` still resolves internally a second time via
`get_predictors()` — that's pre-existing, harmless (the one real cost, `list_cards()`, is cached
after its first call per `WarmModelWorker` instance), and out of scope to change here.

- [ ] **Step 7: Run every new test from Steps 1–2 to verify they pass**

Run:
```bash
uv run pytest tests/test_batch.py -k "changed_params_causes_repredict or changed_images_checksum_causes_repredict or changed_predict_code_sha_causes_repredict or changed_model_ref_causes_repredict or corrupt_previous_manifest or manifest_missing_sidecar_does_not_abort" -v
```
Expected: all PASS.

- [ ] **Step 8: Run the full test_batch.py file, with special attention to the highest-risk pre-existing tests**

Run: `uv run pytest tests/test_batch.py -v`

Expected: 100% pass, including (name them explicitly and check each one in the output):
`test_rerun_skips_completed_scan`, `test_zero_resolved_models_is_failed`,
`test_one_failing_scan_does_not_abort_batch`, `test_sidecar_copy_failure_leaves_no_manifest`,
`test_resume_mixed_skip_and_predict`, `test_run_batch_writes_outputs_and_copies_sidecar`,
`test_run_batch_constructs_single_worker`.

- [ ] **Step 9: Run the full project test suite**

Run: `uv run pytest -m "not gpu and not acceptance and not wandb and not parity" tests/`

Expected: 100% pass.

- [ ] **Step 10: Commit (single commit for this entire task)**

```bash
git add sleap_roots_predict/batch.py tests/test_batch.py
git commit -m "feat: idempotency-key skip-if-done, replacing Path.exists() resume"
```

---

## Task 7: OpenSpec validation gate

**Files:** none (verification only)

- [ ] **Step 1: Validate**

Run: `openspec validate consume-run-manifest --strict`

Expected: `Change 'consume-run-manifest' is valid`. If not, fix the delta spec (it should already
be correct from the review round — this is a final confirmation, not expected to need edits).

---

## Task 8: Docs

**Files:**
- Modify: `CHANGELOG.md` (`[Unreleased]` → "Predict container CLI" bullet)
- Modify: `API.md` (`run_batch` section prose)
- Modify: `README.md` (~line 211, "Running the predict container" section)
- Modify: `openspec/project.md` (External Dependencies version literal + Roadmap note)

- [ ] **Step 1: Update CHANGELOG.md**

In the `[Unreleased]` → "Predict container CLI" bullet, replace the clause
`"and per scan skips-if-done (existence-based resume),"` with:
```
and per scan, when a `run_manifest.json` (`RunManifest`, `sleap-roots-contracts==0.1.0a7`) is
staged in `input_dir`, scopes discovery to exactly its `scan_keys` (an out-of-scope sidecar is
silently excluded); skip-if-done now compares a recomputed idempotency key
(`compute_idempotency_key`) against the prior run's own artifacts, skipping only on an exact
match and otherwise (re)predicting — no new storage. Note: this means `resolve()` (and its
one-time model-registry fetch) now runs once per batch invocation even when every scan is
already done, which it previously did not.
```

- [ ] **Step 2: Update API.md**

In the `run_batch` section, replace `"skips if the manifest already exists (resume)"` with
wording matching Step 1 (manifest-scoped discovery + idempotency-key comparison).

- [ ] **Step 3: Update README.md**

Around line 211, replace `"It skips a scan whose manifest already exists (resume) and exits
non-zero if any scan failed."` with wording matching Step 1, plus a sentence noting
`run_manifest.json`-scoped discovery when a pipeline stages one.

- [ ] **Step 4: Update openspec/project.md**

- Bump the External Dependencies `sleap-roots-contracts` version literal: `==0.1.0a6` →
  `==0.1.0a7`.
- In the Roadmap note at the top of the file, add a credit for this change closing the
  `sleap-roots-predict` row of `sleap-roots-pipeline#37`, parallel to the existing `#15` credit
  for the parity harness.

- [ ] **Step 5: Grep sweep**

Run (adjust for your shell):
```bash
grep -rniE "skip.{0,20}(exist|manifest)|exists.{0,20}resume" README.md API.md CHANGELOG.md openspec/project.md
```
Confirm no stale existence-based-resume phrasing remains anywhere.

- [ ] **Step 6: Commit**

```bash
git add CHANGELOG.md API.md README.md openspec/project.md
git commit -m "docs: update run-manifest scoping and idempotency-key skip-if-done docs"
```

---

## Task 9: Pre-merge gate

**Files:** none (verification only)

- [ ] **Step 1: Run the pre-merge gate**

Invoke `/pre-merge`. Confirm its pytest invocation uses `-m "not gpu and not acceptance and not
wandb and not parity"` (matching `ci.yml`'s exact filter), not `/pre-merge`'s own default `-m
"not gpu"` (which would pull in flaky wandb-registry tests) — override explicitly if needed.

- [ ] **Step 2: Confirm every tasks.md item is checked off**

Open `openspec/changes/consume-run-manifest/tasks.md` and mark every item `- [x]`.

```bash
git add openspec/changes/consume-run-manifest/tasks.md
git commit -m "chore: mark consume-run-manifest tasks complete"
```
