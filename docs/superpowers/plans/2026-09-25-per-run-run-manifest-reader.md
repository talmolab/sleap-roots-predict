# Per-run run-manifest reader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Predict resolves its run manifest through `sleap-roots-contracts 0.1.0a9`'s per-run policy (fail loud under a known run id), scopes discovery to it, forwards it under the name it read, and — predict#43 — gives its three per-scan atomic writes per-writer temp names.

**Architecture:** One private resolver in `run_manifest.py` wraps contracts' `load_run_manifest(..., allow_legacy=True)` plus two log-only diagnostics; `run_batch` calls it once and threads the result into `discover_scans` (scope) and `copy_run_manifest_forward` (publish `read.data` under `read.filename`). `__main__` adds `RunManifestError` to its staging-error tuple. A name-only `_unique_tmp_path` helper in `output_contract.py` replaces `dst.name + ".tmp"` at three sites.

**Tech Stack:** Python 3.11, pytest, `sleap-roots-contracts==0.1.0a9`, sleap-io, `uv`.

**Spec:** `openspec/changes/adopt-per-run-run-manifest-reader/` (proposal, design, tasks, deltas) and `docs/superpowers/specs/2026-09-25-per-run-run-manifest-reader-design.md`. `tasks.md` is the checkbox source of truth — tick its items in the commit that does the work.

## Global Constraints

- Contracts pin stays `sleap-roots-contracts==0.1.0a9` — no dependency or `uv.lock` change.
- Every `load_run_manifest` / `read_run_manifest` call passes `allow_legacy=True` literally, with a comment pointing at `talmolab/sleap-roots-pipeline#82`.
- The run id comes only from `pipeline_run_id_from_env()` — never `os.environ` directly.
- Paths: `pathlib.Path`; path strings in logs/errors via `.as_posix()` (lab convention).
- Tests: set `ARGO_WORKFLOW_NAME` only with `monkeypatch.setenv`; manifests written with `write_bytes`; residue asserted by exact directory contents; directory-at-path failures asserted as `OSError`; any `run_batch` reached without explicit `source=` injects `_recording_source()` and uses `clean_wandb_env`; never stage a manifest into `tests/assets/scans/`.
- Commit messages: repo style (`feat(scope)!:`, `fix(scope):`, `test:`, `docs:`), ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; no closing keyword (close/fix/resolve + #) next to srp#71, #40, #41, #44, #46.
- Test command (ci.yml verbatim): `SRP_DEVICE=cpu uv run pytest -m "not gpu and not acceptance and not wandb" tests/`. Lint: `uv run black --check .`, `uv run ruff check sleap_roots_predict/ scripts/`, `uv run codespell`.

## Review Focus

1. **`ARGO_WORKFLOW_NAME` with incidental whitespace** (`"wf-a\n"` from a templated env) — must resolve `run_manifest.wf-a.json` (contracts strips); pinned in Task 3.
2. **Input path is a regular file, not a directory** (mis-mount) — must fail the batch with an `OSError` on every OS, never "success"; pinned in Task 3.
3. **Standalone `copy_run_manifest_forward` on a nonexistent input directory** — now raises `FileNotFoundError` (contracts) where it was a silent no-op; pinned in Task 3 as intended behavior.
4. **A retried predict in the same workflow** — the output already holds `run_manifest.wf-a.json`; it must be replaced byte-identically, no residue; pinned in Task 3.
5. **A stale legacy `run_manifest.json` already in the output under a per-run read** — must be left untouched (neither deleted nor overwritten), since only the per-run file is this run's; pinned in Task 3.

---

### Task 1: Clear `ARGO_WORKFLOW_NAME` for the default suite

**Files:**
- Modify: `tests/conftest.py` (add an autouse fixture after the imports)

**Interfaces:**
- Produces: every test starts with `ARGO_WORKFLOW_NAME` unset.

- [ ] **Step 1: Add the fixture**

```python
@pytest.fixture(autouse=True)
def _no_pipeline_run_id(monkeypatch):
    """Run every test as a local, identity-less run unless it opts in.

    ``ARGO_WORKFLOW_NAME`` is the run identity predict reads through contracts'
    ``pipeline_run_id_from_env()``; set, a missing run manifest fails the batch. A
    developer shell (or CI) that happens to export it must not change what the default
    suite tests. Tests that need an identity ``monkeypatch.setenv`` it themselves.
    """
    monkeypatch.delenv("ARGO_WORKFLOW_NAME", raising=False)
```

- [ ] **Step 2: Run the suite**

Run: `SRP_DEVICE=cpu uv run pytest -m "not gpu and not acceptance and not wandb" tests/ -q`
Expected: PASS (nothing reads the variable yet).

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test: clear ARGO_WORKFLOW_NAME in an autouse fixture so the default suite has no run identity"
```

---

### Task 2: predict#43 — per-writer unique temp names

**Files:**
- Modify: `sleap_roots_predict/output_contract.py` (add `_unique_tmp_path`; `.slp` site ~L193, manifest site ~L235)
- Modify: `sleap_roots_predict/batch.py` (sidecar site ~L458-465 and the comment above it)
- Test: `tests/test_output_contract.py`, `tests/test_batch.py`

**Interfaces:**
- Produces: `sleap_roots_predict.output_contract._unique_tmp_path(dst: Path) -> Path`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_output_contract.py`:

```python
def test_unique_tmp_path_is_private_hidden_and_inert(tmp_path):
    from sleap_roots_predict.output_contract import _unique_tmp_path

    slp = tmp_path / "scan1.modelx.rootprimary.slp"
    first, second = _unique_tmp_path(slp), _unique_tmp_path(slp)
    assert first != second
    for tmp in (first, second):
        assert tmp.parent == slp.parent
        assert tmp.name.startswith(".") and tmp.name.endswith(".tmp")
        # never matched by the stale-.slp sweep or a consumer glob
        assert not (tmp.name.startswith("scan1.model") and tmp.name.endswith(".slp"))
        assert not tmp.name.endswith(".predictions.json")


def _record_and_fail_for(monkeypatch, suffix):
    """Patch os.replace to record + raise only for destinations ending ``suffix``."""
    import sleap_roots_predict.output_contract as oc_mod

    real_replace = oc_mod.os.replace
    seen = []

    def _replace(src, dst, *a, **k):
        if str(dst).endswith(suffix):
            seen.append(str(src))
            raise OSError("simulated interruption")
        return real_replace(src, dst, *a, **k)

    monkeypatch.setattr(oc_mod.os, "replace", _replace)
    return seen


@pytest.mark.parametrize("suffix", [".slp", ".predictions.json"])
def test_concurrent_writers_of_one_scan_use_private_temp_files(
    rice_source, video, tmp_path, monkeypatch, suffix
):
    worker = WarmModelWorker(rice_source)
    labels = worker.predict(_params(), video)
    refs = worker.resolve(_params())
    seen = _record_and_fail_for(monkeypatch, suffix)
    for _ in range(2):
        with pytest.raises(OSError):
            write_prediction_outputs(
                labels,
                refs,
                tmp_path,
                scan_key="scan0731",
                inference_config=worker.inference_config(),
                output_params=worker.output_params(),
            )
        assert not [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert len(seen) == 2 and seen[0] != seen[1]


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_written_artifacts_keep_a_direct_writes_permissions(rice_source, video, tmp_path):
    """Guard (green before and after #43): a private 0600 temp mode must never leak."""
    old = os.umask(0o022)
    try:
        worker = WarmModelWorker(rice_source)
        write_prediction_outputs(
            worker.predict(_params(), video),
            worker.resolve(_params()),
            tmp_path,
            scan_key="scan0731",
            inference_config=worker.inference_config(),
            output_params=worker.output_params(),
        )
        control = tmp_path / "control"
        control.write_bytes(b"x")
        want = control.stat().st_mode & 0o777
        written = list(tmp_path.glob("scan0731*"))
        assert written
        assert all(p.stat().st_mode & 0o777 == want for p in written)
    finally:
        os.umask(old)
```

Add `import os` to the module's imports. Then tighten the two existing residue checks in the same file — in `test_slp_write_leaves_no_partial_file_if_replace_fails` and `test_manifest_write_leaves_no_partial_file_if_replace_fails` replace `assert not list(tmp_path.glob("*.tmp"))` with:

```python
    assert not [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
```

In `tests/test_batch.py`, make `test_sidecar_copy_leaves_no_partial_file_if_replace_fails` path-conditional and exact, and add a uniqueness test:

```python
def _fail_replace_for_sidecars(monkeypatch):
    """Record + raise only for the sidecar's destination; everything else is real."""
    import os as _os

    real_replace = _os.replace
    seen = []

    def _replace(src, dst, *a, **k):
        if str(dst).endswith(".scan_metadata.json"):
            seen.append(str(src))
            raise OSError("simulated interruption")
        return real_replace(src, dst, *a, **k)

    monkeypatch.setattr("os.replace", _replace)
    return seen


def test_sidecar_copy_leaves_no_partial_file_if_replace_fails(
    scan_input_dir: Path, all_roots_source, tmp_path: Path, monkeypatch
):
    _fail_replace_for_sidecars(monkeypatch)
    out = tmp_path / "out"
    result = run_batch(scan_input_dir, out, source=all_roots_source)
    assert [s.status for s in result.scans] == ["failed"]
    scan_dir = out / "scanCPTEST0"
    assert not (scan_dir / "scanCPTEST0.scan_metadata.json").exists()
    assert not (scan_dir / "scanCPTEST0.predictions.json").exists()
    assert not [p.name for p in scan_dir.iterdir() if p.name.endswith(".tmp")]


def test_concurrent_sidecar_copies_use_private_temp_files(
    scan_input_dir: Path, all_roots_source, tmp_path: Path, monkeypatch
):
    seen = _fail_replace_for_sidecars(monkeypatch)
    out = tmp_path / "out"
    for _ in range(2):
        run_batch(scan_input_dir, out, source=all_roots_source)
    assert len(seen) == 2 and seen[0] != seen[1]
    scan_dir = out / "scanCPTEST0"
    assert not [p.name for p in scan_dir.iterdir() if p.name.endswith(".tmp")]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
def test_copied_sidecar_keeps_a_direct_writes_permissions(
    scan_input_dir: Path, all_roots_source, tmp_path: Path
):
    """Guard (green before and after #43)."""
    import os as _os

    old = _os.umask(0o022)
    try:
        out = tmp_path / "out"
        run_batch(scan_input_dir, out, source=all_roots_source)
        control = out / "control"
        control.write_bytes(b"x")
        sidecar = out / "scanCPTEST0" / "scanCPTEST0.scan_metadata.json"
        assert sidecar.stat().st_mode & 0o777 == control.stat().st_mode & 0o777
    finally:
        _os.umask(old)
```

`test_sidecar_copy_failure_leaves_no_manifest` patches `batch_mod.shutil.copyfile`, which the run-manifest forward never calls (it writes through `mkstemp`'s fd) — leave it, but add the comment `# path-safe: only the sidecar copy uses shutil.copyfile`.

- [ ] **Step 2: Run to verify failure**

Run: `SRP_DEVICE=cpu uv run pytest tests/test_output_contract.py tests/test_batch.py -q -k "unique_tmp or private_temp or partial_file or permissions"`
Expected: FAIL — `ImportError: cannot import name '_unique_tmp_path'`, and the uniqueness tests fail on `seen[0] != seen[1]` (both are `…​.tmp` fixed names). Mode guards PASS (they are guards).

- [ ] **Step 3: Implement** — in `output_contract.py` add `import uuid` and, below `predictions_json_path`:

```python
def _unique_tmp_path(dst: Path) -> Path:
    """A temporary path beside ``dst``, private to this writer.

    Unique (not derivable from ``dst`` alone), so two concurrent writers of the same scan
    never share a temp file and one's ``os.replace`` can never publish the other's
    half-written bytes (predict#43). In ``dst``'s own directory, so the replace is never
    cross-device on NFS. Dot-prefixed and ``.tmp``-suffixed, so no consumer glob
    (``*.predictions.json``, ``*.slp``, the ``{scan_key}.model…`` sweep) ever matches it --
    which is also why an orphan left by a SIGKILL is inert and not reclaimed. Name-only
    rather than ``mkstemp``: ``mkstemp`` creates at ``0600``, and ``shutil.copyfile`` /
    ``write_text`` would keep that mode, making the artifact unreadable to the downstream
    uid. ``uuid4`` rather than a pid, since every container is PID 1; 16 hex digits
    (64 bits) keeps deep Windows paths clear of ``MAX_PATH``.
    """
    return dst.with_name(f".{dst.name}.{uuid.uuid4().hex[:16]}.tmp")
```

Replace `tmp_slp_path = slp_path.with_name(slp_path.name + ".tmp")` with `tmp_slp_path = _unique_tmp_path(slp_path)` and `tmp_manifest_path = manifest_path.with_name(manifest_path.name + ".tmp")` with `tmp_manifest_path = _unique_tmp_path(manifest_path)`. Update the `.slp` comment's "the \".tmp\"-suffixed temp name" wording to "the temp name".

In `batch.py` import `_unique_tmp_path` alongside the other `output_contract` imports; replace `tmp_sidecar_dst = sidecar_dst.with_name(sidecar_dst.name + ".tmp")` with `tmp_sidecar_dst = _unique_tmp_path(sidecar_dst)`; delete the "Note for test authors: … never add one." paragraph of the comment above it (the tests are now path-conditional).

- [ ] **Step 4: Run to verify pass**

Run: `SRP_DEVICE=cpu uv run pytest tests/test_output_contract.py tests/test_batch.py -q`
Expected: PASS.

- [ ] **Step 5: Tick `tasks.md` 2.1/2.2, lint, commit**

```bash
uv run black --check . && uv run ruff check sleap_roots_predict/ scripts/
git add sleap_roots_predict/output_contract.py sleap_roots_predict/batch.py tests/test_output_contract.py tests/test_batch.py openspec/changes/adopt-per-run-run-manifest-reader/tasks.md
git commit -m "fix(output): per-writer unique temp names for per-scan atomic writes (#43)"
```

---

### Task 3: Per-run run-manifest reader (resolve once, scope, forward under the name read, CLI)

**Files:**
- Create: `tests/manifest_builders.py`
- Modify: `sleap_roots_predict/run_manifest.py` (whole module), `sleap_roots_predict/batch.py` (imports, `discover_scans`, `run_batch`, module docstring), `sleap_roots_predict/__main__.py` (except tuple, docstrings, comment), `sleap_roots_predict/__init__.py` (docstring)
- Test: `tests/test_run_manifest.py`, `tests/test_batch.py`

**Interfaces:**
- Consumes (contracts 0.1.0a9): `load_run_manifest(directory, pipeline_run_id, *, allow_legacy) -> LoadedRunManifest | None`; `read_run_manifest(...) -> RunManifestRead | None`; `pipeline_run_id_from_env() -> str | None`; `RunManifestRead(filename, data, mode, is_per_run)`; `LoadedRunManifest(manifest, read)`; `RunManifestError`, `RunManifestMissingError`, `RunManifestIdentityError`.
- Produces: `run_manifest._resolve_run_manifest(input_dir, pipeline_run_id) -> LoadedRunManifest | None`; `copy_run_manifest_forward(input_dir, output_dir, *, read=_UNREAD)`; `discover_scans(input_dir, *, manifest=_UNREAD)`.

- [ ] **Step 1: Create the fixture helper** `tests/manifest_builders.py`:

```python
"""Write RunManifest JSON fixtures for tests (importable like card_builders)."""

import json
from pathlib import Path


def write_run_manifest(
    directory: Path, filename: str, *, pipeline_run_id: str, scan_keys
) -> Path:
    """Write a manifest as raw bytes (never write_text: CRLF on Windows)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    body = {"pipeline_run_id": pipeline_run_id, "scan_keys": list(scan_keys)}
    path.write_bytes(json.dumps(body).encode("utf-8"))
    return path
```

- [ ] **Step 2: Write the failing `test_run_manifest.py` tests** — append (new private symbols are imported inside tests so the module still collects while red):

```python
from manifest_builders import write_run_manifest

_PER_RUN = "run_manifest.wf-a.json"


def _resolve(input_dir, run_id):
    from sleap_roots_predict.run_manifest import _resolve_run_manifest

    return _resolve_run_manifest(input_dir, run_id)


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def test_resolver_prefers_the_per_run_manifest_silently(tmp_path, caplog):
    write_run_manifest(tmp_path, RUN_MANIFEST_FILENAME, pipeline_run_id="hpdpf", scan_keys=["s1", "s2"])
    write_run_manifest(tmp_path, _PER_RUN, pipeline_run_id="wf-a", scan_keys=["s1"])
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        loaded = _resolve(tmp_path, "wf-a")
    assert loaded.read.is_per_run and loaded.read.filename == _PER_RUN
    assert loaded.manifest.scan_keys == ["s1"]
    assert _warnings(caplog) == []


def test_resolver_warns_when_a_legacy_manifest_names_another_run(tmp_path, caplog):
    write_run_manifest(tmp_path, RUN_MANIFEST_FILENAME, pipeline_run_id="hpdpf", scan_keys=["s1"])
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        loaded = _resolve(tmp_path, "wf-a")
    assert loaded.read.filename == RUN_MANIFEST_FILENAME
    (msg,) = _warnings(caplog)
    assert "wf-a" in msg and "hpdpf" in msg


def test_resolver_is_silent_when_a_legacy_manifest_names_this_run(tmp_path, caplog):
    write_run_manifest(tmp_path, RUN_MANIFEST_FILENAME, pipeline_run_id="wf-a", scan_keys=["s1"])
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        _resolve(tmp_path, "wf-a")
    assert _warnings(caplog) == []


def test_resolver_warns_about_unread_per_run_manifests_under_an_identity(tmp_path, caplog):
    write_run_manifest(tmp_path, RUN_MANIFEST_FILENAME, pipeline_run_id="wf-a", scan_keys=["s1"])
    write_run_manifest(tmp_path, "run_manifest.wf-b.json", pipeline_run_id="wf-b", scan_keys=["s2"])
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        _resolve(tmp_path, "wf-a")
    assert any("run_manifest.wf-b.json" in m for m in _warnings(caplog))


def test_resolver_raises_for_a_known_run_with_no_manifest(tmp_path):
    from sleap_roots_contracts import RunManifestMissingError

    with pytest.raises(RunManifestMissingError):
        _resolve(tmp_path, "wf-a")


def test_resolver_rejects_a_per_run_manifest_naming_another_run(tmp_path):
    from sleap_roots_contracts import RunManifestIdentityError

    write_run_manifest(tmp_path, _PER_RUN, pipeline_run_id="wf-b", scan_keys=["s1"])
    with pytest.raises(RunManifestIdentityError):
        _resolve(tmp_path, "wf-a")


def test_resolver_without_identity_ignores_per_run_files_but_warns(tmp_path, caplog):
    write_run_manifest(tmp_path, _PER_RUN, pipeline_run_id="wf-a", scan_keys=["s1"])
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        assert _resolve(tmp_path, None) is None
    assert any(_PER_RUN in m for m in _warnings(caplog))


def test_a_directory_at_the_legacy_path_raises(tmp_path):
    (tmp_path / RUN_MANIFEST_FILENAME).mkdir()
    with pytest.raises(OSError):
        _resolve(tmp_path, None)


def _per_run_read(input_dir):
    from sleap_roots_contracts import read_run_manifest

    return read_run_manifest(input_dir, "wf-a", allow_legacy=True)


def test_forward_publishes_under_the_per_run_name_only(tmp_path, caplog):
    src = write_run_manifest(tmp_path / "in", _PER_RUN, pipeline_run_id="wf-a", scan_keys=["s1"])
    out = tmp_path / "out"
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        copy_run_manifest_forward(tmp_path / "in", out, read=_per_run_read(tmp_path / "in"))
    assert _names(out) == {_PER_RUN}
    assert (out / _PER_RUN).read_bytes() == src.read_bytes()
    infos = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert any(_PER_RUN in m for m in infos)


def test_standalone_forward_of_unparsable_per_run_bytes(tmp_path, monkeypatch):
    monkeypatch.setenv("ARGO_WORKFLOW_NAME", "wf-a")
    src = tmp_path / "in" / _PER_RUN
    src.parent.mkdir()
    src.write_bytes(b"{not valid json")
    copy_run_manifest_forward(tmp_path / "in", tmp_path / "out")
    assert (tmp_path / "out" / _PER_RUN).read_bytes() == b"{not valid json"


def test_standalone_forward_does_not_check_run_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("ARGO_WORKFLOW_NAME", "wf-a")
    src = write_run_manifest(tmp_path / "in", _PER_RUN, pipeline_run_id="wf-b", scan_keys=["s1"])
    copy_run_manifest_forward(tmp_path / "in", tmp_path / "out")
    assert (tmp_path / "out" / _PER_RUN).read_bytes() == src.read_bytes()


def test_standalone_forward_fails_loud_for_a_known_run_with_no_manifest(
    tmp_path, monkeypatch, caplog
):
    from sleap_roots_contracts import RunManifestMissingError

    monkeypatch.setenv("ARGO_WORKFLOW_NAME", "wf-a")
    (tmp_path / "in").mkdir()
    out = tmp_path / "out"
    with caplog.at_level(logging.ERROR, logger=_LOGGER):
        with pytest.raises(RunManifestMissingError):
            copy_run_manifest_forward(tmp_path / "in", out)
    assert not out.exists()
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert errors and (tmp_path / "in").as_posix() in errors[0] and out.as_posix() in errors[0]


def test_standalone_forward_on_a_missing_input_directory_raises(tmp_path):
    """Review focus 3: was a silent no-op; a mis-mounted stage-in must not pass."""
    with pytest.raises(FileNotFoundError):
        copy_run_manifest_forward(tmp_path / "nope", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_per_run_same_directory_forward_is_a_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("ARGO_WORKFLOW_NAME", "wf-a")
    both = tmp_path / "both"
    src = write_run_manifest(both, _PER_RUN, pipeline_run_id="wf-a", scan_keys=["s1"])
    before, mtime = src.read_bytes(), src.stat().st_mtime_ns
    copy_run_manifest_forward(both, both / ".." / "both")  # a different spelling
    assert src.read_bytes() == before and src.stat().st_mtime_ns == mtime
    assert _names(both) == {_PER_RUN}


def test_retried_forward_replaces_this_runs_per_run_manifest(tmp_path, monkeypatch):
    """Review focus 4: a same-workflow retry finds its own earlier forward."""
    monkeypatch.setenv("ARGO_WORKFLOW_NAME", "wf-a")
    src = write_run_manifest(tmp_path / "in", _PER_RUN, pipeline_run_id="wf-a", scan_keys=["s1", "s2"])
    write_run_manifest(tmp_path / "out", _PER_RUN, pipeline_run_id="wf-a", scan_keys=["s1"])
    copy_run_manifest_forward(tmp_path / "in", tmp_path / "out")
    assert (tmp_path / "out" / _PER_RUN).read_bytes() == src.read_bytes()
    assert _names(tmp_path / "out") == {_PER_RUN}


def test_per_run_forward_leaves_a_stale_legacy_output_manifest_untouched(tmp_path, monkeypatch):
    """Review focus 5."""
    monkeypatch.setenv("ARGO_WORKFLOW_NAME", "wf-a")
    write_run_manifest(tmp_path / "in", _PER_RUN, pipeline_run_id="wf-a", scan_keys=["s1"])
    stale = write_run_manifest(tmp_path / "out", RUN_MANIFEST_FILENAME, pipeline_run_id="old", scan_keys=["s9"])
    before = stale.read_bytes()
    copy_run_manifest_forward(tmp_path / "in", tmp_path / "out")
    assert stale.read_bytes() == before
    assert _names(tmp_path / "out") == {_PER_RUN, RUN_MANIFEST_FILENAME}


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_per_run_forward_keeps_the_source_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("ARGO_WORKFLOW_NAME", "wf-a")
    src = write_run_manifest(tmp_path / "in", _PER_RUN, pipeline_run_id="wf-a", scan_keys=["s1"])
    os.chmod(src, 0o640)
    copy_run_manifest_forward(tmp_path / "in", tmp_path / "out")
    assert (tmp_path / "out" / _PER_RUN).stat().st_mode & 0o777 == 0o640
```

Then **rewrite** `test_permission_error_from_the_presence_check_is_reported_cleanly` so the failure is injected at the read:

```python
def test_permission_error_from_the_read_is_reported_cleanly(tmp_path: Path, caplog, monkeypatch):
    """The read is the first filesystem step, so the first that can fail; an unreadable
    source manifest (an NFS ACL misconfiguration) must still get the log naming both
    directories, not only the CLI's generic line."""
    import sleap_roots_predict.run_manifest as rm_mod

    _stage(tmp_path / "in")
    out = tmp_path / "out"

    def _denied(*_a, **_k):
        raise PermissionError(errno.EACCES, "Permission denied")

    monkeypatch.setattr(rm_mod, "read_run_manifest", _denied)
    with caplog.at_level(logging.ERROR, logger=_LOGGER):
        with pytest.raises(PermissionError):
            copy_run_manifest_forward(tmp_path / "in", out)
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert errors, "read failure propagated without the mandated log"
    assert (tmp_path / "in").as_posix() in errors[0].message
    assert out.as_posix() in errors[0].message
```

and change `_raise_eacces_for`'s docstring first paragraph to: "Make stat'ing exactly ``target`` fail with EACCES, delegating everything else. Used for the same-file (identity) check, which stats both paths."

- [ ] **Step 3: Write the failing `test_batch.py` tests** — append:

```python
from manifest_builders import write_run_manifest

_PER_RUN = "run_manifest.wf-a.json"
_TWELVE = [f"scan_{i}" for i in range(1, 13)]


def _stage_accumulated_union(inp: Path, *, with_per_run: bool) -> Path:
    """The measured srp#71 shape: a legacy manifest carrying every run's keys."""
    for key in _TWELVE:
        _write_scan(inp, key, _RICE)
    write_run_manifest(inp, _MANIFEST, pipeline_run_id="sleap-roots-pipeline-hpdpf", scan_keys=_TWELVE)
    if with_per_run:
        return write_run_manifest(inp, _PER_RUN, pipeline_run_id="wf-a", scan_keys=["scan_7"])
    return inp / _MANIFEST


def test_a_per_run_manifest_stops_the_accumulated_union(tmp_path, monkeypatch):
    monkeypatch.setenv("ARGO_WORKFLOW_NAME", "wf-a")
    inp, out = tmp_path / "in", tmp_path / "out"
    src = _stage_accumulated_union(inp, with_per_run=True)
    assert [s.scan_key for s in discover_scans(inp)] == ["scan_7"]
    source, _ = _recording_source()
    result = run_batch(inp, out, source=source)
    assert [s.scan_key for s in result.scans] == ["scan_7"]
    assert (out / _PER_RUN).read_bytes() == src.read_bytes()
    assert not (out / _MANIFEST).exists()


def test_a_legacy_union_is_honored_and_flagged_during_the_rollout(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("ARGO_WORKFLOW_NAME", "wf-a")
    inp, out = tmp_path / "in", tmp_path / "out"
    src = _stage_accumulated_union(inp, with_per_run=False)
    source, _ = _recording_source()
    with caplog.at_level(logging.WARNING, logger="sleap_roots_predict.run_manifest"):
        result = run_batch(inp, out, source=source)
    assert sorted(s.scan_key for s in result.scans) == sorted(_TWELVE)
    assert any("hpdpf" in r.getMessage() and "wf-a" in r.getMessage() for r in caplog.records)
    assert {p.name for p in out.iterdir() if p.is_file()} == {_MANIFEST}
    assert (out / _MANIFEST).read_bytes() == src.read_bytes()


def _no_manifest(inp):
    _write_scan(inp, "scanA", _RICE)


def _foreign_per_run(inp):
    _write_scan(inp, "scanA", _RICE)
    write_run_manifest(inp, _PER_RUN, pipeline_run_id="wf-b", scan_keys=["scanA"])


def _directory_at_legacy_path(inp):
    _write_scan(inp, "scanA", _RICE)
    (inp / _MANIFEST).mkdir()


@pytest.mark.parametrize(
    "run_id, stage, error",
    [
        ("wf-a", _no_manifest, "RunManifestMissingError"),
        ("wf-a", _foreign_per_run, "RunManifestIdentityError"),
        ("../x", _no_manifest, "ValueError"),
        (None, _directory_at_legacy_path, "OSError"),
    ],
    ids=["missing", "foreign", "unusable-id", "directory-at-path"],
)
def test_manifest_staging_errors_abort_before_any_work(tmp_path, monkeypatch, run_id, stage, error):
    import sleap_roots_contracts as contracts

    if run_id is not None:
        monkeypatch.setenv("ARGO_WORKFLOW_NAME", run_id)
    inp, out = tmp_path / "in", tmp_path / "out"
    stage(inp)
    expected = {"ValueError": ValueError, "OSError": OSError}.get(error) or getattr(contracts, error)
    source, calls = _recording_source()
    with pytest.raises(expected):
        run_batch(inp, out, source=source)
    assert calls["n"] == 0
    assert not out.exists()


def test_standalone_discovery_fails_loud_for_a_known_run(tmp_path, monkeypatch):
    from sleap_roots_contracts import RunManifestMissingError

    monkeypatch.setenv("ARGO_WORKFLOW_NAME", "wf-a")
    _write_scan(tmp_path, "scanA", _RICE)
    with pytest.raises(RunManifestMissingError):
        discover_scans(tmp_path)


def test_discovery_without_identity_ignores_per_run_manifests(tmp_path):
    _write_scan(tmp_path, "scanA", _RICE)
    _write_scan(tmp_path, "scanB", _RICE)
    write_run_manifest(tmp_path, _PER_RUN, pipeline_run_id="wf-a", scan_keys=["scanA"])
    assert [s.scan_key for s in discover_scans(tmp_path)] == ["scanA", "scanB"]


@pytest.mark.parametrize("value", ["   ", ""])
def test_a_blank_run_identity_is_no_identity(tmp_path, monkeypatch, value):
    monkeypatch.setenv("ARGO_WORKFLOW_NAME", value)
    _write_scan(tmp_path, "scanA", _RICE)
    assert [s.scan_key for s in discover_scans(tmp_path)] == ["scanA"]


def test_a_run_identity_with_incidental_whitespace_still_resolves(tmp_path, monkeypatch):
    """Review focus 1."""
    monkeypatch.setenv("ARGO_WORKFLOW_NAME", "wf-a\n")
    _write_scan(tmp_path, "scanA", _RICE)
    _write_scan(tmp_path, "scanB", _RICE)
    write_run_manifest(tmp_path, _PER_RUN, pipeline_run_id="wf-a", scan_keys=["scanB"])
    assert [s.scan_key for s in discover_scans(tmp_path)] == ["scanB"]


def test_an_input_path_that_is_a_file_fails_the_batch(tmp_path):
    """Review focus 2: a mis-mount must never pass as success, on any OS."""
    not_a_dir = tmp_path / "in"
    not_a_dir.write_bytes(b"x")
    source, calls = _recording_source()
    with pytest.raises((OSError, ValueError)):
        run_batch(not_a_dir, tmp_path / "out", source=source)
    assert calls["n"] == 0


def test_an_invalid_per_run_manifest_is_never_forwarded(tmp_path, monkeypatch):
    monkeypatch.setenv("ARGO_WORKFLOW_NAME", "wf-a")
    inp, out = tmp_path / "in", tmp_path / "out"
    _write_scan(inp, "scanA", _RICE)
    (inp / _PER_RUN).write_bytes(b"{not valid json")
    source, _ = _recording_source()
    with pytest.raises(ValueError):
        run_batch(inp, out, source=source)
    assert not out.exists()


def test_run_batch_forwards_the_resolved_per_run_bytes_even_if_rewritten(tmp_path, monkeypatch):
    import sleap_roots_predict.batch as batch_mod

    monkeypatch.setenv("ARGO_WORKFLOW_NAME", "wf-a")
    inp, out = tmp_path / "in", tmp_path / "out"
    _write_scan(inp, "scanA", _RICE)
    src = write_run_manifest(inp, _PER_RUN, pipeline_run_id="wf-a", scan_keys=["scanA"])
    resolved = src.read_bytes()
    real = batch_mod._resolve_run_manifest

    def _resolve_then_rewrite(*args, **kwargs):
        loaded = real(*args, **kwargs)
        src.write_bytes(b'{"pipeline_run_id":"wf-a","scan_keys":["scanA","scanZ"]}')
        return loaded

    monkeypatch.setattr(batch_mod, "_resolve_run_manifest", _resolve_then_rewrite)
    source, _ = _recording_source()
    run_batch(inp, out, source=source)
    assert (out / _PER_RUN).read_bytes() == resolved


def test_missing_input_dir_wins_over_an_unusable_run_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("ARGO_WORKFLOW_NAME", "../x")
    with pytest.raises(FileNotFoundError, match="input scan directory does not exist"):
        run_batch(tmp_path / "nope", tmp_path / "out")


def _main_with_recording_source(monkeypatch):
    import sleap_roots_predict.batch as batch_mod

    source, calls = _recording_source()
    real_run_batch = batch_mod.run_batch

    def _with_source(*args, **kwargs):
        kwargs.setdefault("source", source)
        return real_run_batch(*args, **kwargs)

    monkeypatch.setattr(batch_mod, "run_batch", _with_source)
    return calls


@pytest.mark.parametrize("stage, error", [(_no_manifest, "RunManifestMissingError"), (_foreign_per_run, "RunManifestIdentityError")])
def test_cli_logs_run_manifest_errors_as_staging_errors(tmp_path, monkeypatch, caplog, clean_wandb_env, stage, error):
    import sleap_roots_contracts as contracts
    from sleap_roots_predict.__main__ import main

    monkeypatch.setenv("ARGO_WORKFLOW_NAME", "wf-a")
    inp, out = tmp_path / "in", tmp_path / "out"
    stage(inp)
    calls = _main_with_recording_source(monkeypatch)
    with caplog.at_level("ERROR"):
        with pytest.raises(getattr(contracts, error)):
            main([str(inp), str(out)])
    assert any("Batch aborted" in r.getMessage() for r in caplog.records)
    assert calls["n"] == 0 and not out.exists()
```

Also add the per-run-invalid case to nothing else — the existing `test_run_batch_never_forwards_a_manifest_that_fails_validation` stays as-is (legacy name, no identity).

- [ ] **Step 4: Run to verify failure**

Run: `SRP_DEVICE=cpu uv run pytest tests/test_run_manifest.py tests/test_batch.py -q -m "not gpu and not acceptance and not wandb"`
Expected: new tests FAIL — `ImportError: cannot import name '_resolve_run_manifest'`, the contamination test returns 12 scans (red for the right reason), `copy_run_manifest_forward() got an unexpected keyword argument 'read'`, CLI tests see no `Batch aborted`. Existing tests still PASS.

- [ ] **Step 5: Implement `run_manifest.py`** — replace the module with:

```python
"""Resolve this run's run manifest, and forward it from a stage's input to its output.

``RunManifest`` (bloomctl's cross-repo run-scoping shape, keyed by ``scan_keys``) is a
*different* concept from ``PredictionManifest`` (one scan's predict output, written by
``output_contract.py``). Kept in its own module so "manifest" never means two things at
once -- see ``sleap-roots-pipeline#37``.

Resolution is ``sleap-roots-contracts``' own policy (0.1.0a9, talmolab/sleap-roots-pipeline#71):
with a run identity (``ARGO_WORKFLOW_NAME``, via ``pipeline_run_id_from_env``) the per-run
``run_manifest.<pipeline_run_id>.json`` first, then -- while ``allow_legacy=True`` -- the
legacy ``run_manifest.json``; a known run with neither raises rather than falling back to
unscoped discovery. Without an identity only ``run_manifest.json`` is a candidate.

One deliberate divergence from the sibling ``trait_extractor/run_manifest.py`` in
``sleap-roots``: a copy failure **raises** rather than being logged as best-effort (a
silently skipped copy is the bug this exists to prevent, recurring undetected), argued in
the archived ``forward-run-manifest-to-output`` change's ``design.md``. Do not "harmonize"
it back without re-reading that rationale.
"""

import logging
import os
import tempfile
from pathlib import Path

from sleap_roots_contracts import (
    RUN_MANIFEST_FILENAME,
    LoadedRunManifest,
    load_run_manifest,
    pipeline_run_id_from_env,
    read_run_manifest,
)

logger = logging.getLogger(__name__)

# "No read supplied -- read the manifest yourself." Distinct from None, which is a
# caller asserting it already looked and the manifest is absent.
_UNREAD = object()

# Matches the per-run names contracts' run_manifest_filename() builds
# ("run_manifest.<pipeline_run_id>.json") but never the legacy "run_manifest.json" (the
# pattern needs a second dot) nor a dot-prefixed temp file. Contracts keeps its
# prefix/suffix private, so the pattern is spelled out here, as traits does.
_PER_RUN_MANIFEST_GLOB = "run_manifest.*.json"


def _resolve_run_manifest(
    input_dir: str | Path, pipeline_run_id: str | None
) -> LoadedRunManifest | None:
    """Load this run's manifest once, and log the cases where its scope may be wrong.

    Resolution, parsing and the per-run identity cross-check are contracts'
    ``load_run_manifest``; this adds only traceability for what that function
    deliberately leaves unchecked. Neither warning changes the scope:

    - the read is not per-run (unscoped, or scoped by the legacy file) while per-run
      manifests sit unread at the top level of ``input_dir`` -- typically a copied
      cluster tree re-run locally;
    - a legacy manifest read under a known identity names a different run -- under Argo
      the old writer stamps each merge with the current run, so this means another
      workflow merged into the shared directory afterwards.

    Args:
        input_dir: Directory whose top level holds the manifest.
        pipeline_run_id: This run's identity, from ``pipeline_run_id_from_env()``.

    Returns:
        The loaded manifest and the read it came from, or ``None`` when the run has no
        identity and no ``run_manifest.json`` exists.

    Raises:
        sleap_roots_contracts.RunManifestMissingError: If the identity is known but no
            manifest resolves for it.
        sleap_roots_contracts.RunManifestIdentityError: If a per-run manifest names a
            different run.
        ValueError: If the identity is unusable as a filename component, or the
            manifest fails ``RunManifest`` validation (pydantic ``ValidationError``).
        OSError: If a candidate exists but cannot be read (a directory there, a
            permission error, a dangling symlink), or ``input_dir`` is missing.
    """
    # allow_legacy=True while any stage may still write the legacy name; flipping it to
    # False is fleet-wide (talmolab/sleap-roots-pipeline#82).
    loaded = load_run_manifest(input_dir, pipeline_run_id, allow_legacy=True)
    input_dir = Path(input_dir)
    if loaded is None or not loaded.read.is_per_run:
        per_run = sorted(p.name for p in input_dir.glob(_PER_RUN_MANIFEST_GLOB))
        if per_run:
            scoped_by = (
                "discovering every scan"
                if loaded is None
                else f"scoping by legacy {loaded.read.filename}"
            )
            logger.warning(
                "Run %r is not scoped by a per-run manifest in %s (%s); per-run "
                "manifest(s) present but unread: %s",
                pipeline_run_id,
                input_dir.as_posix(),
                scoped_by,
                ", ".join(per_run),
            )
    if (
        loaded is not None
        and pipeline_run_id is not None
        and not loaded.read.is_per_run
        and loaded.manifest.pipeline_run_id != pipeline_run_id
    ):
        logger.warning(
            "Run %r is scoped by legacy %s in %s, which names run %r -- honored while "
            "the legacy fallback is enabled, but it is another run's scope",
            pipeline_run_id,
            loaded.read.filename,
            input_dir.as_posix(),
            loaded.manifest.pipeline_run_id,
        )
    return loaded
```

Keep `_is_same_file` exactly as it is. Replace `copy_run_manifest_forward` with:

```python
def copy_run_manifest_forward(
    input_dir: str | Path, output_dir: str | Path, *, read=_UNREAD
) -> None:
    """Republish this run's manifest from ``input_dir`` into ``output_dir``, if present.

    Publishes under the filename it was read from -- the per-run
    ``run_manifest.<pipeline_run_id>.json`` or the legacy ``run_manifest.json`` -- so the
    downstream stage, resolving by the same policy and identity, finds it. A raw byte copy
    -- never a re-serialization through ``RunManifest`` -- with no validation of contents,
    so the forwarded file is byte-identical to what the upstream producer wrote. Written
    atomically via a ``mkstemp`` file in ``output_dir`` (unique per writer) plus
    :func:`os.replace`, at the source's permissions.

    Args:
        input_dir: Directory whose top level holds the manifest (never searched
            recursively).
        output_dir: Directory to publish into. Created if missing, and removed again if
            the copy then fails, so a failed forward never leaves a directory that did
            not exist before.
        read: Package-internal. A ``RunManifestRead`` already taken from ``input_dir``,
            or ``None`` to assert none was found. Omit it and the manifest is located
            here with contracts' non-parsing ``read_run_manifest`` (``allow_legacy=True``,
            identity from ``pipeline_run_id_from_env()``). ``run_batch`` passes the read
            discovery scoped against, so the bytes published are the bytes scoped.

    Returns:
        None. A no-op when no manifest was found (only possible with no run identity), or
        when source and destination are the same file. When none was found but
        ``output_dir`` already holds a ``run_manifest.json`` from an earlier run, that file
        is left in place and a warning is logged.

    Raises:
        sleap_roots_contracts.RunManifestMissingError: Standalone, if a run identity is
            known and no manifest is found for it.
        ValueError: Standalone, if the run identity is unusable as a filename component.
        OSError: If a present manifest cannot be read or forwarded, or ``input_dir`` is
            missing. Deliberately not best-effort: a silently missing forwarded manifest
            sends the downstream stage to unscoped discovery (or, under a run identity,
            to a failure) -- the contamination this exists to prevent.
    """
    source_dir = Path(input_dir)
    destination_dir = Path(output_dir)
    # Until the read resolves no filename is known, so a failure before then is logged
    # generically rather than naming a file that was never found.
    label = "run manifest"

    tmp: str | None = None
    created: list[Path] = []
    try:
        # The read lives INSIDE this block so that a failure there (EACCES on an NFS ACL,
        # a known run with no manifest) still gets the log naming both directories that
        # the spec requires before anything propagates.
        if read is _UNREAD:
            # allow_legacy=True while any stage may still write the legacy name; flipping
            # it to False is fleet-wide (talmolab/sleap-roots-pipeline#82).
            read = read_run_manifest(
                source_dir, pipeline_run_id_from_env(), allow_legacy=True
            )
        if read is None:
            stale = destination_dir / RUN_MANIFEST_FILENAME
            if stale.is_file():
                logger.warning(
                    "No %s under %s, but %s already holds one from an earlier run; "
                    "leaving it in place -- the downstream stage will be scoped to it",
                    RUN_MANIFEST_FILENAME,
                    source_dir.as_posix(),
                    destination_dir.as_posix(),
                )
            else:
                logger.debug(
                    "No run manifest under %s; nothing to forward",
                    source_dir.as_posix(),
                )
            return

        label = read.filename
        source = source_dir / read.filename
        destination = destination_dir / read.filename
        # (unchanged comment block about the identity check stat'ing both operands)
        try:
            if _is_same_file(source, destination):
                return
        except (FileNotFoundError, NotADirectoryError):
            pass

        # (unchanged mkdir-probe block, verbatim from the current module)
        probe = destination_dir
        while not probe.exists():
            created.append(probe)
            if probe.parent == probe:
                break
            probe = probe.parent
        destination_dir.mkdir(parents=True, exist_ok=True)
        # (unchanged mkstemp rationale comment) -- prefix now carries the name read.
        fd, tmp = tempfile.mkstemp(
            dir=destination_dir,
            prefix=f".{read.filename}.",
            suffix=".tmp",
        )
        with os.fdopen(fd, "wb") as handle:
            handle.write(read.data)
        # (unchanged chmod-before-replace comment)
        os.chmod(tmp, read.mode)
        os.replace(tmp, destination)
    except Exception:
        logger.error(
            "Failed to forward %s from %s to %s",
            label,
            source_dir.as_posix(),
            destination_dir.as_posix(),
        )
        # (unchanged tmp-unlink and created-dir rmdir cleanup, verbatim)
        ...
        raise

    logger.info(
        "Forwarded %s from %s to %s",
        read.filename,
        source_dir.as_posix(),
        destination_dir.as_posix(),
    )
```

The three `(unchanged …)` markers mean: copy those comment blocks and the cleanup body verbatim from the current module — they are correct as written; only the `RUN_MANIFEST_FILENAME` references inside them become `read.filename`.

- [ ] **Step 6: Implement `batch.py`**

Imports — replace the contracts and run_manifest imports with:

```python
from sleap_roots_contracts import (
    PredictionManifest,
    ResolvedParams,
    compute_param_hash,
    pipeline_run_id_from_env,
)
...
from sleap_roots_predict.run_manifest import (
    _UNREAD,
    _resolve_run_manifest,
    copy_run_manifest_forward,
)
```

Add, above `discover_scans`:

```python
def _require_input_dir(input_dir: Path) -> None:
    """Fail on a missing input mount before anything else can mask it."""
    if not input_dir.exists():
        raise FileNotFoundError(
            f"input scan directory does not exist: {input_dir.as_posix()}"
        )
```

`discover_scans(input_dir, *, manifest=_UNREAD)` body up to the loop:

```python
    input_dir = Path(input_dir)
    _require_input_dir(input_dir)
    if manifest is _UNREAD:
        manifest = _resolve_run_manifest(input_dir, pipeline_run_id_from_env())
    scoped_keys = None if manifest is None else set(manifest.manifest.scan_keys)
```

Change the excluded-sidecar debug line's text to `"Excluded %d sidecar(s) outside the run manifest's scope: %s"`. Docstring: replace the manifest paragraph with "The run manifest is resolved by contracts' policy (see `run_manifest._resolve_run_manifest`) and discovery is scoped to exactly its `scan_keys` …"; the `manifest` arg doc becomes "Package-internal. A `LoadedRunManifest` already resolved for `input_dir`, or `None` to assert none was found; omit it and it is resolved here"; `Raises:` adds `RunManifestMissingError`, `RunManifestIdentityError`, `ValueError` (unusable identity), `OSError` (unreadable candidate).

`run_batch` — replace the block from `snapshot = …` through the `copy_run_manifest_forward(...)` call with:

```python
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    _require_input_dir(input_dir)
    # ONE resolution of the run manifest for the whole batch. Discovery scopes against
    # exactly this read and the forward-copy publishes exactly its bytes, so the two can
    # never disagree about the run's scope. Re-reading the path for the copy left a window
    # in which a concurrent upstream writer could widen it -- forwarding a scope predict
    # never predicted -- or remove it, making the forward a silent no-op.
    loaded = _resolve_run_manifest(input_dir, pipeline_run_id_from_env())
    scans = discover_scans(input_dir, manifest=loaded)
    if not scans:
        raise ValueError(f"no scans discovered under {input_dir.as_posix()}")
    # (keep the existing "Forward BEFORE the loop …" comment)
    copy_run_manifest_forward(
        input_dir, output_dir, read=None if loaded is None else loaded.read
    )
```

Update `run_batch`'s `Raises:` (add the three manifest errors as batch-level staging errors raised before any prediction) and the module docstring's second paragraph: "This run's manifest (per-run `run_manifest.<pipeline_run_id>.json`, or the legacy `run_manifest.json`) is also forwarded, under the name it was read, to the **top level** of `output_dir` …".

- [ ] **Step 7: Implement `__main__.py`**

Inside `main`, next to the lazy `run_batch` import: `from sleap_roots_contracts import RunManifestError`. Change the handler to `except (OSError, ValueError, RunManifestError) as exc:` and rewrite its comment:

```python
            # A pre-flight staging error: missing input mount, duplicate scan_key,
            # malformed/unresolvable/foreign run manifest, unusable ARGO_WORKFLOW_NAME, a
            # failed run-manifest forward-copy, zero scans discovered, or no readable
            # production model card. RunManifestError is listed explicitly: contracts
            # deliberately does not derive RunManifestMissingError /
            # RunManifestIdentityError from ValueError. OSError rather than
            # FileNotFoundError because the forward-copy raises PermissionError and
            # friends. Deliberately a superset of the staging set (a registry network
            # error subclasses OSError too) -- accepted: exit 1 either way and the
            # traceback still surfaces. Log a clean line, then re-raise (exit 1).
```

Module docstring's exit-`1` list gains "a run manifest that cannot be resolved for a known run, names another run, or an unusable `ARGO_WORKFLOW_NAME`".

`__init__.py` package docstring (L12-15): "forward this run's manifest (per-run or legacy `run_manifest.json`) to the output directory (`copy_run_manifest_forward`), so the downstream traits …".

- [ ] **Step 8: Run to verify pass**

Run: `SRP_DEVICE=cpu uv run pytest -m "not gpu and not acceptance and not wandb" tests/ -q`
Expected: PASS (all, including every pre-existing manifest test).

- [ ] **Step 9: Tick `tasks.md` 3.1/3.2, lint, commit**

```bash
uv run black . && uv run black --check . && uv run ruff check sleap_roots_predict/ scripts/
git add sleap_roots_predict/run_manifest.py sleap_roots_predict/batch.py sleap_roots_predict/__main__.py sleap_roots_predict/__init__.py tests/manifest_builders.py tests/test_run_manifest.py tests/test_batch.py openspec/changes/adopt-per-run-run-manifest-reader/tasks.md
git commit -m "feat(run-manifest)!: resolve the run manifest via contracts 0.1.0a9 per-run policy and forward it under the name read"
```

Body: "Part of talmolab/sleap-roots-pipeline#71 (predict tracking: #46). BREAKING: with ARGO_WORKFLOW_NAME set, a missing manifest aborts the batch (exit 1) instead of unscoped discovery; a directory at a manifest path now raises."

---

### Task 4: Docs

**Files:** `API.md`, `README.md`, `openspec/project.md`, `CHANGELOG.md`

- [ ] **Step 1: `API.md`**
  - `run_batch` (~L210): manifest sentence → "This run's manifest is resolved once (contracts' per-run policy: `run_manifest.<ARGO_WORKFLOW_NAME>.json`, else the legacy `run_manifest.json`), scopes discovery, and is forwarded under the name read. Raises `RunManifestMissingError` (known run, no manifest), `RunManifestIdentityError`, or `ValueError` (unusable `ARGO_WORKFLOW_NAME`) before any prediction."
  - Add `#### discover_scans` after `run_batch`: signature `discover_scans(input_dir)`, returns one `ScanInput` per scoped sidecar (plus failed entries for manifest keys with no sidecar); same resolution and raises as above; `FileNotFoundError` for a missing directory.
  - `copy_run_manifest_forward` (~L250-275): "Copies this run's manifest … under the filename it was read from"; no-op only with no run identity and no `run_manifest.json`; raises `RunManifestMissingError` standalone under a known run; reads with the non-parsing `read_run_manifest`, so contents are never validated.
  - `write_prediction_outputs` (~L190): "temp file + `os.replace`" → "a per-writer, dot-prefixed temp file beside the destination + `os.replace`".
- [ ] **Step 2: `README.md`**
  - Manifest paragraph (~L209-218): per-run resolution + forward under the name read; "All writes are atomic (per-writer temp file + rename)".
  - Configuration table: add row `ARGO_WORKFLOW_NAME` | set by the cluster predictor template; **leave unset locally** | unset → only `run_manifest.json` is read (absent = unscoped discovery); set → this run's manifest (or, during the rollout, the legacy one) is required and its absence exits 1. Not added to `.env.example`.
  - Rollback note (~L228-245), append: "Per-run `run_manifest.<workflow>.json` files need no rollback cleanup — each names one Argo workflow and no later run reads it; never glob-delete them on the shared mount (a concurrent run's manifest would go too). **Once bloomctl writes per-run manifests, this version is predict's rollback floor**: an older image reads only the stale legacy `run_manifest.json` and would re-scope traits to it. Roll the writer back first."
  - Project tree (~L311, L326): "`run_manifest.py` — resolve this run's manifest (contracts policy) and forward it input_dir -> output_dir"; tests line "Run-manifest resolution + forward-copy tests (offline)".
- [ ] **Step 3: `openspec/project.md`** (~L63-65): `run_manifest.py` — "resolves this run's manifest via contracts (`run_manifest.<id>.json` / legacy `run_manifest.json`) and forwards it, under the name read, to the output directory".
- [ ] **Step 4: `CHANGELOG.md`** under `## [Unreleased]`:
  - `### Changed (BREAKING)`: "Run-manifest resolution uses `sleap-roots-contracts` 0.1.0a9's per-run policy (#46, part of talmolab/sleap-roots-pipeline#71): with `ARGO_WORKFLOW_NAME` set, `run_manifest.<id>.json` is preferred, the legacy `run_manifest.json` is accepted during the rollout, and a missing manifest aborts the batch (exit 1) instead of falling back to unscoped discovery. The manifest is forwarded under the name it was read. A directory at a manifest path now raises instead of reading as absent."
  - `### Fixed` (after the Removed section): "Per-scan atomic writes (sidecar copy, `.slp`, `predictions.json`) use per-writer temp names, so concurrent invocations can no longer publish each other's partial files (#43)."
  - In the existing "Predict container CLI" entry, change "concurrency-safe merging across all three hops is a filed follow-up" to "the cross-run half of that follow-up (#40) is dissolved by per-run manifests (above)".
- [ ] **Step 5: Check and commit**

```bash
uv run codespell && SRP_DEVICE=cpu uv run pytest tests/test_env_docs.py -q
git add API.md README.md openspec/project.md CHANGELOG.md openspec/changes/adopt-per-run-run-manifest-reader/tasks.md
git commit -m "docs: per-run run-manifest resolution, name-read forwarding, and per-writer temp names"
```

---

### Task 5: Verification (no commit unless ticking tasks.md)

- [ ] **Step 1:** `SRP_DEVICE=cpu uv run pytest -m "not gpu and not acceptance and not wandb" tests/` → all pass.
- [ ] **Step 2:** `uv sync --extra dev --extra windows_cuda` then `uv run pytest -m gpu tests/ -rs` → confirm tests **ran** (no "skipped: CUDA unavailable"); then `uv sync --extra dev --extra cpu` to restore.
- [ ] **Step 3:** POSIX-only tests under WSL: `wsl bash -lc 'cd /mnt/c/repos/sleap-roots-predict && uv run pytest tests/test_run_manifest.py tests/test_output_contract.py tests/test_batch.py -k "permissions or mode" -rs'` (if the WSL env lacks the deps, record that and confirm them as passed-not-skipped in the ubuntu/macOS CI logs instead).
- [ ] **Step 4:** Per-commit green: `for c in $(git rev-list --reverse main..HEAD); do git checkout -q $c && SRP_DEVICE=cpu uv run pytest -q -m "not gpu and not acceptance and not wandb" tests/ || { echo "RED at $c"; break; }; done; git checkout -q adopt-per-run-run-manifest-reader`.
- [ ] **Step 5:** `uv run black --check . && uv run ruff check sleap_roots_predict/ scripts/ && uv run codespell && uv build && openspec validate adopt-per-run-run-manifest-reader --strict`.
- [ ] **Step 6:** Tick `tasks.md` 5.x and commit: `git add openspec/changes/adopt-per-run-run-manifest-reader/tasks.md && git commit -m "docs: tick verification tasks for adopt-per-run-run-manifest-reader"`.
