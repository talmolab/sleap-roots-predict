> **Commit discipline.** Each numbered section is ONE commit, and each must leave the suite green
> on all three OS legs. Section 3 is deliberately one commit: `batch.py` imports the
> `run_manifest.py` symbols it replaces and `__init__.py` imports `batch` eagerly, so splitting it
> would break collection. "Verify FAIL" steps are working-tree checkpoints, never commit points.
> CI runs only the PR head, so before pushing run the suite at every commit locally (5.4).
>
> **Standing invariants:** never stage a manifest into `tests/assets/scans/`; set
> `ARGO_WORKFLOW_NAME` only via `monkeypatch.setenv` (1.1's autouse fixture clears it); write
> manifest fixtures with `write_bytes` and assert with `read_bytes()` against the source's own
> bytes; assert residue by a directory's **exact contents**, never a `*.tmp` glob; assert
> directory-at-path failures as `OSError` (Windows raises `PermissionError`, POSIX
> `IsADirectoryError`); any test that reaches `run_batch` without an explicit `source=` injects
> `_recording_source()` and uses `clean_wandb_env`, so no red phase can reach W&B. No commit
> message may put a closing keyword next to srp#71, #40, #41, #44 or #46 (the PR body alone closes #46 and #43).

## 0. Proposal

- [x] 0.1 Commit the OpenSpec proposal (`proposal.md`, `design.md`, `tasks.md`, both deltas)
      before any code.
- [x] 0.2 File a predict tracking issue for srp#71's predict half (with the user's go-ahead),
      cross-linking srp#71, srp#37, sleap-roots#269 and the design of record; record its number
      here. **Filed: #46** (2026-09-25). The PR closes it (code scope only); post-merge rollout items live on srp#71.

## 1. Test harness

- [x] 1.1 Autouse fixture in `tests/conftest.py`: `monkeypatch.delenv("ARGO_WORKFLOW_NAME",
      raising=False)`. Verify the default suite is still green (nothing reads it yet).

## 2. predict#43: per-writer unique temp names (TDD)

- [x] 2.1 Write failing tests first:
  - Unit test `_unique_tmp_path(dst)`: same directory as `dst`; name starts with `.` and ends
    `.tmp`; differs across two calls; does not match `{scan_key}.model*…*.slp` nor
    `*.predictions.json`.
  - `write_prediction_outputs` run twice to one `scan_key`/`out_dir` with `os.replace` patched
    path-conditionally: (a) record + raise only when `dst` ends `.slp` → the two runs' `.slp` temp
    paths differ; (b) record + raise only when `dst` ends `.predictions.json` (`.slp` replaces pass
    through) → the two manifest temp paths differ. After each failure, assert `out_dir`'s exact
    contents contain no temp file.
  - Two sidecar copies of one scan with `os.replace` raising only when `dst` ends
    `.scan_metadata.json` → temp paths differ; no residue.
  - **Guard (green before and after, not a red test):** POSIX-only, with `os.umask(0o022)` set and
    restored in `finally`, every `.slp`, the manifest and the copied sidecar have the mode of a
    control file written with `write_bytes` in the same directory.
  - Make `test_sidecar_copy_failure_leaves_no_manifest` and
    `test_sidecar_copy_leaves_no_partial_file_if_replace_fails` path-conditional (sidecar
    destination only); replace the `glob("*.tmp")` residue checks in `test_output_contract.py` and
    `test_batch.py` with exact-contents checks.
  Verify FAIL (the uniqueness and helper tests) for the right reason.
- [x] 2.2 Implement `_unique_tmp_path(dst)` in `output_contract.py` →
      `dst.with_name(f".{dst.name}.{uuid4().hex[:16]}.tmp")`; route the `.slp`, manifest and
      sidecar (`batch.py`) sites through it, keeping each site's write → `os.replace` →
      unlink-on-failure shape. Drop `_predict_one`'s comment that relied on prose to keep the
      sidecar tests pointed at the right call. Verify PASS; lint clean.

## 3. Per-run run-manifest reader: resolve once, scope, forward under the name read, CLI (TDD)

- [x] 3.1 Write failing tests first. Import new private symbols **inside** the new tests so the
      existing modules still collect during the red phase. A small importable helper (next to
      `card_builders.py`) writes a `RunManifest` JSON (`pipeline_run_id`, `scan_keys`) to a given
      filename with `write_bytes`.
  - **`tests/test_run_manifest.py`**
    - `_resolve_run_manifest` under `wf-a` with a per-run and a stale legacy file → per-run read
      (`is_per_run`, `filename == "run_manifest.wf-a.json"`), and **no** warning.
    - id `wf-a`, only a legacy file naming `hpdpf` → returned; WARNING names both runs. Legacy
      naming `wf-a` itself → no warning.
    - id `wf-a`, legacy read while `run_manifest.wf-b.json` is present → "unread per-run" WARNING.
    - id known, nothing present → `RunManifestMissingError`; per-run naming `wf-b` →
      `RunManifestIdentityError`.
    - no id, only `run_manifest.wf-a.json` → `None` plus "unread per-run" WARNING;
      `ARGO_WORKFLOW_NAME="   "` → behaves as unset.
    - no id, a directory at `run_manifest.json` → `OSError`.
    - `copy_run_manifest_forward(read=<per-run read>)` → output's exact contents are
      `{"run_manifest.wf-a.json"}`, byte-identical; INFO line names `run_manifest.wf-a.json`.
    - Standalone under `wf-a`:
      - a per-run source that is not valid JSON → forwarded byte-identically;
      - a per-run source naming `wf-b` → forwarded unchanged (no identity check);
      - no manifest → `RunManifestMissingError`, an ERROR log naming both directories, nothing
        written, output directory not created;
      - input and output the same directory by different spellings with a per-run file → no-op
        (bytes, mtime, exact contents unchanged).
    - POSIX-only: a per-run source at `0o640` → forwarded at `0o640`.
    - Rewrite `test_permission_error_from_the_presence_check_is_reported_cleanly` to inject
      `PermissionError` at the read (`sleap_roots_predict.run_manifest.read_run_manifest`), not
      `os.stat`, and assert the ERROR log names both directories; update `_raise_eacces_for`'s
      docstring.
  - **`tests/test_batch.py`**
    - **Contamination reproduction:** twelve staged scans; legacy `run_manifest.json` listing all
      twelve under `sleap-roots-pipeline-hpdpf`; `run_manifest.wf-a.json` listing one under
      `wf-a`; `ARGO_WORKFLOW_NAME=wf-a`.
      - `discover_scans` returns exactly that key.
      - `run_batch` with `_recording_source()` gives `[s.scan_key for s in result.scans] ==
        [that_key]`.
      - The output's top level holds `run_manifest.wf-a.json`, byte-identical, and no
        `run_manifest.json`.
    - Same tree without the per-run file → twelve scans in scope, the "names another run" WARNING,
      and the output holds exactly `run_manifest.json` (byte-identical) and no per-run file.
    - id known with sidecars but no manifest, a per-run file naming `wf-b`,
      `ARGO_WORKFLOW_NAME="../x"`, and a directory at `run_manifest.json` → `run_batch` raises
      `RunManifestMissingError` / `RunManifestIdentityError` / `ValueError` / `OSError`
      respectively, with `calls["n"] == 0` and the output directory absent.
    - Standalone `discover_scans` under a known id with no manifest → `RunManifestMissingError`;
      no id with only `run_manifest.wf-a.json` → every sidecar returned.
    - Invalid JSON in `run_manifest.wf-a.json` under `wf-a` → raises, nothing forwarded, no output
      directory (a case of `test_run_batch_never_forwards_a_manifest_that_fails_validation`).
    - Single snapshot under a per-run name: patch `batch_mod._resolve_run_manifest` to delegate,
      then rewrite or remove the source → the forwarded bytes equal the resolved bytes.
    - Missing input directory still raises `FileNotFoundError` naming the input directory, even
      with an unusable `ARGO_WORKFLOW_NAME`.
  - **CLI tests** (in-process `main([...])` with `caplog`, `run_batch` wrapped to inject
    `_recording_source()`, `clean_wandb_env`): id known with no manifest, and an identity mismatch
    → each raises its error, logs a `Batch aborted:` ERROR line, `calls["n"] == 0`, no output
    directory.
  Verify FAIL for the right reason.
- [x] 3.2 Implement:
  - **`run_manifest.py`**
    - Delete `_ManifestSnapshot` / `_read_manifest_snapshot`.
    - Add `_resolve_run_manifest(input_dir, pipeline_run_id) -> LoadedRunManifest | None` calling
      `load_run_manifest(..., allow_legacy=True)` (comment → srp#82), plus the two diagnostics.
    - `copy_run_manifest_forward(input_dir, output_dir, *, read=_UNREAD)`:
      - publishes to `output_dir / read.filename`, with the same-file check on
        `input_dir / read.filename` and the temp prefix `.{read.filename}.`;
      - standalone, it reads with `read_run_manifest(input_dir, pipeline_run_id_from_env(),
        allow_legacy=True)` inside the logged `try`;
      - a failure before the read is logged without a filename;
      - the stale-output warning stays keyed to `RUN_MANIFEST_FILENAME`.
  - **`batch.py`**
    - `discover_scans(input_dir, *, manifest=_UNREAD)`: the existence check first, then resolve if
      no manifest was passed, then scope to `manifest.manifest.scan_keys`.
    - `run_batch`: existence check → resolve once → discover → forward `loaded.read` (or `None`).
  - **`__main__.py`**: add `RunManifestError` to the except tuple.
  - **Docstrings and comments**: the module docstrings of `run_manifest.py` (drop the now-stale
    "cleanup" divergence from traits), `batch.py` and `__main__.py`; the public docstrings of
    `copy_run_manifest_forward`, `discover_scans` and `run_batch`, including their `Raises:`
    sections; `batch.py`'s "ONE read of run_manifest.json" comment; `__main__`'s inline
    staging-error comment; and the `__init__.py` package docstring.
  Verify PASS; lint clean.

## 4. Docs

- [ ] 4.1 Update:
  - **`API.md`**:
    - `run_batch` and `copy_run_manifest_forward`: per-run resolution, forwarding under the name
      read, the new raises.
    - `write_prediction_outputs`: per-writer temp names.
    - Add a `discover_scans` entry (it is public).
  - **`README.md`**:
    - The manifest section and "all writes are atomic".
    - The project tree lines for `run_manifest.py` and its tests.
    - A Configuration row for `ARGO_WORKFLOW_NAME`: set by the cluster template; leave it unset
      locally; set means a manifest is required. Do **not** add it to `.env.example`, because
      `test_env_docs.py` requires an exact set there.
    - The rollback note: unchanged before the writer flips. After the flip, this version is
      predict's rollback floor, so roll the writer back first. Per-run forwarded files need no
      cleanup and must never be glob-deleted on the shared mount.
  - **`openspec/project.md`**: the `run_manifest.py` line.
  - **`CHANGELOG.md`**, under Unreleased:
    - `### Changed (BREAKING)`: fail-loud under a run id; the non-file edge.
    - `### Fixed`: #43.
    - Amend the existing entry that calls cross-hop merging (#40) a pending follow-up.
  `codespell` clean.

## 5. Verification

- [ ] 5.1 `SRP_DEVICE=cpu uv run pytest -m "not gpu and not acceptance and not wandb" tests/`
      (ci.yml's expression, verbatim) green.
- [ ] 5.2 GPU subset under a `windows_cuda` sync (`uv sync --extra dev --extra windows_cuda`, then
      `uv run pytest -m gpu tests/`), confirming tests actually ran rather than skipped.
- [ ] 5.3 POSIX-only tests (the `0o640` forward, mode guards) run under WSL, or confirmed as
      *passed, not skipped* in the ubuntu/macOS CI logs.
- [ ] 5.4 Per-commit green: for each commit in `main..HEAD`, check it out and run 5.1's command
      (no interactive rebase).
- [ ] 5.5 `black --check .`, `ruff check sleap_roots_predict/ scripts/`, `codespell`, `uv build`
      clean; `openspec validate adopt-per-run-run-manifest-reader --strict` passes.

Post-merge work is tracked on srp#71, not here, so `/cleanup-merged` never archives an unchecked
item:
- release, then the predictor pin bump in `sleap-roots-pipeline` **after C2 (srp#89) is confirmed
  deployed**, retiring the template's "inert today" comment;
- the traits pin bump before the writer flips;
- the roadmap entry;
- the srp#71 comment;
- predict#41's in-cluster checks, on #41.
