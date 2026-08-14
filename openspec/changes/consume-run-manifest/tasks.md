## 1. Dependency bump (isolated regression baseline)

Land 1.1–1.2 as a single commit — the Dockerfile's `uv sync --frozen` hard-fails if `uv.lock`
doesn't match `pyproject.toml`, so a bumped pin without its relock is never a safe standalone
commit.

- [ ] 1.1 Bump `sleap-roots-contracts` from `==0.1.0a6` to `==0.1.0a7` in `pyproject.toml`.
- [ ] 1.2 Relock scoped to just this dependency: `uv lock -P sleap-roots-contracts` (not a bare
      `uv lock`); confirm the `uv.lock` diff touches only the `sleap-roots-contracts` entry.
- [ ] 1.3 Verify `python -c "from sleap_roots_contracts import RunManifest, RUN_MANIFEST_FILENAME;
      from sleap_roots_contracts.identity import compute_idempotency_key; print('ok')"` succeeds
      at the new pin, with no other code changes yet.
- [ ] 1.4 Run the full test suite (`pytest -m "not gpu and not acceptance and not wandb and not
      parity"`) and confirm it is still green before any further edits — the regression baseline.

## 2. Manifest-scoped discovery (TDD)

Each sub-item: write the failing test first, then the minimal `discover_scans` change to pass
it, per the design doc's data-flow section.

- [ ] 2.1 Test: `discover_scans` scopes to a present `run_manifest.json`'s `scan_keys` — a
      leftover sidecar outside that set is excluded (not returned, no error recorded). Then
      implement: read `input_dir / RUN_MANIFEST_FILENAME` if present, parse via
      `RunManifest.model_validate_json`, filter the `rglob` results to `scan_key in
      manifest.scan_keys` before dup-tracking (so an out-of-scope sidecar is never even
      considered for the duplicate-`scan_key` check).
- [ ] 2.2 Test: no `run_manifest.json` present → `discover_scans` returns exactly what it does
      today (every sidecar found, none excluded) — a regression guard, not new behavior; existing
      fixtures/tests should need no changes to keep passing.
- [ ] 2.3 Test: a manifest `scan_key` with no matching sidecar anywhere under the input directory
      becomes a `ScanInput` with `.error` set (reuses the existing error → `ScanResult(status=
      "failed")` path — verify via `run_batch`, not just `discover_scans`, since that's where the
      status actually surfaces). Implement: after building the scoped+discovered set, compute
      `manifest.scan_keys - {discovered scan_keys}` and append a synthetic errored `ScanInput` per
      missing key.
- [ ] 2.4 Test: a malformed `run_manifest.json` (invalid JSON, or valid JSON failing `RunManifest`
      validation, e.g. empty `scan_keys`) raises from `discover_scans` before any scan is
      returned/processed.
- [ ] 2.5 Run the targeted test file (`pytest tests/test_batch.py -k manifest or discover`) and
      confirm all four new scenarios pass alongside the untouched existing discovery tests.

## 3. Idempotency-key skip-if-done (TDD)

- [ ] 3.1 Add `images_checksum: str = ""` to `ScanInput`, populated in `_load_scan` from
      `meta.get("images_checksum", "")` (no test needed standalone — covered by 3.3's fixtures
      already carrying `images_checksum`, and by the identity-key tests below).
- [ ] 3.2 Write a small helper (private to `batch.py`) that derives the identity key from
      arbitrary `(images_checksum, params_dict, model_refs, predict_code_sha,
      predict_output_params)` via `compute_idempotency_key(..., traits_code_sha="")`, and a
      second helper `_previous_identity_key(out_scan_dir, scan_key)` that reads back
      `{scan_key}.scan_metadata.json` + `{scan_key}.predictions.json` from `out_scan_dir` (via the
      first helper) and returns `None` if either file is missing/unreadable. Unit-test both
      helpers directly first (pure functions / simple file reads — fastest feedback), before
      wiring them into `run_batch`.
- [ ] 3.3 Test: `test_rerun_skips_completed_scan` (existing) still passes essentially unchanged —
      an unchanged re-run (same sidecar, same models, same `predict_code_sha`) skips via key
      match, not just `Path.exists()`. Update its assertions/setup only as needed to reflect that
      the skip is now key-based.
- [ ] 3.4 Test (new): re-run after mutating the *input* sidecar's `params` (different
      `param_hash`) between the two `run_batch` calls → the scan is re-predicted (`status="ok"`,
      manifest mtime changes), not skipped. This is the actual incident-2 fix from the design doc.
- [ ] 3.5 Test (new): re-run after changing the input sidecar's `images_checksum` only (params
      unchanged) → re-predicted, not skipped.
- [ ] 3.6 Test (new): re-run with a different `SRP_PREDICT_CODE_SHA` env value between the two
      `run_batch` calls → re-predicted, not skipped.
- [ ] 3.7 Test (new): a scan with no prior `out_scan_dir` contents at all (first run) always
      predicts — `_previous_identity_key` returns `None`, never raises.
- [ ] 3.8 Implement the `run_batch` loop restructuring: move `refs = worker.resolve(scan.params)`
      up before the skip decision (ahead of `_predict_one`); compute `current_key` and
      `previous_key` via 3.2's helpers; skip iff both exist and are equal; otherwise call
      `_predict_one` (passing `refs` through so it isn't re-resolved) and let it write outputs as
      today. Remove the old `manifest_path.exists()` check entirely.
- [ ] 3.9 Run `pytest tests/test_batch.py` in full and confirm green, including every pre-existing
      test (e.g. `test_resume_mixed_skip_and_predict`, `test_run_batch_writes_outputs_and_copies_
      sidecar`) with no regressions.

## 4. OpenSpec validation gate

- [ ] 4.1 `openspec validate consume-run-manifest --strict` — resolve any issues.

## 5. Docs

- [ ] 5.1 Update `CHANGELOG.md`'s `[Unreleased]` "Predict container CLI" bullet in place: replace
      "skips-if-done (existence-based resume)" with a sentence describing manifest-scoped
      discovery (`RunManifest`, `sleap-roots-contracts==0.1.0a7`) and idempotency-key-verified
      skip-if-done (recomputed from already-written artifacts, no new storage).
- [ ] 5.2 Update `API.md`'s `run_batch` prose ("skips if the manifest already exists (resume)")
      to describe the idempotency-key comparison and manifest-scoped discovery, matching the
      CHANGELOG wording.
- [ ] 5.3 Update `openspec/project.md`'s External Dependencies `sleap-roots-contracts` version
      literal (`==0.1.0a6` → `==0.1.0a7`) and, if useful, note `RunManifest`/
      `compute_idempotency_key` alongside the existing enumerated type list.

## 6. Pre-merge gate

- [ ] 6.1 Full `/pre-merge` gate (format, lint, test, build) before opening the PR.
