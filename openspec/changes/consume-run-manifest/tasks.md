## 1. Dependency bump (isolated regression baseline)

Land 1.1–1.2 as a single commit — the Dockerfile's `uv sync --frozen` hard-fails if `uv.lock`
doesn't match `pyproject.toml`, so a bumped pin without its relock is never a safe standalone
commit.

- [x] 1.1 Bump `sleap-roots-contracts` from `==0.1.0a6` to `==0.1.0a7` in `pyproject.toml`.
- [x] 1.2 Relock scoped to just this dependency: `uv lock -P sleap-roots-contracts` (not a bare
      `uv lock`); confirm the `uv.lock` diff touches only the `sleap-roots-contracts` entry.
- [x] 1.3 Verify `python -c "from sleap_roots_contracts import RunManifest, RUN_MANIFEST_FILENAME;
      from sleap_roots_contracts.identity import compute_idempotency_key; print('ok')"` succeeds
      at the new pin, with no other code changes yet.
- [x] 1.4 Run the full test suite (`pytest -m "not gpu and not acceptance and not wandb and not
      parity"`) and confirm it is still green before any further edits — the regression baseline.
      (290 passed, 7 deselected.)

## 2. Manifest-scoped discovery (TDD)

Each sub-item: write the failing test first, then the minimal `discover_scans` change to pass
it, per the design doc's data-flow section.

- [x] 2.1 Test: `discover_scans` scopes to a present `run_manifest.json`'s `scan_keys` — a
      leftover sidecar outside that set is excluded (not returned, no error recorded). Then
      implement: read `input_dir / RUN_MANIFEST_FILENAME` if present, parse via
      `RunManifest.model_validate_json`, filter the `rglob` results to `scan_key in
      manifest.scan_keys` before dup-tracking (so an out-of-scope sidecar is never even
      considered for the duplicate-`scan_key` check).
- [x] 2.2 Test: no `run_manifest.json` present → `discover_scans` returns exactly what it does
      today (every sidecar found, none excluded) — a regression guard, not new behavior; existing
      fixtures/tests should need no changes to keep passing.
- [x] 2.3 Test: a manifest `scan_key` with no matching sidecar anywhere under the input directory
      becomes a `ScanInput` with `.error` set and `params=None` (reuses the existing error →
      `ScanResult(status="failed")` path — verify via `run_batch`, not just `discover_scans`,
      since that's where the status actually surfaces). Implement: after building the
      scoped+discovered set, compute `manifest.scan_keys - {discovered scan_keys}` and append a
      synthetic errored `ScanInput` per missing key.
- [x] 2.4 Test: a malformed `run_manifest.json` (invalid JSON, or valid JSON failing `RunManifest`
      validation, e.g. empty `scan_keys`) raises from `discover_scans` before any scan is
      returned/processed.
- [x] 2.5 Run the targeted test file (`pytest tests/test_batch.py -k manifest or discover`) and
      confirm all four new scenarios pass alongside the untouched existing discovery tests.
      (28 tests in test_batch.py, all passing; full suite 295 passed, 7 deselected.)

## 3. Idempotency-key skip-if-done (TDD)

**Land 3.1–3.9 as a single commit.** Tests 3.3–3.7 assert key-based skip/re-predict behavior that
doesn't exist until 3.8's loop restructuring lands — committing them separately would leave an
intermediate commit red, unlike group 2 (whose tests are independently satisfiable one at a time).

- [x] 3.1 Add `images_checksum: str = ""` to `ScanInput`, populated in `_load_scan` from
      `meta.get("images_checksum", "")`.
- [x] 3.2 Write two small helpers, private to `batch.py`:
      - `_identity_key(*, scan_key, images_checksum, params_dict, model_refs, predict_code_sha,
        predict_output_params)` — converts `model_refs` (a `dict[RootType, ModelRef]` or
        iterable of `ModelRef`) to the `list[tuple[registry_id, version, weights_checksum]]` shape
        `compute_idempotency_key` expects, recomputes `param_hash` via `compute_param_hash
        (params_dict)`, and calls `compute_idempotency_key(..., traits_code_sha="")`.
      - `_previous_identity_key(out_scan_dir, scan_key)` — reads back
        `{scan_key}.scan_metadata.json` + `{scan_key}.predictions.json` from `out_scan_dir` and
        calls `_identity_key` with their contents. **Must internally catch `OSError`,
        `json.JSONDecodeError`, and `pydantic.ValidationError`** (missing file, unreadable file,
        or present-but-corrupt/schema-invalid content) and return `None` in every one of those
        cases — this is a deliberate design choice (see design doc's ordering section), not
        reliance on the outer per-scan `try/except` added in 3.8, because a corrupt *previous*
        artifact must be treated as "changed → re-predict," not as a scan failure.
      Unit-test both helpers directly first (pure function / simple file reads — fastest
      feedback), including a case where `_previous_identity_key` is pointed at a directory with a
      hand-corrupted `{scan_key}.predictions.json` (valid JSON, invalid schema) and confirmed to
      return `None` rather than raising.
- [x] 3.3 Test: `test_rerun_skips_completed_scan` (existing) still passes essentially unchanged —
      an unchanged re-run (same sidecar, same models, same `predict_code_sha`) skips via key
      match, not just `Path.exists()`. Update its assertions/setup only as needed to reflect that
      the skip is now key-based.
- [x] 3.4 Test (new): re-run after mutating the *input* sidecar's `params` (different
      `param_hash`) between the two `run_batch` calls → the scan is re-predicted (`status="ok"`,
      manifest mtime changes), not skipped. This is the actual incident-2 fix from the design doc.
- [x] 3.5 Test (new): re-run after changing the input sidecar's `images_checksum` only (params
      unchanged) → re-predicted, not skipped.
- [x] 3.6 Test (new): re-run with a different `SRP_PREDICT_CODE_SHA` env value between the two
      `run_batch` calls → re-predicted, not skipped.
- [x] 3.7 Test (new): a scan with no prior `out_scan_dir` contents at all (first run) always
      predicts — `_previous_identity_key` returns `None`, never raises.
- [x] 3.7a Test (new): re-run against a `source` whose card for the same scan resolves to a
      different `registry_id`/`version`/`weights_checksum` than the previous run → re-predicted,
      not skipped (identity-key inputs include model refs; no other test exercises this input).
- [x] 3.7b Test (new): re-run where the existing `{scan_key}.predictions.json` is present but
      hand-corrupted (valid JSON, fails `PredictionManifest` validation) → the scan is
      re-predicted (`status="ok"`), and — critically — is NOT recorded as `failed`; assert on
      `BatchResult.scans` directly, not just that a new manifest was written.
- [x] 3.7c Test (new): a manifest `scan_key` with no sidecar (2.3's synthetic error `ScanInput`,
      `params=None`) is recorded `failed` via `run_batch` without raising — regression guard
      confirming the `scan.error` check still runs before `resolve()` (a naive "move resolve() up"
      implementation would call `resolve(None)` and raise `AttributeError` inside `choose_models`
      instead of isolating this as a per-scan failure).
- [x] 3.8 Implement the `run_batch` loop restructuring, in this exact order per scan:
      1. Check `scan.error is not None` first, unchanged from today — record `failed` and
         `continue` before anything else runs.
      2. Wrap everything else in the existing per-scan `try/except Exception` (widened to cover
         more than just `_predict_one`): `refs = worker.resolve(scan.params)`, then
         `current_key = _identity_key(...)`, then `previous_key =
         _previous_identity_key(out_scan_dir, scan.scan_key)`, then compare.
      3. `previous_key is not None and previous_key == current_key` → record `skipped`,
         `continue`. Otherwise → call `_predict_one` (passing `refs` through so it isn't
         re-resolved) and let it write outputs as today.
      Remove the old `manifest_path.exists()` check entirely.
- [x] 3.9 Run `pytest tests/test_batch.py` in full and confirm green, with special attention to
      the tests most at risk from this restructuring: `test_zero_resolved_models_is_failed`,
      `test_one_failing_scan_does_not_abort_batch`, `test_sidecar_copy_failure_leaves_no_manifest`,
      and `test_resume_mixed_skip_and_predict` — all must stay green with no behavior change.
      (36 tests in test_batch.py, all passing; full suite 304 passed, 7 deselected.)

## 4. OpenSpec validation gate

- [x] 4.1 `openspec validate consume-run-manifest --strict` — resolve any issues.
      (`Change 'consume-run-manifest' is valid`.)

## 5. Docs

- [x] 5.1 Update `CHANGELOG.md`'s `[Unreleased]` "Predict container CLI" bullet in place: replace
      "skips-if-done (existence-based resume)" with: "and per scan, when a `run_manifest.json`
      (`RunManifest`, `sleap-roots-contracts==0.1.0a7`) is staged in `input_dir`, scopes discovery
      to exactly its `scan_keys` (an out-of-scope sidecar is silently excluded); skip-if-done now
      compares a recomputed idempotency key (`compute_idempotency_key`) against the prior run's
      own artifacts, skipping only on an exact match and otherwise (re)predicting — no new
      storage." Note the accepted trade-off (resolve() now runs once per batch even on a full
      resume) in a short follow-up sentence.
- [x] 5.2 Update `API.md`'s `run_batch` prose ("skips if the manifest already exists (resume)")
      to describe the idempotency-key comparison and manifest-scoped discovery, matching the
      CHANGELOG wording from 5.1.
- [x] 5.3 Update `README.md`'s "Running the predict container" section (~line 211, "It skips a
      scan whose manifest already exists (resume)...") with the same replacement wording as 5.1,
      and add a sentence noting `run_manifest.json`-scoped discovery when the operator's pipeline
      stages one.
- [x] 5.4 Update `openspec/project.md`: the External Dependencies `sleap-roots-contracts` version
      literal (`==0.1.0a6` → `==0.1.0a7`), and the Roadmap note at the top of the file to credit
      this change with closing the `sleap-roots-predict` row of `sleap-roots-pipeline#37`,
      parallel to the existing `#15` credit for the parity harness.
- [x] 5.5 Grep sweep: search `README.md`, `API.md`, `CHANGELOG.md`, `openspec/project.md` for
      `skip|manifest|exists|resume` and confirm no stale existence-based-resume phrasing survives
      anywhere (mirrors the precedent PR's closing sweep task). (Clean — no matches.)

## 6. Pre-merge gate

- [x] 6.1 Full `/pre-merge` gate (format, lint, test, build) before opening the PR. Confirm the
      pytest invocation matches `ci.yml`'s exact marker expression (`-m "not gpu and not
      acceptance and not wandb"`), not `/pre-merge`'s own default `-m "not gpu"` (which would pull
      in flaky wandb-registry tests).
      (black/ruff/codespell PASS; 304 passed, 7 deselected via bare `pytest tests/`, which relies
      on `addopts` rather than `/pre-merge`'s own stale `-m "not gpu"` suggestion, for the same
      reason flagged here; `uv build` PASS; Docker image build skipped locally — Dockerfile
      itself is unchanged and CI's build-only PR job covers it; GPU subset: N/A, no CUDA/MPS
      accelerator on this machine — needs verification on a GPU box before merge.)
