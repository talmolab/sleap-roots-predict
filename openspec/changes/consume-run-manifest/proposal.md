## Why

`sleap-roots-pipeline` issue #37 documents two contamination/staleness incidents from PR #33's
testing: (1) a leftover scan directory from a prior run got reprocessed by predict's
directory-wide `discover_scans` scan even though it wasn't in the current run's scope, and (2) a
corrected sidecar failed to propagate on a later run because `run_batch`'s skip-if-done is a
plain `Path.exists()` check, not an identity comparison. `sleap-roots-contracts` 0.1.0a7 now ships
`RunManifest` (written by `bloomctl` into the same shared, already-mounted staging directory as
the per-scan sidecars) specifically to close incident (1); the design doc for this change
(`docs/superpowers/specs/2026-08-14-consume-run-manifest-design.md`) also closes incident (2)
using data predict already has on disk, no new storage or contracts change required.

## What Changes

- Bump `sleap-roots-contracts` pin from `==0.1.0a6` to `==0.1.0a7` in `pyproject.toml` (relock
  scoped to that dependency); `RunManifest` doesn't exist before `a7`.
- `discover_scans` reads an optional `run_manifest.json` (`RunManifest`: `schema_version`,
  `pipeline_run_id`, `scan_keys`; fixed filename via the imported `RUN_MANIFEST_FILENAME`
  constant) from `input_dir`:
  - **Present**: scope discovery to exactly `scan_keys`. A sidecar on disk whose `scan_key` is
    not in that set is silently excluded (not discovered, not reported — the contamination case
    this change fixes). A `scan_key` in the manifest with no matching sidecar on disk becomes an
    isolated failed scan (reuses the existing `ScanInput.error` → `ScanResult(status="failed")`
    path; the batch continues).
  - **Absent**: unchanged — full unscoped `rglob`, exactly today's behavior. Every existing
    local/standalone/test caller that doesn't stage a manifest is unaffected.
  - **Malformed** (present but invalid JSON / fails `RunManifest` validation): raises, a
    batch-level abort before any scan is touched — mirrors the existing duplicate-`scan_key`
    raise, since scope itself can't be trusted.
- `run_batch`'s skip-if-done becomes a real identity comparison instead of `Path.exists()`.
  `worker.resolve(scan.params)` moves up from inside `_predict_one` (cheap — matches against
  already-listed `ModelCard`s, no network/download) so the resolved `ModelRef`s are available
  before the skip decision. The "current" and "previous" idempotency keys are both derived via
  `sleap_roots_contracts.identity.compute_idempotency_key` (imported directly — present in the
  installed package, not re-exported in `__all__`, `traits_code_sha=""` fixed placeholder since
  predict never owns that value and only ever compares against its own prior key). The "previous"
  key needs **no new storage anywhere**: every input is already recoverable from the
  already-copied sidecar (`images_checksum`, and `params` to recompute `param_hash`) and the
  already-written `{scan_key}.predictions.json` (`artifacts[].model`, `predict_code_sha`,
  `predict_output_params`) sitting in `out_scan_dir` from the prior run. Missing or unreadable
  prior artifacts (first run, or a crash mid-write) yield no previous key, which is treated as
  "changed" — the scan is (re)predicted rather than silently skipped on ambiguous state.

## Impact

- Affected specs: `predict-container` (MODIFIED: scan discovery; RENAMED + MODIFIED:
  skip-if-exists resume → skip-if-done resume, idempotency-key verified)
- Affected code: `pyproject.toml`, `uv.lock`, `sleap_roots_predict/batch.py`,
  `tests/test_batch.py`
- No behavior change for any caller that never stages a `run_manifest.json` (the entire existing
  test suite and any standalone/local usage) — scoping and the stricter skip-if-done comparison
  only activate what a manifest and a changed identity actually require; an unchanged re-run
  still skips exactly as before.
- Out of scope (tracked separately): `sleap-roots`/traits consuming the manifest (separate
  repo/handoff); write-back's identical unscoped-glob bug (bloom #678, fixed in `bloomcli`); any
  change to `sleap-roots-contracts` (not needed — see design doc).
- Resolves the `sleap-roots-predict` row of `sleap-roots-pipeline`#37's cross-repo idempotency
  chain (design doc: that repo's
  `docs/superpowers/specs/2026-08-03-manifest-scoped-processing-redesign.md`).
