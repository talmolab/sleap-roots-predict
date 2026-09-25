# Adopt the contracts 0.1.0a9 per-run run-manifest reader

## Why

Every pipeline run shares its output directories by design (srp#37), so every run has shared one
`run_manifest.json`, into which bloomctl unions `scan_keys` and never prunes. Measured 2026-09-21:
a **1-scan request carried a 12-key manifest and write-back ingested 11 scans nobody requested**
(talmolab/sleap-roots-pipeline#71). The fix is per-run manifest identity —
`run_manifest.<pipeline_run_id>.json`, keyed to `ARGO_WORKFLOW_NAME` — shipped as a shared
resolution policy in `sleap-roots-contracts 0.1.0a9` (contracts #38, released via #39, 2026-09-22).
Readers must adopt it before the writer flips (srp design §4); traits already has
(sleap-roots #269, merged 2026-09-24). This change is predict's reader half. Predict already pins
`==0.1.0a9` (PR #45), so no dependency bump is needed.

It also fixes predict#43: three per-scan atomic writes use a temp name derived from the
destination (`dst.name + ".tmp"`), shared by any two concurrent writers of the same scan.

## What Changes

- **Manifest resolution** goes through contracts' `load_run_manifest(input_dir,
  pipeline_run_id_from_env(), allow_legacy=True)`, once per batch: per-run name first, legacy
  `run_manifest.json` as the rollout-era fallback, and — when the run id is known —
  `RunManifestMissingError` instead of unscoped discovery. A per-run manifest naming another run
  raises `RunManifestIdentityError`. With no run id (local, `local-WSL2-*`, tests) scoping is
  unchanged (only the non-regular-entry edge and one new warning apply there).
- **BREAKING**: with `ARGO_WORKFLOW_NAME` set, a missing manifest now aborts the batch
  (`RunManifestMissingError`, exit `1`) where it previously fell back to unscoped discovery. This
  also applies to direct callers of `run_batch` / `discover_scans` / `copy_run_manifest_forward`.
- **Forward-copy** republishes the exact bytes read under **the filename read** (per-run or legacy);
  byte-exact, atomic, mode-preserving, cleanup-on-failure behavior is unchanged. The standalone
  public function reads with the non-parsing `read_run_manifest`, so it still validates nothing.
- **CLI**: `RunManifestError` joins `(OSError, ValueError)` in the staging-error handler, so both
  new errors get the one-line `Batch aborted: …` log and exit `1`.
- **Diagnostics** (log-only): per-run manifests present but unread; a legacy manifest naming
  another run. The existing stale-output warning is kept.
- **BREAKING (edge)**: a directory (or other unreadable entry) at a manifest path now raises
  (exit `1`) instead of reading as absent.
- **predict#43**: the sidecar copy, `.slp` write and `predictions.json` write use a per-writer
  unique, dot-prefixed temp name in the destination directory; file modes are unchanged. Cost: a
  write killed by SIGKILL now leaves a uniquely named hidden orphan instead of a fixed name the
  next run overwrote (no consumer glob matches it).

Out of scope (rationale in the design doc): predict#44, predict#40, predict#41, srp#82.

## Impact

- Affected specs: `predict-container` (MODIFIED: scan discovery; per-scan outputs; failure
  isolation and exit code; run-manifest forward-copy), `prediction-output` (MODIFIED: pure
  per-scan writer API — atomic-write temp names).
- Affected code: `sleap_roots_predict/run_manifest.py`, `batch.py`, `__main__.py`,
  `output_contract.py`, `__init__.py` (docstring); tests `test_run_manifest.py`, `test_batch.py`,
  `test_output_contract.py`, CLI tests, `conftest.py`; docs `API.md`, `README.md` (manifest
  section, rollback note, Configuration table gains `ARGO_WORKFLOW_NAME`), `CHANGELOG.md`,
  `openspec/project.md`.
- Deploy: production-inert while every cluster staging directory holds a legacy manifest (true
  today; the fleet's writer still publishes only that name). The image MUST NOT share a deploy
  with "C2" — the predict#34 selector deploy (srp §4 step 0c, srp#89, merged 2026-09-25, cluster
  apply pending) — and deploys only after C2 is confirmed. Once bloomctl flips the writer, this
  version becomes predict's rollback floor.
- Design: `docs/superpowers/specs/2026-09-25-per-run-run-manifest-reader-design.md`; design of
  record `sleap-roots-pipeline/docs/superpowers/specs/2026-09-21-per-run-run-manifest-identity-design.md`.
