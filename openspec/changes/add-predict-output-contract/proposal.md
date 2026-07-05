## Why

`WarmModelWorker.predict(save_dir=…)` currently writes only raw `save_dir/<root_type>.slp`
— no scan-aware naming, no manifest, no provenance. The downstream traits stage
(sleap-roots / A3-traits) needs a stable **output contract** to load predictions and
assemble a `ResultEnvelope`. This is the deferred task 9.3 slice from the archived
`add-warm-model-worker` change, and the predict→traits handoff on the A4 DAG
(`images-downloader → predict(warm) → traits → write-back`).

## What Changes

- Add a `prediction-output` capability: the on-disk artifacts predict writes per scan.
- New module `sleap_roots_predict/output_contract.py` with:
  - `PredictionArtifact` / `PredictionManifest` pydantic models (predict-local schema;
    reuse `ModelRef` from `sleap-roots-contracts`).
  - `write_prediction_outputs(...)` — pure per-scan writer: named per-root `.slp`
    (`{scan_key}.model{model_id}.root{root_type}.slp`, sleap-roots `Series`-compatible)
    plus a single combined `{scan_key}.predictions.json` (manifest + predict-side
    provenance: resolved `ModelRef`s, inference config, output params, code sha /
    container digest, and per-`.slp` checksum + file_size).
  - `ScanRequest` + `predict_and_write_batch(...)` — drive one warm worker over N scans
    (residents reused), one output subdirectory per scan.
- Fail-soft build identity: `predict_code_sha` / `predict_container_digest` from explicit
  arg → env (`SRP_PREDICT_CODE_SHA` / `SRP_PREDICT_CONTAINER_DIGEST`) → `""`.
- Add `sleap-roots` as a **test-only** dependency (`dev` extra) for the real acceptance
  test (`Series.load`); predict does **not** runtime-depend on it (no cycle).
- Export the new public API from `sleap_roots_predict/__init__.py`.

Non-breaking: `WarmModelWorker` is unchanged (the writer sits on top of it); the existing
raw-`.slp` `save_dir` behavior and its tests are untouched.

## Impact

- Affected specs: **prediction-output** (new capability; ADDED requirements only).
- Affected code: new `sleap_roots_predict/output_contract.py`; `__init__.py` (exports);
  `pyproject.toml` (`dev` extra gains `sleap-roots`) + regenerated `uv.lock`; new
  `tests/test_output_contract.py`; docs (`CLAUDE.md`, `README.md`, `API.md`,
  `CHANGELOG.md`, and the `openspec/project.md` roadmap note).
- Downstream: unblocks A3-traits' `Provenance`/`BlobRef` assembly and `Series` loading.
  Schema is predict-local now; promote to `sleap-roots-contracts` when A3-traits consumes
  it (design decision 1).
- Out of scope: upload/`BlobRef` locations (A4 step G), full `Provenance`/`ResultEnvelope`
  + `idempotency_key` (traits), `InputRef`/`images_checksum` (upstream), plant-metadata CSV
  (upstream), the serving entrypoint, and the Dockerfile build-stamp wiring.
