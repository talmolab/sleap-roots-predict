# prediction-output Specification

## Purpose
TBD - created by archiving change add-predict-output-contract. Update Purpose after archive.
## Requirements
### Requirement: Named per-root prediction files

The system SHALL write one `.slp` file per predicted root type named
`{scan_key}.model{model_id}.root{root_type}.slp`, following the sleap-roots `Series`
filename convention (no underscores; `root` concatenated with the type). Because a scan's
root types may resolve to *different* models (distinct `model_id` slugs), the robust load
path is `Series.load` with the explicit per-root paths from the manifest — not the
single-`model_id` directory scanner `load_series_from_h5s`. `model_id` SHALL be a
filename-safe slug derived from the model's `registry_id` and `version` with every
character outside `[A-Za-z0-9-]` replaced by `-` (so it is slash- and dot-free).
`scan_key` is identity and SHALL NOT be mangled: the writer SHALL raise `ValueError` when
`scan_key` is empty or contains any of `.` `/` `\` `:` `*` `?` `"` `<` `>` `|`, a control
character, or leading/trailing whitespace (so the value is safe as a single path segment
on both POSIX and Windows and preserves the `series_name = filename.split(".")[0]`
invariant). Each written `.slp` SHALL be reloadable via `sio.load_file`.

#### Scenario: Writes a reloadable named .slp per root type

- **WHEN** the writer runs for a scan whose worker resolved `primary` and `lateral` models
- **THEN** it writes `{scan_key}.model{model_id}.rootprimary.slp` and
  `{scan_key}.model{model_id}.rootlateral.slp`, each reloadable via `sio.load_file` with
  labeled frames

#### Scenario: Rejects a non-filename-safe scan_key

- **WHEN** the writer is called with a `scan_key` that is empty or contains a reserved
  character (`.`, a path separator, `:`, `*`, `?`, `"`, `<`, `>`, `|`, a control char, or
  leading/trailing whitespace)
- **THEN** it raises `ValueError` and writes no files

### Requirement: Combined per-scan manifest and provenance sidecar

The system SHALL write a single `{scan_key}.predictions.json` per scan that serializes a
`PredictionManifest`: `schema_version`, `scan_key`, `plant_qr_code` (defaulting to
`scan_key` when not given), and a list of per-root `PredictionArtifact` records. Each
`PredictionArtifact` SHALL carry `kind` (a `BlobKind`, defaulting to `"predictions_slp"`),
`root_type`, `model_id`, the full `ModelRef` (from `sleap-roots-contracts`), `slp_path` (the
basename as a POSIX-style string via `Path.as_posix()`, relative to the manifest's
directory), `checksum` (sha256 hex of the `.slp`), and `file_size` (bytes). The manifest
SHALL also record the predict-side provenance: `predict_inference_config`,
`predict_output_params`, `predict_code_sha`, and `predict_container_digest`. The written
JSON SHALL reload and validate back into an equivalent `PredictionManifest`. When no root
types are resolved, `artifacts` SHALL be an empty list. `PredictionArtifact` and
`PredictionManifest` are defined by `sleap-roots-contracts`' `prediction-manifest-contract`
capability and imported here, not defined locally in this repo.

#### Scenario: Manifest maps each root type to its file, model_id, and full ModelRef

- **WHEN** the writer runs for a scan with resolved `primary` and `lateral` models
- **THEN** `{scan_key}.predictions.json` contains one artifact per root type whose
  `slp_path` basename matches the written `.slp`, whose `model_id` matches the filename
  slug, and whose `model` round-trips to the resolved `ModelRef`

#### Scenario: Checksums and sizes match the written files

- **WHEN** the manifest is reloaded and each artifact's `checksum`/`file_size` is compared
  to the on-disk `.slp`
- **THEN** each `checksum` equals the file's recomputed sha256 hex and each `file_size`
  equals the file's byte length

#### Scenario: Zero resolved roots yields an empty artifacts list

- **WHEN** the writer runs for a scan where no root type resolved to a model
- **THEN** it still writes `{scan_key}.predictions.json` with `artifacts` equal to `[]`

#### Scenario: Manifest JSON round-trips to an equivalent model

- **WHEN** the written `{scan_key}.predictions.json` is read back and validated into a
  `PredictionManifest`
- **THEN** it equals the manifest the writer returned (all fields, including each
  artifact's `ModelRef`)

#### Scenario: Artifact kind defaults to predictions_slp

- **WHEN** the writer constructs a `PredictionArtifact` for a written `.slp` without
  specifying `kind` explicitly
- **THEN** the artifact's `kind` equals `"predictions_slp"` in both the returned manifest and
  the written JSON

### Requirement: Fail-soft build identity

The writer SHALL record `predict_code_sha` and `predict_container_digest` from explicit
arguments when provided, otherwise from the environment variables `SRP_PREDICT_CODE_SHA`
and `SRP_PREDICT_CONTAINER_DIGEST`, otherwise the empty string. It SHALL NOT raise when
these are absent.

#### Scenario: Explicit argument takes precedence

- **WHEN** `write_prediction_outputs` is called with an explicit `predict_code_sha`
- **THEN** the manifest records that exact value regardless of the environment

#### Scenario: Environment fallback

- **WHEN** no `predict_container_digest` argument is given but
  `SRP_PREDICT_CONTAINER_DIGEST` is set
- **THEN** the manifest records the environment value

#### Scenario: Absent identity records empty strings

- **WHEN** neither the argument nor the environment variable is set
- **THEN** the manifest records `""` for that field and no error is raised

### Requirement: Pure per-scan writer API

The system SHALL provide
`write_prediction_outputs(labels_by_root, refs_by_root, out_dir, *, scan_key,
plant_qr_code=None, inference_config, output_params, predict_code_sha=None,
predict_container_digest=None)` that writes the named `.slp` files and the combined JSON
into `out_dir` (creating it if missing) and returns the resulting `PredictionManifest`.
It SHALL raise `ValueError` when `labels_by_root` and `refs_by_root` do not cover the same
set of root types. Re-running for the same `scan_key` into the same `out_dir` SHALL
overwrite prior outputs in place: the manifest is replaced and any prior `.slp` for that
`scan_key` (matched by the `{scan_key}.model…` prefix) is removed first, so a changed
`model_id` slug does not leave orphaned files. The writer SHALL use `pathlib.Path`
for path handling and emit path strings — `slp_path` and any path passed across the
sleap-io / sleap-roots boundary — via `Path.as_posix()` (lab convention; keeps the
manifest portable across POSIX and Windows). It SHALL NOT import or depend on
`sleap-roots` at runtime.

#### Scenario: Writer returns a manifest and writes the artifacts

- **WHEN** `write_prediction_outputs` is called with aligned `labels_by_root` and
  `refs_by_root`
- **THEN** it writes the per-root `.slp` files and `{scan_key}.predictions.json` into
  `out_dir` and returns a `PredictionManifest` describing them

#### Scenario: Mismatched label and ref root types raise

- **WHEN** `labels_by_root` and `refs_by_root` cover different root types
- **THEN** the writer raises `ValueError`

#### Scenario: Re-running overwrites prior outputs in place

- **WHEN** `write_prediction_outputs` runs into an `out_dir` that already holds a prior
  manifest and `.slp` files for the same `scan_key`
- **THEN** it overwrites them in place and the reloaded manifest reflects the new run

#### Scenario: A changed model on re-run does not orphan the prior .slp

- **WHEN** a scan is re-run with a different model for a root type (a new `model_id` slug)
- **THEN** the prior `.slp` for that `scan_key` is removed, leaving only the current run's
  files

### Requirement: Batch prediction-and-write over a warm worker

The system SHALL provide `predict_and_write_batch(worker, requests, out_dir, *,
predict_code_sha=None, predict_container_digest=None)` that drives a single
`WarmModelWorker` over an iterable of `ScanRequest`s, writing one output subdirectory per
scan (`out_dir/{scan_key}/`), reusing the worker's resident `Predictor`s across scans, and
returning one `PredictionManifest` per scan. A `ScanRequest` SHALL carry `scan_key`,
`video`, `params`, and optional `plant_qr_code` and `overrides`. It SHALL raise
`ValueError` if two requests share a `scan_key` (which would otherwise silently overwrite
a scan's subdirectory).

#### Scenario: Batch writes one subdirectory of artifacts per scan

- **WHEN** `predict_and_write_batch` runs over two `ScanRequest`s
- **THEN** it creates `out_dir/{scan_key}/` for each scan containing that scan's `.slp`
  files and `{scan_key}.predictions.json`, and returns a `PredictionManifest` per scan

#### Scenario: Resident predictors are reused across scans

- **WHEN** two scans in one batch resolve to the same model version
- **THEN** the second scan reuses the resident `Predictor` instance loaded for the first
  (verified by object identity), rather than reloading it

#### Scenario: Per-scan override is honored

- **WHEN** a `ScanRequest` carries an explicit `overrides` mapping a root type to a
  `ModelRef`
- **THEN** that scan's manifest records the overridden `ModelRef` for that root type

#### Scenario: Duplicate scan_key in a batch is rejected

- **WHEN** `predict_and_write_batch` is given two `ScanRequest`s with the same `scan_key`
- **THEN** it raises `ValueError`

### Requirement: Series-loadable output verified without mocks

The produced artifacts SHALL load through the downstream consumer
`sleap_roots.Series.load(series_name=scan_key, ...)`, passing each resolved root type's
`.slp` path (e.g. `primary_path` / `lateral_path`; `crown_path` is `None` when no crown
model resolved) as a `Path.as_posix()` string, without error and expose the predicted
labels for the resolved root types. This SHALL be verified by a real, non-mocked test that
drives the warm worker over the vendored native + legacy models (no mocking of the
sleap-nn / sleap-io boundary). `sleap-roots` SHALL be a test-only dependency and the test
SHALL skip cleanly when it is not importable.

#### Scenario: Output loads via sleap-roots Series.load

- **WHEN** the writer's output for a scan is passed to `sleap_roots.Series.load` with the
  resolved root types' `.slp` paths (primary and lateral for the vendored models;
  `crown_path=None`)
- **THEN** the `Series` loads without error and its `primary_labels` / `lateral_labels`
  are populated

#### Scenario: Acceptance test skips cleanly without sleap-roots

- **WHEN** the test suite runs in an environment where `sleap_roots` is not importable
- **THEN** the `Series.load` test skips with a message rather than failing

