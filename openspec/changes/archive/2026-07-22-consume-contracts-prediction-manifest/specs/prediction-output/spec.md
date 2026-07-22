## MODIFIED Requirements

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
