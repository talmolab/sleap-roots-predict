## MODIFIED Requirements

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

Both the `.slp` files and the `{scan_key}.predictions.json` manifest SHALL be written
atomically: each is written to a temporary file in the same directory as its final path, then
moved into place via `os.replace`, so no reader can ever observe a partially-written file at the
final path. The manifest SHALL still be written last, after every `.slp` write completes,
preserving its established role as the resume commit-marker. The `.slp` temp write SHALL pass an
explicit format (e.g. `format="slp"`) to the underlying `sio.save_file` call rather than relying
on the temp filename's extension — `sio.save_file` infers its output format purely from the
destination filename when `format` is omitted, so a temp name that does not itself end in `.slp`
(e.g. one built by appending a `.tmp` suffix) would otherwise fail with an unknown-format error.

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

#### Scenario: Atomic write leaves no partial file visible

- **WHEN** a `.slp` or manifest write is interrupted before its final `os.replace` into place
- **THEN** no file exists at the final path with incomplete content — a reader sees either the
  complete prior version (if any) or nothing, never a truncated one

#### Scenario: Manifest is still written last

- **WHEN** `write_prediction_outputs` runs for a scan with one or more resolved root types
- **THEN** every `.slp`'s atomic write completes before the manifest's atomic write begins

#### Scenario: The .slp temp write does not depend on the temp filename's extension

- **WHEN** the writer's temporary filename for a `.slp` write does not itself end in `.slp` (e.g.
  a `.tmp`-suffixed name)
- **THEN** the write still succeeds, because `format="slp"` is passed explicitly rather than
  inferred from the filename
