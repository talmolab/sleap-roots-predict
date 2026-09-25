## MODIFIED Requirements

### Requirement: Pure per-scan writer API

The system SHALL provide
`write_prediction_outputs(labels_by_root, refs_by_root, out_dir, *, scan_key,
plant_qr_code=None, inference_config, output_params, predict_code_sha=None,
predict_container_digest=None)` that writes the named `.slp` files and the combined JSON
into `out_dir` (creating it if missing) and returns the resulting `PredictionManifest`.
It SHALL raise `ValueError` when `labels_by_root` and `refs_by_root` do not cover the same
set of root types. Re-running for the same `scan_key` into the same `out_dir` SHALL
overwrite prior outputs in place: any prior `.slp` for that `scan_key` (matched by the
`{scan_key}.model…` prefix) that is no longer referenced by the new manifest is removed
only *after* the new manifest has been committed, so a changed `model_id` slug does not
leave orphaned files, and a failure during that removal can never leave a still-current
manifest referencing a file the removal only partially deleted. The writer SHALL use
`pathlib.Path` for path handling and emit path strings — `slp_path` and any path passed
across the sleap-io / sleap-roots boundary — via `Path.as_posix()` (lab convention; keeps
the manifest portable across POSIX and Windows). It SHALL NOT import or depend on
`sleap-roots` at runtime.

Both the `.slp` files and the `{scan_key}.predictions.json` manifest SHALL be written
atomically: each is written to a temporary file in the same directory as its final path, then
moved into place via `os.replace`, so no reader can ever observe a partially-written file at the
final path. Atomicity SHALL also hold **between concurrent writers** of the same `scan_key` into
a shared `out_dir`: each temporary file's name SHALL be private to the writer (not derivable from
the destination alone), so one writer's move can never publish another's incomplete bytes. The
temporary file SHALL NOT match the `{scan_key}.model…` / `.slp` pattern the stale-`.slp` removal
pass matches, nor the `{scan_key}.predictions.json` name consumers look for; a temporary file orphaned by an
uncatchable termination (SIGKILL) is therefore inert and is not reclaimed. The written files'
permissions SHALL be those a direct write produces, never a private temporary-file mode — the
downstream stage reads them as a different user on shared storage. The manifest SHALL be written
after every `.slp` write completes but *before* the
stale-`.slp` removal pass, preserving its role as the resume commit-marker: once the manifest
commit succeeds, it is already correct and complete on its own, so the stale-removal pass that
follows is purely cosmetic cleanup, never load-bearing for the manifest's correctness. The `.slp`
temp write SHALL pass an explicit format (e.g. `format="slp"`) to the underlying `sio.save_file`
call rather than relying on the temp filename's extension — `sio.save_file` infers its output
format purely from the destination filename when `format` is omitted, so a temp name that does
not itself end in `.slp` (e.g. one built by appending a `.tmp` suffix) would otherwise fail with
an unknown-format error.

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

#### Scenario: Concurrent writers of one scan use private temporary files

- **WHEN** `write_prediction_outputs` runs twice for the same `scan_key` into the same `out_dir`
- **THEN** the two runs' temporary paths for each `.slp` and for the manifest differ, and a run
  whose move fails leaves no temporary file behind *(verified via per-writer temp-path uniqueness
  rather than a live race)*

#### Scenario: Written artifacts keep a direct write's permissions

- **WHEN** the writer publishes a `.slp` and the manifest on a POSIX filesystem
- **THEN** each file's permissions equal those of a file created directly in the same directory
  by the same process, not a private temporary-file mode such as `0600`

#### Scenario: Manifest write completes before the stale-.slp removal pass

- **WHEN** `write_prediction_outputs` runs for a scan with one or more resolved root types
- **THEN** every `.slp`'s atomic write completes before the manifest's atomic write begins, and
  the manifest's atomic write completes before the stale-`.slp` removal pass runs

#### Scenario: A stale-removal failure does not corrupt an already-committed manifest

- **WHEN** the stale-`.slp` removal pass fails partway through (e.g. a locked file) after the new
  manifest has already been committed
- **THEN** the committed manifest still correctly references only this run's own artifacts,
  regardless of which stale files the failed removal did or did not delete

#### Scenario: The .slp temp write does not depend on the temp filename's extension

- **WHEN** the writer's temporary filename for a `.slp` write does not itself end in `.slp` (e.g.
  a `.tmp`-suffixed name)
- **THEN** the write still succeeds, because `format="slp"` is passed explicitly rather than
  inferred from the filename
