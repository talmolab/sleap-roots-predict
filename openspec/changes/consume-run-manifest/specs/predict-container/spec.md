## RENAMED Requirements
- FROM: `### Requirement: Skip-if-exists resume`
- TO: `### Requirement: Skip-if-done resume (idempotency-key verified)`

## MODIFIED Requirements

### Requirement: Scan discovery and params from the scan-metadata sidecar

The runner SHALL discover scans by recursively globbing `*.scan_metadata.json` under the input
directory. Each scan's image frames and its sidecar SHALL reside together in a **single
dedicated directory** (the sidecar co-located with the frames it describes); the directory's
name is not significant — the `scan_key` SHALL be the sidecar's filename stem and SHALL equal
the sidecar's internal `scan_key` field. Multiple scans SHALL reside in **separate**
directories (a single-scan input is the degenerate case of one directory). A scan's frames are
the image files co-located with the sidecar, matched by extension
(`.png/.tif/.tiff/.jpg/.jpeg`) **case-insensitively**; any non-image file in that directory
(including the sidecar itself) SHALL be ignored. The scan's `ResolvedParams` SHALL be built
directly from the sidecar's normalized `params` object (`{species, mode, age}`) — the
container does not call `resolve_params` (which runs upstream) and SHALL NOT import the
`trait_extractor` package. Two sidecars resolving to the same `scan_key` anywhere in the tree
SHALL be rejected.

If a `RunManifest` (`sleap-roots-contracts`; fixed filename `RUN_MANIFEST_FILENAME`,
`"run_manifest.json"`) is present directly under the input directory, discovery SHALL be scoped
to exactly its `scan_keys`: a discovered sidecar whose `scan_key` is not in that set SHALL be
silently excluded (not returned, not recorded as an error — a leftover from a prior run is not
this run's concern), and a `scan_key` listed in the manifest with no matching sidecar anywhere
under the input directory SHALL be recorded as a failed scan (isolated, batch continues) rather
than silently omitted. If no `run_manifest.json` is present, discovery SHALL fall back to the
unscoped behavior described above (every sidecar found is discovered), unchanged from before this
manifest-awareness existed. A `run_manifest.json` that is present but fails to parse or validate
as a `RunManifest` SHALL raise (a batch-level error surfaced before any scan is processed), since
scope cannot be trusted from an invalid manifest.

#### Scenario: Discovers a scan and resolves its params

- **WHEN** a directory holds image frames and a co-located `{scan_key}.scan_metadata.json` with
  `params={"species":"rice","mode":"cylinder","age":3}`
- **THEN** discovery yields that scan with `scan_key`, its frame paths, and a `ResolvedParams`
  carrying `species=rice`, `mode=cylinder`, `age=3`

#### Scenario: Non-image files are ignored as frames

- **WHEN** a scan directory contains image frames alongside non-image files (e.g. a stray
  `.txt` and the `.scan_metadata.json` sidecar itself)
- **THEN** only the image files are collected as frames; the non-image files are not ingested

#### Scenario: Sidecar stem must match its scan_key

- **WHEN** a `{stem}.scan_metadata.json` whose internal `scan_key` differs from `stem` is
  discovered
- **THEN** that scan is recorded as failed (not silently mis-keyed) and the batch continues

#### Scenario: A sidecar with missing or incomplete params fails only that scan

- **WHEN** a discovered sidecar has no `params` object, or `params` lacking a required field
  (`species`/`mode`/`age`)
- **THEN** that scan is recorded as failed and the batch continues (other scans still written)

#### Scenario: Duplicate scan_key across the tree is rejected

- **WHEN** two `*.scan_metadata.json` files anywhere under the input directory share a
  `scan_key`
- **THEN** the runner raises rather than silently overwriting a scan's output

#### Scenario: Discovery is scoped to a present RunManifest's scan_keys

- **WHEN** `run_manifest.json` under the input directory lists `scan_keys=["scan_1009"]`, and
  the input directory also contains a leftover `scan_1010/` sidecar from a prior run
- **THEN** discovery returns only `scan_1009`; `scan_1010` is neither discovered nor reported as
  an error

#### Scenario: A manifest scan_key with no matching sidecar fails only that scan

- **WHEN** `run_manifest.json` lists a `scan_key` for which no `*.scan_metadata.json` exists
  anywhere under the input directory
- **THEN** that scan is recorded as failed (isolated; the batch continues for scans that do have
  a sidecar)

#### Scenario: No manifest present falls back to unscoped discovery

- **WHEN** the input directory contains sidecars but no `run_manifest.json`
- **THEN** discovery behaves exactly as it did before manifest-awareness existed — every
  discovered sidecar is returned, none excluded

#### Scenario: A malformed manifest raises before any scan is processed

- **WHEN** `run_manifest.json` is present but is invalid JSON, or fails `RunManifest` validation
  (e.g. an empty `scan_keys` list)
- **THEN** `discover_scans` raises and no scan is processed

### Requirement: Skip-if-done resume (idempotency-key verified)

The runner SHALL skip a scan only when its previously-written outputs are still identical in
effect to what predicting it now would produce, rather than merely checking that
`<output_dir>/{scan_key}/{scan_key}.predictions.json` exists. Both the current scan's identity
key and the previously-recorded identity key SHALL be derived via
`sleap_roots_contracts.identity.compute_idempotency_key`, over: `scan_key`, `images_checksum`,
the resolved `ModelRef`s (`registry_id`, `version`, `weights_checksum`), the params'
`param_hash`, `predict_code_sha`, and `predict_output_params`; `traits_code_sha` SHALL be passed
as a fixed empty-string placeholder (the runner never owns that value and only ever compares
against its own previously-derived key, never against a traits-computed one). The previous key
SHALL require no storage beyond what the runner already writes: `images_checksum` and `params`
(to recompute `param_hash`) come from the already-copied `{scan_key}.scan_metadata.json` sidecar
in `out_scan_dir`, and the resolved models, `predict_code_sha`, and `predict_output_params` come
from the already-written `{scan_key}.predictions.json` there. When either prior file is missing
or unreadable (a first run, or a crash left a partial write), no previous key exists, and the
scan SHALL be (re)predicted rather than skipped.

#### Scenario: An unchanged scan is skipped on re-run

- **WHEN** `run_batch` runs a second time over an output directory that already holds a
  completed scan, with the same sidecar (same `images_checksum`/`params`), the same resolved
  models, and the same `predict_code_sha` as the run that produced it
- **THEN** that scan is skipped (status `skipped`, not re-predicted) while any scan without a
  matching previous key is still predicted

#### Scenario: A changed sidecar causes a re-predict rather than a skip

- **WHEN** a scan already has completed outputs, but its current sidecar's `params` (hence
  `param_hash`) or `images_checksum` differs from what produced those outputs
- **THEN** the scan is re-predicted (status `ok`), not skipped, and its outputs are overwritten

#### Scenario: A changed predict_code_sha causes a re-predict rather than a skip

- **WHEN** a scan already has completed outputs, but the current run's `predict_code_sha` (e.g.
  `SRP_PREDICT_CODE_SHA`) differs from the one recorded in its existing
  `{scan_key}.predictions.json`
- **THEN** the scan is re-predicted (status `ok`), not skipped

#### Scenario: A first run always predicts

- **WHEN** `out_scan_dir` for a scan has no prior `{scan_key}.scan_metadata.json` or
  `{scan_key}.predictions.json` (nothing written yet)
- **THEN** the scan is predicted (no previous key exists to compare against)
