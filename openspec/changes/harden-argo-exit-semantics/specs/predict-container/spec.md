## MODIFIED Requirements

### Requirement: Per-scan failure isolation and batch exit code

A scan whose processing fails SHALL be isolated: the runner records it with status `failed`,
continues the batch, and still produces outputs for the other scans. `run_batch` SHALL return a
`BatchResult` whose per-scan status is one of `ok` / `skipped` / `failed` and which reports
`ok` (batch-level) iff no scan failed. A scan that resolves to **zero** models across all root
types SHALL be treated as `failed` (rather than emitting an empty-artifacts manifest that the
downstream trait-extractor would reject).

The process SHALL exit with one of three driver-owned codes so an Argo step can distinguish an
isolated per-scan failure from a genuine crash:
- `0` — success: at least one scan was discovered and none failed.
- `3` — partial: at least one scan was discovered and the batch ran to completion, but one or
  more scans isolated-failed. The batch's own per-scan isolation already ran (the other scans'
  outputs are written); this exit code exists so Argo does not conflate "retry the whole batch"
  with "this is done, some scans need attention."
- *(Python's default, produced by an uncaught exception, not an explicit `return`)* `1` — every
  other failure: a pre-flight/staging error before any scan ran (a missing input directory, two
  sidecars sharing a `scan_key`, or **zero scans discovered** — an empty-but-present input
  directory, or a `run_manifest.json` scoping discovery to zero `scan_keys`), or a genuine
  pod-level crash (e.g. model-registry authentication failing before any scan is attempted). Both
  are "the batch could not meaningfully run" conditions and are not split into separate codes;
  Argo's `retryStrategy` should retry either.

Exit code `2` is deliberately NOT part of this convention: `argparse` already exits `2` on a CLI
usage error (missing/extra positional arguments), before `run_batch` ever runs. This matches the
identical convention adopted by the sibling `sleap-roots` trait-extractor driver
(`sleap-roots#259`) — both producers report numerically identical codes for numerically identical
situations, per A4's design doc §8 ask to resolve this "the same way for both."

A `run_manifest.json`-scoped batch where every listed `scan_key` has no matching sidecar is
**not** the zero-scans-discovered case: `discover_scans` still returns one (failed) entry per
listed key, so that batch ends `partial` (`3`), not the crash/staging-error code (`1`).

#### Scenario: One failing scan does not abort the batch

- **WHEN** one scan in a multi-scan batch fails (e.g. its frames are unreadable or absent) and
  the others are valid
- **THEN** the valid scans' outputs are written, that scan's status is `failed`, and the
  process exits `3`

#### Scenario: A scan resolving to zero models is failed

- **WHEN** a scan's params match no model for any root type
- **THEN** the scan's status is `failed` (no empty-artifacts manifest is written for it), the
  batch continues, and the process exits `3`

#### Scenario: All scans succeed

- **WHEN** every discovered scan predicts successfully
- **THEN** the process exits `0`

#### Scenario: Empty input directory is a staging error

- **WHEN** a present-but-empty input directory contains no `*.scan_metadata.json` (including a
  `run_manifest.json` that scopes discovery to zero `scan_keys`)
- **THEN** `run_batch` raises before constructing a `WarmModelWorker`, writes nothing, and the CLI
  exits `1` (uncaught — not a special-cased code)

#### Scenario: Missing input directory is an error

- **WHEN** the input directory path does not exist
- **THEN** the runner raises (surfaced by the CLI as exit `1`), rather than reporting
  success with no outputs

#### Scenario: A manifest scoped to missing sidecars ends partial, not a crash

- **WHEN** `run_manifest.json` lists one or more `scan_keys` with no matching sidecar anywhere
  under the input directory, and no other scans are discovered
- **THEN** `discover_scans` returns one failed entry per listed key (not an empty list), and the
  process exits `3`, not `1`

#### Scenario: A CLI usage error is unrelated to the partial/crash codes

- **WHEN** `python -m sleap_roots_predict` is invoked with a missing required argument
- **THEN** the process exits `2` via `argparse`'s own pre-existing usage-error handling, before
  `run_batch` ever runs, and this is unrelated to (does not collide in meaning with) the `0`/`1`/`3`
  driver-owned codes

### Requirement: Per-scan outputs with scan-metadata pass-through

For each predicted scan the runner SHALL write, into `<output_dir>/{scan_key}/`, the
prediction-output artifacts defined by the `prediction-output` capability (the named per-root
`.slp` files and the `{scan_key}.predictions.json` manifest, via `write_prediction_outputs`),
and SHALL additionally copy the scan's `{scan_key}.scan_metadata.json` sidecar **verbatim**
(a byte-for-byte binary copy) into the same directory, so `<output_dir>/{scan_key}/` is a
self-contained trait-extractor input tree (manifest + sidecar + `.slp` co-located). The sidecar
SHALL be copied **before** the manifest is written (the manifest is the resume marker), so the
manifest never exists without its co-located sidecar. The copy SHALL be performed atomically
(written to a temporary file in the same directory, then moved into place via `os.replace`), so
no reader can ever observe a partially-written sidecar at the final path. The runner SHALL NOT
author or modify the sidecar's contents (its `image_ids`/`images_checksum` remain the upstream
downloader's responsibility).

#### Scenario: Writes manifest, .slp, and the copied sidecar

- **WHEN** a scan is predicted into `<output_dir>`
- **THEN** `<output_dir>/{scan_key}/` contains `{scan_key}.predictions.json`, one
  `{scan_key}.model*.root*.slp` per resolved root type, and a `{scan_key}.scan_metadata.json`
  byte-identical to the input sidecar

#### Scenario: Sidecar copy is atomic

- **WHEN** the sidecar copy is interrupted before it completes
- **THEN** no partially-written file is ever visible at the final `{scan_key}.scan_metadata.json`
  path — a reader sees either nothing, a complete prior copy, or the complete new copy, never a
  truncated one

## ADDED Requirements

### Requirement: Graceful SIGTERM handling for Argo preemption

The CLI SHALL install a `SIGTERM` handler before running the batch. `run_batch` SHALL accept an
optional keyword-only `should_stop: Callable[[], bool]` parameter (default a no-op returning
`False`, so existing callers are unaffected) and SHALL check it at the top of each iteration of
its per-scan loop, stopping before starting the next scan when it returns `True` — never
interrupting a scan already in progress (there is no safe interrupt point inside sleap-nn/GPU
inference). When the CLI's handler has fired, `main()` SHALL exit `143` (`128 + SIGTERM`)
regardless of what exit code the completed-so-far scans would otherwise produce, so the
container's reported exit code honestly reflects "asked to stop," distinct from a normal
success/partial/aborted outcome.

#### Scenario: Stops at the next scan boundary, not mid-scan

- **WHEN** `should_stop` becomes `True` while the first of two scans is being predicted
- **THEN** the first scan's outputs are written completely and validly, and the second scan is
  not attempted

#### Scenario: SIGTERM exit code overrides the batch outcome

- **WHEN** the CLI's `SIGTERM` handler fires during a batch that would otherwise exit `0` or `3`
- **THEN** the process exits `143` instead

#### Scenario: No signal received leaves existing behavior unchanged

- **WHEN** the batch runs to completion without `SIGTERM` ever being received
- **THEN** the exit code is determined exactly as before (`0`/`3`/default `1`), unaffected by
  the new handler's presence
