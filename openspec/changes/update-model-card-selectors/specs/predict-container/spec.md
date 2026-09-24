## MODIFIED Requirements

### Requirement: Per-scan failure isolation and batch exit code

A scan whose processing fails SHALL be isolated: the runner records it with status `failed`,
continues the batch, and still produces outputs for the other scans. `run_batch` SHALL return a
`BatchResult` whose per-scan status is one of `ok` / `skipped` / `failed` and which reports
`ok` (batch-level) iff no scan failed. A scan that resolves to **zero** models across all root
types SHALL be treated as `failed` (rather than emitting an empty-artifacts manifest that the
downstream trait-extractor would reject).

`run_batch` SHALL load the model-card catalog once, via `WarmModelWorker.load_catalog()`, **after**
scan discovery, the zero-scans check and the run-manifest forward-copy, and **immediately before
the first scan that has no discovery error is processed** — after that iteration's stop check,
outside the per-scan isolation. The load SHALL NOT add a stop check of its own, so the number and
order of stop checks is unchanged. A catalog that cannot be
listed — missing credentials, a registry/network error, or a registry with production artifacts
none of which is readable (per the `model-management` "Wandb Registry Source With Version Pinning"
requirement) — is therefore a batch-level error (exit `1`) rather than one isolated failure per
scan (exit `3`, which these conditions produced before this change). A catalog that lists **no
cards at all** (e.g. a registry where nothing carries the configured alias, or an empty local
source) SHALL likewise raise a batch-level `ValueError` naming the source, since no scan can be
predicted from it. It follows that the catalog
is not loaded when a stop is requested before the first processable scan, nor when every
discovered scan already carries a discovery error (no scan could use it); in both cases the batch
proceeds exactly as without the load. Scans recorded `failed` for a discovery error before the
first processable scan do not prevent a catalog failure from aborting the batch.

The process SHALL exit with one of three driver-owned codes so an Argo step can distinguish an
isolated per-scan failure from a genuine crash:
- `0` — success: at least one scan was discovered and none failed.
- `3` — partial: at least one scan was discovered and the batch ran to completion, but one or
  more scans isolated-failed. The batch's own per-scan isolation already ran (the other scans'
  outputs are written); this exit code exists so Argo does not conflate "retry the whole batch"
  with "this is done, some scans need attention."
- *(Python's default, produced by an uncaught exception, not an explicit `return`)* `1` — every
  other failure: a pre-flight/staging error before any scan ran (a missing input directory, two
  sidecars sharing a `scan_key`, a `run_manifest.json` that fails to parse or validate — including
  an empty `scan_keys` list, rejected by the "Scan discovery" requirement's own manifest
  validation before discovery ever runs — **a failure to forward `run_manifest.json` to the
  output directory, per the "Run-manifest forward-copy" requirement** — **zero scans
  discovered**: `discover_scans` returns an empty list because no sidecar exists anywhere under a
  present input directory — or **a model-card catalog with no readable production card, or no
  card at all**), or a
  genuine pod-level crash (e.g. model-registry authentication failing before any scan is
  attempted). All are "the batch could not meaningfully run" conditions and are not split into
  separate codes; Argo's `retryStrategy` should retry any of them. The CLI SHALL log a clear
  one-line message before propagating any `OSError` or `ValueError` (which, since
  `FileNotFoundError` subclasses `OSError`, and `json.JSONDecodeError` and
  `pydantic.ValidationError` both subclass `ValueError`, covers all six staging-error cases above —
  missing directory, duplicate `scan_key`, malformed manifest, forward-copy failure,
  zero-scans-discovered, and no readable production card); any other exception type is not
  specially logged and surfaces Python's default traceback. Note `OSError` is deliberately a
  **superset** of the staging set: some pod-level crashes subclass it (a model-registry network
  failure, for instance, since `requests`' exception base subclasses `OSError`), so those are now
  labelled with the same one-line message. This over-capture is accepted — the exit code is `1`
  either way and the traceback still surfaces — in exchange for the staging errors that matter
  being logged cleanly.

Exit code `2` is deliberately NOT part of this convention: `argparse` already exits `2` on a CLI
usage error (missing/extra positional arguments), before `run_batch` ever runs. This matches the
identical convention adopted by the sibling `sleap-roots` trait-extractor driver
(`sleap-roots#259`) — both producers report numerically identical codes for numerically identical
situations, per A4's design doc §8 ask to resolve this "the same way for both."

A `run_manifest.json`-scoped batch where every listed `scan_key` has no matching sidecar is
**not** the zero-scans-discovered case: `discover_scans` still returns one (failed) entry per
listed key, so that batch ends `partial` (`3`), not the crash/staging-error code (`1`).

A forward-copy failure occurring while a stop has already been requested SHALL propagate rather
than being converted into the stop-requested exit code, giving exit `1` — the exception is
raised before the driver's stop-requested check is reached. This is how every pre-flight staging
error already behaves, and `1` is retryable in Argo just as `143` is.
A catalog-load failure occurring while a stop has already been requested behaves the same way:
it propagates as exit `1`, not `143`.

#### Scenario: One failing scan does not abort the batch

- **WHEN** one scan in a multi-scan batch fails (e.g. its frames are unreadable or absent) and
  the others are valid
- **THEN** the valid scans' outputs are written, that scan's status is `failed`, and the
  process exits `3`

#### Scenario: A scan resolving to zero models is failed

- **WHEN** a scan's params match no model for any root type in a non-empty catalog
- **THEN** the scan's status is `failed` (no empty-artifacts manifest is written for it), the
  batch continues, and the process exits `3`

#### Scenario: All scans succeed

- **WHEN** every discovered scan predicts successfully
- **THEN** the process exits `0`

#### Scenario: Empty input directory is a staging error

- **WHEN** a present-but-empty input directory contains no `*.scan_metadata.json`
- **THEN** `run_batch` raises before constructing a `WarmModelWorker`, writes nothing, and the CLI
  logs a clear message and exits `1`

Note: this is distinct from a `run_manifest.json` scoping discovery to zero `scan_keys`, which is
rejected earlier by the "Scan discovery" requirement's own manifest validation (an empty
`scan_keys` list fails `RunManifest` validation before `discover_scans` returns at all) — see the
exit-code bullet above, which already separates the two raise sites. Both land on exit `1` either
way; the mechanism differs, the outcome doesn't.

#### Scenario: Missing input directory is an error

- **WHEN** the input directory path does not exist
- **THEN** the runner raises, the CLI logs a clear message before the exception propagates, and
  the process exits `1`, rather than reporting success with no outputs

#### Scenario: A failed run-manifest forward-copy is a staging error

- **WHEN** a `run_manifest.json` is present but cannot be copied into the output directory
- **THEN** the CLI logs its clear one-line staging-error message before the exception
  propagates, giving the process exit `1` — not the raw-traceback path, and not `3`, since no
  scan was predicted

#### Scenario: A catalog with no readable production card is a staging error

- **WHEN** the model-card source raises because production artifacts exist but none is readable
  (e.g. an upgraded predict deployed against a registry still holding only flat-shaped cards)
- **THEN** `run_batch` raises before any scan is predicted, writes no per-scan outputs, the CLI
  logs its one-line staging-error message, and the process exits `1` — not `3` with every scan
  failed

#### Scenario: An empty catalog is a staging error

- **WHEN** a batch with a processable scan runs against a model-card source that lists no cards
- **THEN** `run_batch` raises a `ValueError` naming the source before any scan is predicted, the
  CLI logs its one-line staging-error message, and the process exits `1` — not `3` with every
  scan failed

#### Scenario: An empty catalog with only errored scans is not loaded

- **WHEN** every discovered scan carries a discovery error and the source would list no cards
- **THEN** the catalog is not loaded, each scan is recorded `failed`, and the process exits `3`

#### Scenario: The catalog is loaded once per batch, before the first scan

- **WHEN** a multi-scan batch with processable scans runs against a model-card source
- **THEN** `list_cards()` is called exactly once, and that call precedes the first scan's model
  resolution

#### Scenario: Missing registry credentials fail the batch, not each scan

- **WHEN** a batch with processable scans runs with the default registry source and no
  `WANDB_API_KEY`
- **THEN** `run_batch` raises before any scan is attempted and the process exits `1`, not `3`

#### Scenario: A stop requested before the first scan skips the catalog load

- **WHEN** a stop has been requested before the per-scan loop begins
- **THEN** the catalog is not loaded, the stop is honored exactly as without the load, and the stop
  callback is called the same number of times as before this change

#### Scenario: A batch of only errored scans does not load the catalog

- **WHEN** every discovered scan carries a discovery error (e.g. a manifest listing only
  `scan_keys` with no sidecar)
- **THEN** the catalog is not loaded, each scan is recorded `failed`, and the process exits `3`

#### Scenario: A manifest scoped to missing sidecars ends partial, not a crash

- **WHEN** `run_manifest.json` lists one or more `scan_keys` with no matching sidecar anywhere
  under the input directory, and no other scans are discovered
- **THEN** `discover_scans` returns one failed entry per listed key (not an empty list), and the
  process exits `3`, not `1`

#### Scenario: A CLI usage error exits via argparse, before the driver runs

- **WHEN** `python -m sleap_roots_predict` is invoked with a missing required argument
- **THEN** the process exits `2` via `argparse`'s own pre-existing usage-error handling, before
  `run_batch` ever runs
