# predict-container Specification

## Purpose
TBD - created by archiving change add-predict-container-cli. Update Purpose after archive.
## Requirements
### Requirement: Batch predict CLI over an input scan directory

The system SHALL provide a batch entrypoint runnable both as `python -m sleap_roots_predict`
and as a `sleap-roots-predict` console script (declared in `[project.scripts]` as
`sleap_roots_predict.__main__:main`), invoked with two positional arguments
`<input_scan_dir> <output_dir>`. It SHALL construct a **single** resident `WarmModelWorker`
for the whole batch and predict every scan discovered under `<input_scan_dir>`, reusing the
worker's cached `Predictor`s so each distinct model version is materialized and loaded at most
once for the batch. It SHALL provide a `run_batch(input_dir, output_dir, *, source=None,
predict_code_sha=None, predict_container_digest=None)` library function that the CLI wraps
(with `source=None` defaulting to the production `WandbRegistrySource`), exported from the
package's public API (`sleap_roots_predict.__all__`).

#### Scenario: Predicts every scan in the input directory

- **WHEN** `run_batch` runs over an input directory containing two scans in separate
  directories
- **THEN** it writes prediction outputs for both scans under `<output_dir>`

#### Scenario: The batch constructs a single worker

- **WHEN** `run_batch` processes a multi-scan batch
- **THEN** it constructs exactly one `WarmModelWorker` for the whole batch (so each distinct
  model version is loaded at most once), rather than one worker per scan

#### Scenario: Console script and module entrypoint invoke the same runner

- **WHEN** the package is installed
- **THEN** `[project.scripts]` declares `sleap-roots-predict = "sleap_roots_predict.__main__:main"`,
  and both `sleap-roots-predict <in> <out>` and `python -m sleap_roots_predict <in> <out>`
  invoke that same `main`

#### Scenario: run_batch is exported from the public API

- **WHEN** a caller runs `from sleap_roots_predict import run_batch`
- **THEN** the import succeeds and `"run_batch"` is in `sleap_roots_predict.__all__`

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
  sidecars sharing a `scan_key`, a `run_manifest.json` that fails to parse or validate — including
  an empty `scan_keys` list, rejected by the "Scan discovery" requirement's own manifest
  validation before discovery ever runs — or **zero scans discovered**: `discover_scans` returns
  an empty list because no sidecar exists anywhere under a present input directory), or a genuine
  pod-level crash (e.g. model-registry authentication failing before any scan is attempted). All
  are "the batch could not meaningfully run" conditions and are not split into separate codes;
  Argo's `retryStrategy` should retry any of them. The CLI SHALL log a clear one-line message
  before propagating any `FileNotFoundError` or `ValueError` (which, since `json.JSONDecodeError`
  and `pydantic.ValidationError` both subclass `ValueError`, covers all four staging-error cases
  above — missing directory, duplicate `scan_key`, malformed manifest, and zero-scans-discovered);
  any other exception type (a genuine pod-level crash outside those two types) is not specially
  logged and surfaces Python's default traceback.

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

#### Scenario: A manifest scoped to missing sidecars ends partial, not a crash

- **WHEN** `run_manifest.json` lists one or more `scan_keys` with no matching sidecar anywhere
  under the input directory, and no other scans are discovered
- **THEN** `discover_scans` returns one failed entry per listed key (not an empty list), and the
  process exits `3`, not `1`

#### Scenario: A CLI usage error exits via argparse, before the driver runs

- **WHEN** `python -m sleap_roots_predict` is invoked with a missing required argument
- **THEN** the process exits `2` via `argparse`'s own pre-existing usage-error handling, before
  `run_batch` ever runs

### Requirement: Single-channel prediction input

The runner SHALL build each scan's inference video as single-channel (greyscale) to match the
single-channel (`in_channels: 1`) cylinder root models, because sleap-nn 0.3.0 does not adapt a
video's channel count to the model (a mismatch is a runtime error, not a silent conversion).
This is an explicit cylinder-scoped assumption; model-derived channel selection (for
color/plate models) is deferred to #25.

#### Scenario: Prediction video is single-channel

- **WHEN** the runner builds the inference video for a scan
- **THEN** the video is single-channel (one channel per frame) and inference against the
  single-channel models completes without a channel-mismatch error

### Requirement: GPU container image with a real exec-form entrypoint

The root `Dockerfile` SHALL install the `linux_cuda` extra (GPU-capable torch whose wheels
bundle the CUDA runtime), set headless matplotlib (`MPLBACKEND=Agg`), and declare an exec-form
`ENTRYPOINT ["python", "-m", "sleap_roots_predict"]` that replaces the prior REPL stub (both
its `ENTRYPOINT` and its `CMD`), so the batch process is PID 1 and its exit code propagates to
the caller. The image SHALL run the batch CLI with no extra install step (dependencies baked
into the venv).

#### Scenario: Image entrypoint is the exec-form batch CLI with no leftover CMD

- **WHEN** `docker inspect` reads the built image
- **THEN** its `Entrypoint` is the exec-form `["python", "-m", "sleap_roots_predict"]` (not a
  shell-form string and not the REPL stub) and no stale `Cmd` (`["-c", "import …"]`) remains

#### Scenario: Container predicts over a mounted scan directory

- **WHEN** the image is run as `docker run <image> <in_dir> <out_dir>` over a fixture scan
  directory
- **THEN** it writes each scan's `{scan_key}.predictions.json` + `.slp` under `<out_dir>` and
  exits `0`

### Requirement: Baked predict_code_sha provenance

The image SHALL bake `SRP_PREDICT_CODE_SHA` via a Dockerfile `ARG SRP_PREDICT_CODE_SHA` →
`ENV SRP_PREDICT_CODE_SHA`, declared after the dependency-install layers so a per-commit SHA
does not bust the dependency cache. The build/push workflow (`docker-build.yml`) SHALL pass
`build-args: SRP_PREDICT_CODE_SHA=${{ github.sha }}` and tag the image with the full commit sha
(`type=sha,format=long` → `sha-<full-sha>`) so the published tag equals the baked
`predict_code_sha`. Because `write_prediction_outputs` already reads `predict_code_sha` from
that environment variable, each emitted manifest's `predict_code_sha` SHALL record the image's
build git sha (feeding the downstream idempotency key), symmetric to the traits
`SRT_TRAITS_CODE_SHA` requirement.

#### Scenario: Baked build-arg lands in the manifest

- **WHEN** the image is built with `--build-arg SRP_PREDICT_CODE_SHA=deadbeef` and run over a
  fixture scan
- **THEN** the emitted `{scan_key}.predictions.json` records `predict_code_sha == "deadbeef"`

#### Scenario: Workflow passes the commit sha and tags to match

- **WHEN** `docker-build.yml` builds the image
- **THEN** it passes `SRP_PREDICT_CODE_SHA=${{ github.sha }}` as a build-arg and publishes the
  `sha-<full-sha>` tag whose value equals the baked `predict_code_sha`

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
from the already-written `{scan_key}.predictions.json` there. When either prior file is missing,
unreadable, or present but corrupt/fails to parse (e.g. invalid JSON or schema-invalid content),
no previous key exists, and the scan SHALL be (re)predicted rather than skipped or recorded as a
failure. A scan already recorded as failed for another reason (an invalid sidecar, a scan_key/stem
mismatch, or a manifest scan_key with no matching sidecar) SHALL NOT reach model resolution or key
computation at all — that isolation check SHALL run first, unchanged from before this
idempotency-key comparison existed.

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

#### Scenario: A changed resolved model causes a re-predict rather than a skip

- **WHEN** a scan already has completed outputs, but the currently-resolved `ModelRef` for some
  root type (`registry_id`/`version`/`weights_checksum`) differs from the one recorded in its
  existing `{scan_key}.predictions.json`
- **THEN** the scan is re-predicted (status `ok`), not skipped

#### Scenario: Corrupt previous artifacts cause a re-predict, not a recorded failure

- **WHEN** a scan's existing `{scan_key}.predictions.json` is present but fails to parse or
  validate (e.g. hand-corrupted, or written by an incompatible schema)
- **THEN** no previous key is derived, the scan is (re)predicted (status `ok`, overwriting the
  corrupt state), and it is NOT recorded as a batch failure

#### Scenario: A scan already recorded as failed never reaches model resolution

- **WHEN** a scan has `.error` set (an invalid sidecar, a scan_key/stem mismatch, or a manifest
  scan_key with no matching sidecar)
- **THEN** the scan is recorded as failed without `resolve()` or any identity-key computation
  ever running for it

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

Note: on Windows, `signal.signal(signal.SIGTERM, ...)` registers without error, but real
cross-process delivery (`os.kill`) invokes `TerminateProcess` rather than the registered handler
— unlike Linux/macOS. Tests SHALL exercise the handler by calling it directly, never via
`os.kill`, so this requirement is verifiable identically across this project's CI matrix.

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

