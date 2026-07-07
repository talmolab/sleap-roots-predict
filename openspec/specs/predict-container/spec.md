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

### Requirement: Per-scan outputs with scan-metadata pass-through

For each predicted scan the runner SHALL write, into `<output_dir>/{scan_key}/`, the
prediction-output artifacts defined by the `prediction-output` capability (the named per-root
`.slp` files and the `{scan_key}.predictions.json` manifest, via `write_prediction_outputs`),
and SHALL additionally copy the scan's `{scan_key}.scan_metadata.json` sidecar **verbatim**
(a byte-for-byte binary copy) into the same directory, so `<output_dir>/{scan_key}/` is a
self-contained trait-extractor input tree (manifest + sidecar + `.slp` co-located). The sidecar
SHALL be copied **before** the manifest is written (the manifest is the resume marker), so the
manifest never exists without its co-located sidecar. The runner SHALL NOT author or modify the
sidecar's contents (its `image_ids`/`images_checksum` remain the upstream downloader's
responsibility).

#### Scenario: Writes manifest, .slp, and the copied sidecar

- **WHEN** a scan is predicted into `<output_dir>`
- **THEN** `<output_dir>/{scan_key}/` contains `{scan_key}.predictions.json`, one
  `{scan_key}.model*.root*.slp` per resolved root type, and a `{scan_key}.scan_metadata.json`
  byte-identical to the input sidecar

### Requirement: Skip-if-exists resume

The runner SHALL skip any scan whose `<output_dir>/{scan_key}/{scan_key}.predictions.json`
already exists, without re-running inference for it, so a re-run resumes a partially-completed
batch after loading models once. (Resume is existence-based in this slice; checksum-verified
skip and atomic writes are deferred together — see the design's Out-of-scope and #26.)

#### Scenario: A completed scan is skipped on re-run

- **WHEN** `run_batch` runs a second time over an output directory that already holds
  `{scan_key}/{scan_key}.predictions.json` for a scan
- **THEN** that scan is skipped (status `skipped`, not re-predicted) while any scan without an
  existing manifest is still predicted

### Requirement: Per-scan failure isolation and batch exit code

A scan whose processing fails SHALL be isolated: the runner records it with status `failed`,
continues the batch, and still produces outputs for the other scans. `run_batch` SHALL return a
`BatchResult` whose per-scan status is one of `ok` / `skipped` / `failed` and which reports
`ok` (batch-level) iff no scan failed. The process SHALL exit `0` when no scan failed and
non-zero when at least one scan failed. A scan that resolves to **zero** models across all root
types SHALL be treated as `failed` (rather than emitting an empty-artifacts manifest that the
downstream trait-extractor would reject). A **present-but-empty** input directory (no
`*.scan_metadata.json`) SHALL be a no-op that logs a warning and exits `0`; a **missing** input
directory SHALL instead be a batch-level error surfaced as a non-zero exit (distinguishing an
idle mount from a mis-configured one).

#### Scenario: One failing scan does not abort the batch

- **WHEN** one scan in a multi-scan batch fails (e.g. its frames are unreadable or absent) and
  the others are valid
- **THEN** the valid scans' outputs are written, that scan's status is `failed`, and the
  process exits non-zero

#### Scenario: A scan resolving to zero models is failed

- **WHEN** a scan's params match no model for any root type
- **THEN** the scan's status is `failed` (no empty-artifacts manifest is written for it) and the
  batch continues

#### Scenario: All scans succeed

- **WHEN** every discovered scan predicts successfully
- **THEN** the process exits `0`

#### Scenario: Empty input directory is a no-op

- **WHEN** a present-but-empty input directory contains no `*.scan_metadata.json`
- **THEN** the runner logs a warning, writes nothing, and exits `0`

#### Scenario: Missing input directory is an error

- **WHEN** the input directory path does not exist
- **THEN** the runner raises (surfaced by the CLI as a non-zero exit), rather than reporting
  success with no outputs

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

