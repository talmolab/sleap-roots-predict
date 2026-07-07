## ADDED Requirements

### Requirement: Batch predict CLI over an input scan directory

The system SHALL provide a batch entrypoint runnable both as `python -m sleap_roots_predict`
and as a `sleap-roots-predict` console script (declared in `[project.scripts]` as
`sleap_roots_predict.__main__:main`), invoked with two positional arguments
`<input_scan_dir> <output_dir>`. It SHALL construct a single resident `WarmModelWorker` and
predict every scan discovered under `<input_scan_dir>`, reusing the worker's cached
`Predictor`s across scans so each distinct model version is loaded at most once for the whole
batch. It SHALL provide a `run_batch(input_dir, output_dir, *, source=None,
predict_code_sha=None, predict_container_digest=None)` library function that the CLI wraps
(with `source=None` defaulting to the production `WandbRegistrySource`), exported from the
package's public API.

#### Scenario: Predicts every scan in the input directory

- **WHEN** `run_batch` runs over an input directory containing two scans
- **THEN** it writes prediction outputs for both scans under `<output_dir>`

#### Scenario: Models are loaded once across the batch

- **WHEN** two scans in one batch resolve to the same model version
- **THEN** the second scan reuses the resident `Predictor` loaded for the first (verified by
  object identity), rather than reloading it

### Requirement: Scan discovery and params from the scan-metadata sidecar

The runner SHALL discover scans by recursively globbing `*.scan_metadata.json` under the
input directory. For each sidecar, the `scan_key` SHALL be the filename stem and SHALL equal
the sidecar's internal `scan_key` field (else that scan is an error); the scan's image frames
are the images co-located in the sidecar's parent directory; and the scan's `ResolvedParams`
SHALL be built directly from the sidecar's normalized `params` object (`{species, mode, age}`)
— the container does not call `resolve_params` (which runs upstream) and SHALL NOT import the
`trait_extractor` package. Two sidecars resolving to the same `scan_key` anywhere in the tree
SHALL be rejected.

#### Scenario: Discovers a scan and resolves its params

- **WHEN** the input directory holds `{scan_key}/<frames> + {scan_key}.scan_metadata.json`
  with `params={"species":"rice","mode":"cylinder","age":3}`
- **THEN** discovery yields that scan with `scan_key`, its frame paths, and a `ResolvedParams`
  carrying `species=rice`, `mode=cylinder`, `age=3`

#### Scenario: Sidecar stem must match its scan_key

- **WHEN** a `{stem}.scan_metadata.json` whose internal `scan_key` differs from `stem` is
  discovered
- **THEN** that scan is recorded as failed (not silently mis-keyed)

#### Scenario: Duplicate scan_key across the tree is rejected

- **WHEN** two `*.scan_metadata.json` files anywhere under the input directory share a
  `scan_key`
- **THEN** the runner raises rather than silently overwriting a scan's output

### Requirement: Per-scan outputs with scan-metadata pass-through

For each predicted scan the runner SHALL write, into `<output_dir>/{scan_key}/`, the
prediction-output artifacts defined by the `prediction-output` capability (the named per-root
`.slp` files and the `{scan_key}.predictions.json` manifest, via `write_prediction_outputs`),
and SHALL additionally copy the scan's `{scan_key}.scan_metadata.json` sidecar **verbatim**
into the same directory, so `<output_dir>/{scan_key}/` is a self-contained trait-extractor
input tree (manifest + sidecar + `.slp` co-located). The runner SHALL NOT author or modify
the sidecar's contents (its `image_ids`/`images_checksum` remain the upstream downloader's
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
- **THEN** that scan is skipped (not re-predicted) while any scan without an existing manifest
  is still predicted

### Requirement: Per-scan failure isolation and batch exit code

A scan whose processing raises SHALL be isolated: the runner records it as failed, continues
the batch, and still produces outputs for the other scans. The process SHALL exit `0` when no
scan failed and non-zero when at least one scan failed. An input directory in which no scans
are discovered SHALL be treated as a no-op that logs a warning and exits `0`.

#### Scenario: One failing scan does not abort the batch

- **WHEN** one scan in a multi-scan batch fails (e.g. its frames are unreadable) and the
  others are valid
- **THEN** the valid scans' outputs are written, the batch result reports the failure, and the
  process exits non-zero

#### Scenario: All scans succeed

- **WHEN** every discovered scan predicts successfully
- **THEN** the process exits `0`

#### Scenario: Empty input directory is a no-op

- **WHEN** the input directory contains no `*.scan_metadata.json`
- **THEN** the runner logs a warning, writes nothing, and exits `0`

### Requirement: Single-channel prediction input

The runner SHALL build each scan's inference video as single-channel (greyscale) to match the
single-channel (`in_channels: 1`) cylinder root models, because sleap-nn 0.3.0 does not adapt
a video's channel count to the model (a mismatch is a runtime error, not a silent
conversion). This is an explicit cylinder-scoped assumption; model-derived channel selection
(for color/plate models) is deferred to #25.

#### Scenario: Prediction runs against single-channel models

- **WHEN** the runner predicts a scan whose resolved root models expect one input channel
- **THEN** it builds the video greyscale (1-channel) and inference completes without a
  channel-mismatch error

### Requirement: GPU container image with a real exec-form entrypoint

The root `Dockerfile` SHALL install the `linux_cuda` extra (GPU-capable torch whose wheels
bundle the CUDA runtime), set headless matplotlib (`MPLBACKEND=Agg`), and declare an
exec-form `ENTRYPOINT ["python", "-m", "sleap_roots_predict"]` that replaces the prior REPL
stub, so the batch process is PID 1 and its exit code propagates to the caller. The image
SHALL run the batch CLI with no extra install step (dependencies baked into the venv).

#### Scenario: Image entrypoint is the exec-form batch CLI

- **WHEN** `docker inspect` reads the built image
- **THEN** its `Entrypoint` is the exec-form `["python", "-m", "sleap_roots_predict"]` (not a
  shell-form string and not the REPL stub)

#### Scenario: Container predicts over a mounted scan directory

- **WHEN** the image is run as `docker run <image> <in_dir> <out_dir>` over a fixture scan
  directory
- **THEN** it writes each scan's `{scan_key}.predictions.json` + `.slp` under `<out_dir>` and
  exits `0`

### Requirement: Baked predict_code_sha provenance

The image SHALL bake `SRP_PREDICT_CODE_SHA` via a Dockerfile `ARG SRP_PREDICT_CODE_SHA` →
`ENV SRP_PREDICT_CODE_SHA`, declared after the dependency-install layers so a per-commit SHA
does not bust the dependency cache. The build/push workflow (`docker-build.yml`) SHALL pass
`build-args: SRP_PREDICT_CODE_SHA=${{ github.sha }}`. Because `write_prediction_outputs`
already reads `predict_code_sha` from that environment variable, each emitted manifest's
`predict_code_sha` SHALL therefore record the image's build git sha (feeding the downstream
idempotency key), symmetric to the traits `SRT_TRAITS_CODE_SHA` requirement.

#### Scenario: Baked build-arg lands in the manifest

- **WHEN** the image is built with `--build-arg SRP_PREDICT_CODE_SHA=deadbeef` and run over a
  fixture scan
- **THEN** the emitted `{scan_key}.predictions.json` records `predict_code_sha == "deadbeef"`

#### Scenario: Workflow passes the commit sha

- **WHEN** `docker-build.yml` builds the image
- **THEN** it passes `SRP_PREDICT_CODE_SHA=${{ github.sha }}` as a build-arg, and still emits
  the `sha-<sha>` image tag
