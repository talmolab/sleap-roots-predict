# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- **Param resolution** (`resolve_params`): maps a single Bloom `cyl_scans_extended` row to a
  `ResolvedParams` (`species`/`mode`/`age`), so `choose_models` can select production models
  from real Bloom metadata (metadata → params → model). Re-exported from
  `sleap_roots_predict`; implemented in `sleap-roots-contracts==0.1.0a4` (predict's local
  copy is deleted — contracts is now the single source of truth across producers). **Behavior
  change vs. predict's prior local copy**: contracts hardens missing-data sentinel handling —
  `pd.NA`/`pd.NaT` species, `np.bool_`/`Decimal`/`inf` age, and non-string species (e.g. `123`)
  now raise `ValueError` instead of silently coercing into a wrong `param_hash`. No well-formed
  input's behavior changes.
- **Predict output contract** (`sleap_roots_predict.output_contract`): the per-scan
  artifacts the downstream traits stage reads. `write_prediction_outputs` writes one
  named per-root `.slp` (`{scan_key}.model{model_id}.root{root_type}.slp`, sleap-roots
  `Series`-compatible) plus a combined `{scan_key}.predictions.json` manifest — per-root
  paths + `model_id` + `plant_qr_code` and the predict-side provenance (resolved
  `ModelRef`s, effective inference config, `predict_code_sha` / `predict_container_digest`,
  and each `.slp`'s sha256 checksum + size). `predict_and_write_batch` drives one warm
  `WarmModelWorker` over many scans (one subdirectory per scan; resident predictors
  reused). New public exports: `PredictionArtifact`, `PredictionManifest`, `ScanRequest`,
  `write_prediction_outputs`, `predict_and_write_batch`. Build identity is read from
  `SRP_PREDICT_CODE_SHA` / `SRP_PREDICT_CONTAINER_DIGEST` (fail-soft to `""`). Added
  `sleap-roots` as a test-only (`dev`) dependency for the `Series.load` acceptance test.
  `PredictionArtifact`/`PredictionManifest` are implemented in
  `sleap-roots-contracts==0.1.0a5` (predict's local copies are deleted — contracts is now
  the single source of truth, mirroring `resolve_params`); `PredictionArtifact` gains a
  `kind` field (`BlobKind`, defaults to `"predictions_slp"`). See the `prediction-output`
  OpenSpec spec.
- **Predict container CLI** (`sleap_roots_predict.batch` + `__main__`): a warm-batch
  entrypoint — `sleap-roots-predict <input_scan_dir> <output_dir>` (also
  `python -m sleap_roots_predict`) and the `run_batch(...)` library function. Discovers scans
  (a `{scan_key}.scan_metadata.json` sidecar co-located with its frames), loads models once
  via a resident `WarmModelWorker`, and per scan, when a `run_manifest.json` (`RunManifest`,
  `sleap-roots-contracts==0.1.0a7`) is staged in `input_dir`, scopes discovery to exactly its
  `scan_keys` (an out-of-scope sidecar is silently excluded); skip-if-done now compares a
  recomputed idempotency key (`compute_idempotency_key`) against the prior run's own artifacts,
  skipping only on an exact match and otherwise (re)predicting — no new storage. Note:
  `resolve()` (and its one-time model-registry fetch) now runs once per batch invocation even
  when every scan is already done, which it previously did not. Predicts (single-channel
  video), writes the output-contract artifacts into `out_dir/{scan_key}/`, and copies the
  sidecar through so the output is a self-contained trait-extractor input tree — the `.slp`,
  manifest, and sidecar are all written atomically (temp file + rename), so no reader ever
  observes a partially-written file. Per-scan failures are isolated. **Exit-code contract
  (Argo-ready, #26):** `0` success, `3` partial (isolated scan failure(s), batch otherwise
  completed), Python's default `1` for every other failure (a staging error — missing input
  directory, duplicate `scan_key`, malformed `run_manifest.json`, or a now-rejected empty
  input directory — or a genuine crash), `143` (`128+SIGTERM`) if a `SIGTERM` (Argo
  preemption) stopped the batch at the next scan boundary; `2` is reserved for a CLI usage
  error (`argparse`) and is never returned by the driver's own logic. **BREAKING**: an empty
  (zero-scan) input directory previously exited `0` as a silent no-op — it now raises; the
  previous `1`="some scan failed" / `2`="staging error" split is now `3`/default-`1`. The root
  `Dockerfile` now ships a real exec-form `ENTRYPOINT ["python","-m","sleap_roots_predict"]`
  on the GPU (`linux_cuda`) stack and bakes the build git sha (`SRP_PREDICT_CODE_SHA`
  build-arg → `ENV` → manifest `predict_code_sha`); `docker-build.yml` tags
  `type=sha,format=long`. New public export: `run_batch`. See the `predict-container`
  OpenSpec spec (closes #24 and #26, and the `sleap-roots-predict` row of
  `sleap-roots-pipeline`#37). Model-derived channel handling (#25) is a follow-up.
- **A3-predict parity harness** (`sleap_roots_predict.parity`): resolves real human-labeled
  ground truth per production `ModelCard` (labels-registry lookup, bundled-labels path
  relinking, or basename search, in that priority order; unresolvable models are reported as
  explicit gaps, never silently dropped), computes sleap-nn-vs-classic-SLEAP parity metrics via
  `sleap_nn.evaluation.run_evaluation` (OKS-based matching, `distance_metrics`/
  `visibility_metrics` only — OKS-derived scores are excluded from the gate as miscalibrated
  for the root-keypoint domain), and asserts a documented, empirically-derived tolerance:
  `distance_p95` within **25% relative delta** of the classic-SLEAP reference, and
  `visibility_recall` no more than **0.10 lower** (sleap-nn scoring higher never fails).
  Measured against all 13 production `ModelCard`s (8 physically distinct weight sets) at up
  to `n=100` sampled frames each (the full resolved count where fewer resolved) — every model
  passes with headroom; see
  `docs/superpowers/specs/2026-08-04-define-parity-tolerance-results.json` for the full
  per-model report. New `parity` pytest marker (gated on `WANDB_API_KEY` + a network-share
  root, deselected by default/in CI, mirroring `gpu`/`acceptance`/`wandb`). Resolves
  sleap-roots-pipeline#15. Bumped `sleap-roots-contracts` `0.1.0a5` → `0.1.0a6` for
  `LabelCard` (used to shape the checked-in ground-truth manifest). Regenerating that report
  is a committed, reusable operation: `run_parity_harness()` loops the per-card evaluation
  over a list of `ModelCard`s, isolating one card's failure as a gap entry
  (`gap_stage="evaluation"`, distinct from a ground-truth-resolution gap's
  `gap_stage="resolution"`) instead of aborting the run, and refuses to overwrite an existing
  report when every card gapped. Driven by the committed `scripts/run_parity_harness.py`
  (`uv run python scripts/run_parity_harness.py`; lab-only — Windows + a `Z:` mapped network
  share, real registry credentials — manual/on-demand, no CI wiring).

### Changed

- **Rebuilt the inference core on sleap-nn 0.3.0.** `make_predictor` now builds a
  reusable `sleap_nn.inference.Predictor` (loaded once, reused across videos) and
  `predict_on_video` runs `predictor.predict(video, make_labels=True)`.
- `make_predictor` now loads **legacy SLEAP** models from a sanitized temporary
  copy when their config carries inert out-of-range augmentation values that
  sleap-nn 0.3.0 rejects (e.g. `brightness_min_val < 0`); the original model
  directory is never modified. See `docs/upstream/sleap-nn-legacy-brightness-issue.md`.
- Dependencies: pinned `sleap-nn==0.3.0`; `sleap-io` follows sleap-nn (`>=0.8.0,<0.9.0`);
  fixed the `linux_cuda` extra and added PyTorch index routing so CUDA extras
  resolve CUDA wheels.
- Added `SRP_DEVICE` env override (used by `"auto"` device resolution).

### Changed (BREAKING)

- **Flipped the default model source to the live production wandb registry.**
  `WandbRegistrySource` now defaults its registry to `sleap-roots-models`, and
  `WarmModelWorker(source=None)` defaults to a `WandbRegistrySource` — so with only
  `WANDB_API_KEY` set the warm worker fetches production models out-of-the-box (no other
  env var required). A missing `WANDB_API_KEY` fails loud on first use; there is no offline
  fallback. Renamed the registry env vars `SRP_WANDB_REGISTRY` → `SRP_WANDB_MODEL_REGISTRY`
  and `SRP_WANDB_ALIAS` → `SRP_WANDB_MODEL_ALIAS` (old names are no longer read), matching
  the `sleap-roots-training` producer. `list_cards()` now skips a single non-conforming
  registry artifact with a logged warning instead of aborting the whole listing.

### Removed (BREAKING)

- Removed `predict_on_h5` and `batch_predict` (they depended on the removed
  sleap-nn 0.0.x `VideoReader`). Build a `sleap_io.Video` (e.g. via
  `make_video_from_images`) and call `predict_on_video` instead.
- Renamed `make_predictor(model_path=...)` → `make_predictor(model_paths=...)`.
- `process_timelapse_experiment` no longer runs prediction — it still builds
  videos/H5/metadata, but `model_paths`/`peak_threshold`/`batch_size`/`device`
  are accepted and ignored, and `predictions_path` in the results is always
  `None`. Use `predict_on_video` directly. (Timelapse-integrated prediction is
  deferred to a future release.)
