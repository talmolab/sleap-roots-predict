# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

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
  See the `prediction-output` OpenSpec spec.

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
