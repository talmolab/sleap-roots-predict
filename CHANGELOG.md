# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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
