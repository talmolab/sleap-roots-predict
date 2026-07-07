# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- **Param resolution** (`sleap_roots_predict.param_resolution`): `resolve_params(metadata,
  overrides=None)` maps a single Bloom `cyl_scans_extended` row (the dict bloomcli writes to
  `scans.csv`) to a `ResolvedParams` (`species`/`mode`/`age`), so `choose_models` can select
  production models from real Bloom metadata (metadata → params → model). `species_name` is
  normalized to the `ModelCard` vocabulary (lowercase passthrough; unknown species pass
  through so the registry stays the authority), `mode` routes through a one-line
  `_mode_for_scan` seam (`"cylinder"` for the cylinder stage-in path; GraviScan/multiscanner
  deferred), and `plant_age_days` is coerced to an `int` (confirmed **days**, matching the
  seeded cards' `age_min`/`age_max`). Overrides win per field with keys restricted to
  `{species, mode, age}`; override values are canonicalized like derived values, and
  blank/lossy inputs fail loud, so `param_hash` is representation-independent. Pure and
  offline — no new dependencies. New public export: `resolve_params`. See the
  `param-resolution` OpenSpec spec.
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
- **Predict container CLI** (`sleap_roots_predict.batch` + `__main__`): a warm-batch
  entrypoint — `sleap-roots-predict <input_scan_dir> <output_dir>` (also
  `python -m sleap_roots_predict`) and the `run_batch(...)` library function. Discovers scans
  (a `{scan_key}.scan_metadata.json` sidecar co-located with its frames), loads models once
  via a resident `WarmModelWorker`, and per scan skips-if-done (existence-based resume),
  predicts (single-channel video), writes the output-contract artifacts into
  `out_dir/{scan_key}/`, and copies the sidecar through so the output is a self-contained
  trait-extractor input tree. Per-scan failures are isolated; the process exits non-zero iff
  any scan failed. The root `Dockerfile` now ships a real exec-form
  `ENTRYPOINT ["python","-m","sleap_roots_predict"]` on the GPU (`linux_cuda`) stack and bakes
  the build git sha (`SRP_PREDICT_CODE_SHA` build-arg → `ENV` → manifest `predict_code_sha`);
  `docker-build.yml` tags `type=sha,format=long`. New public export: `run_batch`. See the
  `predict-container` OpenSpec spec (closes #24). Model-derived channel handling (#25) and
  Argo-readiness hardening (#26) are follow-ups.

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
