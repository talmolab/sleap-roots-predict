## Why

The prediction core in `sleap_roots_predict/predict.py` was written against **sleap-nn 0.0.1** and is broken against the current backend. sleap-nn's entire inference surface was rewritten between 0.0.1 and the latest release, **0.3.0** (2026-06-29): the old `Predictor.from_model_paths(...)` → `VideoReader(video)` → `predictor.predict(video_reader)` flow no longer exists (`sleap_nn/predict.py` was removed, `make_pipeline` was removed from `Predictor`). Every prediction test **mocks** the old boundary, so CI stays green against an API that no longer exists — false confidence. `sleap-io` also jumps from the resolved 0.5.1 to the `>=0.8.0,<0.9.0` that sleap-nn 0.3.0 requires.

This is the first slice of roadmap tier **A3-predict**. It rebuilds the inference core on real sleap-nn 0.3.0 with genuine (no-mock) tests, so the deferred CLI, warm-worker, and parity slices build on a correct foundation.

## What Changes

- **BREAKING:** Rebuild `make_predictor` and `predict_on_video` on the sleap-nn 0.3.0 API: `Predictor.from_model_paths(model_dirs, device=, batch_size=, peak_threshold=)` (keyword-only after `model_paths`) → `predictor.predict(video, make_labels=True)` with the `sio.Video` passed directly. The `Predictor` is persistent (loaded once, reused across videos) to seat the later warm-worker slice.
- **BREAKING:** Remove `predict_on_h5` and `batch_predict` (backward-compat cruft; they depend on the removed `VideoReader` import). Remove the prediction branch and predict-function imports from `plates_timelapse_experiment` — it continues to build videos/H5/metadata but no longer predicts. Real timelapse-prediction rework is **deferred to a future PR**; the cylinder-data path is tested directly via `predict_on_video` + the acceptance test, not through the orchestrator.
- Replace all **mocked** prediction tests with **real** no-mock tests running actual CPU inference against **vendored minimal sleap-nn models** (bottom-up, to mirror production): a native-format model (`best.ckpt` + `training_config.yaml`) and a **legacy SLEAP UNet** model (`training_config.json` + `best_model.h5`) — the exact format the production root models use. Delete `tests/test_predict.py`.
- **BREAKING (deps):** Pin `sleap-nn==0.3.0`; re-lock `uv.lock`. Keep `sleap-io` as an explicit but **unpinned** direct dependency — sleap-nn 0.3.0 already governs it to `>=0.8.0,<0.9.0`, so pinning here would only duplicate that bound.
- Fix the `linux_cuda` extra (`torch-cpu` → `torch-cuda128`) and add `[[tool.uv.index]]` (`pytorch-cpu`, `pytorch-cu128`) + `[tool.uv.sources]` torch routing so CUDA extras resolve CUDA wheels (currently a no-op → CPU-only torch on Windows).
- Add local **GPU tests** (`@pytest.mark.gpu`) runnable via `uv run pytest -m gpu`, and a **CI-skipped acceptance test** (`@pytest.mark.acceptance`) gated on `SRP_CYLINDER_DIR` / `SRP_MODEL_DIRS` that runs the full image-dir → video → prediction path on real cylinder data. Register both markers in `pyproject.toml`.
- Add **legacy SLEAP config sanitization** in `make_predictor`: load from a sanitized temporary copy when a legacy `training_config.json` carries inert out-of-range augmentation values that sleap-nn 0.3.0 rejects (currently `brightness_min_val < 0`), never mutating the original. This unblocks production root models (verified: the `lateral` model loads + predicts after sanitization). Upstream fix drafted in `docs/upstream/sleap-nn-legacy-brightness-issue.md`.

## Impact

- Affected specs: `prediction` (new capability).
- Affected code: `sleap_roots_predict/predict.py`, `sleap_roots_predict/__init__.py`, `sleap_roots_predict/plates_timelapse_experiment.py` (remove prediction branch/imports; prediction within the timelapse flow is deferred), `sleap_roots_predict/video_utils.py` (verify `sio.Video.from_filename` under sleap-io 0.8), `pyproject.toml`, `uv.lock`, `.github/workflows/ci.yml`, `tests/` (new `test_predict.py`, `test_gpu.py`, `test_acceptance.py`, vendored `tests/assets/`).
- Open risk surfaced, not assumed away: whether production legacy SLEAP UNet bottom-up models load under sleap-nn 0.3.0. A vendored legacy minimal model tests this hermetically in CI; the acceptance test confirms it on the real models. A load failure is a finding feeding the parity slice, not a defect of this slice.
- Grounding design: `docs/superpowers/specs/2026-07-02-rebuild-inference-core-design.md`.
