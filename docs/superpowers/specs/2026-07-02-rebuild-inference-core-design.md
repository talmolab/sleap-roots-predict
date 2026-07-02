# Design: Rebuild inference core on sleap-nn 0.3.0

**Date:** 2026-07-02
**Change-id:** `rebuild-inference-core`
**Roadmap tier:** A3-predict (first slice)
**Status:** Approved (brainstorming), pending OpenSpec proposal

## Problem

The current [`predict.py`](../../../sleap_roots_predict/predict.py) was written against **sleap-nn 0.0.1** and is broken against the current backend. Between 0.0.1 and the latest release (**0.3.0**, published 2026-06-29) the entire inference surface was rewritten. Concretely:

- The old flow `Predictor.from_model_paths(...)` → `VideoReader(video)` → `predictor.predict(video_reader)` no longer matches the API. `sleap_nn/predict.py` was removed (demoted to `legacy_predict.py`); `make_pipeline` was removed from the new `Predictor`; the persistent-predictor flow is now `Predictor.from_model_paths(...)` → `predictor.predict(source, make_labels=True)` with the video passed directly.
- Every prediction test **mocks** the old sleap-nn/sleap-io boundary (`Predictor.from_model_paths`, `VideoReader`, `sio.save_file`). The mocks assert a signature that no longer exists, yet CI stays green — a false-confidence trap. The one real-model test is `@pytest.mark.gpu` and self-skips.
- `sleap-nn` is not a hard dependency; it lives only in unpinned platform extras. `sleap-io` resolves to **0.5.1**, but sleap-nn 0.3.0 requires **`sleap-io>=0.8.0,<0.9.0`**.
- The `[project.optional-dependencies]` extra `linux_cuda = ["sleap-nn[torch-cpu]"]` is mislabeled (CPU torch on a CUDA extra).
- This repo declares no `[tool.uv.sources]`/`[[tool.uv.index]]`, so the torch-variant extras are **no-ops**: all extras resolve the same `torch` from PyPI. On Windows that PyPI wheel is CPU-only, so local GPU inference does not work.

This is the "sleap-nn backend has changed a lot" the work was scoped around.

## Scope

**In scope (this slice):**
1. Rebuild the inference core on the sleap-nn 0.3.0 API.
2. Replace all mocked prediction tests with **real** (no-mock) tests running actual CPU inference against vendored minimal models.
3. Pin dependencies to latest compatible versions; fix the extras; add uv index plumbing so CUDA extras work.
4. Add easy-to-run local **GPU tests**.

**Explicitly deferred** to follow-on OpenSpec proposals (each its own slice):
- CLI `main.py` matching the pipeline container contract (`python main.py <images> <models> <output>` → `.slp` per root type).
- Warm long-running GPU worker (model resident across scans).
- Prediction-parity gate vs. the current pipeline (keypoint RMSE / trait-delta tolerance on a reference scan set).

## Design

### Public API (lean, persistent predictor)

Two functions in `sleap_roots_predict.predict`:

```python
make_predictor(model_paths, peak_threshold=0.2, batch_size=4, device="auto") -> Predictor
predict_on_video(predictor, video, save_path=None) -> Path | sio.Labels
```

- **`make_predictor`** resolves `device="auto"` ourselves (`cuda` → `mps` → `cpu` via a lazy `torch` probe), validates each model dir exists (`FileNotFoundError` otherwise), then calls
  `Predictor.from_model_paths(model_dirs, device=resolved, batch_size=batch_size, peak_threshold=peak_threshold)`.
  Note: in 0.3.0 `from_model_paths` is keyword-only after `model_paths` and defaults `device="cpu"`. Returns the **persistent** `Predictor` (loaded once, reused across videos — this is what the later warm-worker slice builds on).
- **`predict_on_video`** calls `labels = predictor.predict(video, make_labels=True)` (the `sio.Video` is passed directly as the first positional `source`). If `save_path` is given, `sio.save_file(labels, save_path)` and return the `Path`; otherwise return the `sio.Labels`.
- **Removed:** `predict_on_h5` and `batch_predict` (backward-compat cruft). The orchestrator's `save_h5` branch will predict on the `sio.Video` built by `make_video_from_images`, regardless of whether an H5 is also written — decoupling storage format from inference.

### Real tests (no mocks)

Vendor sleap-nn 0.3.0 test fixtures into `tests/assets/`:
- `minimal_instance_centroid/` (`best.ckpt` ~551 KB + `training_config.yaml`)
- `minimal_instance_centered_instance/` (`best.ckpt` ~648 KB + `training_config.yaml`)
- `centered_pair_small.mp4` (~1.0 MB)

Top-down pair chosen because sleap-roots models are top-down; ~2.2 MB total in git is acceptable.

Tests assert real behavior against actual CPU inference:
- `make_predictor(model_dirs)` returns a live `Predictor`.
- `predict_on_video(predictor, video)` returns a real `sio.Labels` containing `PredictedInstance`s.
- With `save_path`, the written `.slp` round-trips (`sio.load_file` succeeds and has frames).
- Bad model dir raises `FileNotFoundError`.
- Device `"auto"` returns a valid device string on the host (asserted, not mocked).

The entire mocked `tests/test_predict.py` is **deleted** and replaced.

### Acceptance test on real cylinder images (guarded, CI-skipped)

A separate `@pytest.mark.acceptance` test verifies the rebuilt core end-to-end on **real** data before merge, without coupling CI to large assets or the model registry. It is **skipped unless** environment variables point to local data:

- `SRP_CYLINDER_DIR` — a directory of real cylinder/plate timelapse `.tif` frames.
- `SRP_MODEL_DIRS` — one or more real root-model directories (top-down: centroid + centered_instance), os-pathsep-separated.

When set, the test drives the true pipeline path: `make_video_from_images(SRP_CYLINDER_DIR frames)` → `make_predictor(SRP_MODEL_DIRS)` → `predict_on_video(...)`, then asserts a real `sio.Labels` with `PredictedInstance`s and writes a `.slp` to a temp dir. When unset, it skips with a message explaining how to enable it. It never runs in CI (no runner sets those vars); locally it is one command: `uv run pytest -m acceptance`.

**Verified real acceptance data** (`Z:\users\eberrigan\20260522_Suyash_Patil_Arabidopsis_PGM1-PAC-EFFECT_EXP1`, a full pipeline-run dir):
- Images: `.jpg` frames under `images_downloader_output/images/Wave0/Day21_2026-04-28/<plate>/` (per-plate timelapse subdirs, e.g. `ARB103_1P_R1`). So the acceptance test must accept a configurable image pattern (`*.jpg`), not assume `*.tif`.
- Models: legacy SLEAP **bottom-up** zips in `models_downloader_output/` — `model_paths.csv` maps `primary → 240611_102513.multi_instance.n=743.zip`, `lateral → 240130_140452.multi_instance.n=337.zip`. Each zip contains `training_config.json` + `best_model.h5` (SLEAP `.multi_instance`, UNet). One model per root type; **not** top-down. `SRP_MODEL_DIRS` points at the extracted model dirs (one per root type).

**Open question, deliberately surfaced:** whether these legacy SLEAP UNet bottom-up models load under sleap-nn 0.3.0 is *unknown*. sleap-nn 0.3.0 loads native (`training_config.yaml`+`best.ckpt`) or legacy SLEAP (`training_config.json`+`best_model.h5`, UNet only). This harness is built now; it runs once real models are supplied. A **load failure is a valid, expected outcome** that becomes an explicit finding feeding the deferred parity slice — not a blocker for this slice's hermetic tests.

### Dependencies / CI

- Pin `sleap-nn==0.3.0`; set `sleap-io>=0.8.0,<0.9.0` (0.8.0 is both the latest sleap-io release and exactly what sleap-nn 0.3.0 requires).
- Fix extras: `linux_cuda` → `sleap-nn[torch-cuda128]` (matches `windows_cuda`); `macos` and `cpu` are already correct (markerless CPU/MPS wheel).
- Add `[[tool.uv.index]]` (`pytorch-cpu`, `pytorch-cu128`) and `[tool.uv.sources]` routing torch for the CUDA extras, mirroring sleap-nn. Without this, `--extra windows_cuda`/`linux_cuda` silently resolve CPU wheels. Re-lock (`uv.lock`).
- Ensure the CI **test** job installs the `cpu` extra so real inference runs on all non-GPU runners (already the case: `ci.yml` uses `uv sync --extra dev --extra cpu`).
- Register the `acceptance` pytest marker in `pyproject.toml` (alongside `gpu`); both are deselected on default/CI runs.

**Verified nuance (do not overstate):** the current `linux_cuda → torch-cpu` mislabel is *cosmetically* wrong but functionally harmless on the Linux self-hosted GPU runner, because this repo has no index routing and PyPI's default **Linux** `torch` wheel is already CUDA-enabled (the locked `manylinux_2_28_x86_64` wheel pulls `nvidia-cu12-*`). The real breakage is **local Windows GPU**: PyPI's Windows `torch` wheel is CPU-only, so the index plumbing above is what actually enables local Windows CUDA.

### Local GPU tests

New `tests/test_gpu.py`, all `@pytest.mark.gpu`:
1. An accelerator is present (`torch.cuda.is_available()`), else skip with a clear message.
2. `make_predictor(model_dirs, device="cuda")` builds and its model reports a CUDA device; `predict_on_video` on the vendored fixture runs on CUDA and returns real `sio.Labels`.

Run locally with **`uv run pytest -m gpu`**. On Windows+CUDA: `uv sync --extra dev --extra windows_cuda` then that command. GPU tests already run automatically on the CI self-hosted-gpu and mac runners (`ci.yml` runs the full suite there; other runners use `-m "not gpu"`). This will be documented as a canonical dev command.

## Risks

- The `sleap-io` 0.5.1 → 0.8.x bump can ripple into `video_utils`/`plates_timelapse_experiment` (e.g. `sio.Video.from_filename(..., grayscale=...)`). This slice stays focused on making the prediction path + its video input correct and green; breaks in unrelated `video_utils` functions are noted but not fixed here unless prediction needs them.
- Vendored `.ckpt` size in git (~2.2 MB) — acceptable, verified.
- The 0.3.0 API is pinned exactly (`==0.3.0`), protecting the `predict(source, make_labels=True)` contract against further drift.
- Production sleap-roots models may not load under sleap-nn 0.3.0 (format drift). This is surfaced by the acceptance harness rather than assumed away; a failure is a finding for the parity slice, not a defect in this slice.

## Out of scope

CLI entrypoint, warm GPU worker, parity gate, and any broad refactor of `video_utils`/`plates_timelapse_experiment` beyond what the prediction path requires.
