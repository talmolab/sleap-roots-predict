# Tasks: rebuild-inference-core

Each implementation task is TDD-first: write the failing test, then the
minimum code to pass, then refactor. Run `/lint` and `/test` after each.

## 1. Dependencies, fixtures, and environment

- [ ] 1.1 Pin `sleap-nn==0.3.0` in `pyproject.toml`; keep `sleap-io` as an unpinned direct dep (governed by sleap-nn); fix `linux_cuda` extra (`torch-cpu` → `torch-cuda128`).
- [ ] 1.2 Add `[[tool.uv.index]]` (`pytorch-cpu`, `pytorch-cu128`) and `[tool.uv.sources]` torch routing mirroring sleap-nn; re-lock with `uv lock`.
- [ ] 1.3 Register `gpu` and `acceptance` pytest markers in `pyproject.toml`; ensure default `pytest` deselects them.
- [ ] 1.4 `uv sync --extra dev --extra cpu` and confirm `import sleap_nn` reports `0.3.0` and `import sleap_io` reports `0.8.x` (evidence: version strings).
- [ ] 1.5 Vendor minimal bottom-up fixtures into `tests/assets/`: native (`best.ckpt` + `training_config.yaml`) and legacy SLEAP UNet (`training_config.json` + `best_model.h5`), plus `centered_pair_small.mp4`. Record provenance (sleap-nn v0.3.0 test assets) in a short `tests/assets/README.md`.

## 2. Rebuild `make_predictor` (real tests, no mocks)

- [ ] 2.1 **Test first:** `test_make_predictor_builds_real_predictor` — real `make_predictor(native_model_dir)` returns a live sleap-nn `Predictor` (no mocks). Verifies *Predictor Construction*.
- [ ] 2.2 **Test first:** `test_make_predictor_auto_device` — `device="auto"` returns a valid device string on the host (cuda/mps/cpu), not mocked.
- [ ] 2.3 **Test first:** `test_make_predictor_missing_dir_raises` — nonexistent model dir raises `FileNotFoundError`.
- [ ] 2.4 Implement `make_predictor` on `Predictor.from_model_paths(model_dirs, device=resolved, batch_size=, peak_threshold=)` with our own `"auto"` resolution. Make 2.1–2.3 pass.

## 3. Rebuild `predict_on_video` (real tests, no mocks)

- [ ] 3.1 **Test first:** `test_predict_on_video_returns_labels` — real inference on the vendored native model + `centered_pair_small.mp4` returns `sio.Labels` with predicted instances. Verifies *Video Prediction*.
- [ ] 3.2 **Test first:** `test_predict_on_video_saves_slp` — with `save_path`, writes a `.slp` reloadable via `sio.load_file` with frames; returns `Path`.
- [ ] 3.3 **Test first:** `test_predict_on_video_legacy_model` — the vendored legacy SLEAP UNet model loads under 0.3.0 and produces real `sio.Labels`. Verifies *Real Non-Mocked Test Coverage* (legacy path) and de-risks production model format.
- [ ] 3.4 **Test first:** `test_predictor_reused_across_videos` — one predictor used for two videos returns valid labels both times. Verifies persistent reuse.
- [ ] 3.5 Implement `predict_on_video` via `predictor.predict(video, make_labels=True)` + optional `sio.save_file`. Make 3.1–3.4 pass.

## 4. Remove legacy surface; defer timelapse prediction

- [ ] 4.1 Delete `predict_on_h5` and `batch_predict` and the mocked `tests/test_predict.py`.
- [ ] 4.2 **Test first:** add a test proving `plates_timelapse_experiment` still imports and `process_timelapse_experiment` builds videos/H5/metadata with NO prediction (e.g. supplying `model_paths` logs a "prediction deferred" notice and yields `predictions_path=None`), so the package imports cleanly under 0.3.0.
- [ ] 4.3 Remove the predict-function imports and prediction branch from `plates_timelapse_experiment.py` (keep the signature; make the prediction path an inert, logged no-op deferred to a future PR); update `__init__.py` exports (drop `predict_on_h5`/`batch_predict`). Make 4.2 pass.
- [ ] 4.4 Verify `make_video_from_images` still works under sleap-io 0.8 (`sio.Video.from_filename(list, grayscale=)`); adjust only if prediction requires it, and add a real test asserting a Video is built from a small image dir.

## 5. GPU tests

- [ ] 5.1 **Test first:** `tests/test_gpu.py` (`@pytest.mark.gpu`) — assert an accelerator is present (skip otherwise); `make_predictor(model_dirs, device="cuda")` yields a model on a CUDA device and `predict_on_video` returns real `sio.Labels`. Verifies *GPU Inference Execution*.
- [ ] 5.2 Confirm `uv run pytest -m gpu` runs them locally (Windows+CUDA: `uv sync --extra dev --extra windows_cuda` first) and they are deselected on default runs. Document the command in `project.md`/README dev commands.

## 6. Acceptance test on real cylinder data (guarded, CI-skipped)

- [ ] 6.1 **Test first:** `tests/test_acceptance.py` (`@pytest.mark.acceptance`) — skips unless `SRP_CYLINDER_DIR` and `SRP_MODEL_DIRS` are set; when set, builds a video (configurable pattern, default `*.jpg`), runs `make_predictor` + `predict_on_video`, asserts real `sio.Labels`, writes a `.slp`; a model-load failure fails clearly naming the model dir. Verifies *Local Acceptance Validation*.
- [ ] 6.2 Document how to run it against the reference pipeline dir (extract `models_downloader_output/*.zip` to dirs, point `SRP_MODEL_DIRS` at them; `SRP_CYLINDER_DIR` at an `images/.../<plate>/` dir).

## 7. Hardware-install verification

- [ ] 7.1 Locally verify `uv sync --extra dev --extra windows_cuda` resolves a CUDA torch build (evidence: `torch.version.cuda` is not None / `torch.cuda.is_available()` on the GPU host). Verifies *Hardware-Appropriate Backend Installation*.
- [ ] 7.2 Confirm `ci.yml` test job installs `--extra cpu` for non-GPU runners (already present) and that GPU/mac runners run the full suite.

## 8. Gate and validate

- [ ] 8.1 `openspec validate rebuild-inference-core --strict` passes.
- [ ] 8.2 Run the real acceptance test on the reference data (`Z:\...PGM1-PAC-EFFECT_EXP1`) and record the outcome — especially whether the legacy production models load under 0.3.0 (a finding for the parity slice if not).
- [ ] 8.3 `/pre-merge` (format + lint + test + build) green; open PR referencing this change-id.
