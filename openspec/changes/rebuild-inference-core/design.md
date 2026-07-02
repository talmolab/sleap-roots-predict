## Context

`sleap-roots-predict` is the prediction service for the sleap-roots pipeline (roadmap tier A3-predict). The inference core was written against sleap-nn 0.0.1 and is broken against the current release (0.3.0). Full brainstorming and audit is in `docs/superpowers/specs/2026-07-02-rebuild-inference-core-design.md`; this file records the load-bearing technical decisions.

## Goals / Non-Goals

- Goals: correct inference on sleap-nn 0.3.0; real (no-mock) test coverage; latest-compatible pinned deps; working CUDA extras; local GPU + real-data acceptance tests.
- Non-Goals (deferred slices): CLI `main.py` container entrypoint; warm long-running GPU worker; prediction-parity gate vs the current pipeline; broad `video_utils`/`plates_timelapse_experiment` refactor beyond the prediction path.

## Decisions

- **Persistent `Predictor`, not per-call `run_inference`.** `make_predictor` returns a reusable `Predictor` (loaded once), and `predict_on_video` calls `predictor.predict(video, make_labels=True)`. Alternatives: module-level `sleap_nn.inference.predict` per call (rebuilds each time — fights the warm-worker goal) or the legacy `run_inference` (demoted in 0.3.0). Chosen path seats the later warm-worker slice.
- **0.3.0 API shape.** `Predictor.from_model_paths` is keyword-only after `model_paths` and defaults `device="cpu"`; `make_pipeline`/`VideoReader` are gone; the `sio.Video` is passed straight into `predict()`.
- **Lean surface + deferred timelapse.** Remove `predict_on_h5`/`batch_predict` (they depend on the removed `VideoReader` import). Rather than rewire the timelapse orchestrator now, **drop its prediction branch** (keep video/H5/metadata building; make the prediction path an inert logged no-op) and defer real timelapse-prediction rework to a future PR. The cylinder-data path is exercised directly (`predict_on_video`) and via the acceptance test, not through the orchestrator. Every retained function has a real test.
- **sleap-io unpinned.** Declare `sleap-io` as a direct dependency but do not pin it — sleap-nn 0.3.0 already constrains it to `>=0.8.0,<0.9.0`, so a local pin would only duplicate that bound and add maintenance on future sleap-nn bumps.
- **Vendored fixtures mirror production.** Production root models are legacy SLEAP UNet **bottom-up** (`training_config.json` + `best_model.h5`, per `model_paths.csv`). Vendor both a native bottom-up minimal (`best.ckpt`) and a legacy UNet bottom-up minimal to hermetically cover both loader paths — the legacy one directly de-risks the production format in CI. (This refines the original design's "top-down pair" fixture choice, made before the production model format was known.)
- **uv index plumbing is required downstream.** uv `[tool.uv.sources]`/index routing is not inherited from sleap-nn, so this repo must declare `[[tool.uv.index]]` + `[tool.uv.sources]` for CUDA extras to resolve CUDA wheels. Without it, Windows resolves CPU-only torch.

## Risks / Trade-offs

- **Legacy model load under 0.3.0 (primary risk).** sleap-nn 0.3.0's legacy loader is UNet-only (`training_config.json` + `best_model.h5`). Vendored legacy minimal proves it in CI; the acceptance test confirms on real models. → If load fails, that is a recorded finding for the parity/conversion slice, not a silent pass.
- **sleap-io 0.5.1 → 0.8.x bump** may ripple into `video_utils`. → Scope stays on the prediction path; real tests catch breaks; unrelated `video_utils` breakage is noted, not fixed here.
- **Vendored asset size** (~4 MB total). → Acceptable; models are minimal test assets.

## Migration Plan

`predict_on_h5`/`batch_predict` removal is BREAKING for any direct importer. `process_timelapse_experiment` keeps its signature but its prediction path becomes an inert logged no-op (prediction deferred to a future PR); callers needing prediction use `predict_on_video` directly. Dep bumps require a re-lock and a fresh `uv sync`.

## Open Questions

- Do the production legacy SLEAP UNet bottom-up models load under sleap-nn 0.3.0? Resolved empirically by tasks 3.3 (hermetic) and 8.2 (real data), not assumed.
- Exact `linux_cuda` CUDA target (cu128 vs cu130) — cu128 chosen to match `windows_cuda`; revisit if the self-hosted GPU runner's driver requires otherwise.
