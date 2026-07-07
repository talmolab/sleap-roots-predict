# Predict container CLI + real Dockerfile ENTRYPOINT

**Date:** 2026-07-06
**Issue:** [talmolab/sleap-roots-predict#24](https://github.com/talmolab/sleap-roots-predict/issues/24)
**Branch:** `add-predict-container-cli`
**Follow-up filed:** [#25](https://github.com/talmolab/sleap-roots-predict/issues/25) (model-derived channel handling)

## Context & goal

`WarmModelWorker` (#9), the prediction-output manifest + `predict_and_write_batch` (#16),
and `resolve_params` (#18) all exist as **library** code, but there is **no container
entrypoint**: `[project.scripts]` is empty and the root `Dockerfile` is a REPL stub
(`ENTRYPOINT ["python"]` / `CMD ["-c", "import sleap_roots_predict; ..."]`, with the
comment "The warm-GPU worker entrypoint is defined later in roadmap tier A3-predict").

A4's per-scan Argo pipeline (talmolab/sleap-roots-pipeline#10) needs a **runnable predict
container**. This slice wires a warm-batch CLI over the existing library and replaces the
REPL stub with a real exec-form ENTRYPOINT, so A4's `predict` stage can `docker run` the
published image. It is the **predict analog of the traits image slice**
(talmolab/sleap-roots `add-trait-extractor-image`, sleap-roots#256).

Every prior predict slice explicitly deferred "the serving protocol/entrypoint" to "the
A4 serving/entrypoint slice" — **this is that slice.** The prediction OUTPUT contract and
the `ScanMetadata` sidecar ownership are already decided (see below); the container INPUT
layout and CLI are decided here.

## Container contract (fixed by A4 — do not change without updating A4)

The interface A4's Argo `predictor` template calls:

- **Invocation:** `docker run <image> <input_scan_dir> <output_dir>` (positional). In Argo:
  `args: ["/workspace/images_input", "/workspace/output"]`.
- **Env:** `WANDB_API_KEY` (from a k8s secret) — the worker fetches models from the live
  registry. Fail-loud if missing.
- **GPU:** `resources.limits.nvidia.com/gpu: 1`.
- **Output:** per scan, the #16 prediction-output artifacts written under
  `<output_dir>/{scan_key}/`.
- **Image tag:** A4 pins the immutable `ghcr.io/talmolab/sleap-roots-predict:sha-<sha>`
  tag (never `latest`).

Source: sleap-roots-pipeline `docs/superpowers/plans/2026-07-06-a4-argo-workflow-poc.md`
("Container interface contracts", Task 2) and
`docs/superpowers/specs/2026-07-06-a4-request-driven-pipeline-design.md` (§3 predict-all,
§7 dedup, §8 resume).

## What was already decided (we conform, not re-litigate)

- **Output layout = nested per-scan** `out_dir/{scan_key}/` holding
  `{scan_key}.predictions.json` + `{scan_key}.model{slug}.root{type}.slp`. This is exactly
  what `predict_and_write_batch` (#16) already writes, and what the trait-extractor
  consumes ("mirroring predict's per-scan `out_dir/{scan_key}/` batch layout"; the traits
  fixture `tests/data/rice_3do_pipeline_output/` is nested).
- **Params reach the container via the `{scan_key}.scan_metadata.json` sidecar**, whose
  `params` object uses normalized `{species, mode, age}` keys (not Bloom columns). The
  trait-extractor's `ScanMetadata` model reads exactly these keys.
- **`resolve_params` (Bloom `cyl_scans_extended` row → params) runs UPSTREAM** in Bloom's
  `workflows` service at submit time (A4 design §3 step 2), **not** in the container. The
  container therefore consumes the already-normalized `params` and builds `ResolvedParams`
  directly (symmetric to the trait-extractor's `to_resolved_params`).
- **Predict does NOT author the sidecar.** Its `image_ids` / `images_checksum` are "the
  idempotency inputs predict defers" — the downloader's responsibility. Predict may *copy*
  the sidecar (copy ≠ author); it never computes those fields.
- **Resume = skip-if-`{scan}.predictions.json`-exists** (existence-based for the PoC;
  checksum-verified skip is the deferred "A4 hardening slice", design §8).
- **`predict_code_sha` provenance** already has an env ladder in `write_prediction_outputs`
  (`SRP_PREDICT_CODE_SHA` → manifest; `SRP_PREDICT_CONTAINER_DIGEST` likewise).

## Decisions (the forks this slice resolves)

### D1 — Sidecar handoff: predict copies it through
The A4 plan authors `{scan_key}.scan_metadata.json` into predict's **input** mount but has
the trait-extractor read it from predict's **output** mount (a different hostPath), with
**no bridge specified** — a real hole in the plan. **Resolution:** after writing a scan's
outputs, predict copies the sidecar **verbatim** into `out_dir/{scan_key}/`, next to the
manifest. This closes the gap, makes predict's output a self-contained trait-extractor
input tree (matching the committed `rice_3do_pipeline_output` fixture), and needs no
cross-mount coordination. Predict never edits the sidecar.

### D2 — Build/CI: evolve the existing image (do not add a second one)
Predict already publishes `ghcr.io/talmolab/sleap-roots-predict` via the root `Dockerfile`
+ `docker-build.yml`, and `type=sha` already emits the `sha-<sha>` tag A4 pins. Unlike the
traits slice (which added a *new* image because sleap-roots already had a different main
image), predict's single image **is** the predict service. **Resolution:** evolve the
existing `Dockerfile` + `docker-build.yml` in place. Switch the image's uv extra
`cpu` → `linux_cuda` so it uses `nvidia.com/gpu`, and add the `SRP_PREDICT_CODE_SHA`
build-arg. Deviation from #24's "no `release:` trigger" wording is **intentional**: we keep
the existing workflow's `release:` trigger and `type=semver` tags (evolving, not mirroring
the traits workflow) — they're harmless to A4, which pins `sha-<sha>`.

### D3 — Channel handling: hardcode `grayscale=True` (this slice), model-derived later (#25)
sleap-nn 0.3.0 does **not** auto-match the video's channel count to the model — conversion
in `Predictor._process_batch` only fires on `ensure_rgb`/`ensure_grayscale` (both default
`False` in every config we have), and there is no `in_channels`-based adaptation in
`sleap_nn/inference/`. A mismatch is a runtime shape crash, not a silent fix.
`predict.py` makes no channel decision — it's fixed when the `sio.Video` is built.

Today's cylinder root models (and the vendored test models) are all single-channel
(`in_channels: 1`), so the video must be 1-channel → **`grayscale=True`**. `grayscale=None`
(autodetect) is fragile on JPEG-decoded frames (exact `R==B` test). **Resolution:**
hardcode `grayscale=True` for this slice and record it as an explicit assumption.
**Follow-up [#25]:** derive `grayscale` from the resolved model's `in_channels` so color /
plate models (`in_channels: 3`) work — plates can be color.

## Architecture

Two small new modules over the existing library primitives; no change to `predict.py`,
`warm_worker.py`, or `output_contract.py`.

### `sleap_roots_predict/batch.py` — the container-oriented runner
- **`discover_scans(input_dir) -> list[ScanInput]`** — `rglob "*.scan_metadata.json"`; for
  each sidecar: `scan_key` = filename stem, validated to equal the sidecar's internal
  `scan_key`; the sidecar's **parent dir = that scan's frames**; frames = natural-sorted
  glob of common image extensions (`png/tif/tiff/jpg/jpeg`, case-insensitive); `params` =
  `ResolvedParams(values=sidecar["params"])`. Duplicate `scan_key` across the tree → hard
  error (mirrors the trait-extractor and `predict_and_write_batch`). Predict reads only
  `scan_key` + `params` with a minimal `json.load` — **no import of `trait_extractor`**.
- **`run_batch(input_dir, output_dir, *, source=None, peak_threshold=0.2, batch_size=4,
  predict_code_sha=None, predict_container_digest=None) -> BatchResult`** —
  1. Construct **one** `WarmModelWorker(source=source)` (`source=None` ⇒ the production
     `WandbRegistrySource`; tests inject a `LocalCardSource`). Models load once and are
     cached by `(registry_id, version)` across the whole batch.
  2. For each discovered scan (sorted):
     - `out_scan_dir = output_dir/{scan_key}`.
     - **Resume:** if `out_scan_dir/{scan_key}.predictions.json` exists → record `skipped`,
       continue.
     - **Isolate:** wrap the rest in try/except; on error record `failed(scan_key, err)` and
       continue the batch.
     - Build the video (`grayscale=True`), `refs = worker.resolve(params)`,
       `labels_by_root = worker.predict(params, video)`,
       `write_prediction_outputs(labels_by_root, refs, out_scan_dir, scan_key=...,
       inference_config=worker.inference_config(), output_params=worker.output_params(),
       predict_code_sha=..., predict_container_digest=...)`.
     - **Copy the sidecar** verbatim → `out_scan_dir/{scan_key}.scan_metadata.json`.
     - Record `ok`.
  3. Return `BatchResult` (per-scan statuses; `ok` iff no `failed` scans).
- **Empty input** (no sidecars discovered) → warn, `ok=True`.

### `sleap_roots_predict/__main__.py` — the CLI
`argparse` with two positional args (`input_dir`, `output_dir`); `main(argv=None) -> int`
calls `run_batch` and returns `0 if result.ok else 1`;
`if __name__ == "__main__": sys.exit(main())`. `predict_code_sha` /
`predict_container_digest` are read via the existing env ladder (not CLI flags — the
container contract is strictly positional).

### Packaging
`[project.scripts]`: `sleap-roots-predict = "sleap_roots_predict.__main__:main"` (so both
the console script and `python -m sleap_roots_predict` work). Export `run_batch` from
`__init__`. Adding a console script does not change dependencies, so `uv.lock` should not
need re-locking (`uv sync --frozen` must still pass — verified during implementation).

### Dockerfile (evolve root `Dockerfile`)
- Replace the REPL stub with exec-form `ENTRYPOINT ["python", "-m", "sleap_roots_predict"]`.
- `RUN uv sync --frozen --no-dev --extra linux_cuda --python 3.12` (was `--extra cpu`, with
  dev). The torch-cuda128 wheels bundle the CUDA runtime, so `bookworm-slim` + the nvidia
  container runtime is enough — **no CUDA base image needed**. Keep the existing apt libs
  (`libgl1`, `libglib2.0-0`, `tk`, `build-essential`) and `MPLBACKEND=Agg`.
- `ARG SRP_PREDICT_CODE_SHA=""` → `ENV SRP_PREDICT_CODE_SHA=${SRP_PREDICT_CODE_SHA}`, placed
  **after** the heavy `uv sync`/COPY layers so a per-commit SHA doesn't bust the dep cache.
- **Tradeoff:** `linux_cuda` pulls multi-GB CUDA torch → a much larger image + slower CI
  build than today's `cpu` image. Accepted — GPU is the point.

### CI (evolve existing `docker-build.yml`)
Add `build-args: SRP_PREDICT_CODE_SHA=${{ github.sha }}` to the build step. Existing
triggers and tags stay (`type=sha` already emits `sha-<sha>`; `latest` on main).

## Data flow

```
A4 stages, per scan, into images_input:  {scan_key}/<frames> + {scan_key}.scan_metadata.json
                                          (sidecar authored upstream; params already {species,mode,age})
        │
        ▼  docker run <image> /workspace/images_input /workspace/output   (WANDB_API_KEY, GPU)
run_batch:
  WarmModelWorker(source=None)                      ← load models once (cached across scans)
  for each discovered scan:
    skip if out/{scan_key}/{scan_key}.predictions.json exists      (resume)
    ResolvedParams(sidecar.params) → resolve → predict (grayscale=True)
    write_prediction_outputs → out/{scan_key}/{scan_key}.predictions.json + .model*.root*.slp
    copy sidecar               → out/{scan_key}/{scan_key}.scan_metadata.json    (D1 pass-through)
        │
        ▼
trait-extractor reads out/{scan_key}/ (manifest + sidecar + .slp) → {scan_key}.result.json
```

## Testing (real TDD, no mocks — mirrors #256's gate)

- **CI gate (offline):** drive `run_batch(..., source=LocalCardSource(vendored models))`
  over a **committed fixture input scan dir** (`tests/assets/scans/{scan_key}/` = the
  existing `centered_pair` PNGs + a hand-authored `{scan_key}.scan_metadata.json`, params
  `rice/cylinder/3`). Assert: nested outputs + copied sidecar exist; manifest validates;
  `predict_code_sha` picked up from `SRP_PREDICT_CODE_SHA`; a second run **skips** (resume);
  one deliberately-broken scan → `result.ok is False` but the good scan still writes
  (isolation). Real sleap-nn inference against the vendored 1-channel models — no mocks, no
  network.
- **CLI wiring:** unlike the traits container (which fetches no models), predict's default
  `main()` uses the live `WandbRegistrySource`, so a `subprocess.run([sys.executable, "-m",
  "sleap_roots_predict", in, out])` end-to-end test needs `WANDB_API_KEY` + network and is
  therefore marked **`@pytest.mark.wandb`** (deselected in CI, run in `/pre-merge`). The
  offline, no-mock **CI** gate is the in-process `run_batch(..., source=LocalCardSource)`
  test above, which exercises the same runner; a cheap CI-safe test also asserts
  `python -m sleap_roots_predict` is importable and its `--help`/arg parsing works without
  touching the network.
- **Packaging guard:** a `tomllib` test asserting `[project.scripts]` declares
  `sleap-roots-predict` (mirrors the traits `test_packaging_config_declares_the_extractor_extra`).
- **Manual pre-merge gate** (heavy; not in CI, like #256): `docker build`
  (`--build-arg SRP_PREDICT_CODE_SHA=deadbeef`) + `docker run <image> /in /out` over the
  fixture → manifests emitted; `docker inspect` shows exec-form ENTRYPOINT + baked
  `SRP_PREDICT_CODE_SHA`; the GPU path via the repo's `/pre-merge` GPU step.

## OpenSpec scope

**One new capability** (mirrors the traits slice's single `trait-extractor-image` spec),
`predict-container`, with `## ADDED Requirements` covering: batch CLI execution over an
input scan dir; load-models-once; skip-if-exists resume; per-scan failure isolation +
non-zero exit; sidecar pass-through; `predict_code_sha` stamping; the `linux_cuda` +
exec-form image; and the build-arg-gated workflow. **No delta** to `prediction-output`
(the manifest/`.slp` format is unchanged). Change-id: `add-predict-container-cli`.

## Risks & assumptions

- **`grayscale=True` is a cylinder-only assumption** — plates can be color. Tracked in
  [#25]; this slice records the assumption in the spec.
- **Frame extension glob** (`png/tif/tiff/jpg/jpeg`) — if the images-downloader stages a
  specific format we can narrow it; broad is safer for the PoC.
- **`linux_cuda` image size / CI build time** grows substantially vs. the `cpu` image
  (accepted, D2).
- **Multi-scan input layout** is undecided in the A4 docs (only single-scan/flat is
  described). The `rglob "*.scan_metadata.json"` discovery works for both flat-single and
  nested-per-scan, so predict does not force A4's hand here.
- **Residual model-config uncertainty:** no production cylinder model config is checked in;
  the 1-channel evidence is from the vendored test models + the SLEAP root-tracing
  convention. The manual GPU gate against the real registry models de-risks this before
  merge.

## Out of scope

- Checksum-verified resume, atomic temp→rename writes, per-scan attempt caps (A4 §8
  hardening slice).
- Model-derived channel handling ([#25]).
- boto3/S3 (A4 stages inputs and collects outputs on the shared mount; predict is
  filesystem-only).
- Any change to the #16 manifest/`.slp` format or the `resolve_params`/`WarmModelWorker`
  library APIs.
- The `{scan_key}.scan_metadata.json` schema and its automated producer (owned by
  A3-traits + the A4 downloader).
