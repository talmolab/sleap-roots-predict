## Why

`WarmModelWorker` (#9), the prediction-output manifest + `predict_and_write_batch` (#16),
and `resolve_params` (#18) exist as **library** code, but there is **no container
entrypoint**: `[project.scripts]` is empty and the root `Dockerfile` is a REPL stub. A4's
per-scan Argo pipeline (talmolab/sleap-roots-pipeline#10) needs a runnable predict
container. This slice wires the warm-batch CLI and a real exec-form Dockerfile ENTRYPOINT
so A4's `predict` stage can `docker run` the published image. Closes #24; the predict analog
of the traits image slice (talmolab/sleap-roots#256).

Grounded in `docs/superpowers/specs/2026-07-06-predict-container-cli-design.md`.

## What Changes

- **New `predict-container` capability** — a batch CLI (`python -m sleap_roots_predict
  <input_scan_dir> <output_dir>` + a `sleap-roots-predict` console script) over the existing
  library that:
  - discovers scans in the input dir (`rglob "*.scan_metadata.json"`), reading each scan's
    normalized `{species, mode, age}` params from the sidecar (`resolve_params` already ran
    upstream in Bloom's workflows service — not in the container);
  - **loads models once** across the whole batch via one resident `WarmModelWorker`;
  - writes per scan the #16 outputs into nested `output_dir/{scan_key}/`, and **copies the
    `{scan_key}.scan_metadata.json` sidecar through** so the output is a self-contained
    trait-extractor input tree (closes a documented input→output gap in the A4 plan);
  - **skips** a scan whose `{scan_key}.predictions.json` already exists (existence-based
    resume); **isolates** per-scan failures (continue the batch) and exits non-zero iff any
    scan failed;
  - builds the inference video **single-channel** (`greyscale=True`) to match today's
    1-channel cylinder models (sleap-nn 0.3.0 does not adapt channel count).
- **Evolve the existing GHCR image** — replace the root `Dockerfile` REPL stub with an
  exec-form `ENTRYPOINT ["python","-m","sleap_roots_predict"]`; switch the image's uv extra
  `cpu` → `linux_cuda` (GPU-capable, self-contained CUDA runtime); bake `SRP_PREDICT_CODE_SHA`
  (Dockerfile `ARG`→`ENV`) so each manifest's `predict_code_sha` records the build git sha.
- **Evolve `docker-build.yml`** — pass `build-args: SRP_PREDICT_CODE_SHA=${{ github.sha }}`
  (existing triggers/tags unchanged; `type=sha` already emits the `sha-<sha>` tag A4 pins).

## Impact

- **Affected specs:** new capability `predict-container` (ADDED). No change to
  `prediction`, `prediction-output`, `param-resolution`, or `model-management` — the
  manifest/`.slp` format and the worker/param APIs are reused unchanged.
- **Affected code:** new `sleap_roots_predict/batch.py` (`run_batch`, `discover_scans`) and
  `sleap_roots_predict/__main__.py` (CLI); `pyproject.toml` (`[project.scripts]`); root
  `Dockerfile`; `.github/workflows/docker-build.yml`; `sleap_roots_predict/__init__.py`
  (export `run_batch`); a committed fixture input scan dir under `tests/assets/`;
  `openspec/project.md` (container extra `cpu`→`linux_cuda`; "serving protocol/CLI" now
  landed).
- **Deferred (tracked):** model-derived channel handling (#25, plates can be color);
  Argo-readiness exit-code / empty-input / SIGTERM policy + checksum-verified-skip &
  atomic-writes hardening (#26, symmetric with traits talmolab/sleap-roots#259).
- **Downstream:** A4's `predictor` template consumes `ghcr.io/talmolab/sleap-roots-predict:sha-<sha>`.
