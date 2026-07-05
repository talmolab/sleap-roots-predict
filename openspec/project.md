# Project Context

## Purpose
`sleap-roots-predict` is the **prediction service** in the sleap-roots pipeline: a
lightweight CLI and library that runs `sleap-nn` inference on plant-root images and
timelapse experiments, producing pose-estimation artifacts in the format expected by
downstream `sleap-roots` traits/analysis tooling. It is designed to interoperate with the
sleap-roots model registry.

It is a **service** (runs in a container, GPU-capable), not a pure library: the canonical
distribution artifact is a GHCR Docker image. A PyPI wheel is also published for use as an
importable library.

> **Roadmap note:** tier **A3-predict** is landing. The inference core was rebuilt on the
> sleap-nn 0.3.0 API (PR #6), and the warm in-memory model worker + wandb model-management
> layer (the `model-management` capability) followed. Remaining A3/A4 work: the serving
> protocol/CLI, the `predictions.csv` output contract + `.slp` naming, emitting
> `Provenance`/`ResultEnvelope`, and the prediction-parity harness.

## Tech Stack
- **Python** ≥ 3.11 (CI matrix: 3.11, 3.12)
- **uv** for environment + dependency management (`uv.lock` committed); setuptools build backend
- **sleap-nn** + **sleap-io** for pose estimation / inference
- **numpy, pandas, h5py, imageio** for image/data handling
- **pytest** (+ pytest-cov) for tests; **black** for formatting; **ruff** (pydocstyle `D`,
  google convention) + **codespell** for lint
- **Docker / GHCR** for the service image; **PyPI** for the importable wheel

## Project Conventions

### Code Style
- **black**, line length 88 (`uv run black sleap_roots_predict tests`)
- **ruff** lints docstrings only (`select = ["D"]`, google convention); public functions
  require google-style docstrings
- **codespell** must pass (config in `pyproject.toml`)
- No type-checker (mypy) is configured.

### Architecture Patterns
- Package `sleap_roots_predict/`:
  - `predict.py` — sleap-nn prediction interface (`make_predictor`, `predict_on_video`)
  - `model_selection.py` — pure model-selection matcher (`choose_models`)
  - `model_registry.py` — model-card sources (`ModelCardSource`, `LocalCardSource`,
    `WandbRegistrySource`); all wandb/network access confined here (lazy import)
  - `warm_worker.py` — `WarmModelWorker`, keeps sleap-nn `Predictor`s resident across scans
  - `video_utils.py` — image I/O utilities (natural sort, greyscale, load/save, video build)
  - `plates_timelapse_experiment.py` — timelapse experiment orchestration
  - `__init__.py` — exposes the high-level API only
- Platform-specific install extras (`cpu`, `windows_cuda`, `linux_cuda`, `macos`) select
  the right hardware acceleration; the container uses the CPU extra by default.

### Testing Strategy
- pytest, tests in `tests/`; fixtures in `tests/conftest.py`. Inference tests are
  **real (no mocks)** — they run actual sleap-nn CPU inference against vendored
  minimal models in `tests/assets/` (see `tests/assets/README.md`).
- Default run deselects `gpu`, `acceptance`, and `wandb` markers (`addopts` in `pyproject.toml`;
  CI's explicit `-m` filters exclude them too). The `wandb` tests hit the model registry and are
  gated on `WANDB_API_KEY` (they skip cleanly without it).
- **GPU tests** (`@pytest.mark.gpu`): **not run in CI** (no GPU runner). Run locally on a
  CUDA/MPS machine — on Windows+CUDA: `uv sync --extra dev --extra windows_cuda` then
  `uv run pytest -m gpu`. This is a required step in the `/pre-merge` gate; they skip cleanly
  with no accelerator.
- **Acceptance test** (`@pytest.mark.acceptance`): real-data, CI-skipped. Gate with
  `SRP_CYLINDER_DIR` (image frames) and `SRP_MODEL_DIRS` (os-pathsep-joined model
  dirs; extract legacy `.zip` models first), then `uv run pytest -m acceptance -s`.
- Coverage via `pytest --cov=sleap_roots_predict`.

### Git Workflow
- Branch off `main`; open a PR; CI (`ci.yml`: lint + test matrix) must be green to merge.
- Spec-driven changes use OpenSpec (`/openspec:proposal` → `/openspec:apply` →
  `/openspec:archive`).

## Domain Context
Roots are imaged in cylinders/plates (often as timelapse sequences). This service converts
image directories into video/H5 inputs, runs sleap-nn predictions with sleap-roots models,
and emits artifacts (`.slp`/labels + metadata CSV) consumed by `sleap-roots` for trait
extraction. Correct artifact format and metadata provenance are critical for the
downstream join.

## Important Constraints
- Inference is GPU-heavy; image builds and worker shape must account for model-load cost.
- Output artifact format must stay compatible with downstream `sleap-roots` consumers.
- Keep `main` green: do not add CI steps referencing not-yet-built modules.

## External Dependencies
- **sleap-nn / sleap-io** — inference engine and label I/O.
- **sleap-roots-contracts** (`==0.1.0a3`) — shared `ModelCard`/`ModelRef`/`ResolvedParams`/`RootType`.
- **wandb** — the model registry the warm worker fetches root models from (network confined to
  `WandbRegistrySource`). Also a transitive sleap-nn dependency.
- **sleap-roots model registry** — source of trained models (the wandb registry above).
- **sleap-roots-training** — *coordinating writer*: emits the `ModelCard` selection fields as
  wandb artifact metadata at model promotion (must match the contract's field names).
- **GHCR** (`ghcr.io/talmolab/sleap-roots-predict`) — service image registry.
- **PyPI** — importable wheel distribution.
