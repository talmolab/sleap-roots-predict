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

> **Roadmap note:** the rebuild of the inference core on the new `sleap-nn` API and the
> warm long-running GPU worker shape belong to a **later tier (A3-predict)** in the
> bloom-pipeline-integration roadmap. This repo is currently at the **A0 tooling baseline**
> (OpenSpec + dev commands + Dockerfile + GHCR). Do **not** undertake the sleap-nn rewrite
> or warm-worker work as part of A0.

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
  - `predict.py` — sleap-nn prediction interface (`make_predictor`, `predict_on_video`,
    `predict_on_h5`, `batch_predict`)
  - `video_utils.py` — image I/O utilities (natural sort, greyscale, load/save, video build)
  - `plates_timelapse_experiment.py` — timelapse experiment orchestration
  - `__init__.py` — exposes the high-level API only
- Platform-specific install extras (`cpu`, `windows_cuda`, `linux_cuda`, `macos`) select
  the right hardware acceleration; the container uses the CPU extra by default.

### Testing Strategy
- pytest, tests in `tests/`; fixtures in `tests/conftest.py`.
- GPU tests are marked `@pytest.mark.gpu` and deselected with `-m "not gpu"` on non-GPU
  runners; the self-hosted-gpu and macOS (MPS) runners run the full suite.
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
- **sleap-roots model registry** — source of trained models.
- **GHCR** (`ghcr.io/talmolab/sleap-roots-predict`) — service image registry.
- **PyPI** — importable wheel distribution.
