# Predict Container CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire a warm-batch predict CLI (`python -m sleap_roots_predict <in> <out>`) + a real exec-form GPU Dockerfile ENTRYPOINT over the existing library, so A4's Argo predict stage can `docker run` the published image. Closes #24.

**Architecture:** Two small new modules — `batch.py` (`discover_scans` + `run_batch`, orchestrating discovery/resume/isolation/sidecar-copy over a single resident `WarmModelWorker`, delegating inference to `worker.predict` and writing via `write_prediction_outputs`) and `__main__.py` (argparse CLI). The root `Dockerfile` and `docker-build.yml` evolve in place (`cpu`→`linux_cuda`, `SRP_PREDICT_CODE_SHA` build-arg). No changes to `predict.py`/`warm_worker.py`/`output_contract.py`.

**Tech Stack:** Python ≥3.11, uv, pytest (no mocks; real sleap-nn CPU inference against vendored models), sleap-io, sleap-roots-contracts, Docker/GHCR.

**Spec:** `openspec/changes/add-predict-container-cli/` (capability `predict-container`, 8 requirements / 20 scenarios). **Design:** `docs/superpowers/specs/2026-07-06-predict-container-cli-design.md`.

## Global Constraints

- **OpenSpec tasks are the checklist of record** — mark `- [x]` in `openspec/changes/add-predict-container-cli/tasks.md` as each lands.
- Path handling via `pathlib.Path`; emit path strings with `Path.as_posix()` (lab convention).
- Public functions require google-style docstrings (`ruff select=["D"]`); format with `black` (line length 88); `codespell` must pass over the whole tree (sidecar JSON + docs included).
- Offline CI gate: real inference via injected `LocalCardSource` (never the default `WandbRegistrySource`); anything needing the registry is `@pytest.mark.wandb`, GPU is `@pytest.mark.gpu`, docker is a manual `/pre-merge` gate. Default `addopts` deselects `gpu`/`acceptance`/`wandb`.
- Frame channel: build the inference video `greyscale=True` (1-channel; cylinder models are `in_channels: 1`). Model-derived handling is deferred (#25).
- Resume is existence-based; atomic writes + checksum-skip are deferred together (#26).
- The RED test and its GREEN implementation land in the **same commit** (never commit a standalone failing test). Commit messages: conventional prefix + `#24` in the subject + the trailer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---

## File Structure

- **Create** `sleap_roots_predict/batch.py` — `ScanInput`, `ScanResult`, `BatchResult`, `discover_scans`, `run_batch`.
- **Create** `sleap_roots_predict/__main__.py` — `main(argv)` CLI (lazy `run_batch` import).
- **Modify** `pyproject.toml` — `[project.scripts]`.
- **Modify** `sleap_roots_predict/__init__.py` — export `run_batch`.
- **Modify** `Dockerfile` — real exec-form ENTRYPOINT, `linux_cuda`, code-sha ARG/ENV.
- **Modify** `.github/workflows/docker-build.yml` — build-arg + `type=sha,format=long` + build reliability.
- **Create** `tests/test_batch.py` — the batch runner tests (real inference).
- **Create** `tests/test_predict_container_packaging.py` — `tomllib` guard for `[project.scripts]` + workflow guard.
- **Modify** `tests/conftest.py` — promote `_card`/`rice_source`/`all_roots_source`/`video`; add `scan_input_dir`/`SCAN_KEY`.
- **Create** `tests/assets/scans/scanCPTEST0/` — committed fixture (8 PNG frames + sidecar).
- **Modify** `tests/test_public_api.py` — add `run_batch`.
- **Modify** docs: `API.md`, `README.md`, `openspec/project.md`, `CLAUDE.md`, `CHANGELOG.md`.

---

## Task 1: Shared test fixtures (conftest + committed scan dir)

**Files:**
- Modify: `tests/conftest.py`
- Create: `tests/assets/scans/scanCPTEST0/frame_000.png … frame_007.png` (copied)
- Create: `tests/assets/scans/scanCPTEST0/scanCPTEST0.scan_metadata.json`

**Interfaces:**
- Produces: fixtures `all_roots_source` (LocalCardSource, 3 root types), `rice_source`, `video`, `scan_input_dir` (→ `tests/assets/scans`), and module constant `SCAN_KEY = "scanCPTEST0"`.

- [ ] **Step 1: Add shared fixtures to `tests/conftest.py`** (append near the model fixtures). These mirror the currently module-local copies in `tests/test_output_contract.py:32-77`; adding them to `conftest` makes them visible to `tests/test_batch.py`. Existing modules keep their own local copies (pytest lets a module override a conftest fixture), so no other test file changes.

```python
from sleap_roots_contracts import ModelCard, ResolvedParams

from sleap_roots_predict.model_registry import LocalCardSource
from sleap_roots_predict.video_utils import make_video_from_images

SCAN_KEY = "scanCPTEST0"


def _card(root_type, registry_id, *, species="rice", version="v1", age_min=2, age_max=5):
    """Build a ModelCard for the vendored-model LocalCardSources."""
    return ModelCard(
        species=species,
        mode="cylinder",
        age_min=age_min,
        age_max=age_max,
        root_type=root_type,
        registry_id=registry_id,
        version=version,
    )


@pytest.fixture
def rice_source(native_model_dir: Path, legacy_model_dir: Path) -> LocalCardSource:
    """A source: primary=native model, lateral=legacy model, both for rice."""
    return LocalCardSource(
        [
            (_card("primary", "reg/rice-primary"), native_model_dir),
            (_card("lateral", "reg/rice-lateral"), legacy_model_dir),
        ]
    )


@pytest.fixture
def all_roots_source(native_model_dir: Path, legacy_model_dir: Path) -> LocalCardSource:
    """A source covering all three root types (crown reuses the native model)."""
    return LocalCardSource(
        [
            (_card("primary", "reg/rice-primary"), native_model_dir),
            (_card("lateral", "reg/rice-lateral"), legacy_model_dir),
            (_card("crown", "reg/rice-crown"), native_model_dir),
        ]
    )


@pytest.fixture(scope="module")
def video(centered_pair_image_dir: Path):
    """An 8-frame greyscale video built from the vendored frames."""
    files = sorted(centered_pair_image_dir.glob("*.png"))
    return make_video_from_images(files, greyscale=True)


@pytest.fixture
def scan_input_dir() -> Path:
    """Committed input scan dir: scans/scanCPTEST0/<8 frames> + sidecar."""
    return ASSETS_DIR / "scans"
```

- [ ] **Step 2: Create the committed fixture scan dir.** Copy the 8 frames and write the sidecar.

```bash
mkdir -p tests/assets/scans/scanCPTEST0
cp tests/assets/images/centered_pair/*.png tests/assets/scans/scanCPTEST0/
```

Then create `tests/assets/scans/scanCPTEST0/scanCPTEST0.scan_metadata.json` (co-located with the frames; internal `scan_key` == dir name == filename stem):

```json
{
  "scan_key": "scanCPTEST0",
  "image_ids": ["cp_0", "cp_1", "cp_2", "cp_3", "cp_4", "cp_5", "cp_6", "cp_7"],
  "images_checksum": "sha256:cptest0000000000000000000000000000000000000000000000000000000000",
  "params": {"species": "rice", "mode": "cylinder", "age": 3}
}
```

- [ ] **Step 3: Confirm the suite still collects and the frames copied.**

Run: `uv run pytest -q --collect-only tests/conftest.py 2>/dev/null; ls tests/assets/scans/scanCPTEST0`
Expected: 8 `frame_*.png` + `scanCPTEST0.scan_metadata.json` present; no collection error.

- [ ] **Step 4: Commit.**

```bash
git add tests/conftest.py tests/assets/scans
git commit -m "chore: shared LocalCardSource fixtures + committed scan fixture (#24)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Packaging + CLI skeleton

**Files:**
- Modify: `pyproject.toml` (`[project.scripts]`)
- Create: `sleap_roots_predict/__main__.py`
- Create: `tests/test_predict_container_packaging.py`

**Interfaces:**
- Produces: console script `sleap-roots-predict = "sleap_roots_predict.__main__:main"`; `sleap_roots_predict.__main__.main(argv=None) -> int`.
- Consumes (lazily, at call time): `sleap_roots_predict.batch.run_batch` (Task 4).

- [ ] **Step 1: Write the failing packaging guard test.** Create `tests/test_predict_container_packaging.py`:

```python
"""Guards for the predict-container packaging + workflow wiring (no mocks)."""

import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_console_script_declared():
    data = tomllib.loads((REPO / "pyproject.toml").read_text())
    scripts = data["project"]["scripts"]
    assert scripts["sleap-roots-predict"] == "sleap_roots_predict.__main__:main"
```

- [ ] **Step 2: Run it, verify it fails.**

Run: `uv run pytest tests/test_predict_container_packaging.py::test_console_script_declared -v`
Expected: FAIL (`KeyError: 'sleap-roots-predict'` — `[project.scripts]` is empty).

- [ ] **Step 3: Add the console script to `pyproject.toml`.** Under `[project.scripts]` (currently empty):

```toml
[project.scripts]
sleap-roots-predict = "sleap_roots_predict.__main__:main"
```

- [ ] **Step 4: Run the guard + confirm the lock is still frozen-clean.**

Run: `uv run pytest tests/test_predict_container_packaging.py::test_console_script_declared -v && uv sync --frozen --extra cpu`
Expected: test PASS; `uv sync --frozen` succeeds with no lock change (entry points are not resolution inputs).

- [ ] **Step 5: Write the failing CLI `--help` test.** Append to `tests/test_predict_container_packaging.py`:

```python
import subprocess
import sys


def test_module_help_lists_positional_args():
    proc = subprocess.run(
        [sys.executable, "-m", "sleap_roots_predict", "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "input_dir" in proc.stdout
    assert "output_dir" in proc.stdout
```

- [ ] **Step 6: Run it, verify it fails.**

Run: `uv run pytest tests/test_predict_container_packaging.py::test_module_help_lists_positional_args -v`
Expected: FAIL (no `__main__.py` → `No module named sleap_roots_predict.__main__`).

- [ ] **Step 7: Create `sleap_roots_predict/__main__.py`.** The `run_batch` import is **function-local** so `--help` (which raises `SystemExit` in `parse_args`) never triggers it — keeping this commit green before `batch.py` exists.

```python
"""CLI entrypoint: ``python -m sleap_roots_predict <input_dir> <output_dir>``.

Warm-batch predict over a directory of staged scans. Exit code is ``0`` when no
scan failed and ``1`` otherwise, so an Argo step sees a real batch result.
"""

import argparse
import logging
import sys
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse args, run the batch, and return a process exit code.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        ``0`` if no scan failed, ``1`` otherwise.
    """
    parser = argparse.ArgumentParser(
        prog="sleap_roots_predict",
        description="Warm-batch predict over a directory of staged scans.",
    )
    parser.add_argument(
        "input_dir",
        help="Directory of staged scans (each scan: a directory of image frames "
        "with a co-located {scan_key}.scan_metadata.json sidecar).",
    )
    parser.add_argument(
        "output_dir",
        help="Directory to write per-scan prediction outputs into.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # Lazy import: keeps `--help` import-light and lets this module load before
    # batch.py exists during incremental development.
    from sleap_roots_predict.batch import run_batch

    result = run_batch(args.input_dir, args.output_dir)
    n_ok = sum(1 for s in result.scans if s.status == "ok")
    n_skip = sum(1 for s in result.scans if s.status == "skipped")
    n_fail = sum(1 for s in result.scans if s.status == "failed")
    logging.getLogger(__name__).info(
        "Batch complete: %d ok, %d skipped, %d failed", n_ok, n_skip, n_fail
    )
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 8: Run both guards.**

Run: `uv run pytest tests/test_predict_container_packaging.py -v`
Expected: both PASS (`--help` prints the two positional args and exits 0; the `run_batch` import never runs for `--help`).

- [ ] **Step 9: Commit.**

```bash
git add pyproject.toml sleap_roots_predict/__main__.py tests/test_predict_container_packaging.py
git commit -m "feat: sleap-roots-predict console script + __main__ CLI skeleton (#24)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Scan discovery (`discover_scans`)

**Files:**
- Create: `sleap_roots_predict/batch.py` (discovery portion)
- Create: `tests/test_batch.py`

**Interfaces:**
- Produces: `ScanInput` (frozen dataclass: `scan_key: str`, `sidecar_path: Path`, `frames: list[Path]`, `params: Optional[ResolvedParams]`, `error: Optional[str]`); `discover_scans(input_dir) -> list[ScanInput]` (raises `ValueError` on duplicate `scan_key`; invalid scans returned with `.error` set).

- [ ] **Step 1: Write the failing discovery test.** Create `tests/test_batch.py`:

```python
"""Real, no-mock tests for the warm-batch predict runner."""

import json
from pathlib import Path

import pytest

from sleap_roots_predict.batch import BatchResult, discover_scans, run_batch


def test_discover_scans_reads_sidecar_and_frames(scan_input_dir: Path):
    scans = discover_scans(scan_input_dir)
    assert len(scans) == 1
    scan = scans[0]
    assert scan.scan_key == "scanCPTEST0"
    assert scan.error is None
    assert len(scan.frames) == 8
    assert all(p.suffix.lower() == ".png" for p in scan.frames)
    assert scan.params.values == {"species": "rice", "mode": "cylinder", "age": 3}
```

- [ ] **Step 2: Run it, verify it fails.**

Run: `uv run pytest tests/test_batch.py::test_discover_scans_reads_sidecar_and_frames -v`
Expected: FAIL (`ModuleNotFoundError: sleap_roots_predict.batch`).

- [ ] **Step 3: Create `sleap_roots_predict/batch.py` with the discovery half.**

```python
"""Warm-batch predict runner over a directory of staged scans.

Discovers scans (a ``{scan_key}.scan_metadata.json`` sidecar co-located with its
image frames in a dedicated directory), loads models once via a resident
:class:`~sleap_roots_predict.warm_worker.WarmModelWorker`, predicts each scan,
writes the prediction-output artifacts, and copies the sidecar through so each
``<output_dir>/{scan_key}/`` is a self-contained trait-extractor input tree.
"""

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

from sleap_roots_contracts import ResolvedParams

from sleap_roots_predict.model_registry import ModelCardSource
from sleap_roots_predict.output_contract import write_prediction_outputs
from sleap_roots_predict.video_utils import make_video_from_images
from sleap_roots_predict.warm_worker import WarmModelWorker

logger = logging.getLogger(__name__)

_SIDECAR_SUFFIX = ".scan_metadata.json"
_IMAGE_EXTENSIONS = frozenset({".png", ".tif", ".tiff", ".jpg", ".jpeg"})
_REQUIRED_PARAM_KEYS = ("species", "mode", "age")


@dataclass(frozen=True)
class ScanInput:
    """A discovered scan. ``error`` is set (and params/frames may be empty) when
    the sidecar is invalid — an isolated per-scan failure, not a batch abort."""

    scan_key: str
    sidecar_path: Path
    frames: List[Path] = field(default_factory=list)
    params: Optional[ResolvedParams] = None
    error: Optional[str] = None


def discover_scans(input_dir: Union[str, Path]) -> List[ScanInput]:
    """Discover scans under ``input_dir`` by their scan-metadata sidecars.

    Recursively finds ``*.scan_metadata.json`` files; each sidecar's parent
    directory holds that scan's frames. ``scan_key`` is the sidecar's filename
    stem and must equal the sidecar's internal ``scan_key``. Invalid scans are
    returned with ``.error`` set (isolated failure); a duplicate ``scan_key``
    anywhere in the tree raises.

    Args:
        input_dir: Directory of staged scans.

    Returns:
        One :class:`ScanInput` per discovered sidecar, sorted by path.

    Raises:
        ValueError: If two sidecars share a ``scan_key``.
    """
    input_dir = Path(input_dir)
    scans: List[ScanInput] = []
    seen: dict[str, Path] = {}
    for sidecar in sorted(input_dir.rglob("*" + _SIDECAR_SUFFIX)):
        scan_key = sidecar.name[: -len(_SIDECAR_SUFFIX)]
        if scan_key in seen:
            raise ValueError(
                f"duplicate scan_key {scan_key!r}: "
                f"{seen[scan_key].as_posix()} and {sidecar.as_posix()}"
            )
        seen[scan_key] = sidecar
        scans.append(_load_scan(sidecar, scan_key))
    return scans


def _load_scan(sidecar: Path, scan_key: str) -> ScanInput:
    """Parse one sidecar into a ScanInput (with ``.error`` set if invalid)."""
    try:
        meta = json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return ScanInput(scan_key, sidecar, error=f"unreadable sidecar: {exc}")
    if meta.get("scan_key") != scan_key:
        return ScanInput(
            scan_key,
            sidecar,
            error=(
                f"sidecar scan_key {meta.get('scan_key')!r} != filename stem "
                f"{scan_key!r}"
            ),
        )
    params = meta.get("params")
    if not isinstance(params, dict) or any(
        k not in params for k in _REQUIRED_PARAM_KEYS
    ):
        return ScanInput(
            scan_key, sidecar, error=f"sidecar params missing/incomplete: {params!r}"
        )
    frames = sorted(
        (
            p
            for p in sidecar.parent.iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
        ),
        key=lambda p: p.name,
    )
    resolved = ResolvedParams(values={k: params[k] for k in _REQUIRED_PARAM_KEYS})
    return ScanInput(scan_key, sidecar, frames=frames, params=resolved)
```

- [ ] **Step 4: Run the discovery test.**

Run: `uv run pytest tests/test_batch.py::test_discover_scans_reads_sidecar_and_frames -v`
Expected: PASS.

- [ ] **Step 5: Add the discovery edge-case tests.** Append to `tests/test_batch.py`:

```python
def _write_scan(root: Path, scan_key: str, params, *, stem=None, extra_files=()):
    """Create a scan dir with one PNG frame + a sidecar; return the dir."""
    d = root / scan_key
    d.mkdir(parents=True)
    # a tiny valid PNG frame
    import numpy as np
    from PIL import Image

    Image.fromarray(np.zeros((16, 16), dtype="uint8")).save(d / "frame_000.png")
    stem = stem if stem is not None else scan_key
    body = {"scan_key": scan_key, "image_ids": ["a"], "images_checksum": "sha256:x"}
    if params is not None:
        body["params"] = params
    (d / f"{stem}{'.scan_metadata.json'}").write_text(json.dumps(body))
    for name in extra_files:
        (d / name).write_text("not an image")
    return d


def test_non_image_files_are_ignored(tmp_path: Path):
    _write_scan(
        tmp_path, "scanA", {"species": "rice", "mode": "cylinder", "age": 3},
        extra_files=("readme.txt",),
    )
    (scan,) = discover_scans(tmp_path)
    assert [p.name for p in scan.frames] == ["frame_000.png"]  # .txt + .json excluded


def test_stem_scan_key_mismatch_is_error(tmp_path: Path):
    # sidecar filename stem "scanB" but internal scan_key "scanOTHER"
    _write_scan(tmp_path, "scanB", {"species": "rice", "mode": "cylinder", "age": 3})
    (tmp_path / "scanB" / "scanB.scan_metadata.json").write_text(
        json.dumps({"scan_key": "scanOTHER", "params": {"species": "rice", "mode": "cylinder", "age": 3}})
    )
    (scan,) = discover_scans(tmp_path)
    assert scan.error is not None and "scanOTHER" in scan.error


def test_missing_params_is_error(tmp_path: Path):
    _write_scan(tmp_path, "scanC", None)  # no params key
    (scan,) = discover_scans(tmp_path)
    assert scan.error is not None and "params" in scan.error


def test_duplicate_scan_key_raises(tmp_path: Path):
    _write_scan(tmp_path / "a", "dup", {"species": "rice", "mode": "cylinder", "age": 3})
    _write_scan(tmp_path / "b", "dup", {"species": "rice", "mode": "cylinder", "age": 3})
    with pytest.raises(ValueError, match="duplicate scan_key"):
        discover_scans(tmp_path)


def test_batch_does_not_import_trait_extractor():
    import sleap_roots_predict.batch as batch_mod
    import sys

    # Importing batch must not pull in the trait-extractor package.
    assert "trait_extractor" not in sys.modules
    assert "trait_extractor" not in getattr(batch_mod, "__dict__", {})
```

- [ ] **Step 6: Run the edge-case tests.**

Run: `uv run pytest tests/test_batch.py -v -k "non_image or mismatch or missing_params or duplicate or does_not_import"`
Expected: all PASS.

- [ ] **Step 7: Commit.**

```bash
git add sleap_roots_predict/batch.py tests/test_batch.py
git commit -m "feat: scan discovery + params from sidecar (batch.discover_scans) (#24)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: `run_batch` — per-scan outputs + sidecar pass-through

**Files:**
- Modify: `sleap_roots_predict/batch.py` (add `ScanResult`, `BatchResult`, `run_batch`, `_predict_one`)
- Modify: `sleap_roots_predict/__init__.py` (export `run_batch`)
- Modify: `tests/test_public_api.py`
- Modify: `tests/test_batch.py`

**Interfaces:**
- Produces: `ScanResult(scan_key, status, error=None)` with `status ∈ {"ok","skipped","failed"}`; `BatchResult(scans: list[ScanResult])` with `.ok` property (True iff no `failed`); `run_batch(input_dir, output_dir, *, source=None, peak_threshold=0.2, batch_size=4, predict_code_sha=None, predict_container_digest=None) -> BatchResult`.
- Consumes: `WarmModelWorker.resolve/predict/inference_config/output_params`, `write_prediction_outputs`, `make_video_from_images`.

- [ ] **Step 1: Write the failing happy-path test (real inference).** Append to `tests/test_batch.py`:

```python
def test_run_batch_writes_outputs_and_copies_sidecar(
    scan_input_dir: Path, all_roots_source, tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("SRP_PREDICT_CODE_SHA", "cafef00d")
    out = tmp_path / "out"
    result = run_batch(scan_input_dir, out, source=all_roots_source)

    assert result.ok
    assert [s.status for s in result.scans] == ["ok"]

    scan_dir = out / "scanCPTEST0"
    manifest = scan_dir / "scanCPTEST0.predictions.json"
    assert manifest.exists()
    slps = list(scan_dir.glob("scanCPTEST0.model*.root*.slp"))
    assert len(slps) == 3  # primary, lateral, crown

    # sidecar copied through, byte-identical
    src = scan_input_dir / "scanCPTEST0" / "scanCPTEST0.scan_metadata.json"
    dst = scan_dir / "scanCPTEST0.scan_metadata.json"
    assert dst.read_bytes() == src.read_bytes()

    # provenance sha picked up from the env
    data = json.loads(manifest.read_text())
    assert data["predict_code_sha"] == "cafef00d"
```

- [ ] **Step 2: Run it, verify it fails.**

Run: `uv run pytest tests/test_batch.py::test_run_batch_writes_outputs_and_copies_sidecar -v`
Expected: FAIL (`ImportError: cannot import name 'run_batch'` / `run_batch` not defined).

- [ ] **Step 3: Add `ScanResult`, `BatchResult`, `run_batch`, `_predict_one` to `batch.py`.** Append after `_load_scan`:

```python
@dataclass
class ScanResult:
    """Per-scan outcome. ``status`` is one of ``ok`` / ``skipped`` / ``failed``."""

    scan_key: str
    status: str
    error: Optional[str] = None


@dataclass
class BatchResult:
    """Aggregate batch outcome."""

    scans: List[ScanResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True iff no scan failed (skipped/ok scans are fine)."""
        return all(s.status != "failed" for s in self.scans)


def run_batch(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    source: Optional[ModelCardSource] = None,
    peak_threshold: float = 0.2,
    batch_size: int = 4,
    predict_code_sha: Optional[str] = None,
    predict_container_digest: Optional[str] = None,
) -> BatchResult:
    """Predict every scan under ``input_dir``, writing outputs under ``output_dir``.

    Loads models once via a single resident worker. Per scan: skip if its manifest
    already exists (resume); otherwise resolve + predict, write the #16 artifacts
    into ``output_dir/{scan_key}/``, and copy the sidecar through. A per-scan error
    is isolated (recorded ``failed``, batch continues). An input with no scans is a
    no-op.

    Args:
        input_dir: Directory of staged scans.
        output_dir: Directory to write per-scan outputs into.
        source: Model-card source; ``None`` uses the production WandbRegistrySource.
        peak_threshold: Peak-detection threshold for inference.
        batch_size: Inference batch size.
        predict_code_sha: Provenance sha (falls back to ``SRP_PREDICT_CODE_SHA``).
        predict_container_digest: Provenance digest (env fallback).

    Returns:
        A :class:`BatchResult` with one :class:`ScanResult` per scan.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    scans = discover_scans(input_dir)
    result = BatchResult()
    if not scans:
        logger.warning("No scans discovered under %s", input_dir.as_posix())
        return result

    worker = WarmModelWorker(
        source=source, peak_threshold=peak_threshold, batch_size=batch_size
    )
    for scan in scans:
        out_scan_dir = output_dir / scan.scan_key
        manifest_path = out_scan_dir / f"{scan.scan_key}.predictions.json"
        if manifest_path.exists():
            logger.info("Skipping %s (manifest exists)", scan.scan_key)
            result.scans.append(ScanResult(scan.scan_key, "skipped"))
            continue
        if scan.error is not None:
            logger.error("Scan %s failed: %s", scan.scan_key, scan.error)
            result.scans.append(ScanResult(scan.scan_key, "failed", scan.error))
            continue
        try:
            _predict_one(
                worker, scan, out_scan_dir, predict_code_sha, predict_container_digest
            )
            result.scans.append(ScanResult(scan.scan_key, "ok"))
        except Exception as exc:  # noqa: BLE001 - isolate per-scan failures
            logger.exception("Scan %s failed", scan.scan_key)
            result.scans.append(ScanResult(scan.scan_key, "failed", str(exc)))
    return result


def _predict_one(
    worker: WarmModelWorker,
    scan: ScanInput,
    out_scan_dir: Path,
    predict_code_sha: Optional[str],
    predict_container_digest: Optional[str],
) -> None:
    """Predict one scan and write its outputs + copied sidecar. Raises on failure."""
    if not scan.frames:
        raise ValueError(
            f"no image frames co-located with sidecar {scan.sidecar_path.as_posix()}"
        )
    refs = worker.resolve(scan.params)
    if not refs:
        raise ValueError(f"no models resolved for params {scan.params.values!r}")
    video = make_video_from_images(scan.frames, greyscale=True)
    labels = worker.predict(scan.params, video)
    out_scan_dir.mkdir(parents=True, exist_ok=True)
    write_prediction_outputs(
        labels,
        refs,
        out_scan_dir,
        scan_key=scan.scan_key,
        inference_config=worker.inference_config(),
        output_params=worker.output_params(),
        predict_code_sha=predict_code_sha,
        predict_container_digest=predict_container_digest,
    )
    shutil.copyfile(
        scan.sidecar_path, out_scan_dir / f"{scan.scan_key}{_SIDECAR_SUFFIX}"
    )
```

- [ ] **Step 4: Export `run_batch` from `__init__.py` (same commit).** In `sleap_roots_predict/__init__.py` add the import next to the other exports and append `"run_batch"` to `__all__`:

```python
from sleap_roots_predict.batch import run_batch
```
```python
    "run_batch",
```

- [ ] **Step 5: Update `tests/test_public_api.py`.** Add `run_batch` to whatever expected-export collection it asserts (mirror the existing entries).

- [ ] **Step 6: Run the happy-path + public-API tests.**

Run: `uv run pytest tests/test_batch.py::test_run_batch_writes_outputs_and_copies_sidecar tests/test_public_api.py -v`
Expected: PASS (real inference writes 3 `.slp` + manifest + copied sidecar; `run_batch` importable and in `__all__`).

- [ ] **Step 7: Add the "predicts every scan + single worker" test.** Append to `tests/test_batch.py`:

```python
def test_run_batch_predicts_every_scan(all_roots_source, tmp_path: Path):
    # two scans in separate dirs, both pointing at the vendored frames
    import shutil as _sh

    src_frames = sorted((Path("tests/assets/images/centered_pair")).glob("*.png"))
    inp = tmp_path / "in"
    for key in ("scanX", "scanY"):
        d = inp / key
        d.mkdir(parents=True)
        for f in src_frames:
            _sh.copyfile(f, d / f.name)
        (d / f"{key}.scan_metadata.json").write_text(json.dumps(
            {"scan_key": key, "image_ids": ["a"], "images_checksum": "sha256:x",
             "params": {"species": "rice", "mode": "cylinder", "age": 3}}
        ))
    out = tmp_path / "out"
    result = run_batch(inp, out, source=all_roots_source)
    assert [s.status for s in result.scans] == ["ok", "ok"]
    for key in ("scanX", "scanY"):
        assert (out / key / f"{key}.predictions.json").exists()
```

- [ ] **Step 8: Run it.**

Run: `uv run pytest tests/test_batch.py::test_run_batch_predicts_every_scan -v`
Expected: PASS.

- [ ] **Step 9: Commit.**

```bash
git add sleap_roots_predict/batch.py sleap_roots_predict/__init__.py tests/test_public_api.py tests/test_batch.py
git commit -m "feat: run_batch — per-scan outputs + sidecar pass-through (#24)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Single-channel prediction input (explicit assertion)

**Files:**
- Modify: `tests/test_batch.py`

**Interfaces:** Consumes `discover_scans` + `make_video_from_images` (already built).

- [ ] **Step 1: Write the single-channel test.** Append to `tests/test_batch.py`:

```python
def test_video_is_single_channel(scan_input_dir: Path):
    from sleap_roots_predict.video_utils import make_video_from_images

    (scan,) = discover_scans(scan_input_dir)
    video = make_video_from_images(scan.frames, greyscale=True)
    assert video.shape[-1] == 1  # 1-channel, matching in_channels:1 cylinder models
```

- [ ] **Step 2: Run it.**

Run: `uv run pytest tests/test_batch.py::test_video_is_single_channel -v`
Expected: PASS (`make_video_from_images(..., greyscale=True)` forces 1 channel). If the runner ever regresses to color, this names the failure.

- [ ] **Step 3: Commit.**

```bash
git add tests/test_batch.py
git commit -m "test: assert run_batch builds a single-channel inference video (#24)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Skip-if-exists resume

**Files:**
- Modify: `tests/test_batch.py` (the resume behavior is already implemented in `run_batch` Task 4 — this task's RED test proves it and guards regressions)

**Interfaces:** Consumes `run_batch`.

- [ ] **Step 1: Write the resume test.** Append to `tests/test_batch.py`:

```python
def test_rerun_skips_completed_scan(scan_input_dir: Path, all_roots_source, tmp_path: Path):
    out = tmp_path / "out"
    run_batch(scan_input_dir, out, source=all_roots_source)
    manifest = out / "scanCPTEST0" / "scanCPTEST0.predictions.json"
    mtime = manifest.stat().st_mtime_ns

    result2 = run_batch(scan_input_dir, out, source=all_roots_source)
    assert [s.status for s in result2.scans] == ["skipped"]
    assert manifest.stat().st_mtime_ns == mtime  # not rewritten
```

- [ ] **Step 2: Run it.**

Run: `uv run pytest tests/test_batch.py::test_rerun_skips_completed_scan -v`
Expected: PASS (the skip check in `run_batch` fires on the existing manifest). If `run_batch` lacked the skip, the second run's status would be `ok` and mtime would change — so this is the guarding RED for the resume requirement.

- [ ] **Step 3: Commit.**

```bash
git add tests/test_batch.py
git commit -m "test: run_batch resume skips scans with an existing manifest (#24)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Failure isolation, zero-model, empty-input + CLI exit codes

**Files:**
- Modify: `tests/test_batch.py`
- Modify: `tests/test_predict_container_packaging.py` (or a small CLI test)

**Interfaces:** Consumes `run_batch`, `main`.

- [ ] **Step 1: Write the isolation + zero-model + empty tests.** Append to `tests/test_batch.py`:

```python
def test_one_failing_scan_does_not_abort_batch(
    scan_input_dir: Path, all_roots_source, tmp_path: Path
):
    import shutil as _sh

    inp = tmp_path / "in"
    # a good scan (copy the committed fixture)
    _sh.copytree(scan_input_dir / "scanCPTEST0", inp / "scanGOOD")
    (inp / "scanGOOD" / "scanCPTEST0.scan_metadata.json").rename(
        inp / "scanGOOD" / "scanGOOD.scan_metadata.json"
    )
    (inp / "scanGOOD" / "scanGOOD.scan_metadata.json").write_text(json.dumps(
        {"scan_key": "scanGOOD", "image_ids": ["a"], "images_checksum": "sha256:x",
         "params": {"species": "rice", "mode": "cylinder", "age": 3}}
    ))
    # a bad scan: sidecar present, NO frames -> per-scan failure
    bad = inp / "scanBAD"
    bad.mkdir()
    (bad / "scanBAD.scan_metadata.json").write_text(json.dumps(
        {"scan_key": "scanBAD", "image_ids": ["a"], "images_checksum": "sha256:x",
         "params": {"species": "rice", "mode": "cylinder", "age": 3}}
    ))
    out = tmp_path / "out"
    result = run_batch(inp, out, source=all_roots_source)
    statuses = {s.scan_key: s.status for s in result.scans}
    assert statuses["scanGOOD"] == "ok"
    assert statuses["scanBAD"] == "failed"
    assert result.ok is False
    assert (out / "scanGOOD" / "scanGOOD.predictions.json").exists()


def test_zero_resolved_models_is_failed(scan_input_dir: Path, rice_source, tmp_path: Path):
    # rice_source has no card for species "soybean" -> zero models resolve
    import shutil as _sh, json as _json

    inp = tmp_path / "in"
    _sh.copytree(scan_input_dir / "scanCPTEST0", inp / "scanZ")
    (inp / "scanZ" / "scanCPTEST0.scan_metadata.json").unlink()
    (inp / "scanZ" / "scanZ.scan_metadata.json").write_text(_json.dumps(
        {"scan_key": "scanZ", "image_ids": ["a"], "images_checksum": "sha256:x",
         "params": {"species": "soybean", "mode": "cylinder", "age": 3}}
    ))
    out = tmp_path / "out"
    result = run_batch(inp, out, source=rice_source)
    assert [s.status for s in result.scans] == ["failed"]
    assert not (out / "scanZ" / "scanZ.predictions.json").exists()


def test_empty_input_is_noop(tmp_path: Path):
    result = run_batch(tmp_path / "empty_in", tmp_path / "out")
    # discover_scans over a missing/empty dir yields nothing -> ok, nothing written
    assert isinstance(result, BatchResult)
    assert result.ok and result.scans == []
```

Note: `run_batch` over a **missing** input dir — `Path.rglob` on a non-existent dir raises `FileNotFoundError`. Ensure `test_empty_input_is_noop` creates the empty dir first (adjust: `(tmp_path / "empty_in").mkdir()` before the call), or handle a missing dir as empty in `discover_scans`. Prefer creating it in the test (an Argo mount always exists).

- [ ] **Step 2: Fix the empty-input test to create the dir** (mounts always exist):

```python
def test_empty_input_is_noop(tmp_path: Path):
    empty = tmp_path / "empty_in"
    empty.mkdir()
    result = run_batch(empty, tmp_path / "out")
    assert result.ok and result.scans == []
```

- [ ] **Step 3: Run the isolation/zero-model/empty tests.**

Run: `uv run pytest tests/test_batch.py -v -k "failing_scan or zero_resolved or empty_input"`
Expected: all PASS (isolation implemented in Task 4; zero-model raises inside `_predict_one` → failed; empty → no-op).

- [ ] **Step 4: Write the CLI exit-code test.** Append to `tests/test_batch.py`:

```python
def test_cli_main_exit_codes(scan_input_dir: Path, tmp_path: Path, monkeypatch):
    from sleap_roots_predict.__main__ import main

    # success path uses an injected source via run_batch monkeypatch to stay offline
    import sleap_roots_predict.__main__ as cli

    calls = {}

    class _Res:
        def __init__(self, ok):
            self.ok = ok
            self.scans = []

    def fake_run_batch(inp, out):
        calls["args"] = (inp, out)
        return _Res(calls["ok"])

    monkeypatch.setattr("sleap_roots_predict.batch.run_batch", fake_run_batch)
    calls["ok"] = True
    assert main([str(scan_input_dir), str(tmp_path / "o1")]) == 0
    calls["ok"] = False
    assert main([str(scan_input_dir), str(tmp_path / "o2")]) == 1
```

Note: this test asserts the CLI's **exit-code wiring** only (that `main` returns `0`/`1` from `result.ok`) by patching `run_batch` — it is not an inference test. The real end-to-end CLI-over-the-registry path is the `@pytest.mark.wandb` subprocess test below.

- [ ] **Step 5: Add the `@pytest.mark.wandb` end-to-end subprocess test.** Append to `tests/test_batch.py`:

```python
@pytest.mark.wandb
def test_module_cli_over_registry(scan_input_dir: Path, tmp_path: Path):
    import subprocess, sys

    out = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, "-m", "sleap_roots_predict", str(scan_input_dir), str(out)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert (out / "scanCPTEST0" / "scanCPTEST0.predictions.json").exists()
```

- [ ] **Step 6: Run the CLI tests (wandb deselected by default).**

Run: `uv run pytest tests/test_batch.py -v -k "cli_main_exit"`
Expected: PASS. (`test_module_cli_over_registry` is skipped by the default `-m 'not wandb'`.)

- [ ] **Step 7: Commit.**

```bash
git add tests/test_batch.py
git commit -m "feat: per-scan failure isolation + zero-model/empty handling + CLI exit codes (#24)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 8: Mark OpenSpec runner tasks done.** In `openspec/changes/add-predict-container-cli/tasks.md` check off groups 1-7; run `uv run pytest -m "not gpu and not acceptance and not wandb"` to confirm the whole offline suite is green, then commit the tasks.md update.

---

## Task 8: Dockerfile — real exec-form entrypoint + GPU

**Files:**
- Modify: `Dockerfile` (replace lines 30, 41-44 region; see current stub)

**Interfaces:** Produces the runnable image; consumes `sleap_roots_predict.__main__`.

- [ ] **Step 1: Rewrite the install + entrypoint region of `Dockerfile`.** Change the sync line from `RUN uv sync --frozen --extra cpu` to the GPU extra, and replace the REPL `ENTRYPOINT`/`CMD` stub. Final tail of the file:

```dockerfile
# Install the GPU stack against the committed lock (the lock already resolves
# linux_cuda; the cu128 wheels bundle the CUDA runtime, so bookworm-slim + the
# nvidia container runtime is enough — no CUDA base image needed).
RUN uv sync --frozen --no-dev --extra linux_cuda --python 3.12

ENV PATH="/app/.venv/bin:$PATH"

# Headless matplotlib for any plotting pulled in by the stack.
ENV MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib

# Build git sha baked in AFTER the heavy layers so a per-commit sha does not bust
# the dependency cache. write_prediction_outputs reads SRP_PREDICT_CODE_SHA into
# each manifest's predict_code_sha (the downstream idempotency key).
ARG SRP_PREDICT_CODE_SHA=""
ENV SRP_PREDICT_CODE_SHA=${SRP_PREDICT_CODE_SHA}

# Real exec-form entrypoint (replaces the REPL stub, incl. its CMD) so the batch
# exit code propagates to the caller (Argo). Positional args: <input_dir> <output_dir>.
ENTRYPOINT ["python", "-m", "sleap_roots_predict"]
```

Ensure the old `ENTRYPOINT ["python"]` and `CMD ["-c", "import …"]` lines are removed (no leftover `CMD`).

- [ ] **Step 2: Build locally and inspect (manual gate — needs Docker).**

Run:
```bash
docker build -t srp:test --build-arg SRP_PREDICT_CODE_SHA=deadbeef .
docker inspect --format '{{json .Config.Entrypoint}} {{json .Config.Cmd}}' srp:test
```
Expected: `["python","-m","sleap_roots_predict"]` and `null` (no leftover Cmd).

- [ ] **Step 3: Run the container over the committed fixture (bind-mounted; the fixture is dockerignored).**

Run:
```bash
docker run --rm -e WANDB_API_KEY="$WANDB_API_KEY" \
  -v "$PWD/tests/assets/scans:/in" -v "$PWD/out:/out" \
  srp:test /in /out
python -c "import json;print(json.load(open('out/scanCPTEST0/scanCPTEST0.predictions.json'))['predict_code_sha'])"
```
Expected: exit 0; printed `predict_code_sha == deadbeef`. (CPU host is fine — device auto→cpu; the GPU path needs a Linux+NVIDIA host and is exercised via `/pre-merge`.)

- [ ] **Step 4: Commit.**

```bash
git add Dockerfile
git commit -m "feat: real linux_cuda exec-form Dockerfile ENTRYPOINT + code-sha ARG (#24)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: docker-build.yml — bake the sha + CUDA build reliability

**Files:**
- Modify: `.github/workflows/docker-build.yml`
- Modify: `tests/test_predict_container_packaging.py` (workflow guard)

**Interfaces:** Produces the `sha-<full>` GHCR tag A4 pins, with the code-sha baked.

- [ ] **Step 1: Write the failing workflow guard.** Append to `tests/test_predict_container_packaging.py`:

```python
def test_docker_workflow_bakes_code_sha():
    wf = (REPO / ".github/workflows/docker-build.yml").read_text()
    assert "SRP_PREDICT_CODE_SHA=${{ github.sha }}" in wf
    assert "type=sha,format=long" in wf
```

- [ ] **Step 2: Run it, verify it fails.**

Run: `uv run pytest tests/test_predict_container_packaging.py::test_docker_workflow_bakes_code_sha -v`
Expected: FAIL.

- [ ] **Step 3: Edit `.github/workflows/docker-build.yml`.**
  (a) In the `docker/metadata-action` `tags:` list, change `type=sha` → `type=sha,format=long` (so `sha-<full>` == the baked sha).
  (b) In the `docker/build-push-action` step, add:
```yaml
        build-args: |
          SRP_PREDICT_CODE_SHA=${{ github.sha }}
```
  (c) Change `cache-to: type=gha,mode=max` → `cache-to: type=gha,mode=min`.
  (d) Add a free-disk step before the build (the CUDA image is multi-GB on `ubuntu-latest`):
```yaml
      - name: Free disk space
        run: |
          sudo rm -rf /usr/share/dotnet /opt/ghc /usr/local/lib/android /opt/hostedtoolcache/CodeQL
          df -h
```

- [ ] **Step 4: Run the guard.**

Run: `uv run pytest tests/test_predict_container_packaging.py::test_docker_workflow_bakes_code_sha -v`
Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add .github/workflows/docker-build.yml tests/test_predict_container_packaging.py
git commit -m "ci: bake SRP_PREDICT_CODE_SHA + full-sha tag + CUDA build reliability (#24)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Docs + final validation

**Files:**
- Modify: `API.md`, `README.md`, `openspec/project.md`, `CLAUDE.md`, `CHANGELOG.md`

- [ ] **Step 1: `API.md`** — in the "Output Contract" / public-API section, add `run_batch` with its signature and a one-line description (the warm-batch runner over an input scan dir).
- [ ] **Step 2: `README.md`** — add `batch.py` + `__main__.py` to the Project Structure tree; add a "Run the container" Usage snippet: `docker run <image> <input_scan_dir> <output_dir>` (env `WANDB_API_KEY`); note provenance SHAs (`SRP_PREDICT_CODE_SHA`) are baked at image build time.
- [ ] **Step 3: `openspec/project.md`** — change "the container uses the CPU extra by default" → `linux_cuda`; drop "the serving protocol/CLI" from the remaining-A3/A4 note; add `output_contract.py`, `batch.py`, `__main__.py` to the Architecture Patterns module list.
- [ ] **Step 4: `CLAUDE.md`** — prune (don't expand) its duplicated Package-Structure/`__init__` export/env lists, replacing each with a one-line pointer to `openspec/project.md` / `API.md` / `README.md` (it is being retired — SSOT lives elsewhere).
- [ ] **Step 5: `CHANGELOG.md`** — add under `[Unreleased] → Added`: the predict container CLI (`python -m sleap_roots_predict` / `sleap-roots-predict`), `run_batch`, and the real GPU Dockerfile ENTRYPOINT (#24).
- [ ] **Step 6: Mark OpenSpec tasks 8-10 done** in `openspec/changes/add-predict-container-cli/tasks.md`.
- [ ] **Step 7: Full local gate.**

Run:
```bash
uv run black --check sleap_roots_predict tests
uv run ruff check sleap_roots_predict
uv run codespell
uv run pytest -m "not gpu and not acceptance and not wandb"
openspec validate add-predict-container-cli --strict
```
Expected: all green; validation "valid".

- [ ] **Step 8: Commit docs.**

```bash
git add API.md README.md openspec/project.md CLAUDE.md CHANGELOG.md openspec/changes/add-predict-container-cli/tasks.md
git commit -m "docs: document predict container CLI + run_batch; prune CLAUDE.md (#24)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage** (8 requirements → tasks):
1. Batch CLI over input dir → Tasks 2 (skeleton), 4 (`run_batch`), 7 (exit codes). Console/export scenarios → Tasks 2, 4.
2. Scan discovery + params from sidecar → Task 3 (incl. non-image/mismatch/missing-params/duplicate).
3. Per-scan outputs + sidecar pass-through → Task 4 (byte-identical binary copy).
4. Skip-if-exists resume → Task 6.
5. Failure isolation + exit code (+ zero-model failed, empty no-op) → Task 7.
6. Single-channel prediction input → Task 5.
7. GPU image with real exec-form entrypoint (no leftover CMD) → Task 8.
8. Baked predict_code_sha (`type=sha,format=long`) → Tasks 8 (ARG/ENV) + 9 (build-arg/tag).

**Placeholder scan:** none — every code step carries real code; docs steps name exact files + exact edits.

**Type consistency:** `ScanInput`/`ScanResult`/`BatchResult`/`run_batch`/`discover_scans` names + signatures are consistent across Tasks 3, 4, 7 and the CLI (`main` returns `0 if result.ok else 1`; `result.scans[*].status ∈ {ok,skipped,failed}`). `run_batch(source=...)` injection used consistently in tests. `worker.resolve/predict/inference_config/output_params` and `write_prediction_outputs(...)` match the real signatures verified in `warm_worker.py`/`output_contract.py`.

**Note for the executor:** `test_cli_main_exit_codes` patches `run_batch` to test wiring offline; if you prefer a fully-real CLI exit-code test, gate it `@pytest.mark.wandb` instead. Confirm `ResolvedParams` exposes `.values` (used in a couple of assertions/log messages); if the attribute name differs, adjust the assertion to reconstruct the dict.
