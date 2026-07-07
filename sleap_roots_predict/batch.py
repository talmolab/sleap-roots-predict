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

from sleap_roots_contracts import ResolvedParams

from sleap_roots_predict.model_registry import ModelCardSource
from sleap_roots_predict.output_contract import write_prediction_outputs
from sleap_roots_predict.video_utils import make_video_from_images, natural_sort
from sleap_roots_predict.warm_worker import WarmModelWorker

logger = logging.getLogger(__name__)

_SIDECAR_SUFFIX = ".scan_metadata.json"
_IMAGE_EXTENSIONS = frozenset({".png", ".tif", ".tiff", ".jpg", ".jpeg"})
_REQUIRED_PARAM_KEYS = ("species", "mode", "age")


@dataclass(frozen=True)
class ScanInput:
    """A discovered scan.

    ``error`` is set (and ``params``/``frames`` may be empty) when the sidecar is
    invalid — an isolated per-scan failure, not a batch abort.
    """

    scan_key: str
    sidecar_path: Path
    frames: list[Path] = field(default_factory=list)
    params: ResolvedParams | None = None
    error: str | None = None


@dataclass(frozen=True)
class ScanResult:
    """Per-scan outcome. ``status`` is one of ``ok`` / ``skipped`` / ``failed``."""

    scan_key: str
    status: str
    error: str | None = None


@dataclass
class BatchResult:
    """Aggregate batch outcome."""

    scans: list[ScanResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True iff no scan failed (skipped/ok scans are fine)."""
        return all(s.status != "failed" for s in self.scans)


def discover_scans(input_dir: str | Path) -> list[ScanInput]:
    """Discover scans under ``input_dir`` by their scan-metadata sidecars.

    Recursively finds ``*.scan_metadata.json`` files; each sidecar's parent
    directory holds that scan's frames. ``scan_key`` is the sidecar's filename
    stem and must equal the sidecar's internal ``scan_key``. Invalid scans are
    returned with ``.error`` set (isolated failure); a duplicate ``scan_key``
    anywhere in the tree raises.

    Args:
        input_dir: Directory of staged scans (must exist).

    Returns:
        One :class:`ScanInput` per discovered sidecar, sorted by path.

    Raises:
        FileNotFoundError: If ``input_dir`` does not exist (a mis-configured mount,
            distinct from an empty-but-present directory which is a no-op).
        ValueError: If two sidecars share a ``scan_key``.
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(
            f"input scan directory does not exist: {input_dir.as_posix()}"
        )
    scans: list[ScanInput] = []
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
    # Frames are the co-located images, natural-sorted so frame_2 precedes frame_10
    # (the frame order is the temporal order of the inference video). Non-image files
    # (including the sidecar itself) and subdirectories are excluded. natural_sort
    # returns strings, so map back to Path to keep the list[Path] contract.
    frames = [
        Path(s)
        for s in natural_sort(
            [
                p
                for p in sidecar.parent.iterdir()
                if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
            ]
        )
    ]
    resolved = ResolvedParams(values={k: params[k] for k in _REQUIRED_PARAM_KEYS})
    return ScanInput(scan_key, sidecar, frames=frames, params=resolved)


def run_batch(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    source: ModelCardSource | None = None,
    predict_code_sha: str | None = None,
    predict_container_digest: str | None = None,
) -> BatchResult:
    """Predict every scan under ``input_dir``, writing outputs under ``output_dir``.

    Loads models once via a single resident worker. Per scan: skip if its manifest
    already exists (resume); otherwise resolve + predict, write the prediction-output
    artifacts into ``output_dir/{scan_key}/``, and copy the sidecar through. A
    per-scan error is isolated (recorded ``failed``, batch continues). An empty (but
    present) input directory is a no-op.

    Args:
        input_dir: Directory of staged scans.
        output_dir: Directory to write per-scan outputs into.
        source: Model-card source; ``None`` uses the production WandbRegistrySource.
        predict_code_sha: Provenance sha (falls back to ``SRP_PREDICT_CODE_SHA``).
        predict_container_digest: Provenance digest (env fallback).

    Returns:
        A :class:`BatchResult` with one :class:`ScanResult` per scan.

    Raises:
        FileNotFoundError: If ``input_dir`` does not exist.
        ValueError: If two sidecars share a ``scan_key`` (a batch-level staging error,
            surfaced before any prediction).
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    scans = discover_scans(input_dir)
    result = BatchResult()
    if not scans:
        logger.warning("No scans discovered under %s", input_dir.as_posix())
        return result

    worker = WarmModelWorker(source=source)
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
    predict_code_sha: str | None,
    predict_container_digest: str | None,
) -> None:
    """Predict one scan and write its outputs + copied sidecar. Raises on failure."""
    if not scan.frames:
        raise ValueError(
            f"no image frames co-located with sidecar {scan.sidecar_path.as_posix()}"
        )
    assert scan.params is not None  # run_batch filters error scans (params-None) first
    refs = worker.resolve(scan.params)
    if not refs:
        # A scan matching no model for any root type is a hard per-scan failure rather
        # than an empty-artifacts manifest: write_prediction_outputs permits an empty
        # manifest, but the downstream trait-extractor rejects one, so surface it here.
        raise ValueError(f"no models resolved for params {scan.params.values!r}")
    video = make_video_from_images(scan.frames, greyscale=True)
    labels = worker.predict(scan.params, video)
    out_scan_dir.mkdir(parents=True, exist_ok=True)
    # Copy the sidecar BEFORE the manifest: write_prediction_outputs writes the manifest
    # last as the resume commit-marker, so the sidecar must already be present when it
    # lands — else a crash in between leaves a manifest with no sidecar that resume skips
    # forever and the trait-extractor then rejects (an incomplete input tree).
    shutil.copyfile(
        scan.sidecar_path, out_scan_dir / f"{scan.scan_key}{_SIDECAR_SUFFIX}"
    )
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
