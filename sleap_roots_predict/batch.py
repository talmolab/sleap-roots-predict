"""Warm-batch predict runner over a directory of staged scans.

Discovers scans (a ``{scan_key}.scan_metadata.json`` sidecar co-located with its
image frames in a dedicated directory), loads models once via a resident
:class:`~sleap_roots_predict.warm_worker.WarmModelWorker`, predicts each scan,
writes the prediction-output artifacts, and copies the sidecar through so each
``<output_dir>/{scan_key}/`` is a self-contained trait-extractor input tree.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from sleap_roots_contracts import ResolvedParams

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
    seen: Dict[str, Path] = {}
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
