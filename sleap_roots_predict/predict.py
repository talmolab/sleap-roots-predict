"""SLEAP-NN prediction interface for pose estimation on videos.

Thin wrapper over the sleap-nn 0.3.0 inference API. It builds a reusable
:class:`~sleap_nn.inference.predictor.Predictor` (loaded once, reused across
videos) and runs inference on in-memory ``sleap_io.Video`` objects, handling
automatic device selection (CPU, CUDA, MPS).
"""

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Union

import sleap_io as sio
from sleap_nn.inference import Predictor

logger = logging.getLogger(__name__)

# Legacy SLEAP augmentation fields whose values must satisfy a lower bound in the
# sleap-nn 0.3.0 config schema. The 0.3.0 legacy mapper only clamps *upper*
# bounds, so an inert out-of-range legacy value (e.g. brightness_min_val=-10.0
# when brightness augmentation is disabled) fails config validation on load.
# Augmentation never runs at inference, so clamping these is behavior-preserving.
# Upstream fix tracked separately; extend this map if other fields surface.
_LEGACY_AUG_FLOORS = {
    "brightness_min_val": 0.0,
}


def _clamp_legacy_aug(config: dict) -> bool:
    """Clamp inert out-of-range legacy augmentation values in place.

    Args:
        config: A parsed legacy SLEAP ``training_config.json`` dict.

    Returns:
        ``True`` if any value was changed, else ``False``.
    """
    aug = config.get("optimization", {}).get("augmentation_config")
    if not isinstance(aug, dict):
        return False

    changed = False
    for key, floor in _LEGACY_AUG_FLOORS.items():
        value = aug.get(key)
        if isinstance(value, (int, float)) and value < floor:
            aug[key] = floor
            changed = True
    return changed


def _maybe_sanitize_legacy_config(model_dir: Path) -> Path:
    """Return a model directory that sleap-nn 0.3.0 can load.

    For legacy SLEAP models (``training_config.json`` and no
    ``training_config.yaml``) that carry inert out-of-range augmentation values,
    a temporary copy is made with the config sanitized and its path returned. The
    original model directory is never modified. Otherwise ``model_dir`` is
    returned unchanged.

    Args:
        model_dir: A model directory path.

    Returns:
        Either ``model_dir`` or a temporary directory holding a sanitized copy.
    """
    json_path = model_dir / "training_config.json"
    if (model_dir / "training_config.yaml").exists() or not json_path.exists():
        return model_dir

    config = json.loads(json_path.read_text())
    if not _clamp_legacy_aug(config):
        return model_dir

    tmp_root = Path(tempfile.mkdtemp(prefix="srp_legacy_"))
    sanitized_dir = tmp_root / model_dir.name
    shutil.copytree(model_dir, sanitized_dir)
    (sanitized_dir / "training_config.json").write_text(json.dumps(config))
    logger.warning(
        "Sanitized legacy augmentation config for %s (clamped inert out-of-range "
        "value(s) rejected by sleap-nn 0.3.0); loading from a temporary copy.",
        model_dir,
    )
    return sanitized_dir


def _resolve_device(device: str = "auto") -> str:
    """Resolve a device string, expanding ``"auto"`` to a concrete device.

    When ``device`` is ``"auto"``, the ``SRP_DEVICE`` environment variable (if
    set) takes precedence — useful for forcing ``"cpu"`` in environments where
    auto-detection picks an unusable accelerator (e.g. MPS on CI mac runners).

    Args:
        device: One of ``"auto"``, ``"cpu"``, ``"cuda"`` (or ``"cuda:N"``), or
            ``"mps"``. ``"auto"`` selects CUDA, then MPS, then CPU.

    Returns:
        A concrete device string. Non-``"auto"`` values are returned unchanged.
    """
    if device != "auto":
        return device

    override = os.environ.get("SRP_DEVICE")
    if override:
        return override

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_predictor(
    model_paths: List[Union[str, Path]],
    peak_threshold: float = 0.2,
    batch_size: int = 4,
    device: str = "auto",
) -> Predictor:
    """Create a reusable sleap-nn Predictor from one or more model directories.

    The returned predictor loads the model(s) once and can be reused across many
    videos (see :func:`predict_on_video`).

    Args:
        model_paths: Path(s) to trained model director(ies). For top-down models
            pass both the centroid and centered-instance directories.
        peak_threshold: Confidence threshold for peak detection.
        batch_size: Number of samples per batch for inference.
        device: Device for inference (``"auto"``, ``"cpu"``, ``"cuda"``, or
            ``"mps"``). ``"auto"`` selects the best available device.

    Returns:
        A sleap-nn :class:`~sleap_nn.inference.predictor.Predictor`.

    Raises:
        FileNotFoundError: If any model directory does not exist.
    """
    resolved_paths = []
    for p in model_paths:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"Model dir not found: {path}")
        resolved_paths.append(_maybe_sanitize_legacy_config(path))

    resolved_device = _resolve_device(device)

    return Predictor.from_model_paths(
        [path.as_posix() for path in resolved_paths],
        device=resolved_device,
        batch_size=batch_size,
        peak_threshold=peak_threshold,
    )


def predict_on_video(
    predictor: Predictor,
    video: "sio.Video",
    save_path: Optional[Union[str, Path]] = None,
) -> Union[Path, "sio.Labels"]:
    """Run prediction on a ``sleap_io.Video`` using a Predictor.

    Args:
        predictor: The Predictor to use for inference.
        video: A ``sleap_io.Video`` object.
        save_path: Optional path to save predictions as a ``.slp`` file. If
            ``None``, the ``sio.Labels`` object is returned without saving.

    Returns:
        The saved ``.slp`` :class:`~pathlib.Path` if ``save_path`` is given,
        otherwise the ``sio.Labels`` with predictions.
    """
    labels = predictor.predict(video, make_labels=True)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        sio.save_file(labels, save_path.as_posix())
        return save_path

    return labels
