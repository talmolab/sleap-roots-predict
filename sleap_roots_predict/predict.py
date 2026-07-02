"""SLEAP-NN prediction interface for pose estimation on videos.

Thin wrapper over the sleap-nn 0.3.0 inference API. It builds a reusable
:class:`~sleap_nn.inference.predictor.Predictor` (loaded once, reused across
videos) and runs inference on in-memory ``sleap_io.Video`` objects, handling
automatic device selection (CPU, CUDA, MPS).
"""

from pathlib import Path
from typing import List, Optional, Union

import sleap_io as sio
from sleap_nn.inference import Predictor


def _resolve_device(device: str = "auto") -> str:
    """Resolve a device string, expanding ``"auto"`` to a concrete device.

    Args:
        device: One of ``"auto"``, ``"cpu"``, ``"cuda"`` (or ``"cuda:N"``), or
            ``"mps"``. ``"auto"`` selects CUDA, then MPS, then CPU.

    Returns:
        A concrete device string. Non-``"auto"`` values are returned unchanged.
    """
    if device != "auto":
        return device

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
    resolved_paths = [Path(p) for p in model_paths]
    for path in resolved_paths:
        if not path.exists():
            raise FileNotFoundError(f"Model dir not found: {path}")

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
