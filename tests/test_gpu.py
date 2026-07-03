"""Real GPU inference tests.

Run locally with::

    uv run pytest -m gpu

On Windows + CUDA, install the CUDA profile once first::

    uv sync --extra dev --extra windows_cuda

These are marked ``gpu`` (deselected by default and on non-GPU CI runners) and
skip cleanly when no CUDA device is present.
"""

import pytest
import sleap_io as sio
import torch

from sleap_roots_predict.predict import (
    _resolve_device,
    make_predictor,
    predict_on_video,
)
from sleap_roots_predict.video_utils import make_video_from_images

pytestmark = pytest.mark.gpu

CUDA_AVAILABLE = torch.cuda.is_available()
_SKIP_REASON = "No CUDA device available (install --extra windows_cuda/linux_cuda)"


@pytest.mark.skipif(not CUDA_AVAILABLE, reason=_SKIP_REASON)
def test_cuda_is_available():
    """Sanity check that torch sees a CUDA device (GPU actually works)."""
    assert torch.cuda.device_count() >= 1
    assert torch.version.cuda is not None


@pytest.mark.skipif(not CUDA_AVAILABLE, reason=_SKIP_REASON)
def test_resolve_device_auto_selects_cuda():
    """device='auto' resolves to cuda when a CUDA device is present."""
    assert _resolve_device("auto") == "cuda"


@pytest.mark.skipif(not CUDA_AVAILABLE, reason=_SKIP_REASON)
def test_predict_on_video_runs_on_cuda(native_model_dir, centered_pair_image_dir):
    """make_predictor(device='cuda') runs real inference on the GPU."""
    predictor = make_predictor([native_model_dir], device="cuda")
    # Predictor is configured for CUDA; a successful predict below confirms the
    # model actually loaded and ran on the GPU.
    assert str(predictor.device).startswith("cuda")

    files = sorted(centered_pair_image_dir.glob("*.png"))
    video = make_video_from_images(files, greyscale=True)
    labels = predict_on_video(predictor, video)
    assert isinstance(labels, sio.Labels)
    assert sum(len(lf.instances) for lf in labels) > 0
