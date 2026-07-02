"""Guarded acceptance test: real inference on real cylinder data.

Deselected by default and never run in CI. It runs only when two environment
variables point at real local data:

- ``SRP_CYLINDER_DIR`` — a directory of real cylinder/plate image frames.
- ``SRP_MODEL_DIRS`` — one or more real root-model directories, joined by the
  OS path separator (``;`` on Windows, ``:`` elsewhere). Extract legacy SLEAP
  ``.zip`` models to directories first and point at those.

Optional:

- ``SRP_IMAGE_PATTERN`` — glob for image frames (default ``*.jpg``).
- ``SRP_GREYSCALE`` — ``"1"`` to build a greyscale video (default off; sleap-nn
  preprocessing converts channels per the model config).

Run with::

    SRP_CYLINDER_DIR=... SRP_MODEL_DIRS=... uv run pytest -m acceptance -s

A model that cannot be loaded under sleap-nn 0.3.0 fails the test with a clear
message naming the model directory — an expected, informative outcome that
feeds the prediction-parity slice.
"""

import os
from pathlib import Path

import pytest
import sleap_io as sio

from sleap_roots_predict.predict import make_predictor, predict_on_video
from sleap_roots_predict.video_utils import make_video_from_images

CYLINDER_DIR = os.environ.get("SRP_CYLINDER_DIR")
MODEL_DIRS = os.environ.get("SRP_MODEL_DIRS")
IMAGE_PATTERN = os.environ.get("SRP_IMAGE_PATTERN", "*.jpg")
GREYSCALE = os.environ.get("SRP_GREYSCALE", "") == "1"

pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.skipif(
        not (CYLINDER_DIR and MODEL_DIRS),
        reason="Set SRP_CYLINDER_DIR and SRP_MODEL_DIRS to run the acceptance test",
    ),
]


def _model_dirs() -> list[Path]:
    return [Path(p) for p in MODEL_DIRS.split(os.pathsep) if p]


def test_predict_on_real_cylinder_images(tmp_path):
    """End-to-end: image dir -> video -> real prediction -> .slp, per model."""
    image_dir = Path(CYLINDER_DIR)
    files = sorted(image_dir.glob(IMAGE_PATTERN))
    assert files, f"No images matching {IMAGE_PATTERN!r} in {image_dir}"
    print(f"\n[acceptance] {len(files)} frames from {image_dir}")

    video = make_video_from_images(files, greyscale=GREYSCALE)

    for model_dir in _model_dirs():
        try:
            predictor = make_predictor([model_dir])
        except Exception as e:  # noqa: BLE001 - surface load failures clearly
            pytest.fail(
                f"Model failed to load under sleap-nn 0.3.0: {model_dir}\n"
                f"{type(e).__name__}: {e}"
            )

        out = tmp_path / f"{model_dir.name}.predictions.slp"
        labels = predict_on_video(predictor, video, save_path=out)

        assert out.exists()
        reloaded = sio.load_file(out.as_posix())
        assert isinstance(reloaded, sio.Labels)
        n_inst = sum(len(lf.instances) for lf in reloaded)
        print(
            f"[acceptance] {model_dir.name}: "
            f"labeled_frames={len(reloaded)} instances={n_inst} -> {out.name}"
        )
