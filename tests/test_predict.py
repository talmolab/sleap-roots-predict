"""Real (no-mock) inference tests for ``sleap_roots_predict.predict``.

These run actual sleap-nn 0.3.0 inference on CPU against vendored minimal
bottom-up models (see ``tests/assets/README.md``). Nothing here mocks the
sleap-nn or sleap-io boundary — a passing test means real inference ran.
"""

import json
import shutil
from pathlib import Path

import pytest
import sleap_io as sio
from sleap_nn.inference import Predictor

from sleap_roots_predict.predict import (
    _clamp_legacy_aug,
    make_predictor,
    predict_on_video,
)
from sleap_roots_predict.video_utils import make_video_from_images


@pytest.fixture(scope="session")
def native_predictor(native_model_dir: Path) -> Predictor:
    """A predictor built once from the native minimal bottom-up model."""
    return make_predictor([native_model_dir])


@pytest.fixture(scope="session")
def centered_pair_video(centered_pair_image_dir: Path):
    """A small (8-frame) video built from vendored image frames."""
    files = sorted(centered_pair_image_dir.glob("*.png"))
    return make_video_from_images(files, greyscale=True)


# --- make_predictor -----------------------------------------------------------


def test_make_predictor_builds_real_predictor(native_model_dir: Path):
    """make_predictor returns a live sleap-nn Predictor from a real model dir."""
    predictor = make_predictor([native_model_dir])
    assert isinstance(predictor, Predictor)


def test_make_predictor_auto_device(native_model_dir: Path):
    """device='auto' builds a real predictor (device resolved on the host)."""
    predictor = make_predictor([native_model_dir], device="auto")
    assert isinstance(predictor, Predictor)


def test_make_predictor_missing_dir_raises(tmp_path: Path):
    """A nonexistent model directory raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        make_predictor([tmp_path / "does_not_exist"])


def test_resolve_device_env_override(monkeypatch):
    """SRP_DEVICE overrides auto-detection but not an explicit device."""
    from sleap_roots_predict.predict import _resolve_device

    monkeypatch.setenv("SRP_DEVICE", "cpu")
    assert _resolve_device("auto") == "cpu"
    assert _resolve_device("cuda") == "cuda"  # explicit device is respected


# --- predict_on_video ---------------------------------------------------------


def test_predict_on_video_returns_labels(native_predictor, centered_pair_video):
    """Real inference returns sio.Labels containing predicted instances."""
    labels = predict_on_video(native_predictor, centered_pair_video)
    assert isinstance(labels, sio.Labels)
    assert sum(len(lf.instances) for lf in labels) > 0


def test_predict_on_video_saves_slp(native_predictor, centered_pair_video, tmp_path):
    """With save_path, a reloadable .slp is written and the Path returned."""
    out = tmp_path / "nested" / "preds.slp"
    result = predict_on_video(native_predictor, centered_pair_video, save_path=out)
    assert result == out
    assert out.exists()
    reloaded = sio.load_file(out.as_posix())
    assert len(reloaded) > 0


def test_predict_on_video_legacy_model(legacy_model_dir: Path, centered_pair_video):
    """A legacy SLEAP UNet model loads under sleap-nn 0.3.0 and predicts."""
    predictor = make_predictor([legacy_model_dir])
    labels = predict_on_video(predictor, centered_pair_video)
    assert isinstance(labels, sio.Labels)
    assert sum(len(lf.instances) for lf in labels) > 0


def _make_bad_legacy_model(legacy_model_dir: Path, dest: Path) -> Path:
    """Copy a legacy model and set an out-of-range (inert) brightness_min_val."""
    shutil.copytree(legacy_model_dir, dest)
    cfg_path = dest / "training_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["optimization"]["augmentation_config"]["brightness_min_val"] = -10.0
    cfg_path.write_text(json.dumps(cfg))
    return dest


def test_clamp_legacy_aug_clamps_negative_brightness():
    """The clamp helper raises a negative brightness_min_val to the floor."""
    cfg = {"optimization": {"augmentation_config": {"brightness_min_val": -10.0}}}
    changed = _clamp_legacy_aug(cfg)
    assert changed is True
    assert cfg["optimization"]["augmentation_config"]["brightness_min_val"] == 0.0


def test_clamp_legacy_aug_leaves_valid_values():
    """A valid config is left unchanged (no needless sanitization)."""
    cfg = {"optimization": {"augmentation_config": {"brightness_min_val": 0.9}}}
    assert _clamp_legacy_aug(cfg) is False
    assert cfg["optimization"]["augmentation_config"]["brightness_min_val"] == 0.9


def test_make_predictor_sanitizes_legacy_config(
    legacy_model_dir: Path, tmp_path: Path, centered_pair_video
):
    """A legacy model with an inert negative brightness value loads and predicts.

    This reproduces the real production 'lateral' model failure: sleap-nn 0.3.0
    rejects brightness_min_val < 0. make_predictor must sanitize and load it.
    """
    bad_model = _make_bad_legacy_model(legacy_model_dir, tmp_path / "legacy_bad")
    predictor = make_predictor([bad_model])
    assert isinstance(predictor, Predictor)
    labels = predict_on_video(predictor, centered_pair_video)
    assert sum(len(lf.instances) for lf in labels) > 0


def test_predictor_reused_across_videos(native_predictor, centered_pair_image_dir):
    """One persistent predictor produces labels for two different videos."""
    files = sorted(centered_pair_image_dir.glob("*.png"))
    video_a = make_video_from_images(files[:4], greyscale=True)
    video_b = make_video_from_images(files[4:], greyscale=True)
    labels_a = predict_on_video(native_predictor, video_a)
    labels_b = predict_on_video(native_predictor, video_b)
    assert isinstance(labels_a, sio.Labels) and isinstance(labels_b, sio.Labels)
    assert len(labels_a) == 4 and len(labels_b) == 4
