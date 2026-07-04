"""Real, no-mock tests for the warm model worker.

A ``LocalCardSource`` over the vendored native + legacy models drives the whole
warm path: resolve -> materialize -> ``make_predictor`` -> real CPU inference. No
sleap-nn / sleap-io boundary is mocked. Residency is asserted by object identity
(the same ``Predictor`` instance is reused across calls).
"""

from pathlib import Path

import pytest
import sleap_io as sio
from sleap_nn.inference import Predictor
from sleap_roots_contracts import ModelCard, ResolvedParams

from sleap_roots_predict.model_registry import LocalCardSource
from sleap_roots_predict.video_utils import make_video_from_images
from sleap_roots_predict.warm_worker import WarmModelWorker


def _card(
    root_type, registry_id, *, species="rice", version="v1", age_min=2, age_max=5
):
    return ModelCard(
        species=species,
        mode="cylinder",
        age_min=age_min,
        age_max=age_max,
        root_type=root_type,
        registry_id=registry_id,
        version=version,
    )


def _params(species="rice", age=3):
    return ResolvedParams(values={"species": species, "mode": "cylinder", "age": age})


@pytest.fixture(scope="module")
def video(centered_pair_image_dir: Path):
    """An 8-frame greyscale video built from vendored frames."""
    files = sorted(centered_pair_image_dir.glob("*.png"))
    return make_video_from_images(files, greyscale=True)


@pytest.fixture
def rice_source(native_model_dir: Path, legacy_model_dir: Path) -> LocalCardSource:
    """A source: primary=native model, lateral=legacy model, both for rice."""
    return LocalCardSource(
        [
            (_card("primary", "reg/rice-primary"), native_model_dir),
            (_card("lateral", "reg/rice-lateral"), legacy_model_dir),
        ]
    )


def test_resolve_returns_refs_without_loading(rice_source):
    """Resolve selects ModelRefs but materializes/loads nothing."""
    worker = WarmModelWorker(rice_source)
    refs = worker.resolve(_params())
    assert set(refs) == {"primary", "lateral"}
    assert worker._predictors == {}  # nothing loaded


def test_get_predictors_loads_once_and_reuses(rice_source):
    """A second call returns the same resident Predictor instances (warm)."""
    worker = WarmModelWorker(rice_source)
    first = worker.get_predictors(_params())
    second = worker.get_predictors(_params())
    assert set(first) == {"primary", "lateral"}
    assert all(isinstance(p, Predictor) for p in first.values())
    assert first["primary"] is second["primary"]
    assert first["lateral"] is second["lateral"]


def test_shared_model_version_hits_cache(rice_source):
    """Different params resolving to the same model version share one Predictor."""
    worker = WarmModelWorker(rice_source)
    a = worker.get_predictors(_params(age=3))
    b = worker.get_predictors(_params(age=4))  # same primary card (window 2-5)
    assert a["primary"] is b["primary"]


def test_zero_match_root_type_skipped(native_model_dir, legacy_model_dir):
    """A root type whose only card mismatches the species is skipped end-to-end."""
    source = LocalCardSource(
        [
            (_card("primary", "reg/rice-primary"), native_model_dir),
            (_card("lateral", "reg/soy-lateral", species="soybean"), legacy_model_dir),
        ]
    )
    worker = WarmModelWorker(source)
    result = worker.get_predictors(_params(species="rice"))
    assert set(result) == {"primary"}


def test_predict_returns_labels_per_root_type(rice_source, video):
    """Predict runs real inference and returns sio.Labels per resolved root type."""
    worker = WarmModelWorker(rice_source)
    labels = worker.predict(_params(), video)
    assert set(labels) == {"primary", "lateral"}
    for lab in labels.values():
        assert isinstance(lab, sio.Labels)
        assert sum(len(lf.instances) for lf in lab) > 0


def test_predict_save_dir_writes_raw_slp_only(rice_source, video, tmp_path):
    """save_dir writes one reloadable raw .slp per root type and no manifest."""
    worker = WarmModelWorker(rice_source)
    worker.predict(_params(), video, save_dir=tmp_path)
    for root_type in ("primary", "lateral"):
        slp = tmp_path / f"{root_type}.slp"
        assert slp.exists()
        assert len(sio.load_file(slp.as_posix())) > 0
    # Deferred output contract: no predictions.csv, no scan-aware naming.
    assert not (tmp_path / "predictions.csv").exists()
    assert not list(tmp_path.glob("*.model*.root*.slp"))


def test_get_predictors_fails_loud_on_unloadable_model(tmp_path):
    """An unmaterializable/unloadable model raises, naming root type + id:version."""
    bad_dir = tmp_path / "does_not_exist"
    source = LocalCardSource([(_card("primary", "reg/bad", version="v9"), bad_dir)])
    worker = WarmModelWorker(source)
    with pytest.raises(RuntimeError, match=r"reg/bad:v9") as exc:
        worker.get_predictors(_params())
    assert "primary" in str(exc.value)
