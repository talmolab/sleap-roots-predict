"""Real, no-mock tests for the predict output contract.

The writer is driven by the real warm worker over the vendored native + legacy
models (reusing the `rice_source` / `video` fixtures pattern from
``test_warm_worker.py``); no sleap-nn / sleap-io boundary is mocked.
"""

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest
import sleap_io as sio
from sleap_roots_contracts import ModelCard, ModelRef, ResolvedParams

from sleap_roots_predict.model_registry import LocalCardSource
from sleap_roots_predict.output_contract import (
    _SCAN_KEY_FORBIDDEN,
    PredictionArtifact,
    PredictionManifest,
    ScanRequest,
    _validate_scan_key,
    predict_and_write_batch,
    slugify_model_id,
    write_prediction_outputs,
)
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
    """An 8-frame greyscale video built from the vendored frames."""
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


def _ref(root_type="primary", registry_id="reg/rice-primary", version="v1"):
    return ModelRef(
        registry_id=registry_id,
        version=version,
        sleap_nn_version="0.3.0",
        root_type=root_type,
    )


# --- Task 2: schema models --------------------------------------------------


def test_manifest_round_trips():
    """A manifest with a real ModelRef dumps to JSON and reloads equal."""
    artifact = PredictionArtifact(
        root_type="primary",
        model_id="reg-rice-primary-v1",
        model=_ref(),
        slp_path="s1.modelreg-rice-primary-v1.rootprimary.slp",
        checksum="abc123",
        file_size=42,
    )
    manifest = PredictionManifest(
        scan_key="s1",
        plant_qr_code="s1",
        artifacts=[artifact],
        predict_inference_config={"device": "cpu", "peak_threshold": 0.2},
        predict_output_params={"peak_threshold": 0.2},
    )
    reloaded = PredictionManifest.model_validate_json(manifest.model_dump_json())
    assert reloaded == manifest
    assert reloaded.schema_version == "1"
    assert reloaded.artifacts[0].model == _ref()


def test_plant_qr_code_defaults_to_scan_key():
    """An unset plant_qr_code defaults to scan_key."""
    manifest = PredictionManifest(scan_key="scan0731")
    assert manifest.plant_qr_code == "scan0731"


# --- Task 3: slug + scan_key validation -------------------------------------


def test_model_id_slug_is_filename_safe():
    """A registry_id with a slash and a dot yields a slash/dot-free slug."""
    slug = slugify_model_id(_ref(registry_id="reg/rice.primary", version="v1"))
    assert "/" not in slug and "." not in slug
    assert slug == "reg-rice-primary-v1"


@pytest.mark.parametrize(
    "bad", sorted(_SCAN_KEY_FORBIDDEN) + ["\x00", "\n", "\t", "\r"]
)
def test_rejects_reserved_and_control_chars_in_scan_key(bad):
    """Every reserved char + control chars are rejected (not mangled)."""
    with pytest.raises(ValueError):
        _validate_scan_key(f"scan{bad}1")


def test_rejects_empty_scan_key():
    """An empty scan_key is rejected."""
    with pytest.raises(ValueError):
        _validate_scan_key("")


@pytest.mark.parametrize("key", [" ", "   ", "scan ", " scan"])
def test_rejects_whitespace_scan_key(key):
    """Leading/trailing/all-whitespace keys are rejected (Windows-mangle-prone)."""
    with pytest.raises(ValueError):
        _validate_scan_key(key)


def test_accepts_normal_scan_key():
    """An alphanumeric key with `_`/`-` is accepted."""
    _validate_scan_key("scan_0731-A")  # no raise


# --- Task 4: pure writer (real inference, no mocks) -------------------------


def test_writer_writes_named_slp_and_manifest(rice_source, video, tmp_path):
    """Real inference -> named .slp per root + a manifest mapping each root."""
    worker = WarmModelWorker(rice_source)
    refs = worker.resolve(_params())
    labels = worker.predict(_params(), video)
    manifest = write_prediction_outputs(
        labels,
        refs,
        tmp_path,
        scan_key="scan0731",
        inference_config=worker.inference_config(),
        output_params=worker.output_params(),
    )
    assert {a.root_type for a in manifest.artifacts} == {"primary", "lateral"}
    for art in manifest.artifacts:
        slp = tmp_path / art.slp_path
        assert slp.exists()
        assert len(sio.load_file(slp.as_posix())) > 0
        assert art.model_id == slugify_model_id(refs[art.root_type])
        assert art.model == refs[art.root_type]


def test_slp_path_is_relocatable_basename(rice_source, video, tmp_path):
    """slp_path is a bare POSIX basename (no dir), so the scan dir is relocatable."""
    worker = WarmModelWorker(rice_source)
    manifest = write_prediction_outputs(
        worker.predict(_params(), video),
        worker.resolve(_params()),
        tmp_path,
        scan_key="scan0731",
        inference_config=worker.inference_config(),
        output_params=worker.output_params(),
    )
    for art in manifest.artifacts:
        assert not Path(art.slp_path).is_absolute()
        assert "/" not in art.slp_path and "\\" not in art.slp_path
        assert art.slp_path == (
            f"scan0731.model{slugify_model_id(art.model)}.root{art.root_type}.slp"
        )
        assert (tmp_path / art.slp_path).exists()


def test_writer_covers_all_root_types(all_roots_source, video, tmp_path):
    """All three RootType literals (incl. crown) produce named .slp + artifacts."""
    worker = WarmModelWorker(all_roots_source)
    manifest = write_prediction_outputs(
        worker.predict(_params(), video),
        worker.resolve(_params()),
        tmp_path,
        scan_key="scan0731",
        inference_config=worker.inference_config(),
        output_params=worker.output_params(),
    )
    assert {a.root_type for a in manifest.artifacts} == {"primary", "lateral", "crown"}
    assert (
        tmp_path / f"scan0731.model{slugify_model_id(_ref('crown', 'reg/rice-crown'))}"
        ".rootcrown.slp"
    ).exists()


def test_manifest_json_on_disk_round_trips(rice_source, video, tmp_path):
    """The written JSON file reloads into a manifest equal to the returned one."""
    worker = WarmModelWorker(rice_source)
    manifest = write_prediction_outputs(
        worker.predict(_params(), video),
        worker.resolve(_params()),
        tmp_path,
        scan_key="scan0731",
        inference_config=worker.inference_config(),
        output_params=worker.output_params(),
    )
    on_disk = (tmp_path / "scan0731.predictions.json").read_text(encoding="utf-8")
    assert PredictionManifest.model_validate_json(on_disk) == manifest


def test_checksums_and_sizes_match_files(rice_source, video, tmp_path):
    """Each artifact's checksum/file_size match the on-disk .slp."""
    worker = WarmModelWorker(rice_source)
    manifest = write_prediction_outputs(
        worker.predict(_params(), video),
        worker.resolve(_params()),
        tmp_path,
        scan_key="scan0731",
        inference_config=worker.inference_config(),
        output_params=worker.output_params(),
    )
    for art in manifest.artifacts:
        data = (tmp_path / art.slp_path).read_bytes()
        assert art.checksum == hashlib.sha256(data).hexdigest()
        assert art.file_size == len(data)


def test_mismatched_labels_and_refs_raise(tmp_path):
    """Label root types not matching ref root types raise ValueError."""
    with pytest.raises(ValueError):
        write_prediction_outputs(
            {"primary": sio.Labels()},
            {"lateral": _ref("lateral", "reg/rice-lateral")},
            tmp_path,
            scan_key="s1",
            inference_config={},
            output_params={},
        )


def test_writer_rejects_unsafe_scan_key(tmp_path):
    """The writer rejects an unsafe scan_key and writes nothing."""
    with pytest.raises(ValueError):
        write_prediction_outputs(
            {}, {}, tmp_path, scan_key="scan.1", inference_config={}, output_params={}
        )
    assert not list(tmp_path.iterdir())


def test_zero_roots_writes_empty_artifacts(tmp_path):
    """A scan with no resolved roots still writes a manifest with artifacts=[]."""
    manifest = write_prediction_outputs(
        {}, {}, tmp_path, scan_key="scan0", inference_config={}, output_params={}
    )
    assert manifest.artifacts == []
    assert (tmp_path / "scan0.predictions.json").exists()


def test_rerun_overwrites_in_place(rice_source, video, tmp_path):
    """Re-running the same scan into the same dir overwrites in place."""
    worker = WarmModelWorker(rice_source)
    kwargs = dict(
        scan_key="scan0731",
        inference_config=worker.inference_config(),
        output_params=worker.output_params(),
    )
    write_prediction_outputs(
        worker.predict(_params(), video), worker.resolve(_params()), tmp_path, **kwargs
    )
    second = write_prediction_outputs(
        worker.predict(_params(), video), worker.resolve(_params()), tmp_path, **kwargs
    )
    on_disk = (tmp_path / "scan0731.predictions.json").read_text(encoding="utf-8")
    assert PredictionManifest.model_validate_json(on_disk) == second


def test_rerun_with_changed_model_prunes_stale_slp(rice_source, video, tmp_path):
    """Re-running with a different model for a root type removes the stale .slp."""
    worker = WarmModelWorker(rice_source)
    kwargs = dict(
        scan_key="scan0731",
        inference_config=worker.inference_config(),
        output_params=worker.output_params(),
    )
    # Run 1: primary -> reg/rice-primary (native model, one slug).
    write_prediction_outputs(
        worker.predict(_params(), video), worker.resolve(_params()), tmp_path, **kwargs
    )
    # Run 2: override primary -> reg/rice-lateral (a different slug/filename).
    override = {
        "primary": ModelRef(
            registry_id="reg/rice-lateral",
            version="v1",
            sleap_nn_version="0.3.0",
            root_type="primary",
        )
    }
    write_prediction_outputs(
        worker.predict(_params(), video, overrides=override),
        worker.resolve(_params(), override),
        tmp_path,
        **kwargs,
    )
    # The stale run-1 primary .slp was pruned: exactly one remains.
    assert len(list(tmp_path.glob("*.rootprimary.slp"))) == 1


def test_manifest_records_worker_provenance(rice_source, video, tmp_path):
    """The writer stores the worker's inference config + output params verbatim."""
    worker = WarmModelWorker(rice_source, peak_threshold=0.15)
    manifest = write_prediction_outputs(
        worker.predict(_params(), video),
        worker.resolve(_params()),
        tmp_path,
        scan_key="scan0731",
        inference_config=worker.inference_config(),
        output_params=worker.output_params(),
    )
    assert manifest.predict_inference_config == worker.inference_config()
    assert manifest.predict_output_params == worker.output_params()
    assert manifest.predict_output_params == {"peak_threshold": 0.15}


def test_plant_qr_code_recorded_verbatim(tmp_path):
    """An explicit plant_qr_code is stored as-is (non-default path)."""
    manifest = write_prediction_outputs(
        {},
        {},
        tmp_path,
        scan_key="scan0",
        plant_qr_code="QR-123",
        inference_config={},
        output_params={},
    )
    assert manifest.plant_qr_code == "QR-123"


def test_writer_does_not_import_sleap_roots():
    """Importing the writer module must not pull in the sleap-roots runtime dep."""
    code = (
        "import sys, sleap_roots_predict.output_contract; "
        "assert 'sleap_roots' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


# --- Task 5: fail-soft build identity ---------------------------------------


def test_build_identity_explicit_arg_wins(tmp_path, monkeypatch):
    """An explicit predict_code_sha overrides the environment."""
    monkeypatch.setenv("SRP_PREDICT_CODE_SHA", "from-env")
    manifest = write_prediction_outputs(
        {},
        {},
        tmp_path,
        scan_key="s",
        inference_config={},
        output_params={},
        predict_code_sha="explicit",
    )
    assert manifest.predict_code_sha == "explicit"


def test_build_identity_env_fallback(tmp_path, monkeypatch):
    """With no argument, the environment variable is used."""
    monkeypatch.setenv("SRP_PREDICT_CONTAINER_DIGEST", "sha256:env")
    manifest = write_prediction_outputs(
        {}, {}, tmp_path, scan_key="s", inference_config={}, output_params={}
    )
    assert manifest.predict_container_digest == "sha256:env"


def test_build_identity_absent_is_empty_string(tmp_path, monkeypatch):
    """Absent argument and environment record the empty string without error."""
    monkeypatch.delenv("SRP_PREDICT_CODE_SHA", raising=False)
    monkeypatch.delenv("SRP_PREDICT_CONTAINER_DIGEST", raising=False)
    manifest = write_prediction_outputs(
        {}, {}, tmp_path, scan_key="s", inference_config={}, output_params={}
    )
    assert manifest.predict_code_sha == ""
    assert manifest.predict_container_digest == ""


# --- Task 6: ScanRequest + batch --------------------------------------------


def test_batch_writes_per_scan_subdirs(rice_source, video, tmp_path):
    """Batch writes one subdirectory of artifacts per scan."""
    worker = WarmModelWorker(rice_source)
    requests = [
        ScanRequest(scan_key="s1", video=video, params=_params()),
        ScanRequest(scan_key="s2", video=video, params=_params()),
    ]
    manifests = predict_and_write_batch(worker, requests, tmp_path)
    assert len(manifests) == 2
    for scan_key in ("s1", "s2"):
        scan_dir = tmp_path / scan_key
        assert (scan_dir / f"{scan_key}.predictions.json").exists()
        assert list(scan_dir.glob("*.rootprimary.slp"))
        assert list(scan_dir.glob("*.rootlateral.slp"))


def test_batch_reuses_resident_predictors(rice_source, video, tmp_path):
    """The second scan reuses the resident Predictor loaded for the first."""
    worker = WarmModelWorker(rice_source)
    predict_and_write_batch(worker, [ScanRequest("s1", video, _params())], tmp_path)
    primary = worker._predictors[("reg/rice-primary", "v1")]
    predict_and_write_batch(worker, [ScanRequest("s2", video, _params())], tmp_path)
    assert worker._predictors[("reg/rice-primary", "v1")] is primary


def test_batch_respects_overrides(rice_source, video, tmp_path):
    """A per-scan override reaches resolution and is recorded in the manifest."""
    override = ModelRef(
        registry_id="reg/rice-lateral",
        version="v1",
        sleap_nn_version="0.3.0",
        root_type="primary",
    )
    worker = WarmModelWorker(rice_source)
    manifests = predict_and_write_batch(
        worker,
        [ScanRequest("s1", video, _params(), overrides={"primary": override})],
        tmp_path,
    )
    primary_art = next(a for a in manifests[0].artifacts if a.root_type == "primary")
    assert primary_art.model == override


def test_batch_rejects_duplicate_scan_keys(rice_source, video, tmp_path):
    """Two requests sharing a scan_key raise rather than silently clobber."""
    worker = WarmModelWorker(rice_source)
    reqs = [
        ScanRequest("dup", video, _params()),
        ScanRequest("dup", video, _params()),
    ]
    with pytest.raises(ValueError):
        predict_and_write_batch(worker, reqs, tmp_path)


# --- Task 7: downstream acceptance (sleap_roots.Series.load) -----------------


def test_output_loads_via_sleap_roots_series(rice_source, video, tmp_path):
    """The writer's output loads cleanly via the downstream sleap_roots.Series."""
    sleap_roots = pytest.importorskip("sleap_roots")
    worker = WarmModelWorker(rice_source)
    manifest = write_prediction_outputs(
        worker.predict(_params(), video),
        worker.resolve(_params()),
        tmp_path,
        scan_key="scan0731",
        inference_config=worker.inference_config(),
        output_params=worker.output_params(),
    )
    # Resolve each root type's .slp path (POSIX) from the manifest; no crown here.
    paths = {
        a.root_type: (tmp_path / a.slp_path).as_posix() for a in manifest.artifacts
    }
    series = sleap_roots.Series.load(
        series_name="scan0731",
        primary_path=paths.get("primary"),
        lateral_path=paths.get("lateral"),
        crown_path=None,
    )
    assert series.primary_labels is not None
    assert series.lateral_labels is not None
