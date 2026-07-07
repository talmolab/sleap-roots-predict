"""Real, no-mock tests for the warm-batch predict runner."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sleap_roots_predict.batch import BatchResult, discover_scans, run_batch


def _write_scan(root: Path, scan_key: str, params, *, stem=None, extra_files=()):
    """Create a scan dir with one PNG frame + a sidecar; return the dir."""
    d = root / scan_key
    d.mkdir(parents=True)
    Image.fromarray(np.zeros((16, 16), dtype="uint8")).save(d / "frame_000.png")
    stem = stem if stem is not None else scan_key
    body = {"scan_key": scan_key, "image_ids": ["a"], "images_checksum": "sha256:x"}
    if params is not None:
        body["params"] = params
    (d / f"{stem}.scan_metadata.json").write_text(json.dumps(body))
    for name in extra_files:
        (d / name).write_text("not an image")
    return d


def test_discover_scans_reads_sidecar_and_frames(scan_input_dir: Path):
    scans = discover_scans(scan_input_dir)
    assert len(scans) == 1
    scan = scans[0]
    assert scan.scan_key == "scanCPTEST0"
    assert scan.error is None
    assert len(scan.frames) == 8
    assert all(p.suffix.lower() == ".png" for p in scan.frames)
    assert scan.params.values == {"species": "rice", "mode": "cylinder", "age": 3}


def test_non_image_files_are_ignored(tmp_path: Path):
    _write_scan(
        tmp_path,
        "scanA",
        {"species": "rice", "mode": "cylinder", "age": 3},
        extra_files=("readme.txt",),
    )
    (scan,) = discover_scans(tmp_path)
    assert [p.name for p in scan.frames] == ["frame_000.png"]  # .txt + .json excluded


def test_stem_scan_key_mismatch_is_error(tmp_path: Path):
    d = tmp_path / "scanB"
    d.mkdir()
    Image.fromarray(np.zeros((16, 16), dtype="uint8")).save(d / "frame_000.png")
    # sidecar filename stem "scanB" but internal scan_key "scanOTHER"
    (d / "scanB.scan_metadata.json").write_text(
        json.dumps(
            {
                "scan_key": "scanOTHER",
                "params": {"species": "rice", "mode": "cylinder", "age": 3},
            }
        )
    )
    (scan,) = discover_scans(tmp_path)
    assert scan.error is not None and "scanOTHER" in scan.error


def test_missing_params_is_error(tmp_path: Path):
    _write_scan(tmp_path, "scanC", None)  # no params key
    (scan,) = discover_scans(tmp_path)
    assert scan.error is not None and "params" in scan.error


def test_duplicate_scan_key_raises(tmp_path: Path):
    _write_scan(
        tmp_path / "a", "dup", {"species": "rice", "mode": "cylinder", "age": 3}
    )
    _write_scan(
        tmp_path / "b", "dup", {"species": "rice", "mode": "cylinder", "age": 3}
    )
    with pytest.raises(ValueError, match="duplicate scan_key"):
        discover_scans(tmp_path)


def test_batch_does_not_import_trait_extractor():
    import sleap_roots_predict.batch  # noqa: F401

    assert "trait_extractor" not in sys.modules


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


def test_run_batch_predicts_every_scan(all_roots_source, tmp_path: Path):
    import shutil as _sh

    src_frames = sorted(Path("tests/assets/images/centered_pair").glob("*.png"))
    inp = tmp_path / "in"
    for key in ("scanX", "scanY"):
        d = inp / key
        d.mkdir(parents=True)
        for f in src_frames:
            _sh.copyfile(f, d / f.name)
        (d / f"{key}.scan_metadata.json").write_text(
            json.dumps(
                {
                    "scan_key": key,
                    "image_ids": ["a"],
                    "images_checksum": "sha256:x",
                    "params": {"species": "rice", "mode": "cylinder", "age": 3},
                }
            )
        )
    out = tmp_path / "out"
    result = run_batch(inp, out, source=all_roots_source)
    assert [s.status for s in result.scans] == ["ok", "ok"]
    for key in ("scanX", "scanY"):
        assert (out / key / f"{key}.predictions.json").exists()


def test_video_is_single_channel(scan_input_dir: Path):
    from sleap_roots_predict.video_utils import make_video_from_images

    (scan,) = discover_scans(scan_input_dir)
    video = make_video_from_images(scan.frames, greyscale=True)
    assert video.shape[-1] == 1  # 1-channel, matching in_channels:1 cylinder models


def test_rerun_skips_completed_scan(
    scan_input_dir: Path, all_roots_source, tmp_path: Path
):
    out = tmp_path / "out"
    run_batch(scan_input_dir, out, source=all_roots_source)
    manifest = out / "scanCPTEST0" / "scanCPTEST0.predictions.json"
    mtime = manifest.stat().st_mtime_ns

    result2 = run_batch(scan_input_dir, out, source=all_roots_source)
    assert [s.status for s in result2.scans] == ["skipped"]
    assert manifest.stat().st_mtime_ns == mtime  # not rewritten


_FRAMES = sorted(Path("tests/assets/images/centered_pair").glob("*.png"))
_RICE = {"species": "rice", "mode": "cylinder", "age": 3}


def _real_scan(root: Path, key: str, params):
    """Create a scan dir with the 8 vendored frames + a sidecar."""
    import shutil as _sh

    d = root / key
    d.mkdir(parents=True)
    for f in _FRAMES:
        _sh.copyfile(f, d / f.name)
    (d / f"{key}.scan_metadata.json").write_text(
        json.dumps(
            {
                "scan_key": key,
                "image_ids": ["a"],
                "images_checksum": "sha256:x",
                "params": params,
            }
        )
    )
    return d


def test_one_failing_scan_does_not_abort_batch(all_roots_source, tmp_path: Path):
    inp = tmp_path / "in"
    _real_scan(inp, "scanGOOD", _RICE)
    # a bad scan: sidecar present, NO frames -> per-scan failure
    bad = inp / "scanBAD"
    bad.mkdir()
    (bad / "scanBAD.scan_metadata.json").write_text(
        json.dumps(
            {
                "scan_key": "scanBAD",
                "image_ids": ["a"],
                "images_checksum": "sha256:x",
                "params": _RICE,
            }
        )
    )
    out = tmp_path / "out"
    result = run_batch(inp, out, source=all_roots_source)
    statuses = {s.scan_key: s.status for s in result.scans}
    assert statuses["scanGOOD"] == "ok"
    assert statuses["scanBAD"] == "failed"
    assert result.ok is False
    assert (out / "scanGOOD" / "scanGOOD.predictions.json").exists()


def test_zero_resolved_models_is_failed(rice_source, tmp_path: Path):
    # rice_source has no card for species "soybean" -> zero models resolve
    inp = tmp_path / "in"
    _real_scan(inp, "scanZ", {"species": "soybean", "mode": "cylinder", "age": 3})
    out = tmp_path / "out"
    result = run_batch(inp, out, source=rice_source)
    assert [s.status for s in result.scans] == ["failed"]
    assert not (out / "scanZ" / "scanZ.predictions.json").exists()


def test_empty_input_is_noop(tmp_path: Path):
    empty = tmp_path / "empty_in"
    empty.mkdir()
    result = run_batch(empty, tmp_path / "out")
    assert isinstance(result, BatchResult)
    assert result.ok and result.scans == []


def test_cli_main_exit_codes(scan_input_dir: Path, tmp_path: Path, monkeypatch):
    from sleap_roots_predict.__main__ import main

    class _Res:
        def __init__(self, ok):
            self.ok = ok
            self.scans = []

    state = {"ok": True}

    def fake_run_batch(inp, out):
        return _Res(state["ok"])

    monkeypatch.setattr("sleap_roots_predict.batch.run_batch", fake_run_batch)
    state["ok"] = True
    assert main([str(scan_input_dir), str(tmp_path / "o1")]) == 0
    state["ok"] = False
    assert main([str(scan_input_dir), str(tmp_path / "o2")]) == 1


@pytest.mark.wandb
def test_module_cli_over_registry(scan_input_dir: Path, tmp_path: Path):
    import subprocess

    out = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, "-m", "sleap_roots_predict", str(scan_input_dir), str(out)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert (out / "scanCPTEST0" / "scanCPTEST0.predictions.json").exists()
