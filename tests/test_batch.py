"""Real, no-mock tests for the warm-batch predict runner."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sleap_roots_predict.batch import discover_scans


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
    _write_scan(tmp_path / "a", "dup", {"species": "rice", "mode": "cylinder", "age": 3})
    _write_scan(tmp_path / "b", "dup", {"species": "rice", "mode": "cylinder", "age": 3})
    with pytest.raises(ValueError, match="duplicate scan_key"):
        discover_scans(tmp_path)


def test_batch_does_not_import_trait_extractor():
    import sleap_roots_predict.batch  # noqa: F401

    assert "trait_extractor" not in sys.modules
