"""Real, no-mock tests for the warm-batch predict runner."""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from manifest_builders import write_run_manifest
from sleap_roots_predict.batch import discover_scans, run_batch


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


def test_discover_scans_scopes_to_run_manifest(tmp_path: Path):
    _write_scan(tmp_path, "scan_1009", _RICE)
    _write_scan(tmp_path, "scan_1010", _RICE)  # leftover from a prior run, not in scope
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"pipeline_run_id": "run-1", "scan_keys": ["scan_1009"]})
    )
    scans = discover_scans(tmp_path)
    assert [s.scan_key for s in scans] == ["scan_1009"]


def test_excluded_out_of_scope_sidecar_logs_debug(tmp_path: Path, caplog):
    _write_scan(tmp_path, "scan_1009", _RICE)
    _write_scan(tmp_path, "scan_1010", _RICE)  # leftover from a prior run, not in scope
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"pipeline_run_id": "run-1", "scan_keys": ["scan_1009"]})
    )
    with caplog.at_level("DEBUG", logger="sleap_roots_predict.batch"):
        discover_scans(tmp_path)
    assert any("scan_1010" in r.message for r in caplog.records)


def test_no_exclusion_logs_no_debug_line(tmp_path: Path, caplog):
    _write_scan(tmp_path, "scan_1009", _RICE)
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"pipeline_run_id": "run-1", "scan_keys": ["scan_1009"]})
    )
    with caplog.at_level("DEBUG", logger="sleap_roots_predict.batch"):
        discover_scans(tmp_path)
    assert not any("excluded" in r.message.lower() for r in caplog.records)


def test_manifest_scoped_stem_mismatch_reports_the_real_error(tmp_path: Path):
    # sidecar filename stem "scanB" (matches the manifest's scan_key, so it passes
    # scoping) but its internal scan_key is "scanOTHER" — must surface the actual
    # stem-mismatch error, not a misleading "no sidecar found for manifest scan_key"
    # (which would fire if this sidecar were wrongly treated as never having been found).
    d = tmp_path / "scanB"
    d.mkdir()
    Image.fromarray(np.zeros((16, 16), dtype="uint8")).save(d / "frame_000.png")
    (d / "scanB.scan_metadata.json").write_text(
        json.dumps(
            {
                "scan_key": "scanOTHER",
                "params": {"species": "rice", "mode": "cylinder", "age": 3},
            }
        )
    )
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"pipeline_run_id": "run-1", "scan_keys": ["scanB"]})
    )
    (scan,) = discover_scans(tmp_path)
    assert scan.error is not None
    assert "scanOTHER" in scan.error
    assert "no sidecar found" not in scan.error


def test_no_manifest_falls_back_to_unscoped_discovery(tmp_path: Path):
    _write_scan(tmp_path, "scanA", _RICE)
    _write_scan(tmp_path, "scanB", _RICE)
    scans = discover_scans(tmp_path)
    assert sorted(s.scan_key for s in scans) == ["scanA", "scanB"]


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

    src_frames = sorted(
        (Path(__file__).parent / "assets/images/centered_pair").glob("*.png")
    )
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


_FRAMES = sorted((Path(__file__).parent / "assets/images/centered_pair").glob("*.png"))
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


def test_empty_input_raises(tmp_path: Path):
    empty = tmp_path / "empty_in"
    empty.mkdir()
    with pytest.raises(ValueError, match="no scans discovered"):
        run_batch(empty, tmp_path / "out")


def _recording_source():
    """A model source that records access; returns ``(source, calls)``.

    ``list_cards()`` yields one card no scan can match (an unmodelled species), so every
    scan resolves to zero models and fails at ``_predict_one``'s ``if not refs:`` guard --
    *before* ``out_scan_dir.mkdir()``, so a batch using this source writes nothing at all.
    (Not an *empty* catalog: that is a batch-level error of its own.) That makes it the
    cheap stand-in for tests about the forward-copy, which must hold independently of
    prediction, and it doubles as a probe for "no model-source interaction happened".
    """
    from card_builders import make_card

    calls = {"n": 0}
    unmatched = make_card("primary", "reg/unmodelled", species="no-such-species")

    class _RecordingSource:
        def list_cards(self):
            calls["n"] += 1
            return [unmatched]

        def materialize(self, ref):
            calls["n"] += 1
            raise AssertionError("materialize should never be called")

    return _RecordingSource(), calls


def test_empty_input_raises_before_worker_interaction(tmp_path: Path):
    source, calls = _recording_source()
    empty = tmp_path / "empty_in"
    empty.mkdir()
    with pytest.raises(ValueError):
        run_batch(empty, tmp_path / "out", source=source)
    assert calls["n"] == 0


def test_cli_main_exit_codes(scan_input_dir: Path, tmp_path: Path, monkeypatch):
    from sleap_roots_predict.__main__ import main

    class _Res:
        def __init__(self, ok):
            self.ok = ok
            self.scans = []

    state = {"ok": True}

    def fake_run_batch(inp, out, **kwargs):
        return _Res(state["ok"])

    monkeypatch.setattr("sleap_roots_predict.batch.run_batch", fake_run_batch)
    state["ok"] = True
    assert main([str(scan_input_dir), str(tmp_path / "o1")]) == 0
    state["ok"] = False
    assert main([str(scan_input_dir), str(tmp_path / "o2")]) == 3


def test_cli_usage_error_exits_2_via_argparse():
    from sleap_roots_predict.__main__ import main

    with pytest.raises(SystemExit) as exc_info:
        main([])  # missing both required positional arguments
    assert exc_info.value.code == 2


def test_install_sigterm_handler_sets_event_when_invoked():
    import signal

    from sleap_roots_predict.__main__ import _install_sigterm_handler

    prev_handler = signal.getsignal(signal.SIGTERM)
    try:
        event = _install_sigterm_handler()
        assert not event.is_set()
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        assert event.is_set()
    finally:
        signal.signal(signal.SIGTERM, prev_handler)


def test_sigterm_overrides_success_exit_code(
    scan_input_dir: Path, all_roots_source, tmp_path: Path, monkeypatch
):
    import signal

    import sleap_roots_predict.batch as batch_mod
    from sleap_roots_predict.__main__ import main

    prev_handler = signal.getsignal(signal.SIGTERM)
    try:
        real_run_batch = batch_mod.run_batch

        def _run_then_signal(*args, **kwargs):
            # Delegating spy: run the real batch, then trigger the SIGTERM handler
            # main() already installed, strictly between run_batch returning and
            # main()'s post-run_batch check -- mirrors this file's existing
            # spy_resolve/_Counting(orig) wrap-and-delegate pattern. `source` is
            # injected so this hits the offline test fixture, not the live registry.
            kwargs.setdefault("source", all_roots_source)
            result = real_run_batch(*args, **kwargs)
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            return result

        monkeypatch.setattr(batch_mod, "run_batch", _run_then_signal)
        assert main([str(scan_input_dir), str(tmp_path / "out")]) == 143
    finally:
        signal.signal(signal.SIGTERM, prev_handler)


def test_sigterm_overrides_partial_exit_code(
    all_roots_source, tmp_path: Path, monkeypatch
):
    import signal

    import sleap_roots_predict.batch as batch_mod
    from sleap_roots_predict.__main__ import main

    inp = tmp_path / "in"
    _real_scan(inp, "scanGOOD", _RICE)
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

    prev_handler = signal.getsignal(signal.SIGTERM)
    try:
        real_run_batch = batch_mod.run_batch

        def _run_then_signal(*args, **kwargs):
            kwargs.setdefault("source", all_roots_source)
            result = real_run_batch(*args, **kwargs)
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            return result

        monkeypatch.setattr(batch_mod, "run_batch", _run_then_signal)
        assert main([str(inp), str(tmp_path / "out")]) == 143
    finally:
        signal.signal(signal.SIGTERM, prev_handler)


def test_main_restores_prior_sigterm_handler_on_normal_completion(
    scan_input_dir: Path, tmp_path: Path, monkeypatch
):
    # The staging-error-path restoration test only exercises main()'s except/raise
    # branch; this pins the same guarantee on the plain success-return path, so the
    # unconditional `finally` around the whole function body is tested on both
    # sides, not just the one that happens to also need the log-then-reraise fix.
    import signal

    from sleap_roots_predict.__main__ import main

    prev_handler = signal.getsignal(signal.SIGTERM)
    monkeypatch.setattr(
        "sleap_roots_predict.batch.run_batch",
        lambda *a, **k: type("R", (), {"ok": True, "scans": []})(),
    )
    assert main([str(scan_input_dir), str(tmp_path / "out")]) == 0
    assert signal.getsignal(signal.SIGTERM) is prev_handler


def test_sigterm_composes_with_should_stop_across_a_real_multi_scan_batch(
    all_roots_source, tmp_path: Path, monkeypatch
):
    # The boundary-stop test (test_should_stop_stops_after_first_scan) and the
    # SIGTERM-override tests above each exercise their own half in isolation --
    # this wires a real SIGTERM delivery (via the actual registered handler,
    # never os.kill) into should_stop's real per-scan-loop check over a real
    # two-scan batch, so the full composition is exercised end to end at least
    # once, not just its two halves independently.
    import signal

    import sleap_roots_predict.batch as batch_mod
    from sleap_roots_predict.__main__ import main

    inp = tmp_path / "in"
    _real_scan(inp, "s1", _RICE)
    _real_scan(inp, "s2", _RICE)
    out = tmp_path / "out"

    prev_handler = signal.getsignal(signal.SIGTERM)
    try:
        real_run_batch = batch_mod.run_batch
        calls = {"n": 0}

        def _run_with_signal_after_first_scan(*args, **kwargs):
            orig_should_stop = kwargs.get("should_stop", lambda: False)

            def _wrapped_should_stop():
                calls["n"] += 1
                if calls["n"] == 2:
                    # Fire the real handler main() already installed -- this sets
                    # the real threading.Event backing orig_should_stop, so the
                    # very next line observes it exactly as a genuine delivery
                    # between scans would.
                    signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
                return orig_should_stop()

            kwargs["source"] = all_roots_source
            kwargs["should_stop"] = _wrapped_should_stop
            return real_run_batch(*args, **kwargs)

        monkeypatch.setattr(batch_mod, "run_batch", _run_with_signal_after_first_scan)
        assert main([str(inp), str(out)]) == 143
    finally:
        signal.signal(signal.SIGTERM, prev_handler)

    assert (out / "s1" / "s1.predictions.json").exists()
    assert not (out / "s2").exists()


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


def test_run_batch_constructs_single_worker(all_roots_source, tmp_path, monkeypatch):
    import sleap_roots_predict.batch as batch_mod

    counter = {"n": 0}
    orig = batch_mod.WarmModelWorker

    class _Counting(orig):
        def __init__(self, *a, **k):
            counter["n"] += 1
            super().__init__(*a, **k)

    monkeypatch.setattr(batch_mod, "WarmModelWorker", _Counting)
    inp = tmp_path / "in"
    _real_scan(inp, "s1", _RICE)
    _real_scan(inp, "s2", _RICE)
    result = run_batch(inp, tmp_path / "out", source=all_roots_source)
    assert [s.status for s in result.scans] == ["ok", "ok"]
    assert counter["n"] == 1  # one resident worker for the whole batch


def test_should_stop_stops_after_first_scan(all_roots_source, tmp_path: Path):
    inp = tmp_path / "in"
    _real_scan(inp, "s1", _RICE)
    _real_scan(inp, "s2", _RICE)
    out = tmp_path / "out"
    calls = {"n": 0}

    def _stop():
        calls["n"] += 1
        return calls["n"] > 1  # False the first time (before s1), True thereafter

    result = run_batch(inp, out, source=all_roots_source, should_stop=_stop)
    assert [s.scan_key for s in result.scans] == ["s1"]
    assert result.scans[0].status == "ok"
    assert (out / "s1" / "s1.predictions.json").exists()
    assert list((out / "s1").glob("*.slp"))
    assert not (out / "s2").exists()


def test_should_stop_default_is_unaffected(all_roots_source, tmp_path: Path):
    inp = tmp_path / "in"
    _real_scan(inp, "s1", _RICE)
    _real_scan(inp, "s2", _RICE)
    result = run_batch(inp, tmp_path / "out", source=all_roots_source)
    assert [s.status for s in result.scans] == ["ok", "ok"]


def test_missing_input_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_batch(tmp_path / "does_not_exist", tmp_path / "out")


def test_cli_missing_input_dir_propagates_as_default_exit_1(tmp_path, caplog):
    from sleap_roots_predict.__main__ import main

    # discover_scans raises FileNotFoundError before the worker is built (no
    # network); main logs a clean message then re-raises, so the process exits via
    # Python's default unhandled-exception code (1), not a special-cased return.
    with caplog.at_level("ERROR"):
        with pytest.raises(FileNotFoundError):
            main([str(tmp_path / "nope"), str(tmp_path / "out")])
    assert any("Batch aborted" in r.message for r in caplog.records)


def test_cli_empty_input_propagates_as_default_exit_1(tmp_path):
    from sleap_roots_predict.__main__ import main

    empty = tmp_path / "empty_in"
    empty.mkdir()
    with pytest.raises(ValueError, match="no scans discovered"):
        main([str(empty), str(tmp_path / "out")])


def test_main_restores_prior_sigterm_handler_on_staging_error(tmp_path):
    # main() installs its own SIGTERM handler before running the batch; a staging
    # error (re-raised, not returned) must not leave that handler installed --
    # otherwise a real SIGTERM to whatever process later calls main() again (or
    # to the pytest process itself) is silently swallowed by an orphaned handler
    # closing over a dead threading.Event nobody reads.
    import signal

    from sleap_roots_predict.__main__ import main

    prev_handler = signal.getsignal(signal.SIGTERM)
    with pytest.raises(FileNotFoundError):
        main([str(tmp_path / "nope"), str(tmp_path / "out")])
    assert signal.getsignal(signal.SIGTERM) is prev_handler


def test_sidecar_copy_failure_leaves_no_manifest(
    scan_input_dir: Path, all_roots_source, tmp_path: Path, monkeypatch
):
    import sleap_roots_predict.batch as batch_mod

    def _boom(src, dst):
        raise OSError("disk full")

    # path-safe: only the sidecar copy uses shutil.copyfile (the run-manifest forward
    # writes through the fd mkstemp opened)
    monkeypatch.setattr(batch_mod.shutil, "copyfile", _boom)
    out = tmp_path / "out"
    result = run_batch(scan_input_dir, out, source=all_roots_source)
    assert [s.status for s in result.scans] == ["failed"]
    # sidecar is copied BEFORE the manifest, so a copy failure leaves no manifest ->
    # resume re-runs the scan rather than skipping an incomplete tree.
    assert not (out / "scanCPTEST0" / "scanCPTEST0.predictions.json").exists()


def _fail_replace_for_sidecars(monkeypatch):
    """Record + raise only for the sidecar's destination; everything else is real.

    Path-conditional so the run-manifest forward-copy, which also calls os.replace,
    can never be what these tests intercept.
    """
    import os as _os

    real_replace = _os.replace
    seen = []

    def _replace(src, dst, *a, **k):
        if str(dst).endswith(".scan_metadata.json"):
            seen.append(str(src))
            raise OSError("simulated interruption")
        return real_replace(src, dst, *a, **k)

    monkeypatch.setattr("os.replace", _replace)
    return seen


def test_sidecar_copy_leaves_no_partial_file_if_replace_fails(
    scan_input_dir: Path, all_roots_source, tmp_path: Path, monkeypatch
):
    _fail_replace_for_sidecars(monkeypatch)
    out = tmp_path / "out"
    result = run_batch(scan_input_dir, out, source=all_roots_source)
    assert [s.status for s in result.scans] == ["failed"]
    scan_dir = out / "scanCPTEST0"
    assert not (scan_dir / "scanCPTEST0.scan_metadata.json").exists()
    assert not (scan_dir / "scanCPTEST0.predictions.json").exists()
    assert not [p.name for p in scan_dir.iterdir() if p.name.endswith(".tmp")]


def test_concurrent_sidecar_copies_use_private_temp_files(
    scan_input_dir: Path, all_roots_source, tmp_path: Path, monkeypatch
):
    """Two writers of one scan must never share a sidecar temp path (predict#43)."""
    seen = _fail_replace_for_sidecars(monkeypatch)
    out = tmp_path / "out"
    for _ in range(2):
        run_batch(scan_input_dir, out, source=all_roots_source)
    assert len(seen) == 2 and seen[0] != seen[1]
    scan_dir = out / "scanCPTEST0"
    assert not [p.name for p in scan_dir.iterdir() if p.name.endswith(".tmp")]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
def test_copied_sidecar_keeps_a_direct_writes_permissions(
    scan_input_dir: Path, all_roots_source, tmp_path: Path
):
    """Guard (green before and after #43): a private 0600 temp mode must never leak."""
    import os as _os

    old = _os.umask(0o022)
    try:
        out = tmp_path / "out"
        run_batch(scan_input_dir, out, source=all_roots_source)
        control = out / "control"
        control.write_bytes(b"x")
        sidecar = out / "scanCPTEST0" / "scanCPTEST0.scan_metadata.json"
        assert sidecar.stat().st_mode & 0o777 == control.stat().st_mode & 0o777
    finally:
        _os.umask(old)


def test_unreadable_json_sidecar_is_error(tmp_path: Path):
    d = tmp_path / "scanBad"
    d.mkdir()
    (d / "scanBad.scan_metadata.json").write_text("{not valid json")
    (scan,) = discover_scans(tmp_path)
    assert scan.error is not None and "unreadable" in scan.error


def test_uppercase_extension_frames_collected(tmp_path: Path):
    d = tmp_path / "scanU"
    d.mkdir()
    Image.fromarray(np.zeros((16, 16), dtype="uint8")).save(
        d / "FRAME_000.PNG", format="PNG"
    )
    (d / "scanU.scan_metadata.json").write_text(
        json.dumps({"scan_key": "scanU", "params": _RICE})
    )
    (scan,) = discover_scans(tmp_path)
    assert [p.name for p in scan.frames] == ["FRAME_000.PNG"]  # case-folded match


def test_resume_mixed_skip_and_predict(all_roots_source, tmp_path: Path):
    inp = tmp_path / "in"
    _real_scan(inp, "sDone", _RICE)
    out = tmp_path / "out"
    run_batch(inp, out, source=all_roots_source)  # sDone predicted
    _real_scan(inp, "sNew", _RICE)  # add a second, not-yet-done scan
    result = run_batch(inp, out, source=all_roots_source)
    statuses = {s.scan_key: s.status for s in result.scans}
    assert statuses["sDone"] == "skipped"
    assert statuses["sNew"] == "ok"


def test_manifest_scan_key_with_no_sidecar_is_failed(all_roots_source, tmp_path: Path):
    inp = tmp_path / "in"
    _real_scan(inp, "scanGOOD", _RICE)
    (inp / "run_manifest.json").write_text(
        json.dumps(
            {"pipeline_run_id": "run-1", "scan_keys": ["scanGOOD", "scanMISSING"]}
        )
    )
    out = tmp_path / "out"
    result = run_batch(inp, out, source=all_roots_source)
    statuses = {s.scan_key: s.status for s in result.scans}
    assert statuses["scanGOOD"] == "ok"
    assert statuses["scanMISSING"] == "failed"


def test_malformed_manifest_json_raises(tmp_path: Path):
    _write_scan(tmp_path, "scanA", _RICE)
    (tmp_path / "run_manifest.json").write_text("{not valid json")
    with pytest.raises(Exception):
        discover_scans(tmp_path)


def test_manifest_with_empty_scan_keys_raises(tmp_path: Path):
    _write_scan(tmp_path, "scanA", _RICE)
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"pipeline_run_id": "run-1", "scan_keys": []})
    )
    with pytest.raises(Exception):
        discover_scans(tmp_path)


def test_extra_params_keys_ignored(tmp_path: Path):
    d = tmp_path / "scanE"
    d.mkdir()
    (d / "scanE.scan_metadata.json").write_text(
        json.dumps(
            {
                "scan_key": "scanE",
                "params": {
                    "species": "rice",
                    "mode": "cylinder",
                    "age": 3,
                    "extra": "ignored",
                },
            }
        )
    )
    (scan,) = discover_scans(tmp_path)
    assert scan.error is None
    assert scan.params.values == {"species": "rice", "mode": "cylinder", "age": 3}


def test_identity_key_changes_with_images_checksum():
    import sleap_roots_predict.batch as batch_mod
    from sleap_roots_contracts import ModelRef

    ref = ModelRef(registry_id="reg/x", version="v1", sleap_nn_version="0.3.0")
    base_kwargs = dict(
        scan_key="scanA",
        params_dict={"species": "rice", "mode": "cylinder", "age": 3},
        model_refs={"primary": ref},
        predict_code_sha="sha1",
        predict_output_params={"peak_threshold": 0.2},
    )
    key_a = batch_mod._identity_key(images_checksum="sha256:a", **base_kwargs)
    key_b = batch_mod._identity_key(images_checksum="sha256:b", **base_kwargs)
    assert key_a != key_b


def test_previous_identity_key_none_when_nothing_on_disk(tmp_path: Path):
    import sleap_roots_predict.batch as batch_mod

    assert batch_mod._previous_identity_key(tmp_path, "scanA") is None


def test_previous_identity_key_none_when_predictions_json_corrupt(tmp_path: Path):
    import sleap_roots_predict.batch as batch_mod

    (tmp_path / "scanA.scan_metadata.json").write_text(
        json.dumps(
            {"scan_key": "scanA", "images_checksum": "sha256:x", "params": _RICE}
        )
    )
    (tmp_path / "scanA.predictions.json").write_text('{"not": "a valid manifest"}')
    assert batch_mod._previous_identity_key(tmp_path, "scanA") is None


def test_changed_params_causes_repredict(all_roots_source, tmp_path: Path):
    inp = tmp_path / "in"
    _real_scan(inp, "scanA", _RICE)
    out = tmp_path / "out"
    run_batch(inp, out, source=all_roots_source)
    manifest = out / "scanA" / "scanA.predictions.json"
    mtime1 = manifest.stat().st_mtime_ns

    sidecar = inp / "scanA" / "scanA.scan_metadata.json"
    body = json.loads(sidecar.read_text())
    body["params"] = {"species": "rice", "mode": "cylinder", "age": 4}
    sidecar.write_text(json.dumps(body))

    result2 = run_batch(inp, out, source=all_roots_source)
    assert [s.status for s in result2.scans] == ["ok"]
    assert manifest.stat().st_mtime_ns != mtime1


def test_changed_images_checksum_causes_repredict(all_roots_source, tmp_path: Path):
    inp = tmp_path / "in"
    _real_scan(inp, "scanA", _RICE)
    out = tmp_path / "out"
    run_batch(inp, out, source=all_roots_source)
    manifest = out / "scanA" / "scanA.predictions.json"
    mtime1 = manifest.stat().st_mtime_ns

    sidecar = inp / "scanA" / "scanA.scan_metadata.json"
    body = json.loads(sidecar.read_text())
    body["images_checksum"] = "sha256:changed"
    sidecar.write_text(json.dumps(body))

    result2 = run_batch(inp, out, source=all_roots_source)
    assert [s.status for s in result2.scans] == ["ok"]
    assert manifest.stat().st_mtime_ns != mtime1


def test_changed_predict_code_sha_causes_repredict(
    all_roots_source, tmp_path, monkeypatch
):
    inp = tmp_path / "in"
    _real_scan(inp, "scanA", _RICE)
    out = tmp_path / "out"
    monkeypatch.setenv("SRP_PREDICT_CODE_SHA", "sha-one")
    run_batch(inp, out, source=all_roots_source)
    manifest = out / "scanA" / "scanA.predictions.json"
    mtime1 = manifest.stat().st_mtime_ns

    monkeypatch.setenv("SRP_PREDICT_CODE_SHA", "sha-two")
    result2 = run_batch(inp, out, source=all_roots_source)
    assert [s.status for s in result2.scans] == ["ok"]
    assert manifest.stat().st_mtime_ns != mtime1
    assert json.loads(manifest.read_text())["predict_code_sha"] == "sha-two"


def test_changed_model_ref_causes_repredict(tmp_path: Path, native_model_dir):
    from card_builders import make_card
    from sleap_roots_predict.model_registry import LocalCardSource

    def _source(version):
        card = make_card("primary", "reg/rice-primary", version=version)
        return LocalCardSource([(card, native_model_dir)])

    inp = tmp_path / "in"
    _real_scan(inp, "scanA", _RICE)
    out = tmp_path / "out"
    run_batch(inp, out, source=_source("v1"))
    manifest = out / "scanA" / "scanA.predictions.json"
    mtime1 = manifest.stat().st_mtime_ns

    result2 = run_batch(inp, out, source=_source("v2"))
    assert [s.status for s in result2.scans] == ["ok"]
    assert manifest.stat().st_mtime_ns != mtime1


def test_corrupt_previous_manifest_causes_repredict_not_failure(
    all_roots_source, tmp_path
):
    inp = tmp_path / "in"
    _real_scan(inp, "scanA", _RICE)
    out = tmp_path / "out"
    run_batch(inp, out, source=all_roots_source)
    (out / "scanA" / "scanA.predictions.json").write_text('{"not": "a valid manifest"}')

    result2 = run_batch(inp, out, source=all_roots_source)
    assert [s.status for s in result2.scans] == ["ok"]


def test_scan_error_short_circuits_before_resolve(
    all_roots_source, tmp_path, monkeypatch
):
    import sleap_roots_predict.batch as batch_mod

    inp = tmp_path / "in"
    _real_scan(inp, "scanGOOD", _RICE)
    (inp / "run_manifest.json").write_text(
        json.dumps(
            {"pipeline_run_id": "run-1", "scan_keys": ["scanGOOD", "scanMISSING"]}
        )
    )
    out = tmp_path / "out"

    calls = []
    original_resolve = batch_mod.WarmModelWorker.resolve

    def spy_resolve(self, params, overrides=None):
        calls.append(params)
        return original_resolve(self, params, overrides)

    monkeypatch.setattr(batch_mod.WarmModelWorker, "resolve", spy_resolve)
    result = run_batch(inp, out, source=all_roots_source)
    statuses = {s.scan_key: s.status for s in result.scans}
    assert statuses == {"scanGOOD": "ok", "scanMISSING": "failed"}
    # resolve() runs exactly twice for scanGOOD (once for the identity-key comparison
    # in run_batch, once more inside worker.predict() -> get_predictors()) and never
    # for scanMISSING (whose ScanInput.params is None) — pinning down both the "never
    # called for an error'd scan" invariant and the documented call count together.
    assert len(calls) == 2
    assert all(params is not None for params in calls)


# --- run-manifest forward-copy (predict #39) ---------------------------------------

_MANIFEST = "run_manifest.json"
# Bytes a RunManifest round-trip would not reproduce: non-canonical key order, an
# undeclared extra field (dropped by re-serialization), interior whitespace, and no
# trailing newline.
_NON_CANONICAL = b'{"scan_keys": ["scanA"],  "pipeline_run_id": "run-1", "extra": 1}'


def _stage_manifest(root: Path, body: bytes) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / _MANIFEST
    path.write_bytes(body)
    return path


def test_run_batch_forwards_the_manifest_byte_identically(tmp_path: Path):
    inp = tmp_path / "in"
    _write_scan(inp, "scanA", _RICE)
    src = _stage_manifest(inp, _NON_CANONICAL)
    out = tmp_path / "out"
    source, _ = _recording_source()

    run_batch(inp, out, source=source)

    assert (out / _MANIFEST).read_bytes() == src.read_bytes()
    # forwarded to the TOP level, never into a per-scan subdirectory
    assert not (out / "scanA" / _MANIFEST).exists()


def test_run_batch_writes_no_manifest_when_none_is_staged(
    scan_input_dir: Path, all_roots_source, tmp_path: Path
):
    out = tmp_path / "out"
    result = run_batch(scan_input_dir, out, source=all_roots_source)
    assert not (out / _MANIFEST).exists()
    # the batch is otherwise unaffected: outputs still written as usual
    assert [s.status for s in result.scans] == ["ok"]
    assert (out / "scanCPTEST0" / "scanCPTEST0.predictions.json").is_file()


def test_run_batch_forwards_the_manifest_even_when_stopped_immediately(tmp_path: Path):
    """Pins the placement decision: a copy after the loop would be skipped exactly
    when a preempted partial batch hands off to the downstream stage."""
    inp = tmp_path / "in"
    _write_scan(inp, "scanA", _RICE)
    src = _stage_manifest(inp, _NON_CANONICAL)
    out = tmp_path / "out"
    source, _ = _recording_source()

    result = run_batch(inp, out, source=source, should_stop=lambda: True)

    assert result.scans == []  # nothing predicted at all
    assert (out / _MANIFEST).read_bytes() == src.read_bytes()


def test_run_batch_forwards_the_manifest_when_every_scan_fails(tmp_path: Path):
    """The forward-copy is then the only thing that creates output_dir."""
    inp = tmp_path / "in"
    inp.mkdir()
    src = _stage_manifest(inp, b'{"pipeline_run_id":"r","scan_keys":["scanMISSING"]}')
    out = tmp_path / "out"
    source, _ = _recording_source()

    result = run_batch(inp, out, source=source)

    assert [s.status for s in result.scans] == ["failed"]
    assert {p.name for p in out.iterdir()} == {_MANIFEST}
    assert (out / _MANIFEST).read_bytes() == src.read_bytes()


def test_run_batch_copy_failure_raises_before_any_prediction(
    tmp_path: Path, monkeypatch
):
    import sleap_roots_predict.batch as batch_mod

    inp = tmp_path / "in"
    _write_scan(inp, "scanA", _RICE)
    _stage_manifest(inp, _NON_CANONICAL)
    out = tmp_path / "out"
    source, calls = _recording_source()

    def _boom(_input_dir, _output_dir, **_kwargs):
        raise PermissionError("read-only mount")

    monkeypatch.setattr(batch_mod, "copy_run_manifest_forward", _boom)
    with pytest.raises(PermissionError):
        run_batch(inp, out, source=source)

    assert calls["n"] == 0  # no model-source interaction
    assert not out.exists()  # no per-scan output directory either


_SCOPED = b'{"scan_keys": ["scanA"], "pipeline_run_id": "run-1"}'


def _run_batch_with_upstream_rewrite(tmp_path: Path, monkeypatch, rewrite):
    """Run a batch where the upstream writer mutates the input manifest *after*
    discovery has read and validated it, but before the forward-copy runs.

    Patching `discover_scans` is how the window is opened deterministically: the real
    race is a concurrent bloomctl unioning into the shared input manifest, which no
    test can schedule reliably.
    """
    import sleap_roots_predict.batch as batch_mod

    inp = tmp_path / "in"
    _write_scan(inp, "scanA", _RICE)
    _stage_manifest(inp, _SCOPED)
    out = tmp_path / "out"
    source, _ = _recording_source()
    real_discover = batch_mod.discover_scans

    def _discover_then_rewrite(*args, **kwargs):
        scans = real_discover(*args, **kwargs)
        rewrite(inp / _MANIFEST)
        return scans

    monkeypatch.setattr(batch_mod, "discover_scans", _discover_then_rewrite)
    run_batch(inp, out, source=source)
    return out


def test_run_batch_forwards_what_discovery_validated_not_a_later_rewrite(
    tmp_path: Path, monkeypatch
):
    """The manifest was read twice with no shared snapshot, so an upstream writer
    unioning a new `scan_key` in between made the forwarded file describe a *wider*
    scope than predict actually predicted. Trait-extraction then reports a spurious
    `result.failed` for the extra key, exits 3, and burns retries under
    `retryPolicy: Always`, blocking write-back for scans that genuinely succeeded.
    """
    out = _run_batch_with_upstream_rewrite(
        tmp_path,
        monkeypatch,
        lambda path: path.write_bytes(
            b'{"scan_keys": ["scanA", "scanLATER"], "pipeline_run_id": "run-1"}'
        ),
    )
    assert (out / _MANIFEST).read_bytes() == _SCOPED


def test_run_batch_forwards_the_validated_manifest_even_if_the_source_vanishes(
    tmp_path: Path, monkeypatch
):
    """The other direction of the same race, and the worse one. If the source is gone
    by the time the copy re-reads it, the absent branch makes the forward a silent
    no-op -- predict scopes its own work to a manifest, exits 0, and forwards nothing,
    so the downstream stage falls back to unscoped discovery. That is #39 recurring,
    with a green step and nothing logged above DEBUG.
    """
    out = _run_batch_with_upstream_rewrite(
        tmp_path, monkeypatch, lambda path: path.unlink()
    )
    assert (out / _MANIFEST).read_bytes() == _SCOPED


def test_run_batch_same_input_and_output_leaves_the_manifest_intact(tmp_path: Path):
    inp = tmp_path / "in"
    _write_scan(inp, "scanA", _RICE)
    src = _stage_manifest(inp, _NON_CANONICAL)
    before = src.read_bytes()
    contents_before = {p.name for p in inp.iterdir()}
    # Bytes and directory contents are both unchanged by a copy that round-tripped the
    # file over itself, so they hold even with no identity check at all. The mtime is
    # the discriminator -- see test_run_manifest.py's hard-link test for the unit-level
    # version of this guarantee.
    before_mtime = src.stat().st_mtime_ns
    source, _ = _recording_source()

    run_batch(inp, inp, source=source)

    assert src.read_bytes() == before
    assert {p.name for p in inp.iterdir()} == contents_before
    assert src.stat().st_mtime_ns == before_mtime


def test_run_batch_forward_leaves_neighbouring_scan_outputs_untouched(tmp_path: Path):
    inp = tmp_path / "in"
    _write_scan(inp, "scanA", _RICE)
    _stage_manifest(inp, _NON_CANONICAL)
    out = tmp_path / "out"
    neighbour = out / "scanOTHER"
    neighbour.mkdir(parents=True)
    prior = neighbour / "scanOTHER.predictions.json"
    prior.write_bytes(b'{"prior":"run"}')
    prior_mtime = prior.stat().st_mtime_ns
    source, _ = _recording_source()

    run_batch(inp, out, source=source)

    assert {p.name for p in out.iterdir()} == {"scanOTHER", _MANIFEST}
    assert prior.read_bytes() == b'{"prior":"run"}'
    assert prior.stat().st_mtime_ns == prior_mtime


def test_run_batch_warns_and_keeps_a_stale_output_manifest(tmp_path: Path, caplog):
    """No input manifest, but output_dir holds one from an earlier run: it is left in
    place (a concurrent invocation may have written it) and the condition is logged."""
    inp = tmp_path / "in"
    _write_scan(inp, "scanA", _RICE)
    stale = _stage_manifest(tmp_path / "out", b'{"pipeline_run_id":"old","x":0}')
    before = stale.read_bytes()
    source, _ = _recording_source()

    with caplog.at_level(logging.WARNING, logger="sleap_roots_predict.run_manifest"):
        run_batch(inp, tmp_path / "out", source=source)

    assert stale.read_bytes() == before
    assert [r for r in caplog.records if r.levelno == logging.WARNING]


def test_cli_forward_copy_failure_propagates_as_default_exit_1(
    scan_input_dir: Path, tmp_path: Path, caplog, monkeypatch
):
    """The copy raises PermissionError -- a *sibling* of FileNotFoundError under
    OSError, not a subclass -- so without the widened catch this would bypass the
    mandated staging-error line and surface a bare traceback. `main()` re-raises, so
    exit 1 is the interpreter's default and is not observable in-process."""
    import sleap_roots_predict.batch as batch_mod
    from sleap_roots_predict.__main__ import main

    def _boom(_input_dir, _output_dir, **_kwargs):
        raise PermissionError("read-only mount")

    monkeypatch.setattr(batch_mod, "copy_run_manifest_forward", _boom)
    with caplog.at_level("ERROR"):
        with pytest.raises(PermissionError):
            main([str(scan_input_dir), str(tmp_path / "out")])
    assert any("Batch aborted" in r.message for r in caplog.records)


def test_forward_copy_failure_during_requested_stop_propagates(
    scan_input_dir: Path, tmp_path: Path, caplog, monkeypatch
):
    """A pre-flight staging error is not converted to the stop code: it raises before
    main()'s stop_event check is reached, so 143 never wins over it.

    The spy asserts the precondition it exists to establish, and the absence of the
    stop warning is asserted rather than implied. Without both, the test passes with
    the signal never fired -- PermissionError propagates whether or not a stop was
    requested -- which makes it a duplicate of the test above and leaves this
    scenario uncovered. The precondition is genuinely fragile: it bites only because
    __main__ imports run_batch *inside* main(), so hoisting that import to module
    scope would silently neuter this test.
    """
    import signal

    import sleap_roots_predict.batch as batch_mod
    from sleap_roots_predict.__main__ import main

    def _boom(_input_dir, _output_dir, **_kwargs):
        raise PermissionError("read-only mount")

    monkeypatch.setattr(batch_mod, "copy_run_manifest_forward", _boom)
    real_run_batch = batch_mod.run_batch

    def _fire_sigterm_then_run(*args, **kwargs):
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        assert kwargs["should_stop"](), "stop was not requested before run_batch ran"
        return real_run_batch(*args, **kwargs)

    monkeypatch.setattr(batch_mod, "run_batch", _fire_sigterm_then_run)
    prev_handler = signal.getsignal(signal.SIGTERM)
    try:
        with caplog.at_level(logging.WARNING):
            with pytest.raises(PermissionError):
                main([str(scan_input_dir), str(tmp_path / "out")])
    finally:
        signal.signal(signal.SIGTERM, prev_handler)
    assert not any("Terminated by SIGTERM" in r.message for r in caplog.records)


@pytest.mark.parametrize(
    "body",
    [b"{not valid json", b'{"pipeline_run_id":"r","scan_keys":[]}'],
    ids=["invalid-json", "fails-model-validation"],
)
def test_run_batch_never_forwards_a_manifest_that_fails_validation(
    tmp_path: Path, body: bytes
):
    """Pins Decision 2's strongest argument: discovery validates before the copy runs,
    so a corrupt manifest is structurally unforwardable."""
    inp = tmp_path / "in"
    _write_scan(inp, "scanA", _RICE)
    _stage_manifest(inp, body)
    out = tmp_path / "out"
    source, _ = _recording_source()

    with pytest.raises(ValueError):
        run_batch(inp, out, source=source)

    assert not out.exists()


# --- batch-level catalog load (registry guard + load_catalog) ----------------


def _counting(inner):
    calls = {"list": 0, "order": []}

    class _Counting:
        def list_cards(self):
            calls["list"] += 1
            calls["order"].append("list")
            return inner.list_cards()

        def materialize(self, ref):
            return inner.materialize(ref)

    return _Counting(), calls


def test_catalog_loaded_once_before_the_first_resolve(
    all_roots_source, tmp_path, monkeypatch
):
    from sleap_roots_predict import warm_worker as ww

    inp = tmp_path / "in"
    _real_scan(inp, "scanA", _RICE)
    _real_scan(inp, "scanB", _RICE)
    source, calls = _counting(all_roots_source)
    real_resolve = ww.WarmModelWorker.resolve

    def spy_resolve(self, *a, **k):
        calls["order"].append("resolve")
        return real_resolve(self, *a, **k)

    monkeypatch.setattr(ww.WarmModelWorker, "resolve", spy_resolve)
    run_batch(inp, tmp_path / "out", source=source)
    assert calls["list"] == 1
    assert calls["order"][0] == "list"


def test_unreadable_registry_aborts_the_batch_with_exit_1(
    tmp_path, monkeypatch, caplog
):
    import wandb

    from sleap_roots_predict import batch as batch_mod
    from sleap_roots_predict.__main__ import main
    from sleap_roots_predict.model_registry import (
        NoReadableModelCardsError,
        WandbRegistrySource,
    )
    from registry_fakes import FakeApi, FakeArtifact, _flat_meta

    monkeypatch.setenv("WANDB_API_KEY", "dummy")
    monkeypatch.setattr(
        wandb,
        "Api",
        lambda: FakeApi({"col": [FakeArtifact("reg/flat", metadata=_flat_meta())]}),
    )
    inp = tmp_path / "in"
    _real_scan(inp, "scanA", _RICE)
    out = tmp_path / "out"
    with pytest.raises(NoReadableModelCardsError):
        run_batch(inp, out, source=WandbRegistrySource(entity="ent", registry="reg"))
    assert not out.exists() or {p.name for p in out.iterdir()} <= {"run_manifest.json"}

    real_run_batch = batch_mod.run_batch

    def _with_source(*args, **kwargs):
        kwargs.setdefault("source", WandbRegistrySource(entity="ent", registry="reg"))
        return real_run_batch(*args, **kwargs)

    monkeypatch.setattr(batch_mod, "run_batch", _with_source)
    with caplog.at_level(logging.ERROR):
        with pytest.raises(NoReadableModelCardsError):
            main([str(inp), str(tmp_path / "out2")])
    assert "Batch aborted" in caplog.text


def test_discovery_error_before_first_processable_scan_still_aborts_on_catalog_failure(
    tmp_path, monkeypatch
):
    """A `failed` discovery-error scan ahead of the first processable one doesn't block a catalog-failure abort."""
    import wandb

    from sleap_roots_predict.model_registry import (
        NoReadableModelCardsError,
        WandbRegistrySource,
    )
    from registry_fakes import FakeApi, FakeArtifact, _flat_meta

    monkeypatch.setenv("WANDB_API_KEY", "dummy")
    monkeypatch.setattr(
        wandb,
        "Api",
        lambda: FakeApi({"col": [FakeArtifact("reg/flat", metadata=_flat_meta())]}),
    )
    inp = tmp_path / "in"
    _real_scan(inp, "scanA", _RICE)
    # A present-but-invalid sidecar sorts alongside scanA's in discover_scans' single
    # found-sidecars pass (unlike a manifest-scoped-but-*missing* scan_key, which is
    # always appended in a second pass *after* every found sidecar regardless of its
    # name -- verified empirically; that shape can never precede scanA). Naming it
    # "aaa-ghost" sorts it first, landing it in scans[0] as the discovery-error entry
    # ahead of the processable scanA in scans[1].
    ghost_dir = inp / "aaa-ghost"
    ghost_dir.mkdir(parents=True)
    (ghost_dir / "aaa-ghost.scan_metadata.json").write_text(
        json.dumps(
            {"scan_key": "aaa-ghost", "image_ids": ["a"], "images_checksum": "sha256:x"}
        )
    )
    (inp / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1",
                "pipeline_run_id": "r",
                "scan_keys": ["aaa-ghost", "scanA"],
            }
        )
    )
    scans = discover_scans(inp)
    assert [s.scan_key for s in scans] == ["aaa-ghost", "scanA"]
    assert scans[0].error is not None
    assert scans[1].error is None

    out = tmp_path / "out"
    with pytest.raises(NoReadableModelCardsError):
        run_batch(inp, out, source=WandbRegistrySource(entity="ent", registry="reg"))


def test_missing_key_fails_the_batch_not_each_scan(tmp_path, clean_wandb_env):
    inp = tmp_path / "in"
    _real_scan(inp, "scanA", _RICE)
    with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
        run_batch(inp, tmp_path / "out")


def test_stop_before_first_scan_skips_the_catalog(tmp_path):
    inp = tmp_path / "in"
    _real_scan(inp, "scanA", _RICE)
    source, calls = _recording_source()
    run_batch(inp, tmp_path / "out", source=source, should_stop=lambda: True)
    assert calls["n"] == 0


def test_only_errored_scans_skip_the_catalog(tmp_path):
    inp = tmp_path / "in"
    inp.mkdir()
    (inp / "run_manifest.json").write_text(
        json.dumps(
            {"schema_version": "1", "pipeline_run_id": "r", "scan_keys": ["ghost"]}
        )
    )
    source, calls = _recording_source()
    result = run_batch(inp, tmp_path / "out", source=source)
    assert [s.status for s in result.scans] == ["failed"]
    assert calls["n"] == 0


class _EmptySource:
    """A model source whose catalog is empty (lists no cards at all)."""

    def list_cards(self):
        """Return no cards."""
        return []

    def materialize(self, ref):
        """Never reached: nothing can be selected from an empty catalog."""
        raise AssertionError("materialize should never be called")


def test_empty_catalog_aborts_the_batch(tmp_path):
    """A processable scan against an empty catalog is a batch-level error, not exit 3."""
    from sleap_roots_predict.model_registry import NoReadableModelCardsError

    inp = tmp_path / "in"
    _real_scan(inp, "scanA", _RICE)
    with pytest.raises(NoReadableModelCardsError, match="no model cards"):
        run_batch(inp, tmp_path / "out", source=_EmptySource())


def test_empty_catalog_cli_exits_1_with_the_staging_line(tmp_path, monkeypatch, caplog):
    """The CLI logs its one-line staging message for an empty catalog and re-raises."""
    from sleap_roots_predict import batch as batch_mod
    from sleap_roots_predict.__main__ import main
    from sleap_roots_predict.model_registry import NoReadableModelCardsError

    inp = tmp_path / "in"
    _real_scan(inp, "scanA", _RICE)
    real_run_batch = batch_mod.run_batch

    def _with_empty_source(*args, **kwargs):
        kwargs.setdefault("source", _EmptySource())
        return real_run_batch(*args, **kwargs)

    monkeypatch.setattr(batch_mod, "run_batch", _with_empty_source)
    with caplog.at_level(logging.ERROR):
        with pytest.raises(NoReadableModelCardsError):
            main([str(inp), str(tmp_path / "out")])
    assert "Batch aborted" in caplog.text


def test_empty_catalog_with_only_errored_scans_is_not_loaded(tmp_path):
    """No processable scan means no load, so an empty catalog changes nothing (exit 3)."""
    inp = tmp_path / "in"
    inp.mkdir()
    (inp / "run_manifest.json").write_text(
        json.dumps(
            {"schema_version": "1", "pipeline_run_id": "r", "scan_keys": ["ghost"]}
        )
    )
    result = run_batch(inp, tmp_path / "out", source=_EmptySource())
    assert [s.status for s in result.scans] == ["failed"]


def test_catalog_failure_during_requested_stop_propagates(
    tmp_path, caplog, monkeypatch
):
    """A catalog failure with a stop pending exits 1 (raises), not 143.

    The stop is requested *inside* ``list_cards()``, i.e. after that iteration's stop
    check, which is the only window in which the two can coincide; the spy asserts the
    stop really was pending when the catalog failed.
    """
    import signal

    import sleap_roots_predict.batch as batch_mod
    from sleap_roots_predict.__main__ import main
    from sleap_roots_predict.model_registry import NoReadableModelCardsError

    inp = tmp_path / "in"
    _real_scan(inp, "scanA", _RICE)
    seen = {}

    class _StopThenEmpty:
        def list_cards(self):
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            seen["stop_pending"] = seen["should_stop"]()
            return []

        def materialize(self, ref):
            raise AssertionError("materialize should never be called")

    real_run_batch = batch_mod.run_batch

    def _with_source(*args, **kwargs):
        seen["should_stop"] = kwargs["should_stop"]
        kwargs.setdefault("source", _StopThenEmpty())
        return real_run_batch(*args, **kwargs)

    monkeypatch.setattr(batch_mod, "run_batch", _with_source)
    prev_handler = signal.getsignal(signal.SIGTERM)
    try:
        with caplog.at_level(logging.WARNING):
            with pytest.raises(NoReadableModelCardsError):
                main([str(inp), str(tmp_path / "out")])
    finally:
        signal.signal(signal.SIGTERM, prev_handler)
    assert seen["stop_pending"], "stop was not pending when the catalog failed"
    assert not any("Terminated by SIGTERM" in r.message for r in caplog.records)


_PER_RUN = "run_manifest.wf-a.json"
_TWELVE = [f"scan_{i}" for i in range(1, 13)]


def _stage_accumulated_union(inp: Path, *, with_per_run: bool) -> Path:
    """The measured srp#71 shape: a legacy manifest carrying every run's keys."""
    for key in _TWELVE:
        _write_scan(inp, key, _RICE)
    write_run_manifest(
        inp, _MANIFEST, pipeline_run_id="sleap-roots-pipeline-hpdpf", scan_keys=_TWELVE
    )
    if with_per_run:
        return write_run_manifest(
            inp, _PER_RUN, pipeline_run_id="wf-a", scan_keys=["scan_7"]
        )
    return inp / _MANIFEST


def test_a_per_run_manifest_stops_the_accumulated_union(tmp_path, monkeypatch):
    monkeypatch.setenv("ARGO_WORKFLOW_NAME", "wf-a")
    inp, out = tmp_path / "in", tmp_path / "out"
    src = _stage_accumulated_union(inp, with_per_run=True)
    assert [s.scan_key for s in discover_scans(inp)] == ["scan_7"]
    source, _ = _recording_source()
    result = run_batch(inp, out, source=source)
    assert [s.scan_key for s in result.scans] == ["scan_7"]
    assert (out / _PER_RUN).read_bytes() == src.read_bytes()
    assert not (out / _MANIFEST).exists()


def test_a_legacy_union_is_honored_and_flagged_during_the_rollout(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setenv("ARGO_WORKFLOW_NAME", "wf-a")
    inp, out = tmp_path / "in", tmp_path / "out"
    src = _stage_accumulated_union(inp, with_per_run=False)
    source, _ = _recording_source()
    with caplog.at_level(logging.WARNING, logger="sleap_roots_predict.run_manifest"):
        result = run_batch(inp, out, source=source)
    assert sorted(s.scan_key for s in result.scans) == sorted(_TWELVE)
    assert any(
        "hpdpf" in r.getMessage() and "wf-a" in r.getMessage() for r in caplog.records
    )
    assert {p.name for p in out.iterdir() if p.is_file()} == {_MANIFEST}
    assert (out / _MANIFEST).read_bytes() == src.read_bytes()


def _no_manifest(inp):
    _write_scan(inp, "scanA", _RICE)


def _foreign_per_run(inp):
    _write_scan(inp, "scanA", _RICE)
    write_run_manifest(inp, _PER_RUN, pipeline_run_id="wf-b", scan_keys=["scanA"])


def _directory_at_legacy_path(inp):
    _write_scan(inp, "scanA", _RICE)
    (inp / _MANIFEST).mkdir()


@pytest.mark.parametrize(
    "run_id, stage, error",
    [
        ("wf-a", _no_manifest, "RunManifestMissingError"),
        ("wf-a", _foreign_per_run, "RunManifestIdentityError"),
        ("../x", _no_manifest, "ValueError"),
        (None, _directory_at_legacy_path, "OSError"),
    ],
    ids=["missing", "foreign", "unusable-id", "directory-at-path"],
)
def test_manifest_staging_errors_abort_before_any_work(
    tmp_path, monkeypatch, run_id, stage, error
):
    import sleap_roots_contracts as contracts

    if run_id is not None:
        monkeypatch.setenv("ARGO_WORKFLOW_NAME", run_id)
    inp, out = tmp_path / "in", tmp_path / "out"
    stage(inp)
    expected = {"ValueError": ValueError, "OSError": OSError}.get(error) or getattr(
        contracts, error
    )
    source, calls = _recording_source()
    with pytest.raises(expected):
        run_batch(inp, out, source=source)
    assert calls["n"] == 0
    assert not out.exists()


def test_standalone_discovery_fails_loud_for_a_known_run(tmp_path, monkeypatch):
    from sleap_roots_contracts import RunManifestMissingError

    monkeypatch.setenv("ARGO_WORKFLOW_NAME", "wf-a")
    _write_scan(tmp_path, "scanA", _RICE)
    with pytest.raises(RunManifestMissingError):
        discover_scans(tmp_path)


def test_discovery_without_identity_ignores_per_run_manifests(tmp_path):
    _write_scan(tmp_path, "scanA", _RICE)
    _write_scan(tmp_path, "scanB", _RICE)
    write_run_manifest(tmp_path, _PER_RUN, pipeline_run_id="wf-a", scan_keys=["scanA"])
    assert [s.scan_key for s in discover_scans(tmp_path)] == ["scanA", "scanB"]


@pytest.mark.parametrize("value", ["   ", ""])
def test_a_blank_run_identity_is_no_identity(tmp_path, monkeypatch, value):
    monkeypatch.setenv("ARGO_WORKFLOW_NAME", value)
    _write_scan(tmp_path, "scanA", _RICE)
    assert [s.scan_key for s in discover_scans(tmp_path)] == ["scanA"]


def test_a_run_identity_with_incidental_whitespace_still_resolves(
    tmp_path, monkeypatch
):
    """Review focus 1."""
    monkeypatch.setenv("ARGO_WORKFLOW_NAME", "wf-a\n")
    _write_scan(tmp_path, "scanA", _RICE)
    _write_scan(tmp_path, "scanB", _RICE)
    write_run_manifest(tmp_path, _PER_RUN, pipeline_run_id="wf-a", scan_keys=["scanB"])
    assert [s.scan_key for s in discover_scans(tmp_path)] == ["scanB"]


def test_an_input_path_that_is_a_file_fails_the_batch(tmp_path):
    """Review focus 2: a mis-mount must never pass as success, on any OS."""
    not_a_dir = tmp_path / "in"
    not_a_dir.write_bytes(b"x")
    source, calls = _recording_source()
    with pytest.raises((OSError, ValueError)):
        run_batch(not_a_dir, tmp_path / "out", source=source)
    assert calls["n"] == 0


def test_an_invalid_per_run_manifest_is_never_forwarded(tmp_path, monkeypatch):
    monkeypatch.setenv("ARGO_WORKFLOW_NAME", "wf-a")
    inp, out = tmp_path / "in", tmp_path / "out"
    _write_scan(inp, "scanA", _RICE)
    (inp / _PER_RUN).write_bytes(b"{not valid json")
    source, _ = _recording_source()
    with pytest.raises(ValueError):
        run_batch(inp, out, source=source)
    assert not out.exists()


def test_run_batch_forwards_the_resolved_per_run_bytes_even_if_rewritten(
    tmp_path, monkeypatch
):
    import sleap_roots_predict.batch as batch_mod

    monkeypatch.setenv("ARGO_WORKFLOW_NAME", "wf-a")
    inp, out = tmp_path / "in", tmp_path / "out"
    _write_scan(inp, "scanA", _RICE)
    src = write_run_manifest(inp, _PER_RUN, pipeline_run_id="wf-a", scan_keys=["scanA"])
    resolved = src.read_bytes()
    real = batch_mod._resolve_run_manifest

    def _resolve_then_rewrite(*args, **kwargs):
        loaded = real(*args, **kwargs)
        src.write_bytes(b'{"pipeline_run_id":"wf-a","scan_keys":["scanA","scanZ"]}')
        return loaded

    monkeypatch.setattr(batch_mod, "_resolve_run_manifest", _resolve_then_rewrite)
    source, _ = _recording_source()
    run_batch(inp, out, source=source)
    assert (out / _PER_RUN).read_bytes() == resolved


def test_missing_input_dir_wins_over_an_unusable_run_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("ARGO_WORKFLOW_NAME", "../x")
    with pytest.raises(FileNotFoundError, match="input scan directory does not exist"):
        run_batch(tmp_path / "nope", tmp_path / "out")


def _main_with_recording_source(monkeypatch):
    import sleap_roots_predict.batch as batch_mod

    source, calls = _recording_source()
    real_run_batch = batch_mod.run_batch

    def _with_source(*args, **kwargs):
        kwargs.setdefault("source", source)
        return real_run_batch(*args, **kwargs)

    monkeypatch.setattr(batch_mod, "run_batch", _with_source)
    return calls


@pytest.mark.parametrize(
    "stage, error",
    [
        (_no_manifest, "RunManifestMissingError"),
        (_foreign_per_run, "RunManifestIdentityError"),
    ],
)
def test_cli_logs_run_manifest_errors_as_staging_errors(
    tmp_path, monkeypatch, caplog, clean_wandb_env, stage, error
):
    import sleap_roots_contracts as contracts
    from sleap_roots_predict.__main__ import main

    monkeypatch.setenv("ARGO_WORKFLOW_NAME", "wf-a")
    inp, out = tmp_path / "in", tmp_path / "out"
    stage(inp)
    calls = _main_with_recording_source(monkeypatch)
    with caplog.at_level("ERROR"):
        with pytest.raises(getattr(contracts, error)):
            main([str(inp), str(out)])
    assert any("Batch aborted" in r.getMessage() for r in caplog.records)
    assert calls["n"] == 0 and not out.exists()
