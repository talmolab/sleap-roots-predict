"""Real, no-mock tests for the run-manifest forward-copy.

Assertions on residue use the output directory's *exact* contents rather than a
``*.tmp`` glob: the temp file's name is an implementation detail, and a glob that
happens not to match it would pass against an implementation with no cleanup at all.
"""

import logging
import os
from pathlib import Path

import pytest
from sleap_roots_contracts import RUN_MANIFEST_FILENAME

from sleap_roots_predict.run_manifest import copy_run_manifest_forward

_LOGGER = "sleap_roots_predict.run_manifest"

# Bytes a RunManifest round-trip would not reproduce: non-canonical key order, an
# undeclared extra field (RunManifest is frozen with pydantic's default
# extra="ignore", so this validates but is dropped by re-serialization), interior
# whitespace, and no trailing newline.
_NON_CANONICAL = b'{"scan_keys": ["scan_1"],  "pipeline_run_id": "run-1", "extra": 1}'


def _stage(root: Path, body: bytes = _NON_CANONICAL) -> Path:
    """Create ``root`` with a manifest in it; return the manifest path."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / RUN_MANIFEST_FILENAME
    path.write_bytes(body)
    return path


def _names(directory: Path) -> set:
    return {p.name for p in directory.iterdir()}


def test_forwards_unparsable_bytes_verbatim(tmp_path: Path):
    """A byte copy succeeds where a re-serializing implementation would raise."""
    src = _stage(tmp_path / "in", b"{not valid json")
    out = tmp_path / "out"
    copy_run_manifest_forward(tmp_path / "in", out)
    assert (out / RUN_MANIFEST_FILENAME).read_bytes() == src.read_bytes()


def test_forwards_non_canonical_bytes_verbatim(tmp_path: Path):
    src = _stage(tmp_path / "in")
    out = tmp_path / "out"
    copy_run_manifest_forward(tmp_path / "in", out)
    assert (out / RUN_MANIFEST_FILENAME).read_bytes() == src.read_bytes()


def test_temp_file_is_created_inside_the_output_directory(tmp_path: Path, monkeypatch):
    """A system-temp implementation is invisible to CI but fails with EXDEV in prod."""
    import tempfile

    _stage(tmp_path / "in")
    out = tmp_path / "out"
    seen = []
    real = tempfile.mkstemp

    def _spy(*args, **kwargs):
        seen.append(kwargs.get("dir", args[2] if len(args) > 2 else None))
        return real(*args, **kwargs)

    monkeypatch.setattr(tempfile, "mkstemp", _spy)
    copy_run_manifest_forward(tmp_path / "in", out)
    assert seen, "implementation did not allocate a temp file via tempfile.mkstemp"
    assert Path(seen[0]).resolve() == out.resolve()


def test_temp_name_is_unique_per_writer(tmp_path: Path, monkeypatch):
    """A fixed temp name lets one writer publish another's half-written file."""
    _stage(tmp_path / "in")
    out = tmp_path / "out"
    seen = []

    def _record_and_fail(src, dst, *a, **k):
        seen.append(str(src))
        raise OSError("simulated interruption")

    monkeypatch.setattr("os.replace", _record_and_fail)
    for _ in range(2):
        with pytest.raises(OSError):
            copy_run_manifest_forward(tmp_path / "in", out)

    assert len(seen) == 2, "implementation never reached os.replace"
    assert seen[0] != seen[1]
    assert _names(out) == set()


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_forwarded_permissions_match_the_source(tmp_path: Path):
    """0o640 is neither mkstemp's 0600 nor umask-022's 0644, so this catches both a
    missing copymode and a hardcoded mode."""
    src = _stage(tmp_path / "in")
    os.chmod(src, 0o640)
    out = tmp_path / "out"
    copy_run_manifest_forward(tmp_path / "in", out)
    assert (out / RUN_MANIFEST_FILENAME).stat().st_mode & 0o777 == 0o640


def test_absent_manifest_is_a_noop(tmp_path: Path, caplog):
    inp = tmp_path / "in"
    inp.mkdir()
    out = tmp_path / "out"
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        copy_run_manifest_forward(inp, out)
    assert not out.exists()
    assert any("nothing to forward" in r.message.lower() for r in caplog.records)


def test_absent_manifest_with_stale_destination_warns_and_keeps(tmp_path: Path, caplog):
    inp = tmp_path / "in"
    inp.mkdir()
    stale = _stage(tmp_path / "out", b'{"pipeline_run_id":"old","scan_keys":["s0"]}')
    before = stale.read_bytes()
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        copy_run_manifest_forward(inp, tmp_path / "out")
    assert stale.read_bytes() == before
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "no warning for a stale output manifest"
    assert (tmp_path / "out").as_posix() in warnings[0].message


def test_absent_manifest_with_same_input_and_output_is_a_noop(tmp_path: Path, caplog):
    """Presence is checked before identity, so a nonexistent source never raises --
    and the stale-destination branch must not fire when source and destination are
    the same (absent) path."""
    both = tmp_path / "both"
    both.mkdir()
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        copy_run_manifest_forward(both, both)
    assert _names(both) == set()
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_output_directory_is_created_including_nested_parents(tmp_path: Path):
    _stage(tmp_path / "in")
    out = tmp_path / "out" / "a" / "b"
    copy_run_manifest_forward(tmp_path / "in", out)
    assert (out / RUN_MANIFEST_FILENAME).is_file()


def test_same_file_via_a_different_path_spelling_is_a_noop(tmp_path: Path):
    """Identity is decided by what the paths refer to, not by string comparison."""
    inp = tmp_path / "in"
    src = _stage(inp)
    (inp / "sub").mkdir()  # POSIX resolves '..' component-wise; without this it ENOENTs
    before = src.read_bytes()
    copy_run_manifest_forward(inp, inp / "sub" / "..")
    assert src.read_bytes() == before
    assert _names(inp) == {RUN_MANIFEST_FILENAME, "sub"}


def test_stale_output_manifest_is_fully_replaced(tmp_path: Path):
    src = _stage(tmp_path / "in")
    _stage(tmp_path / "out", b'{"pipeline_run_id":"a-much-longer-previous-run","x":0}')
    copy_run_manifest_forward(tmp_path / "in", tmp_path / "out")
    dst = tmp_path / "out" / RUN_MANIFEST_FILENAME
    assert dst.read_bytes() == src.read_bytes()
    assert b"previous-run" not in dst.read_bytes()


def test_successful_forward_leaves_no_residue(tmp_path: Path):
    _stage(tmp_path / "in")
    out = tmp_path / "out"
    copy_run_manifest_forward(tmp_path / "in", out)
    assert _names(out) == {RUN_MANIFEST_FILENAME}


def test_failed_replace_cleans_up_and_publishes_nothing(tmp_path: Path, monkeypatch):
    """Injects at os.replace, not copyfile: with copyfile patched the temp never
    exists, so a residue assertion would pass against no cleanup at all."""
    _stage(tmp_path / "in")
    out = tmp_path / "out"
    monkeypatch.setattr(
        "os.replace",
        lambda *a, **k: (_ for _ in ()).throw(OSError("simulated interruption")),
    )
    with pytest.raises(OSError):
        copy_run_manifest_forward(tmp_path / "in", out)
    assert _names(out) == set()


def test_failure_before_the_output_directory_is_prepared_reports_cleanly(
    tmp_path: Path, caplog
):
    """A file occupying the output_dir path makes mkdir raise before the temp exists.
    The error must stay an OSError -- a NameError from an unbound temp variable would
    escape the CLI's handler into the raw-traceback path."""
    _stage(tmp_path / "in")
    out = tmp_path / "out"
    out.write_text("i am a file, not a directory")
    with caplog.at_level(logging.ERROR, logger=_LOGGER):
        with pytest.raises(OSError) as excinfo:
            copy_run_manifest_forward(tmp_path / "in", out)
    assert not isinstance(excinfo.value, NameError)
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert errors, "no error logged before propagating"
    assert (tmp_path / "in").as_posix() in errors[0].message
    assert out.as_posix() in errors[0].message


def test_directory_at_the_destination_path_raises(tmp_path: Path):
    _stage(tmp_path / "in")
    out = tmp_path / "out"
    (out / RUN_MANIFEST_FILENAME).mkdir(parents=True)
    with pytest.raises(OSError):  # IsADirectoryError on POSIX, PermissionError on nt
        copy_run_manifest_forward(tmp_path / "in", out)


def test_accepts_str_paths_on_the_success_path(tmp_path: Path):
    src = _stage(tmp_path / "in")
    out = tmp_path / "out"
    copy_run_manifest_forward(str(tmp_path / "in"), str(out))
    assert (out / RUN_MANIFEST_FILENAME).read_bytes() == src.read_bytes()


def test_accepts_str_paths_on_the_failure_path(tmp_path: Path, monkeypatch):
    """The failure arm is the one that matters: logging input_dir.as_posix() on a str
    raises AttributeError, which is not an OSError and would escape the CLI handler."""
    _stage(tmp_path / "in")
    out = tmp_path / "out"
    monkeypatch.setattr(
        "os.replace",
        lambda *a, **k: (_ for _ in ()).throw(OSError("simulated interruption")),
    )
    with pytest.raises(OSError) as excinfo:
        copy_run_manifest_forward(str(tmp_path / "in"), str(out))
    assert not isinstance(excinfo.value, AttributeError)


def test_successful_forward_logs_both_directories_at_info(tmp_path: Path, caplog):
    _stage(tmp_path / "in")
    out = tmp_path / "out"
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        copy_run_manifest_forward(tmp_path / "in", out)
    infos = [r for r in caplog.records if r.levelno == logging.INFO]
    assert infos, "forward not logged at INFO"
    assert (tmp_path / "in").as_posix() in infos[0].message
    assert out.as_posix() in infos[0].message


def test_zero_byte_manifest_is_forwarded_verbatim(tmp_path: Path):
    _stage(tmp_path / "in", b"")
    out = tmp_path / "out"
    copy_run_manifest_forward(tmp_path / "in", out)
    assert (out / RUN_MANIFEST_FILENAME).read_bytes() == b""
