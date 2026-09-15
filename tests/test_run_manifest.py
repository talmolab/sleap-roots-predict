"""Real, no-mock tests for the run-manifest forward-copy.

Assertions on residue use the output directory's *exact* contents rather than a
``*.tmp`` glob: the temp file's name is an implementation detail, and a glob that
happens not to match it would pass against an implementation with no cleanup at all.
"""

import errno
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
    # Stronger than "no residue in out": a failed forward also removes the output
    # directory when it was this call that created it.
    assert not out.exists()


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
    # Byte and name assertions alone cannot see a copy that round-tripped the file over
    # itself -- they hold for an implementation with no identity check at all. The
    # mtime is what discriminates: a replace publishes a *new* file at that path.
    before_mtime = src.stat().st_mtime_ns
    copy_run_manifest_forward(inp, inp / "sub" / "..")
    assert src.read_bytes() == before
    assert _names(inp) == {RUN_MANIFEST_FILENAME, "sub"}
    assert src.stat().st_mtime_ns == before_mtime


def test_hard_linked_destination_in_another_directory_is_a_noop(tmp_path: Path):
    """The discriminating identity case, and the one the production topology needs.

    A hard link gives two path strings sharing no prefix that nonetheless name one
    file -- so neither string comparison nor ``Path.resolve()`` can tell they are the
    same, only ``(st_dev, st_ino)`` can. That is the same property a bind-mounted
    ``output_dir`` has (design.md: one hostPath volume mounted at two container
    paths), tested without the symlink privilege tasks.md 1.1 rules out. The inode
    assertion is the discriminator: a copy that proceeded would ``os.replace`` a new
    file over the destination and break the link.
    """
    inp = tmp_path / "in"
    src = _stage(inp)
    out = tmp_path / "out"
    out.mkdir()
    try:
        os.link(src, out / RUN_MANIFEST_FILENAME)
    except (OSError, NotImplementedError) as exc:  # e.g. a FAT/exFAT temp dir
        pytest.skip(f"filesystem does not support hard links: {exc}")
    before = src.read_bytes()
    linked_ino = (out / RUN_MANIFEST_FILENAME).stat().st_ino

    copy_run_manifest_forward(inp, out)

    assert src.read_bytes() == before
    assert _names(out) == {RUN_MANIFEST_FILENAME}
    assert (out / RUN_MANIFEST_FILENAME).stat().st_ino == linked_ino


def test_a_zero_inode_pair_is_not_treated_as_the_same_file(tmp_path: Path, monkeypatch):
    """Identity compares ``(st_dev, st_ino)``, and on Windows ``os.stat`` falls back to
    a path reporting both as ``0`` when a file cannot be opened (a sharing violation,
    or an ACL permitting attribute reads but not opens). Two such stats compare *equal*,
    which would make an unrelated pair look identical and skip the forward silently --
    the one outcome this module exists to prevent. A zero inode must therefore mean
    "not provably the same file": re-copying a file onto itself is harmless, silently
    skipping a real forward is not.
    """
    src = _stage(tmp_path / "in")
    out = tmp_path / "out"
    _stage(out, b'{"pipeline_run_id":"previous"}')
    real_stat = os.stat
    # Compared as strings, not via resolve(): resolve() itself stats on some platforms,
    # which would recurse through this wrapper.
    degenerate = {
        os.fspath(tmp_path / "in" / RUN_MANIFEST_FILENAME),
        os.fspath(out / RUN_MANIFEST_FILENAME),
    }

    def _zero_inode_stat(path, *args, **kwargs):
        result = real_stat(path, *args, **kwargs)
        try:
            targeted = os.fspath(path) in degenerate
        except TypeError:  # an open fd, not a path
            targeted = False
        if not targeted:
            return result
        fields = list(result)  # st_mode, st_ino, st_dev, ... (10 primary fields)
        fields[1] = 0
        fields[2] = 0
        return os.stat_result(fields)

    monkeypatch.setattr(os, "stat", _zero_inode_stat)
    copy_run_manifest_forward(tmp_path / "in", out)
    assert (out / RUN_MANIFEST_FILENAME).read_bytes() == src.read_bytes()


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
    assert not out.exists()


def test_a_failed_forward_removes_an_output_directory_it_created(
    tmp_path: Path, monkeypatch
):
    """`mkdir` runs before the temp/replace sequence, so a later failure used to leave
    a newly-created, empty ``output_dir`` standing even though the batch raised. That
    is a state that could not occur before this change: orchestration that reads
    "output_dir does not exist" as "predict never ran" would now see an empty
    directory instead. Created parents are removed too, innermost first.
    """
    _stage(tmp_path / "in")
    out = tmp_path / "out" / "nested"
    monkeypatch.setattr(
        "os.replace",
        lambda *a, **k: (_ for _ in ()).throw(OSError("simulated interruption")),
    )
    with pytest.raises(OSError):
        copy_run_manifest_forward(tmp_path / "in", out)
    assert not out.exists()
    assert not (tmp_path / "out").exists()


def test_a_failed_forward_keeps_a_pre_existing_output_directory(
    tmp_path: Path, monkeypatch
):
    """Only directories this call created are removed. A shared output tree that was
    already there must survive untouched -- deleting it would be far worse than the
    empty directory the fix above removes.

    Deliberately a *pre-existing and empty* directory: that is the discriminating
    case. A non-empty one is protected by ``rmdir``'s own semantics, so it would
    survive even an implementation that tried to remove the destination
    unconditionally, and would prove nothing about tracking what this call created.
    """
    _stage(tmp_path / "in")
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(
        "os.replace",
        lambda *a, **k: (_ for _ in ()).throw(OSError("simulated interruption")),
    )
    with pytest.raises(OSError):
        copy_run_manifest_forward(tmp_path / "in", out)
    assert out.is_dir(), "removed an output directory it did not create"


def _raise_eacces_for(monkeypatch, target: Path):
    """Make stat'ing exactly ``target`` fail with EACCES, delegating everything else.

    ``Path.is_file()`` only swallows the errnos in ``pathlib._IGNORED_ERRNOS``
    (ENOENT, ENOTDIR, EBADF, WSAENOTSOCK) -- ``EACCES`` is not among them, so a
    permission problem propagates rather than reading as "absent". Injected at
    ``os.stat`` rather than via ``chmod``: 1.1 bans chmod, and it is a no-op for the
    owner on Windows anyway.
    """
    real_stat = os.stat
    wanted = os.fspath(target)

    def _stat(path, *args, **kwargs):
        try:
            hit = os.fspath(path) == wanted
        except TypeError:  # an open fd, not a path
            hit = False
        if hit:
            raise PermissionError(errno.EACCES, "Permission denied", wanted)
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", _stat)


def test_permission_error_from_the_presence_check_is_reported_cleanly(
    tmp_path: Path, caplog, monkeypatch
):
    """The presence check is the *first* thing that touches the filesystem, so it is
    the first thing that can fail -- and it was the one step left outside the block
    that satisfies the spec's "log both directories before raising". An unreadable
    source manifest (an NFS ACL misconfiguration, or a race narrowing the mode) skipped
    the mandated diagnostic entirely and surfaced only as the CLI's generic line.
    """
    _stage(tmp_path / "in")
    out = tmp_path / "out"
    _raise_eacces_for(monkeypatch, tmp_path / "in" / RUN_MANIFEST_FILENAME)
    with caplog.at_level(logging.ERROR, logger=_LOGGER):
        with pytest.raises(PermissionError):
            copy_run_manifest_forward(tmp_path / "in", out)
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert errors, "presence-check failure propagated without the mandated log"
    assert (tmp_path / "in").as_posix() in errors[0].message
    assert out.as_posix() in errors[0].message


def test_failed_replace_leaves_a_prior_manifest_complete(tmp_path: Path, monkeypatch):
    """The second arm of "either nothing, or the complete prior manifest".

    The test above seeds an *empty* output directory, so it only ever exercises the
    "nothing" arm. An implementation that unlinked the destination before replacing it
    would let a reader observe no manifest at all -- the unscoped-discovery fallback
    this change exists to prevent -- and, if the replace then failed, would leave the
    shared output directory with none.
    """
    _stage(tmp_path / "in")
    out = tmp_path / "out"
    prior = b'{"pipeline_run_id":"the-previous-run","scan_keys":["s1"]}'
    _stage(out, prior)
    monkeypatch.setattr(
        "os.replace",
        lambda *a, **k: (_ for _ in ()).throw(OSError("simulated interruption")),
    )
    with pytest.raises(OSError):
        copy_run_manifest_forward(tmp_path / "in", out)
    assert (out / RUN_MANIFEST_FILENAME).read_bytes() == prior
    assert _names(out) == {RUN_MANIFEST_FILENAME}


def test_a_failing_cleanup_does_not_swallow_the_real_error(
    tmp_path: Path, caplog, monkeypatch
):
    """Cleaning up *before* logging is a real bug, not a style preference: if the
    unlink itself raises, the error log never fires and the secondary cleanup error
    replaces the underlying filesystem one -- violating the spec's "the error raised
    is the underlying filesystem error, not a secondary error from the cleanup path".

    design.md recorded this as uncatchable ("no test in this plan can catch it, since
    1.1 bans chmod"); it needs no chmod, only two injected failures. This is also the
    only coverage of the "Could not remove temporary file" warning branch.
    """
    _stage(tmp_path / "in")
    out = tmp_path / "out"
    monkeypatch.setattr(
        "os.replace",
        lambda *a, **k: (_ for _ in ()).throw(OSError("the underlying failure")),
    )
    monkeypatch.setattr(
        Path, "unlink", lambda *a, **k: (_ for _ in ()).throw(OSError("secondary"))
    )
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        with pytest.raises(OSError, match="the underlying failure"):
            copy_run_manifest_forward(tmp_path / "in", out)
    assert [r for r in caplog.records if r.levelno == logging.ERROR]
    assert any("Could not remove temporary file" in r.message for r in caplog.records)


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


def test_permission_error_from_the_identity_check_is_reported_cleanly(
    tmp_path: Path, caplog, monkeypatch
):
    """The other half of the scenario above: the spec's "or its parent denies
    permission" case. A PermissionError from stat'ing the destination is deliberately
    *not* caught -- but it must still take the logging path, since the spec requires
    the forward-copy itself to name both directories before any OSError propagates.
    Reachable on the shared NFS mount, where the next stage runs as a different uid
    and can leave an output subtree this process cannot search.

    Injected at the identity check rather than via chmod: tasks.md 1.1 bans chmod,
    which is a no-op for the owner on Windows anyway.
    """
    _stage(tmp_path / "in")
    out = tmp_path / "out"

    def _denied(_left, _right):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr("sleap_roots_predict.run_manifest._is_same_file", _denied)
    with caplog.at_level(logging.ERROR, logger=_LOGGER):
        with pytest.raises(PermissionError):
            copy_run_manifest_forward(tmp_path / "in", out)
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert errors, "identity-check failure propagated without the mandated log"
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
