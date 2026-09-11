"""Forward the run-scoping ``run_manifest.json`` from a stage's input to its output.

``RunManifest`` (bloomctl's cross-repo run-scoping shape, keyed by ``scan_keys``) is a
*different* concept from ``PredictionManifest`` (one scan's predict output, written by
``output_contract.py``). Kept in its own module so "manifest" never means two things at
once -- see ``sleap-roots-pipeline#37``.

Two deliberate divergences from the sibling ``trait_extractor/run_manifest.py`` in
``sleap-roots``, both argued in the ``forward-run-manifest-to-output`` change's
``design.md``: a copy failure **raises** rather than being treated as best-effort (a
silently skipped copy is the bug this exists to prevent, recurring undetected), and the
temporary file is cleaned up on failure. Do not "harmonize" either back without
re-reading that rationale.
"""

import logging
import os
import shutil
import tempfile
from pathlib import Path

from sleap_roots_contracts import RUN_MANIFEST_FILENAME

logger = logging.getLogger(__name__)


def _is_same_file(left: Path, right: Path) -> bool:
    """Whether two paths provably refer to the same file, however they are spelled.

    Compares ``(st_dev, st_ino)`` rather than path strings, so a symlinked or
    bind-mounted ``output_dir`` is caught. Open-coded instead of calling
    :func:`os.path.samefile` to reject a degenerate stat: on Windows ``os.stat`` falls
    back to a path that reports ``st_ino`` and ``st_dev`` as ``0`` when the file cannot
    be opened, and two such stats compare equal -- which would make an unrelated pair
    look identical and skip the forward silently. A zero inode therefore means "not
    provably the same file": re-copying a file onto itself is harmless, silently
    skipping a real forward is the bug this module exists to prevent.

    Args:
        left: First path to compare.
        right: Second path to compare.

    Returns:
        True only when both paths provably refer to one file.

    Raises:
        OSError: If either path cannot be stat'd -- including ``FileNotFoundError``
            when the destination does not exist yet, which is the ordinary case.
    """
    left_stat = os.stat(left)
    right_stat = os.stat(right)
    if left_stat.st_ino == 0:
        return False
    return (left_stat.st_dev, left_stat.st_ino) == (
        right_stat.st_dev,
        right_stat.st_ino,
    )


def copy_run_manifest_forward(input_dir: str | Path, output_dir: str | Path) -> None:
    """Copy ``run_manifest.json`` from ``input_dir`` into ``output_dir``, if present.

    A raw byte copy -- never a re-serialization through ``RunManifest`` -- and no
    validation of contents, so the forwarded file is byte-identical to what the
    upstream producer wrote. Written atomically via a temporary file in ``output_dir``
    plus :func:`os.replace`, under a name unique to this process so concurrent
    invocations sharing an output directory can never publish one another's
    partially-written bytes.

    Args:
        input_dir: Directory to look for ``RUN_MANIFEST_FILENAME`` in (top level only;
            never searched recursively).
        output_dir: Directory to copy the manifest into (created if missing).

    Returns:
        None. A no-op when no manifest is present under ``input_dir``, or when source
        and destination are the same file (a caller passing ``input_dir ==
        output_dir``, or a symlinked/bind-mounted ``output_dir``). When no manifest is
        present but ``output_dir`` already holds one from an earlier run, that file is
        left in place and a warning is logged: it cannot be distinguished from a
        concurrent invocation's file, but left silent it would scope the downstream
        stage to an earlier run's ``scan_keys``.

    Raises:
        OSError: If a present manifest cannot be forwarded. Deliberately not
            best-effort: a silently missing forwarded manifest makes the downstream
            stage fall back to unscoped discovery, which is the contamination this
            function exists to prevent.
    """
    source_dir = Path(input_dir)
    source = source_dir / RUN_MANIFEST_FILENAME
    destination_dir = Path(output_dir)
    destination = destination_dir / RUN_MANIFEST_FILENAME

    # Presence before identity: os.path.samefile stats both operands, so an identity
    # check first would raise on a nonexistent source.
    if not source.is_file():
        if destination.is_file():
            logger.warning(
                "No %s under %s, but %s already holds one from an earlier run; "
                "leaving it in place -- the downstream stage will be scoped to it",
                RUN_MANIFEST_FILENAME,
                source_dir.as_posix(),
                destination_dir.as_posix(),
            )
        else:
            logger.debug(
                "No %s under %s; nothing to forward",
                RUN_MANIFEST_FILENAME,
                source_dir.as_posix(),
            )
        return

    tmp: str | None = None
    try:
        # The identity check stats BOTH operands and raises FileNotFoundError when the
        # destination does not exist yet -- the ordinary case -- so it can never be
        # called bare. NotADirectoryError covers a file occupying a path component
        # (POSIX; Windows reports FileNotFoundError there). A PermissionError from
        # stat is a real staging error and is deliberately left to propagate -- but it
        # must do so from INSIDE this block, or it escapes without the log naming both
        # directories that the spec requires before any OSError propagates.
        try:
            if _is_same_file(source, destination):
                return
        except (FileNotFoundError, NotADirectoryError):
            pass

        # mkdir must precede mkstemp: mkstemp does not create its dir= argument.
        destination_dir.mkdir(parents=True, exist_ok=True)
        # A name unique to this writer, inside output_dir. Unique because the
        # destination is shared across concurrent invocations, so a fixed name would
        # let one writer replace another's half-written temp into the final path.
        # Inside output_dir because a system-temp file would make the replace
        # cross-device (EXDEV) on the NFS production mount. Dot-prefixed so an orphan
        # left by a SIGKILL is hidden from any future "run_manifest*" glob.
        fd, tmp = tempfile.mkstemp(
            dir=destination_dir,
            prefix=f".{RUN_MANIFEST_FILENAME}.",
            suffix=".tmp",
        )
        # Close before copying: mkstemp hands back an OPEN fd, and on Windows an open
        # handle makes os.replace fail with WinError 32 -- while copyfile succeeds, so
        # the bug would surface only at the replace, and only on the Windows leg.
        os.close(fd)
        shutil.copyfile(source, tmp)
        # mkstemp creates at 0600 regardless of umask and copyfile does not alter an
        # existing file's mode, so without this the forwarded manifest is the one file
        # predict writes that the downstream container -- a different uid on the same
        # shared NFS mount -- cannot read. Before the replace, never after: after would
        # briefly publish the destination at the private temp mode. No-op on Windows.
        shutil.copymode(source, tmp)
        os.replace(tmp, destination)
    except Exception:
        # Log BEFORE cleaning up: if the unlink itself raises, logging afterwards would
        # never run and the secondary error would replace the real one.
        logger.error(
            "Failed to forward %s from %s to %s",
            RUN_MANIFEST_FILENAME,
            source_dir.as_posix(),
            destination_dir.as_posix(),
        )
        if tmp is not None:
            try:
                Path(tmp).unlink(missing_ok=True)
            except OSError:
                logger.warning(
                    "Could not remove temporary file %s", Path(tmp).as_posix()
                )
        raise

    logger.info(
        "Forwarded %s from %s to %s",
        RUN_MANIFEST_FILENAME,
        source_dir.as_posix(),
        destination_dir.as_posix(),
    )
