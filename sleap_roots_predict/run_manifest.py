"""Resolve this run's run manifest, and forward it from a stage's input to its output.

``RunManifest`` (bloomctl's cross-repo run-scoping shape, keyed by ``scan_keys``) is a
*different* concept from ``PredictionManifest`` (one scan's predict output, written by
``output_contract.py``). Kept in its own module so "manifest" never means two things at
once -- see ``sleap-roots-pipeline#37``.

Resolution is ``sleap-roots-contracts``' own policy (0.1.0a9,
talmolab/sleap-roots-pipeline#71): with a run identity (``ARGO_WORKFLOW_NAME``, read only
through ``pipeline_run_id_from_env``) the per-run ``run_manifest.<pipeline_run_id>.json``
first, then -- while ``allow_legacy=True`` -- the legacy ``run_manifest.json``; a known run
with neither raises rather than falling back to unscoped discovery. Without an identity
only ``run_manifest.json`` is a candidate, so local runs behave as before.

One deliberate divergence from the sibling ``trait_extractor/run_manifest.py`` in
``sleap-roots``: a copy failure **raises** rather than being logged as best-effort (a
silently skipped copy is the bug this exists to prevent, recurring undetected), argued in
the archived ``forward-run-manifest-to-output`` change's ``design.md``. Do not
"harmonize" it back without re-reading that rationale.
"""

import logging
import os
import tempfile
from pathlib import Path

from sleap_roots_contracts import (
    RUN_MANIFEST_FILENAME,
    LoadedRunManifest,
    load_run_manifest,
    pipeline_run_id_from_env,
    read_run_manifest,
)

logger = logging.getLogger(__name__)

# "No read supplied -- read the manifest yourself." Distinct from None, which is a
# caller asserting it already looked and the manifest is absent.
_UNREAD = object()

# Matches the per-run names contracts' run_manifest_filename() builds
# ("run_manifest.<pipeline_run_id>.json") but never the legacy "run_manifest.json" (the
# pattern needs a second dot) nor a dot-prefixed temp file. Contracts keeps its
# prefix/suffix private, so the pattern is spelled out here, as traits does.
_PER_RUN_MANIFEST_GLOB = "run_manifest.*.json"


def _resolve_run_manifest(
    input_dir: str | Path, pipeline_run_id: str | None
) -> LoadedRunManifest | None:
    """Load this run's manifest once, and log the cases where its scope may be wrong.

    Resolution, parsing and the per-run identity cross-check are contracts'
    ``load_run_manifest``; this adds only traceability for what that function
    deliberately leaves unchecked. Neither warning changes the scope:

    - the read is not per-run (unscoped, or scoped by the legacy file) while per-run
      manifests sit unread at the top level of ``input_dir`` -- typically a copied
      cluster tree re-run locally;
    - a legacy manifest read under a known identity names a different run -- under Argo
      the pre-per-run writer stamps each merge with the current run, so this means
      another workflow merged into the shared directory afterwards.

    Args:
        input_dir: Directory whose top level holds the manifest.
        pipeline_run_id: This run's identity, from ``pipeline_run_id_from_env()``.

    Returns:
        The loaded manifest and the read it came from, or ``None`` when the run has no
        identity and no ``run_manifest.json`` exists.

    Raises:
        sleap_roots_contracts.RunManifestMissingError: If the identity is known but no
            manifest resolves for it.
        sleap_roots_contracts.RunManifestIdentityError: If a per-run manifest names a
            different run.
        ValueError: If the identity is unusable as a filename component, or the
            manifest fails ``RunManifest`` validation (pydantic's ``ValidationError``).
        OSError: If a candidate exists but cannot be read (a directory there, a
            permission error, a dangling symlink), or ``input_dir`` is missing.
    """
    # allow_legacy=True while any stage may still write the legacy name; flipping it to
    # False is fleet-wide (talmolab/sleap-roots-pipeline#82).
    loaded = load_run_manifest(input_dir, pipeline_run_id, allow_legacy=True)
    input_dir = Path(input_dir)
    if loaded is None or not loaded.read.is_per_run:
        per_run = sorted(p.name for p in input_dir.glob(_PER_RUN_MANIFEST_GLOB))
        if per_run:
            scoped_by = (
                "discovering every scan"
                if loaded is None
                else f"scoping by legacy {loaded.read.filename}"
            )
            logger.warning(
                "Run %r is not scoped by a per-run manifest in %s (%s); per-run "
                "manifest(s) present but unread: %s",
                pipeline_run_id,
                input_dir.as_posix(),
                scoped_by,
                ", ".join(per_run),
            )
    if (
        loaded is not None
        and pipeline_run_id is not None
        and not loaded.read.is_per_run
        and loaded.manifest.pipeline_run_id != pipeline_run_id
    ):
        logger.warning(
            "Run %r is scoped by legacy %s in %s, which names run %r -- honored while "
            "the legacy fallback is enabled, but it is another run's scope",
            pipeline_run_id,
            loaded.read.filename,
            input_dir.as_posix(),
            loaded.manifest.pipeline_run_id,
        )
    return loaded


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


def copy_run_manifest_forward(
    input_dir: str | Path, output_dir: str | Path, *, read=_UNREAD
) -> None:
    """Republish this run's manifest from ``input_dir`` into ``output_dir``, if present.

    Publishes under the filename it was read from -- the per-run
    ``run_manifest.<pipeline_run_id>.json`` or the legacy ``run_manifest.json`` -- so the
    downstream stage, resolving by the same policy and identity, finds it. A raw byte
    copy -- never a re-serialization through ``RunManifest`` -- and no validation of
    contents, so the forwarded file is byte-identical to what the upstream producer
    wrote. Written atomically via a temporary file in ``output_dir`` plus
    :func:`os.replace`, under a name unique to this process so concurrent invocations
    sharing an output directory can never publish one another's partially-written bytes.

    Args:
        input_dir: Directory whose top level holds the manifest (never searched
            recursively).
        output_dir: Directory to copy the manifest into. Created if missing, and
            removed again if the copy then fails, so a failed forward never leaves a
            directory that did not exist before.
        read: Package-internal. A ``RunManifestRead`` already taken from ``input_dir``,
            or ``None`` to assert none was found. Omit it and the manifest is located
            here with contracts' non-parsing ``read_run_manifest`` (``allow_legacy=True``,
            identity from ``pipeline_run_id_from_env()``). ``run_batch`` passes the read
            discovery scoped against, so the bytes published are the bytes scoped
            against even if the source changes in between.

    Returns:
        None. A no-op when no manifest was found (possible only with no run identity),
        or when source and destination are the same file (a caller passing
        ``input_dir == output_dir``, or a symlinked/bind-mounted ``output_dir``). When
        none was found but ``output_dir`` already holds a ``run_manifest.json`` from an
        earlier run, that file is left in place and a warning is logged: it cannot be
        distinguished from a concurrent invocation's file, but left silent it would
        scope the downstream stage to an earlier run's ``scan_keys``.

    Raises:
        sleap_roots_contracts.RunManifestMissingError: Called standalone, if a run
            identity is known and no manifest is found for it.
        ValueError: Called standalone, if the run identity is unusable as a filename
            component.
        OSError: If a present manifest cannot be read or forwarded, or ``input_dir`` is
            missing. Deliberately not best-effort: a silently missing forwarded manifest
            sends the downstream stage to unscoped discovery (or, under a run identity,
            to a failure) -- the contamination this function exists to prevent.
    """
    source_dir = Path(input_dir)
    destination_dir = Path(output_dir)
    # Until the read resolves no filename is known, so a failure before then is logged
    # generically rather than naming a file that was never found.
    label = "run manifest"

    tmp: str | None = None
    created: list[Path] = []
    try:
        # The read lives INSIDE this block so that a failure there (EACCES from an NFS
        # ACL misconfiguration, a known run with no manifest) still gets the log naming
        # both directories that the spec requires before anything propagates.
        if read is _UNREAD:
            # allow_legacy=True while any stage may still write the legacy name;
            # flipping it to False is fleet-wide (talmolab/sleap-roots-pipeline#82).
            read = read_run_manifest(
                source_dir, pipeline_run_id_from_env(), allow_legacy=True
            )
        if read is None:
            stale = destination_dir / RUN_MANIFEST_FILENAME
            if stale.is_file():
                logger.warning(
                    "No %s under %s, but %s already holds one from an earlier run; "
                    "leaving it in place -- the downstream stage will be scoped to it",
                    RUN_MANIFEST_FILENAME,
                    source_dir.as_posix(),
                    destination_dir.as_posix(),
                )
            else:
                logger.debug(
                    "No run manifest under %s; nothing to forward",
                    source_dir.as_posix(),
                )
            return

        label = read.filename
        source = source_dir / read.filename
        destination = destination_dir / read.filename

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
        # Record what this call creates, innermost first, so a later failure can undo
        # it. Leaving a newly-created empty output_dir behind is a state that could not
        # occur before this hop existed, and orchestration that reads "no output_dir"
        # as "predict never ran, safe to retry into a clean mount" would misread it.
        probe = destination_dir
        while not probe.exists():
            created.append(probe)
            if probe.parent == probe:
                break
            probe = probe.parent
        destination_dir.mkdir(parents=True, exist_ok=True)
        # A name unique to this writer, inside output_dir. Unique because the
        # destination is shared across concurrent invocations, so a fixed name would
        # let one writer replace another's half-written temp into the final path.
        # Inside output_dir because a system-temp file would make the replace
        # cross-device (EXDEV) on the NFS production mount. Dot-prefixed so an orphan
        # left by a SIGKILL is hidden from any future "run_manifest*" glob.
        fd, tmp = tempfile.mkstemp(
            dir=destination_dir,
            prefix=f".{read.filename}.",
            suffix=".tmp",
        )
        # The fd must be closed before the replace: on Windows an open handle makes
        # os.replace fail with WinError 32, and the write itself would succeed, so the
        # bug would surface only at the replace and only on the Windows leg. Writing
        # through the fd mkstemp already opened avoids reopening the path by name.
        with os.fdopen(fd, "wb") as handle:
            handle.write(read.data)
        # mkstemp creates at 0600 regardless of umask, so without this the forwarded
        # manifest is the one file predict writes that the downstream container -- a
        # different uid on the same shared NFS mount -- cannot read. Before the
        # replace, never after: after would briefly publish the destination at the
        # private temp mode. Largely a no-op on Windows.
        os.chmod(tmp, read.mode)
        os.replace(tmp, destination)
    except Exception:
        # Log BEFORE cleaning up: if the unlink itself raises, logging afterwards would
        # never run and the secondary error would replace the real one.
        logger.error(
            "Failed to forward %s from %s to %s",
            label,
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
        for path in created:
            try:
                path.rmdir()
            except OSError:
                # Non-empty (a concurrent invocation is already writing into it) or
                # already gone. Stop at the first directory that is not ours to
                # remove, rather than reaching further up the tree.
                break
        raise

    logger.info(
        "Forwarded %s from %s to %s",
        read.filename,
        source_dir.as_posix(),
        destination_dir.as_posix(),
    )
