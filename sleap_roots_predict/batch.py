"""Warm-batch predict runner over a directory of staged scans.

Discovers scans (a ``{scan_key}.scan_metadata.json`` sidecar co-located with its
image frames in a dedicated directory), loads models once via a resident
:class:`~sleap_roots_predict.warm_worker.WarmModelWorker`, predicts each scan,
writes the prediction-output artifacts, and copies the sidecar through so each
``<output_dir>/{scan_key}/`` is a self-contained trait-extractor input tree.

A ``run_manifest.json`` staged in ``input_dir`` is also forwarded to the **top
level** of ``output_dir`` before any scan is predicted, so the downstream
trait-extraction stage — whose input directory *is* this output directory — stays
scoped to the same run instead of falling back to unscoped discovery.
"""

import json
import logging
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from sleap_roots_contracts import (
    RUN_MANIFEST_FILENAME,
    PredictionManifest,
    ResolvedParams,
    RunManifest,
    compute_param_hash,
)
from sleap_roots_contracts.identity import compute_idempotency_key

from sleap_roots_predict.model_registry import ModelCardSource
from sleap_roots_predict.output_contract import (
    predictions_json_path,
    resolve_identity,
    write_prediction_outputs,
)
from sleap_roots_predict.run_manifest import (
    _UNREAD,
    _read_manifest_snapshot,
    copy_run_manifest_forward,
)
from sleap_roots_predict.video_utils import make_video_from_images, natural_sort
from sleap_roots_predict.warm_worker import WarmModelWorker

logger = logging.getLogger(__name__)

_SIDECAR_SUFFIX = ".scan_metadata.json"
_IMAGE_EXTENSIONS = frozenset({".png", ".tif", ".tiff", ".jpg", ".jpeg"})
_REQUIRED_PARAM_KEYS = ("species", "mode", "age")


@dataclass(frozen=True)
class ScanInput:
    """A discovered scan.

    ``error`` is set (and ``params``/``frames`` may be empty) when the sidecar is
    invalid — an isolated per-scan failure, not a batch abort.
    """

    scan_key: str
    sidecar_path: Path
    frames: list[Path] = field(default_factory=list)
    params: ResolvedParams | None = None
    images_checksum: str = ""
    error: str | None = None


@dataclass(frozen=True)
class ScanResult:
    """Per-scan outcome. ``status`` is one of ``ok`` / ``skipped`` / ``failed``."""

    scan_key: str
    status: str
    error: str | None = None


@dataclass
class BatchResult:
    """Aggregate batch outcome."""

    scans: list[ScanResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True iff no scan failed (skipped/ok scans are fine)."""
        return all(s.status != "failed" for s in self.scans)


def discover_scans(input_dir: str | Path, *, manifest_bytes=_UNREAD) -> list[ScanInput]:
    """Discover scans under ``input_dir`` by their scan-metadata sidecars.

    Recursively finds ``*.scan_metadata.json`` files; each sidecar's parent
    directory holds that scan's frames. ``scan_key`` is the sidecar's filename
    stem and must equal the sidecar's internal ``scan_key``. Invalid scans are
    returned with ``.error`` set (isolated failure); a duplicate ``scan_key``
    anywhere in the tree raises.

    If ``input_dir / RUN_MANIFEST_FILENAME`` exists, discovery is scoped to
    exactly its ``scan_keys``: a discovered sidecar outside that set is
    silently excluded, and a listed ``scan_key`` with no matching sidecar is
    returned as an isolated error entry. Absent a manifest, every sidecar found
    is returned (unscoped), unchanged from before manifest-awareness existed.

    Args:
        input_dir: Directory of staged scans (must exist).
        manifest_bytes: Package-internal. The raw bytes of a ``run_manifest.json``
            already read from ``input_dir``, or ``None`` to assert it was already
            found absent. Omit it and the manifest is read here. ``run_batch`` passes
            one snapshot so that discovery scopes against exactly the bytes the
            forward-copy publishes.

    Returns:
        One :class:`ScanInput` per discovered (or manifest-expected-but-missing)
        sidecar, sorted by path.

    Raises:
        FileNotFoundError: If ``input_dir`` does not exist (a mis-configured mount,
            distinct from an empty-but-present directory which is a no-op).
        ValueError: If two in-scope sidecars share a ``scan_key``.
        pydantic.ValidationError: If a present ``run_manifest.json`` fails to parse
            or validate as a :class:`RunManifest`.
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(
            f"input scan directory does not exist: {input_dir.as_posix()}"
        )

    scoped_keys: set[str] | None = None
    if manifest_bytes is _UNREAD:
        manifest_path = input_dir / RUN_MANIFEST_FILENAME
        # read_bytes, not read_text: model_validate_json accepts bytes, and decoding
        # with the platform locale would reject a manifest on the Windows leg that the
        # sibling stages -- which read these same bytes as UTF-8 -- accept.
        manifest_bytes = manifest_path.read_bytes() if manifest_path.is_file() else None
    if manifest_bytes is not None:
        manifest = RunManifest.model_validate_json(manifest_bytes)
        scoped_keys = set(manifest.scan_keys)

    scans: list[ScanInput] = []
    seen: dict[str, Path] = {}
    excluded: list[str] = []
    for sidecar in sorted(input_dir.rglob("*" + _SIDECAR_SUFFIX)):
        scan_key = sidecar.name[: -len(_SIDECAR_SUFFIX)]
        if scoped_keys is not None and scan_key not in scoped_keys:
            excluded.append(scan_key)
            continue
        if scan_key in seen:
            raise ValueError(
                f"duplicate scan_key {scan_key!r}: "
                f"{seen[scan_key].as_posix()} and {sidecar.as_posix()}"
            )
        seen[scan_key] = sidecar
        scans.append(_load_scan(sidecar, scan_key))

    if excluded:
        logger.debug(
            "Excluded %d sidecar(s) outside run_manifest.json scope: %s",
            len(excluded),
            sorted(excluded),
        )

    if scoped_keys is not None:
        for key in sorted(scoped_keys - set(seen)):
            scans.append(
                ScanInput(
                    key,
                    input_dir / f"{key}{_SIDECAR_SUFFIX}",
                    error=f"no sidecar found for manifest scan_key {key!r}",
                )
            )

    return scans


def _load_scan(sidecar: Path, scan_key: str) -> ScanInput:
    """Parse one sidecar into a ScanInput (with ``.error`` set if invalid)."""
    try:
        meta = json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return ScanInput(scan_key, sidecar, error=f"unreadable sidecar: {exc}")
    if meta.get("scan_key") != scan_key:
        return ScanInput(
            scan_key,
            sidecar,
            error=(
                f"sidecar scan_key {meta.get('scan_key')!r} != filename stem "
                f"{scan_key!r}"
            ),
        )
    params = meta.get("params")
    if not isinstance(params, dict) or any(
        k not in params for k in _REQUIRED_PARAM_KEYS
    ):
        return ScanInput(
            scan_key, sidecar, error=f"sidecar params missing/incomplete: {params!r}"
        )
    # Frames are the co-located images, natural-sorted so frame_2 precedes frame_10
    # (the frame order is the temporal order of the inference video). Non-image files
    # (including the sidecar itself) and subdirectories are excluded. natural_sort
    # returns strings, so map back to Path to keep the list[Path] contract.
    frames = [
        Path(s)
        for s in natural_sort(
            [
                p
                for p in sidecar.parent.iterdir()
                if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
            ]
        )
    ]
    resolved = ResolvedParams(values={k: params[k] for k in _REQUIRED_PARAM_KEYS})
    images_checksum = meta.get("images_checksum", "")
    return ScanInput(
        scan_key,
        sidecar,
        frames=frames,
        params=resolved,
        images_checksum=images_checksum,
    )


def _identity_key(
    *,
    scan_key: str,
    images_checksum: str,
    params_dict: dict,
    model_refs,
    predict_code_sha: str,
    predict_output_params: dict,
) -> str:
    """Derive the predict-scoped identity key for one scan's current or prior state.

    ``model_refs`` may be a ``dict[RootType, ModelRef]`` (as returned by
    ``worker.resolve``) or any iterable of ``ModelRef`` (as read back from a
    ``PredictionManifest``'s ``artifacts``) — both are reduced to the same
    order-independent ``(registry_id, version, weights_checksum)`` tuples
    ``compute_idempotency_key`` expects. ``traits_code_sha`` is a fixed empty-string
    placeholder: predict never owns that value and only ever compares this key
    against its own previously-derived one, never against a traits-computed key.
    """
    refs = model_refs.values() if isinstance(model_refs, dict) else model_refs
    models = [(ref.registry_id, ref.version, ref.weights_checksum) for ref in refs]
    return compute_idempotency_key(
        scan_key=scan_key,
        images_checksum=images_checksum,
        models=models,
        param_hash=compute_param_hash(params_dict),
        predict_code_sha=predict_code_sha,
        traits_code_sha="",
        predict_output_params=predict_output_params,
    )


def _previous_identity_key(out_scan_dir: Path, scan_key: str) -> str | None:
    """Recompute the previous run's identity key from artifacts already on disk.

    Reads back the already-copied sidecar and already-written prediction manifest
    from ``out_scan_dir`` — no new storage is needed. Returns ``None`` if either
    file is missing, unreadable, or present but fails to parse/validate (a
    :class:`pydantic.ValidationError`, which subclasses ``ValueError`` and is
    caught here without importing pydantic directly) — a corrupt or absent
    *previous* state is treated as "changed," never as a failure.
    """
    sidecar_path = out_scan_dir / f"{scan_key}{_SIDECAR_SUFFIX}"
    manifest_path = predictions_json_path(out_scan_dir, scan_key)
    try:
        meta = json.loads(sidecar_path.read_text())
        manifest = PredictionManifest.model_validate_json(manifest_path.read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    params = meta.get("params")
    if not isinstance(params, dict):
        return None
    return _identity_key(
        scan_key=scan_key,
        images_checksum=meta.get("images_checksum", ""),
        params_dict={k: params[k] for k in _REQUIRED_PARAM_KEYS if k in params},
        model_refs=[artifact.model for artifact in manifest.artifacts],
        predict_code_sha=manifest.predict_code_sha,
        predict_output_params=manifest.predict_output_params,
    )


def run_batch(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    source: ModelCardSource | None = None,
    predict_code_sha: str | None = None,
    predict_container_digest: str | None = None,
    should_stop: Callable[[], bool] = lambda: False,
) -> BatchResult:
    """Predict every scan under ``input_dir``, writing outputs under ``output_dir``.

    Loads models once via a single resident worker. Per scan: a scan already
    recorded as invalid (``scan.error`` set) is isolated as ``failed`` immediately,
    before any model resolution runs. Otherwise, the currently-resolved models plus
    the scan's current inputs are compared, via a recomputed idempotency key,
    against the key recomputed from that scan's own previously-written artifacts
    (no new storage — see :func:`_previous_identity_key`); an exact match skips,
    anything else (including no previous artifacts at all) predicts and overwrites.
    A per-scan error is isolated (recorded ``failed``, batch continues). An empty
    (but present) input directory is a batch-level staging error (raises), not a
    no-op — a misconfigured or empty stage-in mount should never look like success.

    Args:
        input_dir: Directory of staged scans.
        output_dir: Directory to write per-scan outputs into.
        source: Model-card source; ``None`` uses the production WandbRegistrySource.
        predict_code_sha: Provenance sha (falls back to ``SRP_PREDICT_CODE_SHA``).
        predict_container_digest: Provenance digest (env fallback).
        should_stop: Checked at the top of each per-scan loop iteration; when it
            returns ``True`` the batch stops before starting the next scan (never
            interrupting a scan already in progress). Defaults to a no-op, so
            existing callers are unaffected.

    Returns:
        A :class:`BatchResult` with one :class:`ScanResult` per scan.

    Raises:
        FileNotFoundError: If ``input_dir`` does not exist.
        ValueError: If two sidecars share a ``scan_key``, or if zero scans are
            discovered under a present ``input_dir`` (both batch-level staging
            errors, surfaced before any prediction or model-source interaction).
        OSError: If a ``run_manifest.json`` staged in ``input_dir`` cannot be
            forwarded to ``output_dir``. Also a batch-level staging error, raised
            before any prediction: a silently missing forwarded manifest would make
            the downstream stage fall back to unscoped discovery.
        NoReadableModelCardsError: (a ``ValueError``) If the registry holds
            production artifacts none of which validates. Raised, along with
            ``RuntimeError`` for missing registry credentials, before the first
            processable scan is predicted.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    # ONE read of run_manifest.json for the whole batch. Discovery validates exactly
    # these bytes and the forward-copy publishes exactly these bytes, so the two can
    # never disagree about the run's scope. Re-reading the path for the copy left a
    # window in which a concurrent upstream writer (bloomctl unions scan_keys into the
    # shared input manifest) could widen it -- forwarding a scope predict never
    # predicted -- or remove it, making the forward a silent no-op that reinstates #39.
    snapshot = _read_manifest_snapshot(input_dir)
    scans = discover_scans(
        input_dir, manifest_bytes=None if snapshot is None else snapshot.data
    )
    if not scans:
        raise ValueError(f"no scans discovered under {input_dir.as_posix()}")
    # Forward BEFORE the loop and before any model-source interaction: discovery has
    # already validated these exact bytes, so a corrupt manifest cannot reach the
    # shared output tree; a failure here costs no prediction work; and a batch stopped
    # early (Argo preemption) still hands the downstream stage a scoped tree.
    # Deliberately not wrapped in try/except -- see the change's design.md.
    copy_run_manifest_forward(input_dir, output_dir, snapshot=snapshot)
    result = BatchResult()

    resolved_code_sha = resolve_identity(predict_code_sha, "SRP_PREDICT_CODE_SHA")
    worker = WarmModelWorker(source=source)
    catalog_loaded = False
    for scan in scans:
        if should_stop():
            logger.warning(
                "Stopping early: requested after %d/%d scans",
                len(result.scans),
                len(scans),
            )
            break
        if scan.error is not None:
            logger.error("Scan %s failed: %s", scan.scan_key, scan.error)
            result.scans.append(ScanResult(scan.scan_key, "failed", scan.error))
            continue

        if not catalog_loaded:
            # Once, outside per-scan isolation: an unreadable or unreachable catalog is
            # a batch-level error (exit 1), not one isolated failure per scan (exit 3,
            # which the pipeline's exit gate passes). Placed after this iteration's stop
            # check so it adds no should_stop() call, and skipped entirely when no scan
            # is processable.
            worker.load_catalog()
            catalog_loaded = True

        out_scan_dir = output_dir / scan.scan_key
        try:
            refs = worker.resolve(scan.params)
            current_key = _identity_key(
                scan_key=scan.scan_key,
                images_checksum=scan.images_checksum,
                params_dict=scan.params.values,
                model_refs=refs,
                predict_code_sha=resolved_code_sha,
                predict_output_params=worker.output_params(),
            )
            previous_key = _previous_identity_key(out_scan_dir, scan.scan_key)
            if previous_key is not None and previous_key == current_key:
                logger.info("Skipping %s (idempotency key unchanged)", scan.scan_key)
                result.scans.append(ScanResult(scan.scan_key, "skipped"))
                continue
            _predict_one(
                worker,
                scan,
                out_scan_dir,
                refs,
                predict_code_sha,
                predict_container_digest,
            )
            result.scans.append(ScanResult(scan.scan_key, "ok"))
        except Exception as exc:  # noqa: BLE001 - isolate per-scan failures
            logger.exception("Scan %s failed", scan.scan_key)
            result.scans.append(ScanResult(scan.scan_key, "failed", str(exc)))
    return result


def _predict_one(
    worker: WarmModelWorker,
    scan: ScanInput,
    out_scan_dir: Path,
    refs: dict,
    predict_code_sha: str | None,
    predict_container_digest: str | None,
) -> None:
    """Predict one scan and write its outputs + copied sidecar. Raises on failure."""
    if not scan.frames:
        raise ValueError(
            f"no image frames co-located with sidecar {scan.sidecar_path.as_posix()}"
        )
    assert scan.params is not None  # run_batch filters error scans (params-None) first
    if not refs:
        # A scan matching no model for any root type is a hard per-scan failure rather
        # than an empty-artifacts manifest: write_prediction_outputs permits an empty
        # manifest, but the downstream trait-extractor rejects one, so surface it here.
        raise ValueError(f"no models resolved for params {scan.params.values!r}")
    video = make_video_from_images(scan.frames, greyscale=True)
    labels = worker.predict(scan.params, video)
    out_scan_dir.mkdir(parents=True, exist_ok=True)
    # Copy the sidecar BEFORE the manifest: write_prediction_outputs writes the manifest
    # last as the resume commit-marker, so the sidecar must already be present when it
    # lands. This isn't required to protect predict's own resume logic -- a missing
    # sidecar already makes _previous_identity_key's read raise OSError, caught and
    # treated as "changed" (re-predict), never a silent skip. It's required because the
    # trait-extractor expects a self-contained input tree (manifest + sidecar + .slp) and
    # would reject a manifest with no sidecar. The copy itself is atomic (temp file +
    # os.replace), so no reader ever observes a partially-written sidecar.
    # Note for test authors: tests/test_batch.py's sidecar-atomicity tests patch the
    # *global* os.replace, which the run-manifest forward-copy that now runs first in
    # run_batch also calls. (Its shutil.copyfile no longer collides -- the forward
    # writes its bytes through the fd mkstemp opened.) They stay pointed at this copy
    # only because tests/assets/scans/ stages no run_manifest.json -- never add one.
    sidecar_dst = out_scan_dir / f"{scan.scan_key}{_SIDECAR_SUFFIX}"
    tmp_sidecar_dst = sidecar_dst.with_name(sidecar_dst.name + ".tmp")
    try:
        shutil.copyfile(scan.sidecar_path, tmp_sidecar_dst)
        os.replace(tmp_sidecar_dst, sidecar_dst)
    except Exception:
        tmp_sidecar_dst.unlink(missing_ok=True)
        raise
    write_prediction_outputs(
        labels,
        refs,
        out_scan_dir,
        scan_key=scan.scan_key,
        inference_config=worker.inference_config(),
        output_params=worker.output_params(),
        predict_code_sha=predict_code_sha,
        predict_container_digest=predict_container_digest,
    )
