"""Predict output contract: the on-disk artifacts predict writes per scan.

For each scan the writer emits one named ``.slp`` per predicted root type
(``{scan_key}.model{model_id}.root{root_type}.slp`` — the sleap-roots ``Series``
naming convention) plus a single combined ``{scan_key}.predictions.json`` that
serializes a :class:`PredictionManifest`: the manifest (per-root paths +
``model_id`` + ``plant_qr_code``) and the predict-side provenance (resolved
``ModelRef``s, effective inference config, code sha / container digest, and each
``.slp``'s checksum + file size).

``PredictionArtifact``/``PredictionManifest`` are defined by
``sleap-roots-contracts``' ``prediction-manifest-contract`` capability and
imported here, not defined locally. Path strings are emitted via
``Path.as_posix()`` (lab convention) so the manifest stays portable across
POSIX and Windows.
"""

import hashlib
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import sleap_io as sio
from sleap_roots_contracts import (
    ModelRef,
    PredictionArtifact,
    PredictionManifest,
    ResolvedParams,
)

if TYPE_CHECKING:
    from sleap_roots_predict.warm_worker import WarmModelWorker


# --- filename helpers -------------------------------------------------------

_SLUG_UNSAFE = re.compile(r"[^A-Za-z0-9-]")

# Characters that break a single path segment on POSIX and/or Windows, plus `.`
# (which would break the ``series_name = filename.split(".")[0]`` invariant that
# sleap-roots relies on).
_SCAN_KEY_FORBIDDEN = frozenset('./\\:*?"<>|')


def slugify_model_id(ref: ModelRef) -> str:
    """Return a filename-safe, dot/slash-free slug for a model ref.

    Combines ``registry_id`` and ``version`` and replaces every character outside
    ``[A-Za-z0-9-]`` with ``-`` (the ``registry_id`` is a slash path, so it is not
    itself a valid filename). The slug is a discovery label; the full identity is
    recorded as the artifact's ``ModelRef``.

    Args:
        ref: The resolved model reference.

    Returns:
        A slug safe to embed in a ``.slp`` filename.
    """
    return _SLUG_UNSAFE.sub("-", f"{ref.registry_id}_{ref.version}")


def _validate_scan_key(scan_key: str) -> None:
    r"""Raise ``ValueError`` if ``scan_key`` is unsafe as a single path segment.

    ``scan_key`` is identity and must not be mangled, so an empty value or one
    containing ``. / \ : * ? " < > |`` or a control character is rejected rather
    than rewritten.

    Args:
        scan_key: The producer-side scan identifier.

    Raises:
        ValueError: If ``scan_key`` is empty or contains a reserved character.
    """
    if not scan_key:
        raise ValueError("scan_key must be a non-empty string")
    if scan_key != scan_key.strip():
        raise ValueError(
            f"scan_key {scan_key!r} has leading/trailing whitespace; it must be a "
            "clean single path segment (Windows trims trailing spaces/dots)"
        )
    bad = {c for c in scan_key if c in _SCAN_KEY_FORBIDDEN or ord(c) < 32}
    if bad:
        raise ValueError(
            f"scan_key {scan_key!r} contains reserved character(s) {sorted(bad)!r}; "
            "it must be safe as a single path segment"
        )


def _resolve_identity(explicit: str | None, env_var: str) -> str:
    """Return the explicit value, else the environment value, else ``""``."""
    if explicit is not None:
        return explicit
    return os.environ.get(env_var, "")


# --- writer -----------------------------------------------------------------


def write_prediction_outputs(
    labels_by_root: dict[str, "sio.Labels"],
    refs_by_root: dict[str, ModelRef],
    out_dir: str | Path,
    *,
    scan_key: str,
    plant_qr_code: str | None = None,
    inference_config: dict[str, Any],
    output_params: dict[str, Any],
    predict_code_sha: str | None = None,
    predict_container_digest: str | None = None,
) -> PredictionManifest:
    """Write the per-scan output contract into ``out_dir``.

    For each resolved root type this writes
    ``{scan_key}.model{model_id}.root{root_type}.slp`` (the sleap-roots ``Series``
    filename convention; loaded downstream via ``Series.load`` with the manifest's
    explicit paths) and then a single combined ``{scan_key}.predictions.json``
    serializing a :class:`PredictionManifest`. Re-running for the same ``scan_key``
    into the same ``out_dir`` overwrites the prior outputs in place, first removing
    any stale ``.slp`` for that scan (so a changed model slug does not orphan files).
    Path strings are emitted via ``Path.as_posix()``. Does not import ``sleap-roots``.

    Args:
        labels_by_root: Predicted ``sio.Labels`` per root type (from the worker's
            ``predict``).
        refs_by_root: The resolved ``ModelRef`` per root type (from the worker's
            ``resolve``); must cover the same root types as ``labels_by_root``.
        out_dir: Directory to write into (created if missing).
        scan_key: Producer-side scan identifier and ``.slp`` filename stem; must be
            safe as a single path segment.
        plant_qr_code: Cross-scan plant key; defaults to ``scan_key`` when ``None``.
        inference_config: Full effective inference config (``worker.inference_config()``).
        output_params: Output-defining subset (``worker.output_params()``).
        predict_code_sha: Explicit predict git sha; falls back to
            ``SRP_PREDICT_CODE_SHA`` then ``""``.
        predict_container_digest: Explicit container digest; falls back to
            ``SRP_PREDICT_CONTAINER_DIGEST`` then ``""``.

    Returns:
        The written :class:`PredictionManifest`.

    Raises:
        ValueError: If ``scan_key`` is unsafe, or ``labels_by_root`` and
            ``refs_by_root`` cover different root types.
    """
    _validate_scan_key(scan_key)
    if set(labels_by_root) != set(refs_by_root):
        raise ValueError(
            f"labels_by_root root types {sorted(labels_by_root)} do not match "
            f"refs_by_root root types {sorted(refs_by_root)}"
        )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Idempotent re-run: remove this scan's prior `.slp` artifacts first. Their
    # filenames embed the model slug, so a model/version/override change would
    # otherwise orphan the old files (unreferenced by the new manifest, yet matched
    # by glob-based consumers). Scoped by the validated (separator-free) scan_key
    # prefix, so other scans in the same directory are untouched.
    slp_prefix = f"{scan_key}.model"
    for stale in list(out.iterdir()):
        if stale.name.startswith(slp_prefix) and stale.name.endswith(".slp"):
            stale.unlink()

    artifacts: list[PredictionArtifact] = []
    for root_type in sorted(refs_by_root):
        ref = refs_by_root[root_type]
        slug = slugify_model_id(ref)
        filename = f"{scan_key}.model{slug}.root{root_type}.slp"
        slp_path = out / filename
        sio.save_file(labels_by_root[root_type], slp_path.as_posix())
        data = slp_path.read_bytes()
        artifacts.append(
            PredictionArtifact(
                root_type=root_type,
                model_id=slug,
                model=ref,
                slp_path=Path(filename).as_posix(),
                checksum=hashlib.sha256(data).hexdigest(),
                file_size=len(data),
            )
        )

    manifest = PredictionManifest(
        scan_key=scan_key,
        plant_qr_code=plant_qr_code or scan_key,
        artifacts=artifacts,
        predict_inference_config=dict(inference_config),
        predict_output_params=dict(output_params),
        predict_code_sha=_resolve_identity(predict_code_sha, "SRP_PREDICT_CODE_SHA"),
        predict_container_digest=_resolve_identity(
            predict_container_digest, "SRP_PREDICT_CONTAINER_DIGEST"
        ),
    )
    (out / f"{scan_key}.predictions.json").write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )
    return manifest


# --- batch ------------------------------------------------------------------


@dataclass(frozen=True)
class ScanRequest:
    """One scan's inputs for :func:`predict_and_write_batch`.

    Attributes:
        scan_key: Producer-side scan identifier (and ``.slp`` filename stem).
        video: The ``sio.Video`` to predict on.
        params: Resolved scan params (species/mode/age).
        plant_qr_code: Optional cross-scan plant key (defaults to ``scan_key``).
        overrides: Optional explicit ``ModelRef`` per root type.
    """

    scan_key: str
    video: "sio.Video"
    params: ResolvedParams
    plant_qr_code: str | None = None
    overrides: dict[str, ModelRef] | None = None


def predict_and_write_batch(
    worker: "WarmModelWorker",
    requests: "Iterable[ScanRequest]",
    out_dir: str | Path,
    *,
    predict_code_sha: str | None = None,
    predict_container_digest: str | None = None,
) -> list[PredictionManifest]:
    """Drive one warm worker over N scans, writing one subdirectory per scan.

    The worker's resident ``Predictor``s are reused across scans (models loaded
    once), so a batch amortizes model-load cost. Each scan is written into
    ``out_dir/{scan_key}/`` via :func:`write_prediction_outputs`.

    Args:
        worker: A ``WarmModelWorker`` kept resident across the batch.
        requests: The scans to predict and write.
        out_dir: Parent directory; each scan gets an ``out_dir/{scan_key}/`` subdir.
        predict_code_sha: Passed through to each scan's manifest.
        predict_container_digest: Passed through to each scan's manifest.

    Returns:
        One :class:`PredictionManifest` per scan, in request order.

    Raises:
        ValueError: If two requests share a ``scan_key`` (which would otherwise
            silently overwrite a scan's subdirectory).
    """
    out = Path(out_dir)
    reqs = list(requests)
    keys = [r.scan_key for r in reqs]
    dups = sorted({k for k in keys if keys.count(k) > 1})
    if dups:
        raise ValueError(
            f"duplicate scan_key(s) in batch: {dups}; each scan must be unique "
            "(a repeat would silently overwrite an earlier scan's subdirectory)"
        )
    inference_config = worker.inference_config()
    output_params = worker.output_params()
    manifests: list[PredictionManifest] = []
    for req in reqs:
        refs = worker.resolve(req.params, req.overrides)
        labels = worker.predict(req.params, req.video, overrides=req.overrides)
        manifests.append(
            write_prediction_outputs(
                labels,
                refs,
                out / req.scan_key,
                scan_key=req.scan_key,
                plant_qr_code=req.plant_qr_code,
                inference_config=inference_config,
                output_params=output_params,
                predict_code_sha=predict_code_sha,
                predict_container_digest=predict_container_digest,
            )
        )
    return manifests
