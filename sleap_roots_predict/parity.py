"""A3-predict parity harness: sleap-nn vs. classic-SLEAP inference, same weights.

Ground truth for parity is the real, human-labeled validation split bundled with
each production model's wandb artifact (``labels_gt.val.slp``), not the legacy
pipeline's own predictions. Metrics are computed via ``sleap_nn.evaluation``
(the vendored port of classic SLEAP's own eval code) using its OKS-based
instance *matching* (the library default, permissive at the default
``match_threshold=0.0`` — any spatially plausible pairing counts as a match) —
but this module never reads the OKS-derived *scores* (``mOKS``,
``voc_metrics``). Per ``sleap-roots-training#17``, those scores collapse near
zero on the root-keypoint domain regardless of model quality (uncalibrated
sigma constants), while the raw ``distance_metrics``/``visibility_metrics``
this module does read are unaffected by that miscalibration. Centroid-mode
matching was considered and rejected: it is designed for single-node/
centroid-only predictions, not per-node distance between two full multi-node
skeletons — confirmed empirically to produce nonsensical distances (nonzero
for identical inputs) when tried against a real 2-node skeleton. See
``docs/superpowers/specs/2026-08-03-define-parity-tolerance-design.md`` for
the full design rationale.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import sleap_io as sio
from sleap_nn.evaluation import load_metrics, run_evaluation
from sleap_roots_contracts import LabelCard, ModelCard

logger = logging.getLogger(__name__)

#: Instance-matching method for all parity metric computation. OKS is used for
#: *matching* (deciding which GT/predicted instances correspond), at the
#: maximally permissive default threshold below — this module never reads the
#: OKS-derived *scores* (see module docstring).
MATCH_METHOD = "oks"

#: Deliberately the library's own default (0.0): in OKS mode, any
#: correspondence with OKS > 0 counts as a match, decoupling "which instances
#: correspond" from "how good is the match" — the raw per-node
#: ``distance_metrics`` this module reads are computed on whatever pairing
#: results, independent of the (possibly miscalibrated) OKS score itself.
#: Named as a constant so it's a deliberate, documented choice rather than an
#: implicit library default.
OKS_MATCH_THRESHOLD = 0.0

_GT_FILENAME = "labels_gt.val.slp"
_PR_FILENAME = "labels_pr.val.slp"
_METRICS_FILENAME = "metrics.val.npz"


@dataclass(frozen=True)
class GapRecord:
    """A production ``ModelCard`` whose ground truth could not be resolved."""

    registry_id: str
    version: str
    reason: str


@dataclass(frozen=True)
class ResolvedGroundTruth:
    """Ground truth resolved for one production ``ModelCard``."""

    card: ModelCard
    ground_truth_path: Path
    bundle_dir: Path
    source: str  # "labels_registry" | "relinked_bundle"


@dataclass(frozen=True)
class ParityMetrics:
    """The parity signal for one model: distance + visibility recall, no OKS.

    Deliberately excludes ``mOKS``/``voc_metrics`` — see the module docstring.
    """

    distance_avg: float
    distance_p95: float
    visibility_recall: float
    settings: str  # "recomputed" | "stored"


def relink_ground_truth(
    bundle_dir: Path, prefix_map: dict, out_path: Path
) -> Optional[Path]:
    """Relink a model bundle's ``labels_gt.val.slp`` and verify it loads real pixels.

    Loads ``bundle_dir/labels_gt.val.slp``, rewrites its embedded video paths via
    ``sio.Labels.replace_filenames(prefix_map=...)``, and confirms the first
    labeled frame's image actually loads (proving the relink resolved real
    pixels, not just changed a string). On success the relinked labels are
    saved to ``out_path`` and that path is returned; on any failure (missing
    bundle file, or the first frame still not loadable after relinking) this
    returns ``None`` rather than raising, so callers can treat it as one
    priority-ordered resolution attempt among several.

    Args:
        bundle_dir: A materialized model artifact directory (e.g. from
            ``WandbRegistrySource.materialize``) expected to contain
            ``labels_gt.val.slp``.
        prefix_map: Passed through to ``sio.Labels.replace_filenames``, e.g.
            ``{"D:/SLEAP": "Z:/users/eberrigan/SLEAP"}``.
        out_path: Where to save the relinked labels on success.

    Returns:
        ``out_path`` on success, else ``None``.
    """
    gt_path = bundle_dir / _GT_FILENAME
    if not gt_path.exists():
        return None
    labels = sio.load_slp(gt_path.as_posix())
    labels.replace_filenames(prefix_map=prefix_map)
    try:
        if labels[0].image is None:
            return None
    except Exception as e:  # noqa: BLE001 - any backend failure means "not resolved"
        logger.info("Relink did not resolve real pixels for %s: %s", gt_path, e)
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sio.save_slp(labels, out_path.as_posix())
    return out_path


def resolve_ground_truth(
    card: ModelCard,
    bundle_dir: Path,
    workdir: Path,
    *,
    labels_registry_lookup: Optional[Callable[[ModelCard], Optional[Path]]] = None,
    prefix_map: Optional[dict] = None,
):
    """Resolve real ground truth for one production ``ModelCard``.

    Tries, in order: (1) a matching collection in the ``wandb-registry-sleap-roots-labels``
    registry via ``labels_registry_lookup`` (species/root-type/node-count join,
    injected so this stays testable offline); (2) the model bundle's own
    ``labels_gt.val.slp`` with ``prefix_map`` relinking. Neither resolving is not
    an error — it is recorded as an explicit :class:`GapRecord` so a single
    unresolvable model never aborts resolution for the rest of the batch.

    Args:
        card: The production ``ModelCard`` to resolve ground truth for.
        bundle_dir: The card's materialized model artifact directory.
        workdir: Scratch directory for relinked-labels output files.
        labels_registry_lookup: Optional callable returning a local path to a
            matching labels-registry collection's labels file, or ``None``.
        prefix_map: Optional path-prefix map for bundle relinking (see
            :func:`relink_ground_truth`).

    Returns:
        A :class:`ResolvedGroundTruth` on success, else a :class:`GapRecord`.
    """
    if labels_registry_lookup is not None:
        found = labels_registry_lookup(card)
        if found is not None:
            return ResolvedGroundTruth(
                card=card,
                ground_truth_path=found,
                bundle_dir=bundle_dir,
                source="labels_registry",
            )
    if prefix_map is not None:
        out_path = (
            workdir
            / f"{card.registry_id.replace('/', '_')}.{card.version}.relinked.slp"
        )
        relinked = relink_ground_truth(bundle_dir, prefix_map, out_path)
        if relinked is not None:
            return ResolvedGroundTruth(
                card=card,
                ground_truth_path=relinked,
                bundle_dir=bundle_dir,
                source="relinked_bundle",
            )
    return GapRecord(
        registry_id=card.registry_id,
        version=card.version,
        reason="no ground truth source resolved (labels registry and bundle relinking both failed)",
    )


def compute_metrics(
    ground_truth_path: Path,
    predicted_path: Path,
    *,
    match_threshold: float = OKS_MATCH_THRESHOLD,
) -> ParityMetrics:
    """Compute the parity signal between two label sets.

    Args:
        ground_truth_path: Path to the ground-truth ``.slp``.
        predicted_path: Path to the predicted ``.slp``.
        match_threshold: OKS-mode matching threshold (see
            :data:`OKS_MATCH_THRESHOLD`) — governs which instances are
            considered corresponding, not the reported distance/recall.

    Returns:
        A :class:`ParityMetrics` with ``settings="recomputed"``. OKS-derived
        score fields from the underlying evaluation are not read or exposed.
    """
    metrics = run_evaluation(
        ground_truth_path.as_posix(),
        predicted_path.as_posix(),
        match_method=MATCH_METHOD,
        match_threshold=match_threshold,
    )
    return ParityMetrics(
        distance_avg=float(metrics["distance_metrics"]["avg"]),
        distance_p95=float(metrics["distance_metrics"]["p95"]),
        visibility_recall=float(metrics["visibility_metrics"]["recall"]),
        settings="recomputed",
    )


def reference_metrics(
    bundle_dir: Path,
    ground_truth_path: Path,
    *,
    match_threshold: float = OKS_MATCH_THRESHOLD,
) -> Optional[ParityMetrics]:
    """Get classic-SLEAP's reference parity signal for a resolved model.

    Recomputes via :func:`compute_metrics` against the bundle's
    ``labels_pr.val.slp`` (classic-SLEAP's own predictions) when present, so the
    comparison uses identical settings to the sleap-nn side.

    Falls back to the bundle's stored ``metrics.val.npz`` when
    ``labels_pr.val.slp`` is absent — but that file was pickled by classic
    SLEAP's own (TensorFlow-based) ``sleap`` package, which this repo does not
    and should not depend on. When it cannot be unpickled with only
    ``sleap_nn`` installed (the expected case), this is treated the same as
    "no reference available" — an explicit, logged, non-fatal gap for this
    model's classic-SLEAP comparison, not a crash.

    Args:
        bundle_dir: The model's materialized artifact directory.
        ground_truth_path: The resolved ground-truth path (may be a relinked
            copy, not necessarily ``bundle_dir``'s own file).
        match_threshold: Forwarded to :func:`compute_metrics` when recomputing.

    Returns:
        A :class:`ParityMetrics` with ``settings="recomputed"`` when
        ``labels_pr.val.slp`` was used, ``"stored"`` when the stored
        ``metrics.val.npz`` could be read, or ``None`` when neither is
        available/readable.
    """
    labels_pr = bundle_dir / _PR_FILENAME
    if labels_pr.exists():
        return compute_metrics(
            ground_truth_path, labels_pr, match_threshold=match_threshold
        )
    metrics_path = bundle_dir / _METRICS_FILENAME
    if not metrics_path.exists():
        return None
    try:
        stored = load_metrics(metrics_path.as_posix())
        recall = float(stored["visibility_metrics"]["recall"])
    except Exception as e:  # noqa: BLE001 - e.g. missing legacy `sleap` package
        logger.warning(
            "Could not read stored reference metrics %s: %s", metrics_path, e
        )
        return None
    return ParityMetrics(
        distance_avg=float(stored["distance_metrics"]["avg"]),
        distance_p95=float(stored["distance_metrics"]["p95"]),
        visibility_recall=recall,
        settings="stored",
    )


def build_label_card(
    labels_path: Path,
    card: ModelCard,
    *,
    images_embedded: bool,
    source_experiment: Optional[str] = None,
    bloom_experiment_id: Optional[str] = None,
    accessions: Optional[tuple] = None,
    labeler: Optional[str] = None,
    box_link: Optional[str] = None,
    source_sha256: Optional[str] = None,
) -> LabelCard:
    """Build a ``LabelCard`` record for the ground-truth manifest.

    Content fields (frame/instance/skeleton counts) are derived directly from
    ``labels_path``. Provenance fields the caller cannot determine MUST be left
    at their default (``None``) rather than fabricated — this mirrors
    ``sleap-roots-training#11``'s own stated backfill policy.

    Args:
        labels_path: The resolved ground-truth ``.slp`` to derive content stats
            from.
        card: The production ``ModelCard`` this ground truth backs.
        images_embedded: Whether ``labels_path``'s frames are self-contained
            (``True`` for a labels-registry package, ``True`` for a
            successfully relinked bundle since its pixels are now reachable).
        source_experiment: Provenance, if known.
        bloom_experiment_id: Provenance, if known.
        accessions: Provenance, if known.
        labeler: Provenance, if known.
        box_link: Provenance, if known.
        source_sha256: Provenance, if known.

    Returns:
        A populated ``LabelCard``.
    """
    labels = sio.load_slp(labels_path.as_posix())
    node_names = tuple(n.name for n in labels.skeleton.nodes)
    n_instances = sum(len(lf.instances) for lf in labels)
    n_videos = len(labels.videos)
    return LabelCard(
        species=card.species,
        mode=card.mode,
        root_type=card.root_type,
        age_min=card.age_min,
        age_max=card.age_max,
        skeleton_name=labels.skeleton.name or f"{card.species}_{card.root_type}",
        node_count=len(node_names),
        node_names=node_names,
        n_frames=len(labels),
        n_instances=n_instances,
        n_plants=n_videos,
        n_scans=n_videos,
        images_embedded=images_embedded,
        source_experiment=source_experiment,
        bloom_experiment_id=bloom_experiment_id,
        accessions=accessions,
        labeler=labeler,
        box_link=box_link,
        source_sha256=source_sha256,
        sleap_io_version=sio.__version__,
        registry_id=card.registry_id,
        version=card.version,
    )


def within_tolerance(
    sleap_nn: ParityMetrics,
    classic_sleap: ParityMetrics,
    *,
    distance_tolerance_px: float,
    recall_tolerance: float,
) -> bool:
    """Check whether sleap-nn's metrics are within tolerance of classic-SLEAP's.

    Args:
        sleap_nn: This harness's own computed metrics.
        classic_sleap: The reference metrics (recomputed or stored).
        distance_tolerance_px: Maximum allowed ``|Δ distance_p95|``, in pixels.
        recall_tolerance: Maximum allowed ``|Δ visibility_recall|``.

    Returns:
        ``True`` when both deltas are within their tolerances.
    """
    distance_delta = abs(sleap_nn.distance_p95 - classic_sleap.distance_p95)
    recall_delta = abs(sleap_nn.visibility_recall - classic_sleap.visibility_recall)
    return distance_delta <= distance_tolerance_px and recall_delta <= recall_tolerance
