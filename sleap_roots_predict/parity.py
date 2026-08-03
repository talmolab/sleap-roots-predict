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

import importlib.util
import logging
import os
import re
import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
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

#: Matches a day/age hint embedded in a lab folder name, e.g. "Day10_..." or
#: "3_do"/"3do" ("do" = "days old"). Used to disambiguate basename-search
#: candidates by which one falls inside a ModelCard's age range.
_AGE_HINT_RE = re.compile(r"day(\d+)|(\d+)[\s_]*do\b", re.IGNORECASE)


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
    source: str  # "labels_registry" | "relinked_bundle" | "basename_search"
    n_frames_resolved: int
    n_frames_total: int


@dataclass(frozen=True)
class ParityMetrics:
    """The parity signal for one model: distance + visibility recall, no OKS.

    Deliberately excludes ``mOKS``/``voc_metrics`` — see the module docstring.
    """

    distance_avg: float
    distance_p95: float
    visibility_recall: float
    settings: str  # "recomputed" | "stored"


def _filter_to_loadable_frames(labels: sio.Labels) -> list:
    """Keep only the labeled frames whose video actually loads real pixels."""
    kept = []
    for lf in labels:
        try:
            if lf.image is not None:
                kept.append(lf)
        except Exception as e:  # noqa: BLE001 - unresolved/unopenable video, skip frame
            logger.info("Frame %s not loadable: %s", lf, e)
    return kept


def relink_ground_truth(bundle_dir: Path, prefix_map: dict, out_path: Path):
    """Relink a model bundle's ``labels_gt.val.slp`` via a path-prefix substitution.

    Loads ``bundle_dir/labels_gt.val.slp``, rewrites its embedded video paths via
    ``sio.Labels.replace_filenames(prefix_map=...)``, and keeps only the
    labeled frames whose video actually loads real pixels afterward — a
    prefix map that fixes most, but not all, of a bundle's video paths (seen
    in practice: a handful of stray videos under a different prefix mixed
    into an otherwise-uniform bundle) still yields a real, if smaller, ground
    truth rather than being silently treated as fully resolved or a total
    gap. Returns ``None`` (not a raise) on any failure, including a missing
    bundle file or zero frames resolving, so callers can treat this as one
    priority-ordered resolution attempt among several.

    Args:
        bundle_dir: A materialized model artifact directory (e.g. from
            ``WandbRegistrySource.materialize``) expected to contain
            ``labels_gt.val.slp``.
        prefix_map: Passed through to ``sio.Labels.replace_filenames``, e.g.
            ``{"D:/SLEAP": "Z:/users/eberrigan/SLEAP"}``.
        out_path: Where to save the relinked, filtered labels on success.

    Returns:
        ``(out_path, n_frames_resolved, n_frames_total)`` when at least one
        frame resolves, else ``None``.
    """
    gt_path = bundle_dir / _GT_FILENAME
    if not gt_path.exists():
        return None
    labels = sio.load_slp(gt_path.as_posix())
    n_total = len(labels)
    labels.replace_filenames(prefix_map=prefix_map)
    kept = _filter_to_loadable_frames(labels)
    if not kept:
        return None
    filtered = sio.Labels(
        labeled_frames=kept, videos=labels.videos, skeletons=labels.skeletons
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sio.save_slp(filtered, out_path.as_posix())
    return out_path, len(kept), n_total


def build_basename_index(search_root: Path) -> dict:
    """Index every ``.h5`` file under ``search_root`` by lowercase basename.

    Building this walks the whole tree — O(files under ``search_root``), a
    real cost on a large network share. Build it **once per harness run** and
    reuse it across every model's resolution, not once per model.

    Args:
        search_root: Directory tree to index (e.g. a species-specific SLEAP
            project root on the network share).

    Returns:
        ``{lowercase_basename: [full_path, ...]}`` — usually one path per
        basename, but the same short ID can recur across day/timepoint
        folders in a longitudinal study (see :func:`_pick_best_candidate`).
    """
    index: dict = {}
    for dirpath, _, filenames in os.walk(search_root):
        for fn in filenames:
            if fn.lower().endswith(".h5"):
                index.setdefault(fn.lower(), []).append(os.path.join(dirpath, fn))
    return index


def _age_hint(path: str) -> Optional[int]:
    """Extract a day/age number from a folder name, e.g. ``Day10`` or ``3_do`` -> 10, 3."""
    m = _AGE_HINT_RE.search(path)
    if not m:
        return None
    return int(m.group(1) or m.group(2))


def _pick_best_candidate(
    broken_path: str, candidates: list, card: ModelCard
) -> Optional[str]:
    """Disambiguate multiple same-basename candidates for one broken video path.

    The same short plant/scan ID can recur across day/timepoint folders in a
    longitudinal study (confirmed by comparing file content: same basename,
    different bytes, different day folder) — so basename alone is not enough.
    Tried in order, each only when the previous step leaves more than one
    candidate: (1) a single candidate is unambiguous by definition; (2) an
    exact match on the immediate parent folder name (case/punctuation
    normalized) — the day/date/batch is usually encoded there; (3) among
    remaining candidates, one whose path contains a day/age hint (see
    :func:`_age_hint`) inside the card's ``[age_min, age_max]``; (4) the
    candidate(s) whose path shares the most normalized path segments with the
    broken path. Returns ``None`` (an explicit non-match, not a guess) if a
    step still leaves more than one candidate tied.
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    def normalize(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", s.lower())

    def normalize_segment(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

    def segments(path: str) -> list:
        return [s for s in re.split(r"[\\/]", path)[:-1] if s]

    broken_parent = normalize(Path(broken_path).parent.name)
    parent_matches = [
        c for c in candidates if normalize(Path(c).parent.name) == broken_parent
    ]
    if len(parent_matches) == 1:
        return parent_matches[0]

    pool = parent_matches if parent_matches else candidates
    age_matches = [
        c
        for c in pool
        if (hint := _age_hint(c)) is not None and card.age_min <= hint <= card.age_max
    ]
    if len(age_matches) == 1:
        return age_matches[0]

    pool = age_matches if age_matches else pool
    broken_segments = {normalize_segment(s) for s in segments(broken_path)}
    scored = sorted(
        (
            (sum(1 for s in segments(c) if normalize_segment(s) in broken_segments), c)
            for c in pool
        ),
        key=lambda pair: -pair[0],
    )
    top_score = scored[0][0]
    winners = [c for score, c in scored if score == top_score]
    return winners[0] if len(winners) == 1 else None


def relink_ground_truth_by_basename_search(
    bundle_dir: Path,
    basename_index: dict,
    card: ModelCard,
    out_path: Path,
):
    """Relink a bundle's ground truth by basename search, partially if needed.

    Unlike :func:`relink_ground_truth` (a single prefix substitution, all
    frames or none), this searches ``basename_index`` per video and keeps
    only the labeled frames whose video resolves — a model with some
    unresolvable frames still gets a real (smaller) ground-truth set instead
    of being treated as a total gap. Use when :func:`relink_ground_truth`
    doesn't apply (the bundle's video paths were reorganized, not just moved
    under a new drive letter/root).

    Args:
        bundle_dir: A materialized model artifact directory expected to
            contain ``labels_gt.val.slp``.
        basename_index: From :func:`build_basename_index`, built once and
            reused across models.
        card: The production ``ModelCard`` (used for age-range disambiguation
            in :func:`_pick_best_candidate`).
        out_path: Where to save the filtered, relinked labels on success.

    Returns:
        ``(out_path, n_frames_resolved, n_frames_total)`` when at least one
        frame resolves, else ``None``.
    """
    gt_path = bundle_dir / _GT_FILENAME
    if not gt_path.exists():
        return None
    labels = sio.load_slp(gt_path.as_posix())
    n_total = len(labels)

    filename_map = {}
    for video in labels.videos:
        fn = getattr(video, "filename", None)
        if not isinstance(fn, str):
            continue
        candidates = basename_index.get(Path(fn).name.lower(), [])
        winner = _pick_best_candidate(fn, candidates, card)
        if winner is not None:
            filename_map[fn] = winner
    if not filename_map:
        return None

    labels.replace_filenames(filename_map=filename_map)
    kept = _filter_to_loadable_frames(labels)
    if not kept:
        return None

    filtered = sio.Labels(
        labeled_frames=kept, videos=labels.videos, skeletons=labels.skeletons
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sio.save_slp(filtered, out_path.as_posix())
    return out_path, len(kept), n_total


def resolve_ground_truth(
    card: ModelCard,
    bundle_dir: Path,
    workdir: Path,
    *,
    labels_registry_lookup: Optional[Callable[[ModelCard], Optional[Path]]] = None,
    prefix_map: Optional[dict] = None,
    basename_index: Optional[dict] = None,
):
    """Resolve real ground truth for one production ``ModelCard``.

    Tries, in order: (1) a matching collection in the
    ``wandb-registry-sleap-roots-labels`` registry via `labels_registry_lookup`
    (species/root-type/node-count join, injected so this stays testable
    offline) — full coverage, since those collections are self-contained;
    (2) the model bundle's own ``labels_gt.val.slp`` with `prefix_map`
    relinking (see :func:`relink_ground_truth`) — full or partial coverage;
    (3) `basename_index` search with disambiguation (see
    :func:`relink_ground_truth_by_basename_search`) for bundles whose video
    paths were reorganized, not just moved under a new drive letter/root —
    typically partial coverage. None resolving is not an error — it is
    recorded as an explicit :class:`GapRecord` so a single unresolvable model
    never aborts resolution for the rest of the batch.

    Args:
        card: The production ``ModelCard`` to resolve ground truth for.
        bundle_dir: The card's materialized model artifact directory.
        workdir: Scratch directory for relinked-labels output files.
        labels_registry_lookup: Optional callable returning a local path to a
            matching labels-registry collection's labels file, or ``None``.
        prefix_map: Optional path-prefix map for bundle relinking (see
            :func:`relink_ground_truth`).
        basename_index: Optional basename index (see
            :func:`build_basename_index`) for the basename-search fallback.

    Returns:
        A :class:`ResolvedGroundTruth` on success, else a :class:`GapRecord`.
    """
    if labels_registry_lookup is not None:
        found = labels_registry_lookup(card)
        if found is not None:
            n_frames = len(sio.load_slp(found.as_posix()))
            return ResolvedGroundTruth(
                card=card,
                ground_truth_path=found,
                bundle_dir=bundle_dir,
                source="labels_registry",
                n_frames_resolved=n_frames,
                n_frames_total=n_frames,
            )

    safe_id = card.registry_id.replace("/", "_")
    if prefix_map is not None:
        out_path = workdir / f"{safe_id}.{card.version}.relinked.slp"
        result = relink_ground_truth(bundle_dir, prefix_map, out_path)
        if result is not None:
            path, n_resolved, n_total = result
            return ResolvedGroundTruth(
                card=card,
                ground_truth_path=path,
                bundle_dir=bundle_dir,
                source="relinked_bundle",
                n_frames_resolved=n_resolved,
                n_frames_total=n_total,
            )

    if basename_index is not None:
        out_path = workdir / f"{safe_id}.{card.version}.basename_search.slp"
        result = relink_ground_truth_by_basename_search(
            bundle_dir, basename_index, card, out_path
        )
        if result is not None:
            path, n_resolved, n_total = result
            return ResolvedGroundTruth(
                card=card,
                ground_truth_path=path,
                bundle_dir=bundle_dir,
                source="basename_search",
                n_frames_resolved=n_resolved,
                n_frames_total=n_total,
            )

    return GapRecord(
        registry_id=card.registry_id,
        version=card.version,
        reason=(
            "no ground truth source resolved (labels registry, bundle "
            "relinking, and basename search all failed)"
        ),
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


@contextmanager
def _legacy_sleap_unpickle_shim():
    """Temporarily register a minimal stand-in for the legacy ``sleap`` package.

    A real ``metrics.val.npz`` is pickled by classic SLEAP's own
    (TensorFlow-based) ``sleap`` package, referencing exactly one custom class:
    ``sleap.instance.PointArray`` (confirmed by disassembling the pickle
    opcodes of a real stored file — no other legacy-package classes are
    referenced). Installing the full legacy package here would reintroduce
    the TensorFlow-based stack this repo exists to move away from, just to
    read a few small archival files. Instead, register bare-minimum
    ``numpy.ndarray`` subclasses under fake ``sleap``/``sleap.instance``
    modules — enough for ``pickle`` to reconstruct the array data — for the
    duration of the ``with`` block only, then remove them.

    Does nothing if a real ``sleap`` is already importable (never shadow a
    genuine install).
    """
    if "sleap" in sys.modules or importlib.util.find_spec("sleap") is not None:
        yield
        return

    class PointArray(np.ndarray):
        """Stand-in for ``sleap.instance.PointArray`` (unpickling only)."""

    class PredictedPointArray(np.ndarray):
        """Stand-in for ``sleap.instance.PredictedPointArray`` (unpickling only)."""

    sleap_mod = types.ModuleType("sleap")
    instance_mod = types.ModuleType("sleap.instance")
    instance_mod.PointArray = PointArray
    instance_mod.PredictedPointArray = PredictedPointArray
    sleap_mod.instance = instance_mod
    sys.modules["sleap"] = sleap_mod
    sys.modules["sleap.instance"] = instance_mod
    try:
        yield
    finally:
        del sys.modules["sleap"]
        del sys.modules["sleap.instance"]


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
    ``labels_pr.val.slp`` is absent, read under :func:`_legacy_sleap_unpickle_shim`.
    Note this stored file uses classic SLEAP's own flat, dot-separated key
    schema (``dist.p95``, ``vis.recall``, ...) — a different shape from
    ``sleap_nn.evaluation``'s nested ``distance_metrics``/``visibility_metrics``
    dicts read in the recomputed branch above. If the file is missing or still
    cannot be read even with the shim, this is treated as "no reference
    available" — an explicit, logged, non-fatal gap for this model's
    classic-SLEAP comparison, not a crash.

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
        with _legacy_sleap_unpickle_shim():
            stored = load_metrics(metrics_path.as_posix())
        distance_avg = float(stored["dist.avg"])
        distance_p95 = float(stored["dist.p95"])
        recall = float(stored["vis.recall"])
    except Exception as e:  # noqa: BLE001 - any unpickle/shape surprise is a gap
        logger.warning(
            "Could not read stored reference metrics %s: %s", metrics_path, e
        )
        return None
    return ParityMetrics(
        distance_avg=distance_avg,
        distance_p95=distance_p95,
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
