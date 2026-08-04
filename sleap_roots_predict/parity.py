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
from dataclasses import dataclass, replace
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
    #: classic-SLEAP's ``labels_pr.val.slp``, relinked with the *same* mapping
    #: used for `ground_truth_path` and filtered to the same frames, so the
    #: two align for `reference_metrics`. ``None`` when the bundle has no
    #: `labels_pr.val.slp`, or when the source is `"labels_registry"` (a
    #: separate collection with no corresponding bundle predictions).
    predicted_path: Optional[Path] = None


@dataclass(frozen=True)
class ParityMetrics:
    """The full metric set from one ``run_evaluation`` (or stored-npz) call.

    ``distance_p95`` and ``visibility_recall`` are what :func:`within_tolerance`
    gates on (see the module docstring for why: unaffected by the root-domain
    OKS-sigma miscalibration `sleap-roots-training#17` found). Every other
    field here is captured for completeness/analysis — e.g. to sanity-check an
    outlier model's ``p95`` against its ``pck_at_10px`` or
    ``visibility_precision`` — but is **not** read by the gate. ``moks``/
    ``voc_oks_map``/``voc_oks_mar`` are captured too, purely informational;
    they must never be used for gating (that miscalibration is exactly why
    this harness exists in its current shape).
    """

    # Gated (see within_tolerance)
    distance_p95: float
    visibility_recall: float

    # Captured, not gated
    distance_avg: float
    distance_p50: float
    distance_p75: float
    distance_p90: float
    distance_p99: float
    visibility_precision: float
    pck_mean: float
    pck_at_5px: float
    pck_at_10px: float
    moks: float
    voc_oks_map: float
    voc_oks_mar: float
    voc_pck_map: float
    voc_pck_mar: float

    settings: str  # "recomputed" | "stored"

    def to_dict(self) -> dict:
        """Return a plain-``float``/``str`` dict, safe for ``json.dump``."""
        return {
            "distance_p95": float(self.distance_p95),
            "visibility_recall": float(self.visibility_recall),
            "distance_avg": float(self.distance_avg),
            "distance_p50": float(self.distance_p50),
            "distance_p75": float(self.distance_p75),
            "distance_p90": float(self.distance_p90),
            "distance_p99": float(self.distance_p99),
            "visibility_precision": float(self.visibility_precision),
            "pck_mean": float(self.pck_mean),
            "pck_at_5px": float(self.pck_at_5px),
            "pck_at_10px": float(self.pck_at_10px),
            "moks": float(self.moks),
            "voc_oks_map": float(self.voc_oks_map),
            "voc_oks_mar": float(self.voc_oks_mar),
            "voc_pck_map": float(self.voc_pck_map),
            "voc_pck_mar": float(self.voc_pck_mar),
            "settings": self.settings,
        }


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


def _hashable_filename(video) -> object:
    """Return a hashable form of ``video.filename``.

    As-is for a single-file video, a tuple for a multi-file (image-sequence)
    video (``list`` isn't hashable).
    """
    fn = getattr(video, "filename", None)
    return tuple(fn) if isinstance(fn, list) else fn


def _original_keys(labels: sio.Labels) -> dict:
    """Map each labeled frame's ``id()`` to its ``(video_filename, frame_idx)`` before relinking.

    ``replace_filenames`` mutates ``labels`` in place without recreating its
    ``LabeledFrame`` objects, so ``id(lf)`` stays stable across the call —
    this lets a caller recover, after relinking, which *original* video path
    each surviving frame came from.
    """
    return {id(lf): (_hashable_filename(lf.video), lf.frame_idx) for lf in labels}


def _save_filtered(labels: sio.Labels, kept: list, out_path: Path) -> Path:
    """Save ``kept`` frames, dropping any video not referenced by them.

    A real bundle can list hundreds of videos while only a handful have
    labeled/kept frames (e.g. one video per plant, most unused in a sample).
    ``sleap_nn.evaluation.run_evaluation``'s video-matching step scales with
    the *listed* video count, not just the frame count — confirmed to take
    ~10 minutes for a real 355-video/20-frame file — so keeping only the
    referenced videos matters for more than file size.
    """
    kept_videos = {id(lf.video) for lf in kept}
    videos = [v for v in labels.videos if id(v) in kept_videos]
    filtered = sio.Labels(
        labeled_frames=kept, videos=videos, skeletons=labels.skeletons
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sio.save_slp(filtered, out_path.as_posix())
    return out_path


def _relink_predictions_aligned(
    bundle_dir: Path, keep_keys: set, out_path: Path, **relink_kwargs
) -> Optional[Path]:
    """Relink ``labels_pr.val.slp`` with the same mapping used for its ground truth.

    Classic-SLEAP's own predictions must be relinked with the *identical*
    mapping applied to the ground truth they'll be compared against —
    otherwise their video paths point at different locations than the
    (already-relinked) ground truth and ``run_evaluation`` cannot pair any
    frames (confirmed: this previously raised ``Empty Frame Pairs`` against
    real data). Filters to ``keep_keys`` — the ``(original_filename,
    frame_idx)`` pairs that survived in the ground truth — so the two stay
    frame-for-frame aligned even when the ground truth itself was filtered or
    sampled.

    Args:
        bundle_dir: The model's materialized artifact directory.
        keep_keys: ``{(original_video_filename, frame_idx), ...}`` from the
            ground truth's kept frames, keyed by their *pre-relink* identity.
        out_path: Where to save the relinked, filtered predictions.
        **relink_kwargs: Forwarded to ``sio.Labels.replace_filenames`` — pass
            the *same* ``prefix_map=`` or ``filename_map=`` used for the
            ground truth.

    Returns:
        ``out_path`` if any frame matches ``keep_keys``, else ``None``.
    """
    pr_path = bundle_dir / _PR_FILENAME
    if not pr_path.exists():
        return None
    pr_labels = sio.load_slp(pr_path.as_posix())
    original = _original_keys(pr_labels)
    pr_labels.replace_filenames(**relink_kwargs)
    matched = [lf for lf in pr_labels if original[id(lf)] in keep_keys]
    if not matched:
        return None
    return _save_filtered(pr_labels, matched, out_path)


def relink_ground_truth(bundle_dir: Path, prefix_map: dict, out_path: Path):
    """Relink a model bundle's ``labels_gt.val.slp`` via a path-prefix substitution.

    Loads ``bundle_dir/labels_gt.val.slp``, rewrites its embedded video paths via
    ``sio.Labels.replace_filenames(prefix_map=...)``, and keeps only the
    labeled frames whose video actually loads real pixels afterward — a
    prefix map that fixes most, but not all, of a bundle's video paths (seen
    in practice: a handful of stray videos under a different prefix mixed
    into an otherwise-uniform bundle) still yields a real, if smaller, ground
    truth rather than being silently treated as fully resolved or a total
    gap. Also relinks ``labels_pr.val.slp`` (if present) with the same
    prefix map, aligned to the same kept frames (see
    :func:`_relink_predictions_aligned`). Returns ``None`` (not a raise) on
    any failure, including a missing bundle file or zero frames resolving,
    so callers can treat this as one priority-ordered resolution attempt
    among several.

    Args:
        bundle_dir: A materialized model artifact directory (e.g. from
            ``WandbRegistrySource.materialize``) expected to contain
            ``labels_gt.val.slp``.
        prefix_map: Passed through to ``sio.Labels.replace_filenames``, e.g.
            ``{"D:/SLEAP": "Z:/users/eberrigan/SLEAP"}``.
        out_path: Where to save the relinked, filtered ground truth on
            success. The aligned predictions (if any) are saved alongside it.

    Returns:
        ``(out_path, n_frames_resolved, n_frames_total, predicted_path)``
        when at least one ground-truth frame resolves — `predicted_path` is
        ``None`` when the bundle has no `labels_pr.val.slp` or none of it
        aligns to the kept frames. Else ``None``.
    """
    gt_path = bundle_dir / _GT_FILENAME
    if not gt_path.exists():
        return None
    labels = sio.load_slp(gt_path.as_posix())
    n_total = len(labels)
    original = _original_keys(labels)
    labels.replace_filenames(prefix_map=prefix_map)
    kept = _filter_to_loadable_frames(labels)
    if not kept:
        return None
    keep_keys = {original[id(lf)] for lf in kept}
    gt_out = _save_filtered(labels, kept, out_path)
    pr_out_path = out_path.with_name(out_path.stem + ".pr" + out_path.suffix)
    pr_out = _relink_predictions_aligned(
        bundle_dir, keep_keys, pr_out_path, prefix_map=prefix_map
    )
    return gt_out, len(kept), n_total, pr_out


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

    Also relinks ``labels_pr.val.slp`` (if present) with the same discovered
    ``filename_map``, aligned to the same kept frames (see
    :func:`_relink_predictions_aligned`) — necessary for the same reason as
    :func:`relink_ground_truth`.

    Args:
        bundle_dir: A materialized model artifact directory expected to
            contain ``labels_gt.val.slp``.
        basename_index: From :func:`build_basename_index`, built once and
            reused across models.
        card: The production ``ModelCard`` (used for age-range disambiguation
            in :func:`_pick_best_candidate`).
        out_path: Where to save the filtered, relinked ground truth on
            success. The aligned predictions (if any) are saved alongside it.

    Returns:
        ``(out_path, n_frames_resolved, n_frames_total, predicted_path)``
        when at least one ground-truth frame resolves — `predicted_path` is
        ``None`` when the bundle has no `labels_pr.val.slp` or none of it
        aligns to the kept frames. Else ``None``.
    """
    gt_path = bundle_dir / _GT_FILENAME
    if not gt_path.exists():
        return None
    labels = sio.load_slp(gt_path.as_posix())
    n_total = len(labels)
    original = _original_keys(labels)

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

    keep_keys = {original[id(lf)] for lf in kept}
    gt_out = _save_filtered(labels, kept, out_path)
    pr_out_path = out_path.with_name(out_path.stem + ".pr" + out_path.suffix)
    pr_out = _relink_predictions_aligned(
        bundle_dir, keep_keys, pr_out_path, filename_map=filename_map
    )
    return gt_out, len(kept), n_total, pr_out


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
            path, n_resolved, n_total, predicted_path = result
            return ResolvedGroundTruth(
                card=card,
                ground_truth_path=path,
                bundle_dir=bundle_dir,
                source="relinked_bundle",
                n_frames_resolved=n_resolved,
                n_frames_total=n_total,
                predicted_path=predicted_path,
            )

    if basename_index is not None:
        out_path = workdir / f"{safe_id}.{card.version}.basename_search.slp"
        result = relink_ground_truth_by_basename_search(
            bundle_dir, basename_index, card, out_path
        )
        if result is not None:
            path, n_resolved, n_total, predicted_path = result
            return ResolvedGroundTruth(
                card=card,
                ground_truth_path=path,
                bundle_dir=bundle_dir,
                source="basename_search",
                n_frames_resolved=n_resolved,
                n_frames_total=n_total,
                predicted_path=predicted_path,
            )

    return GapRecord(
        registry_id=card.registry_id,
        version=card.version,
        reason=(
            "no ground truth source resolved (labels registry, bundle "
            "relinking, and basename search all failed)"
        ),
    )


def sample_ground_truth(
    resolved: ResolvedGroundTruth, n: int, workdir: Path
) -> ResolvedGroundTruth:
    """Cap a resolved ground truth to at most ``n`` frames, keeping alignment.

    Real bundles can carry hundreds of resolved frames; running sleap-nn
    inference on all of them for every production model is not necessary to
    get a meaningful empirical baseline. Takes the first ``n`` frames
    (deterministic, not random, for reproducible runs) and, when
    `resolved.predicted_path` is set, filters it to the same
    ``(video_filename, frame_idx)`` keys so the sample stays aligned for
    :func:`reference_metrics`.

    `n_frames_resolved`/`n_frames_total` are left unchanged — they describe
    true resolution coverage, not this working sample.

    Args:
        resolved: A resolved ground truth (see :func:`resolve_ground_truth`).
        n: Maximum number of frames to keep.
        workdir: Scratch directory for the sampled output files.

    Returns:
        `resolved` unchanged if it already has ``<= n`` frames, else a copy
        with `ground_truth_path`/`predicted_path` pointing at the sample.
    """
    gt_labels = sio.load_slp(resolved.ground_truth_path.as_posix())
    if len(gt_labels) <= n:
        return resolved

    kept = list(gt_labels)[:n]
    keep_keys = {(_hashable_filename(lf.video), lf.frame_idx) for lf in kept}
    safe_id = resolved.card.registry_id.replace("/", "_")
    gt_out = _save_filtered(
        gt_labels, kept, workdir / f"{safe_id}.{resolved.card.version}.sample_gt.slp"
    )

    pr_out = None
    if resolved.predicted_path is not None:
        pr_labels = sio.load_slp(resolved.predicted_path.as_posix())
        pr_kept = [
            lf
            for lf in pr_labels
            if (_hashable_filename(lf.video), lf.frame_idx) in keep_keys
        ]
        if pr_kept:
            pr_out = _save_filtered(
                pr_labels,
                pr_kept,
                workdir / f"{safe_id}.{resolved.card.version}.sample_pr.slp",
            )

    return replace(resolved, ground_truth_path=gt_out, predicted_path=pr_out)


def run_sleap_nn_predictions(
    ground_truth_path: Path, model_dir: Path, out_path: Path
) -> Path:
    """Run sleap-nn inference on a resolved ground truth's own images.

    This is the sleap-nn side of the parity comparison: the same production
    model weights (``model_dir``, the card's materialized artifact), run
    through sleap-nn on the exact same images the ground truth's labeled
    frames reference, producing a predicted ``.slp`` aligned frame-for-frame
    with the ground truth for :func:`compute_metrics`.

    Predicts once per distinct video referenced by the ground truth, passing
    only the specific labeled frame indices needed for that video (real
    bundle videos are full multi-frame scans, not one frame each — asking
    ``sleap_nn`` to predict only the frames actually labeled, rather than the
    whole video, avoids doing 10-50x more inference than the comparison
    needs).

    Args:
        ground_truth_path: A resolved ground-truth ``.slp`` (see
            :func:`resolve_ground_truth`).
        model_dir: The production model's materialized directory (loadable
            by ``sleap_roots_predict.predict.make_predictor``).
        out_path: Where to save the predicted labels.

    Returns:
        ``out_path``.
    """
    from sleap_roots_predict.predict import make_predictor

    ground_truth = sio.load_slp(ground_truth_path.as_posix())
    gt_frame_idxs_by_video = {}
    for lf in ground_truth:
        gt_frame_idxs_by_video.setdefault(lf.video, set()).add(lf.frame_idx)

    predictor = make_predictor([model_dir])
    predicted_frames = []
    predicted_skeletons = ground_truth.skeletons
    for video, frame_idxs in gt_frame_idxs_by_video.items():
        predicted = predictor.predict(
            video, make_labels=True, frames=sorted(frame_idxs)
        )
        predicted_skeletons = predicted.skeletons
        predicted_frames.extend(predicted)

    predicted_labels = sio.Labels(
        labeled_frames=predicted_frames,
        videos=ground_truth.videos,
        skeletons=predicted_skeletons,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sio.save_slp(predicted_labels, out_path.as_posix())
    return out_path


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
    return _parity_metrics_from_run_evaluation(metrics, settings="recomputed")


def _parity_metrics_from_run_evaluation(
    metrics: dict, *, settings: str
) -> ParityMetrics:
    """Build a :class:`ParityMetrics` from a raw ``run_evaluation`` result."""
    dist = metrics["distance_metrics"]
    vis = metrics["visibility_metrics"]
    pck = metrics["pck_metrics"]
    voc = metrics["voc_metrics"]
    return ParityMetrics(
        distance_p95=float(dist["p95"]),
        visibility_recall=float(vis["recall"]),
        distance_avg=float(dist["avg"]),
        distance_p50=float(dist["p50"]),
        distance_p75=float(dist["p75"]),
        distance_p90=float(dist["p90"]),
        distance_p99=float(dist["p99"]),
        visibility_precision=float(vis["precision"]),
        pck_mean=float(pck["mPCK"]),
        pck_at_5px=float(pck["PCK@5"]),
        pck_at_10px=float(pck["PCK@10"]),
        moks=float(metrics["mOKS"]["mOKS"]),
        voc_oks_map=float(voc["oks_voc.mAP"]),
        voc_oks_mar=float(voc["oks_voc.mAR"]),
        voc_pck_map=float(voc["pck_voc.mAP"]),
        voc_pck_mar=float(voc["pck_voc.mAR"]),
        settings=settings,
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
    resolved: ResolvedGroundTruth,
    *,
    match_threshold: float = OKS_MATCH_THRESHOLD,
) -> Optional[ParityMetrics]:
    """Get classic-SLEAP's reference parity signal for a resolved model.

    Recomputes via :func:`compute_metrics` against `resolved.predicted_path`
    (classic-SLEAP's own predictions, already relinked and frame-aligned to
    `resolved.ground_truth_path` by :func:`resolve_ground_truth` — using the
    *same* mapping for both is required, not optional: comparing a relinked
    ground truth against still-broken-path predictions yields zero matched
    frames) when present, so the comparison uses identical settings to the
    sleap-nn side.

    Falls back to the bundle's stored ``metrics.val.npz`` when
    `resolved.predicted_path` is unavailable (no `labels_pr.val.slp` in the
    bundle, none of it survived relinking, or the ground truth came from the
    labels registry — a separate collection with no corresponding bundle
    predictions to align against), read under
    :func:`_legacy_sleap_unpickle_shim`. Note this stored file uses classic
    SLEAP's own flat, dot-separated key schema (``dist.p95``, ``vis.recall``,
    ...) — a different shape from ``sleap_nn.evaluation``'s nested
    ``distance_metrics``/``visibility_metrics`` dicts read in the recomputed
    branch above. If the file is missing or still cannot be read even with
    the shim, this is treated as "no reference available" — an explicit,
    logged, non-fatal gap for this model's classic-SLEAP comparison, not a
    crash.

    Args:
        resolved: The :class:`ResolvedGroundTruth` for this model.
        match_threshold: Forwarded to :func:`compute_metrics` when recomputing.

    Returns:
        A :class:`ParityMetrics` with ``settings="recomputed"`` when
        `resolved.predicted_path` was used, ``"stored"`` when the stored
        ``metrics.val.npz`` could be read, or ``None`` when neither is
        available/readable.
    """
    if resolved.predicted_path is not None:
        return compute_metrics(
            resolved.ground_truth_path,
            resolved.predicted_path,
            match_threshold=match_threshold,
        )
    metrics_path = resolved.bundle_dir / _METRICS_FILENAME
    if not metrics_path.exists():
        return None
    try:
        with _legacy_sleap_unpickle_shim():
            stored = load_metrics(metrics_path.as_posix())
        return _parity_metrics_from_stored_npz(stored)
    except Exception as e:  # noqa: BLE001 - any unpickle/shape surprise is a gap
        logger.warning(
            "Could not read stored reference metrics %s: %s", metrics_path, e
        )
        return None


def _parity_metrics_from_stored_npz(stored: dict) -> ParityMetrics:
    """Build a :class:`ParityMetrics` from a legacy ``metrics.val.npz`` dict.

    Classic SLEAP's own stored file uses flat, dot-separated keys (``dist.p95``,
    ``vis.recall``, ...) rather than ``sleap_nn.evaluation``'s nested shape —
    see :func:`reference_metrics`. ``PCK@5``/``PCK@10`` are never present in a
    stored file (confirmed: ``run_evaluation`` computes them only for its log
    output, *after* the branch that writes ``save_metrics``) — recomputed here
    from the stored raw per-instance ``dist.dists`` array the same way
    ``run_evaluation`` does, so the "stored" and "recomputed" settings report
    symmetric fields rather than leaving two blank.
    """
    dists = np.asarray(stored["dist.dists"], dtype=float)
    dists_clean = np.where(np.isnan(dists), np.inf, dists)
    return ParityMetrics(
        distance_p95=float(stored["dist.p95"]),
        visibility_recall=float(stored["vis.recall"]),
        distance_avg=float(stored["dist.avg"]),
        distance_p50=float(stored["dist.p50"]),
        distance_p75=float(stored["dist.p75"]),
        distance_p90=float(stored["dist.p90"]),
        distance_p99=float(stored["dist.p99"]),
        visibility_precision=float(stored["vis.precision"]),
        pck_mean=float(stored["pck.mPCK"]),
        pck_at_5px=float((dists_clean < 5).mean()),
        pck_at_10px=float((dists_clean < 10).mean()),
        moks=float(stored["oks.mOKS"]),
        voc_oks_map=float(stored["oks_voc.mAP"]),
        voc_oks_mar=float(stored["oks_voc.mAR"]),
        voc_pck_map=float(stored["pck_voc.mAP"]),
        voc_pck_mar=float(stored["pck_voc.mAR"]),
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


def build_report_entry(
    resolved: ResolvedGroundTruth,
    n_frames_evaluated: int,
    sleap_nn_metrics: ParityMetrics,
    reference: Optional[ParityMetrics],
) -> dict:
    """Build one model's JSON-serializable entry for a persisted parity report.

    Every field ``run_evaluation`` produces is included (via
    :meth:`ParityMetrics.to_dict`) for both sides, plus the two gated deltas,
    so a saved report is a complete, inspectable record — not just the two
    numbers the tolerance decision reads.

    Args:
        resolved: The model's :class:`ResolvedGroundTruth`.
        n_frames_evaluated: How many frames the metrics below were computed
            over (may be less than `resolved.n_frames_resolved` if sampled —
            see :func:`sample_ground_truth`).
        sleap_nn_metrics: This harness's own computed metrics.
        reference: Classic-SLEAP's reference metrics, or ``None`` if
            unavailable for this model (see :func:`reference_metrics`).

    Returns:
        A plain dict, safe for ``json.dump``.
    """
    entry = {
        "registry_id": resolved.card.registry_id,
        "version": resolved.card.version,
        "species": resolved.card.species,
        "mode": resolved.card.mode,
        "root_type": resolved.card.root_type,
        "age_min": resolved.card.age_min,
        "age_max": resolved.card.age_max,
        "weights_checksum": resolved.card.weights_checksum,
        "ground_truth_source": resolved.source,
        "n_frames_resolved": resolved.n_frames_resolved,
        "n_frames_total": resolved.n_frames_total,
        "n_frames_evaluated": n_frames_evaluated,
        "sleap_nn": sleap_nn_metrics.to_dict(),
        "classic_sleap_reference": reference.to_dict() if reference else None,
    }
    if reference is not None:
        entry["distance_p95_delta"] = abs(
            sleap_nn_metrics.distance_p95 - reference.distance_p95
        )
        entry["visibility_recall_delta"] = abs(
            sleap_nn_metrics.visibility_recall - reference.visibility_recall
        )
    else:
        entry["distance_p95_delta"] = None
        entry["visibility_recall_delta"] = None
    return entry


def write_parity_report(entries: list, out_path: Path) -> Path:
    """Persist a list of :func:`build_report_entry` dicts as an indented JSON file.

    Args:
        entries: Per-model report entries.
        out_path: Where to write the report.

    Returns:
        ``out_path``.
    """
    import json

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(entries, f, indent=2)
    return out_path
