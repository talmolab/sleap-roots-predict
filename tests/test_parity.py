"""Tests for the A3-predict parity harness (``sleap_roots_predict.parity``).

The offline tests below build real, non-mocked ``sio.Labels``/``sio.Video``
objects and run them through the real ``sleap_nn.evaluation`` machinery — no
network access. The real multi-model harness run is gated behind
``@pytest.mark.parity`` (``SRP_PARITY_DATA_DIR`` + ``WANDB_API_KEY``), added at
the bottom of this file, mirroring the ``acceptance``/``wandb`` precedent.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import sleap_io as sio

from sleap_roots_predict.parity import (
    GapRecord,
    ParityMetrics,
    ResolvedGroundTruth,
    build_label_card,
    compute_metrics,
    reference_metrics,
    relink_ground_truth,
    resolve_ground_truth,
    within_tolerance,
)

ASSETS_DIR = Path(__file__).parent / "assets"


def _card(root_type="primary", registry_id="reg/arabidopsis-primary"):
    from sleap_roots_contracts import ModelCard

    return ModelCard(
        species="arabidopsis",
        mode="cylinder",
        age_min=2,
        age_max=14,
        root_type=root_type,
        registry_id=registry_id,
        version="v0",
    )


def _make_labels(video, skeleton, points_list, cls):
    lfs = [
        sio.LabeledFrame(
            video=video,
            frame_idx=i,
            instances=[
                cls.from_numpy(np.array(pts, dtype="float64"), skeleton=skeleton)
            ],
        )
        for i, pts in enumerate(points_list)
    ]
    return sio.Labels(labeled_frames=lfs, videos=[video], skeletons=[skeleton])


@pytest.fixture
def image_files():
    return sorted((ASSETS_DIR / "images" / "centered_pair").glob("*.png"))


@pytest.fixture
def skeleton():
    return sio.Skeleton(nodes=["A", "B"])


# --- relink_ground_truth -----------------------------------------------------


def _broken_single_file_video(name="frame_000.png"):
    # A single-file video reference, matching the real model bundles' own
    # embedded video refs (one H5/image file per Video, not an image sequence).
    return sio.Video(filename=f"D:/SLEAP/fake_project/{name}", open_backend=False)


def test_relink_ground_truth_resolves_real_pixels(tmp_path, image_files, skeleton):
    broken_video = _broken_single_file_video()
    labels = _make_labels(broken_video, skeleton, [[[10, 10], [20, 20]]], sio.Instance)
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    sio.save_slp(labels, (bundle_dir / "labels_gt.val.slp").as_posix())

    out_path = tmp_path / "relinked.slp"
    prefix_map = {"D:/SLEAP/fake_project": str(image_files[0].parent)}
    result = relink_ground_truth(bundle_dir, prefix_map, out_path)

    assert result == out_path
    reloaded = sio.load_slp(out_path.as_posix())
    assert reloaded[0].image is not None
    assert reloaded[0].image.shape[0] > 0


def test_relink_ground_truth_returns_none_when_unresolvable(
    tmp_path, image_files, skeleton
):
    broken_video = _broken_single_file_video()
    labels = _make_labels(broken_video, skeleton, [[[10, 10], [20, 20]]], sio.Instance)
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    sio.save_slp(labels, (bundle_dir / "labels_gt.val.slp").as_posix())

    out_path = tmp_path / "relinked.slp"
    prefix_map = {"E:/NoSuchDrive": "Z:/still/wrong"}
    result = relink_ground_truth(bundle_dir, prefix_map, out_path)

    assert result is None
    assert not out_path.exists()


def test_relink_ground_truth_returns_none_when_bundle_missing_file(tmp_path):
    bundle_dir = tmp_path / "empty_bundle"
    bundle_dir.mkdir()
    result = relink_ground_truth(
        bundle_dir, {"D:/SLEAP": "Z:/whatever"}, tmp_path / "out.slp"
    )
    assert result is None


# --- resolve_ground_truth -----------------------------------------------------


def test_resolve_ground_truth_prefers_labels_registry(tmp_path):
    card = _card()
    sentinel_path = tmp_path / "from_labels_registry.slp"
    sentinel_path.touch()

    def lookup(_card):
        return sentinel_path

    result = resolve_ground_truth(
        card,
        bundle_dir=tmp_path,
        workdir=tmp_path,
        labels_registry_lookup=lookup,
        prefix_map={"D:/should-not-be-tried": "Z:/nope"},
    )

    assert isinstance(result, ResolvedGroundTruth)
    assert result.source == "labels_registry"
    assert result.ground_truth_path == sentinel_path


def test_resolve_ground_truth_falls_back_to_relinked_bundle(
    tmp_path, image_files, skeleton
):
    card = _card()
    broken_video = _broken_single_file_video()
    labels = _make_labels(broken_video, skeleton, [[[1, 1], [2, 2]]], sio.Instance)
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    sio.save_slp(labels, (bundle_dir / "labels_gt.val.slp").as_posix())

    result = resolve_ground_truth(
        card,
        bundle_dir=bundle_dir,
        workdir=tmp_path,
        labels_registry_lookup=lambda _card: None,
        prefix_map={"D:/SLEAP/fake_project": str(image_files[0].parent)},
    )

    assert isinstance(result, ResolvedGroundTruth)
    assert result.source == "relinked_bundle"


def test_resolve_ground_truth_reports_gap_without_raising(tmp_path):
    card = _card(registry_id="reg/unresolvable")
    bundle_dir = tmp_path / "empty_bundle"
    bundle_dir.mkdir()

    result = resolve_ground_truth(
        card,
        bundle_dir=bundle_dir,
        workdir=tmp_path,
        labels_registry_lookup=lambda _card: None,
        prefix_map={"D:/no-match": "Z:/no-match-either"},
    )

    assert isinstance(result, GapRecord)
    assert result.registry_id == "reg/unresolvable"
    assert result.version == "v0"
    assert result.reason


def test_resolve_ground_truth_gap_does_not_block_other_models(tmp_path):
    unresolvable_card = _card(registry_id="reg/unresolvable")
    resolvable_card = _card(registry_id="reg/resolvable")
    sentinel_path = tmp_path / "found.slp"
    sentinel_path.touch()

    def lookup(card):
        return sentinel_path if card.registry_id == "reg/resolvable" else None

    gap_result = resolve_ground_truth(
        unresolvable_card,
        bundle_dir=tmp_path,
        workdir=tmp_path,
        labels_registry_lookup=lookup,
    )
    ok_result = resolve_ground_truth(
        resolvable_card,
        bundle_dir=tmp_path,
        workdir=tmp_path,
        labels_registry_lookup=lookup,
    )

    assert isinstance(gap_result, GapRecord)
    assert isinstance(ok_result, ResolvedGroundTruth)


# --- compute_metrics / reference_metrics -------------------------------------


def test_compute_metrics_gives_real_per_node_distance_and_excludes_oks(
    tmp_path, image_files, skeleton
):
    video = sio.Video(filename=[str(f) for f in image_files])
    gt = _make_labels(
        video, skeleton, [[[10, 10], [20, 20]], [[15, 15], [25, 25]]], sio.Instance
    )
    pr = _make_labels(
        video,
        skeleton,
        [[[11, 11], [21, 21]], [[16, 16], [26, 26]]],
        sio.PredictedInstance,
    )
    gt_path = tmp_path / "gt.slp"
    pr_path = tmp_path / "pr.slp"
    sio.save_slp(gt, gt_path.as_posix())
    sio.save_slp(pr, pr_path.as_posix())

    result = compute_metrics(gt_path, pr_path)

    assert isinstance(result, ParityMetrics)
    assert result.settings == "recomputed"
    # Known shift of (1, 1) px per point -> distance = sqrt(2) ~= 1.414
    assert result.distance_avg == pytest.approx(1.4142, abs=1e-3)
    assert result.distance_p95 == pytest.approx(1.4142, abs=1e-3)
    assert result.visibility_recall == pytest.approx(1.0)
    # ParityMetrics has no OKS-derived fields at all - structurally excluded.
    assert not hasattr(result, "mOKS")
    assert not hasattr(result, "oks_map")


def test_reference_metrics_recomputes_when_labels_pr_present(
    tmp_path, image_files, skeleton
):
    video = sio.Video(filename=[str(f) for f in image_files])
    gt = _make_labels(video, skeleton, [[[10, 10], [20, 20]]], sio.Instance)
    pr = _make_labels(video, skeleton, [[[10, 10], [20, 20]]], sio.PredictedInstance)
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    gt_path = bundle_dir / "labels_gt.val.slp"
    sio.save_slp(gt, gt_path.as_posix())
    sio.save_slp(pr, (bundle_dir / "labels_pr.val.slp").as_posix())

    result = reference_metrics(bundle_dir, gt_path)

    assert result is not None
    assert result.settings == "recomputed"
    assert result.distance_avg == pytest.approx(0.0)


def test_reference_metrics_returns_none_when_nothing_available(tmp_path):
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    gt_path = tmp_path / "gt.slp"

    result = reference_metrics(bundle_dir, gt_path)

    assert result is None


def test_reference_metrics_reads_real_legacy_stored_npz_via_shim(tmp_path):
    # A real classic-SLEAP-produced metrics.val.npz (from the live production
    # rice-cylinder-primary-age2-5 model), pickled by the legacy `sleap`
    # package. Confirms _legacy_sleap_unpickle_shim actually unpickles real
    # data, not just a synthetic stand-in.
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    real_npz = (
        ASSETS_DIR / "legacy_metrics" / "rice_cylinder_primary_age2-5.metrics.val.npz"
    )
    (bundle_dir / "metrics.val.npz").write_bytes(real_npz.read_bytes())
    gt_path = tmp_path / "gt.slp"  # unused in the stored branch

    result = reference_metrics(bundle_dir, gt_path)

    assert result is not None
    assert result.settings == "stored"
    assert 0.0 <= result.visibility_recall <= 1.0
    assert result.distance_p95 > 0.0
    # The shim is temporary - it must not leak into the ambient environment.
    assert "sleap" not in sys.modules


# --- build_label_card ---------------------------------------------------------


def test_build_label_card_derives_content_fields(tmp_path, image_files, skeleton):
    video = sio.Video(filename=[str(f) for f in image_files])
    labels = _make_labels(
        video, skeleton, [[[1, 1], [2, 2]], [[3, 3], [4, 4]]], sio.Instance
    )
    labels_path = tmp_path / "gt.slp"
    sio.save_slp(labels, labels_path.as_posix())
    card = _card()

    result = build_label_card(labels_path, card, images_embedded=True)

    assert result.species == "arabidopsis"
    assert result.root_type == "primary"
    assert result.node_count == 2
    assert result.node_names == ("A", "B")
    assert result.n_frames == 2
    assert result.n_instances == 2
    assert result.images_embedded is True
    assert result.registry_id == card.registry_id
    assert result.version == card.version
    # Unrecoverable provenance fields default to None, never fabricated.
    assert result.source_experiment is None
    assert result.bloom_experiment_id is None
    assert result.accessions is None
    assert result.labeler is None


# --- within_tolerance ---------------------------------------------------------


def test_within_tolerance_true_when_deltas_are_small():
    a = ParityMetrics(
        distance_avg=1.0,
        distance_p95=1.5,
        visibility_recall=0.98,
        settings="recomputed",
    )
    b = ParityMetrics(
        distance_avg=1.0, distance_p95=1.6, visibility_recall=0.99, settings="stored"
    )

    assert within_tolerance(a, b, distance_tolerance_px=2.0, recall_tolerance=0.05)


def test_within_tolerance_false_when_distance_delta_too_large():
    a = ParityMetrics(
        distance_avg=1.0,
        distance_p95=10.0,
        visibility_recall=0.98,
        settings="recomputed",
    )
    b = ParityMetrics(
        distance_avg=1.0, distance_p95=1.0, visibility_recall=0.98, settings="stored"
    )

    assert not within_tolerance(a, b, distance_tolerance_px=2.0, recall_tolerance=0.05)


def test_within_tolerance_false_when_recall_delta_too_large():
    a = ParityMetrics(
        distance_avg=1.0, distance_p95=1.0, visibility_recall=0.5, settings="recomputed"
    )
    b = ParityMetrics(
        distance_avg=1.0, distance_p95=1.0, visibility_recall=0.98, settings="stored"
    )

    assert not within_tolerance(a, b, distance_tolerance_px=2.0, recall_tolerance=0.05)


# --- parity marker (real-data, network-gated) --------------------------------

PARITY_DATA_DIR = os.environ.get("SRP_PARITY_DATA_DIR")
WANDB_API_KEY = os.environ.get("WANDB_API_KEY")


@pytest.mark.parity
@pytest.mark.skipif(
    not (PARITY_DATA_DIR and WANDB_API_KEY),
    reason="Set SRP_PARITY_DATA_DIR and WANDB_API_KEY to run the parity harness",
)
def test_parity_harness_reports_all_production_models(tmp_path):
    """End-to-end: resolve GT + compute metrics for every live production model.

    Implemented incrementally in a follow-up task once the labels-registry
    lookup and per-species path-relinking prefixes are wired up against the
    live registry (see tasks.md tasks 5.2/6).
    """
    pytest.skip("Full multi-model harness wiring is tracked as a follow-up task.")
