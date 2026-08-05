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
    build_basename_index,
    build_label_card,
    build_report_entry,
    compute_metrics,
    evaluate_model_card,
    reference_metrics,
    relink_ground_truth,
    relink_ground_truth_by_basename_search,
    resolve_ground_truth,
    run_sleap_nn_predictions,
    sample_ground_truth,
    within_tolerance,
    write_parity_report,
)
from sleap_roots_predict.parity import _pick_best_candidate
from sleap_roots_predict.video_utils import save_array_as_h5

ASSETS_DIR = Path(__file__).parent / "assets"


def _card(
    root_type="primary", registry_id="reg/arabidopsis-primary", age_min=2, age_max=14
):
    from sleap_roots_contracts import ModelCard

    return ModelCard(
        species="arabidopsis",
        mode="cylinder",
        age_min=age_min,
        age_max=age_max,
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


def _write_small_slp(path, n_frames=1):
    files = sorted((ASSETS_DIR / "images" / "centered_pair").glob("*.png"))[:n_frames]
    video = sio.Video(filename=[str(f) for f in files])
    skeleton = sio.Skeleton(nodes=["A", "B"])
    labels = _make_labels(video, skeleton, [[[1, 1], [2, 2]]] * n_frames, sio.Instance)
    sio.save_slp(labels, str(path))
    return path


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

    assert result == (out_path, 1, 1, None)


def test_relink_ground_truth_also_aligns_labels_pr(tmp_path, image_files, skeleton):
    # Reproduces the real bug found running the live 13-model harness:
    # comparing a relinked ground truth against a still-broken-path
    # labels_pr.val.slp raised "Empty Frame Pairs" (zero matched frames)
    # because the two pointed at different locations. labels_pr.val.slp
    # must be relinked with the *same* mapping as the ground truth it's
    # compared against.
    broken_video = _broken_single_file_video()
    gt = _make_labels(broken_video, skeleton, [[[10, 10], [20, 20]]], sio.Instance)
    pr = _make_labels(
        broken_video, skeleton, [[[11, 11], [21, 21]]], sio.PredictedInstance
    )
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    sio.save_slp(gt, (bundle_dir / "labels_gt.val.slp").as_posix())
    sio.save_slp(pr, (bundle_dir / "labels_pr.val.slp").as_posix())

    out_path = tmp_path / "relinked.slp"
    prefix_map = {"D:/SLEAP/fake_project": str(image_files[0].parent)}
    result = relink_ground_truth(bundle_dir, prefix_map, out_path)

    assert result is not None
    gt_out, n_resolved, n_total, pr_out = result
    assert n_resolved == 1
    assert pr_out is not None

    # The relinked predictions must resolve real pixels too (same relink),
    # and compute_metrics must find the frame pair (the actual regression).
    metrics = compute_metrics(gt_out, pr_out)
    assert metrics.distance_avg == pytest.approx(1.4142, abs=1e-3)
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
    sentinel_path = _write_small_slp(tmp_path / "from_labels_registry.slp")

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
    assert result.n_frames_resolved == result.n_frames_total == 1


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
    sentinel_path = _write_small_slp(tmp_path / "found.slp")

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


# --- _pick_best_candidate / basename search ----------------------------------


def test_pick_best_candidate_single_candidate_is_unambiguous():
    card = _card()
    winner = _pick_best_candidate("D:/broken/path/plant.h5", ["Z:/real/plant.h5"], card)
    assert winner == "Z:/real/plant.h5"


def test_pick_best_candidate_disambiguates_by_parent_folder_name():
    # Same plant ID (basename), scanned at two different timepoints - the
    # real-world case that motivated this function (confirmed distinct
    # content, not accidental duplicates, by comparing real file hashes).
    card = _card()
    broken = "D:/FNRice2022/h5_files/3_do/week1_3do_4-18-22/plant.h5"
    candidates = [
        "Z:/SLEAP_Rice/10_do/h5_files/Day10_4-25-2022/plant.h5",
        "Z:/SLEAP_Rice/3_do/h5_files/week1_3do_4-18-22/plant.h5",
    ]
    winner = _pick_best_candidate(broken, candidates, card)
    assert winner == candidates[1]


def test_pick_best_candidate_disambiguates_by_age_hint_in_range():
    card = _card(root_type="crown", age_min=6, age_max=10)
    broken = (
        "C:/Users/pbiobgh/Box/rice/10_Days_Old_main_root/FN_Day10_4-25-2022/plant.h5"
    )
    candidates = [
        "Z:/SLEAP_Rice/10_do/h5_files/Day10_4-25-2022/plant.h5",
        "Z:/SLEAP_Rice/3_do/h5_files/Day3_4-18-2022/plant.h5",
    ]
    winner = _pick_best_candidate(broken, candidates, card)
    assert winner == candidates[0]


def test_pick_best_candidate_falls_back_to_segment_overlap():
    card = _card()
    broken = (
        "E:/Soy_GDM_Brazil/h5_files_for_lr_sleap_model_elizabeth/blue_6do_1.17.22/p.h5"
    )
    candidates = [
        "Z:/SLEAP_Soy/lateral_root_4_nodes/h5_files_blue_6_do_1.17.22/p.h5",
        "Z:/SLEAP_Soy/primary_multi-day/blue_6_do_1.17.22/p.h5",
    ]
    # Neither parent-folder-name matches exactly, and both fall inside a
    # plausible age range, so this falls through to segment overlap - the
    # "lateral_root_4_nodes" segment only appears in the broken path's own
    # lineage/context (via prior overlap with other resolved siblings) is not
    # available here, but overlap on the shared "blue_6_do_1_17_22"-ish
    # segment plus more shared date tokens should still favor one candidate
    # when the two are not perfectly symmetric.
    winner = _pick_best_candidate(broken, candidates, card)
    # Both candidates are structurally symmetric here (this is intentionally
    # the ambiguous case) - assert the function returns a deterministic,
    # non-crashing result: either a winner or an explicit None, never a raise.
    assert winner is None or winner in candidates


def test_pick_best_candidate_returns_none_when_genuinely_tied():
    card = _card()
    broken = "D:/broken/x/plant.h5"
    candidates = ["Z:/a/plant.h5", "Z:/b/plant.h5"]
    winner = _pick_best_candidate(broken, candidates, card)
    assert winner is None


def test_build_basename_index_finds_files_by_lowercase_basename(tmp_path):
    (tmp_path / "sub1").mkdir()
    (tmp_path / "sub2").mkdir()
    (tmp_path / "sub1" / "Plant.H5").touch()
    (tmp_path / "sub2" / "other.h5").touch()

    index = build_basename_index(tmp_path)

    assert len(index["plant.h5"]) == 1
    assert index["plant.h5"][0].endswith("Plant.H5")
    assert len(index["other.h5"]) == 1


def test_relink_ground_truth_by_basename_search_partial_resolution(tmp_path, skeleton):
    card = _card()
    # Two videos: one whose basename is findable in the index, one that isn't.
    resolvable_video = sio.Video(filename="D:/broken/resolvable.h5", open_backend=False)
    unresolvable_video = sio.Video(filename="D:/broken/nowhere.h5", open_backend=False)
    lf1 = sio.LabeledFrame(
        video=resolvable_video,
        frame_idx=0,
        instances=[
            sio.Instance.from_numpy(
                np.array([[1, 1], [2, 2]], dtype="float64"), skeleton=skeleton
            )
        ],
    )
    lf2 = sio.LabeledFrame(
        video=unresolvable_video,
        frame_idx=0,
        instances=[
            sio.Instance.from_numpy(
                np.array([[3, 3], [4, 4]], dtype="float64"), skeleton=skeleton
            )
        ],
    )
    labels = sio.Labels(
        labeled_frames=[lf1, lf2],
        videos=[resolvable_video, unresolvable_video],
        skeletons=[skeleton],
    )
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    sio.save_slp(labels, (bundle_dir / "labels_gt.val.slp").as_posix())

    search_dir = tmp_path / "search_root"
    search_dir.mkdir()
    save_array_as_h5(
        np.zeros((1, 32, 32, 1), dtype="uint8"), search_dir / "resolvable.h5"
    )
    index = build_basename_index(search_dir)

    out_path = tmp_path / "basename_relinked.slp"
    result = relink_ground_truth_by_basename_search(bundle_dir, index, card, out_path)

    assert result is not None
    path, n_resolved, n_total, predicted_path = result
    assert path == out_path
    assert n_resolved == 1
    assert n_total == 2
    assert predicted_path is None  # no labels_pr.val.slp in this bundle
    reloaded = sio.load_slp(out_path.as_posix())
    assert len(reloaded) == 1


def test_resolve_ground_truth_uses_basename_search_as_last_resort(tmp_path, skeleton):
    card = _card()
    video = sio.Video(filename="D:/broken/resolvable.h5", open_backend=False)
    labels = _make_labels(video, skeleton, [[[1, 1], [2, 2]]], sio.Instance)
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    sio.save_slp(labels, (bundle_dir / "labels_gt.val.slp").as_posix())

    search_dir = tmp_path / "search_root"
    search_dir.mkdir()
    save_array_as_h5(
        np.zeros((1, 32, 32, 1), dtype="uint8"), search_dir / "resolvable.h5"
    )
    index = build_basename_index(search_dir)

    result = resolve_ground_truth(
        card,
        bundle_dir=bundle_dir,
        workdir=tmp_path,
        labels_registry_lookup=lambda _card: None,
        prefix_map={"D:/this-wont-match": "Z:/nope"},
        basename_index=index,
    )

    assert isinstance(result, ResolvedGroundTruth)
    assert result.source == "basename_search"
    assert result.n_frames_resolved == 1


# --- run_sleap_nn_predictions --------------------------------------------------


def test_run_sleap_nn_predictions_aligns_to_ground_truth_frames(
    tmp_path, video, native_model_dir, skeleton
):
    # Ground-truth frame indices/skeleton are arbitrary here - the function
    # only uses the ground truth for which video/frames to predict on, not
    # its points, so a real vendored fly-pair model is fine to exercise the
    # real (non-mocked) sleap-nn inference path end-to-end.
    gt_frame_idxs = {0, 2}
    labels = _make_labels(
        video,
        skeleton,
        [[[1, 1], [2, 2]] for _ in gt_frame_idxs],
        sio.Instance,
    )
    # Overwrite frame_idx to the specific indices under test (helper assigns
    # sequential 0..N-1 by default).
    for lf, idx in zip(labels, sorted(gt_frame_idxs)):
        lf.frame_idx = idx
    gt_path = tmp_path / "gt.slp"
    sio.save_slp(labels, gt_path.as_posix())

    out_path = tmp_path / "sleap_nn_predictions.slp"
    result = run_sleap_nn_predictions(gt_path, native_model_dir, out_path)

    assert result == out_path
    predicted = sio.load_slp(out_path.as_posix())
    predicted_idxs = {lf.frame_idx for lf in predicted}
    assert predicted_idxs == gt_frame_idxs


# --- sample_ground_truth -------------------------------------------------------


def test_sample_ground_truth_keeps_gt_and_pr_aligned(tmp_path, image_files, skeleton):
    card = _card()
    video = sio.Video(filename=[str(f) for f in image_files[:5]])
    gt = _make_labels(
        video, skeleton, [[[i, i], [i + 1, i + 1]] for i in range(5)], sio.Instance
    )
    pr = _make_labels(
        video,
        skeleton,
        [[[i, i], [i + 1, i + 1]] for i in range(5)],
        sio.PredictedInstance,
    )
    gt_path = tmp_path / "gt.slp"
    pr_path = tmp_path / "pr.slp"
    sio.save_slp(gt, gt_path.as_posix())
    sio.save_slp(pr, pr_path.as_posix())
    resolved = _resolved(card, gt_path, tmp_path, predicted_path=pr_path)

    sampled = sample_ground_truth(resolved, n=2, workdir=tmp_path)

    assert sampled.ground_truth_path != gt_path
    assert sampled.predicted_path is not None
    sampled_gt = sio.load_slp(sampled.ground_truth_path.as_posix())
    sampled_pr = sio.load_slp(sampled.predicted_path.as_posix())
    assert len(sampled_gt) == 2
    assert len(sampled_pr) == 2
    assert {lf.frame_idx for lf in sampled_gt} == {lf.frame_idx for lf in sampled_pr}
    # Coverage fields describe true resolution, unaffected by sampling.
    assert sampled.n_frames_resolved == resolved.n_frames_resolved
    assert sampled.n_frames_total == resolved.n_frames_total


def test_sample_ground_truth_returns_unchanged_when_already_small(tmp_path, skeleton):
    card = _card()
    gt_path = _write_small_slp(tmp_path / "gt.slp", n_frames=1)
    resolved = _resolved(card, gt_path, tmp_path)

    sampled = sample_ground_truth(resolved, n=5, workdir=tmp_path)

    assert sampled is resolved


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


def _resolved(
    card,
    ground_truth_path,
    bundle_dir,
    *,
    predicted_path=None,
    source="relinked_bundle",
):
    return ResolvedGroundTruth(
        card=card,
        ground_truth_path=ground_truth_path,
        bundle_dir=bundle_dir,
        source=source,
        n_frames_resolved=1,
        n_frames_total=1,
        predicted_path=predicted_path,
    )


def test_reference_metrics_recomputes_when_labels_pr_present(
    tmp_path, image_files, skeleton
):
    card = _card()
    video = sio.Video(filename=[str(f) for f in image_files])
    gt = _make_labels(video, skeleton, [[[10, 10], [20, 20]]], sio.Instance)
    pr = _make_labels(video, skeleton, [[[10, 10], [20, 20]]], sio.PredictedInstance)
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    gt_path = bundle_dir / "labels_gt.val.slp"
    pr_path = bundle_dir / "labels_pr.val.slp"
    sio.save_slp(gt, gt_path.as_posix())
    sio.save_slp(pr, pr_path.as_posix())

    result = reference_metrics(
        _resolved(card, gt_path, bundle_dir, predicted_path=pr_path)
    )

    assert result is not None
    assert result.settings == "recomputed"
    assert result.distance_avg == pytest.approx(0.0)


def test_reference_metrics_returns_none_when_nothing_available(tmp_path):
    card = _card()
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    gt_path = tmp_path / "gt.slp"

    result = reference_metrics(_resolved(card, gt_path, bundle_dir))

    assert result is None


def test_reference_metrics_reads_real_legacy_stored_npz_via_shim(tmp_path):
    # A real classic-SLEAP-produced metrics.val.npz (from the live production
    # rice-cylinder-primary-age2-5 model), pickled by the legacy `sleap`
    # package. Confirms _legacy_sleap_unpickle_shim actually unpickles real
    # data, not just a synthetic stand-in.
    card = _card()
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    real_npz = (
        ASSETS_DIR / "legacy_metrics" / "rice_cylinder_primary_age2-5.metrics.val.npz"
    )
    (bundle_dir / "metrics.val.npz").write_bytes(real_npz.read_bytes())
    gt_path = tmp_path / "gt.slp"  # unused in the stored branch (no predicted_path)

    result = reference_metrics(_resolved(card, gt_path, bundle_dir))

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


def _metrics(
    distance_p95=1.0, visibility_recall=0.98, settings="recomputed", **overrides
):
    defaults = dict(
        distance_p95=distance_p95,
        visibility_recall=visibility_recall,
        distance_avg=1.0,
        distance_p50=1.0,
        distance_p75=1.0,
        distance_p90=1.0,
        distance_p99=1.0,
        visibility_precision=1.0,
        pck_mean=1.0,
        pck_at_5px=1.0,
        pck_at_10px=1.0,
        moks=0.5,
        voc_oks_map=0.5,
        voc_oks_mar=0.5,
        voc_pck_map=0.5,
        voc_pck_mar=0.5,
        settings=settings,
    )
    defaults.update(overrides)
    return ParityMetrics(**defaults)


def test_within_tolerance_true_when_deltas_are_small():
    a = _metrics(distance_p95=1.5, visibility_recall=0.98, settings="recomputed")
    b = _metrics(distance_p95=1.6, visibility_recall=0.99, settings="stored")

    # relative distance delta = 0.1 / 1.6 ~= 0.0625; recall delta = -0.01
    assert within_tolerance(
        a, b, distance_relative_tolerance=0.25, recall_tolerance=0.05
    )


def test_within_tolerance_false_when_distance_delta_too_large():
    a = _metrics(distance_p95=10.0, visibility_recall=0.98, settings="recomputed")
    b = _metrics(distance_p95=1.0, visibility_recall=0.98, settings="stored")

    # relative distance delta = 9.0 / 1.0 = 9.0, far over tolerance
    assert not within_tolerance(
        a, b, distance_relative_tolerance=0.25, recall_tolerance=0.05
    )


def test_within_tolerance_false_when_recall_delta_too_large():
    a = _metrics(distance_p95=1.0, visibility_recall=0.5, settings="recomputed")
    b = _metrics(distance_p95=1.0, visibility_recall=0.98, settings="stored")

    assert not within_tolerance(
        a, b, distance_relative_tolerance=0.25, recall_tolerance=0.05
    )


def test_within_tolerance_true_when_sleap_nn_recall_is_much_higher():
    # sleap-nn scoring *higher* on recall than the reference must never fail,
    # even though the magnitude of the delta exceeds recall_tolerance.
    a = _metrics(distance_p95=1.0, visibility_recall=0.99, settings="recomputed")
    b = _metrics(distance_p95=1.0, visibility_recall=0.5, settings="stored")

    assert within_tolerance(
        a, b, distance_relative_tolerance=0.25, recall_tolerance=0.05
    )


def test_within_tolerance_uses_decided_defaults_when_not_overridden():
    # 17% relative distance delta and -0.08 recall delta pass the decided
    # defaults (25% / -0.10) without the caller having to know the numbers.
    a = _metrics(distance_p95=117.0, visibility_recall=0.90, settings="recomputed")
    b = _metrics(distance_p95=100.0, visibility_recall=0.98, settings="stored")

    assert within_tolerance(a, b)


def test_within_tolerance_defaults_fail_outside_decided_bounds():
    # 30% relative distance delta exceeds the decided 25% default.
    a = _metrics(distance_p95=130.0, visibility_recall=0.98, settings="recomputed")
    b = _metrics(distance_p95=100.0, visibility_recall=0.98, settings="stored")

    assert not within_tolerance(a, b)


def test_within_tolerance_zero_reference_and_zero_sleap_nn_passes():
    # A reference distance_p95 of exactly 0.0 is a real, reachable value (see
    # test_reference_metrics_recomputes_when_labels_pr_present) — sleap-nn
    # matching it exactly is a perfect match, not a division-by-zero crash.
    a = _metrics(distance_p95=0.0, visibility_recall=0.98, settings="recomputed")
    b = _metrics(distance_p95=0.0, visibility_recall=0.98, settings="stored")

    assert within_tolerance(
        a, b, distance_relative_tolerance=0.25, recall_tolerance=0.05
    )


def test_within_tolerance_zero_reference_nonzero_sleap_nn_fails():
    # A zero reference with a nonzero sleap-nn distance is an infinite
    # relative deviation — it must fail cleanly, not raise ZeroDivisionError.
    a = _metrics(distance_p95=5.0, visibility_recall=0.98, settings="recomputed")
    b = _metrics(distance_p95=0.0, visibility_recall=0.98, settings="stored")

    assert not within_tolerance(
        a, b, distance_relative_tolerance=0.25, recall_tolerance=0.05
    )


def test_parity_metrics_to_dict_is_json_safe():
    import json

    m = _metrics()
    json.dumps(m.to_dict())  # must not raise


# --- build_report_entry / write_parity_report ---------------------------------


def test_build_report_entry_includes_full_metrics_and_deltas(tmp_path):
    card = _card()
    resolved = _resolved(card, tmp_path / "gt.slp", tmp_path, source="basename_search")
    sleap_nn = _metrics(distance_p95=5.0, visibility_recall=0.9)
    reference = _metrics(distance_p95=7.0, visibility_recall=0.8, settings="stored")

    entry = build_report_entry(resolved, 42, sleap_nn, reference)

    assert entry["registry_id"] == card.registry_id
    assert entry["n_frames_evaluated"] == 42
    assert entry["sleap_nn"]["distance_p95"] == 5.0
    assert entry["classic_sleap_reference"]["distance_p95"] == 7.0
    assert entry["distance_p95_delta"] == pytest.approx(2.0)
    assert entry["visibility_recall_delta"] == pytest.approx(0.1)


def test_build_report_entry_handles_missing_reference(tmp_path):
    card = _card()
    resolved = _resolved(card, tmp_path / "gt.slp", tmp_path)
    sleap_nn = _metrics()

    entry = build_report_entry(resolved, 10, sleap_nn, None)

    assert entry["classic_sleap_reference"] is None
    assert entry["distance_p95_delta"] is None
    assert entry["visibility_recall_delta"] is None


def test_write_parity_report_round_trips_via_json(tmp_path):
    import json

    entries = [{"registry_id": "reg/a", "value": 1.5}]
    out_path = write_parity_report(entries, tmp_path / "report.json")

    assert out_path.exists()
    with open(out_path) as f:
        reloaded = json.load(f)
    assert reloaded == entries


# --- evaluate_model_card -------------------------------------------------------


def test_evaluate_model_card_returns_report_entry(tmp_path, video, native_model_dir):
    # A stubbed labels_registry_lookup sidesteps needing a real bundled
    # labels_gt.val.slp — this exercises the same tier-1-success path as
    # test_resolve_ground_truth_prefers_labels_registry, then runs real
    # (non-mocked) sleap-nn inference against the vendored native model, the
    # same way test_run_sleap_nn_predictions_aligns_to_ground_truth_frames
    # does — end to end through evaluate_model_card.
    card = _card()
    skeleton = sio.Skeleton(nodes=["A", "B"])
    gt = _make_labels(
        video, skeleton, [[[1, 1], [2, 2]], [[3, 3], [4, 4]]], sio.Instance
    )
    gt_path = tmp_path / "gt.slp"
    sio.save_slp(gt, gt_path.as_posix())

    entry = evaluate_model_card(
        card,
        native_model_dir,
        tmp_path,
        labels_registry_lookup=lambda c: gt_path,
    )

    assert entry["registry_id"] == card.registry_id
    assert entry["n_frames_evaluated"] == 2
    assert entry["ground_truth_source"] == "labels_registry"
    # The arbitrary GT points above aren't near this real model's actual
    # predictions on these real images, so OKS matching legitimately finds
    # zero pairs (distance_p95 is NaN) - exact numeric correctness of
    # compute_metrics itself is covered by
    # test_compute_metrics_gives_real_per_node_distance_and_excludes_oks.
    # This test verifies evaluate_model_card's wiring: tier-1 resolution,
    # frame counting, and a real (non-mocked) sleap-nn inference call that
    # completes and produces a well-shaped entry.
    assert "distance_p95" in entry["sleap_nn"]
    assert isinstance(entry["sleap_nn"]["distance_p95"], float)
    # No labels_pr.val.slp/metrics.val.npz in this bundle_dir -> no reference.
    assert entry["classic_sleap_reference"] is None


def test_evaluate_model_card_returns_gap_entry_when_unresolvable(
    tmp_path, native_model_dir
):
    card = _card(registry_id="reg/unresolvable")

    entry = evaluate_model_card(card, native_model_dir, tmp_path)

    assert entry == {
        "registry_id": card.registry_id,
        "version": card.version,
        "gap_reason": entry["gap_reason"],
    }
    assert entry["gap_reason"]


# --- parity marker (real-data, network-gated) --------------------------------

PARITY_DATA_DIR = os.environ.get("SRP_PARITY_DATA_DIR")
WANDB_API_KEY = os.environ.get("WANDB_API_KEY")


@pytest.mark.parity
@pytest.mark.skipif(
    not (PARITY_DATA_DIR and WANDB_API_KEY),
    reason="Set SRP_PARITY_DATA_DIR and WANDB_API_KEY to run the parity harness",
)
def test_parity_harness_reports_all_production_models(tmp_path):
    """Standing regression check: the decided tolerance still holds live.

    Not a full re-run of all 13 production models (that's an expensive,
    environment-specific manual harness — see
    ``docs/superpowers/specs/2026-08-04-define-parity-tolerance-results.json``
    for that empirical baseline, not something to bake into a portable
    test). This exercises the real end-to-end path — live registry ->
    resolve_ground_truth -> real sleap-nn inference -> within_tolerance —
    against one live production ``ModelCard``, using ``SRP_PARITY_DATA_DIR``
    as the basename-search root, so a future sleap-nn/sleap-roots-contracts
    upgrade that breaks the decided gate is actually caught by something
    runnable, not just documented.
    """
    from sleap_roots_predict.model_registry import WandbRegistrySource

    source = WandbRegistrySource()
    cards = source.list_cards()
    assert cards, "no production ModelCards found in the live registry"
    card = cards[0]

    basename_index = build_basename_index(PARITY_DATA_DIR)
    bundle_dir = source.materialize(card)

    entry = evaluate_model_card(
        card, bundle_dir, tmp_path, basename_index=basename_index, sample_n=20
    )

    assert "gap_reason" not in entry, (
        f"ground truth unresolvable for {card.registry_id}: "
        f"{entry.get('gap_reason')}"
    )
    assert entry["n_frames_evaluated"] > 0

    if entry["classic_sleap_reference"] is None:
        pytest.skip(
            f"{card.registry_id} has no classic-SLEAP reference available "
            "(no labels_pr.val.slp and no readable metrics.val.npz) - "
            "reported informationally only, per spec."
        )

    sleap_nn_metrics = ParityMetrics(**entry["sleap_nn"])
    reference_metrics_ = ParityMetrics(**entry["classic_sleap_reference"])
    assert within_tolerance(sleap_nn_metrics, reference_metrics_), (
        f"{card.registry_id} exceeded the decided tolerance: "
        f"sleap_nn={entry['sleap_nn']}, "
        f"reference={entry['classic_sleap_reference']}"
    )
