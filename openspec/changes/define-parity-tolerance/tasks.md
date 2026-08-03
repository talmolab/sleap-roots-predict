## 1. Dependency bump (no behavior change)

- [x] 1.1 Bump `sleap-roots-contracts` `0.1.0a5` → `0.1.0a6` in `pyproject.toml`; run `uv lock`
      (diff scoped to the `sleap-roots-contracts` entry, per the #29/#30 precedent). No test
      written for this task — it's a pure version bump. Verification: full existing test suite
      still green (regression baseline before any code changes), confirming the retyped
      `ModelCard.mode` doesn't break `model_registry.py`'s existing behavior or tests. Also
      re-ran the `wandb`-marked tests against the live registry post-bump (2 passed) — direct
      confirmation of predict#32's actual behavior, not just a static read of the code.
- [x] 1.2 Add the `parity` marker to `pyproject.toml`'s `markers` list and `addopts`, alongside
      `gpu`/`acceptance`/`wandb`.

## 2. Ground-truth resolution

- [x] 2.1 **Test first:** `tests/test_parity.py::test_resolve_ground_truth_prefers_labels_registry`
      — given a fake `ModelCard` and a stub `labels_registry_lookup` callable returning a
      sentinel path, resolution returns that path without attempting bundled-labels relinking.
      **Implemented:** `parity.resolve_ground_truth(card, bundle_dir, workdir,
      labels_registry_lookup=..., prefix_map=...)`, checking the labels-registry branch first.
- [x] 2.2 **Test first:** `test_resolve_ground_truth_falls_back_to_relinked_bundle` — given no
      matching labels-registry collection, but a fixture `labels_gt.val.slp` (built in-test with
      a real `sio.Video`/`sio.Labels`, not a committed binary fixture) whose broken video path
      matches a configured prefix map entry pointing at the vendored
      `tests/assets/images/centered_pair` frames, resolution returns those relinked, loadable
      frames. **Implemented:** `parity.relink_ground_truth` (`Labels.replace_filenames()` +
      verify frame 0's image actually loads) wired as the fallback branch.
- [x] 2.3 **Test first:** `test_resolve_ground_truth_reports_gap_without_raising` +
      `test_resolve_ground_truth_gap_does_not_block_other_models` — given no labels-registry
      match and an unresolvable bundle path, resolution returns a `GapRecord` (not an exception),
      and a second, resolvable `ModelCard` processed independently still resolves normally.
      **Implemented:** the `GapRecord` branch.
- [x] 2.4 **Investigated against the live network share (not originally scoped, found
      necessary):** checked `Z:\users\eberrigan\SLEAP\SLEAP_Rice`/`SLEAP_Soy` per the design
      doc's open question. Two more confirmed prefix-map entries
      (`C:/Users/pbiobgh/Desktop/SLEAP` → the same `Z:/users/eberrigan/SLEAP` tree) brought
      prefix-map coverage to **8/13 models**. The remaining 5 (rice ×3, soybean ×2) don't resolve
      via any prefix map — `SLEAP_Rice`'s layout was reorganized since training, and 2 of the 3
      rice models plus both soybean models reference paths not on the network share at all
      (Box-synced folders, `E:`/`F:` drives). A basename search across the whole indexed
      `SLEAP_Rice`/`SLEAP_Soy` tree finds every basename (0 missing), but often **ambiguously**
      (the same basename recurs in multiple folders) — confirmed via file-content-hash
      comparison that these are genuinely different scans of the same plant at different
      timepoints, not accidental duplicate files, so basename alone cannot disambiguate.
      **Test first:** `test_pick_best_candidate_*` (single-candidate shortcut, parent-folder-name
      exact match, age-hint-in-range, segment-overlap fallback, genuine-tie-returns-None).
      **Implemented:** `_pick_best_candidate`, `build_basename_index`. Measured resolution rate
      on the live tree: `rice-crown-age2-5` 100%, `rice-crown-age6-10` 54%, `rice-primary-age2-5`
      9%, both soybean models 100%.
- [x] 2.5 **Test first:** `test_relink_ground_truth_by_basename_search_partial_resolution` +
      `test_resolve_ground_truth_uses_basename_search_as_last_resort` — a bundle with one
      resolvable and one unresolvable video yields a filtered ground truth containing only the
      resolved frame, wired as `resolve_ground_truth`'s third tier. **Implemented:**
      `relink_ground_truth_by_basename_search`; `ResolvedGroundTruth` gained
      `n_frames_resolved`/`n_frames_total` so partial coverage is reported, not collapsed to a
      binary pass/gap. **Also upgraded** `relink_ground_truth` (the prefix-map tier) to the same
      keep-only-loadable-frames behavior — it previously only checked frame 0, which would have
      silently included any other unresolved frames in a mostly-working prefix map.
      `specs/prediction-parity/spec.md`'s "Ground Truth Resolution Per Model" requirement
      rewritten for three tiers + frame-level (not per-model) resolution; new "Basename Search
      Disambiguation" requirement added.

## 3. Metric computation

- [x] 3.1 **Test first:** `test_compute_metrics_gives_real_per_node_distance_and_excludes_oks` —
      given two small in-test `.slp` fixtures (ground truth + a known +1px-per-node shifted
      "prediction"), call the `parity.py` wrapper and assert the returned distance matches the
      known shift (`sqrt(2)`), and that `ParityMetrics` has no `mOKS`/`oks_map` attributes.
      **Implemented:** `parity.compute_metrics(ground_truth_path, predicted_path)`.
      **Correction found during implementation:** the original plan called for
      `match_method="centroid"` to avoid OKS. Tried it first and rejected it — confirmed
      empirically that centroid mode is designed for single-node/centroid-only predictions, not
      per-node distance between two full multi-node skeletons (it produced a nonzero distance
      even for two *identical* instances on a real 2-node skeleton). The actual fix:
      `match_method="oks"` at the library's own permissive default `match_threshold=0.0` for
      *matching*, while still never reading OKS-derived *score* fields (`mOKS`, `voc_metrics`) —
      this is what `sleap-roots-training`#17 actually does. `ParityMetrics.visibility_recall`
      (not `detection_recall` — that field only exists in centroid mode's differently-shaped
      result) is populated from `visibility_metrics.recall`, which only OKS mode returns.
      `design.md`/`proposal.md`/`specs/prediction-parity/spec.md` updated to match.
- [x] 3.2 **Test first:** `test_reference_metrics_recomputes_when_labels_pr_present` — given a
      fixture bundle with both `labels_gt.val.slp` and `labels_pr.val.slp` (identical points),
      the classic-SLEAP reference number comes from a fresh `run_evaluation` call with the same
      settings as 3.1 (`distance_avg == 0.0`), not from any stored file. **Implemented:** the
      recompute branch in `parity.reference_metrics`.
- [x] 3.3 **Test first:** `test_reference_metrics_returns_none_when_nothing_available` (no
      `labels_pr.val.slp`/`metrics.val.npz` at all → `None`, not a raise) and
      `test_reference_metrics_reads_real_legacy_stored_npz_via_shim` (a real stored
      `metrics.val.npz`, committed as `tests/assets/legacy_metrics/
      rice_cylinder_primary_age2-5.metrics.val.npz`, reads successfully). **Correction found
      during implementation, then resolved:** every real `metrics.val.npz` checked (materialized
      from the live registry) is pickled by classic SLEAP's own (TensorFlow-based) `sleap`
      package — `load_metrics()` raises `ModuleNotFoundError: No module named 'sleap'` with only
      `sleap_nn` installed. Adding that legacy dependency here would undermine this repo's whole
      purpose. Disassembling the pickle opcodes showed only one custom class is referenced
      (`sleap.instance.PointArray`), so **implemented** `parity._legacy_sleap_unpickle_shim`: a
      temporary, minimal `numpy.ndarray`-subclass stand-in registered in `sys.modules` only for
      the duration of the read, then removed. Verified against all 13 live production models'
      real stored files (not just the committed fixture) — full stored-reference coverage, not a
      rare fallback. Also found: the stored file's key schema is classic SLEAP's own flat,
      dot-separated one (`dist.p95`, `vis.recall`), different from `sleap_nn.evaluation`'s nested
      `distance_metrics`/`visibility_metrics` — `reference_metrics` translates it.
      `reference_metrics` returns `Optional[ParityMetrics]`; `None` is reserved for the residual
      case where neither source is available/readable at all (expected to be rare).
      `specs/prediction-parity/spec.md`'s "Classic-SLEAP Reference Number" and "Documented,
      Enforced Tolerance" requirements updated accordingly.

## 4. LabelCard-shaped manifest

- [x] 4.1 **Test first:** `test_build_label_card_derives_content_fields` — building a
      `LabelCard` from a real in-test `.slp` fixture derives `node_count`/`node_names`/
      `n_frames`/`n_instances` correctly and carries through the `ModelCard`'s
      species/mode/root_type/age/registry_id/version. **Implemented:**
      `parity.build_label_card(labels_path, card, images_embedded=..., **provenance)`.
- [x] 4.2 **Test first (same test):** unrecoverable provenance fields
      (`source_experiment`/`bloom_experiment_id`/`accessions`/`labeler`) default to `None` when
      not passed, not fabricated. **Verified:** no fabricated placeholder values in
      `build_label_card`'s implementation (all optional args default `None`, no synthesized
      strings). The actual checked-in multi-model manifest (one record per production
      `ModelCard`, built by calling `build_label_card` against each resolved ground truth from
      task 2, plus an explicit gap list) is produced together with task 5.2/6's live run — it
      depends on which of the 13 models actually resolve, which isn't known until that run.

## 5. Parity marker + real harness test

- [x] 5.1 **Test first:** `test_relink_ground_truth_returns_none_when_bundle_missing_file` (unit)
      plus the `@pytest.mark.parity` + `skipif(not (SRP_PARITY_DATA_DIR and WANDB_API_KEY))`
      guard on `test_parity_harness_reports_all_production_models`, mirroring the existing
      `acceptance`/`wandb` skip pattern exactly. **Implemented and verified:** collecting
      `tests/test_parity.py` with no env vars set skips that test (confirmed via the full suite
      run: `251 passed, 7 deselected`, one more deselected than before this change landed).
- [ ] 5.2 **Not yet implemented — real-data, gated, tracked as a follow-up within this same
      change before merge:** `test_parity_harness_reports_all_production_models` currently
      `pytest.skip()`s with an explicit note. Wiring this up requires, per model card: (a) a
      real `labels_registry_lookup` implementation against `wandb-registry-sleap-roots-labels`
      (species/root-type/node-count join against the 8 real collections), (b) the real
      per-species `prefix_map` entries (confirmed working: `D:/SLEAP` → the arabidopsis/
      canola/pennycress video pool; still to check: `Z:\users\eberrigan\SLEAP\SLEAP_Rice` and
      `...\SLEAP_Soy`, pointed to directly by the repo owner, for the `D:/FNRice*`,
      `C:/Users/pbiobgh`, `E:/Soy_GDM_Brazil`, `F:/Soy_GDM_Brazil` prefixes), and (c) materializing
      all 13 live `ModelCard`s via `WandbRegistrySource`. **Test:** the harness produces one
      result (metrics + delta, or an explicit gap) per production `ModelCard`, with no model
      silently missing from the report.

## 6. Empirical tolerance

- [ ] 6.1 Run the wired-up `parity` marker test locally (`uv run pytest -m parity -s`) against
      the live registry + resolved ground truth. Record the observed `distance_metrics`/
      `visibility_metrics.recall` deltas per model in this file and in
      `docs/superpowers/specs/2026-08-03-define-parity-tolerance-design.md`.
- [ ] 6.2 **Test first:** `test_within_tolerance_true_when_deltas_are_small` /
      `test_within_tolerance_false_when_*_delta_too_large` — already implemented and passing
      against placeholder tolerance values; once 6.1's real numbers are in, revisit whether the
      placeholder margins still make sense and update `parity.py`'s tolerance constants with a
      comment citing 6.1's measurement.

## 7. Docs, cleanup, closing

- [ ] 7.1 Update `CHANGELOG.md` (`[Unreleased]`), `openspec/project.md` (contracts version
      literal, a `parity.py` bullet in Architecture Patterns, and the roadmap note at the top
      that currently lists "the prediction-parity harness" as remaining A3/A4 work — drop or
      reword now that it's landing), and any stale doc references (grep sweep).
- [ ] 7.2 Full `/pre-merge` gate: format, lint, test (including `-m gpu` locally per the
      standing requirement), build.
- [ ] 7.3 Close predict#32 with a comment citing the existing skip-with-warning implementation
      (`model_registry.py`) and spec — no code change.
- [ ] 7.4 Draft the sleap-roots-pipeline#15 closing comment (decided tolerance + measured
      baseline + reference-set coverage/gaps) and the `docs/bloom-integration/roadmap.md`
      A3-predict row diff; show both for approval before posting/committing.
- [ ] 7.5 Cross-link this change on `sleap-roots-training`#10/#11/#22 (downstream `LabelCard`
      consumer) and predict#8 (shared reference set + reusable `parity.py`) — comment drafts
      shown for approval first.
