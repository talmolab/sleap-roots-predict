## 1. Dependency bump (no behavior change)

- [ ] 1.1 Bump `sleap-roots-contracts` `0.1.0a5` → `0.1.0a6` in `pyproject.toml`; run `uv lock`
      (diff scoped to the `sleap-roots-contracts` entry, per the #29/#30 precedent). No test
      written for this task — it's a pure version bump. Verification: full existing test suite
      still green (regression baseline before any code changes), confirming the retyped
      `ModelCard.mode` doesn't break `model_registry.py`'s existing behavior or tests.
- [ ] 1.2 Add the `parity` marker to `pyproject.toml`'s `markers` list, alongside
      `gpu`/`acceptance`/`wandb`.

## 2. Ground-truth resolution

- [ ] 2.1 **Test first:** `tests/test_parity.py::test_resolve_ground_truth_prefers_labels_registry`
      — given a fake `ModelCard` and a stub labels-source object exposing a matching
      collection (species/root-type/node-count), resolution returns that collection's labels
      without attempting bundled-labels path relinking. **Then implement:**
      `parity.resolve_ground_truth(card, labels_source, bundle_dir, prefix_map)` in
      `sleap_roots_predict/parity.py`, checking the labels-registry branch first.
- [ ] 2.2 **Test first:** `test_resolve_ground_truth_falls_back_to_relinked_bundle` — given no
      matching labels-registry collection, but a small fixture `labels_gt.val.slp` (in
      `tests/assets/`) whose embedded video path matches a configured prefix map entry pointing
      at a fixture image directory, resolution returns those relinked, loadable frames.
      **Then implement:** the `Labels.replace_filenames()` fallback branch.
- [ ] 2.3 **Test first:** `test_resolve_ground_truth_reports_explicit_gap` — given no
      labels-registry match and an unresolvable bundle path, resolution returns a gap marker
      (not an exception), and a second, resolvable `ModelCard` processed in the same batch still
      resolves normally. **Then implement:** the gap branch + batch-level "continue on gap"
      behavior.

## 3. Metric computation

- [ ] 3.1 **Test first:** `test_compute_parity_metrics_uses_centroid_matching` — given two small
      synthetic `.slp` fixtures (ground truth + a near-identical "prediction"), call the
      `parity.py` wrapper around `run_evaluation` and assert it was invoked with
      `match_method="centroid"`, and that the returned structure exposes `distance_metrics`/
      `visibility_metrics` but does not surface `mOKS`/`voc_metrics` as part of the gating
      result. **Then implement:** `parity.compute_metrics(ground_truth_path, predicted_path)`.
- [ ] 3.2 **Test first:** `test_reference_metrics_recomputed_when_labels_pr_present` — given a
      fixture bundle with both `labels_gt.val.slp` and `labels_pr.val.slp`, the classic-SLEAP
      reference number comes from a fresh `run_evaluation` call with the same settings as 3.1,
      not from any stored file. **Then implement:** the recompute branch.
- [ ] 3.3 **Test first:** `test_reference_metrics_fall_back_to_stored_npz` — given a fixture
      bundle with only `metrics.val.npz` (no `labels_pr.val.slp`), the classic-SLEAP reference
      number comes from `sleap_nn.evaluation.load_metrics()`, and the result is marked
      `settings="stored"` (vs. `"recomputed"` in 3.2). **Then implement:** the fallback branch +
      the settings-provenance marker.

## 4. LabelCard-shaped manifest

- [ ] 4.1 **Test first:** `test_manifest_records_validate_as_labelcard` — every record in the
      checked-in ground-truth manifest file constructs a valid `sleap_roots_contracts.LabelCard`
      when loaded. **Then implement:** build the manifest (JSON, one record per production
      `ModelCard` resolved in task 2, gaps included as an explicit list alongside it) and a
      small loader in `parity.py`.
- [ ] 4.2 **Test first:** `test_manifest_unrecoverable_fields_are_none_not_fabricated` — for a
      manifest record sourced via bundled-labels relinking (which carries no Bloom/accession
      provenance), the corresponding `LabelCard` provenance fields are `None`. **Then verify:**
      no fabricated placeholder values anywhere in the manifest (a self-review grep, not just
      the unit test).

## 5. Parity marker + real harness test

- [ ] 5.1 **Test first:** `test_parity_marker_skips_without_env_vars` — collecting
      `tests/test_parity.py` with the required env vars (network-share root, `WANDB_API_KEY`)
      unset skips the `@pytest.mark.parity` tests at collection time, mirroring the existing
      `acceptance`/`wandb` skip tests. **Then implement:** the `skipif` guard.
- [ ] 5.2 **Test first (real-data, gated):** `test_parity_harness_reports_all_production_models`
      — with env vars set, running the harness produces one result (metrics + delta, or an
      explicit gap) per production `ModelCard`, with no model silently missing from the report.
      **Then implement:** the end-to-end harness entry point wiring tasks 2–4 together.

## 6. Empirical tolerance

- [ ] 6.1 Run the `parity` marker test locally (`uv run pytest -m parity -s`) against the live
      registry + resolved ground truth. Record the observed `distance_metrics`/
      `visibility_metrics.recall` deltas per model in this file and in
      `docs/superpowers/specs/2026-08-03-define-parity-tolerance-design.md`.
- [ ] 6.2 **Test first:** `test_tolerance_assertion_passes_within_bound` /
      `test_tolerance_assertion_fails_beyond_bound` — using the values from 6.1 plus a
      documented margin, unit-test the pass/fail assertion logic directly (not the full network
      harness) with synthetic deltas just inside and just outside the tolerance. **Then
      implement:** the tolerance constants (in `parity.py`, with a comment citing 6.1's
      measurement) and wire the assertion into the task-5.2 harness test.

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
