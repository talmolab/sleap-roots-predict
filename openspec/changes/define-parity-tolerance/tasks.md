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
- [x] 5.2 **Implemented via a scratch harness script** (not a committed test — the real,
      network/registry-touching run is a one-time empirical measurement, not a standing test;
      the standing `parity`-marked test still exercises the reusable functions against
      fixtures). Ran against all 13 live `ModelCard`s with the real `prefix_map`
      (`D:/SLEAP` and `C:/Users/pbiobgh/Desktop/SLEAP` → `Z:/users/eberrigan/SLEAP`) and
      basename-search fallback at `n=100` sampled frames/model. Every one of the 13 produced a
      result (metrics + delta) — full coverage, no gaps, no model silently missing. Persisted to
      `docs/superpowers/specs/2026-08-04-define-parity-tolerance-results.json` via the new
      `build_report_entry()`/`write_parity_report()` (also newly added: full `ParityMetrics`
      capture — distance percentiles, PCK, visibility precision, OKS/VOC fields — not just the
      two gated numbers).

## 6. Empirical tolerance

- [x] 6.1 Ran the harness (see 5.2) at `n=100`. Deduplicating by `weights_checksum` (several
      `registry_id`s share physical weights) leaves 8 distinct models; measured `distance_p95`/
      `visibility_recall` deltas recorded in
      `docs/superpowers/specs/2026-08-03-define-parity-tolerance-design.md` §6 and the full
      per-model JSON report (2026-08-04). One model (`rice-cylinder-crown-age6-10`) diverged
      2-8x more than the rest in absolute px — investigated (not assumed): coarser metrics
      (pck@10px, visibility_precision) agree almost exactly, instance density is ~2x its own
      age2-5 sibling, and the growth stage is independently documented as less precise in
      Berrigan et al. 2024 (10.34133/plantphenomics.0175) — not an engine-parity bug.
- [x] 6.2 **Test first, then implementation:** added
      `test_within_tolerance_true_when_sleap_nn_recall_is_much_higher` (new — verifies the
      directional recall check) alongside the existing `within_tolerance` tests, updated to the
      new relative-distance signature. Then changed `within_tolerance()` in `parity.py` from a
      fixed-pixel `distance_tolerance_px` to a relative `distance_relative_tolerance` (measured
      max 17.0% across all 8 distinct models; gate set at 25% for headroom), and made
      `recall_tolerance` directional (only fails when sleap-nn scores lower; measured max
      -0.085, gate set at -0.10). No fixed-pixel exception needed for any model — see 6.1.
      (Corrected 2026-08-04, task 8.5: this previously said "-0.053" —
      `rice-cylinder-crown-age6-10`'s own figure, cited above for the outlier investigation —
      not the actual worst value across all 8 distinct models, which is -0.085
      (arabidopsis/canola/pennycress-primary-age2-14 shared group; see design.md §6's table).
      Both pass the -0.10 gate, so the decision is unaffected — caught by an adversarial
      `/review-pr` that verified this claim against the raw results JSON.)

## 7. Docs, cleanup, closing

- [x] 7.1 Updated `CHANGELOG.md` (`[Unreleased]`), `openspec/project.md` (contracts version
      literal `0.1.0a6`, a `parity.py` bullet in Architecture Patterns, a `parity` marker
      bullet in Testing Strategy, and the roadmap note reworded now that the harness has
      landed). Swept for stale `0.1.0a5` references — the remaining ones are legitimate
      historical mentions (this change's own docs describing the bump, and an archived prior
      change), not stale current-state claims. `API.md` intentionally unchanged —
      `parity.py` is not re-exported from `__init__.py` (an internal/harness module, per the
      design), so it's outside that doc's documented public-API surface.
- [x] 7.2 Full `/pre-merge` gate: `black --check` and `ruff check sleap_roots_predict/` and
      `codespell` all clean; `pytest -m "not gpu and not acceptance and not wandb" tests/` →
      269 passed, 1 skipped (parity test self-skips cleanly, no env vars set), 6 deselected;
      `pytest -m gpu tests/` → 3 skipped (this machine has no CUDA/MPS — this change touches
      no GPU-relevant code path, so that's a clean N/A, not a deferred verification); `uv
      build` → wheel builds successfully with `parity.py` included; no Dockerfile/image
      change in this branch, so no `docker build` needed.
- [x] 7.3 Closed predict#32 with a comment citing the existing skip-with-warning implementation
      (`model_registry.py`) and spec — no code change.
- [x] 7.4 Posted the sleap-roots-pipeline#15 closing comment (decided tolerance + measured
      baseline + reference-set coverage/gaps) and closed the issue. Opened
      [sleap-roots-pipeline#39](https://github.com/talmolab/sleap-roots-pipeline/pull/39) with
      the `docs/bloom-integration/roadmap.md` A3-predict row diff (top-line summary, sub-table
      cell, tracking-column row, and a new status-log entry).
- [x] 7.5 Cross-linked this change on `sleap-roots-training`#10/#11/#22 (downstream `LabelCard`
      consumer) and predict#8 (shared reference set + reusable `parity.py`). Also filed
      [sleap-roots-training#39](https://github.com/talmolab/sleap-roots-training/issues/39)
      (the shared-model-registry-duplication finding surfaced while building this harness) —
      drafted, shown, and approved before posting.
- [x] 7.6 Pushed the `define-parity-tolerance` branch and opened
      [sleap-roots-predict#33](https://github.com/talmolab/sleap-roots-predict/pull/33).

## 8. Post-review fixes (adversarial `/review-pr`, 5-lens)

A 5-lens adversarial review of PR #33 (Mode B, not posted to GitHub) found the harness's
resolution/matching/metrics logic sound, but the change's actual deliverable — an *enforced*
tolerance — wasn't functional as committed. Fixing the findings the user asked for, TDD-first.

- [x] 8.1 **Test first:** `test_within_tolerance_uses_decided_defaults_when_not_overridden` +
      `test_within_tolerance_defaults_fail_outside_decided_bounds` — calling
      `within_tolerance(a, b)` with no tolerance kwargs uses the decided constants.
      **Fix:** added `DECIDED_DISTANCE_RELATIVE_TOLERANCE = 0.25` / `DECIDED_RECALL_TOLERANCE
      = 0.10` module-level constants to `parity.py` (the numbers previously existed only in
      prose docs — nothing in code linked them to "the decided gate"), defaulted
      `within_tolerance`'s two kwargs to them. Existing tests that pass explicit tolerance
      values keep working unchanged.
- [x] 8.2 **Test first:** `test_within_tolerance_zero_reference_and_zero_sleap_nn_passes` +
      `test_within_tolerance_zero_reference_nonzero_sleap_nn_fails` — a `classic_sleap.
      distance_p95 == 0.0` reference must not raise `ZeroDivisionError` (reachable today:
      `test_reference_metrics_recomputes_when_labels_pr_present` already constructs exactly
      this case). Zero reference + zero sleap-nn distance is a perfect match → passes; zero
      reference + any nonzero sleap-nn distance is an infinite relative deviation → fails
      cleanly (no crash). **Fix:** guarded the division in `within_tolerance`.
- [x] 8.3 **Test first:** `test_evaluate_model_card_returns_report_entry` +
      `test_evaluate_model_card_returns_gap_entry_when_unresolvable` (fixture-based, real
      vendored sleap-nn model, no network) — a small, reusable orchestration function wrapping
      the existing resolve_ground_truth → sample_ground_truth → run_sleap_nn_predictions →
      compute_metrics → reference_metrics → build_report_entry pipeline for **one**
      `ModelCard`, extracted from the uncommitted scratch harness script that produced the
      checked-in results JSON (per the review: "the empirical results... came entirely from an
      uncommitted scratch script," "predict#8... can reuse `parity.py`... rather than
      duplicating instance-matching code"). **Fix:** added `evaluate_model_card(card,
      bundle_dir, workdir, *, labels_registry_lookup=None, prefix_map=None,
      basename_index=None, sample_n=None) -> dict` to `parity.py` — takes an already-
      materialized `bundle_dir` (not a `ModelCardSource`), matching `resolve_ground_truth`'s
      existing convention and keeping `parity.py` decoupled from `model_registry.py`.
- [x] 8.4 Rewrote `test_parity_harness_reports_all_production_models` to actually call
      `evaluate_model_card` against one real production `ModelCard` (via `WandbRegistrySource`
      + a basename search rooted at `SRP_PARITY_DATA_DIR`, which the skip reason had always
      named but no code ever read) and assert `within_tolerance` on the result when a
      reference is available — replacing the unconditional `pytest.skip()` that made this
      test a no-op regardless of env vars (confirmed: tasks.md's prior claim that "the
      standing parity-marked test still exercises the reusable functions against fixtures"
      was false). Verified it still skips cleanly, at collection time, without
      `SRP_PARITY_DATA_DIR`/`WANDB_API_KEY` set (`pytest -m parity` with both unset → 1
      skipped).
- [x] 8.5 Fixed the doc-accuracy bug the review caught: tasks.md 6.2 stated the "measured max"
      `visibility_recall` delta was `-0.053` — that's `rice-cylinder-crown-age6-10`'s own
      figure (cited there for the outlier investigation); the actual worst value across all 8
      distinct models is `-0.085` (the arabidopsis/canola/pennycress-primary shared group, per
      design.md §6's own table — which already had the correct number, confirmed; only 6.2's
      "measured max" prose was wrong). Both pass the `-0.10` gate, so the decision is
      unaffected, but the stated basis was wrong. Corrected 6.2 above.
- [x] 8.6 Full test suite (275 passed, 7 deselected) + `black`/`ruff`/`codespell` clean;
      `openspec validate --strict` passing; checkmarks updated; committed.
- [x] 8.7 Fixed a second stale-percentage bug (user-caught, 2026-08-05): §2's basename-search
      coverage claim ("`rice-cylinder-crown-age6-10` 54%, `rice-cylinder-primary-age2-5` 9%")
      in both `docs/superpowers/specs/2026-08-03-define-parity-tolerance-design.md` and
      `openspec/changes/define-parity-tolerance/design.md` was an early, pre-final-tuning
      measurement — the actual final harness run resolved **both to 100%** (confirmed directly
      against the results JSON's `n_frames_resolved`/`n_frames_total`). Reframed both passages
      as historical ("an early measurement found... later resolved to 100%") and added an
      explicit pointer to the results JSON as the living source of truth for current coverage,
      so a future reader (or harness re-run) doesn't have to re-derive this and a hardcoded
      percentage doesn't go stale silently again.

## 9. Reusable harness runner + results-schema docs

Root cause behind both round-3 staleness bugs: the results JSON was produced by an uncommitted
scratch script, so nobody could re-run the full 13-model harness without rebuilding it from
scratch. Brainstormed and designed in
`docs/superpowers/specs/2026-08-05-define-parity-tolerance-harness-runner-design.md` (approved,
revised after a `/review-openspec` adversarial round caught a data-loss-risk design flaw and a
mislabeled-schema doc bug before any code was written — see that doc's revision note).

- [ ] 9.1 **Test first, then implement** (one commit — matches this branch's established
      granularity; e.g. `3cb64cb` bundled four test+fix pairs, and the isolation *scenario* is
      already committed to the spec, so splitting the runner itself across two commits would
      leave one commit where committed spec text describes behavior the code doesn't have
      yet). Tests, all fixture-based/no-network (real `LocalCardSource`, not a hand-rolled
      stub — its `materialize` already raises `KeyError` for a card it has no entry for):
      - `test_run_parity_harness_writes_one_entry_per_card` — two resolvable fixture
        `ModelCard`s through a real `LocalCardSource` produce a JSON file with one entry per
        card, in input order, each with the full `build_report_entry` key set.
      - `test_run_parity_harness_returns_out_path` — the return value equals `out_path`.
      - `test_run_parity_harness_accepts_str_out_path` — a `str` `out_path` is coerced to
        `Path` up front, not left to fail only after every card has been evaluated.
      - `test_run_parity_harness_forwards_resolution_and_sampling_options` — `sample_n`/
        `prefix_map`/`basename_index`/`labels_registry_lookup` observably reach
        `evaluate_model_card` (via `ground_truth_source`/`n_frames_evaluated` in the output;
        no mocks needed).
      - `test_run_parity_harness_isolates_a_failing_card_and_warns` — a card absent from the
        `LocalCardSource` (real `KeyError`) becomes a gap entry naming the exception, tagged
        `gap_stage="evaluation"` (distinct from `evaluate_model_card`'s own
        `gap_stage="resolution"` gaps — see the sub-bullet below), with exactly one WARNING
        logged (`caplog`, mirroring `test_collect_cards_skips_malformed_and_warns`'s existing
        convention) naming that card's `registry_id`.
      - `test_run_parity_harness_failing_card_does_not_block_others_and_preserves_order` —
        cards `[good, bad, good]` all produce entries, in order, both good ones unaffected.
      - `test_run_parity_harness_does_not_swallow_keyboard_interrupt` — a stub source whose
        `materialize` raises `KeyboardInterrupt` propagates it (guards `except Exception`, not
        a bare `except`).
      - `test_run_parity_harness_all_cards_failing_does_not_clobber_an_existing_report` — an
        all-gap run does not overwrite a pre-existing report file at `out_path`.
      - `test_run_parity_harness_with_no_cards_writes_empty_report` — an empty `cards` list
        writes `[]` and returns `out_path` (pinned deliberately: no existing report to protect
        here, since there is nothing to compare non-gap-ness against).

      **Then implement:** `run_parity_harness(cards, source, workdir, out_path, *,
      labels_registry_lookup=None, prefix_map=None, basename_index=None, sample_n=None) ->
      Path` in `parity.py`. Coerces `out_path` to `Path` immediately. Converts each card via
      `card.to_model_ref(sleap_nn.__version__)` before calling `source.materialize(...)`
      (mirrors `warm_worker.py`'s existing conversion, rather than the `ModelCard`/`ModelRef`
      duck-typing shortcut one existing test uses). Wraps *only* the per-card
      `materialize`+`evaluate_model_card` call in `except Exception` (never bare `except`) —
      not `list_cards()` or any credential/setup step, which continue to propagate fail-loud,
      matching `model_registry.py`'s `_collect_cards` precisely (its own docstring: the `try`
      wraps only per-artifact construction). A caught exception becomes a gap entry with
      `gap_stage="evaluation"` plus the exception type/message. Before writing, if every entry
      is a gap and a report already exists at `out_path`, raises instead of overwriting. Uses
      `TYPE_CHECKING` for the `ModelCardSource` annotation (no runtime `model_registry` import
      into `parity.py`, preserving the decoupling task 8.3 established).

      **Also touches `evaluate_model_card`'s existing gap branch** (`parity.py:~1101-1106`):
      add `gap_stage="resolution"` to the dict it already returns, and update
      `test_evaluate_model_card_returns_gap_entry_when_unresolvable` to assert the new field
      (red without it, green after) — this is the discriminator the isolation tests above
      depend on to prove the two gap kinds are distinguishable.
- [ ] 9.2 Add `scripts/run_parity_harness.py` (the repo's first top-level `scripts/`
      directory): a committed, thin script hardcoding the two real prefix-map *source* keys
      (`D:/SLEAP`, `C:/Users/pbiobgh/Desktop/SLEAP` — already documented in `design.md`'s
      Decision 2, and immutable facts baked into the training-time `.slp` files), exposing
      `--share-root` (default the real `Z:/users/eberrigan/SLEAP` value — a CLI arg, not
      hardcoded, since this value is exactly as machine-specific as `SRP_PARITY_DATA_DIR`) and
      `--out` (default anchored via `Path(__file__).resolve().parents[1] /
      "docs/superpowers/specs/2026-08-04-define-parity-tolerance-results.json"` — not a
      CWD-relative string). Reads `WANDB_API_KEY`/`SRP_PARITY_DATA_DIR` from the environment
      (no new env vars — and `SRP_PARITY_DATA_DIR` is documented only in the script's own
      docstring + README, never added to `.env.example`/`EXPECTED_VARS`, since
      `tests/test_env_docs.py` asserts exact set equality there and that set is scoped to
      production/operator config, not test-gating vars). Calls
      `WandbRegistrySource().list_cards()` + `build_basename_index(...)`, then
      `run_parity_harness(..., sample_n=100)`. Module docstring states plainly: Windows + a
      `Z:` mapped share, this lab only, not portable, not shipped in the wheel/image, invoke
      via `uv run python scripts/run_parity_harness.py`. Not unit-tested beyond a `--help`
      smoke check (task 9.7) — thin, credential-requiring argument wiring, no other logic.
- [ ] 9.3 **Chore, separate commit:** extend the `black`/`ruff` targets to include `scripts` in
      `.github/workflows/ci.yml` and the three `.claude/commands/*.md` copies (`lint.md`,
      `fix-formatting.md`, `pre-merge.md`) — verified empirically that bare `codespell`
      already covers `scripts/` (no path arg) but `black --check`/`ruff check` currently do
      not, so the new script would otherwise be permanently exempt from this repo's formatting
      convention. Lands with or immediately after 9.2 (never referencing a not-yet-committed
      path in CI config).
- [ ] 9.4 Expand **and correct** `build_report_entry`'s docstring (`parity.py`): document
      `ground_truth_source`'s three values, `n_frames_resolved`/`n_frames_total`/
      `n_frames_evaluated`'s distinct meanings, each metrics dict's `settings` field (always
      `"recomputed"` on the `sleap_nn` side; `"recomputed"` or `"stored"` on the reference
      side — not two-valued on both sides as an earlier draft of this task's own design doc
      said), the nullability of `classic_sleap_reference`, and a note that `weights_checksum`
      is required to dedupe entries before summarizing (13 raw entries → 8 physically distinct
      models). **Correct an existing inaccuracy already shipped in this docstring**
      (`parity.py:991`, "plus the two gated deltas"): `distance_p95_delta`/
      `visibility_recall_delta` are unsigned absolute differences, informational only — the
      real gate (`within_tolerance`) computes a relative distance delta and a signed,
      directional recall delta from the two full metrics dicts, neither of which the `*_delta`
      fields can reproduce. Move the two gap-entry shapes (now distinguished by `gap_stage`)
      onto `write_parity_report`'s and `run_parity_harness`'s own docstrings instead of
      `build_report_entry`'s (which can't itself produce a gap entry). Add one sentence: this
      docstring is the living schema description; treat any other prose description of it
      (including the dated design docs) as a snapshot that can go stale. No signature change.
- [ ] 9.5 Add `specs/prediction-parity/spec.md`'s "Reusable Multi-Model Harness Runner"
      requirement, including the propagation and no-clobber scenarios added on revision
      (already drafted, done alongside this task list and its revision — landed in `0abf2b4`
      and the follow-up revision commit; no separate implementation commit needed for this
      task).
- [ ] 9.6 **Docs sweep**, mirroring task 7.1's precedent (missing from this section's first
      draft): fold an addition into `CHANGELOG.md`'s existing `[Unreleased]` parity bullet
      (not a new bullet — the harness hasn't shipped in a release yet) describing
      `run_parity_harness`/the script; add the regeneration command
      (`uv run python scripts/run_parity_harness.py`) and an explicit naming of
      `SRP_PARITY_DATA_DIR` to README's Parity Harness section, plus a `scripts/` entry in
      README's repo-tree diagram; add `run_parity_harness` + the regeneration script to
      `openspec/project.md`'s `parity.py` Architecture-Patterns bullet, and name
      `SRP_PARITY_DATA_DIR` in its parity Testing-Strategy bullet. State explicitly that
      `API.md` stays unchanged (no new public re-export), so it isn't re-litigated later.
- [ ] 9.7 Full `/pre-merge` gate (format, lint, test, build — now including `scripts/` per
      9.3), plus `uv run python scripts/run_parity_harness.py --help` exits 0 with no
      traceback, plus `openspec validate --strict`; commit.
