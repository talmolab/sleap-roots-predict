# Change: Select past-window scans with their species' highest-age window

Phase 1 of Salk-Harnessing-Plants-Initiative/bloom#971, predict side. Decisions:
[bloom#971 decisions](https://github.com/Salk-Harnessing-Plants-Initiative/bloom/issues/971#issuecomment-5937500723),
[update](https://github.com/Salk-Harnessing-Plants-Initiative/bloom/issues/971#issuecomment-5938030166).
The traits side (`choose_pipeline` in sleap-roots) is a separate change in that repo.

## Why

Every production selector ends at a fixed `age_max`, so a scan older than every window for its
species and mode matches no card for any root type, and predict fails it with "no models
resolved" (`batch.py`). Per bloom#971, ~21.9k staging cyl scans were past their species' window;
bloom#971 decided to run them with the species' highest-age window by default.

## What Changes

- `choose_models` matches a past-window scan at its species and mode's **window maximum** (the
  highest `age_max` across all cards), so a root type whose cards don't reach it is skipped. The
  real params, `param_hash` and idempotency key keep the scan's real age. Rules and examples are
  in the `model-management` delta.
- New pure helper `model_selection.past_window_age(params, cards, overrides=None)`, module-level
  and **not** exported from the package. `choose_models` stays pure and silent.
- `run_batch` and `predict_and_write_batch` log one warning per clamped scan **after it is
  predicted successfully** (not on resume-skip, and not for a scan that fails, so a failing scan
  isn't re-warned on every rerun). The message starts `past-window age:`, like the trait
  extractor's, so one search finds both. The warning lives in the batch entry points because
  selection runs twice per scan there (`resolve`, then again inside `predict`), so a warning in
  `choose_models` would log twice.
- Nothing new in provenance: saved `params` + `predict_models` + those cards' selector windows
  identify a clamp. This assumes a card version's `selectors` metadata is never edited in place:
  W&B allows editing it, and `ModelRef` doesn't record selectors.
- The ambiguity error names both the scan age and the matching age when they differ.

## Impact

- Affected specs: `model-management` (MODIFIED: Model Selection From Scan Params; ADDED:
  Past-Window Matching Age Helper), `predict-container` (ADDED: Past-window scan warning),
  `prediction-output` (ADDED: Past-window scan warning in batch prediction-and-write).
- Affected code: `sleap_roots_predict/model_selection.py` (helper + clamp),
  `sleap_roots_predict/batch.py` and `sleap_roots_predict/output_contract.py` (one warning per
  clamped scan). No worker API change; no dependency change.
- Affected tests: `tests/card_builders.py` (shared `production_cards()`),
  `tests/test_model_selection.py`, `tests/test_batch.py`,
  `tests/test_output_contract.py`, `tests/test_param_resolution.py`, and
  `tests/test_canary_check.py` (`test_every_context_is_checked` uses rice day 9 on a 2–5 card as
  its "resolves nothing" context; it moves to a below-window age).
- Affected docs: `CHANGELOG.md`. `API.md` and `openspec/project.md` defer to the spec for the
  matching rules.
- Other callers whose output changes: `scripts/canary_check.py` and
  `scripts/a1_selection_oracle.py` call `choose_models` directly, so they now report matches for
  past-window contexts. A1 tables regenerated after this change differ in those cells.
- **Selected refs for in-window scans are unchanged.** Resume-skip is not preserved across the
  re-pin regardless: the idempotency key includes `predict_code_sha`, which every CI-built image
  bakes in (`docker-build.yml`), so a new image re-predicts every scan.
- **Deploy ordering (other repos):** after merge, re-pin `sleap-roots-predictor-template.yaml`
  in sleap-roots-pipeline: the `image:` `:sha-<commit>` tag and digest, and
  `SRP_PREDICT_CONTAINER_DIGEST`, together (`scripts/check_manifests.py` enforces both). The code
  PRs can merge in either order, but **re-pin both templates in one sleap-roots-pipeline PR, or
  traits first**:
  - traits first: a past-window scan fails as today, at predict ("no models resolved", exit 3,
    retried);
  - predict first: it runs GPU inference, writes predictions, then fails at traits ("No pipeline
    matches", exit 3, retried twice). The exit gate passes exit 3 either way, and those
    predictions are reused once traits is re-pinned.

  Bloom's dialog warning ships last.
- **Arabidopsis multiplant cylinder past day 14** now gets predictions (cpa-primary and
  arabidopsis-lateral carry multiplant selectors), but traits' scan-grain guard still rejects
  multi-plant scans (talmolab/sleap-roots#252), as it does in-window ones. This adds a GPU pass to
  an existing gap rather than creating one.
- **Parity rests on data.** Predict's window maximum comes from the live registry; traits' from
  its packaged `pipeline_selection.yaml`. They match today (arabidopsis 14, canola 13, pennycress
  14, soybean 8, rice 10), but a promotion that changes a window would break parity without a code
  change. A guard (canary or promotion-checklist item) is a follow-up.
- **Out of scope:** per-root-type model choice / overrides threading (bloom#897, predict#22);
  species with no cards (bloom#993); younger-than-window (bloom#994); a shared matcher in
  sleap-roots-contracts (contracts#13/#14, phase 2).

## Rollback

Revert the squash commit and re-pin the previous image (tag, digest and
`SRP_PREDICT_CONTAINER_DIGEST`). New past-window scans go back to failing at predict with "no
models resolved", before anything is written. Predictions already written for clamped scans are
not removed, and a clamping traits image still reads them, so roll traits back too to stop
past-window results. Every other scan re-predicts once, because the code sha changes. The squash
also carries the unrelated `.claude/commands/new-feature.md` fix (step 8 → `/openspec:apply`); a
revert undoes it too, so re-apply it if needed.
