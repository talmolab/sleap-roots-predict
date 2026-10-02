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
- `run_batch` and `predict_and_write_batch` log one warning per clamped scan they predict (not on
  resume-skip). The warning lives in the batch entry points because selection runs twice per scan
  there (`resolve`, then again inside `predict`), so a warning in `choose_models` would log twice.
- Nothing new in provenance: saved `params` + `predict_models` + the registry windows suffice.
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
- **Deploy ordering (other repos):** after merge, re-pin the image in sleap-roots-pipeline
  (`sleap-roots-predictor-template.yaml` `image:` digest and `SRP_PREDICT_CONTAINER_DIGEST`
  together) alongside the traits re-pin. Until both are live a past-window scan fails as today, so
  either can ship first. Bloom's dialog warning ships last.
- **Out of scope:** per-root-type model choice / overrides threading (bloom#897, predict#22);
  species with no cards (bloom#993); younger-than-window (bloom#994); a shared matcher in
  sleap-roots-contracts (contracts#13/#14, phase 2).

## Rollback

Revert the squash commit and re-pin the previous image in sleap-roots-pipeline. Past-window scans
go back to failing with "no models resolved", before anything is written, so outputs from a
clamped run stay on disk but the scan is reported `failed`. Every other scan re-predicts once,
because the code sha changes.
