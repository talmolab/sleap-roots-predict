# Tasks: update-past-window-model-selection

Each code task is TDD (`/tdd`): **red** tasks are written and run failing before the code that
makes them pass; *characterization* tasks are expected green on first run and say what a red
means. Record each observed red in the commit body. Push only green heads. Tick each task in its
own section's commit.

**Git hygiene.**
- Stage by explicit path; never `git add -A` or `git add .`.
- The repo squashes with `COMMIT_MESSAGES`, so every commit body lands on `main`. No commit body or
  PR description may contain a closing keyword for bloom#971: write
  `Part of Salk-Harnessing-Plants-Initiative/bloom#971`, and drop the `Closes #…` line that
  `/pr-description` emits.

**The gate.** CI's commands (`ci.yml`) plus two local-only checks:

```
uv run pytest -m "not gpu and not acceptance and not wandb" tests/
uv run black --check sleap_roots_predict tests scripts
uv run ruff check sleap_roots_predict/ scripts/
uv run codespell
# local only, not in CI:
uv lock --check
openspec validate update-past-window-model-selection --strict
```

**Commit plan.** Commits 1–2 are pushed together (CI's path filter skips a docs-only head).

1. `docs: OpenSpec proposal for past-window model selection` — this change dir.
2. `test(selection): production-shaped card catalog` — section 1.
3. `feat(selection): clamp past-window ages to the species' highest window` — section 2.
4. `feat(batch): warn once per past-window scan` — section 3.
5. `docs: CHANGELOG for past-window selection` — section 4.
6. After the PR opens: `docs: tick live-registry checks` — 1.1 and 4.4, once recorded in the PR.

## 1. Production-shaped fixture

- [x] 1.1 Re-verify the live production catalog read-only (`WandbRegistrySource().list_cards()`)
      and record root type, model and selectors in the PR.
- [x] 1.2 Add `production_cards()` to `tests/card_builders.py`, mirroring the 8 cards' root types,
      registry-id stems and selectors as recorded in 1.1 (including both arabidopsis
      multiplant-cylinder selectors).
- [x] 1.3 *Characterization* (`test_model_selection.py`): in-window selection on that catalog —
      arabidopsis 10 and 14 → cpa-primary + arabidopsis-lateral; rice 4 → rice-younger-primary +
      rice-younger-crown; rice 8 → rice-older-crown; canola 13 → cpa-primary + canola-lateral;
      soybean 8 → soybean-primary + soybean-lateral. Pins today's in-window refs, so section 2
      can't change them.

## 2. Clamp in `choose_models` and `past_window_age`

Write and run 2.1–2.9 red before 2.10.

- [x] 2.1 Red: past-window cases on `production_cards()` — arabidopsis 28 → cpa-primary +
      arabidopsis-lateral; rice 18 → `rice-older-crown` only; soybean 10 → soybean-primary +
      soybean-lateral; canola 14 → cpa-primary + canola-lateral; pennycress 15 → cpa-primary +
      canola-lateral. Assert the exact root-type set and `registry_id`s. Red today: each is `{}`.
- [x] 2.2 Red: each past-window result equals the result at the window maximum (arabidopsis 28 ==
      14, rice 18 == 10, soybean 10 == 8, canola 14 == 13, pennycress 15 == 14); age 365 and
      string age `"28"` clamp too.
- [x] 2.3 Red: the window maximum is scoped by mode — cards (canola, cylinder, 2–13) and (canola,
      multiplant cylinder, 2–20), scan canola cylinder 15 → the cylinder card is selected.
- [x] 2.4 Red: no lower-window fallback — lateral cards (arabidopsis 2–10) and (arabidopsis 2–14),
      scan 28 → only the 2–14 card, no ambiguity error.
- [x] 2.5 Red: ambiguity at the matching age — two lateral cards both covering arabidopsis 14,
      scan 28 → `ValueError` whose message names 28 and 14.
- [x] 2.6 Overrides: arabidopsis 28 with primary overridden → the override plus arabidopsis-lateral
      (red: lateral is `{}` today). *Characterization:* primary card rice 2–10 overridden, lateral
      rice 2–5, scan 18 → only the override (lateral is skipped today too).
- [x] 2.7 *Characterization* controls: younger than every window (arabidopsis 1, canola 0) → `{}`;
      no-card species (`sorghum` 30) → `{}`; cards only in another mode (canola, multiplant
      cylinder, 20) → `{}`; `cards=[]` at age 100 → `{}`; overrides only, no cards → the
      overrides; gap age 7 on the disjoint card → `{}`; a non-integer age with `cards=[]` still
      raises. After a clamped call, `params.values["age"]` is unchanged, `param_hash` equals a
      fresh `ResolvedParams` with the real age, and `caplog` shows no record from
      `sleap_roots_predict.model_selection`.
- [x] 2.8 Red: `past_window_age` unit tests, one per spec scenario of "Past-Window Matching Age
      Helper": 14 for arabidopsis 28 and `"28"`; 13 for the mode-scoped case; 10 for the
      overridden-primary case; `None` for arabidopsis 10 and 14, rice 4, canola 13, arabidopsis 1,
      no-card, other-mode and `cards=[]`; `None` when every root type with a selector for the
      species and mode is overridden (rice 18, primary rice 2–10 overridden, only lateral card
      arabidopsis); the same `ValueError` as `choose_models` for a bad age with `cards=[]`.
- [x] 2.9 Red: end-to-end — `test_batch.py`: `run_batch` with `rice_source` (rice 2–5) and one
      rice day-9 scan ends `ok`; its manifest's artifacts carry the rice refs; the copied sidecar
      (`out/<key>/<key>.scan_metadata.json`) keeps `params.age == 9`. Re-running skips it;
      rewriting the input sidecar to day 5 then re-predicts (`ok`), so the key carries the real
      age. `test_param_resolution.py`: `choose_models(resolve_params(_row("Pennycress",
      plant_age_days=20)), cards)` over the multi-selector card already in
      `test_round_trip_selects_a_multi_selector_card` (canola 2–13, pennycress 2–14) equals the
      day-14 result. Red today: `failed` "no models resolved", and `{}`.
- [x] 2.10 Update the tests whose "no match" examples are above-window:
      - `test_string_age_at_a_per_selector_boundary[canola-14-False]` and
        `test_inclusive_boundaries_per_selector[14-False]` → expect a match (clamped to 13). These
        two go red before 2.11. Add `("canola", "1", False)` to the string-age test
        (*characterization*).
      - *Characterization* (green after the edit): `test_age_outside_window_skips_root_type` →
        age 1, renamed to match the spec scenario;
        `test_age_compared_against_the_matching_selectors_window_only` → canola 5–13 + pennycress
        2–14, canola age 3 → `{}`; `tests/test_canary_check.py::test_every_context_is_checked` →
        context rice age 1, asserting `"rice|cylinder|1"`.
- [x] 2.11 Green: in `sleap_roots_predict/model_selection.py`, factor the existing param validation
      and age coercion into one helper used by both functions; add `past_window_age` (validate
      first, then `max(..., default=None)`); match at its window maximum in `choose_models`; name
      both ages in the ambiguity error. `params` is never mutated and nothing is logged. Update the
      module, `choose_models` and `_card_matches` docstrings (the `age` argument is the matching
      age).
- [x] 2.12 *Characterization*: `test_zero_resolved_models_is_failed` (soybean on `rice_source`, no
      cards for the species) stays `failed`.

## 3. One warning per clamped scan

- [x] 3.1 (`test_batch.py`): extend 2.9's test functions with `caplog` assertions, so no new
      runs. Red: the first day-9 `run_batch` logs exactly one WARNING from logger
      `sleap_roots_predict.batch` naming the scan key, `rice`, `cylinder`, 9 and 5.
      *Characterization:* the resumed (skipped) run and an in-window day-3 scan log none.
- [x] 3.2 (`test_output_contract.py`): red: `predict_and_write_batch` with `rice_source` logs
      exactly one such WARNING from `sleap_roots_predict.output_contract` for a day-9 request.
      *Characterization:* none for a day-3 request, or for a day-9 request overriding both root
      types (`{"primary": _ref("primary"), "lateral": _ref("lateral", "reg/rice-lateral")}`).
- [x] 3.3 Green:
      - `batch.py` `run_batch`: inside the per-scan `try`, between the resume-skip `continue`
        and `_predict_one` (so a validation raise stays isolated to the scan), call
        `past_window_age(scan.params, worker.load_catalog())` (no overrides; `load_catalog()` is
        cached) and log the warning when it returns an age.
      - `output_contract.py`: add `import logging` and `logger = logging.getLogger(__name__)`; in
        `predict_and_write_batch`, after `resolve` and before `predict`, call
        `past_window_age(req.params, worker.load_catalog(), req.overrides)` and log.
      - Add one sentence on the warning to the `run_batch` and `predict_and_write_batch`
        docstrings.

## 4. Docs and validation

- [x] 4.1 `CHANGELOG.md` `[Unreleased]`: append one short entry to the existing `### Changed`
      (not `### Changed (BREAKING)`; no API breaks): the clamp rule, the real age kept, the
      warning, a pointer to the `model-management` spec, and
      `Part of Salk-Harnessing-Plants-Initiative/bloom#971`.
- [x] 4.2 Run the gate; all green.
- [x] 4.3 `openspec list` shows no other active change touching these requirements.
- [x] 4.4 Read-only check against the live registry from the branch, recorded in the PR:
      `uv run python scripts/canary_check.py --context arabidopsis,cylinder,28 --expect-roots
      primary,lateral` and `--context rice,cylinder,18 --expect-roots crown`.
