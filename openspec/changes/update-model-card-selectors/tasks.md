# Tasks: update-model-card-selectors

Every code task is red → green → refactor: write the named test(s) first, run them, see them fail
for the stated reason, then write the minimum code to pass. Record each observed red in the commit
body. Commit red states locally if useful, but push only green heads. Stage by explicit path in
every commit (`HANDOFF-predict34.md` is untracked and excluded via `.git/info/exclude`; never
`git add -A` on this branch). Tick a box only after its verification has actually run.

**Verification commands** (referred to below as "the gate"):

```
uv run pytest -m "not gpu and not acceptance and not wandb" tests/   # CI's exact expression
uv run ruff check .
uv run black --check .
uv run codespell
uv lock --check
openspec validate update-model-card-selectors --strict
```

## 1. Prep refactor — green on contracts a7 (commit: `refactor(tests): ...`)

- [ ] 1.1 Replace the seven local `_card` helpers (`conftest.py`, `test_model_selection.py`,
      `test_output_contract.py`, `test_param_resolution.py`, `test_parity.py`,
      `test_warm_worker.py`, the inline one in `test_model_registry.py`) and the inline
      `ModelCard(` in `test_batch.py` with one shared builder in `tests/conftest.py` taking
      `selectors=((species, mode, age_min, age_max), ...)`. On a7 it asserts exactly one selector
      and builds a flat card. Also add a shared raw-metadata builder that the wandb fixtures
      (`_good_meta`, `_malformed_artifact`, `test_model_registry.py:164-197`) use. No assertion
      changes. Verify: the gate is green on a7, and
      `grep -rnE "ModelCard\((.|\n)*?(species|age_min|age_max)=" tests/` finds only the builder.

## 2. The breaking commit — pin a9 + selection (commit: `feat(selection)!: ...`)

This commit is atomic by necessity: after the pin bump, `model_selection.py` and the parity
report entry both fail on a flat read, so neither can land separately green.

- [ ] 2.1 Bump `sleap-roots-contracts` to `==0.1.0a9` with
      `uv lock --upgrade-package sleap-roots-contracts`. Verify: `git diff uv.lock` changes only
      that package; `uv run python -c "import sleap_roots_contracts as c; print(c.Selector)"`.
      Switch the two shared builders to emit `Selector`s / JSON `"selectors": [{...}]`. Record the
      red count of the suite in this state.
- [ ] 2.2 Contract-assumption tests (deploy ordering rests on them): `ModelCard.model_validate` of
      a flat dict raises; `selectors=[]` raises.
- [ ] 2.3 Selection tests first (`test_model_selection.py`): any-selector (pennycress/14 on
      canola 2–13 + pennycress 2–14); per-selector age (canola/14 on the same card does not match);
      disjoint windows (canola 2–5 + canola 10–13, age 7 → no match); no cross product (canola
      2–13 + arabidopsis multiplant 2–14; canola/multiplant at ages 2, 5, 13, 14 → no match);
      overlapping selectors on one card → selected, no raise; two distinct matching cards → still
      `ValueError("Ambiguous...")`; inclusive boundaries per selector; a multi-selector variant of
      the Bloom round trip in `test_param_resolution.py`.
- [ ] 2.4 Implement the any-selector predicate in `choose_models`; update the module docstring.
      Verify `git diff` leaves the `len(matches) > 1` raise and override/skip logic untouched.
- [ ] 2.5 **Mutation check** (not committed): temporarily replace the predicate with (a) a
      card-level min/max age envelope and (b) any-species ∧ any-mode ∧ any-window; confirm 2.3's
      tests fail against each. Record both results in the commit body.
- [ ] 2.6 Minimal parity migration so the suite is green: `build_report_entry` emits `selectors`
      (tests first: two-selector card → two dicts in order, no top-level flat keys,
      `json.dumps` succeeds; one-selector card → one-element list); `build_label_card` and
      `_pick_best_candidate` read the sole selector (full selector rule lands in §4).
- [ ] 2.7 Fix flat reads in tests CI never runs: `grep -rnE "\.(species|mode|age_min|age_max)\b"
      tests/ scripts/ sleap_roots_predict/` — including the wandb-gated
      `test_model_registry.py:315` — until only `Selector`/`LabelCard`/params reads remain.
- [ ] 2.8 The gate is green. Commit with a `BREAKING CHANGE:` footer (report-entry shape).

## 3. Registry guard + batch-level catalog load (commit: `feat(registry): ...`)

- [ ] 3.1 Characterization test first (expected green on first run; if red, stop — #34 fact 1 is
      false): artifacts under `production` mixing flat-shaped and selector-shaped metadata → each
      flat one skipped with a warning naming it (assert the warning mentions `selectors`), the
      selector-shaped ones returned. Not `@pytest.mark.wandb` (fake artifacts), so CI runs it.
- [ ] 3.2 Tests first: every alias-matching artifact invalid → `list_cards()` raises naming the
      registry, alias and skipped count; zero alias-matching artifacts → unchanged behavior
      (empty list, no raise).
- [ ] 3.3 Tests first (`test_batch.py`): a source whose `list_cards()` raises → `run_batch` raises
      before any scan is predicted, no per-scan outputs written, CLI exits `1` with the one-line
      staging message; a counting fake source → `list_cards()` called exactly once per batch.
- [ ] 3.4 Implement: the guard in `WandbRegistrySource._collect_cards` (raise a `ValueError`
      subclass so the CLI's one-line staging path logs it); `run_batch` loads the catalog once
      after the run-manifest forward-copy and before the per-scan loop, outside its `try`.
- [ ] 3.5 Record on #34 that fact 1 now has one deliberate exception (all-invalid → raise), with
      the rationale (a premature deploy otherwise exits `3`, which the pipeline's exit gate
      passes).

## 4. Parity selector rule (commit: `feat(parity): ...`)

- [ ] 4.1 Tests first for `_resolve_selector(card, selector)`: explicit on card → returned;
      value-equal fresh `Selector` → accepted; not on card → `ValueError` naming the card; none +
      one → that selector; none + several → `None`.
- [ ] 4.2 Tests first for `build_label_card(..., selector=None)`: single-selector, no selector →
      that selector's fields; multi-selector, no selector → `ValueError`; not on card →
      `ValueError`; explicit → its fields; unnamed skeleton → fallback name uses the resolved
      selector's species.
- [ ] 4.3 Tests first for `_pick_best_candidate`, with fixtures that discriminate: a multi-selector
      card, no selector, where an envelope window would pick A but segment scoring picks B →
      assert B; a variant where segment scoring ties → `None`; explicit selector on a
      multi-selector card, both selectors checked (Day3 vs Day11, windows 2–5 and 10–13).
- [ ] 4.4 Tests first for `resolve_ground_truth`: a selector not on the card raises before any
      tier, even when `labels_registry_lookup` would resolve; a supplied selector reaches the
      basename tier (the Day3/Day11 fixture through `resolve_ground_truth(..., basename_index=...,
      selector=S)`).
- [ ] 4.5 Test first: `run_parity_harness` end to end with a two-selector card (`LocalCardSource`
      + `labels_registry_lookup`) → a full entry (no `gap_stage`), `selectors` of two objects in
      order, file round-trips through `json.loads`. Its broad `except` would otherwise hide a
      leftover flat read as an evaluation gap.
- [ ] 4.6 Implement 4.1–4.5; thread `selector` through `resolve_ground_truth` →
      `relink_ground_truth_by_basename_search` → `_pick_best_candidate`. Update the comments and
      docstrings that describe the flat fields: `parity.py:67-69`, `:346-348`, `:423-424`,
      `:480`, `:1006-1008`.
- [ ] 4.7 Change `scripts/run_parity_harness.py`'s default `--out` so it never overwrites the
      committed `2026-08-04-define-parity-tolerance-results.json` (require `--out`, or default to
      a new dated path); test the default.

## 5. Docs (commit: `docs: ...`)

- [ ] 5.1 Update `openspec/project.md:24,129` (pin, "13 production `ModelCard`s" → dated
      measurement), `API.md:209`, the `CLAUDE.md:93` parenthetical (point to the spec; add no
      content), README's parity-regenerate paragraph (`README.md:270-277`: pass a new `--out`; the
      2026-08-04 JSON is a pre-selectors snapshot), and `CHANGELOG.md` via `/update-changelog`
      (note: merging moves `:latest`/`:main` to an image that reads no production card until the
      re-seed). Then grep for the claim across the repo, excluding `.venv`,
      `openspec/changes/archive` and dated `docs/superpowers/` snapshots, comments included:
      `0\.1\.0a7`, `13 production`, `card's age`, `card's \[age_min`, `species/root-type`,
      `species ==`, `age_min <=`, `card-level`.
- [ ] 5.2 Re-grep for persisted state keyed on `registry_id` beyond design §4's list; record the
      result in the PR.

## 6. Merge gates — real models and the live canary

- [ ] 6.1 **A1 — selection-equivalence oracle.** Commit `scripts/a1_selection_oracle.py` (calls
      only `list_cards`/`LocalCardSource` + `choose_models`, never card fields, so the same file
      runs on `main` and the branch). Three environments exchange JSON: (i) the training worktree
      dumps its 8 planned selector cards (`model_dump_json`, no `--execute`), with placeholder
      identity (`collection_id` + `v0`) and `source_model_id`; (ii) a `main` worktree (contracts
      a7, needs `WANDB_API_KEY`/`SRP_WANDB_ENTITY`) dumps its table over the 13 live flat cards;
      (iii) this branch dumps its table over (i). Grid: every species in the selectors plus one
      unmodelled species × all three `Mode`s × ages min(age_min)−1 … max(age_max)+1 × root type.
      Join key: the new side's `source_model_id` ↔ the old side's selected `registry_id` mapped
      through training's committed 6.0(a) baseline
      (`docs/migration/2026-09-22-pre-reseed-baseline.json`, collection → `source_model_id`).
      Pass iff each cell's
      outcome class (selected model / skip / raise) and selected model agree, and the new side
      never raises. Commit the inputs and both tables under
      `docs/superpowers/specs/2026-09-2X-model-card-selectors-a1-*.json`.
- [ ] 6.2 **A2 — real-inference equivalence.** Commit a driver `scripts/a2_run_local.py` calling
      `run_batch(in, out, source=LocalCardSource([...]))` over the models-downloader snapshot
      (`c:\repos\models-downloader\tests\data\models_downloader_input\20250204_models`, record
      the card → directory map). Copy `scan_289`, `scan_577`, `scan_1009` from
      `Z:\users\eberrigan\pipeline_orchestration_tests\a4_poc\input` to scratch; never write into
      `a4_poc`. With `SRP_DEVICE=cpu` in both, and after confirming the two environments differ
      only in `sleap-roots-contracts` (`uv pip freeze` diff): run `main` twice (control), then the
      branch. Per scan, per root type, per frame: equal instance counts and
      `np.array_equal(points, equal_nan=True)` and equal scores — bitwise if the control was
      bitwise, else within the control's max deviation. Manifests may differ only in
      `registry_id`, `version`, `weights_checksum`, the `.slp` filename/path slug, the idempotency
      key and timestamps. Summarize in the PR; do not commit outputs.
- [ ] 6.3 Before the canary, confirm the wandb-gated tests fail as designed against today's flat
      registry (`uv run pytest -m wandb`): the guard's error, not an empty list. Record it.
- [ ] 6.4 **B — live canary (training 6.1)**, only on the user's explicit confirmation of
      `seed-registry --execute --only <collection>` (irreversible, single-operator). Then:
      `uv run pytest -m wandb` on this branch passes; a scripted check shows this branch resolves
      the new collection's `registry_id` without raising and `main` still resolves the old flat
      card; the expected skip warnings appear on both sides.
- [ ] 6.5 The gate, plus `uv build`, plus the `gpu` subset under a `windows_cuda` venv (it builds
      no cards, so A2 is the device-relevant check; record that it ran rather than skipped).
- [ ] 6.6 `/review-pr`.

## 7. Post-merge gates → tracked on #34

- [ ] 7.1 Post the post-merge gate checklist on #34 and open the PR with "Part of #34" (not
      "Closes"): **C1** re-run A1 with the new side read from the live selector registry after
      training 6.2; **C2** bump the predictor pin in `sleap-roots-pipeline`, run a small real Argo
      batch (selection resolves for every scan, exit `0`, predictions match pre-migration outputs,
      the one-time recompute occurs), re-baseline the A4 batch oracle; **C3** re-run
      `scripts/run_parity_harness.py` with a new `--out` against the re-seeded registry and compare
      with the 2026-08-04 results per physical model (old `registry_id` → `source_model_id` via
      the 6.0(a) baseline); then report C2's confirmed deployment to
      training so 6.3 becomes eligible. Tick this task when the checklist is posted.
