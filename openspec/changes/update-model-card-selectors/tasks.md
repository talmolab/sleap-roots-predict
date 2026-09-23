# Tasks: update-model-card-selectors

Every code task is red → green → refactor: write the named test(s) first, run them, see them fail
for the stated reason, then write the minimum code to pass. Record each observed red in the commit
body. Tasks marked *characterization* are expected green on first run and say what a red means.
Commit red states locally if useful, but push only green heads.

**Git hygiene.** Stage by explicit path in every commit; never `git add -A` on this branch.
`HANDOFF-predict34.md` is excluded via `.git/info/exclude`. Worktrees for A1/A2 go under the
scratchpad or `.claude/worktrees/` (the repo's `.worktrees/` is not ignored) and are removed with
`git worktree remove` before merge. No commit body may contain a closing keyword for #34 or
sleap-roots-training#39 (the repo squashes with `COMMIT_MESSAGES`, so every body lands on `main`);
use "(#34)" or "Part of #34".

**The gate** (CI's exact commands; `ruff check .` is not used — it reports 162 pre-existing errors
in `tests/`, outside CI's lint scope):

```
uv run pytest -m "not gpu and not acceptance and not wandb and not parity" tests/
uv run black --check sleap_roots_predict tests scripts
uv run ruff check sleap_roots_predict/ scripts/
uv run codespell
uv lock --check
openspec validate update-model-card-selectors --strict
```

`/pre-merge`'s `-m "not gpu"` is knowingly replaced by the pytest line above: it overrides
`addopts` and selects the `wandb` tests, which fail by design against the flat registry (§6.3).

## 1. Prep refactor — green on contracts a7 (commit: `refactor(tests): share card builders`)

- [x] 1.1 Positive control first: record that
      `rg -nU "ModelCard(\.model_validate)?\([^)]*\b(species|mode|age_min|age_max)\s*=" tests/ | rg -c ModelCard`
      prints `8` today.
- [x] 1.2 Add `tests/card_builders.py`: `make_card(root_type, registry_id=None, *, selectors=None,
      species=, mode=, age_min=, age_max=, version=, weights_checksum=, sleap_nn_version=)` (either
      `selectors` or the flat kwargs; `registry_id` defaults from the first selector) and
      `raw_card_meta(..., drop=())` for the wandb fixtures. On a7 both assert exactly one selector
      and emit the flat shape. Keep each module's thin `_card` wrapper so call sites and per-module
      defaults are untouched; route the inline `ModelCard(` in `test_batch.py:827` and the
      `_good_meta`/`_malformed_artifact` fixtures (`test_model_registry.py:164-197`) through them.
- [x] 1.3 Verify: the gate is green on a7; `git diff -U0 tests/ | rg "^[-+]\s*assert"` prints
      nothing; 1.1's command prints `1` (the builder); `rg -n "[\"'](age_min|age_max)[\"']\s*:" tests/`
      hits only `raw_card_meta`.

## 2. The breaking commit — pin a9 + selection (commit: `feat(selection)!: ...`)

Atomic by necessity: after the bump, `model_selection.py:86-88` and `parity.py:904,1071-1075`
fail on flat reads, so neither lands separately green. Put a `BREAKING CHANGE:` footer (parity
report-entry shape) in the body.

- [x] 2.1 Contract-assumption tests **before** the bump (red on a7, where a flat dict validates):
      `ModelCard.model_validate(<flat dict>)` raises; `selectors=[]` raises.
- [x] 2.2 Edit `pyproject.toml:24` to `sleap-roots-contracts==0.1.0a9`, then
      `uv lock --upgrade-package sleap-roots-contracts`. Verify `git diff uv.lock` changes only
      that package's block and the project's `requires-dist` specifier. Switch the builders to emit
      `Selector`s / JSON `"selectors": [{...}]` (`drop=("species",)` drops it from selector 0; the
      pydantic location `selectors.0.species` still satisfies `test_model_registry.py:197`). 2.1 is
      now green; record the suite's red count.
- [x] 2.3 Interim: make `choose_models` and the two parity reads use `card.selectors[0]`; get the
      gate green. Then write the selection tests (`test_model_selection.py`) and watch the
      any-selector, per-selector-age and disjoint-window ones go red for the stated reason:
      any-selector (pennycress/14 on canola 2–13 + pennycress 2–14); per-selector age (canola/14
      on that card → no match); disjoint windows (canola 2–5 + canola 10–13, age 7 → no match);
      no cross product (canola 2–13 + arabidopsis multiplant 2–14; canola/multiplant at ages 2, 5,
      13 → no match); overlapping selectors on one card → selected, no raise; two cards each
      matching through a different selector → `ValueError("Ambiguous...")`; inclusive boundaries
      per selector; a multi-selector variant of the Bloom round trip in `test_param_resolution.py`.
- [x] 2.4 Implement the any-selector predicate; update the module docstring. Verify `git diff`
      leaves the `len(matches) > 1` raise and the override/skip logic untouched.
- [x] 2.5 **Mutation check** (not committed): temporarily substitute (a) a card-level min/max
      envelope, (b) any-species ∧ any-mode ∧ any-window, (c) `selectors[0]` only; confirm 2.3's
      tests fail against each. Record the results in the commit body.
- [x] 2.6 `build_report_entry` emits `selectors` (tests first: two-selector card → two dicts in
      order, no top-level flat keys, `json.dumps` succeeds; one-selector card → one-element list).
- [x] 2.7 Flat reads nothing in CI exercises:
      `rg -n --type py "\.(species|mode|age_min|age_max)\b|\[[\"'](species|mode|age_min|age_max)[\"']\]" tests/ scripts/ sleap_roots_predict/`
      until only the allowed residue remains — params reads (`model_selection.py:56-57`), `LabelCard`
      reads (`test_parity.py:635`), `snapshot.mode` (`run_manifest.py:218`) — including the
      wandb-gated `test_model_registry.py:315`.
- [x] 2.8 The gate is green.

## 3. Registry guard + batch-level catalog load (commit: `feat(registry): ...`)

- [x] 3.1 *Characterization* (red means #34 fact 1 is false — stop and revisit the plan): fake
      artifacts under `production` mixing flat- and selector-shaped metadata → each flat one skipped
      with a warning naming it and mentioning `selectors`; the selector-shaped ones returned. No
      `@pytest.mark.wandb`, so CI runs it.
- [x] 3.2 Tests first: **invert** `test_collect_cards_all_malformed_returns_empty`
      (`test_model_registry.py:200`, which pins #32's "empty, not an exception") into
      `pytest.raises(<GuardError>)` matching the registry, alias and count `1` — an intentional
      assertion change, named in the commit body; keep `test_collect_cards_alias_filtered_is_silent`
      as the zero-alias case (empty list, no raise).
- [x] 3.3 Tests first for `WarmModelWorker.load_catalog()`: `load_catalog(); load_catalog();
      resolve(); resolve()` → one `list_cards` call; with no `WANDB_API_KEY`, `load_catalog()`
      raises naming it.
- [x] 3.4 Tests first (`test_batch.py`):
      - (a) the **real** guard: `WandbRegistrySource` with `WANDB_API_KEY=dummy` and
        `wandb.Api` monkeypatched to `FakeApi` holding only flat artifacts, passed into `run_batch`
        and (via the `kwargs.setdefault("source", …)` spy) into `main()` → raises a `ValueError`
        before any scan, "Batch aborted" in `caplog`, `out` holds at most `run_manifest.json`, exit
        `1`;
      - (b) a counting source whose `should_stop` stays false: exactly one `list_cards` call for a
        two-scan batch, and it precedes the first `resolve` (spy);
      - (c) default source with `clean_wandb_env` → `run_batch` raises `RuntimeError` before any
        scan (was: every scan failed, exit `3`);
      - (d) stop requested before the loop → `list_cards` not called, exit `143`;
      - (e) a manifest listing only missing sidecars → `list_cards` not called, exit `3`;
      - (f) the forward-copy-failure test (`calls["n"] == 0`) still passes unchanged.
- [x] 3.5 Implement: the guard in `WandbRegistrySource._collect_cards` (a `ValueError` subclass
      naming the full registry path, the alias, the failed count, and a hint "this consumer
      requires selector-shaped cards; has the registry been re-seeded?"); `WarmModelWorker.load_catalog()`;
      in `run_batch`'s loop, after the `should_stop()` check and the `scan.error` skip and before the
      per-scan `try`, call it once (guarded by a loaded flag); add no extra `should_stop()` call —
      `test_should_stop_stops_after_first_scan` and the SIGTERM compose test count them. Update the `run_batch` `Raises:`
      docstring (`batch.py:322-330`), `__main__.py`'s module docstring and staging-error comment
      (`:1-12`, `:86-90`), and the `_collect_cards` docstring (`model_registry.py:196-214`).
- [ ] 3.6 Draft (for the user to approve before posting) a #34 comment: fact 1 now has one
      deliberate exception (all-invalid → raise, reversing #32's test), why (a premature deploy
      otherwise exits `3`, which the exit gate passes), and its limit (it cannot see an
      incomplete-but-readable catalog, e.g. mid-re-seed).

## 4. Parity selector rule (commit: `feat(parity): ...`)

- [ ] 4.1 Tests first for `_resolve_selector(card, selector)`: explicit on card → returned;
      value-equal fresh `Selector` → accepted; not on card → `ValueError` naming the card; none +
      one → that selector; none + several → `None`.
- [ ] 4.2 Tests first for `build_label_card(..., selector=None)`: single-selector, no selector →
      that selector's fields; multi-selector, no selector → `ValueError`; not on card →
      `ValueError`; value-equal selector → accepted; explicit → its fields; unnamed skeleton →
      fallback name uses the resolved selector's species.
- [ ] 4.3 Tests first for `_pick_best_candidate`, with discriminating fixtures: multi-selector
      card, no selector, where an envelope window would pick A but segment scoring picks B → B;
      a variant where segment scoring ties → `None`; explicit selector on a multi-selector card
      (Day3 vs Day11, windows 2–5 and 10–13, parent matching leaving both) → each selector picks
      its day. Name the existing single-selector age-hint tests (e.g. `test_parity.py:295`) as the
      sole-selector scenario's coverage.
- [ ] 4.4 Tests first for `resolve_ground_truth`: a selector not on the card raises before any
      tier — the lookup spy is never called and `workdir` stays empty — even when the lookup would
      resolve; a supplied selector reaches the basename tier (the Day3/Day11 fixture via
      `basename_index=` and `selector=`).
- [ ] 4.5 Test first: `run_parity_harness` end to end with a two-selector card (`LocalCardSource`
      + `labels_registry_lookup`) → a full entry (no `gap_stage`), `selectors` of two objects in
      order, file round-trips through `json.loads`. Its broad `except` (`parity.py:1306`) would
      otherwise hide a leftover flat read as an evaluation gap.
- [ ] 4.6 Implement 4.1–4.5; thread `selector` through `resolve_ground_truth` →
      `relink_ground_truth_by_basename_search` → `_pick_best_candidate` (only direct callers can
      pass one; `evaluate_model_card`/`run_parity_harness` pass none). Update the comments and
      docstrings describing the flat fields: `parity.py:67-69`, `:346-348`, `:423-424`, `:480`,
      `:1006-1008`.
- [ ] 4.7 `scripts/run_parity_harness.py`: make `--out` required, so a run can never overwrite the
      committed `2026-08-04-define-parity-tolerance-results.json`. Test via
      `importlib.util.spec_from_file_location` + `main([])` → `SystemExit(2)`.

## 5. Docs (commit: `docs: ...`)

- [ ] 5.1 Update `openspec/project.md:23-24,129` (pin; "13 production `ModelCard`s" → the dated
      2026-08-04 measurement), `API.md:209` and the `run_batch` paragraph (`:214-227`), the
      `CLAUDE.md:93` parenthetical (point to the spec; add no content), README's parity-regenerate
      paragraph (`README.md:270-277`: `--out` is required; the 2026-08-04 JSON is a pre-selectors
      snapshot), and `CHANGELOG.md` via `/update-changelog`: the pin; any-selector matching; the
      report-entry shape; the guard; **exit `3` → `1`** for missing credentials, registry/network
      errors and an unreadable catalog; and that after merge `:latest`/`:main` read only the
      canary's card until the full re-seed.
- [ ] 5.2 Repo-wide claim grep with `git grep` (skips ignored `htmlcov/`, `dist/`, `artifacts/`),
      excluding `openspec/changes/archive` and dated `docs/superpowers/` snapshots, comments
      included: `0\.1\.0a7`, `13 production`, `card's age`, `card's \[age_min`, `species/root-type`,
      `species ==`, `age_min <=`, `card-level`, `lazily once`.
- [ ] 5.3 Re-grep for persisted state keyed on `registry_id` beyond design §4's list; record the
      result in the PR.

## 6. PR and merge gates

- [ ] 6.0 Push and open a **draft** PR titled `feat(selection)!: migrate to ModelCard.selectors
      (contracts 0.1.0a9)`, body "Part of #34" (replace `/pr-description`'s literal
      `Closes #…` line; do not link #34 in the Development sidebar). Include a ready-made squash
      message ending with the `BREAKING CHANGE:` footer. Check each commit is green:
      `GIT_SEQUENCE_EDITOR=: git rebase -x "<the gate>" main`.
- [ ] 6.1 **A1 — selection-equivalence oracle.** Commit `scripts/a1_selection_oracle.py` (calls
      only `list_cards`/`LocalCardSource` + `choose_models`, never card fields; copied unmodified
      into the `main` worktree, sha256 recorded on both sides) and `scripts/a1_compare.py` (exits
      non-zero on any mismatch). Three environments exchange JSON:
      - (i) the training worktree runs the pure library path
        `chooser.load_selection_matrix()` → `cards.expand_rows_to_cards(...)` → per card
        `card_to_metadata(card)` + placeholder identity (`collection_id(card)`, `v0`), plus a
        `collection_id → source_model_id` map (carried beside the card; `ModelCard` would drop
        it). Not `seed-registry`, which extracts zips;
      - (ii) a `main` worktree (contracts a7; needs `WANDB_API_KEY`, `SRP_WANDB_ENTITY`) tables the
        13 live flat cards;
      - (iii) this branch tables (i).
      Grid: the union of both sides' species plus one unmodelled species × all three `Mode`s × ages
      min−1 … max+1 over the union of both sides' windows × root type. Join: the old `registry_id`'s
      basename (the collection) → `source_model_id` via training's 6.0(a) baseline
      (`docs/migration/2026-09-22-pre-reseed-baseline.json` on training branch
      `migrate-model-card-selectors` @ `b43464a`, not yet on training `main` — copy it into the
      committed inputs with its sha). Precondition (recorded): the old side has zero `raise` cells.
      Pass iff every cell's outcome class (selected model / skip / raise) and selected model agree,
      the new side never raises, and both sides select in the same, non-zero number of cells.
      Commit inputs and tables as `docs/superpowers/specs/<date>-model-card-selectors-a1-*.json`.
- [ ] 6.2 **A2 — real-inference equivalence.** Commit `scripts/a2_run_local.py`, calling
      `run_batch(in, out, source=LocalCardSource([...]))` with cards loaded from A1's dumps ((ii) on
      `main`, (i) on the branch) and directories mapped by `source_model_id` into the
      models-downloader snapshot (`c:\repos\models-downloader\tests\data\models_downloader_input\20250204_models`,
      laid out by `source_model_id`). Copy `scan_289`, `scan_577`, `scan_1009` from
      `Z:\users\eberrigan\pipeline_orchestration_tests\a4_poc\input` to scratch; never write into
      `a4_poc`. Confirm the two environments differ only in `sleap-roots-contracts` (`uv pip freeze`
      diff, excluding the editable `sleap-roots-predict` line). With `SRP_DEVICE=cpu`, run `main`
      twice (control), then the branch, **each into a fresh empty output directory**. Require every
      run's `BatchResult` all `ok` (no `skipped`/`failed`), and each scan the same non-empty set of
      root types with an equal, non-zero frame count on every run. Per scan, root type and frame:
      equal instance counts, `np.array_equal(points, equal_nan=True)` and equal scores — bitwise if
      the control was bitwise, else within the control's max deviation; unequal instance counts in
      the control make A2 inconclusive, not passed. Manifests may differ only in `registry_id`,
      `version`, `weights_checksum`, the `.slp` filename/path slug, the idempotency key and
      timestamps. Record the commit sha, snapshot path and device in the PR; do not commit outputs.
- [ ] 6.3 Before the canary: `uv run pytest -m wandb -rA` with `WANDB_API_KEY` set → 0 skipped;
      both wandb tests fail, each failure naming the guard's class, registry and alias; the CLI
      test shows "Batch aborted" and exit `1`. (Without the guard the failure would be
      `assert cards` — this check tells them apart.) Record it.
- [ ] 6.4 **B — live canary (training 6.1)**, only on the user's explicit confirmation of
      `seed-registry --execute --only <collection>` (irreversible, single-operator). Propose a rice
      cylinder 2–5 collection (crown or primary), which makes `test_module_cli_over_registry`
      (rice/cylinder/3) meaningful; with any other canary, pre-declare that test's expected exit
      `3`. Then, with a committed check script that exits non-zero on failure: this branch resolves
      the new collection's `registry_id` without raising; `main` still resolves the old flat card;
      skip warnings number exactly 13 on the branch and 1 on `main`. Do not run it while training's
      6.0(b)/(e) alias-drop rehearsal is live on the canary collection.
- [ ] 6.5 The gate, plus `uv build`, plus `uv run pytest -m gpu -rs` under a `windows_cuda` venv
      reporting 0 skipped (it builds no cards, so A2 is the device-relevant check).
- [ ] 6.6 `/review-pr`.
- [ ] 6.7 Draft (for the user to approve before posting) a correction to the pipeline design of
      record (`2026-09-21-per-run-run-manifest-identity-design.md` §2.7 and §4 step 0a say merging
      #34 is ungated and pins a8; this change pins a9 and holds the merge for the canary).

## 7. Post-merge gates → tracked on #34

- [ ] 7.1 Post (after user approval) the post-merge checklist on #34: **C1** re-run A1 with the new
      side read from the live selector registry after training 6.2; **C2** bump the predictor pin
      in `sleap-roots-pipeline` (tag + digest + `SRP_PREDICT_CONTAINER_DIGEST`,
      `sleap-roots-predictor-template.yaml:80,107`), run a small real Argo batch (selection
      resolves for every scan, exit `0`, predictions match pre-migration outputs, the one-time
      recompute occurs), re-baseline the A4 batch oracle, and report C2's confirmed deployment to
      training so **training** 6.3 becomes eligible; **C3** re-run `scripts/run_parity_harness.py`
      with a new `--out` against the re-seeded registry and compare with the 2026-08-04 results per
      physical model (old `registry_id` → `source_model_id` via the 6.0(a) baseline). C3 does not
      gate training 6.3. Tick this task when the checklist is posted.
