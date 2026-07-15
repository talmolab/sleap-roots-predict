## 1. Dependency bump (regression baseline)

- [x] 1.1 Bump `sleap-roots-contracts` to `==0.1.0a4` in `pyproject.toml`; run `uv lock` to
      relock `uv.lock` in the same step. Run `git diff uv.lock` and confirm the diff touches
      only the `sleap-roots-contracts` entry (version/hash/url) — abort and investigate if
      anything in the torch/sleap-nn stack moved (that stack is version-fragile per this
      repo's own pyproject.toml comments).
- [x] 1.2 Verify: `python -c "import sleap_roots_contracts; print(sleap_roots_contracts.__version__)"`
      prints `0.1.0a4`, and the full existing test suite (`pytest`) is still green with no
      code changes yet — this is the baseline the rest of the change diffs against.

## 2. Swap the implementation (import, delete, re-point tests)

- [x] 2.1 In `sleap_roots_predict/__init__.py`, replace
      `from sleap_roots_predict.param_resolution import resolve_params` with
      `from sleap_roots_contracts import resolve_params` (keep it in `__all__`).
- [x] 2.2 Delete `sleap_roots_predict/param_resolution.py`.
- [x] 2.3 Prune `tests/test_param_resolution.py` to only
      `test_round_trip_selects_expected_models` and
      `test_round_trip_unknown_species_selects_nothing`; update its imports to pull
      `resolve_params` from `sleap_roots_contracts` directly (not via predict's re-export)
      and drop now-unused imports (`param_resolution` module, the private helpers,
      `PLANT_AGE_DAYS_FIELD`/`SPECIES_NAME_FIELD`, `math`). Update the module docstring to
      describe its narrowed scope (the round-trip interop proof, not the oracle itself).
- [x] 2.4 Verify: `pytest` is green — no `ModuleNotFoundError`, the two kept tests pass
      against the contracts-produced `ResolvedParams`, `tests/test_public_api.py` still
      passes unchanged (re-export intact). Additionally run
      `pytest --collect-only tests/test_param_resolution.py` and confirm it collects exactly
      2 items, both passing — "green" alone wouldn't catch a botched prune that silently
      leaves 0 tests or reduces both to no-ops.

## 3. Doc sweep

- [x] 3.1 Grep the repo for `param_resolution`, `param-resolution`, and `resolve_params`
      (excluding `openspec/changes/archive/`, which is immutable history) to confirm the
      complete set of remaining references.
- [x] 3.2 Update `README.md`: remove `param_resolution.py` from the architecture tree; update
      the test-tree line for `test_param_resolution.py` (currently "Param-resolution oracle
      tests (offline)") to reflect its narrowed scope, e.g. "choose_models round-trip vs.
      contracts' resolve_params (offline)" — the file survives (pruned, not deleted), so its
      old description would otherwise go stale. (The "See the `param-resolution` and
      `model-management` OpenSpec specs" line lives in API.md, not README.md — handled in 3.3.)
- [x] 3.3 Update `API.md`: same spec-reference fix, plus fold the new failure modes into the
      existing semicolon-joined `resolve_params` Raises sentence (matching its style, not a
      bolted-on note) — e.g. "...or a malformed/sentinel input (non-string `species`,
      `pd.NA`/`pd.NaT`, non-finite/`Decimal`/bool-like `age`) — enforced by
      `sleap-roots-contracts`; see CHANGELOG for the behavior-change history."
- [x] 3.4 Update `openspec/project.md`: remove the `param_resolution.py` module-layout bullet,
      and bump its `sleap-roots-contracts (0.1.0a3)` version-string reference to `0.1.0a4`.
- [x] 3.5 Update `CLAUDE.md`: remove `param_resolution.py` from the Package Structure blurb.
- [x] 3.6 Re-run the grep from 3.1 to confirm zero remaining stale references outside
      `openspec/changes/archive/`.

## 4. CHANGELOG

- [x] 4.1 Edit the existing `[Unreleased]` "Param resolution" bullet in `CHANGELOG.md` in
      place: this edit should net *shrink* the bullet, not grow it. Cut the now-redundant
      internals prose (normalization/mode-seam detail — that's an implementation detail
      owned by contracts now, not predict's to document) down to: what predict re-exports,
      the `sleap-roots-contracts==0.1.0a4` version pin, and one compact behavior-change
      sentence (not a restated table) covering the stricter sentinel handling (`pd.NA`/
      `pd.NaT` species, `np.bool_`/`Decimal`/`inf` age, non-string species now raise
      `ValueError` instead of silently coercing).

## 5. Spec validation

- [x] 5.1 Run `openspec validate consume-contracts-params --strict` and fix every issue.

## 6. Pre-merge gate

- [x] 6.1 Run `/pre-merge` (format check, lint, full test suite, build) and fix any failures.
      Format/lint/codespell clean; CPU subset 238 passed (verified against the exact ci.yml
      marker expression — the pre-merge skill's documented `-m "not gpu"` command is stale,
      overrides `addopts`, and pulls in flaky `wandb`-marked tests; noted separately, not a
      regression from this change); GPU subset 3 passed locally (CUDA available); `uv build`
      succeeds. Local `docker build` skipped per user decision — no Dockerfile change and no
      new resolved dependencies (verified), CI's PR-triggered build-only validation covers it.

## 7. PR

- [x] 7.1 Open a PR referencing predict#28 and this change-id; note in the PR body that the
      `sleap-roots-pipeline` roadmap doc update is a pipeline-repo follow-up (not done here),
      and that bloom#411 is already unblocked by the `0.1.0a4` release. (PR #29.)

## 8. Post-merge (not part of this PR's tasks; tracked separately)

- [ ] 8.1 After merge: archive this OpenSpec change (`openspec archive consume-contracts-params`),
      applying the REMOVED/MODIFIED spec deltas to `openspec/specs/`.
- [ ] 8.2 Comment on predict#20 to re-point its reference oracle at
      `sleap_roots_contracts.resolve_params`.
