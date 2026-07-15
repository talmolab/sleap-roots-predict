## 1. Dependency bump (regression baseline)

- [ ] 1.1 Bump `sleap-roots-contracts` to `==0.1.0a4` in `pyproject.toml`; run `uv lock` to
      relock `uv.lock` in the same step.
- [ ] 1.2 Verify: `python -c "import sleap_roots_contracts; print(sleap_roots_contracts.__version__)"`
      prints `0.1.0a4`, and the full existing test suite (`pytest`) is still green with no
      code changes yet — this is the baseline the rest of the change diffs against.

## 2. Swap the implementation (import, delete, re-point tests)

- [ ] 2.1 In `sleap_roots_predict/__init__.py`, replace
      `from sleap_roots_predict.param_resolution import resolve_params` with
      `from sleap_roots_contracts import resolve_params` (keep it in `__all__`).
- [ ] 2.2 Delete `sleap_roots_predict/param_resolution.py`.
- [ ] 2.3 Prune `tests/test_param_resolution.py` to only
      `test_round_trip_selects_expected_models` and
      `test_round_trip_unknown_species_selects_nothing`; update its imports to pull
      `resolve_params` from `sleap_roots_contracts` directly (not via predict's re-export)
      and drop now-unused imports (`param_resolution` module, the private helpers,
      `PLANT_AGE_DAYS_FIELD`/`SPECIES_NAME_FIELD`, `math`). Update the module docstring to
      describe its narrowed scope (the round-trip interop proof, not the oracle itself).
- [ ] 2.4 Verify: `pytest` is green — no `ModuleNotFoundError`, the two kept tests pass
      against the contracts-produced `ResolvedParams`, `tests/test_public_api.py` still
      passes unchanged (re-export intact).

## 3. Doc sweep

- [ ] 3.1 Grep the repo for `param_resolution`, `param-resolution`, and `resolve_params`
      (excluding `openspec/changes/archive/`, which is immutable history) to confirm the
      complete set of remaining references.
- [ ] 3.2 Update `README.md`: remove `param_resolution.py` from the architecture tree; update
      the "See the `param-resolution` and `model-management` OpenSpec specs" line to
      reference `sleap-roots-contracts` + `model-management` instead.
- [ ] 3.3 Update `API.md`: same spec-reference fix, plus a short note that malformed/sentinel
      inputs (non-string species, non-finite/`Decimal`/bool-like age) raise `ValueError`,
      sourced from `sleap-roots-contracts`.
- [ ] 3.4 Update `openspec/project.md`: remove the `param_resolution.py` module-layout bullet.
- [ ] 3.5 Update `CLAUDE.md`: remove `param_resolution.py` from the Package Structure blurb.
- [ ] 3.6 Re-run the grep from 3.1 to confirm zero remaining stale references outside
      `openspec/changes/archive/`.

## 4. CHANGELOG

- [ ] 4.1 Edit the existing `[Unreleased]` "Param resolution" bullet in `CHANGELOG.md` in
      place: attribute the implementation to `sleap-roots-contracts>=0.1.0a4`, note the
      local copy is deleted, and call out the stricter sentinel handling (`pd.NA`/`pd.NaT`
      species, `np.bool_`/`Decimal`/`inf` age, non-string species now raise `ValueError`
      instead of silently coercing) as a behavior change.

## 5. Spec validation

- [ ] 5.1 Run `openspec validate consume-contracts-params --strict` and fix every issue.

## 6. Pre-merge gate

- [ ] 6.1 Run `/pre-merge` (format check, lint, full test suite, build) and fix any failures.

## 7. PR

- [ ] 7.1 Open a PR referencing predict#28 and this change-id; note in the PR body that the
      `sleap-roots-pipeline` roadmap doc update is a pipeline-repo follow-up (not done here),
      and that bloom#411 is already unblocked by the `0.1.0a4` release.

## 8. Post-merge (not part of this PR's tasks; tracked separately)

- [ ] 8.1 After merge: archive this OpenSpec change (`openspec archive consume-contracts-params`),
      applying the REMOVED/MODIFIED spec deltas to `openspec/specs/`.
- [ ] 8.2 Comment on predict#20 to re-point its reference oracle at
      `sleap_roots_contracts.resolve_params`.
