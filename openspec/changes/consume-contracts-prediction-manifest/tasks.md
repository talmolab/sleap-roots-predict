## 1. Dependency bump (isolated regression baseline)

- [ ] 1.1 Bump `sleap-roots-contracts` from `==0.1.0a4` to `==0.1.0a5` in `pyproject.toml`.
- [ ] 1.2 Relock `uv.lock`; confirm the diff is scoped to just the `sleap-roots-contracts`
      entry (per predict#29's verified precedent).
- [ ] 1.3 Verify `python -c "import sleap_roots_contracts as c; print(c.PredictionArtifact,
      c.PredictionManifest)"` succeeds at the new pin, with no other code changes yet.
- [ ] 1.4 Run the full test suite (`pytest -m "not gpu and not acceptance and not wandb"`) and
      confirm it is still green before any further edits — this is the regression baseline.

## 2. Import swap + test prune (interdependent — land together)

TDD framing: this is a subtractive refactor (the logic under test already lives, and is
tested, in contracts), so the "test first" step here is confirming the *replacement* tests
(contracts' own suite) already cover what predict's two pure-model tests assert, then pruning
predict's copies and swapping the import in the same pass — pruning first would leave the old
tests failing on stale imports; swapping first would leave duplicate-but-still-green coverage
that then needs a second pass. Land both together, verified by one gate.

- [ ] 2.1 In `sleap_roots_predict/output_contract.py`: replace the local `PredictionArtifact`
      and `PredictionManifest` class bodies with `from sleap_roots_contracts import
      PredictionArtifact, PredictionManifest` (alongside the existing `ModelRef`,
      `ResolvedParams`, `RootType` import from the same package).
- [ ] 2.2 Delete the now-dead `_FROZEN = ConfigDict(frozen=True, protected_namespaces=())` and
      its explanatory comment (it existed solely for the two removed class bodies).
- [ ] 2.3 Update the module docstring's "The schema is predict-local for now... it is
      promoted to the shared contract once the traits stage (A3-traits) consumes it" note —
      it's promoted now; rewrite to state the shape is imported from `sleap-roots-contracts`.
- [ ] 2.4 In `tests/test_output_contract.py`: delete `test_manifest_round_trips` and
      `test_plant_qr_code_defaults_to_scan_key`; remove the now-unused `PredictionArtifact`
      import (confirmed its only other use was inside the deleted `test_manifest_round_trips`).
- [ ] 2.5 Verify: `pytest --collect-only tests/test_output_contract.py` collects exactly 26
      items (28 today minus the 2 pruned). Run the full suite; confirm green, and confirm
      `sleap_roots_predict/__init__.py`'s re-export needs no edit (`from sleap_roots_predict
      import PredictionArtifact, PredictionManifest` still works — covered by
      `tests/test_public_api.py`, unmodified).
- [ ] 2.6 Empirically confirm the `kind` field: run a real writer test (e.g.
      `test_writer_writes_named_slp_and_manifest`) and inspect the produced
      `{scan_key}.predictions.json` on disk for `"kind": "predictions_slp"` per artifact.

## 3. OpenSpec spec delta

- [ ] 3.1 Write the MODIFIED delta in
      `openspec/changes/consume-contracts-prediction-manifest/specs/prediction-output/spec.md`
      for `### Requirement: Combined per-scan manifest and provenance sidecar` — full existing
      text plus: `kind` added to `PredictionArtifact`'s field list, a sentence noting the shape
      is now defined by `sleap-roots-contracts`' `prediction-manifest-contract` capability and
      imported (not locally defined), and a new scenario "Artifact kind defaults to
      predictions_slp".

## 4. Docs and changelog

- [ ] 4.1 Edit `CHANGELOG.md`'s `[Unreleased]` "Predict output contract" bullet in place to
      attribute `PredictionArtifact`/`PredictionManifest` to `sleap-roots-contracts==0.1.0a5`
      and note the new `kind` field.
- [ ] 4.2 Update `API.md`'s "kept single-sourced there to avoid drift" line (now stale since
      the schema is imported, not defined, in `output_contract.py`'s docstrings).
- [ ] 4.3 Update `openspec/project.md` in both spots: the Architecture Patterns
      `output_contract.py` bullet's "+ the manifest/artifact models" phrasing (match the
      `model_selection.py` bullet's existing "consumes... predict carries no local copy"
      phrasing for `resolve_params`), and the External Dependencies section's
      `sleap-roots-contracts` version literal (`==0.1.0a4` → `==0.1.0a5`).
- [ ] 4.4 Update `CLAUDE.md`'s Package Structure blurb with the same treatment as the
      Architecture Patterns bullet.
- [ ] 4.5 Grep sweep for stray references to predict's local class definitions across code and
      docs; confirm zero remaining (aside from immutable `openspec/changes/archive/` history).

## 5. Validation gate

- [ ] 5.1 `openspec validate consume-contracts-prediction-manifest --strict` — resolve any
      issues.
- [ ] 5.2 Full `/pre-merge` gate (format, lint, test, build) before opening the PR.
