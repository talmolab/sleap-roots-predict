## 1. Dependency bump (isolated regression baseline)

Land 1.1 and 1.2 as a single commit — the Dockerfile's `uv sync --frozen` hard-fails if
`uv.lock` doesn't match `pyproject.toml`, so a bumped pin without its relock is never a safe
standalone commit.

- [ ] 1.1 Bump `sleap-roots-contracts` from `==0.1.0a4` to `==0.1.0a5` in `pyproject.toml`.
- [ ] 1.2 Relock scoped to just this dependency: `uv lock -P sleap-roots-contracts` (not a
      bare `uv lock`) so the resolver can't silently pull in unrelated bumps; confirm the
      `uv.lock` diff touches only the `sleap-roots-contracts` entry.
- [ ] 1.3 Verify `python -c "import sleap_roots_contracts as c; print(c.PredictionArtifact,
      c.PredictionManifest, c.ModelRef, c.ResolvedParams, c.RootType)"` succeeds at the new
      pin, with no other code changes yet (also touches the three symbols
      `output_contract.py` already imports from contracts, not just the two new ones, in case
      `0.1.0a5` changed anything about them).
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
      Add a cheap identity-guard test that catches the one regression this whole change exists
      to prevent — a future accidental reintroduction of a local shadow class:
      ```python
      def test_prediction_classes_are_reexported_from_contracts():
          import sleap_roots_contracts as c
          assert PredictionArtifact is c.PredictionArtifact
          assert PredictionManifest is c.PredictionManifest
      ```
- [ ] 2.5 Verify: `grep -c '^def test_' tests/test_output_contract.py` reports 27 (28 today,
      minus the 2 pruned, plus the 1 identity-guard test added in 2.4). Note
      `pytest --collect-only` will report a higher *item* count than this because two
      unrelated tests in the file are `@pytest.mark.parametrize`d
      (`test_rejects_reserved_and_control_chars_in_scan_key`,
      `test_rejects_whitespace_scan_key`) — collected-item count is not the same thing as
      test-function count and is not the right verification here. Run the full suite; confirm
      green, and confirm `sleap_roots_predict/__init__.py`'s re-export needs no edit (`from
      sleap_roots_predict import PredictionArtifact, PredictionManifest` still works — covered
      by `tests/test_public_api.py`, unmodified).
- [ ] 2.6 Extend two existing writer tests (not an ad hoc one-off check, so the assertion
      stays in CI going forward) to cover the new `kind` field, matching the spec delta's
      "Artifact kind defaults to predictions_slp" scenario, which requires the value to hold
      in both the in-memory manifest and the written JSON:
      - `test_writer_writes_named_slp_and_manifest`: add
        `assert all(a.kind == "predictions_slp" for a in manifest.artifacts)`.
      - `test_manifest_json_on_disk_round_trips`: add an assertion that the on-disk JSON's
        `artifacts[*].kind` equals `"predictions_slp"` (parse the JSON directly, don't only
        rely on `PredictionManifest.model_validate_json` round-tripping, since a wrong-but-
        internally-consistent default wouldn't be caught by equality alone).

## 3. OpenSpec spec delta

- [ ] 3.1 Write the MODIFIED delta in
      `openspec/changes/consume-contracts-prediction-manifest/specs/prediction-output/spec.md`
      for `### Requirement: Combined per-scan manifest and provenance sidecar` — full existing
      text plus: `kind` added to `PredictionArtifact`'s field list, a sentence noting the shape
      is now defined by `sleap-roots-contracts`' `prediction-manifest-contract` capability and
      imported (not locally defined), and a new scenario "Artifact kind defaults to
      predictions_slp".

## 4. Docs and changelog

- [ ] 4.1 Edit `CHANGELOG.md`'s `[Unreleased]` "Predict output contract" bullet in place.
      Unlike the adjacent "Param resolution" bullet, this one never named
      `sleap-roots-contracts==0.1.0a4` at all — this is an added attribution, not just a
      version bump. Insert a sentence styled after that adjacent bullet's precedent, e.g.:
      "`PredictionArtifact`/`PredictionManifest` are implemented in
      `sleap-roots-contracts==0.1.0a5` (predict's local copies are deleted — contracts is now
      the single source of truth, mirroring `resolve_params`); `PredictionArtifact` gains a
      `kind` field (`BlobKind`, defaults to `"predictions_slp"`)."
- [ ] 4.2 Update `API.md`'s "kept single-sourced there to avoid drift" line (now stale since
      the schema is imported, not defined, in `output_contract.py`'s docstrings). Also update
      the inline field-list comment on the `PredictionArtifact` import in the same section's
      code block (currently `# one per-root record (model_id, ModelRef, slp_path, checksum,
      size)`) to include `kind` — this is a second, independent copy of the field list that
      the "single-sourced" line doesn't cover and would otherwise silently drift.
- [ ] 4.3 Update `openspec/project.md` in both spots: the Architecture Patterns
      `output_contract.py` bullet's "+ the manifest/artifact models" phrasing (match the
      `model_selection.py` bullet's existing "consumes... predict carries no local copy"
      phrasing for `resolve_params`), and the External Dependencies section's
      `sleap-roots-contracts` bullet — bump the version literal (`==0.1.0a4` → `==0.1.0a5`)
      and append `PredictionArtifact`/`PredictionManifest` to that bullet's enumerated type
      list (`ModelCard`/`ModelRef`/`ResolvedParams`/`RootType`), since they're now equally
      contracts-shared types.
- [ ] 4.4 Update `CLAUDE.md`'s Package Structure blurb with the same treatment as the
      Architecture Patterns bullet.
- [ ] 4.5 Grep sweep for stray references to predict's local class definitions across code and
      docs; confirm zero remaining (aside from immutable `openspec/changes/archive/` history).

## 5. Validation gate

- [ ] 5.1 `openspec validate consume-contracts-prediction-manifest --strict` — resolve any
      issues.
- [ ] 5.2 Full `/pre-merge` gate (format, lint, test, build) before opening the PR.
