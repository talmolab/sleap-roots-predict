## Why

`PredictionArtifact`/`PredictionManifest` were promoted into `sleap-roots-contracts`
(`0.1.0a5` on PyPI) so bloom#407 can read predict's manifest without a git dependency on this
unpublished repo. predict still defines both classes locally, a duplication risk now that the
two definitions already differ by one field (`kind`) with no shared source of truth — the same
problem `resolve_params` had before predict#28/PR#29 consumed it from contracts.

## What Changes

- Bump `sleap-roots-contracts` pin from `==0.1.0a4` to `==0.1.0a5` in `pyproject.toml`; relock
  `uv.lock` (diff scoped to the `sleap-roots-contracts` entry).
- Replace `sleap_roots_predict/output_contract.py`'s locally-defined `PredictionArtifact`/
  `PredictionManifest` classes with `from sleap_roots_contracts import PredictionArtifact,
  PredictionManifest`; delete the now-dead `_FROZEN` config (only the two removed class bodies
  used it) and update the module docstring's stale "schema is predict-local for now" note.
  `sleap_roots_predict/__init__.py` needs no change — it already imports both names from
  `sleap_roots_predict.output_contract`, which continues to bind them.
- Prune `tests/test_output_contract.py`'s two pure-model tests (`test_manifest_round_trips`,
  `test_plant_qr_code_defaults_to_scan_key`) — contracts' own `tests/test_prediction_manifest.py`
  now owns this coverage; predict's remaining 26 tests exercise the real writer and are
  untouched. Remove the now-unused `PredictionArtifact` import.
- Update `openspec/specs/prediction-output/spec.md`'s "Combined per-scan manifest and
  provenance sidecar" requirement (MODIFIED) to document the new `kind` field and that the
  shape is now imported from contracts rather than defined locally.
- Edit the CHANGELOG's `[Unreleased]` "Predict output contract" bullet in place (predict has
  never cut a release) to attribute the shape to `sleap-roots-contracts==0.1.0a5`.
- Update stale doc references in `API.md`, `openspec/project.md` (two spots: the
  `output_contract.py` architecture bullet, and the External Dependencies version literal),
  and `CLAUDE.md`.

## Impact

- Affected specs: `prediction-output` (MODIFIED)
- Affected code: `pyproject.toml`, `uv.lock`, `sleap_roots_predict/output_contract.py`,
  `tests/test_output_contract.py`, `CHANGELOG.md`, `API.md`, `openspec/project.md`,
  `CLAUDE.md`
- No behavior change for well-formed callers: `kind` has a default (`"predictions_slp"`), and
  no `PredictionArtifact(...)` construction site in the repo uses positional arguments.
- `docker-build.yml`'s PR trigger watches `pyproject.toml`/`uv.lock`/`sleap_roots_predict/**`,
  so this PR will automatically run its build-only (no push) validation job — expected, not a
  sign something is wrong. If it fails (e.g. an incompatible transitive dependency in
  `0.1.0a5`), revert the dependency-bump commit and file an issue against
  `sleap-roots-contracts` rather than pinning around it in this repo.
- Closes sleap-roots-predict#30. Downstream: unblocks Salk-Harnessing-Plants-Initiative/
  bloom#407 (not touched here).
