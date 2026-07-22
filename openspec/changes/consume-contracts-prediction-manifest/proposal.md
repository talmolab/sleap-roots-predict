## Why

`PredictionArtifact`/`PredictionManifest` were promoted into `sleap-roots-contracts`
(contracts PR #23, released `0.1.0a5` on PyPI) because bloom#407 needs a PyPI-installable way
to read predict's manifest without a git dependency on this unpublished repo. predict still
defines both classes locally in `sleap_roots_predict/output_contract.py` — a duplication risk
once bloomctl constructs/validates against contracts' version, since the two definitions have
no shared source of truth if they ever drift (contracts' version already differs by one field,
`kind`). This mirrors exactly how `resolve_params` was consumed from contracts in predict#28/
PR#29.

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
- Closes sleap-roots-predict#30. Downstream: unblocks Salk-Harnessing-Plants-Initiative/
  bloom#407 (not touched here).
