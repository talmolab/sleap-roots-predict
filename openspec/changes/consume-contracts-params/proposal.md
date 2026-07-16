## Why

`resolve_params` was promoted into `sleap-roots-contracts` as the single source of truth
(contracts PR #16, released `0.1.0a4`), which additionally hardened pandas/numpy
missing-data sentinel handling that predict's local `param_resolution.py` copy got wrong
(e.g. `pd.NA` species silently hashed to `"<na>"`, `np.bool_` age silently coerced to `1`).
Keeping predict's fork risks two producers computing `param_hash` differently for the same
logical scan, silently breaking the first-writer-wins idempotency contract. predict#28 is
this migration; predict#18 was the origin (now superseded).

## What Changes

- Bump `sleap-roots-contracts` to `==0.1.0a4` in `pyproject.toml`; relock `uv.lock`.
- Replace `sleap_roots_predict/__init__.py`'s `resolve_params` import with
  `from sleap_roots_contracts import resolve_params` (stays in `__all__` — re-export
  preserved for existing predict consumers).
- Delete `sleap_roots_predict/param_resolution.py` entirely.
- Prune `tests/test_param_resolution.py` to the two `choose_models` round-trip tests
  (the only remaining coverage of the metadata → params → model wiring, since
  `choose_models` lives in predict); import `resolve_params` from `sleap_roots_contracts`
  directly.
- **BREAKING (behavior)**: malformed scan-metadata inputs that predict's old copy silently
  coerced now raise `ValueError` instead — `pd.NA`/`pd.NaT` species, `np.bool_`/`Decimal`/
  `inf` age, non-string species (e.g. `123`). No well-formed input's behavior changes.
- Update doc references (`README.md`, `API.md`, `openspec/project.md`, `CLAUDE.md`) that
  named the deleted module/spec.
- Edit the CHANGELOG's existing (still `[Unreleased]`) "Param resolution" bullet in place
  to attribute the implementation to contracts and note the behavior change.

## Impact

- Affected specs:
  - `param-resolution` — **REMOVED** wholesale (contracts now owns this capability).
  - `model-management` — **MODIFIED** (`Model Selection From Scan Params` gains a clause +
    scenario documenting `choose_models`' interop with an externally-produced
    `ResolvedParams`).
- Affected code: `pyproject.toml`, `uv.lock`, `sleap_roots_predict/__init__.py`,
  `sleap_roots_predict/param_resolution.py` (deleted), `tests/test_param_resolution.py`,
  `README.md`, `API.md`, `CLAUDE.md`, `openspec/project.md`, `CHANGELOG.md`.
- Not in scope: any TS port of `resolve_params` (predict#20 stays Python-side, re-pointed
  at the contracts oracle post-merge); the `sleap-roots-pipeline` roadmap doc (flagged as a
  pipeline-repo follow-up in the PR body only).
