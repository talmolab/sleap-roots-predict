# Design: consume `PredictionArtifact`/`PredictionManifest` from `sleap-roots-contracts` (drop predict's local definitions)

- **Date:** 2026-07-22
- **Repo:** `sleap-roots-predict`
- **Branch:** `consume-contracts-prediction-manifest`
- **Issue:** predict#30 ("Consume PredictionArtifact/PredictionManifest from
  sleap-roots-contracts (drop local output_contract.py definition)"). Depends on:
  contracts#22 (merged). Precedent: predict#28/PR#29 (the `resolve_params` migration).
  Downstream: Salk-Harnessing-Plants-Initiative/bloom#407.
- **Status:** APPROVED — brainstorming complete. This doc is the settled design. Proceed to
  the OpenSpec proposal (`/openspec:proposal`), suggested change-id
  `consume-contracts-prediction-manifest`.

## Motivation

`PredictionArtifact`/`PredictionManifest` were promoted into `sleap-roots-contracts` (contracts
PR #23, released `0.1.0a5` on PyPI — confirmed live) because bloom#407 needs a
PyPI-installable way to read predict's manifest without a git dependency on this unpublished
repo. predict still defines both classes locally in `sleap_roots_predict/output_contract.py`.
Keeping both is a duplication risk once bloomctl starts constructing/validating against
contracts' version: predict would be writing manifests shaped by one definition while a
downstream consumer validates against another, with no shared source of truth if the two ever
drift (e.g. contracts' new `kind` field). This change makes predict a pure consumer of the
promoted models, exactly mirroring how `resolve_params` was consumed in #29.

## Settled decisions (from brainstorming 2026-07-22)

1. **Exact version pin.** `sleap-roots-contracts==0.1.0a4` → `==0.1.0a5` in `pyproject.toml`,
   matching predict's existing style for pre-1.0 pins. `uv.lock` relocked in the same commit,
   diff scoped to just the `sleap-roots-contracts` entry (verify per #29's precedent).
2. **Import lives in `output_contract.py`, not `__init__.py`.** Unlike `param_resolution.py`
   (deleted outright in #29), `output_contract.py` is not going away — it still owns the
   writer functions (`write_prediction_outputs`, `predict_and_write_batch`, `ScanRequest`,
   `slugify_model_id`, `_validate_scan_key`, `_resolve_identity`), which are untouched by this
   change. `output_contract.py`'s two local class *definitions* are replaced with
   `from sleap_roots_contracts import PredictionArtifact, PredictionManifest`, mirroring how
   it already imports `ModelRef`/`ResolvedParams`/`RootType` from contracts at the top of the
   file today. `sleap_roots_predict/__init__.py` is **unchanged** — it keeps importing both
   names from `sleap_roots_predict.output_contract`, so the re-export and `__all__` entries
   need no edit and `tests/test_public_api.py` needs no change.
3. **The 2 pure-model tests in `tests/test_output_contract.py` are pruned to zero**
   (`test_manifest_round_trips`, `test_plant_qr_code_defaults_to_scan_key`). Contracts' own
   `tests/test_prediction_manifest.py` now owns this coverage (round-trip, immutability,
   `plant_qr_code` default, `kind` default/validation, `root_type` validation). Unlike
   `resolve_params` in #29 — a function predict calls at runtime through `choose_models`,
   where the 2 kept tests verified predict's *own* wiring against an externally-produced
   value — `PredictionArtifact`/`PredictionManifest` are now just imported data models with no
   predict-specific logic wrapping them. There is no predict-side wiring left to test once the
   import swaps; a thin smoke test would only re-assert "the import works," which
   `tests/test_public_api.py`'s existing `hasattr`/`__all__` check plus every other test in the
   file (all of which construct real manifests through the writer) already cover incidentally.
   The other 24 tests in the file — which exercise the real warm-worker/filesystem writer —
   are untouched.
4. **`openspec/specs/prediction-output/spec.md`: MODIFIED delta restates the full shape.**
   Both affected requirements ("Combined per-scan manifest and provenance sidecar" and
   the artifact fields it lists) get their complete requirement text pasted and edited per
   OpenSpec's MODIFIED convention — add `kind` (`BlobKind`, defaults to `"predictions_slp"`)
   to `PredictionArtifact`'s field list, and add one sentence noting the shape is now defined
   by `sleap-roots-contracts`' `prediction-manifest-contract` capability and imported, not
   defined locally in this repo. Chosen over a thin pointer-only delta so predict's own
   OpenSpec spec remains a complete, standalone reference for "what does predict write to
   disk" without a cross-repo lookup — consistent with how contracts and predict are
   independent OpenSpec installations with no cross-repo linking mechanism, so predict's spec
   needs to stand on its own.
5. **CHANGELOG: edit in place.** predict has never cut a release — everything sits under
   `[Unreleased]`. The existing "Predict output contract" bullet (the one immediately below
   the already-migrated "Param resolution" bullet) is edited in place to attribute
   `PredictionArtifact`/`PredictionManifest` to `sleap-roots-contracts==0.1.0a5` and note the
   new `kind` field, rather than adding a second bullet for a shape that was never shipped
   under the old (local) definition.
6. **No compatibility shim.** `output_contract.py` keeps the same import *names* available at
   the same module path (so nothing outside the file needs to change), but there is no
   separate re-export shim module — the two class bodies are simply replaced by the import
   line. This matches #29's "clean subtraction" framing without also matching #29's full
   file deletion, since here the module survives for other reasons.

## Behavior change

None for existing well-formed callers. `PredictionArtifact` gains one new field, `kind:
BlobKind = "predictions_slp"`, which has a default — `write_prediction_outputs`'s existing
construction call site (`output_contract.py`'s `write_prediction_outputs`) does not need to
pass it and continues to work unchanged (to be confirmed empirically during implementation,
per the issue's own caution against assuming). All other fields match name-for-name and
type-for-type between predict's old local classes and contracts' promoted versions.

## Components touched

- `pyproject.toml` / `uv.lock` — version bump + relock (§1).
- `sleap_roots_predict/output_contract.py` — replace the two local class bodies with a
  contracts import; docstring's "schema is predict-local for now... promoted... once traits
  consumes it" note is now stale and gets updated to reflect the promotion has happened (§2).
- `sleap_roots_predict/__init__.py` — unchanged (§2).
- `tests/test_output_contract.py` — remove `test_manifest_round_trips` and
  `test_plant_qr_code_defaults_to_scan_key`; remove the now-unused `PredictionArtifact`
  import if nothing else in the file constructs one directly (confirm during implementation;
  several of the remaining 24 tests reference `PredictionManifest` return values, so that
  import likely stays) (§3).
- `openspec/specs/prediction-output/spec.md` — MODIFIED delta at archive time (§4).
- `CHANGELOG.md` — edited in place (§5).
- Doc references to update (grep-verified as the complete set): `API.md` (the "kept
  single-sourced there to avoid drift" line, now stale since the schema is imported, not
  defined, in `output_contract.py`'s docstrings), `openspec/project.md` (the
  `output_contract.py` bullet's "+ the manifest/artifact models" phrasing, updated to match
  the `model_selection.py` bullet's existing "consumes... predict carries no local copy"
  phrasing for `resolve_params`), `CLAUDE.md` (Package Structure blurb, same treatment).
  `README.md`'s architecture-tree comment (`output_contract.py # Per-scan output artifacts`)
  needs no change — it doesn't describe the models' origin.

## Testing approach

Subtractive refactor, not new-behavior work — same shape as #29:

- Dependency bump first, in isolation: confirm `sleap_roots_contracts.PredictionArtifact` /
  `.PredictionManifest` importable at `0.1.0a5`, full predict suite still green (regression
  baseline before any code changes).
- Import swap in `output_contract.py` + test-file pruning done together (interdependent —
  pruning first would leave dead imports; swapping first still passes since the field sets
  match). Verification: full suite green, remaining 24 tests in `test_output_contract.py`
  still pass unmodified (they exercise the writer, not the class definitions).
- Empirically confirm the `kind` field's default means `write_prediction_outputs`'s
  construction call site needs no changes (per the issue's explicit caution) — run the
  existing writer tests and inspect a produced manifest's JSON for `"kind":
  "predictions_slp"`.
- Grep sweep for stray references to predict's local class definitions across code and docs
  before calling it done.
- `openspec validate --strict` on the MODIFIED (`prediction-output`) delta.
- Full `/pre-merge` gate (format, lint, test, build) before opening the PR.

## Out of scope

- Any change to `bloomctl`/bloom#407 itself (that's the consuming side, already unblocked by
  contracts' `0.1.0a5` release).
- The paused `salk-bloom` OpenSpec change `add-cyl-blob-upload` — resumed separately after
  this merges (already flagged inline in that change's `proposal.md`/`design.md`).
- Any further change to contracts itself — already shipped (PR #23, `0.1.0a5`).
- Emitting the full `Provenance`/`ResultEnvelope` or the prediction-parity harness (separate
  A3/A4 roadmap items per `openspec/project.md`).

## Acceptance

- predict imports `PredictionArtifact`/`PredictionManifest` from contracts;
  `output_contract.py`'s local class bodies are gone; full test suite green.
- `from sleap_roots_predict import PredictionArtifact, PredictionManifest` still works
  (re-export preserved, no `__init__.py` edit needed).
- A produced manifest's JSON includes `"kind": "predictions_slp"` per artifact.
- CHANGELOG documents the shape now coming from contracts; docs no longer claim the schema is
  "single-sourced" locally.

## Cross-repo references

- **contracts**: `sleap-roots-contracts` PR #23 (the promotion, merged), `0.1.0a5` on PyPI,
  `src/sleap_roots_contracts/prediction_manifest.py` (implementation),
  `tests/test_prediction_manifest.py` (the full model-level test suite),
  `openspec/specs/prediction-manifest-contract/spec.md` (the capability spec).
- **predict**: predict#30 (this), predict#28/PR#29 (the `resolve_params` precedent this
  mirrors).
- **bloom**: Salk-Harnessing-Plants-Initiative/bloom#407 (downstream consumer, unblocked by
  contracts' release, not touched here).
