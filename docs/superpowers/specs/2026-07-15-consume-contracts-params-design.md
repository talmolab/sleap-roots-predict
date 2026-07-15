# Design: consume `resolve_params` from `sleap-roots-contracts` (drop predict's local copy)

- **Date:** 2026-07-15
- **Repo:** `sleap-roots-predict`
- **Branch:** `consume-contracts-params`
- **Issue:** predict#28 ("Consume resolve_params from sleap-roots-contracts (drop local
  param_resolution.py)"). Origin: predict#18 (merged) — predict's local oracle. Related:
  predict#20 (TS-parity, re-pointed at the contracts oracle post-merge, not touched here).
- **Status:** APPROVED — brainstorming complete. This doc is the settled design. Proceed to
  the OpenSpec proposal (`/openspec:proposal`), suggested change-id `consume-contracts-params`.

## Motivation

`resolve_params` was promoted into `sleap-roots-contracts` as the single source of truth
(contracts PR #16, released `0.1.0a4` on PyPI) and additionally hardened the pandas/numpy
missing-data sentinel handling that predict's local copy got wrong (see "Behavior change"
below). predict still carries its own `param_resolution.py`, a byte-for-byte fork of the
pre-hardening implementation. Keeping both is a correctness risk: `resolve_params`'s output
feeds `ResolvedParams.param_hash` → `Provenance.idempotency_key` (first-writer-wins across
producers), so two implementations that diverge on an edge case would both "win" the same
dedup race with no error raised anywhere. This change makes predict a pure consumer.

## Settled decisions (from brainstorming 2026-07-15)

1. **Re-export preserved.** `sleap_roots_predict/__init__.py` keeps `resolve_params` in
   `__all__`, now sourced via `from sleap_roots_contracts import resolve_params`. Existing
   predict consumers (`from sleap_roots_predict import resolve_params`) see no break;
   `tests/test_public_api.py` needs no edit.
2. **Exact version pin.** `sleap-roots-contracts==0.1.0a3` → `==0.1.0a4`, matching predict's
   existing style for other pre-1.0 pins (e.g. `sleap-nn==0.3.0`). `uv.lock` relocked in the
   same commit (a version bump without relocking hard-fails the release build).
3. **`tests/test_param_resolution.py` re-pointed, not deleted or renamed.** Its remaining job
   changes: it no longer tests the oracle's internals (that's now contracts'
   `tests/test_params.py`, confirmed to have ported every predict case plus the hardening
   cases) — it tests that predict's `choose_models` still round-trips correctly against a
   `ResolvedParams` produced by an *external* implementation. Kept tests:
   `test_round_trip_selects_expected_models`, `test_round_trip_unknown_species_selects_nothing`.
   These import `resolve_params` from `sleap_roots_contracts` directly (not through predict's
   re-export), since the file's job is now specifically the cross-repo interop proof.
4. **`param_resolution.py` deleted outright**, no compatibility shim. Rejected alternative: a
   thin shim module that re-exports contracts' `resolve_params` under the old module path —
   softer migration, but explicitly ruled out (the issue calls for a clean subtraction); the
   grep sweep in tasks.md covers the "missed call site" risk instead.
5. **OpenSpec: `param-resolution` capability removed wholesale**; the one requirement worth
   keeping (`Interoperability With Model Selection` — the round-trip contract) is migrated
   into `model-management`'s existing `Model Selection From Scan Params` requirement as a
   MODIFIED delta, so the cross-repo contract stays documented at the spec level and isn't
   only implicit in test code.
6. **CHANGELOG: edit in place.** predict has never cut a release — everything so far sits
   under `[Unreleased]`. The existing "Param resolution" bullet is edited in place to
   attribute the implementation to `sleap-roots-contracts` and to call out the stricter
   sentinel handling as a behavior change, rather than adding a second bullet for a feature
   that was never actually shipped under the old behavior.

## Behavior change (must be called out, not silently absorbed)

Contracts' `resolve_params` is **not** byte-identical to predict's old copy — verified
identical on all 2,268 well-formed inputs contracts ported from predict's suite, but
diverges on inputs predict was resolving *wrongly*:

| Input | predict's old copy | contracts (`0.1.0a4`) |
|---|---|---|
| `pd.NA` / `pd.NaT` species | `str()` → `species="<na>"`, silently hashed | raises `ValueError` |
| `np.bool_(True)` age | `int()` → `age=1`, silently hashed | raises `ValueError` |
| non-string species (e.g. `123`) | `str()` → `"123"`, silently hashed | raises `ValueError` |
| `Decimal("14.5")` age | truncated to `14` | raises `ValueError` |
| `float("inf")` age | uncaught `OverflowError` | raises `ValueError` |

None of predict's own tests asserted the old (wrong) behavior on these inputs — confirmed by
reading `tests/test_param_resolution.py` — so no predict test needs to change to reflect this
divergence; it's purely a CHANGELOG-documented behavior change for any external caller that
happened to depend on the old silent coercion.

## Components touched

- `pyproject.toml` / `uv.lock` — version bump + relock.
- `sleap_roots_predict/__init__.py` — import swap (§1).
- `sleap_roots_predict/param_resolution.py` — deleted.
- `tests/test_param_resolution.py` — pruned to the two round-trip tests (§3).
- `openspec/specs/param-resolution/` — removed at archive time (§5).
- `openspec/specs/model-management/spec.md` — MODIFIED requirement at archive time (§5).
- `CHANGELOG.md` — edited in place (§6).
- Doc references to update (grep-verified as the complete set): `README.md` (architecture
  tree + the "See the `param-resolution` and `model-management` OpenSpec specs" line),
  `API.md` (same spec reference, plus a one-line note on the stricter sentinel handling),
  `openspec/project.md` (module-layout bullet listing `param_resolution.py`), `CLAUDE.md`
  (Package Structure blurb listing `param_resolution.py`).
- predict#20 — commented post-merge to re-point its reference oracle at
  `sleap_roots_contracts.resolve_params`; confirmed with the user before posting (visible
  cross-repo GitHub action).

## Testing approach

Subtractive refactor, not new-behavior work — the logic under test already lives (and is
tested, including the hardening cases) in contracts. TDD's usual red/green doesn't map
cleanly; the verification gates instead:

- Dependency bump first, in isolation: confirm `sleap_roots_contracts.__version__ ==
  "0.1.0a4"` importable, full predict suite still green (regression baseline before any code
  changes).
- Import swap + file deletion + test-file pruning done together (they're interdependent —
  deleting the module first would break the old test file's imports before it's re-pointed).
  Verification: full suite green, the two round-trip tests pass against the
  contracts-produced `ResolvedParams`.
- Grep sweep for stray `param_resolution`/`param-resolution` references across code and docs
  before calling it done (the 4 doc spots above were found this way; re-run after edits to
  confirm zero remaining stale references apart from historical `openspec/changes/archive/`
  entries, which are immutable history and intentionally left alone).
- `openspec validate --strict` on the REMOVED (`param-resolution`) and MODIFIED
  (`model-management`) deltas.
- Full `/pre-merge` gate (format, lint, test, build) before opening the PR.

## Out of scope

- Any TS port of `resolve_params` (predict#20 stays Python-side/re-pointed only, per the
  issue).
- The `sleap-roots-pipeline` roadmap doc update (`docs/bloom-integration/roadmap.md`) — flagged
  in the PR body as a pipeline-repo follow-up; that repo owns its roadmap.
- Any change to contracts itself — already shipped (PR #16, `0.1.0a4`).

## Acceptance

- predict imports `resolve_params` from contracts; `param_resolution.py` is gone; full test
  suite green; the two round-trip tests still pass against the contracts-produced
  `ResolvedParams`.
- `from sleap_roots_predict import resolve_params` still works (re-export preserved).
- CHANGELOG documents the behavior change; docs no longer reference the deleted module/spec.
- predict#20 commented (post-merge) re-pointing it at the contracts oracle.

## Cross-repo references

- **contracts**: `sleap-roots-contracts` PR #16 (the promotion + hardening, merged),
  `0.1.0a4` on PyPI, `src/sleap_roots_contracts/params.py` (implementation),
  `tests/test_params.py` (the full ported + hardening test suite).
- **predict**: predict#28 (this), predict#18 (origin, merged), predict#20 (TS-parity,
  re-pointed post-merge).
- **pipeline** (not touched here): `sleap-roots-pipeline`
  `docs/bloom-integration/roadmap.md` — flag as a follow-up in the PR body only.
