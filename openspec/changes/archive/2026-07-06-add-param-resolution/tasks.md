## 1. Scaffold module and field constants

- [x] 1.1 Create the **flat module** `sleap_roots_predict/param_resolution.py` (not a
  subpackage — see design.md) with a module docstring and the field-name constants
  (`SPECIES_NAME_FIELD == "species_name"`, `PLANT_AGE_DAYS_FIELD == "plant_age_days"`)
  documenting the bloomcli coupling. (Constants are exercised through the oracle-table test
  in 5.1 — no separate tautological constant-value test.)

## 2. Species normalization (TDD)

- [x] 2.1 Write failing tests for `_normalize_species`: a seeded passthrough
  (`Pennycress`→`pennycress`), whitespace/case (`"  Rice  "`→`"rice"`), unknown-species
  passthrough (`"Sorghum"`→`"sorghum"`), and blank/non-str inputs (`""`, `"  "`, `None`,
  `float("nan")`) → returns `""` (blank) without raising.
- [x] 2.2 Implement `_ALIASES = {}` (ships empty with a comment marking it the extension
  seam for names that don't lowercase cleanly; keys, when added, MUST be lowercase) +
  `_normalize_species` (coerce non-str/blank to `""`, else `strip().lower()` then alias
  lookup with lowercase passthrough fallback); make 2.1 pass. No whitelist / no hard-fail.

## 3. Imaging-mode seam (TDD)

- [x] 3.1 Write a failing test asserting `_mode_for_scan(row)` returns `"cylinder"` and that
  the string matches the seeded `ModelCard` mode vocabulary used in selection tests.
- [x] 3.2 Implement the one-line `_mode_for_scan` seam returning `"cylinder"` with a docstring
  documenting it as the single mode-decision point (GraviScan/multiscanner deferred here).

## 4. Age coercion in days (TDD)

- [x] 4.1 Write failing tests for `_coerce_age`: `14`→`14`; int-coercible string `"14"`→`14`
  (result is an `int`); `0`→`0` (valid, not treated as missing); non-whole/non-coercible
  (`14.5`, `"14.5"`, `"abc"`, `True`) → `ValueError` naming `age` (no silent truncation).
- [x] 4.2 Implement `_coerce_age` (reject bool/non-whole-float/non-coercible with a
  param-named `ValueError`; accept int and whole-number string); make 4.1 pass.

## 5. resolve_params core: mapping, override-wins, canonicalization, strict validation (TDD)

- [x] 5.1 Write the oracle-table test: a sample `cyl_scans_extended` row
  (`species_name="Pennycress"`, `plant_age_days=14`) → `values ==
  {"species": "pennycress", "mode": "cylinder", "age": 14}`, `param_hash` populated; plus a
  full-column row asserting only the 3 fields are used and `metadata` is not mutated.
- [x] 5.2 Write override tests: full and partial `overrides` win per field; `overrides={}`
  equals no overrides; an **unknown override key** (`{"specis": ...}`) raises `ValueError`
  naming the key; **override values are canonicalized** — `overrides={"species": "Rice",
  "age": "14"}` → `species="rice"`, `age=14`, and the resulting `param_hash` equals the
  equivalent derived run (representation-independent).
- [x] 5.3 Write validation tests: a row missing both `species_name` and `plant_age_days`
  (no overrides) raises `ValueError` naming **both** `species` and `age`; a **blank**
  `species_name` (`""`) with no override raises naming `species` (blank ≠ empty param); a
  missing `species_name` compensated by `overrides={"species": ...}` succeeds
  (tolerant-read → override → validate ordering).
- [x] 5.4 Implement `resolve_params(metadata, overrides=None)` per design.md: reject unknown
  override keys; read fields tolerantly via `.get` (mode always; species/age only when
  non-None); merge `{**derived, **overrides}` (override wins); canonicalize per field
  (`_normalize_species`, `_coerce_age`) dropping a blank species; `_require(...)` fail-loud
  by key-membership naming every absent param; return `ResolvedParams(values=values)`. Make
  5.1–5.3 pass.

## 6. Public API export (TDD)

- [x] 6.1 Extend `tests/test_public_api.py` asserting both
  `from sleap_roots_predict import resolve_params` works **and**
  `"resolve_params" in sleap_roots_predict.__all__` (the existing test only does `hasattr`).
- [x] 6.2 Add the import + `__all__` entry (and mention it in the module docstring) in
  `sleap_roots_predict/__init__.py`. Make 6.1 pass.

## 7. Payoff round-trip with choose_models (TDD)

- [x] 7.1 Write the demoable round-trip test: build a small real `ModelCard` list (mirroring
  `tests/test_model_selection.py`), then assert `choose_models(resolve_params(row), cards)`
  selects the expected `ModelRef` per root type (metadata → params → model).
- [x] 7.2 Add the end-to-end unknown-species case that proves the passthrough decision:
  a `species_name="Sorghum"` row → `choose_models(resolve_params(row), cards) == {}`
  (zero-match skip, no error).

## 8. Confirm age units (days) — decision, not a wandb regression test

- [x] 8.1 Confirm the seeded production cards' `age_min`/`age_max` are in **days** (to match
  `plant_age_days`): query a real seeded card via a gated `uv run pytest -m wandb` check or
  ask training. **Record the confirmed answer in `design.md` Decisions** (do NOT commit a
  `-m wandb` age-units test). If a conversion is required, add it to `_coerce_age` with an
  **offline** unit test, and file tracking issue #2 if it needs training-side coordination.

## 9. Docs + validation gate

- [x] 9.1 Document `resolve_params` as the metadata → params oracle feeding `choose_models`
  in each place that lists the public surface: `README.md` (Project Structure tree + prose),
  the `sleap_roots_predict/__init__.py` module docstring, the `openspec/project.md`
  Architecture-Patterns list, `API.md` (High-Level API entry), and a `CHANGELOG.md`
  `[Unreleased]/Added` entry. **Do NOT edit `CLAUDE.md`** (being retired).
- [x] 9.2 Run `openspec validate add-param-resolution --strict` and fix any issues.
- [x] 9.3 Run `/pre-merge` (format check, lint, test, build) and ensure it is green.
