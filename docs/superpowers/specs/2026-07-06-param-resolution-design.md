# Design: A3-params — `resolve_params` (Bloom scan metadata → `ResolvedParams`)

- **Date:** 2026-07-06
- **Repo:** `sleap-roots-predict`
- **Branch:** `add-param-resolution`
- **Roadmap:** tier **A3-params** ("Bloom dataset metadata → `ResolvedParams`; oracle:
  given metadata X → expected params; user override wins") — currently `⬜` in
  `sleap-roots-pipeline/docs/bloom-integration/roadmap.md`.
- **Status:** **APPROVED — brainstorming complete.** This doc is the settled design.
  **Resuming in a new session: run `/new-feature`, but SKIP brainstorming** (it's done —
  this file is its output) and go straight to the **OpenSpec proposal** →
  `/review-openspec` → TDD. Suggested change-id: `add-param-resolution`.

## Motivation

Predict's `choose_models(params, cards)` consumes a `ResolvedParams` (`species`, `mode`,
`age`) and selects a production model per root type. Nothing yet **produces** that
`ResolvedParams` from real Bloom scan metadata — the roadmap's A3-params gap. This change
adds a pure oracle `resolve_params(metadata) -> ResolvedParams` so the metadata a scan
already carries in Bloom drives model selection end-to-end (metadata → params →
`choose_models` → warm worker → prediction).

## Settled decisions (from brainstorming 2026-07-06)

1. **Home = `sleap-roots-predict`.** The only dependency the pure mapper needs is
   `sleap-roots-contracts` (already pinned `==0.1.0a3`; predict already imports
   `ResolvedParams`). Co-located with `choose_models`; no new deps. Alternatives rejected:
   contracts (would add mapping *behavior* to a pure-schema lib), pipeline/A4 (not a Python
   lib; most scaffolding), salk-bloom/TS (different stack). A pure Python **oracle** is
   valuable regardless of where production emission eventually runs — including as the
   reference a future TS Bloom-side version is tested against.
2. **Call model = Model A** (predict resolves internally) **with overridability.** The
   predict step receives the raw scan metadata + an optional overrides dict, and calls
   `resolve_params(metadata, overrides)` itself. Overridability is preserved by the
   `overrides` argument (user override wins per field); overrides are collected at submit
   time and passed into predict. Model A keeps everything in predict; the one thing it
   gives up vs "resolve at submit" (Model B) is showing the user the *fully-resolved*
   params at submit for direct editing — acceptable when the override UX is "force these
   fields."
3. **Input = one `cyl_scans_extended` row** — the dict the bloomcli download writes to
   `scans.csv` (see "Input contract" below).
4. **Unknown species → lowercased passthrough** (NOT hard-fail, NOT skip-visibility). The
   **registry (ModelCards) is the single authority** on which species have models;
   `resolve_params` only *translates* Bloom's naming. New seeded species then work with no
   resolver edit; a species with no card degrades to `choose_models` zero-match → skip.
   Safe because Bloom's `species_name` is FK-controlled (a `species` table via
   `species_id`), so it's already a curated vocabulary — "unknown to the resolver" means "a
   real Bloom species not yet modelled," exactly where hard-fail would be wrong.
5. **Strict post-override validation.** After merging overrides, require `species`, `mode`,
   `age` all present → else raise a clear `ValueError` naming what's missing. Do not ship a
   half-resolved params object.
6. **Mode = scanner-determined, scoped to cylinder now, via a one-line seam.** See "Mode".

## Input contract (what `resolve_params` consumes)

A single `cyl_scans_extended` row — the exact shape bloomcli's
`bloomcli/src/bloomctl/download.py` (`_COLUMNS`) writes to `scans.csv`. Confirmed against
`salk-bloom` PRs #350/#351/#353 (bloomcli, authored by blm3886) and
`test_download_metadata.py`. Load-bearing fields (all others ignored):

| Bloom field | → param | Notes |
|---|---|---|
| `species_name` | `species` | e.g. `"Pennycress"`; FK-controlled via `species_id` → `species` table. Also present: `species_genus` (`"Thlaspi"`), `species_species` (`"arvense"`) — available as cross-checks. |
| `plant_age_days` | `age` | int, **days** (e.g. `14`). |
| (the scan's scanner) | `mode` | Cylinder pipeline (`cyl_scans_extended`) ⇒ `"cylinder"`. |

Field names are module constants matching bloomcli's column names, so the coupling is
explicit and greppable.

## The mapping

```python
def resolve_params(metadata, overrides=None) -> ResolvedParams:
    derived = {
        "species": _normalize_species(metadata["species_name"]),
        "mode":    _mode_for_scan(metadata),        # "cylinder" now; scanner→mode seam
        "age":     int(metadata["plant_age_days"]),
    }
    values = {**derived, **(overrides or {})}        # user override wins, per field
    _require(values, ("species", "mode", "age"))     # fail-loud if any missing post-override
    return ResolvedParams(values=values)             # contract computes param_hash
```

- **species** — `_normalize_species(name)`: an explicit alias map for the seeded species
  (`Pennycress→pennycress`, `Arabidopsis→arabidopsis`, `Rice→rice`, `Soybean→soybean`,
  `Canola→canola`) with a **lowercase passthrough fallback** for anything not in the map.
  ```python
  def _normalize_species(name):
      return _ALIASES.get(name.strip().lower(), name.strip().lower())
  ```
- **mode** — `_mode_for_scan(metadata)` returns `"cylinder"` (the cylinder pipeline yields
  cylinder scans only). See "Mode" for the seam rationale.
- **age** — `plant_age_days` straight through as `int`.
- **overrides** — a **param-space** dict (`{"mode": "..."}`, `{"species": "..."}`), merged
  last; override wins per field. Overrides express final param values, not metadata fields.

## Mode: the scanner→mode seam

Mode is a property of the scan's **scanner** (Bloom models modality by separate table
families: `cyl_*` cylinder scanners vs `gravi_*` GraviScan plate scanners; multiscanner is
**not implemented**; multiplant-cylinder is **retired**). For the current **cylinder**
stage-in path, mode is unconditionally `"cylinder"`.

Rather than inline that literal, mode routes through a one-line function:

```python
def _mode_for_scan(metadata):
    """Imaging modality for a scan. The cylinder pipeline yields cylinder scans only;
    this is the single place mode is decided — where GraviScan / multiscanner modes
    slot in once their scanners + models exist. Mode strings MUST match the exact
    seeded ModelCard vocabulary."""
    return "cylinder"
```

This **seam** (not the future implementation — just its shape) means: when GraviScan
models are seeded and a graviscan stage-in path exists, mode derivation becomes a
scanner→modality lookup confined to this one function; `resolve_params`'s body, its
callers, and its output shape (`{species, mode, age}`) do not change. It also centralizes
the "send the exact seeded mode string" contract in one place. We do **NOT** build the
scanner lookup / `_SCANNER_MODE` table now (rejected as speculative — graviscan models are
deferred and its download isn't wired).

## Module & API

- New file `sleap_roots_predict/param_resolution.py` (beside `model_selection.py` /
  `model_registry.py`).
- Public `resolve_params`, exported from `__init__.py` (+ `__all__`).
- Pure: imports only `ResolvedParams` from the contract; no network, no filesystem I/O.

## Testing — the oracle (+ the payoff round-trip)

Real, no-mock, offline (matches repo convention). TDD.

- **Oracle table** (`metadata X → expected params.values`): the sample bloomcli row →
  `{species: "pennycress", mode: "cylinder", age: 14}`; species-normalization cases
  (`Pennycress`, `Arabidopsis`, `Rice`, `Soybean`, `Canola`, plus a Latin/genus variant if
  Bloom uses one); age passthrough; unknown-species passthrough (`"Sorghum"`→`"sorghum"`).
- **Override-wins**: `resolve_params(row, {"mode": "x", "species": "y"})` → those win per
  field; partial overrides leave other fields derived.
- **Strict validation**: metadata missing `species_name`/`plant_age_days` with no override
  → clear `ValueError` naming the missing param.
- **The demoable round-trip** (ties A3-params to the shipped selection layer): build a
  small real `ModelCard` list (as the existing `choose_models` tests do), then
  `choose_models(resolve_params(row), cards)` selects the right `ModelRef` per root type.
  This is the metadata → params → model slide.

## Confirm during implementation

- **Age units.** Verify the seeded production cards' `age_min`/`age_max` are in **days**
  (to match `plant_age_days`). Check a real seeded card (gated `pytest -m wandb`, or ask
  training). If they are NOT days, add a documented conversion in the age mapping. (If this
  turns out to need training coordination, file the issue below rather than block.)
- **Species strings.** Confirm the seeded cards' exact `species` strings are the lowercase
  common names assumed here; extend `_ALIASES` for any Bloom `species_name` that doesn't
  lowercase cleanly to the card string (e.g. Latin binomials).

## Out of scope (seams / deferred)

- GraviScan / multiscanner **modes** — the `_mode_for_scan` seam.
- **Fetching** the metadata (Supabase / bloomcli) — stays in A4/bloomcli; the mapper takes
  a dict it's handed.
- **Where `resolve_params` is called** in production (predict-internal vs a CLI vs a TS
  port) — a thin adapter, an A4-era decision; the oracle is call-site-agnostic.

## Future work — issues to file

Create these to track the deferred pieces (repo · one-liner · dependency):

1. **predict — GraviScan/plate `mode` support in `resolve_params`.** Implement the
   `_mode_for_scan` scanner→modality lookup (cylinder vs gravi) once GraviScan plate models
   are seeded and a graviscan stage-in path exists. *Depends on:* training #3 (plate
   models) + a bloomcli graviscan download.
2. **predict — confirm/handle age units.** Verify seeded card `age_min/age_max` are in days
   vs `plant_age_days`; add a conversion if not. (May be resolved inside this change; file
   only if it needs training-side coordination.)
3. **pipeline/A4 — wire `resolve_params` into the predict step (call-site adapter).** Decide
   Model A internal call vs a `python -m ... resolve_params` CLI; pass scan metadata +
   user overrides into the predict step; log the resolved params in provenance. *Depends
   on:* A4 start.
4. **salk-bloom / A4 — override plumbing UX.** How user overrides are collected at submit
   time and carried into predict (the "user override wins" surface).
5. **bloom / training — multiscanner (multi-scan) modality.** Not implemented; when it
   lands, add its scanner family + mode string + models, then extend `_mode_for_scan`.
6. **(watch) TS parity.** If a Bloom-side (TS) params resolver is ever needed, test it
   against this Python oracle as the reference (given/expected table).

## Acceptance

- `resolve_params(cyl_scans_extended_row)` returns a `ResolvedParams` with
  `{species, mode, age}` matching the oracle table; overrides win per field; missing
  required fields fail loud.
- `choose_models(resolve_params(row), seeded_cards)` selects the expected model(s).
- Pure/offline; zero new dependencies; exported from the public API.
- Advances roadmap A3-params; report back so the roadmap row + tracking issues update.

## Cross-repo references (for the next session)

- **bloomcli** (input contract): `salk-bloom` `bloomcli/src/bloomctl/download.py` (`_COLUMNS`,
  `build_scan_row`) + `bloomcli/tests/test_download_metadata.py`; PRs #350/#351/#353 (author
  blm3886, unreviewed). Package dir `bloomcli/`, import name `bloomctl`.
- **Contract**: `sleap-roots-contracts` `ResolvedParams` (`values: dict`, `param_hash`) at
  `src/sleap_roots_contracts/models.py`; pinned `==0.1.0a3`.
- **Consumer**: predict `choose_models` (`sleap_roots_predict/model_selection.py`) + the
  `model-management` OpenSpec spec.
- **Not related** (clarified this session): the source-aware trait **read** RPC
  (`get_scan_traits` + `source_id`, salk-bloom #373, authored by eberrigan) is the
  results-read-OUT path; A3-params is the stage-IN side — they don't overlap.
