## Why

Predict's `choose_models(params, cards)` consumes a `ResolvedParams` (`species`,
`mode`, `age`) and selects a production model per root type, but nothing yet
**produces** that `ResolvedParams` from real Bloom scan metadata — the roadmap's
**A3-params** gap. Without an oracle that maps the metadata a scan already carries
in Bloom to params, the selection layer cannot be driven from production data.

## What Changes

- Add a pure oracle `resolve_params(metadata, overrides=None) -> ResolvedParams`
  in a new `sleap_roots_predict/param_resolution.py`, mapping a single
  `cyl_scans_extended` row (the dict bloomcli's download writes to `scans.csv`)
  to `{species, mode, age}`:
  - **species** — `species_name` normalized via an alias map with a lowercase
    passthrough fallback (unknown species pass through; the registry/`ModelCard`s
    are the sole authority on which species have models).
  - **mode** — derived through a one-line `_mode_for_scan` **seam** that returns
    `"cylinder"` for the current cylinder stage-in path (GraviScan/multiscanner
    modes are deferred to that seam).
  - **age** — `plant_age_days` as an `int` (days).
- **Override-wins per field:** an optional param-space `overrides` dict is merged
  last, so a caller-supplied field replaces the derived one. Override keys are
  restricted to `{species, mode, age}` — an unknown key raises rather than
  silently polluting `values`. Override **values** are canonicalized by the same
  per-field rules as derived values (species normalized, age coerced), so a run is
  hashed representation-independently (an `age` override of `"14"` and a derived
  `14` produce the same `param_hash`).
- **Strict post-override validation:** blank/absent load-bearing fields are treated
  as not provided (a blank `species_name` fails loud rather than resolving an empty
  species); after merging, require `species`, `mode`, and `age` all present (by
  key, so `age=0` is valid), else raise a clear `ValueError` naming every missing
  param. A lossy `plant_age_days` (non-whole/non-coercible) raises naming `age`.
  These guards protect the reproducibility hash and close holes `choose_models`
  cannot reach (it never recomputes the hash).
- Export `resolve_params` from the package public API (`__init__.py` + `__all__`).
- Pure/offline: the only dependency is `sleap-roots-contracts` (already pinned,
  already imported by predict); **zero new dependencies**; no network, no I/O.

## Impact

- Affected specs: **param-resolution** (new capability). No change to the existing
  `model-management` spec — `choose_models` is the unchanged consumer of the output.
- Affected code: new `sleap_roots_predict/param_resolution.py`;
  `sleap_roots_predict/__init__.py` (export + module docstring); new
  `tests/test_param_resolution.py`; `tests/test_public_api.py` (extend).
- Affected docs: `README.md`, `API.md`, `CHANGELOG.md`, `openspec/project.md`
  (public-surface + architecture list). `CLAUDE.md` is intentionally **not** touched
  (being retired).
- Advances roadmap tier **A3-params** (`sleap-roots-pipeline/docs/bloom-integration/roadmap.md`).
- Settled design (approved 2026-07-06):
  `docs/superpowers/specs/2026-07-06-param-resolution-design.md`.
