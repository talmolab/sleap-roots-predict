## Context

Full brainstorming output (approved 2026-07-06):
`docs/superpowers/specs/2026-07-06-param-resolution-design.md`. This file records the
decisions load-bearing for the spec deltas plus the refinements surfaced by two
`/review-openspec` rounds, and the one open question to resolve during implementation.

The selection layer (`choose_models`, `model-management` spec) already consumes a
`ResolvedParams(values={species, mode, age})`. This change adds the missing **producer**:
a pure oracle mapping a Bloom `cyl_scans_extended` row to that `ResolvedParams`. Home is
`sleap-roots-predict` because the only dependency is `sleap-roots-contracts` (already
pinned `==0.1.0a3`, already imported by predict) and the oracle belongs beside its
consumer; a pure Python oracle is also valuable as the reference a future TS Bloom-side
resolver is tested against.

## Goals / Non-Goals

- **Goals:** a pure, offline, zero-new-dependency `resolve_params(metadata, overrides=None)`
  that returns a valid `choose_models` input with a **representation-independent
  `param_hash`**; override-wins per field; fail-loud on missing required params, unknown
  override keys, and lossy age coercion; exported from the public API.
- **Non-Goals (seams/deferred):** the scanner→mode lookup for GraviScan/multiscanner
  (only the `_mode_for_scan` seam is built now); **fetching** the metadata (A4/bloomcli);
  the production call-site adapter (predict-internal vs CLI vs TS); the override submit-time
  UX. These are tracked as follow-up issues.

## Decisions

- **Call model = Model A (predict resolves internally) with an `overrides` arg.** Predict
  receives the raw scan metadata + an optional param-space overrides dict and calls
  `resolve_params(metadata, overrides)` itself. Overridability is preserved by the argument
  (override wins per field). Alternative "resolve at submit" (Model B) was rejected — it
  only buys showing the fully-resolved params at submit for direct editing, unnecessary
  when the override UX is "force these fields." (Note: because Model B is rejected, the
  human never sees the resolved params, so the resolver — not the caller — must canonicalize
  override values; see below.)
- **Read raw → merge → canonicalize per field → validate.** Fields are read with `.get`
  and blank values (None / non-str / empty-after-strip) are omitted (not a read-time
  `KeyError`, not a blank param); this is what lets an override supply a missing field and
  lets validation name what is still missing. Crucially, **the same per-field
  canonicalization is applied to derived and override values** so `param_hash` is
  representation-independent. Reference shape:
  ```python
  def resolve_params(metadata, overrides=None):
      overrides = overrides or {}
      _reject_unknown_override_keys(overrides)          # keys ⊆ {species, mode, age}
      values = {"mode": _mode_for_scan(metadata)}
      if metadata.get(SPECIES_NAME_FIELD) is not None:
          values["species"] = metadata[SPECIES_NAME_FIELD]   # raw; canonicalized below
      if metadata.get(PLANT_AGE_DAYS_FIELD) is not None:
          values["age"] = metadata[PLANT_AGE_DAYS_FIELD]      # raw; canonicalized below
      values = {**values, **overrides}                  # override wins, per field
      # canonicalize derived OR override values identically
      if "species" in values:
          species = _normalize_species(values["species"])     # non-str/blank -> ""
          values["species"] = species if species else _drop("species", values)
      if "age" in values:
          values["age"] = _coerce_age(values["age"])           # int; ValueError('age') if lossy
      _require(values, ("species", "mode", "age"))     # `k in values`; names EVERY absent param
      return ResolvedParams(values=values)             # contract computes param_hash
  ```
  (`_drop` illustrates "blank species → treat as absent → let `_require` name it"; the real
  implementation simply deletes the key. `_require` uses key-membership, not truthiness, so
  `age == 0` is valid.)
- **Override values are canonicalized, not stored raw (surfaced by review, verified against
  the contract).** `compute_param_hash` hashes the entire `values` dict via canonical JSON;
  `int 14` and `str "14"` hash **differently** (integer-valued floats collapse to int, but
  strings pass through untouched). CLI/JSON overrides are always strings, so storing an
  override `age="14"` raw would give the same scan a different `param_hash` (→ different
  `idempotency_key` → duplicate work) than the derived `14` path — the exact failure this
  module exists to prevent. Normalizing/coercing override values preserves "override wins"
  (the override still wins the field) while making the hash representation-independent.
- **Unknown species → lowercased passthrough, not hard-fail.** The registry (`ModelCard`s)
  is the single authority on which species have models; `resolve_params` only *translates*
  Bloom's naming. A new seeded species then works with no resolver edit; a species with no
  card degrades to `choose_models` zero-match → skip. Safe because Bloom's `species_name`
  is FK-controlled (a curated `species` table), so "unknown to the resolver" means "a real
  Bloom species not yet modelled." A **blank** `species_name` is a different case (corrupt/
  null row, not a real species) → treated as absent → fail loud, so it is not laundered as
  an unmodelled species.
- **Alias map keyed by lowercased name; ships empty.** `_normalize_species` does
  `_ALIASES.get(name.strip().lower(), name.strip().lower())`, so any map keys MUST be
  lowercase or they never fire. Because the seeded species' common names lowercase cleanly
  to the card strings, identity entries would be dead no-ops; `_ALIASES` therefore **ships
  empty** with a comment marking it the extension seam for Bloom names that do *not*
  lowercase cleanly (e.g. a Latin binomial). The verified seeded-species set is recorded in
  the payoff round-trip test (its executable home), not as no-op production entries. Mode
  strings and species strings MUST equal the exact seeded `ModelCard` vocabulary.
- **Mode via a one-line seam (`_mode_for_scan`), scoped to cylinder now.** Mode is a
  property of the scan's scanner (`cyl_*` vs `gravi_*`; multiscanner not implemented;
  multiplant-cylinder retired). We build only the seam's *shape*, not the future
  `_SCANNER_MODE` table (rejected as speculative while GraviScan models are deferred).
- **Strict validation, hardened (both verified load-bearing in review round 2).** Beyond
  "require species/mode/age present," the resolver (a) rejects unknown override keys so a
  typo'd key cannot silently enter `values` and change `param_hash`, and (b) rejects a lossy
  age coercion naming `age`. Both protect the reproducibility hash and both close holes
  `choose_models` structurally cannot reach (it never recomputes the hash).
- **Field names as module constants** (`SPECIES_NAME_FIELD`, `PLANT_AGE_DAYS_FIELD`)
  matching bloomcli's column names, so the cross-repo coupling to `cyl_scans_extended` is
  explicit and greppable.
- **Flat module, not a subpackage.** `pyproject.toml` packages the project via an explicit
  `packages = ["sleap_roots_predict"]` list, so `param_resolution.py` MUST be a flat module;
  a `param_resolution/` subpackage would be silently omitted from the built wheel.

## Risks / Trade-offs

- **Age units — CONFIRMED days (2026-07-06); issue #2 resolved, not deferred.** Queried the
  live production registry (`sleap-roots-models`, 13 cards). The seeded cards' age windows
  are `age_min`/`age_max` in **days**: arabidopsis/pennycress `2–14`, canola `2–13`, rice
  crown `2–5` and `6–10` (two non-overlapping windows), rice primary `2–5`, soybean `2–8`.
  These match `plant_age_days` (days), so the resolver passes `age` straight through as an
  `int` with **no conversion**. The same query confirmed the species vocabulary is exactly
  the lowercase common names `_normalize_species` produces (`arabidopsis`, `canola`,
  `pennycress`, `rice`, `soybean`), so `_ALIASES` correctly ships empty. (Observation: the
  registry also carries a distinct `multiplant cylinder` mode on two arabidopsis cards; the
  current cylinder stage-in path emits `"cylinder"` and does not target those — consistent
  with multiplant-cylinder being retired. Recorded for the `_mode_for_scan` seam / issue #1.)
- **`param_hash` is not self-versioning (verified against the contract).** `ResolvedParams`
  carries only `values` + `param_hash`; its hash's meaning depends on the resolver's emitted
  key-set and normalization rules (`_ALIASES`, `_mode_for_scan`, `_coerce_age`). A future
  change that adds a param, edits an alias, or changes the mode seam makes the **same scan
  hash differently**, with nothing in `ResolvedParams` signalling the shift. Mitigated at
  the provenance level: `Provenance.idempotency_key` folds in `predict_code_sha`, so a
  resolver code change yields a different idempotency key (no false dedup). Acceptable to
  defer, but **any future add/remove of an emitted param or alias-map edit is a
  `param_hash`-breaking change to call out explicitly**, since `git revert` does not restore
  already-persisted hashes.
- **No defensive copy needed.** pydantic v2 copies the input dict on validation
  (`ResolvedParams(values=d).values is d` → `False`), and `resolve_params` builds fresh
  dicts anyway, so there is no mutation-after-construct aliasing from the caller. (The
  contract's `values` field is only shallow-frozen — an upstream limitation of
  `sleap-roots-contracts`, out of scope here.)
- **Species alias drift.** If a Bloom `species_name` doesn't lowercase cleanly to the card
  string, selection silently zero-matches. Mitigation: `_ALIASES` is the one place to
  extend, and the payoff round-trip test catches a mismatch for the seeded species.

## Migration Plan

Additive only — a new flat module, a new public export, and new tests. No existing behavior
changes; `choose_models` and the `model-management` spec are untouched. `choose_models`
does not import `resolve_params`, so nothing on `main` depends on the new code and a single
`git revert` of the squash-merge reverses it cleanly.

## Open Questions

- **Age units** — see Risks (Task 8). Record the confirmed answer in Decisions; file
  tracking issue #2 only if it needs training-side coordination.
