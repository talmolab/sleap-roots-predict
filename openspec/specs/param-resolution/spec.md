# param-resolution Specification

## Purpose
TBD - created by archiving change add-param-resolution. Update Purpose after archive.
## Requirements
### Requirement: Scan Metadata Resolution To Params

The system SHALL provide a pure function `resolve_params(metadata, overrides=None) ->
ResolvedParams` that maps a single Bloom `cyl_scans_extended` scan-metadata record (the
dict shape bloomcli's download writes to `scans.csv`) to a `ResolvedParams` carrying
`species`, `mode`, and `age`. It SHALL read the load-bearing fields via module constants
matching bloomcli's column names: `species_name` → `species` (normalized), the scan's
scanner → `mode` (via the imaging-mode seam), and `plant_age_days` → `age`. A load-bearing
field that is **absent or blank** (missing key, `None`, a non-string sentinel such as a
`NaN`, or an empty/whitespace-only string) SHALL be treated as not provided — the function
SHALL omit that derived param (deferring to `overrides` and then to post-override
validation) rather than raising at read time or emitting a blank param. It SHALL construct
`ResolvedParams(values=…)` so the contract computes `param_hash`. The function SHALL be
pure: it SHALL perform no network access and no filesystem I/O, and SHALL NOT mutate the
input `metadata`.

#### Scenario: A sample cylinder scan row resolves to species/mode/age

- **WHEN** `resolve_params` is called with a `cyl_scans_extended` row carrying
  `species_name="Pennycress"` and `plant_age_days=14`
- **THEN** it returns a `ResolvedParams` whose `values` are
  `{"species": "pennycress", "mode": "cylinder", "age": 14}` and whose `param_hash` is
  populated by the contract

#### Scenario: Extra columns are ignored and the input is not mutated

- **WHEN** `resolve_params` is called with a full `cyl_scans_extended` row carrying many
  non-load-bearing columns (e.g. `species_genus`, `scan_id`, timestamps)
- **THEN** only `species`, `mode`, and `age` appear in the resolved `values`, and the
  passed-in `metadata` dict is unchanged

#### Scenario: A blank species value is treated as not provided

- **WHEN** `species_name` is present but blank (`""`, `"   "`, `None`, or a `NaN`) with no
  `species` override
- **THEN** `resolve_params` raises a `ValueError` naming `species` (the blank is treated as
  missing, not resolved to an empty species), and does not crash

### Requirement: Species Name Normalization

The system SHALL normalize the Bloom `species_name` to the `ModelCard` species vocabulary
via `_normalize_species`, which strips and lowercases the name and then applies an alias
map keyed by the **lowercased** name. The alias map is the single extension point for any
Bloom name that does not lowercase cleanly to the card string (e.g. a Latin binomial);
names not in the map fall through as their stripped, lowercased form (**lowercase
passthrough fallback**). Unknown species SHALL pass through rather than being rejected or
dropped — the registry/`ModelCard`s are the single authority on which species have models,
so an unmodelled species degrades to a `choose_models` zero-match (skip), not a resolver
error. The resolver SHALL NOT maintain a whitelist or hard-fail on species.

#### Scenario: A seeded species normalizes to the card vocabulary

- **WHEN** `species_name` is `"Pennycress"`
- **THEN** the resolved `species` is `"pennycress"`

#### Scenario: Surrounding whitespace and case are normalized

- **WHEN** `species_name` is `"  Rice  "`
- **THEN** the resolved `species` is `"rice"`

#### Scenario: An unknown species passes through lowercased

- **WHEN** `species_name` is `"Sorghum"` (a real Bloom species with no seeded model)
- **THEN** the resolved `species` is `"sorghum"` (passthrough), and no error is raised

### Requirement: Imaging Mode Resolution Seam

The system SHALL derive `mode` through a single `_mode_for_scan(metadata)` function that
returns `"cylinder"` for the current cylinder stage-in path (the cylinder pipeline yields
cylinder scans only). This function SHALL be the one place mode is decided, so future
GraviScan/multiscanner modes slot in here without changing `resolve_params`'s body, its
callers, or its output shape. The mode strings it returns MUST equal the exact seeded
`ModelCard` mode vocabulary. The scanner→mode lookup table for deferred modalities is
explicitly out of scope for this change.

#### Scenario: A cylinder scan resolves mode "cylinder"

- **WHEN** `resolve_params` resolves a `cyl_scans_extended` row
- **THEN** the resolved `mode` is `"cylinder"`

### Requirement: Age Resolution In Days

The system SHALL resolve `age` from `plant_age_days` as an integer number of days via
`_coerce_age`. It SHALL accept both an integer and an int-coercible whole-number string
(Bloom metadata read from `scans.csv` may arrive as a string, e.g. `"14"`), canonicalizing
both to the same `int` so the resolved `age` — and therefore `param_hash` — does not depend
on the incoming representation. A present, non-blank, but non-whole or non-coercible value
(e.g. `14.5`, `"14.5"`, `"abc"`, or a bool) SHALL raise a `ValueError` naming `age`, rather
than silently truncating — mirroring `choose_models`'s age handling. A resolved `age` of
`0` SHALL be valid (validation checks key presence, not truthiness).

#### Scenario: An integer age passes through as days

- **WHEN** the row's `plant_age_days` is `14`
- **THEN** the resolved `age` is the integer `14`

#### Scenario: An int-coercible string age is coerced to the same int

- **WHEN** the row's `plant_age_days` is the string `"14"`
- **THEN** the resolved `age` is the integer `14`, and the resulting `param_hash` is
  identical to the `plant_age_days=14` case (representation-independent)

#### Scenario: A non-whole or non-coercible age raises naming age

- **WHEN** the row's `plant_age_days` is `14.5`, `"14.5"`, or `"abc"`
- **THEN** `resolve_params` raises a `ValueError` whose message names `age`

#### Scenario: An age of zero is valid

- **WHEN** the row's `plant_age_days` is `0`
- **THEN** the resolved `age` is `0` and resolution succeeds (0 is not treated as missing)

#### Scenario: A blank age is treated as not provided

- **WHEN** the row's `plant_age_days` is blank (`""`, whitespace, `None`, or `NaN`) with no
  `age` override
- **THEN** `resolve_params` raises a `ValueError` naming `age` as **missing** (treated as not
  provided — the same as a blank `species_name`), rather than a "not a whole number" error

### Requirement: Override Merge And Strict Post-Override Validation

The optional `overrides` argument SHALL be a param-space dict merged last so a supplied
field replaces the derived one (**override wins per field**). Override keys SHALL be
restricted to the resolvable params `{species, mode, age}`; an unrecognized override key
SHALL raise a `ValueError` naming the offending key. Override **values** SHALL be
canonicalized by the same per-field rules as derived values (`species` normalized via
`_normalize_species`, `mode` normalized via `_normalize_mode`, `age` coerced via
`_coerce_age`), so that a logically identical run produces an identical `param_hash`
regardless of whether a value arrived derived or as an override (e.g. an `age` override of
`"14"` and a derived `14` hash identically; a `mode` override of `"Cylinder"` and a derived
`"cylinder"` hash identically). A blank override value is treated as not provided (the field
is dropped, then named by validation). After merging and canonicalizing, `resolve_params`
SHALL require that `species`, `mode`, and `age` are all present (by key) and SHALL raise a
clear `ValueError` naming every missing param when any is absent. It SHALL NOT return a
half-resolved `ResolvedParams`.

#### Scenario: Override wins per field

- **WHEN** `resolve_params(row, overrides={"mode": "graviscan", "species": "canola"})` is
  called
- **THEN** `mode` is `"graviscan"` and `species` is `"canola"` (the overrides), while any
  field not present in `overrides` (e.g. `age`) keeps its derived value

#### Scenario: Override values are canonicalized like derived values

- **WHEN** `resolve_params(row, overrides={"species": "Rice", "age": "14"})` is called
- **THEN** the resolved `species` is `"rice"` and `age` is the integer `14` (the override
  values are normalized/coerced, not stored raw), so the `param_hash` matches the
  equivalent derived run

#### Scenario: A mode override is canonicalized

- **WHEN** `resolve_params(row, overrides={"mode": "  Cylinder "})` is called
- **THEN** the resolved `mode` is `"cylinder"` (stripped/lowercased like a derived mode), so
  the `param_hash` matches the equivalent derived run and selection is not broken by casing

#### Scenario: An unknown override key is rejected

- **WHEN** `resolve_params(row, overrides={"specis": "rice"})` is called (a typo'd key)
- **THEN** it raises a `ValueError` whose message names the unrecognized key

#### Scenario: Missing required fields raise naming each missing param

- **WHEN** `resolve_params` is called with a row missing both `species_name` and
  `plant_age_days` and no compensating overrides
- **THEN** it raises a `ValueError` whose message names both missing params (`species` and
  `age`)

#### Scenario: A missing derived field can be supplied by an override

- **WHEN** a row is missing `species_name` but the call passes
  `overrides={"species": "rice"}`
- **THEN** resolution succeeds with `species="rice"` (the override satisfies the
  post-override validation)

### Requirement: Public API Export

The system SHALL export `resolve_params` from the package's public API — importable as
`from sleap_roots_predict import resolve_params` and listed in `__all__` — so it is
callable alongside `choose_models`.

#### Scenario: resolve_params is importable and in __all__

- **WHEN** a caller runs `from sleap_roots_predict import resolve_params`
- **THEN** the import succeeds and `"resolve_params"` is present in
  `sleap_roots_predict.__all__`

### Requirement: Interoperability With Model Selection

The `ResolvedParams` produced by `resolve_params` SHALL be a valid input to
`choose_models`, so a Bloom scan-metadata row drives model selection end-to-end
(metadata → params → model). The detailed selection semantics (matching, skipping,
ambiguity) remain owned by the `model-management` capability; this requirement asserts only
that the produced params are consumable by that layer and yield the expected selection for a
row whose `species`/`mode`/`age` match seeded cards.

#### Scenario: A resolved Bloom row selects the expected model(s)

- **WHEN** `choose_models(resolve_params(row), cards)` is called for a Bloom row whose
  normalized `species`/`mode` and coerced `age` match a small real `ModelCard` list
- **THEN** it returns the expected `ModelRef` per matching root type (the metadata → params
  → model round-trip), without raising a missing-param error

