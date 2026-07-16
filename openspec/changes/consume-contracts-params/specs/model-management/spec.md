## MODIFIED Requirements

### Requirement: Model Selection From Scan Params

The system SHALL provide a pure function `choose_models(params, cards, overrides=None)` that maps
resolved scan params (`species`, `mode`, `age` read from a `ResolvedParams`) and a list of
`ModelCard`s to a `dict[RootType, ModelRef]` — at most one model per root type. For each root type
it SHALL apply, in order: an explicit override when provided; otherwise select the card(s) matching
`species ==`, `mode ==`, and `age_min <= age <= age_max` (inclusive). Exactly one matching card
SHALL be selected; zero matches SHALL skip that root type; more than one match SHALL raise an error
identifying the ambiguity. The selected `ModelCard` SHALL be converted to a `ModelRef` via
`ModelCard.to_model_ref`, which copies the card's already-concrete `registry_id`, `version`,
`root_type`, and `weights_checksum` and stamps the **runtime** sleap-nn version. `choose_models`
SHALL rely on cards already carrying a concrete `version`/`weights_checksum` (it does not resolve
aliases) and SHALL perform no network access and no per-call filesystem I/O (the runtime sleap-nn
version is resolved once at import). It SHALL raise a clear error when a required param (`species`,
`mode`, or `age`) is absent.

The accepted `ResolvedParams` SHALL be accepted regardless of which module produced it — in
particular, a `ResolvedParams` built by `sleap_roots_contracts.resolve_params` (the Bloom scan
metadata → params oracle promoted into `sleap-roots-contracts`, consumed by predict rather than
implemented by it) SHALL drive selection identically to a hand-built one, so a Bloom scan
metadata row selects production models end-to-end (metadata → params → model), across the
predict/contracts repo boundary.

#### Scenario: Exactly one match selects a model per root type

- **WHEN** `choose_models` is called with params and cards where each present root type has exactly
  one card matching `species`, `mode`, and an age within `[age_min, age_max]`
- **THEN** it returns a `dict[RootType, ModelRef]` with one `ModelRef` per matched root type, each
  carrying the card's concrete `version`/`weights_checksum` and the runtime `sleap_nn_version`

#### Scenario: Age window boundaries are inclusive

- **WHEN** the scan `age` equals a card's `age_min` or its `age_max`
- **THEN** that card matches (the window is inclusive at both ends)

#### Scenario: Age outside a card's window does not match

- **WHEN** the scan `age` is outside a card's `[age_min, age_max]` window
- **THEN** that card is not selected for its root type

#### Scenario: Zero matches skips the root type

- **WHEN** no card matches for a given root type (e.g. a species with no crown model)
- **THEN** that root type is absent from the returned mapping (skipped, not an error); if no root
  type matches, the returned mapping is empty

#### Scenario: Ambiguous match raises

- **WHEN** more than one card matches the same root type for the given params
- **THEN** `choose_models` raises an error identifying the ambiguous root type

#### Scenario: Explicit override bypasses matching

- **WHEN** an explicit override `ModelRef` is provided for a root type
- **THEN** that override is used for the root type and the card-matching filter is not applied to it,
  even when no card would match that root type

#### Scenario: Missing required param raises

- **WHEN** `choose_models` is called with params missing `species`, `mode`, or `age`
- **THEN** it raises a clear error naming the missing param

#### Scenario: A resolved Bloom row selects the expected model(s)

- **WHEN** `choose_models(resolve_params(row), cards)` is called for a Bloom row whose
  normalized `species`/`mode` and coerced `age` (resolved by
  `sleap_roots_contracts.resolve_params`) match a small real `ModelCard` list
- **THEN** it returns the expected `ModelRef` per matching root type (the metadata → params →
  model round trip), without raising a missing-param error
