## MODIFIED Requirements

### Requirement: Model Selection From Scan Params

The system SHALL provide a pure function `choose_models(params, cards, overrides=None)` that maps
resolved scan params (`species`, `mode`, `age` read from a `ResolvedParams`) and a list of
`ModelCard`s to a `dict[RootType, ModelRef]` — at most one model per root type. For each root type
it SHALL apply, in order: an explicit override when provided; otherwise select the card(s) of that
root type that **match** the params. A card SHALL match if and only if **some single one** of its
`selectors` has `species ==` the scan species, `mode ==` the scan mode, and
`age_min <= age <= age_max` (inclusive) for **that same selector's** window. The system SHALL NOT
match against a card-level age window (any minimum or maximum taken across a card's selectors)
and SHALL NOT match the cross product of different selectors' species, modes and windows. A card
SHALL count once however many of its selectors match. Exactly one matching card SHALL be
selected; zero matches SHALL skip that root type; more than one matching card SHALL raise an
error identifying the ambiguity. The selected `ModelCard` SHALL be converted to a `ModelRef` via
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
  one card with a selector matching `species`, `mode`, and an age within that selector's
  `[age_min, age_max]`
- **THEN** it returns a `dict[RootType, ModelRef]` with one `ModelRef` per matched root type, each
  carrying the card's concrete `version`/`weights_checksum` and the runtime `sleap_nn_version`

#### Scenario: Age window boundaries are inclusive

- **WHEN** the scan `age` equals a selector's `age_min` or its `age_max`, and that selector's
  species and mode match
- **THEN** the card carrying that selector matches (the window is inclusive at both ends)

#### Scenario: Age outside every species/mode-matching selector's window does not match

- **WHEN** the scan `age` is outside the `[age_min, age_max]` window of every selector whose species
  and mode match the scan
- **THEN** that card is not selected for its root type

#### Scenario: A card matches through any one of its selectors

- **WHEN** a card carries selectors (canola, cylinder, 2–13) and (pennycress, cylinder, 2–14), and
  the scan is pennycress, cylinder, age 14
- **THEN** the card matches, through its pennycress selector

#### Scenario: Age is compared against the matching selector's window only

- **WHEN** a card carries selectors (canola, cylinder, 2–13) and (pennycress, cylinder, 2–14), and
  the scan is canola, cylinder, age 14
- **THEN** the card does not match, even though another of its selectors' windows includes 14

#### Scenario: Disjoint windows of one species are not merged

- **WHEN** a card carries selectors (canola, cylinder, 2–5) and (canola, cylinder, 10–13), and the
  scan is canola, cylinder, age 7
- **THEN** the card does not match, because no single selector's window contains 7

#### Scenario: Selectors are never combined across a card

- **WHEN** a card carries selectors (canola, cylinder, 2–13) and (arabidopsis, multiplant
  cylinder, 2–14), and the scan is canola, multiplant cylinder, age 5 (inside both windows)
- **THEN** the card does not match, because no single selector matches both species and mode

#### Scenario: Overlapping selectors on one card are one match

- **WHEN** exactly one card of a root type matches, through two of its selectors at once
- **THEN** that card is selected and no ambiguity error is raised

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

### Requirement: Wandb Registry Source With Version Pinning

The system SHALL provide a `WandbRegistrySource` implementing `ModelCardSource` whose `list_cards()`
reads the production registry's artifacts' selection metadata into `ModelCard`s and, when an artifact
is referenced via a moving alias (e.g. `production`), resolves the alias to a concrete artifact
version and populates each card's `version` and `weights_checksum` with that concrete pin (never the
alias). The source SHALL NOT construct `ModelRef`s — `ModelRef` construction and runtime
`sleap_nn_version` stamping are performed by `choose_models` via `ModelCard.to_model_ref`.
`materialize(ref)` SHALL download the pinned artifact version to a local cache directory and reuse it
on repeat calls. All network access SHALL be confined to this class. Authentication SHALL use
`WANDB_API_KEY`; the entity, registry, and alias SHALL be configurable via `SRP_WANDB_ENTITY`,
`SRP_WANDB_MODEL_REGISTRY`, and `SRP_WANDB_MODEL_ALIAS`; the cache SHALL be configurable via
`SRP_MODEL_CACHE_DIR`. When no registry is configured (neither a constructor argument nor
`SRP_WANDB_MODEL_REGISTRY`), the source SHALL default the registry to `sleap-roots-models` (the live
production registry), so a source constructed with only `WANDB_API_KEY` set reads production;
likewise, when no alias is configured, the alias SHALL default to `production`. The legacy environment
names `SRP_WANDB_REGISTRY` and `SRP_WANDB_ALIAS` SHALL NOT be read. When `WANDB_API_KEY` is not set,
the source SHALL raise a clear error naming the missing variable before any network call, rather than
returning an empty result. `list_cards()` SHALL isolate per-artifact failures: when a single
artifact's metadata cannot be validated into a `ModelCard`, that artifact SHALL be skipped with a
logged warning that names it and includes the underlying error, no exception SHALL be raised, and the
remaining conforming artifacts SHALL still be returned (one malformed artifact SHALL NOT abort the
listing). This isolation SHALL be scoped to per-artifact card construction only; genuine failures
(missing credentials, registry/network errors) SHALL still propagate fail-loud rather than being
swallowed per artifact. When at least one artifact carries the configured alias and **none** of
them validates into a `ModelCard`, `list_cards()` SHALL raise an error naming the registry, the
alias and the number of artifacts skipped, rather than returning an empty catalog: a registry whose
every production card is unreadable (for example, flat-shaped cards read by a consumer pinned to a
selector-shaped contract) is a deployment fault, not a set of per-artifact defects.

#### Scenario: Registry defaults to the live production registry

- **WHEN** a `WandbRegistrySource` is constructed with no registry argument and `SRP_WANDB_MODEL_REGISTRY`
  unset
- **THEN** its configured registry is `sleap-roots-models`, so `list_cards()` / `materialize()` read the
  live production registry with only `WANDB_API_KEY` set

#### Scenario: Renamed registry env var configures the registry and legacy name is ignored

- **WHEN** `SRP_WANDB_MODEL_REGISTRY` is set (and no constructor registry is passed)
- **THEN** the source uses that registry; and a value set only in the legacy `SRP_WANDB_REGISTRY` is not
  read (the `sleap-roots-models` default applies instead)

#### Scenario: Renamed alias env var configures the alias and legacy name is ignored

- **WHEN** `SRP_WANDB_MODEL_ALIAS` is set (and no constructor alias is passed)
- **THEN** the source resolves cards against that alias; and a value set only in the legacy
  `SRP_WANDB_ALIAS` is not read (the `production` default alias applies instead)

#### Scenario: A genuine credential/network error is not swallowed per artifact

- **WHEN** `list_cards()` fails for a non-per-artifact reason (e.g. missing `WANDB_API_KEY`, or a
  registry/network error while traversing artifacts)
- **THEN** the error propagates fail-loud (it is not caught by the per-artifact skip-and-warn), so a
  degraded or empty catalog is never silently returned

#### Scenario: Alias is pinned to a concrete version in the card

- **WHEN** `list_cards()` returns a card for an artifact referenced by a moving alias
- **THEN** the card's `version` is the concrete artifact version and its `weights_checksum` is
  populated (not the alias), so the `ModelRef` later built by `choose_models` carries the concrete pin

#### Scenario: A non-conforming artifact is skipped with a warning

- **WHEN** `list_cards()` encounters an artifact carrying the configured alias whose metadata cannot be
  validated into a `ModelCard` (e.g. missing a required selection field), and at least one other
  artifact carrying the alias does validate
- **THEN** that artifact is skipped, a warning naming it is logged, no exception is raised, and the
  remaining conforming cards are still returned

#### Scenario: A legacy flat-shaped card is skipped alongside readable ones

- **WHEN** artifacts carrying the configured alias include flat-shaped cards (top-level `species`/
  `mode`/`age_min`/`age_max`, no `selectors`) and at least one selector-shaped card
- **THEN** each flat-shaped card is skipped with a warning naming it, and the selector-shaped cards
  are returned

#### Scenario: A registry with no readable production card fails loud

- **WHEN** one or more artifacts carry the configured alias and none of them validates into a
  `ModelCard`
- **THEN** `list_cards()` raises an error naming the registry, the alias and the skipped count, and
  does not return an empty card list

#### Scenario: A materialized artifact is cached and reused

- **WHEN** `materialize(ref)` is called twice for the same pinned version
- **THEN** the artifact is downloaded at most once and the cached local directory is reused

#### Scenario: Missing credentials raise a clear error

- **WHEN** `WandbRegistrySource` is used with no `WANDB_API_KEY` set
- **THEN** it raises an error whose message names `WANDB_API_KEY`, before any network call, and does
  not return an empty card list
