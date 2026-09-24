# model-management Specification

## Purpose
TBD - created by archiving change add-warm-model-worker. Update Purpose after archive.
## Requirements
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

#### Scenario: Two cards matching through different selectors are ambiguous

- **WHEN** two different cards of one root type each match the params through one of their
  selectors
- **THEN** `choose_models` raises the ambiguity error (the raise is not relaxed for selector-shaped
  cards)

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

### Requirement: Model Card Source Protocol And Local Source

The system SHALL define a `ModelCardSource` protocol with two methods: `list_cards() -> list[ModelCard]`
(the catalog used for matching) and `materialize(ref: ModelRef) -> Path` (a local model directory
loadable by `make_predictor`). Because `ModelCard` carries no filesystem path, a source SHALL map a
`ModelRef` back to a local directory by its identity `(registry_id, version)`. It SHALL provide a
`LocalCardSource` implementation constructed from `(ModelCard, Path)` pairs that holds a
`(registry_id, version) -> Path` mapping, so `list_cards()` returns the cards and `materialize(ref)`
returns the mapped directory without any network access — making the full selection→load path
exercisable offline.

#### Scenario: LocalCardSource lists cards offline

- **WHEN** `list_cards()` is called on a `LocalCardSource` built from `(card, path)` pairs
- **THEN** it returns the corresponding `ModelCard`s without any network access

#### Scenario: materialize resolves a ModelRef to its mapped directory

- **WHEN** `materialize(ref)` is called on a `LocalCardSource` for a card's `ModelRef`
- **THEN** it returns the on-disk directory mapped to that `ref`'s `(registry_id, version)`, which
  `make_predictor` can load

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
logged warning that names it and includes the underlying error, no exception SHALL be raised for
that artifact alone, and the remaining conforming artifacts SHALL still be returned (one malformed
artifact SHALL NOT abort the listing), subject to the all-invalid rule below. This isolation SHALL
be scoped to per-artifact card construction only; genuine failures (missing credentials,
registry/network errors) SHALL still propagate fail-loud rather than being swallowed per artifact.
When at least one artifact carries the configured alias and **none** of them can be read into a
`ModelCard`, `list_cards()` SHALL raise a `ValueError` (or subclass) naming the registry, the alias
and the number of alias-carrying artifacts that could not be read, rather than returning an empty
catalog: a registry whose every production card is unreadable (for example, flat-shaped cards read
by a consumer pinned to a selector-shaped contract) is a deployment fault, not a set of
per-artifact defects. The error SHALL distinguish validation failures, for which it points at the
registry's card shape, from any other per-artifact error, which it names and chains as the cause,
so a transient registry fault is never reported as a shape mismatch. Building the error SHALL NOT
itself be able to raise. A registry in which **no** artifact carries the configured alias SHALL still
return an empty list without raising (the batch runner separately treats an empty catalog as a
staging error — see the `predict-container` capability). This rule guards only a catalog with zero
readable cards; it does not detect a catalog that is readable but incomplete.

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
- **THEN** `list_cards()` raises a `ValueError` naming the registry, the alias and the number of
  alias-carrying artifacts that could not be read, points at the registry's card shape, and does
  not return an empty card list

#### Scenario: A non-validation failure is named, not reported as a shape mismatch

- **WHEN** every alias-carrying artifact fails to build a card for a reason other than validation
  (e.g. a registry error while reading its digest)
- **THEN** `list_cards()` raises the same error type, naming the underlying error type and message,
  chaining it as the cause, and not suggesting a re-seed

#### Scenario: A registry with no artifact carrying the alias is empty, not an error

- **WHEN** no artifact in the registry carries the configured alias
- **THEN** `list_cards()` returns an empty list and does not raise

#### Scenario: A materialized artifact is cached and reused

- **WHEN** `materialize(ref)` is called twice for the same pinned version
- **THEN** the artifact is downloaded at most once and the cached local directory is reused

#### Scenario: Missing credentials raise a clear error

- **WHEN** `WandbRegistrySource` is used with no `WANDB_API_KEY` set
- **THEN** it raises an error whose message names `WANDB_API_KEY`, before any network call, and does
  not return an empty card list

### Requirement: Warm Model Residency

The system SHALL provide a `WarmModelWorker(source=None, ...)` that keeps `Predictor`s resident across
scans. When `source` is omitted (or `None`), the worker SHALL default to a `WandbRegistrySource`
reading the live production registry — the default source is the registry, not an offline/stub source,
and there SHALL be no silent `LocalCardSource` fallback. Constructing the worker SHALL perform no
network access; a missing `WANDB_API_KEY` SHALL surface fail-loud on the first call that lists cards —
`load_catalog()`, or the first `resolve()` / `get_predictors()` when `load_catalog()` has not been
called — not at construction. `load_catalog()` SHALL list the source's cards, cache them and
return the cached list, and SHALL be idempotent: repeat calls, and every later `resolve()` / `get_predictors()`, SHALL reuse the
cached cards without listing again. `resolve(params)` SHALL return `dict[RootType, ModelRef]`
without loading weights. `get_predictors(params)` SHALL resolve, `materialize` each `ModelRef`,
build a `Predictor` via `make_predictor`, and cache it keyed by `(registry_id, version)` so a model
is fetched at most once and loaded at most once and reused across scans. Cards SHALL be loaded at
most once per worker, lazily unless `load_catalog()` is called first. If any resolved root type
cannot be materialized or loaded, `get_predictors` SHALL raise an error identifying the root type
and `registry_id:version` and SHALL NOT return partial results (fail-loud). A thin
`predict(params, video, save_dir=None)` convenience SHALL compose `get_predictors` with
`predict_on_video` and return per-root-type results in memory. When `save_dir` is given, it SHALL
write one raw `.slp` per root type via `predict_on_video`'s `save_path` (e.g. `save_dir/<root_type>.slp`)
and SHALL NOT write a `predictions.csv` manifest or apply the
`{scan}.model{id}.root{type}.slp` naming — that naming and manifest are deferred to the
output-contract slice.

#### Scenario: Default source is the live registry

- **WHEN** a `WarmModelWorker` is constructed with no `source` argument
- **THEN** its source is a `WandbRegistrySource` (the live production registry) and construction performs
  no network access

#### Scenario: Missing credentials fail loud on first use

- **WHEN** a `WarmModelWorker` constructed with the default source is used with no `WANDB_API_KEY` set
- **THEN** the first call that lists cards (`load_catalog()`, `resolve()` or `get_predictors()`)
  raises an error naming `WANDB_API_KEY` (construction itself does not raise), and there is no
  offline fallback

#### Scenario: load_catalog lists once and resolve reuses it

- **WHEN** `load_catalog()` is called, then `load_catalog()` again, then `resolve(params)` twice
- **THEN** the source's `list_cards()` has been called exactly once

#### Scenario: resolve does not load weights

- **WHEN** `resolve(params)` is called
- **THEN** it returns the selected `ModelRef`s per root type and the worker's predictor cache remains
  empty (no model materialized or loaded)

#### Scenario: A model is loaded once and reused (warm)

- **WHEN** `get_predictors` is called twice for params that resolve to the same model version
- **THEN** the second call returns the same cached `Predictor` instance without re-fetching or reloading

#### Scenario: Different params sharing a model version hit the cache

- **WHEN** two different param sets resolve a root type to the same `(registry_id, version)`
- **THEN** both use the same resident `Predictor` (keyed by model identity, not by scan or root type)

#### Scenario: Unmaterializable root type fails loud

- **WHEN** a resolved root type's model cannot be materialized or loaded
- **THEN** `get_predictors` raises an error naming the root type and `registry_id:version`, and returns no predictors

#### Scenario: predict returns labels per root type

- **WHEN** `predict(params, video)` is called with no `save_dir`
- **THEN** it returns a `dict[RootType, sio.Labels]` with real predicted instances for each resolved root type

#### Scenario: predict with save_dir writes raw per-root .slp only

- **WHEN** `predict(params, video, save_dir=…)` is called
- **THEN** it writes one reloadable `.slp` per root type under `save_dir` and writes no
  `predictions.csv` and applies no `{scan}.model{id}.root{type}.slp` naming (deferred)

### Requirement: Effective Inference Config Capture

The `WarmModelWorker` SHALL expose the effective inference config it used via `inference_config()`,
including the resolved `device`, `peak_threshold`, and `batch_size`. It SHALL distinguish the
output-defining subset (containing `peak_threshold`) — which downstream layers fold into the
reproducibility hash — from the diagnostic-only hardware/throughput knobs (`device`, `batch_size`),
which SHALL NOT be part of the output-defining subset.

#### Scenario: Effective config reports the values used

- **WHEN** `inference_config()` is called on a worker
- **THEN** it returns the resolved `device`, `peak_threshold`, and `batch_size` actually used to build predictors

#### Scenario: Output-defining subset excludes hardware knobs

- **WHEN** the worker exposes the output-defining subset of its inference config
- **THEN** the subset contains `peak_threshold` and excludes `device` and `batch_size`

### Requirement: Real Non-Mocked Test Coverage

Selection and warm-worker tests SHALL run offline with no mocking of the sleap-nn, sleap-io, or
wandb boundaries: the matcher SHALL be tested with small real `ModelCard` lists, and the warm worker
SHALL be tested via a `LocalCardSource` over the vendored `tests/assets/models/` fixtures — covering
both the native and the legacy SLEAP model in a single resolution (e.g. one root type per format) —
asserting that a model is loaded once and the same `Predictor` is reused across calls, that a
zero-match root type is skipped end-to-end, and that real `sio.Labels` are produced. Tests exercising
`WandbRegistrySource` SHALL be marked `@pytest.mark.wandb` and gated with a collection-time
`skipif(not WANDB_API_KEY)` (mirroring the acceptance-test pattern), deselected by default and in CI,
so they skip cleanly without credentials and never hit the network on shared runners.

#### Scenario: Offline warm-worker test uses real predictors, no mocks

- **WHEN** the default test suite runs the warm-worker tests on CPU with a `LocalCardSource` over the
  vendored native and legacy models
- **THEN** real `Predictor`s are built and reused and real `sio.Labels` are produced, with no mocks

#### Scenario: Wandb tests skip without credentials

- **WHEN** the `@pytest.mark.wandb` tests run with no `WANDB_API_KEY` set
- **THEN** they skip at collection time rather than failing or hitting the network

