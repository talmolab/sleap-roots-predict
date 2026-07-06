## MODIFIED Requirements

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
swallowed per artifact.

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
  validated into a `ModelCard` (e.g. missing a required selection field)
- **THEN** that artifact is skipped, a warning naming it is logged, no exception is raised, and the
  remaining conforming cards are still returned

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
network access; a missing `WANDB_API_KEY` SHALL surface fail-loud on the first
`resolve()` / `get_predictors()` call (which lists cards), not at construction.
`resolve(params)` SHALL return `dict[RootType, ModelRef]` without loading weights.
`get_predictors(params)` SHALL resolve, `materialize` each `ModelRef`, build a `Predictor` via
`make_predictor`, and cache it keyed by `(registry_id, version)` so a model is fetched at most once
and loaded at most once and reused across scans. Cards SHALL be loaded lazily once. If any resolved
root type cannot be materialized or loaded, `get_predictors` SHALL raise an error identifying the
root type and `registry_id:version` and SHALL NOT return partial results (fail-loud). A thin
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
- **THEN** the first `resolve()` / `get_predictors()` raises an error naming `WANDB_API_KEY`
  (construction itself does not raise), and there is no offline fallback

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
