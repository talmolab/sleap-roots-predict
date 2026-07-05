## ADDED Requirements

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
`WANDB_API_KEY`; the entity and registry SHALL be configurable via `SRP_WANDB_ENTITY` and
`SRP_WANDB_REGISTRY`; the cache SHALL be configurable via `SRP_MODEL_CACHE_DIR`. When `WANDB_API_KEY`
is not set, the source SHALL raise a clear error naming the missing variable before any network call,
rather than returning an empty result.

#### Scenario: Alias is pinned to a concrete version in the card

- **WHEN** `list_cards()` returns a card for an artifact referenced by a moving alias
- **THEN** the card's `version` is the concrete artifact version and its `weights_checksum` is
  populated (not the alias), so the `ModelRef` later built by `choose_models` carries the concrete pin

#### Scenario: A materialized artifact is cached and reused

- **WHEN** `materialize(ref)` is called twice for the same pinned version
- **THEN** the artifact is downloaded at most once and the cached local directory is reused

#### Scenario: Missing credentials raise a clear error

- **WHEN** `WandbRegistrySource` is used with no `WANDB_API_KEY` set
- **THEN** it raises an error whose message names `WANDB_API_KEY`, before any network call, and does
  not return an empty card list

### Requirement: Warm Model Residency

The system SHALL provide a `WarmModelWorker(source, ...)` that keeps `Predictor`s resident across
scans. `resolve(params)` SHALL return `dict[RootType, ModelRef]` without loading weights.
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
