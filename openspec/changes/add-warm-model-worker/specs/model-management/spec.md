## ADDED Requirements

### Requirement: Model Selection From Scan Params

The system SHALL provide a pure function `choose_models(params, cards, overrides=None)` that maps
resolved scan params (`species`, `mode`, `age` read from a `ResolvedParams`) and a list of
`ModelCard`s to a `dict[RootType, ModelRef]` — at most one model per root type. For each root type
it SHALL apply, in order: an explicit override when provided; otherwise select the card(s) matching
`species ==`, `mode ==`, and `age_min <= age <= age_max`. Exactly one matching card SHALL be
selected (converted to a `ModelRef`); zero matches SHALL skip that root type; more than one match
SHALL raise an error identifying the ambiguity. The function SHALL be pure (no I/O) and SHALL raise
a clear error when a required param (`species`, `mode`, or `age`) is absent.

#### Scenario: Exactly one match selects a model per root type

- **WHEN** `choose_models` is called with params and cards where each present root type has exactly
  one card matching `species`, `mode`, and an age within `[age_min, age_max]`
- **THEN** it returns a `dict[RootType, ModelRef]` with one `ModelRef` per matched root type

#### Scenario: Age outside a card's window does not match

- **WHEN** the scan `age` is outside a card's `[age_min, age_max]` window
- **THEN** that card is not selected for its root type

#### Scenario: Zero matches skips the root type

- **WHEN** no card matches for a given root type (e.g. a species with no crown model)
- **THEN** that root type is absent from the returned mapping (skipped, not an error)

#### Scenario: Ambiguous match raises

- **WHEN** more than one card matches the same root type for the given params
- **THEN** `choose_models` raises an error identifying the ambiguous root type

#### Scenario: Explicit override bypasses matching

- **WHEN** an explicit override `ModelRef` is provided for a root type
- **THEN** that override is used for the root type and the card-matching filter is not applied to it

#### Scenario: Missing required param raises

- **WHEN** `choose_models` is called with params missing `species`, `mode`, or `age`
- **THEN** it raises a clear error naming the missing param

### Requirement: Model Card Source Protocol And Local Source

The system SHALL define a `ModelCardSource` protocol with two methods: `list_cards() -> list[ModelCard]`
(the catalog used for matching) and `materialize(ref: ModelRef) -> Path` (a local model directory
loadable by `make_predictor`). It SHALL provide a `LocalCardSource` implementation whose cards
reference on-disk model directories and whose `materialize` returns those directories without any
network access, so the full selection→load path is exercisable offline.

#### Scenario: LocalCardSource lists cards offline

- **WHEN** `list_cards()` is called on a `LocalCardSource` built over on-disk model directories
- **THEN** it returns the corresponding `ModelCard`s without any network access

#### Scenario: materialize returns a loadable model directory

- **WHEN** `materialize(ref)` is called on a `LocalCardSource` for a card's `ModelRef`
- **THEN** it returns a filesystem path to a model directory that `make_predictor` can load

### Requirement: Wandb Registry Source With Version Pinning

The system SHALL provide a `WandbRegistrySource` implementing `ModelCardSource` that reads the
production registry's artifacts' selection metadata into `ModelCard`s and downloads a `ModelRef`'s
pinned artifact version to a local cache directory. All network access SHALL be confined to this
class. Authentication SHALL use `WANDB_API_KEY`; the entity and registry SHALL be configurable via
`SRP_WANDB_ENTITY` and `SRP_WANDB_REGISTRY`; the download cache SHALL be configurable via
`SRP_MODEL_CACHE_DIR`. When resolving a moving alias (e.g. `production`), the source SHALL record
the concrete version and `weights_checksum` into the produced `ModelRef` (pinning for
reproducibility) and stamp the `ModelRef`'s `sleap_nn_version` with the runtime sleap-nn version.

#### Scenario: Alias resolves to a concrete pinned version

- **WHEN** `WandbRegistrySource` produces a `ModelRef` for a card selected via a moving alias
- **THEN** the `ModelRef` records the concrete artifact version and its `weights_checksum` (not the
  alias), and its `sleap_nn_version` is the runtime sleap-nn version

#### Scenario: A materialized artifact is cached and reused

- **WHEN** `materialize(ref)` is called twice for the same pinned version
- **THEN** the artifact is downloaded at most once and the cached local directory is reused

#### Scenario: Missing credentials are surfaced

- **WHEN** `WandbRegistrySource` is used to access the registry with no `WANDB_API_KEY` configured
- **THEN** the failure is surfaced as a clear error rather than a silent empty result

### Requirement: Warm Model Residency

The system SHALL provide a `WarmModelWorker(source, ...)` that keeps `Predictor`s resident across
scans. `resolve(params)` SHALL return `dict[RootType, ModelRef]` without loading weights.
`get_predictors(params)` SHALL resolve, `materialize` each `ModelRef`, build a `Predictor` via
`make_predictor`, and cache it keyed by `(registry_id, version)` so a model is fetched at most once
and loaded at most once and reused across scans. Cards SHALL be loaded lazily once. If any resolved
root type cannot be materialized or loaded, `get_predictors` SHALL raise an error identifying the
root type and `registry_id:version` and SHALL NOT return partial results (fail-loud). A thin
`predict(params, video, save_dir=None)` convenience SHALL compose `get_predictors` with
`predict_on_video` and return in-memory results per root type.

#### Scenario: resolve does not load weights

- **WHEN** `resolve(params)` is called
- **THEN** it returns the selected `ModelRef`s per root type without materializing or loading any model

#### Scenario: A model is loaded once and reused (warm)

- **WHEN** `get_predictors` is called twice for params that resolve to the same model version
- **THEN** the second call reuses the same cached `Predictor` instance without re-fetching or reloading

#### Scenario: Different params sharing a model version hit the cache

- **WHEN** two different param sets resolve a root type to the same `(registry_id, version)`
- **THEN** both use the same resident `Predictor` (keyed by model identity, not by scan or root type)

#### Scenario: Unmaterializable root type fails loud

- **WHEN** a resolved root type's model cannot be materialized or loaded
- **THEN** `get_predictors` raises an error naming the root type and `registry_id:version`, and returns no predictors

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
SHALL be tested via a `LocalCardSource` over the vendored `tests/assets/models/` fixtures (native
and legacy), asserting that a model is loaded once and the same `Predictor` is reused across calls
and that real `sio.Labels` are produced. Tests exercising `WandbRegistrySource` SHALL be marked
`@pytest.mark.wandb`, deselected by default (like `gpu`/`acceptance`), and SHALL skip cleanly when
`WANDB_API_KEY` is not set.

#### Scenario: Offline warm-worker test uses real predictors, no mocks

- **WHEN** the default test suite runs the warm-worker tests on CPU with a `LocalCardSource` over the
  vendored models
- **THEN** real `Predictor`s are built and reused and real `sio.Labels` are produced, with no mocks

#### Scenario: Wandb tests skip without credentials

- **WHEN** the `@pytest.mark.wandb` tests run with no `WANDB_API_KEY` set
- **THEN** they skip with a message rather than failing or hitting the network
