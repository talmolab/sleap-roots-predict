## Context

Full design (brainstormed + approved):
`docs/superpowers/specs/2026-07-03-warm-model-worker-design.md`. This file records only the
OpenSpec-relevant decisions; see the design doc for the complete rationale, data-flow diagram,
and the reproducibility discussion.

Layer 1 (`predict.py`) already exists: `make_predictor(model_paths, …)` returns a persistent
`sleap_nn.inference.Predictor` (loaded once, reused) and `predict_on_video`. This change adds
Layer 2 (selection + fetch) and a thin Layer 3 (warm residency) on top of it.

## Goals / Non-Goals

- **Goals:** fetch root models from the wandb registry; choose models per root type from Bloom
  scan metadata; keep `Predictor`s resident across scans; keep the core offline/no-mock testable.
- **Non-Goals (this slice):** external serving protocol / CLI, the `predictions.csv` output
  contract + `.slp` naming, and emitting `Provenance`/`ResultEnvelope`.

## Decisions

- **Registry-as-source-of-truth via a pure matcher + pluggable source.** `choose_models` is a
  pure function over `list[ModelCard]`; the *source* of cards is a `ModelCardSource` protocol with
  two responsibilities — `list_cards()` (catalog for matching) and `materialize(ref) -> Path`
  (the model dir for loading). This is what preserves **no-mock** testability: tests inject a
  `LocalCardSource` over vendored `tests/assets/models/` and exercise the whole warm path with real
  sleap-nn Predictors; only `WandbRegistrySource` touches the network, and it is covered by gated
  tests. Alternatives (a hand-maintained YAML table; a single monolithic `predict()` that calls
  wandb directly) were rejected — see the design doc.
- **Pinning lives in the card, not the matcher or a `ModelRef` built by the source.** `ModelCard`
  (contracts `0.1.0a3`) carries the concrete `registry_id`/`version`/`weights_checksum`;
  `list_cards()` is where a moving alias (`production`) is resolved to a concrete version and baked
  into the card. `choose_models` then builds the `ModelRef` via `ModelCard.to_model_ref(runtime)`,
  which copies the card's pin and stamps the runtime sleap-nn version. The source never constructs
  `ModelRef`s. The runtime version is resolved once at import so the matcher stays pure per call.
- **`ModelCard` has no path field**, so `LocalCardSource` is built from `(card, path)` pairs and
  holds a `(registry_id, version) -> Path` map; `materialize(ref)` resolves through it. This keeps
  the offline path real (no mocks, no fake path field on the contract model).
- **Identity-keyed residency.** `WarmModelWorker` caches `Predictor`s by `(registry_id, version)`,
  not by root type or scan, so different scan types that resolve to the same model version are a
  cache hit (e.g. canola/pennycress/arabidopsis share a primary model). Fetch-once, load-once.
- **Fail-loud on partial results.** If a resolved root type cannot be materialized/loaded,
  `get_predictors` raises identifying the root type + `registry_id:version`; a scan never produces
  partial predictions.
- **Config: whole recorded, output-defining subset hash-bearing.** The worker exposes the effective
  inference config; `peak_threshold` is the output-defining knob; `device`/`batch_size` are
  diagnostic-only. Recording into `Provenance.predict_inference_config` / `predict_output_params`
  (and the `idempotency_key`) is the downstream slice's job — contracts `0.1.0a3` already has the
  fields.

## Risks / Trade-offs

- **wandb network boundary** → confined to `WandbRegistrySource` with a lazy `import wandb`;
  unit/warm tests never touch it (offline via `LocalCardSource`). `wandb` is already a transitive
  sleap-nn dependency, so declaring it direct is honest and does not enlarge the image.
- **CI marker override** → CI passes an explicit `-m` that overrides `addopts`, so the `wandb`
  deselection must also be added to `ci.yml`'s `-m` filters (incl. the self-hosted runner), and the
  gated tests carry `skipif(not WANDB_API_KEY)` so they never hit the network on shared runners.
- **Concurrency** → v1 assumes sequential scans; the residency cache is unguarded (documented
  limitation; a per-key load lock is future work).

## Migration Plan

N/A — new capability, additive; no existing behavior changes. Rollback = revert the additive commits.

## Open Questions

- Exact production-registry identity (a dedicated registry vs. curating the `production` alias in
  `sleap-roots-models`) — configurable via `SRP_WANDB_ENTITY`/`SRP_WANDB_REGISTRY`; settled with
  `sleap-roots-training`. Does not block offline work.
- Best `peak_threshold` per dataset (the output-defining knob) is an open empirical question tracked
  in issue #8; the worker records whatever value is used, so tuning does not change this design.
