> **TDD discipline:** every sub-group writes the failing test FIRST (RED), then the minimum
> implementation to pass (GREEN). No mocks of the sleap-nn / sleap-io / wandb boundaries — the
> matcher uses tiny real `ModelCard` lists; the warm worker uses a `LocalCardSource` over the
> vendored `tests/assets/models/` fixtures (native + legacy) and runs real CPU inference.
> **Blocked start:** groups 1+ require `sleap-roots-contracts==0.1.0a3` importable — group 0 gates it.

## 0. Dependencies + markers (gates the rest)

- [ ] 0.1 Add `sleap-roots-contracts==0.1.0a3` and `wandb` to `pyproject.toml` `dependencies`;
      `uv sync --extra dev --extra cpu`; confirm `from sleap_roots_contracts import ModelCard, ModelRef, ResolvedParams, RootType` imports.
- [ ] 0.2 Register a `wandb` pytest marker in `pyproject.toml` `[tool.pytest.ini_options].markers`
      and add it to the default deselection (`addopts = "-m 'not gpu and not acceptance and not wandb'"`).

## 1. Model-selection matcher (pure, test-first)

- [ ] 1.1 RED: `tests/test_model_selection.py` — build tiny real `ModelCard` lists and assert
      `choose_models` returns one `ModelRef` per root type on exact `(species, mode, age∈[min,max])`
      match; age outside the window does not match; zero matches skips the root type; >1 match raises
      (ambiguous, names the root type); an explicit override bypasses matching; missing
      `species`/`mode`/`age` raises a clear error. Assert purity (no filesystem/network).
- [ ] 1.2 GREEN: implement `sleap_roots_predict/model_selection.py` — `choose_models(params, cards, overrides=None)`
      reading `species`/`mode`/`age` from `ResolvedParams.values`, filtering per the rules, and
      converting the chosen `ModelCard` to a `ModelRef` (stamping runtime `sleap_nn_version`). Pure.

## 2. Card-source protocol + LocalCardSource (offline, test-first)

- [ ] 2.1 RED: `tests/test_warm_worker.py` (or `tests/test_model_registry.py` for the offline part) —
      assert a `LocalCardSource` built over the vendored model dirs returns `ModelCard`s from
      `list_cards()` and that `materialize(ref)` returns an on-disk directory (no network), that
      `make_predictor` can subsequently load.
- [ ] 2.2 GREEN: implement `sleap_roots_predict/model_registry.py` — the `ModelCardSource` `Protocol`
      (`list_cards`, `materialize`) and `LocalCardSource`. Keep any `wandb` import lazy/local so the
      module imports without network and without touching wandb for the offline path.

## 3. Warm worker residency (offline no-mock, test-first)

- [ ] 3.1 RED: in `tests/test_warm_worker.py`, using a `LocalCardSource` over `tests/assets/models/`
      (native + legacy) and a small vendored video, assert: `resolve(params)` returns `ModelRef`s
      without loading weights; `get_predictors(params)` builds `Predictor`s and a second call for the
      same model returns the **same** `Predictor` instance (warm reuse, no reload); two different
      param sets resolving to the same `(registry_id, version)` share one resident `Predictor`;
      `predict(params, video)` returns real `sio.Labels` per root type.
- [ ] 3.2 GREEN: implement `sleap_roots_predict/warm_worker.py` — `WarmModelWorker` with
      `resolve`/`get_predictors`/`predict`, lazy card load, and an internal `(registry_id, version) ->
      Predictor` cache (fetch-once/load-once/reuse).
- [ ] 3.3 RED: assert **fail-loud** — when a resolved root type's model cannot be materialized/loaded,
      `get_predictors` raises an error naming the root type and `registry_id:version` and returns no
      predictors (no partial results).
- [ ] 3.4 GREEN: implement the fail-loud path in `get_predictors`.

## 4. Effective inference-config capture (test-first)

- [ ] 4.1 RED: assert `WarmModelWorker.inference_config()` returns the resolved `device`,
      `peak_threshold`, and `batch_size` actually used, and that the exposed output-defining subset
      contains `peak_threshold` and excludes `device`/`batch_size`.
- [ ] 4.2 GREEN: implement `inference_config()` and the output-defining-subset accessor.

## 5. WandbRegistrySource (network path, gated test-first)

- [ ] 5.1 RED: `tests/test_model_registry.py` — a `@pytest.mark.wandb` test that **skips cleanly**
      when `WANDB_API_KEY` is unset; when set (local/self-hosted), asserts `list_cards()` yields
      `ModelCard`s and `materialize(ref)` downloads the pinned version into `SRP_MODEL_CACHE_DIR`,
      recording the concrete version + `weights_checksum` into the `ModelRef`, and that a second
      `materialize` reuses the cache.
- [ ] 5.2 GREEN: implement `WandbRegistrySource` in `model_registry.py` — `wandb.Api()`-based listing
      + `artifact.download(root=cache_dir)`; read entity/registry/cache from
      `SRP_WANDB_ENTITY`/`SRP_WANDB_REGISTRY`/`SRP_MODEL_CACHE_DIR`; alias→concrete-version pinning +
      `weights_checksum` + runtime `sleap_nn_version` stamping. All network confined here.

## 6. Public API + wiring

- [ ] 6.1 Export the public surface from `sleap_roots_predict/__init__.py`
      (`WarmModelWorker`, `choose_models`, `LocalCardSource`, `WandbRegistrySource`,
      `ModelCardSource`) and confirm importing the package does not require network or `WANDB_API_KEY`.

## 7. Verify

- [ ] 7.1 `/lint` (black + ruff docstrings + codespell) and `/test` — the default suite is green with
      `gpu`, `acceptance`, and `wandb` deselected (offline warm-worker tests run real CPU inference).
- [ ] 7.2 `openspec validate add-warm-model-worker --strict` passes.

## 8. Post-merge (NOT part of this PR)

- [ ] 8.1 After merge: `/cleanup-merged` → `/openspec:archive add-warm-model-worker`.
- [ ] 8.2 **Update the roadmap** (`sleap-roots-pipeline/docs/bloom-integration/roadmap.md`, on the
      up-to-date roadmap branch): A3-predict progress, repo table, retire the standalone
      models-downloader stage from the A4 warm path (per the roadmap skill format).
