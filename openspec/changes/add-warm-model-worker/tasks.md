> **TDD discipline:** every sub-group writes the failing test FIRST (RED), then the minimum
> implementation to pass (GREEN). **Commit each group's RED test together with its GREEN
> implementation** — never commit a bare RED step. No mocks of the sleap-nn / sleap-io / wandb
> boundaries — the matcher uses tiny real `ModelCard` lists; the warm worker uses a `LocalCardSource`
> over the vendored `tests/assets/models/` fixtures (native + legacy) and runs real CPU inference.
> **Not blocked:** `sleap-roots-contracts==0.1.0a3` is already published on PyPI (pure-Python wheel)
> and `wandb` is already a transitive dependency of sleap-nn 0.3.0 — so the offline groups can start
> immediately.

## 0. Dependencies + markers (gates the rest)

- [x] 0.1 Add `sleap-roots-contracts==0.1.0a3` and `wandb` to `pyproject.toml` `dependencies`
      (`wandb` is already transitive via sleap-nn; declaring it direct is honest since
      `WandbRegistrySource` imports it). `uv sync --extra dev --extra cpu`; confirm
      `from sleap_roots_contracts import ModelCard, ModelRef, ResolvedParams, RootType` imports.
- [x] 0.2 **Regenerate and commit `uv.lock`** (it currently lacks `sleap-roots-contracts`); the
      `Dockerfile` runs `uv sync --frozen`, so a stale lock fails the docker build.
- [x] 0.3 Register a `wandb` pytest marker in `pyproject.toml` `[tool.pytest.ini_options].markers`
      and add it to the default deselection (`addopts = "-m 'not gpu and not acceptance and not wandb'"`).
- [x] 0.4 Add `not wandb` to the `-m` expressions in `.github/workflows/ci.yml` (all runners,
      including the self-hosted GPU runner) — CI passes an explicit `-m` that overrides `addopts`, so
      without this the wandb tests would be collected in CI (and could hit the network on a runner
      that carries `WANDB_API_KEY`).

## 1. Model-selection matcher (pure, test-first)

- [x] 1.1 RED: `tests/test_model_selection.py` — build tiny real `ModelCard` lists and assert
      `choose_models` returns one `ModelRef` per root type on exact `(species, mode, age∈[min,max])`
      match (with the card's concrete `version`/`weights_checksum` and the runtime `sleap_nn_version`
      stamped); age at the `age_min`/`age_max` boundary matches (inclusive); age outside does not;
      zero matches skips the root type and no-match-at-all returns an empty mapping; >1 match raises
      (names the root type); an explicit override bypasses matching even with zero cards; missing
      `species`/`mode`/`age` raises. Assert no network and no per-call filesystem I/O.
- [x] 1.2 GREEN: implement `sleap_roots_predict/model_selection.py` — `choose_models(params, cards, overrides=None)`
      reading `species`/`mode`/`age` from `ResolvedParams.values`, filtering per the rules, and
      converting the chosen `ModelCard` to a `ModelRef` via `card.to_model_ref(runtime_version)`.
      Resolve the runtime sleap-nn version **once at import** (`importlib.metadata.version("sleap-nn")`)
      so the function does no per-call filesystem I/O.

## 2. Card-source protocol + LocalCardSource (offline, test-first)

- [x] 2.1 RED: `tests/test_warm_worker.py` — assert a `LocalCardSource` built from `(card, path)`
      pairs over the vendored model dirs returns the `ModelCard`s from `list_cards()` and that
      `materialize(ref)` returns the on-disk directory mapped to `ref`'s `(registry_id, version)`
      (no network), which `make_predictor` can load.
- [x] 2.2 GREEN: implement `sleap_roots_predict/model_registry.py` — the `ModelCardSource` `Protocol`
      (`list_cards`, `materialize`) and `LocalCardSource` holding a `(registry_id, version) -> Path`
      mapping. Keep any `wandb` import **lazy/local** so the module imports without touching wandb.

## 3. Warm worker residency (offline no-mock, test-first)

- [x] 3.1 RED: in `tests/test_warm_worker.py`, using a `LocalCardSource` over `tests/assets/models/`
      (native + legacy, one root type per format) and a small vendored video, assert: `resolve(params)`
      returns `ModelRef`s while the predictor cache stays empty (no load); `get_predictors(params)`
      builds `Predictor`s and a second call for the same model returns the **same** `Predictor`
      instance (warm reuse); two different param sets resolving to the same `(registry_id, version)`
      share one resident `Predictor`; a params that matches one root type and skips another returns
      results only for the matched type; `predict(params, video)` returns real `sio.Labels` per root
      type; `predict(params, video, save_dir=…)` writes one reloadable raw `.slp` per root type under
      `save_dir` and no `predictions.csv`.
- [x] 3.2 GREEN: implement `sleap_roots_predict/warm_worker.py` — `WarmModelWorker` with
      `resolve`/`get_predictors`/`predict`, lazy card load, and an internal `(registry_id, version) ->
      Predictor` cache (fetch-once/load-once/reuse). `predict` composes `predict_on_video`; with
      `save_dir` it writes `save_dir/<root_type>.slp` only (no manifest, no scan-aware naming).
- [x] 3.3 RED: assert **fail-loud** — when a resolved root type's model cannot be materialized/loaded,
      `get_predictors` raises an error naming the root type and `registry_id:version` and returns no
      predictors (no partial results).
- [x] 3.4 GREEN: implement the fail-loud path in `get_predictors`.

## 4. Effective inference-config capture (test-first)

- [x] 4.1 RED: assert `WarmModelWorker.inference_config()` returns the resolved `device`,
      `peak_threshold`, and `batch_size` actually used, and that the exposed output-defining subset
      contains `peak_threshold` and excludes `device`/`batch_size`.
- [x] 4.2 GREEN: implement `inference_config()` and the output-defining-subset accessor.

## 5. WandbRegistrySource (network path, gated test-first)

- [x] 5.1 RED (ungated, offline): assert `WandbRegistrySource` raises a clear error naming
      `WANDB_API_KEY` (before any network call) when the key is unset — this behavior is testable
      without the `wandb` marker.
- [x] 5.2 RED (gated): `tests/test_model_registry.py` — a `@pytest.mark.wandb` test carrying a
      collection-time `skipif(not WANDB_API_KEY)` (mirror `tests/test_acceptance.py`) that, when creds
      are present, asserts `list_cards()` yields `ModelCard`s with alias→concrete-version pinning and
      that `materialize(ref)` downloads the pinned version into `SRP_MODEL_CACHE_DIR`, reusing the
      cache on a second call.
- [x] 5.3 GREEN: implement `WandbRegistrySource` in `model_registry.py` — `wandb.Api()`-based listing
      + `artifact.download(root=cache_dir)`; read entity/registry/cache from
      `SRP_WANDB_ENTITY`/`SRP_WANDB_REGISTRY`/`SRP_MODEL_CACHE_DIR`; alias→concrete-version pinning +
      `weights_checksum` into the card. All network confined here; `import wandb` stays lazy/local.

## 6. Public API + wiring

- [x] 6.1 RED: add a test asserting the package imports with **no network and no `WANDB_API_KEY`** and
      that the public surface is importable from the package root.
- [x] 6.2 GREEN: export the public surface from `sleap_roots_predict/__init__.py`
      (`WarmModelWorker`, `choose_models`, `LocalCardSource`, `WandbRegistrySource`, `ModelCardSource`)
      and update the module docstring to name the model-management capability.

## 7. Docs

- [x] 7.1 Run `/docs-review` and sync the living docs to the new inventory: `CLAUDE.md` (Package
      Structure — add the 3 modules; Key Dependencies — add `wandb`, `sleap-roots-contracts`; the
      public-API line; a new "Runtime configuration (env)" subsection documenting `WANDB_API_KEY`,
      `SRP_WANDB_ENTITY`, `SRP_WANDB_REGISTRY`, `SRP_MODEL_CACHE_DIR`); `openspec/project.md`
      (Architecture Patterns module list, External Dependencies, the `wandb` marker); `README.md`
      (Project Structure tree). While there, fix pre-existing inaccuracies: `project.md`'s
      `predict_on_h5`/`batch_predict` (do not exist), the stale A0 "do not do the warm worker" note,
      and CLAUDE.md's hardcoded "105 tests passing".

## 8. Verify

- [ ] 8.1 `/lint` (black + ruff docstrings + codespell) and `/test` — the default suite is green with
      `gpu`, `acceptance`, and `wandb` deselected (offline warm-worker tests run real CPU inference).
- [ ] 8.2 Run `uv run pytest -m wandb` and confirm the wandb tests **skip** (not error) with no creds.
- [ ] 8.3 `openspec validate add-warm-model-worker --strict` passes.

## 9. Post-merge (NOT part of this PR)

- [ ] 9.1 After merge: `/cleanup-merged` → `/openspec:archive add-warm-model-worker`.
- [ ] 9.2 **Update the roadmap** (`sleap-roots-pipeline/docs/bloom-integration/roadmap.md`, on the
      up-to-date roadmap branch): A3-predict progress, repo table, retire the standalone
      models-downloader stage from the A4 warm path (per the roadmap skill format).
- [ ] 9.3 Output-contract follow-up: the deferred slice picks up `predict()`'s per-root `.slp` output
      and adds the `predictions.csv` manifest + `{scan}.model{id}.root{type}.slp` naming.
