## Why

Model selection + preparation currently lives in a separate, filesystem-only tool
(`models-downloader`). Roadmap tier **A3-predict** needs the predict service itself to fetch
root models from the wandb registry, choose one per root type from Bloom scan metadata, and
keep them resident in memory across scans — building on the already-shipped Layer-1 inference
core (`make_predictor` / `predict_on_video`). Full design:
`docs/superpowers/specs/2026-07-03-warm-model-worker-design.md`.

## What Changes

- Add a new **`model-management`** capability, implemented as three small modules on top of
  the untouched Layer-1 `predict.py`:
  - `model_selection.py` — a **pure** matcher `choose_models(params, cards) -> dict[RootType, ModelRef]`
    (explicit override wins; else `species ==`, `mode ==`, `age_min <= age <= age_max`;
    exactly-one selects, zero skips, more-than-one is an error).
  - `model_registry.py` — a `ModelCardSource` protocol (`list_cards()` + `materialize(ref) -> Path`)
    with a **`LocalCardSource`** (offline) and a **`WandbRegistrySource`** (all network confined
    here; alias→concrete-version pinning + `weights_checksum` + runtime `sleap_nn_version`).
  - `warm_worker.py` — a `WarmModelWorker` that resolves → fetches-once → loads-once → holds
    `Predictor`s resident keyed by `(registry_id, version)` and reuses them across scans;
    **fail-loud** if any resolved root type cannot be materialized/loaded.
- Add a **`sleap-roots-contracts==0.1.0a3`** dependency (`ModelCard`, `ModelRef`, `ResolvedParams`,
  `RootType`) and declare **`wandb`** directly (it is already a transitive dependency of sleap-nn
  0.3.0; `WandbRegistrySource` imports it, kept lazy). Regenerate + commit `uv.lock`.
- Add a **`@pytest.mark.wandb`** marker, deselected by default **and in CI** (`addopts` plus the
  explicit `-m` filters in `ci.yml`, which override `addopts`); wandb tests carry a collection-time
  `skipif(not WANDB_API_KEY)`.
- The existing **`prediction`** capability and `predict.py` are **unchanged** — the worker only
  composes `make_predictor` / `predict_on_video`.

## Out of Scope (deferred — stated explicitly)

- External serving protocol / CLI entrypoint (A4 orchestration / CLI slice).
- The `predictions.csv` manifest and `{scan}.model{id}.root{type}.slp` naming (output-contract slice).
  `predict()`'s `save_dir` writes only **raw** per-root `.slp` (`save_dir/<root_type>.slp`); the
  manifest + scan-aware naming are the output-contract slice's job (tracked in tasks 9.3).
- Emitting `Provenance` / `ResultEnvelope` (orchestration slice). This slice only *produces* pinned
  `ModelRef`s and *exposes* the effective inference config for those layers to record.

## Impact

- **Affected specs:** `model-management` (NEW capability — ADDED requirements).
- **Affected code:**
  - `sleap_roots_predict/model_selection.py`, `model_registry.py`, `warm_worker.py` (new).
  - `sleap_roots_predict/__init__.py` — export the new public API.
  - `pyproject.toml` — add `sleap-roots-contracts==0.1.0a3` + `wandb`; register the `wandb` marker.
  - `uv.lock` — regenerate + commit (Dockerfile uses `uv sync --frozen`).
  - `.github/workflows/ci.yml` — add `not wandb` to the `-m` filters (all runners).
  - `CLAUDE.md`, `openspec/project.md`, `README.md` — sync module inventory / deps / marker / env vars.
  - `tests/test_model_selection.py`, `tests/test_warm_worker.py` (offline, no mocks),
    `tests/test_model_registry.py` (gated `@pytest.mark.wandb`); `tests/conftest.py` fixtures.
- **Dependency/sequencing:** `sleap-roots-contracts==0.1.0a3` is **already published on PyPI** and
  `wandb` is already a transitive sleap-nn dependency, so the offline groups are **not blocked** —
  implementation can start immediately.
