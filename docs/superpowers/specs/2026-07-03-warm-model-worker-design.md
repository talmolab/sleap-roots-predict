# Warm model worker — design

**Date:** 2026-07-03
**Repo:** `sleap-roots-predict`
**Branch:** `add-warm-model-worker`
**Roadmap tier:** A3-predict (bloom-integration roadmap)
**Status:** design approved in brainstorming; pending OpenSpec proposal

## Context

The inference core (Layer 1) already shipped (PR #6): `make_predictor(model_paths, …)`
returns a persistent `sleap_nn.inference.Predictor` (weights loaded once, reused across
videos) and `predict_on_video(predictor, video, save_path=None)`. Legacy SLEAP configs
load via a sanitizer. All inference tests are real (no mocks) against vendored minimal
models in `tests/assets/`.

Today, model selection/prep lives in a separate GitLab tool, `models-downloader`: it reads
a hand-authored `model_params.json` (`species`, `mode`, `age`, optional explicit
`*_model_id`) plus a `model_chooser_table.xlsx`, picks primary/lateral/crown model IDs,
copies legacy SLEAP `.zip`s to an output dir, and writes `model_paths.csv`. It is
filesystem-only — no network, no wandb.

Meanwhile `sleap-roots-training` publishes trained models to a wandb **registry**
(`sleap-roots-models`, entity `eberrigan-salk-institute-for-biological-studies`, project
`sleap-roots`) as **native model directories** (`run.log_artifact` + `add_dir`), with
collections per experiment/root-type (e.g. `…-primary`) and aliases like `production`,
`staging`. Species and root_type ride as tags/boolean flags; **mode and age are not
attached to artifacts today.**

## Goal

Consolidate model selection + fetch into this repo and keep models resident in memory
across scans:

1. **Fetch** root models from the wandb registry (retiring the filesystem-only
   `models-downloader` stage).
2. **Choose** models from Bloom scan metadata (species/mode/age → a model per root type:
   primary/lateral/crown), with an explicit override.
3. **Keep models resident** — fetch-once, load-once, reuse across many scans.

## Scope

**In scope (this slice):**

- Layer 2: a pure model-selection matcher + a pluggable model-card source (wandb + local)
  with fetch/cache.
- Layer 3 (thin): an in-process warm worker that resolves → fetches-once → loads-once →
  holds `Predictor`s resident and reuses them across scans.
- Reproducibility plumbing: produce fully-pinned `ModelRef`s and capture the effective
  inference config.

**Out of scope (deferred to other slices):**

- External serving protocol (RPC/queue/HTTP/CLI entrypoint) — A4 orchestration / CLI slice.
- Output contract: `predictions.csv` manifest and the `{scan}.model{id}.root{type}.slp`
  naming — output-contract/CLI slice.
- Emitting `Provenance`/`ResultEnvelope` — orchestration/output slice. This slice only
  *produces the model half* (pinned `ModelRef`s) and *exposes* the inference config for
  those layers to record.
- `images_checksum`, `predict_code_sha`, `predict_container_digest` — input/orchestration
  layers.

## Decisions (settled in brainstorming)

1. **Slice depth:** Layer 2 + thin in-process warm holder. Defer serving + output contract.
2. **Selection model:** registry-as-source-of-truth via a **pure matcher** over
   `list[ModelCard]` → `dict[RootType, ModelRef]`, with a **pluggable card source**
   (wandb-registry source for real use; injected/local source for offline no-mock tests).
   No hand-maintained policy file in this repo. Flip the default source to the live
   production registry once training tags models.
3. **`ModelCard` schema home:** add to `sleap-roots-contracts` first (next to `ModelRef`),
   then depend on it here.
4. **Params + identity:** consume `ResolvedParams`, return `ModelRef`, both from
   `sleap-roots-contracts`.
5. **Partial results:** **fail loud** — if a resolved root type cannot be
   materialized/loaded, `get_predictors` raises identifying the root type +
   `registry_id:version`; the scan produces nothing rather than partial predictions.
6. **Inference config:** capture the **full effective** config for audit, but only the
   **output-defining subset** (e.g. `peak_threshold`) is reproducibility/hash-bearing.
   Hardware/throughput knobs (`device`, `batch_size`) are recorded as diagnostics but
   **not** hashed (hashing them would break cross-node idempotency dedup).
7. **Config home in contracts:** add a dedicated `predict_inference_config` field to
   `Provenance` and fold its output-defining subset into the `idempotency_key` derivation.
8. **GPU env fingerprint:** roadmap now, contract later (within-tolerance parity does not
   require bit-exact host pinning).

## Architecture

Three small, single-purpose units on top of the built Layer 1.

| Module | Purpose | Network? | Depends on |
|---|---|---|---|
| `sleap_roots_predict/model_selection.py` | **Pure matcher** `choose_models(params, cards) -> dict[RootType, ModelRef]` | none | contracts (`ResolvedParams`, `ModelRef`, `ModelCard`, `RootType`) |
| `sleap_roots_predict/model_registry.py` | **Card source + fetch**: wandb `list_cards()` + `materialize(ref) -> Path`; local/injected source for offline use | **only here** | wandb, contracts |
| `sleap_roots_predict/warm_worker.py` | **Warm holder**: resolve → fetch-once → load-once → hold `Predictor`s resident, reuse across scans; fail-loud | none (delegates) | the two above + Layer 1 |

`predict.py` (Layer 1) is untouched except as a consumer.

### The source protocol

One protocol, two responsibilities, two implementations:

```python
class ModelCardSource(Protocol):
    def list_cards(self) -> list[ModelCard]: ...          # catalog (metadata only) — for matching
    def materialize(self, ref: ModelRef) -> Path: ...     # ref -> local model dir (the bytes)
```

- **`WandbRegistrySource`** — `list_cards()` reads the production registry's artifacts'
  selection metadata (`species, mode, age_min, age_max, root_type`) into `ModelCard`s;
  `materialize()` = `artifact.download(root=cache_dir)`. All network lives here. Uses
  `wandb.Api()` (no `wandb.init` run pollution). Auth via `WANDB_API_KEY`.
- **`LocalCardSource`** — cards carry an on-disk path; `materialize()` returns it. Points at
  vendored `tests/assets/models/` fixtures so the **entire warm path** runs offline with
  **real sleap-nn Predictors and zero mocks**. Also usable as a real filesystem mode.

Bundling both methods on one object keeps them configured consistently (same entity,
registry, cache dir) — you always download from the registry you browsed.

*Seam noted for later:* `list_cards()` reads the whole catalog. Fine while the registry is
small; add a filtered `list_cards(species=…, mode=…)` if it grows.

### The warm worker surface

```python
class WarmModelWorker:
    def __init__(self, source, *, cache_dir=None, device="auto",
                 peak_threshold=0.2, batch_size=4): ...
    def resolve(self, params) -> dict[RootType, ModelRef]:          # pure decision, no I/O
    def get_predictors(self, params) -> dict[RootType, Predictor]:  # resolve → materialize → make_predictor, CACHED
    def predict(self, params, video, save_dir=None) -> dict[RootType, Labels | Path]:  # thin convenience
    def inference_config(self) -> dict:                             # effective config used (for provenance)
```

- **Residency** = internal `dict[(registry_id, version) -> Predictor]`. Keyed by **model
  identity**, not by root type or scan — so two different scan types that resolve to the
  same model version are a cache hit (e.g. canola/pennycress/arabidopsis share a primary
  model). Fetch-once, load-once, reuse.
- Cards are lazy-loaded once on first `resolve`/`get_predictors`.
- **Fail loud:** any resolved root type whose model fails to materialize/load raises,
  naming the root type + `registry_id:version`.
- `predict()` is deliberately thin: returns in-memory `Labels` (optionally saves plain
  per-root `.slp`); it does **not** write `predictions.csv` or apply scan-aware naming.

### Matcher semantics (mirrors proven models-downloader logic)

For each root type: explicit override wins; else filter cards by `species ==`, `mode ==`,
`age_min ≤ age ≤ age_max`. **Exactly one** card per root type → select it; **zero** → skip
that root type (e.g. soybean has no crown); **more than one** → error (ambiguous).

### Data flow (one scan)

```
ResolvedParams{values: species, mode, age, [model_overrides]}
  │  choose_models(params, cards)                  cards ← source.list_cards()
  ▼
dict[RootType -> ModelRef]  (age∈[min,max]; exactly-one-per-present-root-type)
  │  for each ModelRef: source.materialize(ref)    wandb download OR local dir
  ▼
dict[RootType -> local model dir]
  │  make_predictor([dir])   cached by (registry_id, version)
  ▼
dict[RootType -> Predictor]  (RESIDENT / warm)
  │  predict_on_video(predictor, video)            (per root type; convenience)
  ▼
dict[RootType -> sio.Labels]
```

## Reproducibility

- **Alias pinning (the core act):** scans select via a moving alias (`production`); the
  resolver pins it to the **concrete version** at resolve time and records that (not the
  alias) into `ModelRef.version`, plus `weights_checksum` and `sleap_nn_version`.
- **`ModelRef` field sourcing:**
  - `version` — the concrete wandb version the alias resolved to.
  - `weights_checksum` — wandb artifact content digest (wandb source) / weights file hash
    (local source). Always present; feeds `idempotency_key`.
  - `sleap_nn_version` — the **runtime** `importlib.metadata.version("sleap-nn")` (what
    actually runs inference), with a warning on mismatch against the artifact's
    trained-with version.
- **Inference config:** the worker captures the **full effective** config and exposes it
  via `inference_config()`. The output-defining subset (`peak_threshold`, …) is
  hash-bearing; `device`/`batch_size` are diagnostic-only. The orchestration/output slice
  folds this into the new `Provenance.predict_inference_config` (and the `idempotency_key`).
- **Environment:** the `Dockerfile` builds with `uv sync --frozen` from a committed
  `uv.lock`, so the image is a faithful, content-addressed materialization of the locked
  dependency graph. `uv.lock` therefore does **not** need separate archival — the immutable
  container **digest** pins the whole environment, tied back to source via `code_sha`.
  Requirement: provenance must record the immutable `@sha256` **digest**, never a mutable
  tag. (Owned by the orchestration slice; noted here so the chain is explicit.)
- **Host/GPU:** the digest does not pin the NVIDIA driver, GPU SKU, or cudnn/TF32 behavior
  (host-level). This yields *within-tolerance* parity, not bit-exact. Node pinning +
  determinism policy + an env fingerprint are roadmap items (see below), not this slice.

## Cross-repo prerequisites & coordination

- **`sleap-roots-contracts` (hard prerequisite):**
  1. Add `ModelCard` (selection metadata `{species, mode, age_min, age_max, root_type}` +
     identity fields incl. the artifact's trained-with `sleap_nn_version`), next to
     `ModelRef`. The resolver builds `ModelRef` from a card and **stamps the runtime**
     `sleap_nn_version` (warning on mismatch vs. the card's trained-with value).
  2. Add `Provenance.predict_inference_config` and fold its output-defining subset into
     `idempotency_key`.
  3. Version bump + release.
  During development, pin `sleap-roots-predict` to that contracts ref so TDD here is not
  blocked. This contracts change is driven as its own small PR first.
- **`sleap-roots-training` (coordination, not blocking):** promotion must write the
  structured selection metadata to a curated **production** registry/collection for the
  *live* default source. Not required for this slice's merge — offline tests use
  `LocalCardSource`.
- **Open config item:** exact production-registry identity — a new dedicated registry (e.g.
  `wandb-registry-sleap-roots-models-production`) vs. curating the `production` alias inside
  `sleap-roots-models`. Configurable via `SRP_WANDB_ENTITY` / `SRP_WANDB_REGISTRY`; settled
  with training.

## Configuration (env)

- `WANDB_API_KEY` — auth (k8s secret).
- `SRP_WANDB_ENTITY` — default `eberrigan-salk-institute-for-biological-studies`.
- `SRP_WANDB_REGISTRY` — production registry/collection name (default settled with training).
- `SRP_MODEL_CACHE_DIR` — download cache (fallback `WANDB_CACHE_DIR`, then wandb default);
  point at a persistent/hostPath volume in k8s.
- `SRP_DEVICE` — device override (already honored by Layer 1).

## Testing strategy (real TDD, no mocks)

- **Matcher** (`model_selection`): unit tests over tiny real `ModelCard` lists — exact
  match, age-range membership, ambiguity → error, zero → skip, override wins. Pure, offline.
- **Warm worker**: `LocalCardSource` → vendored `tests/assets/models/` (native + legacy) →
  assert the **same `Predictor` object is reused** across two `get_predictors` calls
  (warmth), real `Labels` produced, and **fail-loud** on an unmaterializable root type.
  Offline, no mocks.
- **wandb source** (`model_registry`): gated integration test (new `@pytest.mark.wandb`,
  CI-deselected like `acceptance`) against real wandb; skips cleanly without `WANDB_API_KEY`.
- Reuse the vendored fixtures and the existing no-mock conventions from `tests/`.

## Post-merge roadmap update (required)

Update `sleap-roots-pipeline/docs/bloom-integration/roadmap.md` (A3-predict + A4):

1. A3-predict progress; repo table; **retire the standalone models-downloader stage** (the
   warm worker fetches in-process — no separate stage in the A4 DAG).
2. Add reproducibility-assurance items to A3/A4:
   - Provenance-emission responsibilities split by layer (predict emits pinned `ModelRef`s +
     inference config; A4 owns `container_digest` + `code_sha`).
   - Container-identity discipline: reference images by immutable `@sha256`; stamp
     `code_sha` into the image.
   - GPU determinism policy + node pinning + a best-effort env fingerprint
     (`torch.version.cuda`, device name, driver).
   - A parity harness + the actual tolerance number (keypoint RMSE ≤ N px / trait-summary
     delta ≤ X% on a reference scan set).

## Open questions / future work

- Concurrency: v1 assumes sequential scans; `_predictors` is unguarded. Add a per-key load
  lock if scans ever run concurrently in one process (YAGNI now).
- Eviction/refresh: no eviction v1; refresh = restart (versions pinned). LRU cap later.
- Filtered `list_cards` if the registry grows large.
