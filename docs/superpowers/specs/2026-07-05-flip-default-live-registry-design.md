# Design: flip predict's default model source to the live production registry

- **Date:** 2026-07-05
- **Branch:** `flip-default-live-registry`
- **Issues:** closes predict **#11** (default source + env rename) and **#12**
  (per-artifact error isolation)
- **Roadmap:** advances tier **A3-predict** ("flip predict's default source to the
  live registry")
- **Status:** design — awaiting approval before the OpenSpec proposal

## Motivation

`WandbRegistrySource` today only works when the operator sets `SRP_WANDB_REGISTRY`,
which has **no default** and a generic name; and `WarmModelWorker` requires an
explicit `source`. So fetching production models is opt-in and manually wired.

The production registry is now seeded, verified, and finalized (`pytest -m wandb` is
green against it): entity `eberrigan-salk-institute-for-biological-studies`, registry
`sleap-roots-models`, alias `production`, 13 production `ModelCard`s over 8
SHA256-pinned models. This change makes reading that live registry the **default**, so
the warm worker fetches production models out-of-the-box with only `WANDB_API_KEY` set.

## Decisions (settled in brainstorming)

1. **Default source, fail-loud.** `WarmModelWorker(source=None)` builds a
   `WandbRegistrySource` by default; `WandbRegistrySource()` defaults its registry to
   `sleap-roots-models`. A missing `WANDB_API_KEY` fails loud when cards are listed —
   **no silent `LocalCardSource` fallback**. `LocalCardSource` stays an explicit opt-in
   for offline/tests.
2. **Hard rename** the env vars (no deprecated alias). The package is alpha
   (`0.0.1a0`); the old names had no default and were referenced only by one gated test
   and docs.
3. **Defer predict #7** (reshape the worker's config accessors to the `Provenance`
   contract fields `predict_inference_config` / `predict_output_params`). It is an
   orthogonal provenance-shape concern — the `ModelCard` carries no inference config —
   and deserves its own focused slice.

## The five changes

### 1. Default registry (#11)

In `model_registry.py`, add `_DEFAULT_REGISTRY = "sleap-roots-models"` and resolve the
registry with that default:

```python
self._registry = registry or os.environ.get("SRP_WANDB_MODEL_REGISTRY", _DEFAULT_REGISTRY)
```

`WandbRegistrySource()` — no args, only `WANDB_API_KEY` in the environment — now targets
the live registry. Entity and alias already default correctly. The
`_registry_project()` "no registry configured" guard becomes effectively unreachable in
normal use; keep a minimal defensive guard (an explicit `registry=""` is still possible).

### 2. Env-var rename, hard (#11)

- `SRP_WANDB_REGISTRY` → **`SRP_WANDB_MODEL_REGISTRY`**
- `SRP_WANDB_ALIAS` → **`SRP_WANDB_MODEL_ALIAS`**

Old names are dropped entirely (not read as a fallback), matching the producer's
`SLEAP_ROOTS_MODEL_REGISTRY` / `SLEAP_ROOTS_MODEL_ALIAS` and leaving room for a future
`sleap-roots-labels` (data) registry. Updates land in `model_registry.py` (the
`os.environ.get` reads, docstrings, and the `_registry_project` error message) and the
OpenSpec/README docs (see Change 5). `SRP_WANDB_ENTITY` and `SRP_MODEL_CACHE_DIR` are
unchanged (already correctly scoped). Archived OpenSpec changes and historical design
docs under `docs/superpowers/specs/` are left as-is (historical record).

> **Doc home:** `CLAUDE.md` is being retired in favor of OpenSpec best practices, so
> env-var documentation goes in `openspec/project.md` + `README.md`, **not** CLAUDE.md.
> The two stale old-name references currently in CLAUDE.md are corrected to the new
> names in passing (so the interim file isn't left pointing at dead vars), but no new
> content is added there and it is not treated as the canonical doc home.

### 3. Default source in the warm worker (#11)

In `warm_worker.py`, make `source` optional and default to the live registry:

```python
def __init__(self, source: Optional[ModelCardSource] = None, *, ...):
    self._source = source if source is not None else WandbRegistrySource()
```

Construction does zero network. The first `resolve()` / `get_predictors()` triggers
`list_cards()`, which already fail-loud raises `RuntimeError` naming `WANDB_API_KEY`
when the key is absent. Backward-compatible: every existing caller passes `source`
positionally.

### 4. Per-artifact error isolation (#12)

Split `list_cards()` into a **network part** and a **pure collector**:

- `_iter_registry_artifacts(api)` — the `artifact_collections` / `artifacts` traversal
  (network).
- `_collect_cards(artifacts)` — applies the alias filter, then wraps
  `_card_from_artifact` in `try/except Exception`; on failure it logs
  `logger.warning("Skipping non-conforming model artifact %r: %s", label, err)` and
  continues. One bad artifact never aborts the list.

Adds a module logger to `model_registry.py` (`logging.getLogger(__name__)`, matching the
repo convention already used in `predict.py`/`video_utils.py`). "Malformed" covers
metadata missing required selection fields (pydantic `ValidationError`) and an invalid
age window (`ValueError`) — both caught. This split is also the offline test seam.

### 5. Env-var setup: reference docs + `.env.example` (#11)

After this change the only *required* variable is `WANDB_API_KEY`; everything else has a
sensible default. To make that discoverable:

- **Reference table** — a required-vs-optional + default table for the full env surface,
  added as a new **"Configuration"** section in the operator-facing `README.md`, and
  mirrored in `openspec/project.md` (the dev/ops conventions home). `CLAUDE.md` is being
  retired in favor of OpenSpec best practices, so it is **not** the doc home — its two
  stale old-name references are just corrected to the new names in passing. Surface:

  | Variable | Required? | Default |
  |---|---|---|
  | `WANDB_API_KEY` | **yes** | — (fail-loud if missing) |
  | `SRP_WANDB_ENTITY` | no | `eberrigan-salk-institute-for-biological-studies` |
  | `SRP_WANDB_MODEL_REGISTRY` | no | `sleap-roots-models` |
  | `SRP_WANDB_MODEL_ALIAS` | no | `production` |
  | `SRP_MODEL_CACHE_DIR` | no | falls back to `WANDB_CACHE_DIR`, then wandb default |
  | `SRP_DEVICE` | no | auto-detect |

- **`.env.example`** — a committed template: `WANDB_API_KEY=` uncommented (the one thing
  to fill in) and every optional var commented out with its default shown, so local dev
  is copy-`.env`-and-go. Not auto-loaded (no `python-dotenv`; the service uses real
  k8s env/secrets) — it is documentation-by-example.
- A light consistency test guards the rename: `.env.example` mentions the new
  `SRP_WANDB_MODEL_REGISTRY` / `SRP_WANDB_MODEL_ALIAS` and **not** the old
  `SRP_WANDB_REGISTRY` / `SRP_WANDB_ALIAS`.

## Failure posture

- Missing `WANDB_API_KEY` → **fail loud** before any network call; **no** silent
  `LocalCardSource` fallback.
- The default source **tolerates registry growth** — no card count is hardcoded
  anywhere (arabidopsis `plate` and others will be added later; the default source must
  keep working as the set grows).

## Testing (TDD, minimal mocking)

**Offline unit tests** (default suite, no network):

- Default registry: `WandbRegistrySource()` → `._registry == "sleap-roots-models"`.
- Env rename: `SRP_WANDB_MODEL_REGISTRY` set → honored; old `SRP_WANDB_REGISTRY` set →
  **ignored** (proves the hard rename).
- Default source: `WarmModelWorker()._source` is a `WandbRegistrySource`; no network on
  construction.
- Fail-loud: `WarmModelWorker()` with no `WANDB_API_KEY` → `resolve(params)` raises
  `RuntimeError` naming `WANDB_API_KEY` (construction does not raise).
- Skip-and-warn: `_collect_cards([good_fake, malformed_fake])` returns `[good_card]`,
  emits a `WARNING` (asserted via `caplog`) naming the bad artifact, and does not raise.
  Uses simple duck-typed fake artifact objects (data holders with
  `.metadata`/`.aliases`/`.version`/`.digest`/`.qualified_name`), **not** wandb-boundary
  mocks.

**Gated `@pytest.mark.wandb`** (live registry, only `WANDB_API_KEY`):

- Existing `test_wandb_source_lists_and_materializes` stays green; add
  `monkeypatch.delenv("SRP_WANDB_MODEL_REGISTRY", raising=False)` to prove the default
  path needs no registry env. Assertions stay **count-agnostic** (tolerate the growing
  card set — do not assert exactly 13).

## OpenSpec impact

One capability: **`model-management`**. Deltas:

- MODIFY "Wandb Registry Source With Version Pinning" — default registry, renamed env
  vars, and a skip-and-warn scenario.
- MODIFY "Warm Model Residency" — optional `source` defaulting to the live registry,
  with a fail-loud-on-missing-key scenario.

## Out of scope (linked, not done here)

- #7 — reshape config accessors to `predict_inference_config` / `predict_output_params`.
- #14 — dedupe warm cache by `weights_checksum`.
- #13 — pin a wandb version floor (hygiene; no actual skew).
- #10 — `sleap_nn_version` mismatch warning (dormant).
- The A3-predict parity gate (tolerance + reference scan set).

## Acceptance

- A fresh `WandbRegistrySource` / `WarmModelWorker` with only `WANDB_API_KEY` set (no
  `SRP_WANDB_MODEL_REGISTRY`) fetches the production models from the live registry.
- A malformed/non-conforming production artifact is skipped-with-warning, not fatal, in
  `list_cards()`.
- `uv run pytest -m wandb -q` stays green against the live registry.
- Closes predict #11 + #12; report back so training #6 and the pipeline roadmap
  A3-predict row advance.
