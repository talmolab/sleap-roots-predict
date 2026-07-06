## Why

Fetching production models is opt-in today: `WandbRegistrySource` only works when the
operator sets `SRP_WANDB_REGISTRY` (no default, generic name), and `WarmModelWorker`
requires an explicit `source`. The production registry is now seeded, verified, and
finalized (`pytest -m wandb` is green against `sleap-roots-models` / alias `production`),
so reading it should be the **default** — the warm worker should fetch production models
out-of-the-box with only `WANDB_API_KEY` set. Closes predict **#11** (default source +
env rename) and **#12** (per-artifact error isolation); advances roadmap **A3-predict**.

## What Changes

- Default `WandbRegistrySource`'s registry to `sleap-roots-models` (the live production
  registry), so a source constructed with only `WANDB_API_KEY` set reads production.
- Make `WarmModelWorker(source=None)` default to a `WandbRegistrySource` reading the live
  registry — **fail-loud** on a missing `WANDB_API_KEY`, **no** offline `LocalCardSource`
  fallback. Construction stays network-free; the failure surfaces on first use.
- **BREAKING**: hard-rename the env vars `SRP_WANDB_REGISTRY` → `SRP_WANDB_MODEL_REGISTRY`
  and `SRP_WANDB_ALIAS` → `SRP_WANDB_MODEL_ALIAS` (old names no longer read), matching the
  producer's `SLEAP_ROOTS_MODEL_REGISTRY` / `SLEAP_ROOTS_MODEL_ALIAS`. Alpha package;
  old names had no default and were referenced only by one gated test + docs.
- Isolate per-artifact errors in `list_cards()`: a non-conforming production artifact is
  skipped with a logged warning naming it, not fatal — one bad artifact never aborts the
  listing.
- Add env-var setup docs (a required-vs-optional + defaults table in `README.md`,
  mirrored in `openspec/project.md`) and a committed `.env.example` template.

## Impact

- **Affected specs:** `model-management` — MODIFY "Wandb Registry Source With Version
  Pinning" (default registry + renamed env vars + skip-and-warn); MODIFY "Warm Model
  Residency" (optional `source` defaulting to the live registry, fail-loud).
- **Affected code:** `sleap_roots_predict/model_registry.py`,
  `sleap_roots_predict/warm_worker.py`; new `.env.example`; docs `README.md` (new
  "Configuration" section — the canonical env table), `openspec/project.md` (pointer to it),
  `CHANGELOG.md` (`[Unreleased]` BREAKING entry), and an in-passing single old-name
  correction in `CLAUDE.md` (line 71; being retired in favor of OpenSpec docs — not a doc
  home); tests `tests/test_model_registry.py`, `tests/test_warm_worker.py`.
- **Ripples out (tracked elsewhere):** training **#6** (update its README/design env
  refs), pipeline roadmap row **A3-predict**.
- **Explicitly deferred (linked, not here):** #7 (reshape config accessors to
  `predict_inference_config`/`predict_output_params`), #14 (dedupe warm cache by
  `weights_checksum`), #13 (wandb version floor), #10 (`sleap_nn_version` mismatch), the
  A3-predict parity gate.
