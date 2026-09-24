# Change: Migrate to `ModelCard.selectors` (contracts `0.1.0a9`)

Part of talmolab/sleap-roots-predict#34 — the consumer half of talmolab/sleap-roots-training#39.
Design of record: [`docs/superpowers/specs/2026-09-23-model-card-selectors-design.md`](../../../docs/superpowers/specs/2026-09-23-model-card-selectors-design.md).

## Why

Contracts `0.1.0a8` replaced the flat `ModelCard.species/mode/age_min/age_max` with
`ModelCard.selectors`, so one card describes one physical model; predict pins `0.1.0a7` and reads
the flat fields on its runtime selection path, so it cannot read the re-seeded registry.

## What Changes

- **BREAKING (dependency):** pin `sleap-roots-contracts==0.1.0a9`. The a9 run-manifest API is
  **not** adopted here; that is a separate follow-up change (pipeline design
  `2026-09-21-per-run-run-manifest-identity-design.md` §2.6, §4 step 0c).
- `choose_models` matches a card iff **some single selector** matches species, mode and age, the
  age compared against **that selector's** window — never a card-level window, never the cross
  product. The collect-then-raise ambiguity error is **unchanged**.
- **New fail-loud guard:** `WandbRegistrySource.list_cards()` raises a `ValueError` when
  production artifacts exist but **none** validates. This is the one deliberate exception to #34's
  fact 1, and it reverses #32's pinned "all-malformed listing is empty, not an exception". Mixed
  skips still continue, as the canary requires.
- **Catalog loaded once per batch, before the first processable scan** (new `WarmModelWorker.load_catalog()`),
  so catalog failures are batch-level. **Exit-code change:** missing `WANDB_API_KEY`, a
  registry/network error while listing, and an unreadable catalog now exit `1` (retried by Argo)
  instead of `3` with every scan failed (passed by the exit gate) — which also makes the existing
  spec text "model-registry authentication failing before any scan is attempted" → `1` true for
  the first time. Not loaded when a stop is requested before the first scan, or when every scan
  already has a discovery error, so those paths are unchanged.
- **What the guard does not cover:** a catalog that is readable but incomplete. Between the canary
  and the full re-seed, the one canary card defeats the guard, and a deploy then would fail scans
  outside the canary's contexts with exit `3`. Only deploy ordering (the pin bump waits for
  training 6.2) protects that window.
- Parity harness:
  - **BREAKING** for report readers: entries carry `selectors: [{species, mode, age_min,
    age_max}, …]` in place of the four flat fields;
  - `build_label_card` takes an explicit `selector=`, required for a multi-selector card; the
    LabelCard requirement is **relaxed**: it no longer claims a checked-in manifest (none exists)
    and no longer mandates one record per model card, but one per (card, labeling package);
  - ground-truth resolution validates a supplied selector before any tier; the basename age
    tie-breaker uses the supplied or sole selector's window and is skipped when neither exists;
  - `scripts/run_parity_harness.py` no longer defaults to overwriting the committed
    2026-08-04 report.
- Validation against real models and the live registry (A1, A2, canary B) gates the merge;
  deploy-time gates (C1–C3) are tracked on #34.

## Impact

- Affected specs: `model-management` (MODIFIED: Model Selection From Scan Params; Wandb Registry
  Source With Version Pinning; Warm Model Residency), `predict-container` (MODIFIED: Per-scan failure isolation and batch
  exit code), `prediction-parity` (MODIFIED: Ground Truth Resolution Per Model, Basename Search
  Disambiguation, LabelCard-Shaped Ground Truth Manifest; ADDED: Parity Report Entry Selection
  Fields).
- Affected code: `pyproject.toml`, `uv.lock`, `sleap_roots_predict/{model_selection,model_registry,
  warm_worker,batch,__main__,parity}.py`, `scripts/run_parity_harness.py`; new
  `scripts/{a1_selection_oracle,a1_compare,a2_run_local}.py` and a canary check script; new
  `tests/card_builders.py`; tests across `conftest.py`, `test_batch.py`, `test_model_registry.py`,
  `test_model_selection.py`, `test_output_contract.py`, `test_param_resolution.py`,
  `test_parity.py`, `test_warm_worker.py`.
- Cross-repo text to correct (drafted for approval, task 6.7): the pipeline design of record says
  #34 merges ungated and pins a8.
- Affected docs: `openspec/project.md`, `API.md`, `README.md`, `CLAUDE.md` (one parenthetical),
  `CHANGELOG.md`.
- **Merge timing:** the PR merges only after the live canary (B) passes from the branch, so
  `main` stays deployable until then. After merge, `:latest`/`:main` read only the canary's card
  until the full re-seed; the pipeline pins by sha + digest, so nothing auto-deploys.
- **Deploy ordering (other repos):** the predictor pin bump waits for the re-seed (training
  6.0–6.2); retiring the flat collections (training 6.3) waits for this change to be confirmed
  **deployed**.
- **One-time full recompute and `.slp` rename** — see design §4.

## Rollback

- **Merged, not deployed:** revert the squash commit; the pipeline is unaffected.
- **Deployed, flat collections not yet retired (before training 6.3):** re-pin the predictor to
  the previous image — tag, digest and `SRP_PREDICT_CONTAINER_DIGEST` together
  (`ghcr.io/talmolab/sleap-roots-predict:sha-e025e309…@sha256:4d4064c6…`,
  `sleap-roots-predictor-template.yaml:80,107`; `check_manifests.py` fails if they drift). It still
  reads the 13 flat collections. Costs a second full recompute and `.slp` rename, since
  `registry_id` flips back.
- **After training 6.3:** an image rollback alone is **silent** — the previous image predates the
  guard, so it skips all 8 selector cards, fails every scan and exits `3`, which the exit gate
  passes. Restore `production` on the flat collections from training's 6.0(a) snapshot (the
  procedure rehearsed in training 6.0(e)) before or together with the re-pin. This is why training
  6.3 is gated on confirmed deployment.

## Findings recorded, not fixed here

- `build_label_card` copies the **model** card's `registry_id`/`version` into `LabelCard`
  fields that identify the **label** artifact (`parity.py:924-925`) — a fabricated value by that
  requirement's own rule. Only tests call it; to be filed as its own issue.
- A generalist model bundle's `labels_gt.val.slp` may mix species, contrary to the one-species
  label rule — training#46 / #11.
