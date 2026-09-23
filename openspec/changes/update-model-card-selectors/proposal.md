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
- **New fail-loud guard:** `WandbRegistrySource.list_cards()` raises when production artifacts
  exist but **none** validates, and `run_batch` loads the catalog once before its per-scan loop,
  so a premature deploy exits `1` at startup instead of `3` with every scan failed (which the
  pipeline's exit gate passes). Mixed skips still continue, as the canary requires. This is the
  one deliberate exception to #34's fact 1.
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
  Source With Version Pinning), `predict-container` (MODIFIED: Per-scan failure isolation and batch
  exit code), `prediction-parity` (MODIFIED: Ground Truth Resolution Per Model, Basename Search
  Disambiguation, LabelCard-Shaped Ground Truth Manifest; ADDED: Parity Report Entry Selection
  Fields).
- Affected code: `pyproject.toml`, `uv.lock`, `sleap_roots_predict/{model_selection,model_registry,
  batch,parity}.py`, `scripts/run_parity_harness.py`, new `scripts/a1_selection_oracle.py` and
  `scripts/a2_run_local.py`; tests across `conftest.py`, `test_batch.py`, `test_model_registry.py`,
  `test_model_selection.py`, `test_output_contract.py`, `test_param_resolution.py`,
  `test_parity.py`, `test_warm_worker.py`.
- Affected docs: `openspec/project.md`, `API.md`, `README.md`, `CLAUDE.md` (one parenthetical),
  `CHANGELOG.md`.
- **Merge timing:** the PR merges only after the live canary (B) passes from the branch, so
  `main` stays deployable until then. After merge, `:latest`/`:main` read no production card until
  the full re-seed; the pipeline pins by sha + digest, so nothing auto-deploys.
- **Deploy ordering (other repos):** the predictor pin bump waits for the re-seed (training
  6.0–6.2); retiring the flat collections (training 6.3) waits for this change to be confirmed
  **deployed**.
- **One-time full recompute and `.slp` rename** — see design §4.

## Rollback

- **Merged, not deployed:** revert the squash commit; the pipeline is unaffected.
- **Deployed, flat collections not yet retired (before training 6.3):** re-pin the predictor to
  the previous image (`sha-e025e30…`); it still reads the 13 flat collections. Costs a second full
  recompute and `.slp` rename, since `registry_id` flips back.
- **After training 6.3:** an image rollback alone yields the guard's startup error (no readable
  card). Rollback then requires restoring `production` on the flat collections from training's
  6.0(a) snapshot — which is why 6.3 is gated on confirmed deployment.

## Findings recorded, not fixed here

- `build_label_card` copies the **model** card's `registry_id`/`version` into `LabelCard`
  fields that identify the **label** artifact (`parity.py:924-925`) — a fabricated value by that
  requirement's own rule. Only tests call it; to be filed as its own issue.
- A generalist model bundle's `labels_gt.val.slp` may mix species, contrary to the one-species
  label rule — training#46 / #11.
