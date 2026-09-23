# Migrate to `ModelCard.selectors` (predict#34) — design

Date: 2026-09-23 · Branch: `migrate-model-card-selectors` · Issue: talmolab/sleap-roots-predict#34

## 1. What and why

`sleap-roots-contracts` `0.1.0a8` replaced `ModelCard.species/mode/age_min/age_max` with
`ModelCard.selectors: tuple[Selector, ...]` — one card per physical model, each selector one whole
validated (species, mode, age window) context. It has **no tolerant read** of the flat shape. This
change bumps the pin `==0.1.0a7` → `==0.1.0a9` and migrates every reader of the flat fields.

The live registry is 13 flat / 0 selector-shaped (verified 2026-09-22). A deployed upgrade today
would validate zero cards, and because `WandbRegistrySource.list_cards` skips an unvalidatable card
with a warning (predict#32), the failure is an **empty catalog** — every scan fails with
`no models resolved for params …` (`batch.py:418`), never an error naming the registry (§3.2 adds a
guard that turns the zero-readable-card case into a startup error). **Deploying is
gated on the W&B re-seed** (sleap-roots-training `update-model-card-selectors` group 6), and the
re-seed's canary (6.1) is in turn gated on this code existing. By decision, the PR itself merges
only after that canary passes from this branch, so `main` stays deployable in the meantime.

## 2. Scope decisions

| Decision | Choice | Reason |
|---|---|---|
| Adopt the a9 run-manifest API here? | **No** — separate follow-up PR | Design of record `sleap-roots-pipeline/docs/superpowers/specs/2026-09-21-per-run-run-manifest-identity-design.md` §2.6 and §4 step 0c: the #34 deploy must answer one falsifiable question, since an empty catalog is silent. The pin bump makes the API *available*; no call site opts in. |
| Relax the ambiguity raise? | **No** | #34 fact 2 — the producer's additive migration depends on it never firing. A `weights_checksum` dedupe remains the recorded alternative, as a deliberate later decision. |
| Parity report entry shape | `selectors: [{species, mode, age_min, age_max}, …]` replaces the four flat fields | Nothing upstream decides it (checked: contracts archive, training change, roadmap, predict issues). Mirrors the card; one entry per card so shared weights are not duplicated further. The committed `2026-08-04-define-parity-tolerance-results.json` stays as a historical snapshot in the old shape. |
| `build_label_card` on a multi-selector card | explicit `selector=` argument | A labeling package is one species (structural in the contracts: `LabelCard.species` is scalar; training's labeling-package generator is keyed on `(species, root_type)`; training#46 links a model to "the `LabelCard`(s)"). The card cannot say which of its selectors a package is. |
| Parity basename-search age tie-breaker | the explicit/sole selector's window; **skip** the age step when no selector is known | The age window is per species; the harness does not know a ground-truth video's species from a multi-selector card. Skipping falls through to path-segment scoring, and a remaining tie is already an explicit non-match — never a guess. |

## 3. Design

### 3.1 Model selection — runtime path (`model_selection.py`)

A card matches when **some single selector** matches species, mode **and** age together, the age
compared against **that selector's** window:

```python
def _card_matches(card, species, mode, age):
    return any(
        s.species == species and s.mode == mode and s.age_min <= age <= s.age_max
        for s in card.selectors
    )
```

Matching stays per card: two overlapping selectors on one card that both match count as one match.
The collect-then-raise structure (`if len(matches) > 1: raise ValueError("Ambiguous model
selection...")`), override-wins and zero-match-skips are unchanged.

### 3.2 Registry listing (`model_registry.py`) and batch catalog load (`batch.py`)

Per-card isolation stays (#34 fact 1), and a characterization test pins it against the a9
contract: a flat-shaped artifact under `production` is skipped with a warning naming it while the
selector-shaped ones are returned. **One deliberate exception, added after review:** when
alias-matching artifacts exist and **none** validates, `list_cards()` raises. Review traced why
ordering alone is not enough — every card skipped → every scan raises `no models resolved`
(`batch.py:414-418`) → `main()` returns `3` even at 100% failure (`__main__.py:113`) → the
pipeline's exit gate passes `3` by design (`sleap-roots-exit-gate-template.yaml:132-135,141`). Because
`WarmModelWorker.resolve` loads the catalog lazily inside `run_batch`'s per-scan `try`
(`warm_worker.py:85-86`, `batch.py:370-396`), `run_batch` also loads it once, just before the first processable scan's `try`, so
the raise is a batch-level staging error (exit `1`), not one isolated failure per scan. It loads
through a new public `WarmModelWorker.load_catalog()`, skipped when a stop is already requested or
no scan is processable. Side effect, recorded as a change: missing credentials and registry/network
errors also move from exit `3` to `1`. Limit: the guard sees only a catalog with **zero** readable
cards; between the canary and the full re-seed, one readable card defeats it and only deploy
ordering protects.

### 3.3 Parity harness (`parity.py`)

A shared private helper resolves "which selector": the given one (validated by value equality
against `card.selectors` before any work, else `ValueError`), else the card's only selector, else
`None`.

- `build_label_card(..., selector=None)`: `None` on a multi-selector card raises `ValueError`
  naming the card; the unnamed-skeleton fallback uses the resolved selector's species.
- `resolve_ground_truth(..., selector=None)` → `relink_ground_truth_by_basename_search` →
  `_pick_best_candidate`: the age step uses the resolved selector's window, and is skipped when
  it is `None`.
- `run_parity_harness` passes no selector.
- `build_report_entry` emits `selectors` as a list of dicts in card order.
- The `labels_registry_lookup` callable is caller-injected and receives the card; its join
  criterion is the caller's.
- `scripts/run_parity_harness.py` stops defaulting `--out` to the committed 2026-08-04 report,
  which a default re-run would otherwise overwrite in the new shape.

### 3.4 Specs

- `model-management`: MODIFIED Model Selection From Scan Params (any-selector, per-selector age,
  no cross product; ambiguity raise unchanged); MODIFIED Wandb Registry Source With Version
  Pinning (the all-invalid guard).
- `predict-container`: MODIFIED Per-scan failure isolation and batch exit code (catalog loaded once
  before the first processable scan; no readable card → exit `1`).
- `prediction-parity`: MODIFIED Ground Truth Resolution Per Model, Basename Search Disambiguation,
  LabelCard-Shaped Ground Truth Manifest; ADDED Parity Report Entry Selection Fields (no existing
  requirement lists the entry fields, so the Runner requirement is left unchanged).

## 4. Idempotency and other `registry_id`-keyed state

`registry_id` changes for all 8 models under the producer's new collection-id scheme, and
`compute_idempotency_key` hashes `(registry_id, version, weights_checksum)`, so every key changes
once: the first post-migration run recomputes every scan (predict's skip-if-done builds the same
tuples at `batch.py:244` and compares them at `batch.py:380-383`). Expected, not a regression. The
A4 batch oracle ("re-run a done batch → 0 GPU pods") is re-baselined **after** the migration,
never compared across it.

Other `registry_id`-keyed state in this repo (grepped 2026-09-23):

- **Per-root `.slp` filenames** embed `slugify_model_id(ref)` = `registry_id` + `version`
  (`output_contract.py:66`), so every output file is renamed once. The writer removes a prior
  `.slp` left by a changed model slug only after every new file and the manifest are written
  (`output_contract.py:245-263`), so a failed write never deletes a still-valid prior
  artifact; a failed sweep can leave inert clutter, never a wrong manifest.
- **Warm-worker predictor cache** `(registry_id, version)` (`warm_worker.py:118`) and
  **`LocalCardSource`'s path map** (`model_registry.py:65`) are in-process only; nothing persists.

## 5. Validation against real models and the live registry

Unit tests cannot see the dominant failure — a registry this code cannot read — so validation is
staged by risk. The procedure is single-sourced in the OpenSpec change's `tasks.md` §6–§7; this
section records only why each stage exists.

- **A1, selection-equivalence oracle** (merge gate): the only check that exercises the real cards'
  species/mode/age boundaries — canola 2–13 against pennycress 2–14 on one card — before anything
  irreversible happens. It runs as three environments exchanging JSON, because training pins
  contracts a8 and `sleap-nn<0.3.0`, and the old side needs contracts a7.
- **A2, real-inference equivalence** (merge gate): same weights, so predictions must not change;
  run on copies of real canola scans (`a4_poc` `scan_289`/`577`/`1009`), never in `a4_poc`, which
  is production's working tree.
- **B, live canary** (merge gate, by decision: the PR merges only after it passes, so `main` stays
  deployable until then): the first check against live selector-shaped data, and the only one
  that can prove both consumer generations are cleanly partitioned.
- **C1–C3** (tracked on #34): A1 re-run against the live selector registry after the full
  re-seed, the deploy with a real Argo run, and a parity re-run to a new report path.

**Coverage gap.** Real staged scans are canola only; rice `scan_6791737`'s input is gone, and no
pennycress/arabidopsis/soybean scans are staged in `a4_poc`. A1 covers every species' selection;
A2 covers inference for canola only.

## 6. Out of scope

- Run-manifest adoption (`load_run_manifest`, per-run forward naming) — follow-up PR.
- The predictor pin bump in `sleap-roots-pipeline` (deploy) — after the re-seed.
- A per-card selector mapping for `run_parity_harness` — until someone needs it.
- A generalist bundle's `labels_gt.val.slp` possibly mixing species, against the one-species label
  rule — training#46 / #11.
- The correcting comment on predict#40 — already posted 2026-09-23 (not by this change).
