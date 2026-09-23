# Migrate to `ModelCard.selectors` (predict#34) — design

Date: 2026-09-23 · Branch: `migrate-model-card-selectors` · Issue: talmolab/sleap-roots-predict#34

## 1. What and why

`sleap-roots-contracts` `0.1.0a8` replaced `ModelCard.species/mode/age_min/age_max` with
`ModelCard.selectors: tuple[Selector, ...]` — one card per physical model, each selector one whole
validated (species, mode, age window) context. It has **no tolerant read** of the flat shape. This
change bumps the pin `==0.1.0a7` → `==0.1.0a9` and migrates every reader of the flat fields.

The live registry is 13 flat / 0 selector-shaped (verified 2026-09-22). A deployed upgrade today
would validate zero cards, and because `WandbRegistrySource.list_cards` skips an unvalidatable card
with a warning (predict#32), the failure is an **empty catalog** — "cannot select a model", never an
error naming the registry. Merging is ungated; **deploying is gated on the W&B re-seed**
(sleap-roots-training `update-model-card-selectors` group 6), and the re-seed's canary (6.1) is in
turn gated on this code existing.

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

### 3.2 Registry listing (`model_registry.py`)

No code change. A test pins #34 fact 1 against the a9 contract: a flat-shaped artifact under the
`production` alias is skipped with a warning naming it, and the listing continues with the rest.

### 3.3 Parity harness (`parity.py`)

A shared private helper resolves "which selector": the given one (must be in `card.selectors`,
else `ValueError`), else the card's only selector, else `None`.

- `build_label_card(..., selector=None)`: `None` from the helper on a multi-selector card raises
  `ValueError` naming the card.
- `resolve_ground_truth(..., selector=None)` → `relink_ground_truth_by_basename_search` →
  `_pick_best_candidate`: the age step uses the resolved selector's window, and is skipped when
  it is `None`.
- `run_parity_harness` passes no selector.
- `build_report_entry` emits `selectors` as a list of dicts in card order.
- The `labels_registry_lookup` callable is caller-injected and receives the card; its contract
  wording moves from "the card's species" to "one of the card's selectors' species".

### 3.4 Specs

- `model-management`: MODIFIED model-selection requirement (any-selector, per-selector age, no
  cross product; ambiguity raise unchanged).
- `prediction-parity`: MODIFIED Ground Truth Resolution (lookup join), Basename Search
  Disambiguation (selector window / skip), LabelCard-Shaped Ground Truth Manifest (explicit
  selector), Reusable Multi-Model Harness Runner (entry shape).

## 4. Idempotency and other `registry_id`-keyed state

`registry_id` changes for all 8 models under the producer's new collection-id scheme, and
`compute_idempotency_key` hashes `(registry_id, version, weights_checksum)`, so every key changes
once: the first post-migration run recomputes every scan (predict's own skip-if-done compares the
same tuples, `batch.py:244`). Expected, not a regression. The A4 batch oracle ("re-run a done
batch → 0 GPU pods") is re-baselined **after** the migration, never compared across it.

Other `registry_id`-keyed state in this repo (grepped 2026-09-23):

- **Per-root `.slp` filenames** embed `slugify_model_id(ref)` = `registry_id` + `version`
  (`output_contract.py:66`), so every output file is renamed once. The writer already removes a
  prior `.slp` left by a changed model slug, and only after every new file and the manifest are
  written (`output_contract.py:257-263`), so no orphans remain and a failed write never deletes a
  still-valid prior artifact. Traits reads the manifest's explicit paths, not a glob.
- **Warm-worker predictor cache** `(registry_id, version)` (`warm_worker.py:118`) and
  **`LocalCardSource`'s path map** (`model_registry.py:65`) are in-process only; nothing persists.

## 5. Validation against real models and the real pipeline

Unit tests cannot see the dominant failure (a silent empty catalog), so validation is staged by
risk. A1 and A2 gate the merge; B and C are post-merge gates recorded in `tasks.md`,
unticked until actually run.

**A1 — selection-equivalence oracle (offline, read-only).** Over species × mode × age 0–20 × root
type: the *old* side is `main`'s flat `choose_models` over the 13 live flat cards (read-only
registry listing); the *new* side is this branch's `choose_models` over the 8 selector cards
training's card builder would write (built in-process, no `--execute`). Pass iff every cell selects
the same `weights_checksum`, the same cells skip, and nothing raises. `registry_id` is expected to
differ. This is the only check that exercises the canola-2–13 / pennycress-2–14 boundary on real
card data.

**A2 — real-inference equivalence (offline).** Copy `scan_289`, `scan_577`, `scan_1009` (real
canola cylinder, ages 2/7/9, 72 frames) from
`Z:\users\eberrigan\pipeline_orchestration_tests\a4_poc\input` to scratch; run
`python -m sleap_roots_predict` from `main` (flat `LocalCardSource`) and from this branch (selector
`LocalCardSource`) over the same real weights from the models-downloader snapshot, same device.
Pass iff, per scan and **per root type** (filenames differ by model slug, §4), predictions are
numerically identical and the manifests differ only in model identity fields. Never
write into `a4_poc/predictions` — it is production's working tree.

**B — live canary (training 6.1).** `seed-registry --execute --only <one collection>` —
irreversible and single-operator, **run only on explicit user confirmation**. Then: this branch
against the live registry resolves the new collection's `registry_id` without raising; `main`
still resolves the old flat card; the expected skip warnings appear on both sides.

**C — full re-seed, deploy, real run (training 6.2 → pipeline 0c).** Re-run A1 with the new side
read from the live selector registry. Bump the predictor pin; run a small real Argo batch; assert
selection resolves for every scan, exit 0, predictions match the pre-migration outputs, and the
one-time recompute occurs. Re-run the parity harness (`-m parity`) against the re-seeded registry.
Only then is training 6.3 (retiring the flat collections) eligible.

**Coverage gap.** Real staged scans are canola only; rice `scan_6791737`'s input is gone, and no
pennycress/arabidopsis/soybean scans are staged in `a4_poc`. A1 covers every species' selection;
A2 covers inference for canola only.

## 6. Out of scope

- Run-manifest adoption (`load_run_manifest`, per-run forward naming) — follow-up PR.
- The predictor pin bump in `sleap-roots-pipeline` (deploy) — after the re-seed.
- A per-card selector mapping for `run_parity_harness` — until someone needs it.
- A generalist bundle's `labels_gt.val.slp` possibly mixing species, against the one-species label
  rule — training#46 / #11.
- The correcting comment owed on predict#40 — drafted separately for the user to post.
