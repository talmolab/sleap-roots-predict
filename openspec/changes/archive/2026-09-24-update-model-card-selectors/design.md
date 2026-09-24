# Design: Migrate to `ModelCard.selectors`

The design of record, with the evidence behind each decision, is
[`docs/superpowers/specs/2026-09-23-model-card-selectors-design.md`](../../../docs/superpowers/specs/2026-09-23-model-card-selectors-design.md).
This file records only the decisions that shape the spec deltas; the validation procedure lives in
`tasks.md` §6–§7.

## Context

- Contracts `0.1.0a8` `Selector` is frozen, `extra="ignore"`, with `species: str`, `mode: Mode`,
  inclusive `age_min`/`age_max`. `ModelCard.selectors` is a non-empty tuple. There is no tolerant
  read of the flat card shape. a9 adds only the run-manifest API; `RunManifest`,
  `PredictionManifest`, `compute_idempotency_key` and `resolve_params` are unchanged.
- #34 records two facts the producer's migration depends on: `list_cards` filters on the
  `production` alias before validating and isolates failures per card; `choose_models` raises on
  an ambiguous match.

## Decisions

1. **Any-selector, per card.** A card matches iff some single selector matches all of species,
   mode and age. Overlapping selectors on one card count as one matching card.
2. **Ambiguity raise unchanged.** A `weights_checksum` dedupe stays the recorded alternative, to be
   taken only as a deliberate later decision.
3. **All-invalid catalog fails loud.** Per-card isolation stays; the one exception is a listing
   where alias-matching artifacts exist and none validates (a `ValueError`, so the CLI's one-line
   staging log names it). The catalog was loaded lazily inside `run_batch`'s per-scan `try`, where
   a raise would be isolated per scan, so a public `WarmModelWorker.load_catalog()` now runs once
   before the first processable scan (after that iteration's stop check, adding none) — skipped when a stop is already requested or no
   scan is processable. A deploy against a registry with **zero** readable cards therefore exits
   `1`. It cannot detect a readable but **incomplete** catalog (between the canary and the full
   re-seed); deploy ordering covers that window.
4. **Run-manifest adoption deferred** to its own change, so this deploy answers one question.
5. **One selector rule for parity.** The supplied selector (validated by value equality before any
   work), else the card's only selector, else none. `build_label_card` raises on none; the
   basename age step skips on none.
6. **Report entry shape** mirrors the card: a `selectors` list, one entry per card.

## Alternatives considered

- **Bundle the run-manifest adoption** — rejected: one image and one deploy with two possible
  causes of failure, and the manifest fix would wait behind the re-seed.
- **Card-level age envelope for the tie-breaker** — rejected: it is the card-level window #34
  warns against; an age window belongs to one species.
- **Raise on a multi-selector card in ground-truth resolution** — rejected as needlessly strict:
  in the 2026-08-04 run, tier 3 (the only tier where the age step can run) resolved only five
  single-species cards (rice and soybean, five distinct checksums), each of which becomes a
  single-selector card after the re-seed; every card sharing weights resolved at tier 2.
- **One report entry per (card, selector)** — rejected: duplicates one model's metrics.
- **Rely on deploy ordering alone** — rejected: a deploy against a registry with no readable card
  would exit `3` with every scan failed, and the pipeline's exit gate passes `3` by design.

## Risks

- **Premature deploy** — against a registry with no readable card it fails loud at startup
  (Decision 3); against a partly re-seeded registry only deploy ordering prevents a silent `3`.
- **A guard exit `1` under Argo** retries (`limit: 3`, about 14 minutes) and then fails the node;
  `continueOn` carries the DAG on, and write-back may already have delivered earlier scans'
  results before the Workflow goes red — as for any predict exit `1` today.
- **Coverage** — see the design of record §5.
