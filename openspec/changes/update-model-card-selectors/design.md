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
   where alias-matching artifacts exist and none validates. Together with loading the catalog once
   before the per-scan loop (it was loaded lazily inside the per-scan `try`, so a raise there
   would have been isolated per scan), this turns a premature deploy into exit `1`.
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
  in the 2026-08-04 run the age tie-breaker (tier 3) ran only for five single-species cards (rice
  and soybean, five distinct checksums), each of which becomes a single-selector card after the
  re-seed; every card sharing weights resolved at tier 2, where the age step never runs.
- **One report entry per (card, selector)** — rejected: duplicates one model's metrics.
- **Rely on deploy ordering alone** — rejected: a premature deploy would exit `3` with every scan
  failed, and the pipeline's exit gate passes `3` by design.

## Risks

- **Premature deploy** — now fails loud at startup (Decision 3) as well as being prevented by
  ordering.
- **Real-scan coverage is canola only** (`a4_poc` `scan_289`/`577`/`1009`); A1 covers every
  species' selection, A2 covers inference for canola only.
