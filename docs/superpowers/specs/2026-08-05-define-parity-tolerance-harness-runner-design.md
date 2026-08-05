# Design: reusable parity harness runner + results-schema docs (define-parity-tolerance)

- **Date:** 2026-08-05
- **Repo:** `sleap-roots-predict`
- **Branch:** `define-parity-tolerance` (same branch/PR as the parent change — see
  [#33](https://github.com/talmolab/sleap-roots-predict/pull/33))
- **Parent design:** `2026-08-03-define-parity-tolerance-design.md` — this doc only covers the
  additions below; it does not re-derive anything already settled there.
- **Status:** APPROVED — brainstorming complete. Proceed to implementation.

## Motivation

Both staleness bugs found during PR #33's post-review rounds (task 8.7, and the earlier
`-0.053` vs. `-0.085` figure at task 8.5) trace to the same root cause: the checked-in results
JSON (`docs/superpowers/specs/2026-08-04-define-parity-tolerance-results.json`) was produced by
an **uncommitted scratch script**. Nobody can re-run the full 13-model harness today without
rebuilding that script from scratch, and every number about it that lands in prose is a
manually-copied snapshot with no way to check it's still current. This change makes
regeneration a committed, reusable operation, and documents the JSON's schema so a reader
doesn't have to reverse-engineer `build_report_entry()` to understand it.

## Settled decisions (from brainstorming 2026-08-05)

### 1. `run_parity_harness()` — a new orchestration function in `parity.py`

```python
def run_parity_harness(
    cards: Sequence[ModelCard],
    source: ModelCardSource,
    workdir: Path,
    out_path: Path,
    *,
    labels_registry_lookup=None,
    prefix_map=None,
    basename_index=None,
    sample_n=None,
) -> Path:
```

Loops `cards`, calling `source.materialize(card)` then `evaluate_model_card(...)` per card, and
persists the accumulated entries via the existing `write_parity_report()`. Plain function
arguments only — no new env vars, no config file — mirroring `evaluate_model_card`'s own
signature exactly, so it composes the same way `evaluate_model_card` composes
`resolve_ground_truth`/`compute_metrics`/etc. (task 8.3's own precedent). Unit-testable with
fixture `ModelCard`s and a fake `ModelCardSource`, no network.

**Per-card failure isolation.** Each card's `materialize` + `evaluate_model_card` call is
wrapped in a `try`/`except`; a failure is logged as a warning and turned into a gap-shaped entry
(`registry_id`/`version`/`gap_reason`) rather than aborting the run. This is the same
per-artifact isolation pattern already implemented in `model_registry.py`'s `_collect_cards`
(lines 195-230) — a single non-conforming or network-flaky model must not silently drop the rest
of a 30+-minute run. `evaluate_model_card` itself already isolates *ground-truth-resolution*
gaps (returns a `GapRecord`-derived entry rather than raising); this adds the same isolation one
level up, for failures anywhere else in the per-card pipeline (materialize, inference, metrics).

### 2. `prefix_map` is a committed script constant, not a new config mechanism

The lab-specific `prefix_map` (`D:/SLEAP` and `C:/Users/pbiobgh/Desktop/SLEAP` →
`Z:/users/eberrigan/SLEAP`) is a stable fact about this lab's data reorganization, already
committed as prose in the parent design doc — not a secret, and only two entries. Building a
JSON-config-file format and a new env var for it would be more machinery than a two-entry,
rarely-changing map needs (YAGNI). Instead, a new committed script,
`scripts/run_parity_harness.py`, hardcodes the real values as a module-level constant — this
*is* the fix for the root cause (the mapping is now committed and reusable, not living only in
an uncommitted scratch file or someone's head).

The basename-search root, by contrast, genuinely is machine-specific (depends on how the network
share happens to be mounted on whoever runs this) — that's exactly what the existing
`SRP_PARITY_DATA_DIR` env var is for (already wired into the `parity`-marked test), so no new
env var is introduced for it either.

The script:
- Reads `WANDB_API_KEY` and `SRP_PARITY_DATA_DIR` from the environment (existing convention).
- Hardcodes `PREFIX_MAP` (the real, already-public two entries).
- Calls `WandbRegistrySource().list_cards()` and `build_basename_index(SRP_PARITY_DATA_DIR)`.
- Calls `run_parity_harness(..., sample_n=100)` — `100` matches the original empirical run
  (task 5.2), kept as the script's own default; `run_parity_harness` itself defaults `sample_n`
  to `None` (no magic number baked into the reusable library function).
- Defaults `out_path` to the existing canonical
  `docs/superpowers/specs/2026-08-04-define-parity-tolerance-results.json` path, so a future
  regeneration overwrites the one file every doc already points at rather than producing a new
  dated file each run (overridable via a CLI arg for anyone who wants a dated snapshot instead).

### 3. Regeneration stays manual/on-demand — no CI wiring

This hits a live external wandb registry with real credentials and requires access to a
lab-specific mapped network share (`Z:\`) that no CI runner has. A `workflow_dispatch` job would
need those as repo secrets plus VPN/self-hosted-runner access — out of scope for this change.
Someone with `WANDB_API_KEY` and network-share access runs `scripts/run_parity_harness.py`
locally when a refresh is needed, then commits the updated JSON themselves, same as the original
round that produced it (minus the "uncommitted" part).

### 4. Results-JSON schema lives in `build_report_entry`'s docstring

Expanding the existing docstring on `build_report_entry` (`parity.py:982`) to document every
field it emits keeps the schema right next to the code that produces it — it cannot drift
silently the way a separate companion doc could. Documented:

- `ground_truth_source`: `"labels_registry" | "relinked_bundle" | "basename_search"` — which
  resolution tier (see parent design §2) produced this model's ground truth.
- `n_frames_resolved` / `n_frames_total`: ground-truth resolution coverage (frame-level, not
  per-model — see parent design §2, tier 4). `n_frames_evaluated`: how many of the resolved
  frames the metrics below were actually computed over (may be less than `n_frames_resolved`
  when `sample_n` caps it).
- `sleap_nn` / `classic_sleap_reference`: full `ParityMetrics.to_dict()` shape for each side;
  `classic_sleap_reference` is `None` when no reference is available (no `labels_pr.val.slp` and
  no readable `metrics.val.npz` — parent design §4). Each side's own `settings` field is
  `"recomputed" | "stored"`.
- `distance_p95_delta` / `visibility_recall_delta`: the two gated deltas (parent design §6),
  `None` when `classic_sleap_reference` is `None` (no reference to compare against).
- A **gap entry** (unresolvable ground truth, or an isolated per-card failure per Decision 1)
  has only `registry_id` / `version` / `gap_reason` — no metrics fields at all.

## Behavior change

None for existing callers. New additions only:
- `run_parity_harness()` in `sleap_roots_predict/parity.py`.
- `scripts/run_parity_harness.py` (new, committed).
- Expanded docstring on `build_report_entry` (no signature change).

## Testing approach

TDD, fixture-based, no network — same convention as `evaluate_model_card`'s own tests
(vendored sleap-nn model + fixture `ModelCard`s):
- A card whose `evaluate_model_card` call succeeds produces a full report entry.
- A card whose `source.materialize` (or downstream call) raises is isolated: it becomes a gap
  entry with `gap_reason` set, and does not prevent other cards' entries from being written.
- The persisted output is exactly `write_parity_report`'s own format (no new serialization
  logic to test).

`scripts/run_parity_harness.py` itself is not unit-tested (it's a thin, credential-requiring
wrapper with no logic beyond argument wiring) — same treatment as the original scratch script,
except now it's committed instead of thrown away.

## Out of scope

- A config-file mechanism for `prefix_map` (Decision 2) — revisit only if a second lab/network
  layout needs a different mapping; not needed for one lab's two known entries.
- CI/scheduled regeneration (Decision 3).
- Re-running the harness in this session (no network/registry access from here) — this change
  ships the reusable mechanism; the next actual regeneration is a separate, manual, on-demand
  action by whoever has credentials and network-share access.

## Acceptance

- `run_parity_harness()` exists in `parity.py`, is unit-tested (success + per-card isolation)
  without network access, and persists via the existing `write_parity_report()`.
- `scripts/run_parity_harness.py` is committed, reads only existing env vars
  (`WANDB_API_KEY`/`SRP_PARITY_DATA_DIR`), and hardcodes the real, documented `prefix_map`.
- `build_report_entry`'s docstring documents every field in its output dict, including the gap-
  entry shape.
- `specs/prediction-parity/spec.md` gains a requirement for the reusable runner + isolation
  behavior.
