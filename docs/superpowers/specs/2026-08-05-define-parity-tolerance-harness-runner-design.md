# Design: reusable parity harness runner + results-schema docs (define-parity-tolerance)

- **Date:** 2026-08-05 (revised same day, after a `/review-openspec` adversarial round)
- **Repo:** `sleap-roots-predict`
- **Branch:** `define-parity-tolerance` (same branch/PR as the parent change — see
  [#33](https://github.com/talmolab/sleap-roots-predict/pull/33))
- **Parent design:** `2026-08-03-define-parity-tolerance-design.md` — this doc only covers the
  additions below; it does not re-derive anything already settled there.
- **Status:** APPROVED (revised). A first draft of this doc went through a 5-lens adversarial
  `/review-openspec` before any code was written; three lenses independently found the same
  design flaw (Decision 1's isolation was too broad) and two independently found the same
  schema-docs bug (Decision 4 mislabeled the report's delta fields). Both are fixed below,
  before implementation starts — cheaper to fix in text now than after the JSON schema or the
  isolation behavior is committed.

## Motivation

Both staleness bugs found during PR #33's post-review rounds (task 8.7, and the earlier
`-0.053` vs. `-0.085` figure at task 8.5) trace to the same root cause: the checked-in results
JSON (`docs/superpowers/specs/2026-08-04-define-parity-tolerance-results.json`) was produced by
an **uncommitted scratch script**. Nobody can re-run the full 13-model harness today without
rebuilding that script from scratch, and every number about it that lands in prose is a
manually-copied snapshot with no way to check it's still current. This change makes
regeneration a committed, reusable operation, and documents the JSON's schema so a reader
doesn't have to reverse-engineer `build_report_entry()` to understand it.

**A meta-motivation added on revision:** the review that caught this doc's own two bugs found
them by direct comparison against the real code and the real committed JSON — exactly the
discipline task 8.7 already learned the hard way. Decision 4 below adds a mechanism (not just
another paragraph of prose) so a future reader gets the same protection without needing another
adversarial pass.

## Settled decisions (from brainstorming 2026-08-05, revised same day)

### 1. `run_parity_harness()` — a new orchestration function in `parity.py`

```python
def run_parity_harness(
    cards: Sequence[ModelCard],
    source: "ModelCardSource",  # TYPE_CHECKING import only — see below
    workdir: Path,
    out_path: Union[str, Path],
    *,
    labels_registry_lookup=None,
    prefix_map=None,
    basename_index=None,
    sample_n=None,
) -> Path:
```

Coerces `out_path` to `Path` immediately (a `str` must not survive an entire multi-model run
only to blow up at the final write). For each card, converts it via
`card.to_model_ref(sleap_nn.__version__)` before calling `source.materialize(...)` — mirroring
`warm_worker.py`'s existing `ModelCard → ModelRef` conversion, rather than the shortcut one
existing test uses (`source.materialize(card)` directly, which only works because both
`ModelCardSource` implementations happen to read only `.registry_id`/`.version`). That shortcut
is fine in a test; baking it into a new public function is a bigger commitment against a
Protocol that explicitly types its parameter as `ModelRef`, and `ModelRef`/`ModelCard` diverge
on other fields (`root_type` optionality, `sleap_nn_version`'s meaning) that a future
`ModelCardSource` implementation could legitimately read.

Then calls `evaluate_model_card(...)`, and persists the accumulated entries via the existing
`write_parity_report()`, in input-card order, returning the report's path. Plain function
arguments only — no new env vars, no config file — mirroring `evaluate_model_card`'s own
keyword-only tail (it adds `source`/`out_path` and drops `bundle_dir`, so "mirrors" means the
keyword options, not an identical signature). Unit-testable with fixture `ModelCard`s and a real
`LocalCardSource`, no network.

**Per-card failure isolation, scoped narrowly — not a blanket wrapper.** The first draft of this
decision claimed to mirror `model_registry.py`'s `_collect_cards` isolation pattern, but
over-claimed how broad that mirror was. Re-reading `_collect_cards`'s own docstring
(`model_registry.py:198-205`): "The `try` wraps *only* per-artifact card construction... so
credential errors... and errors raised while traversing the registry propagate fail-loud — only
a single non-conforming artifact's card build is isolated here." `run_parity_harness` now
matches that precisely: only the per-card `source.materialize(...)` + `evaluate_model_card(...)`
call is wrapped in `except Exception` (never a bare `except`, so `KeyboardInterrupt`/
`SystemExit` still interrupt a 30+-minute run cleanly) — not `list_cards()`, not any
credential/setup step, which continue to propagate fail-loud exactly as they do today.

This scoping matters concretely: three independent review lenses flagged that a *blanket* wrap
(the original framing) would convert a systemic failure — an expired `WANDB_API_KEY`, an
unmounted `Z:` share — into 13 gap entries at exit 0, silently overwriting the empirical
baseline the whole tolerance decision rests on. That is a *worse* version of the exact
staleness bug this slice exists to fix. Two further guards close this:

- A caught per-card exception becomes a gap entry carrying the exception type/message **and a
  discriminator distinguishing it from a ground-truth-resolution gap**
  (`evaluate_model_card`'s own existing, pre-this-change gap path). The two are different
  failure kinds — "no ground truth available" is expected and benign; "the pipeline crashed on
  this card" is a bug or an outage — and collapsing them into an identical
  `{registry_id, version, gap_reason}` shape (the original plan) makes that distinction
  unrecoverable from the persisted JSON. Both gap paths now set a `gap_stage` field
  (`"resolution"` for the existing `evaluate_model_card` path, `"evaluation"` for a
  `run_parity_harness`-caught exception).
- Before writing, `run_parity_harness` checks whether *every* entry is a gap. If so, **and** a
  report already exists at `out_path`, it raises rather than silently overwriting — an all-gap
  result is far more likely to mean "something in the environment is broken" than "all 13
  models genuinely have zero resolvable ground truth" (which has never happened in this
  project's history; the committed JSON has zero gaps across all 13).

Avoid a runtime `parity.py → model_registry.py` import for the `ModelCardSource` type hint:
this file currently imports nothing from `model_registry` (verified: stdlib + numpy + sleap_io +
sleap_nn + sleap_roots_contracts only), and has no `from __future__ import annotations`, so a
bare `source: ModelCardSource` annotation would be a real import — reintroducing the coupling
task 8.3 deliberately removed ("keeping `parity.py` decoupled from `model_registry.py`"). Import
`ModelCardSource` under `if TYPE_CHECKING:` instead.

### 2. `prefix_map`'s *source* keys are a committed script constant; the share-root *target* is a CLI arg

The lab-specific prefix-map **keys** (`D:/SLEAP`, `C:/Users/pbiobgh/Desktop/SLEAP`) are
immutable facts baked into the training-time `.slp` files' embedded video paths themselves —
they cannot change without re-training, already committed as prose in the parent design doc,
and only two entries. Hardcoding them in `scripts/run_parity_harness.py` is the right call
(Decision 2, first draft) — that *is* the fix for the root cause (the mapping is now committed
and reusable, not living only in an uncommitted scratch file or someone's head).

The **target** (`Z:/users/eberrigan/SLEAP`) is a Windows mapped-drive letter — exactly as
machine-specific as `SRP_PARITY_DATA_DIR` (the basename-search root), which this same decision
correctly refuses to hardcode, for that same reason. The first draft applied that test
inconsistently: it kept the search root as an env var but fully hardcoded the relink target.
Fixed: the script exposes `--share-root` (argparse, defaulting to the current value) so the two
source-side prefixes get remapped against a caller-overridable target, without introducing any
new env var or config-file mechanism (still just a CLI default — Decision 3's YAGNI reasoning
is unaffected, since this is one more argparse default, not new machinery).

### 3. Regeneration stays manual/on-demand — no CI wiring

This hits a live external wandb registry with real credentials and requires access to a
lab-specific mapped network share (`Z:\`) that no CI runner has. A `workflow_dispatch` job would
need those as repo secrets plus VPN/self-hosted-runner access — out of scope for this change.
Someone with `WANDB_API_KEY` and network-share access runs `scripts/run_parity_harness.py`
locally when a refresh is needed, then commits the updated JSON **as its own standalone commit**
(not bundled with a code change) — a bad regeneration is then a clean single-commit revert.

`SRP_PARITY_DATA_DIR` is documented in the script's own module docstring and in README's
existing Parity Harness section only — it is **not** added to `.env.example` or the
`EXPECTED_VARS` set `tests/test_env_docs.py` asserts exact equality against
(`test_env_example_lists_exactly_the_expected_vars`). That set is scoped to production/operator
runtime config (`WANDB_API_KEY`, `SRP_WANDB_*`, `SRP_MODEL_CACHE_DIR`, `SRP_DEVICE`); the
acceptance test's own analogous test-gating vars (`SRP_CYLINDER_DIR`/`SRP_MODEL_DIRS`) are
correctly absent from it too, for the same reason — confirmed by reading both files. Adding
`SRP_PARITY_DATA_DIR` there would fail that test in CI.

The script:
- Reads `WANDB_API_KEY` and `SRP_PARITY_DATA_DIR` from the environment (existing convention, no
  new env vars).
- Accepts `--share-root` (default the real, documented value — see Decision 2) and `--out`
  (default anchored via `Path(__file__).resolve().parents[1] / "docs/superpowers/specs/
  2026-08-04-define-parity-tolerance-results.json"` — **not** a bare CWD-relative string, which
  would be exactly the kind of hardcoded-path fragility this slice exists to eliminate).
- Calls `WandbRegistrySource().list_cards()` and `build_basename_index(SRP_PARITY_DATA_DIR)`.
- Calls `run_parity_harness(..., sample_n=100)` — `100` matches the original empirical run
  (task 5.2), kept as the script's own default; `run_parity_harness` itself defaults `sample_n`
  to `None` (no magic number baked into the reusable library function).
- Carries a module docstring stating plainly: Windows + a `Z:` mapped network share, this lab
  only, not portable, not shipped in the wheel or the container image, invoke via
  `uv run python scripts/run_parity_harness.py` (so `sleap_roots_predict` resolves via the
  editable venv rather than a bare `python`, which would put `scripts/` on `sys.path[0]`
  instead of the repo root).

### 4. Results-JSON schema lives in `build_report_entry`'s docstring — corrected before it lands

Two inaccuracies in this design doc's own first draft, both caught independently by two review
lenses (with matching hand-computed numbers), must be fixed in the *docstring* task 9.4 writes —
not just here, since the first draft's wrong phrasing traces back to language **already present
in the shipped code**, not something this slice invented:

- **`distance_p95_delta` / `visibility_recall_delta` are NOT "the gated deltas."** They are
  unsigned, raw-unit `abs()` differences (`parity.py:1023-1029`) — informational only.
  `within_tolerance` (`parity.py:963-979`) recomputes its own values from the two full metrics
  dicts: a **relative** distance delta (`|Δ| / reference.distance_p95`) and a **signed**,
  directional recall delta (`sleap_nn.visibility_recall - reference.visibility_recall`, where
  sleap-nn scoring higher never fails). Neither of the parent design's §6 table figures (signed,
  relative — e.g. `-0.085`, `17.0%`) is readable directly from these two fields; both must be
  recomputed from the full metrics dicts. Verified against the real JSON: e.g.
  `rice-cylinder-crown-age6-10`'s `visibility_recall_delta` field reads `0.0530` where the
  actual signed gate value is `-0.0530`. **This wrong phrase ("plus the two gated deltas") is
  already in the shipped `build_report_entry` docstring today** (`parity.py:991`) and in the
  parent design doc's own prose (`2026-08-03-...-design.md:157`) — task 9.4 fixes the docstring
  (the living copy); a small follow-up doc fix should correct the parent doc's prose too, since
  it's the same wrong claim in a second place.
- **`settings` is not two-valued on both sides.** The `sleap_nn` side is always `"recomputed"`
  by construction (`parity.py:687`); only `classic_sleap_reference` can be `"stored"`
  (`parity.py:851`). All 13 entries in the currently-committed JSON are `recomputed`/
  `recomputed`, so the `"stored"` branch is documented-but-currently-unexercised in the
  artifact — worth saying so.
- **The gap-entry shape (now two kinds — see Decision 1's `gap_stage`) belongs on
  `write_parity_report`'s and `run_parity_harness`'s docstrings**, not solely on
  `build_report_entry` (which only ever produces the full shape and cannot itself emit a gap
  entry) — cross-reference between them rather than documenting the gap shape on a function
  that can't produce it.
- **`weights_checksum` dedup is required before summarizing** — 13 raw entries collapse to 8
  physically distinct models when grouped by it (several `registry_id`s share weights). A future
  reader computing a "measured max" delta across the JSON without deduping first would
  double-count exactly the way task 8.5's bug happened. Say so explicitly.
- **One anti-drift sentence**, mirroring the parent design doc's own successful pattern at its
  §2 ("a percentage hardcoded into this prose will go stale... as this sentence itself already
  did once"): state that this docstring is the living schema description, and any other prose
  description of the schema (including this design doc, once it's a few months old) is a
  snapshot that can go stale — read the docstring, not prose, for the current shape.

### 5. `scripts/` needs its own lint-gate fix, landed alongside it

`scripts/` will be the repo's first top-level non-package directory holding Python source
(`docs/`, `openspec/` hold no `.py` files; `examples/` is a separate precedent but isn't in any
lint target either). Checked directly: CI's lint job runs `black --check sleap_roots_predict
tests` and `ruff check sleap_roots_predict/` (`ci.yml:36,39`) — neither covers `scripts/`. But
bare `codespell` (`ci.yml:41`, no path argument) *does* cover it — verified empirically by
planting a typo in a throwaway file under `scripts/` and confirming `codespell` flagged it and
exited nonzero. So the new script is exempt from formatting/lint checks but not from spelling
checks, an inconsistency worth closing rather than leaving as a surprise for whoever next edits
it. Fixed via a small, separate `chore:` commit extending the `black`/`ruff` targets in
`ci.yml` and the three `.claude/commands/*.md` copies (`lint.md`, `fix-formatting.md`,
`pre-merge.md`) to include `scripts` — landed *with or immediately after* `scripts/`'s own
commit, per this project's own standing rule against referencing not-yet-built paths in CI
config (`openspec/project.md`'s "do not add CI steps referencing not-yet-built modules").

### 6. Docs sweep, mirroring task 7.1's precedent

The first draft of this slice had no docs task at all, unlike the parent change's task 7.1. Add
one: fold a short addition into `CHANGELOG.md`'s existing `[Unreleased]` parity bullet (not a
new bullet — the harness hasn't shipped in a release yet); add the regeneration command and an
explicit naming of `SRP_PARITY_DATA_DIR` to README's existing Parity Harness section (currently
that var is named nowhere a reader would look — only in `pyproject.toml`, test code, and
OpenSpec files); add a `scripts/` entry to README's repo-tree diagram; add `run_parity_harness`
and the regeneration script to `openspec/project.md`'s `parity.py` Architecture-Patterns bullet,
and name `SRP_PARITY_DATA_DIR` in its parity Testing-Strategy bullet. State explicitly that
`API.md` is intentionally unchanged (no new public re-export), so it isn't re-litigated by a
future reviewer the way task 7.1 had to state it once already.

## Behavior change

None for existing callers. New additions only:
- `run_parity_harness()` in `sleap_roots_predict/parity.py`.
- A `gap_stage` field added to `evaluate_model_card`'s existing ground-truth-resolution gap
  entries (additive; existing consumers reading the other three keys are unaffected).
- `scripts/run_parity_harness.py` (new, committed) — the repo's first top-level `scripts/` dir.
- Corrected + expanded docstrings on `build_report_entry`/`write_parity_report` (no signature
  change).
- `black`/`ruff` CI targets extended to include `scripts/`.

## Testing approach

TDD, fixture-based, no network — same convention as `evaluate_model_card`'s own tests
(`LocalCardSource` + vendored sleap-nn model + fixture `ModelCard`s):
- All-cards-succeed: one entry per card, in input order, in `build_report_entry`'s shape;
  the function returns `out_path`; a `str` `out_path` is coerced up front.
- Resolution/sampling options (`sample_n`/`prefix_map`/`basename_index`/
  `labels_registry_lookup`) are observably forwarded to `evaluate_model_card` — no mocks needed,
  just distinct fixtures that resolve through distinct tiers.
- A card whose `materialize`/`evaluate_model_card` call raises is isolated as a `gap_stage=
  "evaluation"` entry, distinct from `evaluate_model_card`'s own `gap_stage="resolution"` gaps;
  exactly one warning is logged (asserted via `caplog`, mirroring
  `test_collect_cards_skips_malformed_and_warns`'s existing convention); the run continues and
  preserves order for the remaining cards.
- `KeyboardInterrupt` from a card's `materialize` propagates rather than becoming a gap entry
  (guards the `except Exception`, not bare `except`, choice).
- An all-gap run does not overwrite a pre-existing report at `out_path`.
- An empty `cards` list writes `[]` and returns `out_path` (pinned deliberately — no existing
  report to protect in this case, since there's nothing to compare against).

`scripts/run_parity_harness.py` itself stays a thin, credential-requiring wrapper with the two
now-testable pieces (`PREFIX_MAP`'s source keys, the default `--out` path) kept as simple
module-level constants a smoke test can assert against; `--help` parsing (exit 0, no traceback)
is checked in the verification step rather than pytest, since the script isn't collected by
`testpaths`.

## Out of scope

- A config-file mechanism for `prefix_map`/`SRP_PARITY_DATA_DIR` — revisit only if a second
  lab/network layout needs a different mapping; not needed for one lab's two known entries.
- CI/scheduled regeneration (Decision 3).
- A provenance envelope on the persisted report (generation timestamp, sample_n, code version,
  etc.) — genuinely useful, but changes the report's top-level shape from a bare list to an
  object, which would need its own scenario/schema decision; deferred to a future slice rather
  than folded in here under time pressure.
- Re-running the harness in this session (no network/registry access from here) — this change
  ships the reusable mechanism; the next actual regeneration is a separate, manual, on-demand
  action by whoever has credentials and network-share access, committed as its own standalone
  commit per Decision 3.

## Acceptance

- `run_parity_harness()` exists in `parity.py`, is unit-tested (success, isolation with a
  distinguishable gap kind, non-card-specific-error propagation, the no-clobber-on-all-gap
  guard, `KeyboardInterrupt` safety, empty-input behavior) without network access, and persists
  via the existing `write_parity_report()`.
- `evaluate_model_card`'s existing ground-truth-resolution gap entries carry a `gap_stage`
  field, distinguishing them from `run_parity_harness`'s own isolated-failure gaps.
- `scripts/run_parity_harness.py` is committed, reads only existing env vars
  (`WANDB_API_KEY`/`SRP_PARITY_DATA_DIR`), exposes `--share-root`/`--out` CLI args with the real
  documented defaults, and is covered by CI's `black`/`ruff` targets.
- `build_report_entry`'s (and `write_parity_report`'s) docstrings document every field in their
  output, including the corrected delta-field semantics, the `settings` asymmetry, the
  `weights_checksum` dedup note, and both gap-entry shapes.
- `specs/prediction-parity/spec.md` gains a requirement for the reusable runner covering
  success, isolation, non-card-specific propagation, and the no-clobber guard.
- `CHANGELOG.md`/`README.md`/`openspec/project.md` reflect the new runner/script.
