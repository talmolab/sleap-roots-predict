# Design: reusable parity harness runner + results-schema docs (define-parity-tolerance)

- **Date:** 2026-08-05 (revised twice same day, after two `/review-openspec` adversarial rounds)
- **Repo:** `sleap-roots-predict`
- **Branch:** `define-parity-tolerance` (same branch/PR as the parent change — see
  [#33](https://github.com/talmolab/sleap-roots-predict/pull/33))
- **Parent design:** `2026-08-03-define-parity-tolerance-design.md` — this doc only covers the
  additions below; it does not re-derive anything already settled there.
- **Status:** APPROVED (revised twice). Round 1 caught a data-loss-risk design flaw and a
  mislabeled-schema doc bug before any code was written. Round 2 found round 1's isolation fix
  was cosmetic — the two failure modes it named still bypass it — and several attribution/count
  errors. Both rounds ran entirely against text; no code exists yet for either round to have
  broken. This revision narrows the isolation claim to what is structurally true and makes the
  no-clobber guard the explicitly primary defense, rather than trying to make per-card exception
  handling do a job it structurally cannot.

## Motivation

Both staleness bugs found during PR #33's post-review rounds (task 8.7, and the earlier
`-0.053` vs. `-0.085` figure at task 8.5) trace to the same root cause: the checked-in results
JSON (`docs/superpowers/specs/2026-08-04-define-parity-tolerance-results.json`) was produced by
an **uncommitted scratch script**. Nobody can re-run the full 13-model harness today without
rebuilding that script from scratch, and every number about it that lands in prose is a
manually-copied snapshot with no way to check it's still current. This change makes
regeneration a committed, reusable operation, and documents the JSON's schema so a reader
doesn't have to reverse-engineer `build_report_entry()` to understand it.

## Settled decisions (from brainstorming 2026-08-05, revised twice same day)

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
`card.to_model_ref(version("sleap-nn"))` (stdlib `importlib.metadata.version`) before calling
`source.materialize(...)`, **inside** the per-card try/except described below (a conversion
failure is card-specific too). This mirrors the *actual* existing conversion at
`model_selection.py:98` (`matches[0].to_model_ref(_RUNTIME_SLEAP_NN_VERSION)`, where
`_RUNTIME_SLEAP_NN_VERSION = version("sleap-nn")`) — **not** `warm_worker.py`, which never calls
`to_model_ref` at all (round 1's mirroring claim named the wrong file; round 2 caught it). Using
the same `importlib.metadata.version("sleap-nn")` call, rather than `sleap_nn.__version__`,
also avoids a real bug: `parity.py` does not bind the bare name `sleap_nn` today (only
`from sleap_nn.evaluation import load_metrics, run_evaluation`), so `sleap_nn.__version__` as
originally drafted would raise `NameError` the first time it ran.

Then calls `evaluate_model_card(...)`, and persists the accumulated entries via the existing
`write_parity_report()`, in input-card order, returning the report's path. Plain function
arguments only — no new env vars, no config file. Unit-testable with fixture `ModelCard`s and a
real `LocalCardSource`, no network.

**Per-card failure isolation — scoped to exactly what it can actually catch, no more.** Only
the per-card `card.to_model_ref(...)` + `source.materialize(...)` + `evaluate_model_card(...)`
call is wrapped in `except Exception` (never a bare `except`, so `KeyboardInterrupt`/
`SystemExit` still interrupt a 30+-minute run cleanly). A caught exception becomes a gap entry
carrying the exception type/message and `gap_stage="evaluation"` — distinct from
`evaluate_model_card`'s own pre-existing ground-truth-resolution gaps, now tagged
`gap_stage="resolution"`. `evaluate_model_card`'s own `Returns:` docstring is updated in the
same commit to describe the new field (it currently documents the gap entry as exactly three
keys; adding a fourth without updating that docstring would make it stale the moment this
lands).

**Round 1 claimed this mirrors `model_registry.py`'s `_collect_cards` isolation pattern — true
of the *code*, not of what it actually protects against, and round 2 found the practical
consequence:** neither of the two failure modes round 1 was written to guard against is
actually caught by this scoping, because neither is structurally a per-card exception in this
design:

- `WandbRegistrySource.materialize` calls `self._require_key()` **inside** the call this
  isolation wraps — an expired/missing `WANDB_API_KEY` surfaces *as* a per-card exception, not
  before it, so it still becomes 13 gap entries. There is no way to distinguish "this card's
  ground truth is genuinely unavailable" from "the registry rejected our credentials" by
  exception type/message alone without adding real credential validation, which is out of scope
  here (see Out of Scope).
- An unmounted `Z:` share never reaches the runner's per-card isolation at all:
  `build_basename_index` (called by the *script*, before `run_parity_harness` is ever invoked)
  does `os.walk(search_root)` on a missing root and returns `{}` — no exception, silently — so
  every card resolves through `evaluate_model_card`'s own existing `gap_stage="resolution"` path
  instead.

So the spec's original "a non-card-specific error propagates instead of becoming a gap"
scenario asked for something this design cannot deliver without new preflight-validation
machinery this slice doesn't need (Out of Scope). **The actual, sole protection against a
systemic failure silently overwriting the empirical baseline is the no-clobber guard below** —
the spec is revised to say so plainly rather than asserting an isolation boundary that doesn't
exist. This is now explicit in the Risks section, not glossed over.

**The no-clobber guard, corrected for an edge case round 2 found:** before writing, if **no
card produced a full (non-gap) entry** — whether because every card gapped, or because `cards`
was empty — **and** a report already exists at `out_path`, the runner raises (a
`RuntimeError` naming `out_path` and the entry count) instead of overwriting. Naive phrasing
("if every entry is a gap") is vacuously true for an empty list, which would incorrectly block
the legitimate "no cards to evaluate" case from ever writing `[]` when nothing exists yet at
`out_path` — but round 2 found the *inverse* is the actual intended behavior: an empty run
should also refuse to clobber an *existing* report, since writing `[]` over a 13-entry baseline
is exactly the kind of silent data loss this guard exists to prevent. Both cases are covered by
one condition: `if not any(is_full_entry(e) for e in entries) and out_path.exists(): raise`.
This correctly still allows the very first run (no report yet) to write, gap-only or not.

**A residual, accepted risk, stated rather than silently left uncaught:** a *partial* failure —
12 gaps and 1 success, say — still overwrites a 13-good baseline, since "at least one non-gap
entry" satisfies the guard. This is a real gap in what the guard protects against; closing it
would need either a minimum-success-count threshold or a `--force` flag, neither of which this
slice adds (see Out of Scope) — flagged explicitly in Risks below rather than implied to be
handled.

Avoid a runtime `parity.py → model_registry.py` import for the `ModelCardSource` type hint:
this file currently imports nothing from `model_registry` (verified: stdlib + numpy + sleap_io +
sleap_nn.evaluation + sleap_roots_contracts only), and has no `from __future__ import
annotations`, so a bare `source: ModelCardSource` annotation would be a real import —
reintroducing the coupling task 8.3 deliberately removed. Import `ModelCardSource` under
`if TYPE_CHECKING:` instead, with the annotation itself quoted (`source: "ModelCardSource"`).

### 2. `prefix_map`'s *source* keys are a committed script constant; the share-root *target* is a CLI arg

The lab-specific prefix-map **keys** (`D:/SLEAP`, `C:/Users/pbiobgh/Desktop/SLEAP`) are
immutable facts baked into the training-time `.slp` files' embedded video paths themselves —
they cannot change without re-training, already committed as prose in the parent design doc,
and only two entries. Hardcoding them in `scripts/run_parity_harness.py` is the right call —
that *is* the fix for the root cause (the mapping is now committed and reusable, not living only
in an uncommitted scratch file or someone's head).

The **target** (`Z:/users/eberrigan/SLEAP`) is a Windows mapped-drive letter — exactly as
machine-specific as `SRP_PARITY_DATA_DIR` (the basename-search root), which this same decision
correctly refuses to hardcode, for that same reason. Fixed by exposing `--share-root` (argparse,
defaulting to the current value) so the two source-side prefixes get remapped against a
caller-overridable target, without introducing any new env var or config-file mechanism (still
just one more argparse default). **The script's `--share-root` default is the single
authoritative, executable copy of this value going forward** — every mention of
`Z:/users/eberrigan/SLEAP` in the dated design docs and in `tasks.md` is a historical record of
the empirical run that produced the checked-in JSON, not a value anything needs to keep in sync
with the script.

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
correctly absent from it too, for the same reason. Adding `SRP_PARITY_DATA_DIR` there would fail
that test in CI.

The script:
- Reads `WANDB_API_KEY` and `SRP_PARITY_DATA_DIR` from the environment (existing convention, no
  new env vars).
- Accepts `--share-root` (default the real, documented value — see Decision 2) and `--out`
  (default anchored via `Path(__file__).resolve().parents[1] / "docs/superpowers/specs/
  2026-08-04-define-parity-tolerance-results.json"` — **not** a bare CWD-relative string).
- Calls `WandbRegistrySource().list_cards()` and `build_basename_index(SRP_PARITY_DATA_DIR)`.
- Calls `run_parity_harness(..., sample_n=100)` — `100` matches the original empirical run
  (task 5.2), kept as the script's own default; `run_parity_harness` itself defaults `sample_n`
  to `None`.
- Carries a module docstring stating plainly: Windows + a `Z:` mapped network share, this lab
  only, not portable, not shipped in the wheel or the container image, invoke via
  `uv run python scripts/run_parity_harness.py` (so `sleap_roots_predict` resolves via the
  editable venv rather than a bare `python`, which would put `scripts/` on `sys.path[0]`).
- Is **not** unit-tested beyond a `--help` argparse smoke check (run manually in task 9.7's
  verification step, not under pytest — `scripts/` is outside `testpaths`). The two constants it
  defines (`PREFIX_MAP`'s source keys, the default `--out` path) are simple enough that a smoke
  check + code review substitutes for a dedicated test, given the script itself is otherwise
  exempt from pytest collection.

### 4. Results-JSON schema documentation — corrected, and consolidated to one canonical location

Two inaccuracies in this design doc's own first draft, both caught independently by two review
lenses (with matching hand-computed numbers), are fixed in the docstring task 9.4 writes — and
this design doc's own text is corrected to match, since round 2 found round 1's fix repeated a
small factual slip (see below):

- **`distance_p95_delta` / `visibility_recall_delta` are NOT "the gated deltas."** They are
  unsigned, raw-unit `abs()` differences (`parity.py:1023-1029`) — informational only.
  `within_tolerance` (`parity.py:963-979`) recomputes its own values from the two full metrics
  dicts: a **relative** distance delta (`|Δ| / reference.distance_p95`) and a **signed**,
  directional recall delta. Neither of the parent design's §6 table figures is readable
  directly from these two fields. **This wrong phrase is already in the shipped
  `build_report_entry` docstring today**, verbatim: `parity.py:991` reads "plus the two gated
  deltas" (round 1 quoted this correctly). The parent design doc's own prose has the same
  claim, minus the word "two": `2026-08-03-...-design.md:156-157` reads "...for both sides
  **plus the gated deltas** to..." — task 9.4 fixes the shipped docstring (the living copy);
  task 9.4b fixes the parent doc's prose, mirroring task 8.7's own precedent of tracking a
  stale-dated-doc fix as an explicit task rather than a passing mention.
- **`settings` is not two-valued on both sides.** The `sleap_nn` side is always `"recomputed"`
  by construction (`parity.py:687`); only `classic_sleap_reference` can be `"stored"`
  (`parity.py:851`).
- **The two gap-entry shapes (distinguished by `gap_stage`) belong on `write_parity_report`'s
  and `run_parity_harness`'s docstrings**, not `build_report_entry`'s (which only ever produces
  the full shape). `write_parity_report`'s current one-line summary ("Persist a list of
  `build_report_entry` dicts...") is itself already slightly inaccurate today — gap entries were
  never `build_report_entry` output — and becomes more so once two gap kinds exist; task 9.4
  corrects that summary line too, not just adds to it.
- **`weights_checksum` dedup is required before summarizing** — 13 raw entries collapse to 8
  physically distinct models when grouped by it.
- **One canonical source, named explicitly rather than implied**: `build_report_entry`'s
  docstring documents the full-entry schema; `write_parity_report`/`run_parity_harness`'s
  docstrings document the two gap-entry shapes and cross-reference `build_report_entry` rather
  than restating it. Anywhere else the schema is described in prose (this doc included, once
  it's a few months old) is a snapshot that can go stale — read the docstrings, not prose, for
  the current shape.

### 5. `scripts/` needs its own lint-gate fix, landed alongside it

`scripts/` will be the repo's first top-level non-package directory holding Python source.
Checked directly: CI's lint job runs `black --check sleap_roots_predict tests` and `ruff check
sleap_roots_predict/` (`ci.yml:36,39`) — neither covers `scripts/`. Bare `codespell` (`ci.yml`,
no path argument) *does* cover it — verified empirically. **Also found on this revision's
second pass, and previously missed: `ci.yml`'s own `paths:` trigger filter
(`sleap_roots_predict/**`, `tests/**`, `.github/workflows/ci.yml`, `pyproject.toml`) does not
include `scripts/**` either** — so a future PR touching only `scripts/run_parity_harness.py`
would not trigger CI at all, making the lint-target fix inert for exactly the changes it exists
to police. Both gaps are fixed in the same commit: extend the `black`/`ruff` targets **and** add
`scripts/**` to the `paths:` filter in `ci.yml`, plus the lint-target strings in five
`.claude/commands/*.md` files that hardcode them (`lint.md`, `fix-formatting.md`,
`pre-merge.md`, and two more round 1 missed: `ci-debug.md`, `pr-description.md` — the former is
specifically the command used to reproduce a broken CI run, so leaving it stale defeats its own
purpose). Landed in a separate `chore:` commit, immediately after `scripts/run_parity_harness.py`
itself exists and is already clean against `black`/`ruff` (verified locally before this commit,
since `ruff`'s `select = ["D"]`/google-convention rule will require full docstrings on the new
script the moment this lands).

### 6. Docs sweep, mirroring task 7.1's precedent — its own commit

Fold a short addition into `CHANGELOG.md`'s existing `[Unreleased]` parity bullet (not a new
bullet). Add the regeneration command, an explicit naming of `SRP_PARITY_DATA_DIR`, and the
`--share-root` flag (Decision 2 made it overridable specifically so a different operator/machine
could set it — a regeneration doc that omits it defeats that purpose) to README's Parity Harness
section. README's "Project Structure" section is two separate package-scoped blocks (not one
tree diagram) — insert a fourth block for `scripts/`. Add `run_parity_harness` and the
regeneration script to `openspec/project.md`'s `parity.py` Architecture-Patterns bullet, and name
`SRP_PARITY_DATA_DIR` in its parity Testing-Strategy bullet. State explicitly that `API.md` is
unchanged (no new public re-export). Lands as its own `docs:` commit, per task 7.1's actual
precedent (`dfb0624`, `1c9a67d` — two standalone docs commits, never folded into the pre-merge
gate commit) — never bundled with 9.7's final gate commit.

## Behavior change

None for existing callers. New additions only:
- `run_parity_harness()` in `sleap_roots_predict/parity.py`.
- A `gap_stage` field added to `evaluate_model_card`'s existing ground-truth-resolution gap
  entries (additive; its `Returns:` docstring updated in the same commit).
- `scripts/run_parity_harness.py` (new, committed) — the repo's first top-level `scripts/` dir.
- Corrected + expanded docstrings on `build_report_entry`/`write_parity_report` (no signature
  change).
- `black`/`ruff` CI targets and `ci.yml`'s `paths:` filter extended to include `scripts/`.

## Testing approach

TDD, fixture-based, no network — `LocalCardSource` + vendored sleap-nn model + fixture
`ModelCard`s:
- All-cards-succeed: one entry per card, in input order, in `build_report_entry`'s shape; the
  function returns `out_path` (a `Path`, even when passed a `str`).
- `sample_n` forwarding, tested cheaply and separately from `prefix_map`/`basename_index`
  forwarding (bundling all four options into one test hides which one a future regression
  actually broke).
- A card whose conversion/materialize/evaluation call raises is isolated as a `gap_stage=
  "evaluation"` entry, distinct from `evaluate_model_card`'s own `gap_stage="resolution"` gaps;
  exactly one warning is logged, naming the failing card, not just counted; the run continues
  and preserves order for the remaining cards.
- `KeyboardInterrupt` propagates rather than becoming a gap entry, and leaves no partial report
  file behind.
- The no-clobber guard: an all-gap run with an existing report at `out_path` raises (a real
  pre-written sentinel report, compared by content after — never by `mtime`, which is flaky on
  same-second rewrites); an all-gap run with **no** existing report still writes (the guard must
  not block the legitimate first run); an empty `cards` list with an existing report also
  raises (round 2's fix); an empty `cards` list with no existing report writes `[]`.

`scripts/run_parity_harness.py` stays a thin, credential-requiring wrapper verified only by a
manual `--help` smoke check plus code review — not pytest-collected.

## Out of scope

- A config-file mechanism for `prefix_map`/`SRP_PARITY_DATA_DIR` — not needed for one lab's two
  known entries.
- CI/scheduled regeneration (Decision 3).
- Real preflight credential/share-root validation that would let the runner genuinely
  distinguish a systemic failure from a per-card one before it happens — the no-clobber guard is
  the accepted, cheaper mitigation for this slice; a smarter distinction is a follow-up if the
  partial-gap residual risk (noted in Risks) turns out to matter in practice.
- A provenance envelope on the persisted report (generation timestamp, sample_n, code version) —
  would change the report's top-level shape from a bare list to an object; deferred to a future
  slice.
- Re-running the harness in this session (no network/registry access from here).

## Risks / Trade-offs

- **The per-card isolation does not, and structurally cannot, distinguish a systemic failure
  (bad credentials, an unmounted share) from a genuine per-card resolution gap.** The no-clobber
  guard is the sole protection against that specific failure silently overwriting the committed
  baseline, and it only covers the *all*-gap case. A *partial* failure (most cards gap, one or
  two succeed) still overwrites a fully-resolved baseline undetected. Accepted for this slice;
  a minimum-success-count threshold or `--force` flag would close this if it becomes a real
  problem in practice.
- **One tolerance number, heterogeneous models** — unchanged from the parent design's own
  accepted trade-off; not reopened here.

## Acceptance

- `run_parity_harness()` exists in `parity.py`, is unit-tested (success; per-card isolation with
  a distinguishable `gap_stage`; the no-clobber guard covering both all-gap and empty-input
  cases, on both first-run and existing-report paths; `KeyboardInterrupt` safety) without
  network access, and persists via the existing `write_parity_report()`.
- `evaluate_model_card`'s existing ground-truth-resolution gap entries carry a `gap_stage`
  field, distinguishing them from `run_parity_harness`'s own isolated-failure gaps, and its own
  docstring is updated to match.
- `scripts/run_parity_harness.py` is committed, reads only existing env vars
  (`WANDB_API_KEY`/`SRP_PARITY_DATA_DIR`), exposes `--share-root`/`--out` CLI args with the real
  documented defaults, and is covered by CI's `black`/`ruff` targets *and* trigger paths.
- `build_report_entry`'s (and `write_parity_report`'s) docstrings document every field in their
  output, including the corrected delta-field semantics, the `settings` asymmetry, the
  `weights_checksum` dedup note, and both gap-entry shapes — with `build_report_entry` as the
  one canonical schema source the others cross-reference.
- `specs/prediction-parity/spec.md` gains a requirement for the reusable runner covering success,
  isolation with a named `gap_stage` discriminator, and the no-clobber guard as the stated
  primary defense — not a claim of non-card-specific-error propagation this design cannot
  deliver.
- The parent 2026-08-03 design doc's "gated deltas" prose is corrected (task 9.4b).
- `CHANGELOG.md`/`README.md`/`openspec/project.md` reflect the new runner/script, including the
  `--share-root` flag.
