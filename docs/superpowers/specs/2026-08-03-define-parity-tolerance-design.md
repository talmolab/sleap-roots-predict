# Design: A3-predict parity tolerance + reference scan set (define-parity-tolerance)

- **Date:** 2026-08-03
- **Repo:** `sleap-roots-predict`
- **Branch:** `define-parity-tolerance`
- **Issue:** sleap-roots-pipeline#15 ("Decide A3-predict parity tolerance + reference scan
  set"), part of the A3 epic (sleap-roots-pipeline#9). Coordinates with predict#8
  ("Quantitatively tune peak_threshold per dataset") and predict#32 ("ModelCard.mode becomes a
  controlled vocabulary in contracts 0.1.0a6 — decide list_cards() behavior").
- **Status:** APPROVED — brainstorming complete. This doc is the settled design. Proceed to
  the OpenSpec proposal (`/openspec:proposal`), suggested change-id `define-parity-tolerance`.

## Motivation

`sleap-roots-predict`'s sleap-nn rewrite is functionally complete and already running real
cluster traffic, but its "A3-predict ✅" status has been blocked since 2026-07-06 on an
undefined parity gate: nobody has decided what accuracy tolerance vs. the legacy classic-SLEAP
pipeline is acceptable, or which scans/models to measure it on. This is **not** a
model-retraining/quality question — the production wandb registry models are still legacy
classic-SLEAP-trained weights (`training_config.json` + `best_model.h5`), loaded into sleap-nn
via a sanitization workaround. So "parity" here is an **inference-engine correctness**
question: given the *same already-trained weights*, does sleap-nn's forward pass reproduce the
same predictions as classic-SLEAP's? This is a narrower, stricter question than the
"TF-trained vs. PyTorch-trained model quality" parity question `sleap-roots-training`'s roadmap
already addresses (and deliberately relaxes — "exact numeric parity is not the bar" — for
that different, backend-training axis). Because nothing is being retrained here, there is no
reason to expect the two engines' outputs to diverge more than floating-point/implementation
noise, so this harness can and should demand a tight tolerance.

## Settled decisions (from brainstorming 2026-08-03)

### 1. Ground truth, not predictions-vs-predictions

Initial framing mistakenly treated the legacy pipeline's own *predictions* (real production
scans + legacy `.slp` outputs found on `Z:\users\eberrigan\...` experiment directories) as
"ground truth." Corrected: ground truth means human-labeled data. Every production model's
wandb artifact bundles a classic-SLEAP training export — `labels_gt.train.slp`,
`labels_gt.val.slp` (real human-labeled validation split), `labels_pr.val.slp` (classic-SLEAP's
own predictions on that split, present in most but not all exports), and `metrics.val.npz`
(classic-SLEAP's own precomputed eval, in the same `.npz` shape `sleap_nn.evaluation.
load_metrics()` reads). Confirmed by materializing a live production artifact
(`arabidopsis-cylinder-primary-age2-14:v0`) — the bundle structure matches exactly.

The harness's ground truth is therefore **`labels_gt.val.slp`, per model**, not any
field-experiment scan set.

### 2. Ground-truth image resolution, in priority order

`labels_gt.val.slp`'s embedded video references point at the *original training machine's*
file paths (e.g. `D:/SLEAP/SLEAP_arabidopsis/...`, `C:/Users/pbiobgh/...`,
`E:/Soy_GDM_Brazil/...`) and are not portable as-is. Resolve per model, in this order:

1. **`wandb-registry-sleap-roots-labels`** — a separate, dedicated wandb registry (distinct
   from `sleap-roots-models`) holding 8 named label collections, several already re-embedded
   with `images_embedded: True` (fully portable, no path resolution needed) specifically to fix
   this exact problem. Match to a `ModelCard` by species/root-type/node-count (collection
   descriptions make this legible, e.g. "Soybean lateral root labels (4 nodes...)" →
   `soybean-cylinder-lateral-age2-8`). There is currently no programmatic join between the two
   registries (see §5) — the mapping is curated by hand for this change.
2. **Model bundle's own `labels_gt.val.slp` + path relinking** — for models not covered by
   (1). Confirmed working: `D:/SLEAP/` → `Z:/users/eberrigan/SLEAP/` resolves exactly (verified
   by loading a real frame's image after `Labels.replace_filenames()`). Other original-machine
   prefixes (`C:/Users/pbiobgh`, `D:/FNRice*`, `E:/Soy_GDM_Brazil`, `F:/Soy_GDM_Brazil`) are not
   yet known to resolve; check `Z:\users\eberrigan\SLEAP\SLEAP_Rice` and
   `Z:\users\eberrigan\SLEAP\SLEAP_Soy` (pointed to directly by the user) during implementation.
3. **Documented gap.** Not every one of the 13 production `ModelCard`s is expected to resolve.
   The harness reports what it could and couldn't verify — no silent coverage claims.

### 3. Metric engine: reuse `sleap_nn.evaluation`, not custom code

`sleap_nn.evaluation.run_evaluation(ground_truth_path, predicted_path, ...)` (a vendored port of
classic SLEAP's own `sleap.nn.evals`) already provides instance matching, distance metrics, OKS,
VOC mAP/mAR, PCK, and visibility precision/recall. No custom keypoint-matching/distance code.

**Do not use OKS-derived *scores* for the gate; OKS-based *matching* is fine and necessary.**
`sleap-roots-training`#17 (open, assigned) found `oks_map`/VOC OKS metrics collapse near-zero on
the root-keypoint domain regardless of model quality — likely inherited human/animal-pose sigma
constants, uncalibrated for roots. Their team already works around this by using
`distance_metrics` and `visibility_metrics` instead of OKS-based ones, while still running the
library's normal OKS-based instance matching. This harness follows the same precedent:
`match_method="oks"` at the library's own maximally permissive default `match_threshold=0.0`
(any OKS > 0 counts as a correspondence, decoupling "which instances correspond" from "how good
is the match"), gating on `distance_metrics.p95` (or `.avg`) and `visibility_metrics.recall` —
never `mOKS`/`voc_metrics`.

Centroid-mode matching (`match_method="centroid"`) was considered and rejected during
implementation: it is designed for single-node/centroid-only predictions (e.g. a
centroid-detection model), not per-node distance between two full multi-node skeletons.
Confirmed empirically — trying it against a real 2-node skeleton produced a nonzero distance
even for two exactly identical instances, an unusable result for this harness's purpose.

### 4. Reference number: recompute classic-SLEAP's own eval when possible, else read the stored one

For apples-to-apples settings, prefer recomputing classic-SLEAP's number via
`run_evaluation(labels_gt.val.slp, labels_pr.val.slp)` using the *same* `match_method`/
`match_threshold` the harness uses for sleap-nn, when `labels_pr.val.slp` is present in the
bundle. Fall back to the stored `metrics.val.npz` when `labels_pr` is absent.

That stored file is pickled by classic SLEAP's own (TensorFlow-based) `sleap` package, which
this repo does not and should not depend on — `sleap_nn.evaluation.load_metrics()` raises
`ModuleNotFoundError: No module named 'sleap'` on a real file with only `sleap_nn` installed.
Disassembling the pickle opcodes of a real stored file showed it references exactly one custom
class, `sleap.instance.PointArray` — not deep framework code. A minimal, temporary shim
(bare `numpy.ndarray` subclasses registered under fake `sleap`/`sleap.instance` modules, removed
immediately after reading) unpickles it successfully — verified against all 13 live production
models' real stored files, not just one. This gives **full stored-reference coverage** rather
than "usually unavailable": the "no reference at all" gap only applies to a model with neither
`labels_pr.val.slp` nor a readable `metrics.val.npz`, expected to be rare. One more correction
found this way: the stored file uses classic SLEAP's own flat, dot-separated key schema
(`dist.p95`, `vis.recall`, ...), not `sleap_nn.evaluation`'s nested `distance_metrics`/
`visibility_metrics` dicts — the two schemas must be read differently even though both ultimately
populate the same `ParityMetrics` shape.

### 5. Coverage: all 13 production `ModelCard`s, not a curated field-experiment subset

The live `wandb-registry-sleap-roots-models` registry currently holds 13 `production`-aliased
cards across 5 species (rice, arabidopsis, pennycress, canola, soybean; modes `cylinder` and
`multiplant cylinder`; root types primary/lateral/crown — rice is the only crown coverage).
Reference set = the ground truth resolved (per §2) for each of these 13, not a hand-picked
sample of field-experiment scans. This automatically tracks the registry's real species/root-
type spread instead of requiring manual curation of "representative" scans.

### 6. Tolerance: empirical, set after measuring

Run the harness across all 13 models first to observe the real sleap-nn-vs-classic-SLEAP
distance/recall deltas, then set the tolerance as a documented margin above the observed
baseline (e.g., "observed max Δp95 = 0.8px → tolerance Δp95 ≤ 2px"). Not an a priori guess.

### 7. `LabelCard`-shaped ground-truth manifest (bump the contracts pin)

`sleap-roots-contracts` `0.1.0a6` (on PyPI, confirmed) added `LabelCard` — a typed contract
mirroring `ModelCard`, built by `sleap-roots-training` specifically for "which labels trained
this model" provenance (`sleap-roots-training`#10, merged upstream; backfill of the real shared
registry is `sleap-roots-training`#11, separately scoped and not started — see §8 for why this
change doesn't wait on it). Bump predict's own `sleap-roots-contracts` pin `0.1.0a5` → `0.1.0a6`
and shape this change's curated ground-truth manifest (§2) as `LabelCard`-typed records —
**stored locally in this repo** (not published to the shared `wandb-registry-sleap-roots-labels`
registry; that publish path belongs to `sleap-roots-training`#11/#26). Unrecoverable provenance
fields are marked `None`, per #11's own stated policy ("mark unrecoverable fields honestly...
do not invent values") — this keeps the manifest forward-compatible with a future migration to
a live `LabelCard` query once the real backfill lands.

### 8. Why this doesn't wait on `sleap-roots-training`#10/#11

`LabelCard` the *contract* is merged and released (`0.1.0a6`, confirmed on PyPI 2026-07-31).
The *backfilled, normalized shared registry* is not: `sleap-roots-training`#10 remains open,
blocked on #26 (porting `/build-labeling-package`'s generator scripts out of a personal vault —
zero comments, not started); #11 (backfill) is blocked on #10 and is explicitly scoped as
open-ended archaeology ("expect gaps... do not invent values"), with no fixed ETA; and
`sleap-roots-training` itself still pins contracts `0.1.0a3`. Waiting is unbounded. This change
proceeds now with the curated, explicitly-interim mapping from §2/§7, and cross-links
sleap-roots-pipeline#15 and this change on `sleap-roots-training`#10/#11/#22 so that team knows
there's a live downstream consumer once the real backfill exists.

### 9. predict#32 is already resolved in code — no fix needed, just close it

Bumping the contracts pin to `0.1.0a6` retypes `ModelCard.mode` to a strict `Mode` Literal.
predict#32 asks whether `WandbRegistrySource.list_cards()` should skip-with-warning or hard-fail
when one artifact's `mode` fails validation. Checked directly against the current code
(`model_registry.py:195-230`, `_collect_cards`) and the current `model-management` spec (already
has the full requirement + a "A non-conforming artifact is skipped with a warning" scenario):
**skip-with-warning is already implemented and already spec'd**, isolating per-artifact
`ModelCard.model_validate` failures with a logged warning while letting credential/network
errors still propagate fail-loud. This predates #32 (present since the original warm-worker
PR) — #32's premise ("the spec currently says nothing about a malformed artifact") was already
stale when written. Combined with n-tehranchi's live-registry enumeration on #32 (all 13
`production`-aliased cards already carry an in-vocabulary `mode`; `list_cards()` filters to the
alias before validating, so the pin bump introduces zero new failures), there is nothing to
implement here. This change closes predict#32 with a comment citing the existing code/spec,
rather than bundling any code change.

### 10. Harness form: a new `parity` pytest marker, not a one-off script

Follows the existing `gpu`/`acceptance`/`wandb` convention (`pyproject.toml` markers,
deselected by default, gated on env vars — network-share root for path-relinked models,
`WANDB_API_KEY` for registry/labels-registry access). This makes the parity check a standing
regression guard against future sleap-nn/sleap-roots-contracts upgrades, not just a one-time
report. The metric-computation logic lives in a new, reusable `sleap_roots_predict/parity.py`
module wrapping `run_evaluation` — predict#8 (peak_threshold tuning, which explicitly says it
should "reuse the same reference scan set intended for the A3-predict parity harness") can
import this module rather than duplicating instance-matching/metric code.

## Behavior change

None for existing callers of `sleap_roots_predict`'s public API. New additions only:
- `sleap_roots_predict/parity.py` (new module: ground-truth resolution + metric wrapper).
- A `parity` pytest marker + new gated test module.
- `sleap-roots-contracts` pin `0.1.0a5` → `0.1.0a6` (adds `LabelCard`; retypes `ModelCard.mode`
  — confirmed safe against the live registry and the existing skip-with-warning isolation, §9).
  No `model-management` spec delta needed — `list_cards()`'s malformed-artifact behavior is
  unchanged (already correct) and the spec already documents it.

## Components touched

- `pyproject.toml` / `uv.lock` — contracts version bump + relock; new `parity` marker.
- `sleap_roots_predict/parity.py` (new) — ground-truth resolution (labels-registry lookup +
  path-relinking fallback), `LabelCard`-shaped local manifest, `run_evaluation` wrapper,
  tolerance constants.
- `tests/test_parity.py` (new, `@pytest.mark.parity`) — gated on env vars pointing at the
  network share + `WANDB_API_KEY`; skips cleanly without them, per the `acceptance`/`wandb`
  precedent.
- A checked-in ground-truth manifest (small JSON/YAML — `LabelCard`-shaped records, one per
  resolvable production `ModelCard`, with explicit `None`s for unrecoverable fields and an
  explicit gap list for unresolvable models).
- `openspec/specs/` — new or modified capability spec(s) for the parity harness (and the
  `model-management` capability's `list_cards()` robustness behavior).
- Cross-repo: comments on `sleap-roots-training`#10/#11/#22 noting this change as a downstream
  `LabelCard` consumer; closes predict#32 (comment-only, citing the pre-existing
  skip-with-warning implementation, §9); closes sleap-roots-pipeline#15 (after recording the
  decided tolerance + measured results); cross-links predict#8.

## Testing approach

- TDD per task in the OpenSpec `tasks.md`. Ground-truth resolution, `LabelCard` manifest
  construction, and the `run_evaluation` wrapper are all unit-testable against small fixtures
  without network/GPU access.
- The actual multi-model parity run is real-data/network-gated (`parity` marker), mirroring
  `acceptance`/`wandb` — not run in CI, run locally/on-demand, `-m parity -s` to see the
  measured numbers used to set the tolerance.
- After the pin bump, re-run the existing `model_registry.py` test suite unmodified — it already
  covers the skip-with-warning path (§9); this is a regression check, not new test-writing.
- Full `/pre-merge` gate (format, lint, test, build; GPU tests run locally per the standing
  requirement) before opening the PR.

## Out of scope

- `sleap-roots-training`#10/#11/#26 themselves (LabelCard backfill, `publish-labels`, vault
  script porting) — a different repo's work, cross-linked but not performed here.
- predict#8's actual `peak_threshold` sweep — this change provides the reusable metrics module
  and reference set #8 says it wants to share, not the sweep itself.
- Publishing anything to the shared `wandb-registry-sleap-roots-labels` registry — this
  change's manifest is local/curated only.
- Trait-summary delta as a *gated* metric — computed informationally only if time allows;
  keypoint distance + visibility recall are the gate (per the earlier scoping decision).
- Chasing full ground-truth resolution for every one of the 13 models to 100% — documented gaps
  are an acceptable outcome for this change.

## Acceptance

- `sleap_roots_predict/parity.py` exists, wraps `sleap_nn.evaluation.run_evaluation`, and is
  covered by unit tests independent of network/GPU access.
- A `parity`-marked test runs the harness across all resolvable production `ModelCard`s,
  reports classic-SLEAP-vs-sleap-nn distance/recall deltas per model, and asserts the decided
  tolerance.
- The ground-truth manifest is checked in, `LabelCard`-shaped, with explicit `None`s for
  unrecoverable fields and an explicit list of unresolved models (no silent gaps).
- predict#32 closed with a comment citing the pre-existing skip-with-warning implementation and
  spec (no code change — see §9).
- sleap-roots-pipeline#15 closed with the decided tolerance, the reference set (13 models,
  coverage/gaps noted), and the measured baseline recorded in a closing comment (drafted for
  approval first).
- `docs/bloom-integration/roadmap.md` (sleap-roots-pipeline) A3-predict row updated to reflect
  the parity gate closing (drafted for approval first).

## Cross-repo references

- **sleap-roots-pipeline**: issue #15 (this change resolves it), issue #9 (parent A3 epic),
  `docs/bloom-integration/roadmap.md` (A3-predict row, updated at the end).
- **predict** (this repo): issue #8 (peak_threshold tuning, shares the reference set + reuses
  `parity.py`), issue #32 (list_cards() robustness, closed by this change).
- **sleap-roots-training**: issue #10 (`LabelCard` contract, merged upstream — this change
  consumes the release, not the issue), issue #11 (backfill, not performed here, cross-linked),
  issue #22 (Tier 2 EPIC, cross-linked as a concrete downstream consumer), issue #17 (OKS-sigma
  miscalibration — this change avoids OKS metrics because of this finding), issue #26 (vault
  script porting, not performed here).
- **sleap-roots-contracts**: `0.1.0a6` release (adds `LabelCard`, retypes `ModelCard.mode`) —
  consumed via pin bump, not modified here.
