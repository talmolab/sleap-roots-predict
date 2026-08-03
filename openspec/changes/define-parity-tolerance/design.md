## Context

Full design rationale, alternatives considered, and the cross-repo investigation behind this
change are recorded in
`docs/superpowers/specs/2026-08-03-define-parity-tolerance-design.md` (approved). This file
summarizes the decisions relevant to implementation; see that doc for the "why" behind each one.

## Goals / Non-Goals

**Goals:** decide + measure a keypoint-distance/detection-recall tolerance between sleap-nn and
classic-SLEAP inference on the *same already-trained weights*, against real ground truth, for
as many of the 13 live production `ModelCard`s as ground truth can be resolved for; ship a
reusable metrics module other work (predict#8) can build on; leave an explicit, non-silent
record of any model this couldn't cover.

**Non-Goals:** retraining or model-quality comparison (a different, already-addressed axis —
see `sleap-roots-training`'s roadmap "establish-then-reproduce-or-beat" philosophy, which
explicitly does not apply here since no retraining is involved); gating on trait-summary delta
(informational only, if computed at all); publishing anything to the shared
`wandb-registry-sleap-roots-labels` registry; performing `sleap-roots-training`#10/#11/#26.

## Decisions

1. **Ground truth = `labels_gt.val.slp`** bundled in each production model's wandb artifact (a
   real human-labeled validation split), not the legacy pipeline's own field-experiment
   predictions (an earlier, corrected framing mistake — predictions are not ground truth).

2. **Ground-truth image resolution, in priority order:**
   a. A matching collection in `wandb-registry-sleap-roots-labels` (8 named collections;
      several already carry `images_embedded: True` — fully portable).
   b. The model bundle's own `labels_gt.val.slp` + `sio.Labels.replace_filenames()` with the
      prefix map `{"D:/SLEAP": "Z:/users/eberrigan/SLEAP"}` — confirmed to resolve real frames
      for the `D:/SLEAP/SLEAP_arabidopsis` / `SLEAP_Canola_Pennycress` video pool. Check
      `Z:\users\eberrigan\SLEAP\SLEAP_Rice` and `...\SLEAP_Soy` (pointed to directly by the
      repo owner) for the remaining prefixes (`D:/FNRice*`, `C:/Users/pbiobgh`,
      `E:/Soy_GDM_Brazil`, `F:/Soy_GDM_Brazil`) during implementation — resolvability not yet
      individually confirmed for each.
   c. Otherwise: an explicit, logged gap. Never silently drop a model from the report.

3. **Metric engine: `sleap_nn.evaluation.run_evaluation`, `match_method="centroid"`.** No
   custom keypoint-matching code. OKS-based matching/scoring (`match_method="oks"`, `mOKS`,
   VOC `oks_voc.mAP`/`mAR`) is deliberately avoided — `sleap-roots-training`#17 found OKS
   metrics collapse near-zero on the root-keypoint domain regardless of model quality (likely
   uncalibrated sigma constants inherited from human/animal pose). Gate on
   `distance_metrics.p95` (or `.avg`) and `visibility_metrics.recall`.

4. **Classic-SLEAP's own number:** recompute via `run_evaluation(labels_gt.val.slp,
   labels_pr.val.slp)` with the *same* settings used for sleap-nn, when `labels_pr.val.slp` is
   present in the bundle (most, not all, model exports have it). Fall back to the bundle's
   stored `metrics.val.npz` (via `sleap_nn.evaluation.load_metrics()`) otherwise.

5. **Tolerance is measured, not guessed.** Implementation order: build the harness → run it
   across all resolvable models → record the observed distance/recall deltas → set the
   tolerance as a documented margin above that baseline → encode it as an assertion.

6. **`LabelCard`-shaped local manifest**, not a live registry query. `LabelCard`
   (`sleap-roots-contracts` `0.1.0a6`, confirmed on PyPI) is the right *shape* for this
   metadata, but the shared `wandb-registry-sleap-roots-labels` registry isn't backfilled onto
   it yet (`sleap-roots-training`#10 open, #11 not started, no fixed ETA — see the design doc's
   §8 for the full blocker chain). This change's manifest lives in this repo, typed as
   `LabelCard` records, with unrecoverable fields `None` (per #11's own stated policy) so a
   future migration to a live query is a lookup-swap, not a reshape.

7. **predict#32 needs no code change.** Verified directly against `model_registry.py:195-230`
   and the current `model-management` spec: per-artifact isolation
   (`ValidationError` → logged warning → skip, continue) is already implemented and already
   spec'd, predating #32. Close it with a comment, not a fix.

## Risks / Trade-offs

- **Coverage may be partial.** Not all 13 models' ground-truth images are known to resolve yet
  (rice/soybean paths in particular). Accepted trade-off: document gaps explicitly rather than
  block the whole gate on 100% coverage; the manifest structure supports adding resolved models
  later without redesign.
- **`labels_pr.val.slp` isn't in every bundle.** Falling back to stored `metrics.val.npz` means
  the classic-SLEAP reference number for those models was computed with whatever settings
  classic-SLEAP's own training pipeline used, not necessarily identical to this harness's
  `match_method="centroid"` settings — a slightly less apples-to-apples comparison. Flag this
  per-model in the harness's report rather than treating all 13 numbers as equally comparable.
- **One tolerance number across heterogeneous models/species** may not fit every model equally
  well. Accepted for v1 (matches the issue's ask for *a* decided tolerance); revisit
  per-model/per-species tolerances as a follow-up if the empirical spread is large.

## Migration Plan

Additive only — no existing behavior changes. The contracts pin bump is a version-only change
with no import-shape impact on existing code (confirmed via the existing `model_registry.py`
test suite, which already covers the one behavior the retype could affect).

## Open Questions

- Exact final tolerance values — filled in from the empirical run during implementation, then
  recorded in this change's task list and in the sleap-roots-pipeline#15 closing comment.
- Whether `Z:\users\eberrigan\SLEAP\SLEAP_Rice` / `SLEAP_Soy` fully resolve the remaining
  broken path prefixes — checked during implementation, not assumed here.
