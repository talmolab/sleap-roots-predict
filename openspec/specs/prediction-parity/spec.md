# prediction-parity Specification

## Purpose
TBD - created by archiving change define-parity-tolerance. Update Purpose after archive.
## Requirements
### Requirement: Ground Truth Resolution Per Model

For each production `ModelCard`, the system SHALL attempt to resolve real human-labeled ground
truth in this priority order: (1) a matching collection in the `wandb-registry-sleap-roots-labels`
registry, found by a caller-supplied lookup that receives the card (the join criterion — e.g.
root-type, node-count, and the species of one of the card's `selectors` — is the lookup's); (2)
the model's own bundled `labels_gt.val.slp`, with its embedded video paths relinked via a
configurable prefix map (e.g. `D:/SLEAP` → a network-share root); (3) a basename search across a
configurable search root, for bundles whose video paths were reorganized rather than just moved
under a new prefix; (4) an explicit, logged gap when none resolve. Resolution SHALL accept an
optional `Selector` naming which of the card's selection contexts the ground truth belongs to, and
SHALL validate it against the card **before any tier is attempted**: a selector not equal to one of
the card's `selectors` SHALL raise a `ValueError` naming the card. The supplied selector SHALL be
used only for that validation and for tier (3)'s age step; tiers (1) and (2) do not consult it, so a
caller needing a selector-specific labels-registry join SHALL bind it into the lookup it supplies.
Resolution SHALL be tracked at
the **frame level**, not only per model: tiers (2) and (3) SHALL keep whichever labeled frames
actually resolve and SHALL NOT require every frame in a model's ground truth to resolve for that
model to count as resolved. A model whose ground truth cannot be resolved at all SHALL be recorded
as an explicit gap in the harness's report, tagged `gap_stage="resolution"` (distinguishing it from
an isolated evaluation failure — see the Reusable Multi-Model Harness Runner requirement's
`gap_stage="evaluation"`), and SHALL NOT be silently omitted or cause the harness to fail for the
other models.

#### Scenario: Ground truth resolves via the labels registry

- **WHEN** the caller-supplied labels-registry lookup returns a collection for a `ModelCard`
- **THEN** that collection's labeled frames are used as ground truth for the model, and no
  path-relinking or basename search against the model's own bundle is attempted

#### Scenario: Ground truth resolves via bundled labels with path relinking

- **WHEN** no labels-registry collection matches a `ModelCard`, but one or more of its bundled
  `labels_gt.val.slp`'s embedded video paths resolve after applying the configured prefix map
- **THEN** the frames whose video resolved are used as ground truth for the model, and the
  harness records how many of the model's total frames resolved

#### Scenario: Ground truth resolves via basename search when relinking doesn't apply

- **WHEN** neither the labels registry nor prefix-map relinking resolves a `ModelCard`'s ground
  truth, but one or more of its videos' basenames are found (and unambiguously disambiguated,
  per the Basename Search Disambiguation requirement) under a configured search root
- **THEN** the frames whose video resolved are used as ground truth for the model, and the
  harness records how many of the model's total frames resolved

#### Scenario: Unresolvable ground truth is an explicit, non-fatal gap

- **WHEN** none of the labels registry, bundled-labels path relinking, or basename search
  resolves even one frame of a `ModelCard`'s ground truth
- **THEN** the harness records that model as a named gap tagged `gap_stage="resolution"` in its
  report, continues resolving and evaluating the remaining models, and does not raise

#### Scenario: A selector not on the card is rejected before any tier

- **WHEN** ground-truth resolution is given a `Selector` not equal to any of the card's
  `selectors`, even though the labels-registry lookup would resolve the card
- **THEN** it raises a `ValueError` naming the card, and no tier is attempted

### Requirement: Basename Search Disambiguation

The system SHALL disambiguate multiple same-basename candidates found during basename search
(Ground Truth Resolution Per Model, tier 3) in this order, stopping at the first step that
leaves exactly one candidate: (1) an exact, normalized match on the immediate parent folder
name; (2) among remaining candidates, one whose path contains a day/age hint falling inside the
**resolved selector's** `[age_min, age_max]`; (3) among remaining candidates, the one sharing the
most normalized path segments with the broken path. The resolved selector SHALL be the selector
supplied to ground-truth resolution, else the card's only selector when it carries exactly one.
When neither exists (a multi-selector card with no selector supplied), step (2) SHALL be skipped
— the system SHALL NOT substitute a window taken across the card's selectors. A step that leaves
more than one candidate SHALL narrow the pool passed to the next step (or leave it unchanged when
it matched none); a tie remaining after step (3) SHALL be treated as an unresolved candidate for
that video (contributing to the model's unresolved-frame count) rather than selecting one
arbitrarily.

#### Scenario: A single candidate is unambiguous

- **WHEN** a basename search returns exactly one candidate for a video
- **THEN** that candidate is used without further disambiguation

#### Scenario: Parent folder name disambiguates same-basename candidates

- **WHEN** a basename search returns multiple candidates, and exactly one candidate's immediate
  parent folder name matches the broken path's parent folder name (normalized)
- **THEN** that candidate is used

#### Scenario: Age hint disambiguates when parent names don't match

- **WHEN** parent-folder-name matching leaves more than one candidate, a selector is resolved, and
  exactly one candidate's path contains a day/age hint within that selector's age range
- **THEN** that candidate is used

#### Scenario: A single-selector card uses its sole selector's window

- **WHEN** parent-folder-name matching leaves more than one candidate for a card carrying exactly
  one selector, and no selector was supplied
- **THEN** the age step uses that selector's window

#### Scenario: A supplied selector chooses the window on a multi-selector card

- **WHEN** a card carries selectors with windows 2–5 and 10–13, candidates lie under `Day3` and
  `Day11` folders, parent-folder-name matching leaves both candidates, and the selector with window
  10–13 is supplied
- **THEN** the `Day11` candidate is used

#### Scenario: No resolved selector skips the age step

- **WHEN** parent-folder-name matching leaves more than one candidate for a card carrying several
  selectors, and no selector was supplied
- **THEN** the age step is skipped and disambiguation continues with path-segment scoring, so a
  candidate is never chosen by a window the ground truth's species may not have

#### Scenario: A genuine tie resolves to no match, not a guess

- **WHEN** path-segment scoring still leaves more than one candidate tied
- **THEN** that video is treated as unresolved (its frame does not count toward the model's
  resolved frames), and the system does not guess

### Requirement: Parity Metric Computation Avoids OKS Scores

The system SHALL compute parity metrics between sleap-nn's predictions and classic-SLEAP's
predictions (or stored metrics) against the same resolved ground truth using
`sleap_nn.evaluation.run_evaluation` with `match_method="oks"` at the library's permissive
default `match_threshold=0.0` (any OKS-based correspondence above zero counts as a match,
decoupling which instances correspond from how good the match is). The system SHALL NOT read
OKS-derived score fields (`mOKS`, VOC OKS `mAP`/`mAR`) as a parity signal, because those scores
are known to be miscalibrated for the root-keypoint domain (collapsing near zero regardless of
model quality) — the underlying OKS-based *matching* itself is not affected by this and is used
normally. The gated parity signal SHALL be drawn from `distance_metrics` (keypoint pixel
distance) and `visibility_metrics` (detection precision/recall) only. `match_method="centroid"`
SHALL NOT be used for this comparison — it is designed for single-node/centroid-only
predictions, not per-node distance between two full multi-node skeletons.

#### Scenario: Metrics are computed via OKS-based matching

- **WHEN** the harness computes parity metrics for a resolved model
- **THEN** `run_evaluation` is called with `match_method="oks"`, and the resulting
  `distance_metrics` and `visibility_metrics` are used as the parity signal

#### Scenario: OKS scores are not used as a gate

- **WHEN** `run_evaluation`'s result includes OKS-derived score fields (`mOKS`, `voc_metrics`)
- **THEN** the harness's pass/fail tolerance assertion does not read from those fields

### Requirement: Classic-SLEAP Reference Number

The system SHALL recompute classic-SLEAP's reference metrics via
`run_evaluation(ground_truth, labels_pr.val.slp)`, using the same `match_method` and threshold
settings used for sleap-nn's metrics, when a resolved model's bundle includes
`labels_pr.val.slp` (classic-SLEAP's own predictions on the ground truth). When
`labels_pr.val.slp` is absent, the system SHALL attempt to fall back to the bundle's stored
`metrics.val.npz` via `sleap_nn.evaluation.load_metrics()`. That file is pickled by classic
SLEAP's own (TensorFlow-based) `sleap` package, which this system SHALL NOT depend on; the
system SHALL instead read it under a minimal, temporary unpickling shim (bare `numpy.ndarray`
stand-ins for the one legacy class the pickle references, removed immediately after reading —
not a persistent dependency on the legacy package) and SHALL translate its flat, dot-separated
key schema (e.g. `dist.p95`, `vis.recall`) into the same shape used for the recomputed case. If
the file still cannot be read even with the shim (or is absent), the system SHALL treat
classic-SLEAP's reference as unavailable for that model (not raise), and SHALL report that
model's sleap-nn metrics with an explicit "no reference available" marker rather than a
comparison.

#### Scenario: Reference number is recomputed when predictions are available

- **WHEN** a resolved model's bundle includes `labels_pr.val.slp`
- **THEN** classic-SLEAP's reference `distance_metrics`/`visibility_metrics` are computed via
  `run_evaluation` with the same settings applied to sleap-nn's metrics

#### Scenario: Reference number falls back to stored metrics when readable

- **WHEN** a resolved model's bundle does not include `labels_pr.val.slp`, and its stored
  `metrics.val.npz` can be read
- **THEN** classic-SLEAP's reference metrics are loaded from that file, and the harness's report
  marks that model's comparison as stored-settings rather than recomputed

#### Scenario: Reference number is unavailable when stored metrics cannot be read

- **WHEN** a resolved model's bundle does not include `labels_pr.val.slp`, and its stored
  `metrics.val.npz` cannot be unpickled with only `sleap_nn` installed
- **THEN** the harness reports that model's sleap-nn metrics with no classic-SLEAP reference,
  and does not raise

### Requirement: LabelCard-Shaped Ground Truth Manifest

The system SHALL build ground-truth records shaped as a `LabelCard` (from
`sleap-roots-contracts`), one per (model card, labeling package) pair. Because a labeling package
covers one species, a record's `species`/`mode`/`age_min`/`age_max` SHALL be copied from exactly
one of the card's `selectors`: the selector supplied by the caller, else the card's only selector
when it carries exactly one. Building a record for a card carrying several selectors with no
selector supplied, or with a selector not equal to any of the card's `selectors`, SHALL raise a
`ValueError` naming the card rather than picking one. A fallback skeleton name, when the labels
carry none, SHALL use that resolved selector's species. Provenance fields that cannot be recovered
for a given record SHALL be set to `None` rather than fabricated. Records SHALL NOT be published to
the shared `wandb-registry-sleap-roots-labels` registry.

#### Scenario: A record has an unrecoverable field marked None

- **WHEN** a ground-truth source's provenance (e.g. `bloom_experiment_id`, `labeler`) cannot be
  determined for a record
- **THEN** that `LabelCard` field is `None`, not a fabricated or guessed value

#### Scenario: A single-selector card needs no selector

- **WHEN** a record is built for a card carrying exactly one selector, with no selector supplied
- **THEN** the record's species, mode and age window are that selector's

#### Scenario: A multi-selector card requires an explicit selector

- **WHEN** a record is built for a card carrying several selectors, with no selector supplied
- **THEN** a `ValueError` naming the card is raised, and no species is guessed

#### Scenario: A supplied selector must be on the card

- **WHEN** a record is built with a selector not equal to any of the card's `selectors`
- **THEN** a `ValueError` naming the card is raised

#### Scenario: A value-equal selector is accepted

- **WHEN** a record is built with a newly constructed `Selector` equal in value to one of the
  card's `selectors`
- **THEN** the record is built from that selector (membership is by value, not identity)

### Requirement: Parity Pytest Marker

The system SHALL provide a `parity` pytest marker for the harness's real-data, network-gated
test(s), deselected by default and in CI (mirroring the existing `gpu`/`acceptance`/`wandb`
markers). Tests under this marker SHALL skip cleanly, without hitting the network, when the
required environment variables (the network-share root and `WANDB_API_KEY`) are not set.

#### Scenario: Parity tests skip without configuration

- **WHEN** the `@pytest.mark.parity` tests run with the required environment variables unset
- **THEN** they skip at collection time rather than failing or hitting the network

#### Scenario: Parity tests are deselected by default

- **WHEN** the default test suite (no `-m` filter override) runs
- **THEN** `@pytest.mark.parity` tests are not collected/run, matching the `gpu`/`acceptance`/
  `wandb` precedent

### Requirement: Documented, Enforced Tolerance

The system SHALL assert, for each resolved model that has both a sleap-nn metric and a
classic-SLEAP reference metric (per the Classic-SLEAP Reference Number requirement), that the
measured deltas between them fall within a tolerance that is documented in code and derived
from an empirical baseline run of the harness (not an a priori guess). A delta exceeding the
tolerance SHALL fail the assertion for that model. A model with no classic-SLEAP reference
available SHALL NOT be asserted against the tolerance and SHALL NOT count as a pass or a
failure — it is reported informationally only.

The `distance_p95` tolerance SHALL be relative to the classic-SLEAP reference value
(`|sleap_nn.distance_p95 - reference.distance_p95| / reference.distance_p95`), not a fixed
pixel threshold, because per-model intrinsic localization difficulty varies by an order of
magnitude across the production registry for reasons independent of engine parity (e.g. root
complexity/branching increasing with plant age — confirmed both by this harness's own measured
instance-density and by the published localization-error growth-stage effect in Berrigan et al.
2024, 10.34133/plantphenomics.0175). The `visibility_recall` tolerance SHALL be directional: it
SHALL only fail when sleap-nn's recall is lower than the reference's by more than the tolerance;
sleap-nn scoring higher than the reference SHALL NOT fail regardless of magnitude.

#### Scenario: A delta within tolerance passes

- **WHEN** a resolved model has both metrics, its `distance_p95` relative delta is within the
  documented relative tolerance, and its `visibility_recall` is not lower than the reference by
  more than the documented tolerance
- **THEN** the harness's assertion for that model passes

#### Scenario: A relative distance delta exceeding tolerance fails

- **WHEN** a resolved model's `distance_p95` relative delta (as a fraction of the reference
  value) exceeds the documented tolerance
- **THEN** the harness's assertion for that model fails, naming the model and the measured vs.
  tolerated relative deltas

#### Scenario: sleap-nn recall scoring higher than the reference never fails

- **WHEN** a resolved model's sleap-nn `visibility_recall` is higher than the classic-SLEAP
  reference's, by any margin
- **THEN** the harness's assertion for that model does not fail on the recall check

#### Scenario: A model with no reference is reported without a pass/fail verdict

- **WHEN** a resolved model has no classic-SLEAP reference available
- **THEN** the harness reports that model's sleap-nn metrics informationally and does not assert
  it against the tolerance

### Requirement: Reusable Multi-Model Harness Runner

The system SHALL provide a runner that evaluates a sequence of `ModelCard`s by looping the
single-card evaluation pipeline (`evaluate_model_card`), and SHALL persist the accumulated
entries as one JSON report at a caller-supplied path, in input-card order, returning that path.

A single card's evaluation failure (raised while converting, materializing, or evaluating that
one card) SHALL be isolated: it SHALL be logged as a warning naming the card, and recorded as a
gap entry identifying the card (`registry_id`/`version`), the failure (the exception type and
message), and a `gap_stage` of `"evaluation"` — distinguishing it from a ground-truth-resolution
gap entry (per the Ground Truth Resolution Per Model requirement, whose gap entries SHALL carry
`gap_stage="evaluation"`'s counterpart, `gap_stage="resolution"`). The two gap kinds SHALL NOT be
collapsed into an identical, unlabeled shape. Evaluation SHALL continue for the remaining cards.

This per-card isolation is scoped to exactly the per-card work it wraps: it SHALL NOT be relied
upon to distinguish a systemic failure (e.g. invalid registry credentials, an unreachable
configured search root) from a genuine per-card resolution gap, since such failures may
surface through the same call path as an ordinary per-card gap. The system's protection against
a systemic failure silently overwriting a previously-persisted report is instead: the runner
SHALL NOT overwrite an existing report already present at the target path when no card produced
a non-gap entry — whether because every card produced a gap, or because the input sequence of
cards was empty. This check SHALL NOT prevent writing a report when no report yet exists at the
target path, regardless of how many cards produced a gap.

#### Scenario: All cards produce an entry in the persisted report

- **WHEN** the runner evaluates a list of `ModelCard`s, each resolvable
- **THEN** the persisted report contains one entry per card, in input order, in the same shape
  `build_report_entry` produces for a single card, and the runner returns the report's path

#### Scenario: A single card's evaluation failure is isolated and distinguishable

- **WHEN** one card's conversion, materialization, or evaluation raises an exception, while
  other cards evaluate normally
- **THEN** that card's entry in the persisted report is a gap entry naming the failure, tagged
  `gap_stage="evaluation"`, a warning is logged naming the card, the exception does not
  propagate out of the runner, and every other card's entry is still produced normally

#### Scenario: An all-gap or empty result does not overwrite an existing report

- **WHEN** a run produces no non-gap entries — either because every card produced a gap entry,
  or because the input sequence of cards was empty — and a report already exists at the target
  path
- **THEN** the runner raises, naming the target path, rather than overwriting that existing
  report file

#### Scenario: An all-gap or empty result still writes a first report

- **WHEN** a run produces no non-gap entries, and no report yet exists at the target path
- **THEN** the runner writes the report (containing only gap entries, or being an empty list)
  as normal, since there is no existing baseline to protect

### Requirement: Parity Report Entry Selection Fields

A full (non-gap) parity report entry SHALL identify the evaluated card's selection contexts as a
`selectors` list, one JSON object per card selector in the card's order, each carrying `species`,
`mode`, `age_min` and `age_max`. The entry SHALL NOT carry top-level `species`, `mode`, `age_min`
or `age_max` fields. One entry SHALL be produced per card, not per selector. Gap entries are
unaffected. The multi-model runner SHALL evaluate each card once, supplying no selector, so for a
multi-selector card the basename-search age step is skipped and `selectors` describes the
contexts the card serves, not the contexts its evaluated ground truth is known to cover.

#### Scenario: A multi-selector card yields one entry listing every selector

- **WHEN** a report entry is built for a card carrying two selectors
- **THEN** the entry has a `selectors` list of two objects in the card's order, no top-level
  `species`, `mode`, `age_min` or `age_max`, and it serializes with `json.dumps`

#### Scenario: A single-selector card yields a one-element list

- **WHEN** a report entry is built for a card carrying one selector
- **THEN** the entry's `selectors` list has exactly one object

#### Scenario: A persisted multi-selector report round-trips

- **WHEN** the multi-model runner evaluates a two-selector card end to end and persists the report
- **THEN** the persisted file parses with `json.loads`, the card's entry is a full entry (no
  `gap_stage`), and its `selectors` list has two objects in the card's order

