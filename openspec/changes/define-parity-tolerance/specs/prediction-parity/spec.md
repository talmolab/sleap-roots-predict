## ADDED Requirements

### Requirement: Ground Truth Resolution Per Model

For each production `ModelCard`, the system SHALL attempt to resolve real human-labeled ground
truth in this priority order: (1) a matching collection in the `wandb-registry-sleap-roots-labels`
registry, joined by species/root-type/node-count; (2) the model's own bundled
`labels_gt.val.slp`, with its embedded video paths relinked via a configurable prefix map (e.g.
`D:/SLEAP` → a network-share root); (3) an explicit, logged gap when neither resolves. A model
whose ground truth cannot be resolved SHALL be recorded as an explicit gap in the harness's
report and SHALL NOT be silently omitted or cause the harness to fail for the other models.

#### Scenario: Ground truth resolves via the labels registry

- **WHEN** a `ModelCard`'s species/root-type/node-count matches a `wandb-registry-sleap-roots-labels`
  collection
- **THEN** that collection's labeled frames are used as ground truth for the model, and no
  path-relinking against the model's own bundle is attempted

#### Scenario: Ground truth resolves via bundled labels with path relinking

- **WHEN** no labels-registry collection matches a `ModelCard`, but its bundled
  `labels_gt.val.slp`'s embedded video paths resolve after applying the configured prefix map
- **THEN** those relinked frames are used as ground truth for the model

#### Scenario: Unresolvable ground truth is an explicit, non-fatal gap

- **WHEN** neither the labels registry nor bundled-labels path relinking resolves a `ModelCard`'s
  ground truth
- **THEN** the harness records that model as a named gap in its report, continues resolving and
  evaluating the remaining models, and does not raise

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

The system SHALL maintain a checked-in ground-truth manifest, with one record per production
`ModelCard`, shaped as a `LabelCard` (from `sleap-roots-contracts`). Provenance fields that
cannot be recovered for a given record SHALL be set to `None` rather than fabricated. The
manifest SHALL NOT be published to the shared `wandb-registry-sleap-roots-labels` registry.

#### Scenario: A manifest record has an unrecoverable field marked None

- **WHEN** a ground-truth source's provenance (e.g. `bloom_experiment_id`, `labeler`) cannot be
  determined for a manifest record
- **THEN** that `LabelCard` field is `None`, not a fabricated or guessed value

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

#### Scenario: A delta within tolerance passes

- **WHEN** a resolved model has both metrics and its measured delta is within the documented
  tolerance
- **THEN** the harness's assertion for that model passes

#### Scenario: A delta exceeding tolerance fails

- **WHEN** a resolved model has both metrics and its measured delta exceeds the documented
  tolerance
- **THEN** the harness's assertion for that model fails, naming the model and the measured vs.
  tolerated values

#### Scenario: A model with no reference is reported without a pass/fail verdict

- **WHEN** a resolved model has no classic-SLEAP reference available
- **THEN** the harness reports that model's sleap-nn metrics informationally and does not assert
  it against the tolerance
