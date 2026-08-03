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

### Requirement: Parity Metric Computation Avoids OKS

The system SHALL compute parity metrics between sleap-nn's predictions and classic-SLEAP's
predictions (or stored metrics) against the same resolved ground truth using
`sleap_nn.evaluation.run_evaluation` with `match_method="centroid"`. The system SHALL NOT use
OKS-based matching or scoring (`match_method="oks"`, `mOKS`, VOC OKS `mAP`/`mAR`) as a parity
signal, because OKS metrics are known to be miscalibrated for the root-keypoint domain
(collapsing near zero regardless of model quality). The gated parity signal SHALL be drawn from
`distance_metrics` (keypoint pixel distance) and `visibility_metrics` (detection precision/
recall).

#### Scenario: Metrics are computed via centroid matching

- **WHEN** the harness computes parity metrics for a resolved model
- **THEN** `run_evaluation` is called with `match_method="centroid"`, and the resulting
  `distance_metrics` and `visibility_metrics` are used as the parity signal

#### Scenario: OKS metrics are not used as a gate

- **WHEN** `run_evaluation`'s result includes OKS-derived fields (`mOKS`, `voc_metrics`)
- **THEN** the harness's pass/fail tolerance assertion does not read from those fields

### Requirement: Classic-SLEAP Reference Number

The system SHALL recompute classic-SLEAP's reference metrics via
`run_evaluation(ground_truth, labels_pr.val.slp)`, using the same `match_method` and threshold
settings used for sleap-nn's metrics, when a resolved model's bundle includes
`labels_pr.val.slp` (classic-SLEAP's own predictions on the ground truth). When
`labels_pr.val.slp` is absent, the system SHALL fall back to the bundle's stored
`metrics.val.npz`, loaded via `sleap_nn.evaluation.load_metrics()`, and SHALL mark that model's
comparison as using stored (not freshly recomputed) settings in its report.

#### Scenario: Reference number is recomputed when predictions are available

- **WHEN** a resolved model's bundle includes `labels_pr.val.slp`
- **THEN** classic-SLEAP's reference `distance_metrics`/`visibility_metrics` are computed via
  `run_evaluation` with the same settings applied to sleap-nn's metrics

#### Scenario: Reference number falls back to stored metrics

- **WHEN** a resolved model's bundle does not include `labels_pr.val.slp`
- **THEN** classic-SLEAP's reference metrics are loaded from the bundle's `metrics.val.npz`, and
  the harness's report marks that model's comparison as stored-settings rather than recomputed

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

The system SHALL assert, for each resolved model, that the measured deltas between sleap-nn's
and classic-SLEAP's `distance_metrics` and `visibility_metrics.recall` fall within a tolerance
that is documented in code and derived from an empirical baseline run of the harness (not an a
priori guess). A delta exceeding the tolerance SHALL fail the assertion for that model.

#### Scenario: A delta within tolerance passes

- **WHEN** a resolved model's measured sleap-nn-vs-classic-SLEAP delta is within the documented
  tolerance
- **THEN** the harness's assertion for that model passes

#### Scenario: A delta exceeding tolerance fails

- **WHEN** a resolved model's measured delta exceeds the documented tolerance
- **THEN** the harness's assertion for that model fails, naming the model and the measured vs.
  tolerated values
