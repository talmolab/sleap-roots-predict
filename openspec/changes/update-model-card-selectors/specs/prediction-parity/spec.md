## MODIFIED Requirements

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

## ADDED Requirements

### Requirement: Parity Report Entry Selection Fields

A full (non-gap) parity report entry SHALL identify the evaluated card's selection contexts as a
`selectors` list, one JSON object per card selector in the card's order, each carrying `species`,
`mode`, `age_min` and `age_max`. The entry SHALL NOT carry top-level `species`, `mode`, `age_min`
or `age_max` fields. One entry SHALL be produced per card, not per selector. Gap entries are
unaffected.

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
