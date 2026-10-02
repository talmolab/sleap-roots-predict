## MODIFIED Requirements

### Requirement: Model Selection From Scan Params

The system SHALL provide a pure function `choose_models(params, cards, overrides=None)` that maps
resolved scan params (`species`, `mode`, `age` read from a `ResolvedParams`) and a list of
`ModelCard`s to a `dict[RootType, ModelRef]` — at most one model per root type. For each root type
it SHALL apply, in order: an explicit override when provided; otherwise select the card(s) of that
root type that **match** the params. A card SHALL match if and only if **some single one** of its
`selectors` has `species ==` the scan species, `mode ==` the scan mode, and
`age_min <= matching age <= age_max` (inclusive) for **that same selector's** window, where the
*matching age* is defined below. The system SHALL NOT match against a card-level age window (any
minimum or maximum taken across a card's selectors) and SHALL NOT match the cross product of
different selectors' species, modes and windows. A card SHALL count once however many of its
selectors match. Exactly one matching card SHALL be selected; zero matches SHALL skip that root
type; more than one matching card SHALL raise an error identifying the ambiguity, naming both the
scan age and the matching age when they differ. The selected `ModelCard` SHALL be converted to a
`ModelRef` via `ModelCard.to_model_ref`, which copies the card's already-concrete `registry_id`,
`version`, `root_type`, and `weights_checksum` and stamps the **runtime** sleap-nn version.
`choose_models` SHALL rely on cards already carrying a concrete `version`/`weights_checksum` (it
does not resolve aliases) and SHALL perform no network access, no per-call filesystem I/O and no
logging (the runtime sleap-nn version is resolved once at import). It SHALL raise a clear error
when a required param (`species`, `mode`, or `age`) is absent.

**Past-window ages.** The scan age is coerced to an integer before it is compared. The
*window maximum* SHALL be the largest `age_max` among all selectors, on any card of any root type
in `cards` (including cards of overridden root types), whose `species` and `mode` equal the
scan's. When a window maximum exists and the scan age is greater than it, the matching age SHALL
be the window maximum; otherwise the matching age SHALL be the scan age. There SHALL be no upper
limit on how far past the window an age is clamped. Only ages above every such window SHALL be
clamped: an age below every such selector's `age_min`, or in a gap between two windows, SHALL be
matched unchanged. Because one window maximum is used for the whole species and mode, a root type
none of whose cards reaches that maximum SHALL be skipped, never matched through a lower window.
`choose_models` SHALL NOT modify `params`: `params.values` and `ResolvedParams.param_hash`, and
therefore any idempotency key derived from them, SHALL keep the real scan age. The clamp SHALL NOT
be recorded in the returned `ModelRef`s or the `PredictionManifest`: no field is added to either.
The matching age only affects root types that are not overridden.

The accepted `ResolvedParams` SHALL be accepted regardless of which module produced it — in
particular, a `ResolvedParams` built by `sleap_roots_contracts.resolve_params` (the Bloom scan
metadata → params oracle promoted into `sleap-roots-contracts`, consumed by predict rather than
implemented by it) SHALL drive selection identically to a hand-built one, so a Bloom scan
metadata row selects production models end-to-end (metadata → params → model), across the
predict/contracts repo boundary.

#### Scenario: Exactly one match selects a model per root type

- **WHEN** `choose_models` is called with params and cards where each present root type has exactly
  one card with a selector matching `species`, `mode`, and a matching age within that selector's
  `[age_min, age_max]`
- **THEN** it returns a `dict[RootType, ModelRef]` with one `ModelRef` per matched root type, each
  carrying the card's concrete `version`/`weights_checksum` and the runtime `sleap_nn_version`

#### Scenario: Age window boundaries are inclusive

- **WHEN** the matching age equals a selector's `age_min` or its `age_max`, and that selector's
  species and mode match
- **THEN** the card carrying that selector matches (the window is inclusive at both ends)

#### Scenario: Age below every species/mode-matching selector's window does not match

- **WHEN** the scan `age` is below the `age_min` of every selector whose species and mode match the
  scan
- **THEN** no card is selected through those selectors (a younger-than-window scan is not clamped;
  see bloom#994)

#### Scenario: A card matches through any one of its selectors

- **WHEN** a card carries selectors (canola, cylinder, 2–13) and (pennycress, cylinder, 2–14), and
  the scan is pennycress, cylinder, age 14
- **THEN** the card matches, through its pennycress selector

#### Scenario: Age is compared against the matching selector's window only

- **WHEN** a card carries selectors (canola, cylinder, 5–13) and (pennycress, cylinder, 2–14), and
  the scan is canola, cylinder, age 3
- **THEN** the card does not match, even though another of its selectors' windows includes 3

#### Scenario: Disjoint windows of one species are not merged

- **WHEN** a card carries selectors (canola, cylinder, 2–5) and (canola, cylinder, 10–13), and the
  scan is canola, cylinder, age 7
- **THEN** the card does not match, because no single selector's window contains 7

#### Scenario: Selectors are never combined across a card

- **WHEN** a card carries selectors (canola, cylinder, 2–13) and (arabidopsis, multiplant
  cylinder, 2–14), and the scan is canola, multiplant cylinder, age 5 (inside both windows)
- **THEN** the card does not match, because no single selector matches both species and mode

#### Scenario: Overlapping selectors on one card are one match

- **WHEN** exactly one card of a root type matches, through two of its selectors at once
- **THEN** that card is selected and no ambiguity error is raised

#### Scenario: Zero matches skips the root type

- **WHEN** no card matches for a given root type (e.g. a species with no crown model)
- **THEN** that root type is absent from the returned mapping (skipped, not an error); if no root
  type matches, the returned mapping is empty

#### Scenario: Ambiguous match raises

- **WHEN** more than one card matches the same root type for the given params
- **THEN** `choose_models` raises an error identifying the ambiguous root type

#### Scenario: Two cards matching through different selectors are ambiguous

- **WHEN** two different cards of one root type each match the params through one of their
  selectors
- **THEN** `choose_models` raises the ambiguity error (the raise is not relaxed for selector-shaped
  cards)

#### Scenario: Explicit override bypasses matching

- **WHEN** an explicit override `ModelRef` is provided for a root type
- **THEN** that override is used for the root type and the card-matching filter is not applied to it,
  even when no card would match that root type, and even when the scan age is past the window

#### Scenario: Missing required param raises

- **WHEN** `choose_models` is called with params missing `species`, `mode`, or `age`
- **THEN** it raises a clear error naming the missing param

#### Scenario: A resolved Bloom row selects the expected model(s)

- **WHEN** `choose_models(resolve_params(row), cards)` is called for a Bloom row whose
  normalized `species`/`mode` and coerced `age` (resolved by
  `sleap_roots_contracts.resolve_params`) match a small real `ModelCard` list
- **THEN** it returns the expected `ModelRef` per matching root type (the metadata → params →
  model round trip), without raising a missing-param error

#### Scenario: A past-window scan is matched at its species' window maximum

- **WHEN** the cards are shaped like the production catalog (cpa-primary: canola 2–13,
  pennycress 2–14, arabidopsis 2–14; soybean-primary 2–8; rice-younger-primary 2–5;
  canola-lateral: canola 2–13, pennycress 2–14; arabidopsis-lateral 2–14; soybean-lateral 2–8;
  rice-younger-crown 2–5; rice-older-crown 6–10; all in mode cylinder, and cpa-primary and
  arabidopsis-lateral also carry arabidopsis, multiplant cylinder, 2–14) and the scan is cylinder
  at arabidopsis day 28, soybean day 10, canola day 14 or pennycress day 15
- **THEN** it selects, respectively: cpa-primary + arabidopsis-lateral; soybean-primary +
  soybean-lateral; cpa-primary + canola-lateral; cpa-primary + canola-lateral — the same refs as
  that species at its window maximum (14, 8, 13, 14)

#### Scenario: A past-window multiplant scan is clamped within its own mode

- **WHEN** the cards are shaped like the production catalog and the scan is arabidopsis,
  multiplant cylinder, day 28
- **THEN** it is matched at 14 (the multiplant cylinder window maximum) and selects cpa-primary +
  arabidopsis-lateral, as an in-window multiplant scan does; the trait extractor may still
  reject multi-plant scans downstream

#### Scenario: One window is used for the whole species, not per root type

- **WHEN** the cards are shaped like the production catalog and the scan is rice, cylinder, day 18
- **THEN** it selects only `rice-older-crown`; it selects no primary model and never
  `rice-younger-primary` or `rice-younger-crown`, because rice's window maximum (10) is used for
  every root type

#### Scenario: The window maximum is scoped by species and mode

- **WHEN** one card carries (canola, cylinder, 2–13), another root type's card carries (canola,
  multiplant cylinder, 2–20), and the scan is canola, cylinder, age 15
- **THEN** the canola, cylinder card is selected (matched at 13, not 20)

#### Scenario: Overridden root types still count toward the window maximum

- **WHEN** the primary card carries (rice, cylinder, 2–10), the lateral card carries (rice,
  cylinder, 2–5), primary is overridden, and the scan is rice, cylinder, age 18
- **THEN** lateral is skipped (rice's window maximum is 10, which the lateral card's 2–5 window does
  not contain) and only the override is returned

#### Scenario: In-window scans are not clamped

- **WHEN** the scan age is within some selector window for its species and mode, or equal to the
  window maximum (e.g. arabidopsis day 10 or 14, rice day 4, canola day 13)
- **THEN** the selected refs are those whose selectors contain the scan age itself (e.g.
  arabidopsis day 10: cpa-primary + arabidopsis-lateral)

#### Scenario: A species or mode with no cards is not clamped

- **WHEN** no selector on any card has the scan's species and mode (e.g. a species with no cards,
  a species with cards only in another mode, or an empty `cards` list), at any age
- **THEN** there is no window maximum and the returned mapping holds only overrides (empty when
  there are none)

#### Scenario: The real age is kept in params

- **WHEN** a scan is clamped (e.g. arabidopsis day 28 matched at 14)
- **THEN** `params.values["age"]` is still 28 after `choose_models` returns, `param_hash` is the
  hash of the real age, and `choose_models` has logged nothing

#### Scenario: Ambiguity is checked at the matching age

- **WHEN** a scan is clamped and two cards of one root type both match at the matching age
- **THEN** `choose_models` raises the ambiguity error, naming the scan age and the matching age

#### Scenario: Clamping never selects through a lower window

- **WHEN** two lateral cards carry (arabidopsis, cylinder, 2–10) and (arabidopsis, cylinder,
  2–14), and the scan is arabidopsis, cylinder, age 28
- **THEN** only the 2–14 card is selected and no ambiguity error is raised

## ADDED Requirements

### Requirement: Past-Window Matching Age Helper

The system SHALL provide a pure function `past_window_age(params, cards, overrides=None)` in
`sleap_roots_predict.model_selection` (module-level; not exported from the package) that returns
the matching age, as defined by "Model Selection From Scan Params", when the scan age is above its
species and mode's window maximum **and** some root type that is not overridden has a card with a
selector for the scan's species and mode; otherwise it SHALL return `None`. It SHALL validate
`params` exactly as `choose_models` does (missing `species`/`mode`/`age`, non-integer `age`)
before inspecting `cards`, raising the same errors, and SHALL perform no logging, network access
or filesystem I/O. When it returns an age, `choose_models` matches the scan's non-overridden root
types at that age; callers use it to decide whether to warn.

#### Scenario: A past-window scan returns its window maximum

- **WHEN** the cards are shaped like the production catalog and the scan is arabidopsis, cylinder,
  day 28 (or string age `"28"`)
- **THEN** `past_window_age` returns 14

#### Scenario: Window maximum scoped by mode and counting overridden root types

- **WHEN** the scan is canola, cylinder, 15 over cards (canola, cylinder, 2–13) and (canola,
  multiplant cylinder, 2–20); or rice, cylinder, 18 with an overridden primary card (rice, 2–10)
  and a lateral card (rice, 2–5)
- **THEN** it returns 13, respectively 10

#### Scenario: Not clamped returns None

- **WHEN** the scan age is within or equal to the window maximum (arabidopsis 10 or 14, rice 4,
  canola 13), below every window (arabidopsis 1), or no card has a selector for the scan's species
  and mode (a species with no cards, cards only in another mode, or `cards=[]`)
- **THEN** it returns `None`

#### Scenario: Every species/mode root type overridden returns None

- **WHEN** the scan is past the window but every root type with a selector for the scan's species
  and mode is overridden (e.g. rice 18, primary card rice 2–10 overridden, the only lateral card
  for arabidopsis)
- **THEN** it returns `None`

#### Scenario: Invalid params raise even with no cards

- **WHEN** `past_window_age` or `choose_models` is called with a non-integer age or a missing
  required param and `cards=[]`
- **THEN** it raises the same `ValueError` as `choose_models` does today
