# ModelCard.selectors Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pin `sleap-roots-contracts==0.1.0a9` and migrate predict off the flat `ModelCard` onto
`ModelCard.selectors`, with a fail-loud guard for an unreadable registry, validated against real
models and the live registry before merge.

**Architecture:** `choose_models` gains an any-selector predicate (per-selector age window, no
cross product, ambiguity raise unchanged). `WandbRegistrySource` raises when every production
artifact is unreadable, and `run_batch` loads the catalog once, outside per-scan isolation, via a
new `WarmModelWorker.load_catalog()`. The parity harness resolves "which selector" through one
helper. Real-model gates (A1 selection oracle, A2 inference equivalence, B live canary) run from
committed scripts.

**Tech Stack:** Python 3.11+, pydantic 2 (contracts), pytest, uv, sleap-nn 0.3.0, sleap-io, wandb.

**Spec:** `openspec/changes/update-model-card-selectors/` (proposal, design, tasks, three spec
deltas) and `docs/superpowers/specs/2026-09-23-model-card-selectors-design.md`. OpenSpec
`tasks.md` numbers are cited as **[T x.y]**; tick them there as each lands.

## Global Constraints

- Pin exactly `sleap-roots-contracts==0.1.0a9` (`pyproject.toml:24`); `uv.lock` in the same commit.
- Do **not** adopt any a9 run-manifest API (`load_run_manifest`, `run_manifest_filename`, …).
- Do **not** change the `len(matches) > 1` ambiguity raise in `choose_models`.
- The guard raises a `ValueError` subclass; a registry with zero alias-carrying artifacts returns `[]`.
- The catalog load adds **no** `should_stop()` call (tests count them).
- Paths in code: `pathlib.Path`, strings via `.as_posix()` (lab convention).
- Stage by explicit path; never `git add -A`. No closing keyword for #34 / training#39 in any commit body.
- Never write into `Z:\users\eberrigan\pipeline_orchestration_tests\a4_poc\`.
- Never run `seed-registry --execute` without the user's explicit confirmation in the moment.
- Never post to GitHub (issues, comments) without the user approving the text first.
- The gate (run after every task):

```bash
uv run pytest -m "not gpu and not acceptance and not wandb and not parity" tests/
uv run black --check sleap_roots_predict tests scripts
uv run ruff check sleap_roots_predict/ scripts/
uv run codespell
uv lock --check
openspec validate update-model-card-selectors --strict
```

## Review Focus

- **String age at a per-selector boundary** — Bloom sends `age="14"`; on a (canola 2–13,
  pennycress 2–14) card, pennycress `"14"` must match and canola `"14"` must not. Test in Task 3.
- **Duplicate identical selectors on one card** — a producer emitting the same selector twice must
  still count as one match, not an ambiguity. Test in Task 3.
- **An alias-carrying artifact with `metadata=None`** — it must count toward the guard's
  "failed validation" total, not be silently dropped. Test in Task 4.
- **`load_catalog()` after `resolve()` already listed** — calling it second must not list again.
  Test in Task 4.
- **Mixed-case species in the new registry** (`"Canola"`) — matching is exact and
  `resolve_params` lowercases, so such a card could never match. A1's compare step asserts every
  selector species in the training dump is lowercase. Check in Task 8.

---

### Task 1: Shared card builders (green on a7)  [T 1.1–1.3]

**Files:**
- Create: `tests/card_builders.py`
- Modify: `tests/conftest.py:69-80`, `tests/test_model_selection.py:16-38`,
  `tests/test_output_contract.py:32-43`, `tests/test_param_resolution.py:23-34`,
  `tests/test_parity.py:48-61`, `tests/test_warm_worker.py:22-33`,
  `tests/test_model_registry.py:28-36,164-183`, `tests/test_batch.py:826-835`

**Interfaces:**
- Produces: `make_card(root_type, registry_id=None, *, selectors=None, species="rice",
  mode="cylinder", age_min=2, age_max=5, version="v1", weights_checksum=None,
  sleap_nn_version=None) -> ModelCard`; `raw_card_meta(*, selectors=None, species="rice",
  mode="cylinder", age_min=2, age_max=5, root_type="primary", drop=()) -> dict`.
  `selectors` is a sequence of `(species, mode, age_min, age_max)` tuples.

- [ ] **Step 1: Positive control for the verification check**

Run: `rg -nU "ModelCard(\.model_validate)?\([^)]*\b(species|mode|age_min|age_max)\s*=" tests/ | rg -c ModelCard`
Expected: `8`. Record it.

- [ ] **Step 2: Create `tests/card_builders.py` (a7 body)**

```python
"""Shared ``ModelCard`` builders for tests.

The one place that knows the card's selection shape, so a contract reshape touches only this
file. Each test module keeps a thin ``_card`` wrapper with its own defaults.
"""

from typing import Iterable, Optional, Sequence, Tuple

from sleap_roots_contracts import ModelCard

SelectorTuple = Tuple[str, str, int, int]


def _selector_tuples(
    selectors: Optional[Iterable[Sequence]],
    species: str,
    mode: str,
    age_min: int,
    age_max: int,
) -> Tuple[SelectorTuple, ...]:
    if selectors is None:
        return ((species, mode, age_min, age_max),)
    return tuple(tuple(s) for s in selectors)


def make_card(
    root_type,
    registry_id=None,
    *,
    selectors=None,
    species="rice",
    mode="cylinder",
    age_min=2,
    age_max=5,
    version="v1",
    weights_checksum=None,
    sleap_nn_version=None,
) -> ModelCard:
    """Build a ``ModelCard`` from one or more ``(species, mode, age_min, age_max)`` contexts."""
    sels = _selector_tuples(selectors, species, mode, age_min, age_max)
    assert len(sels) == 1, "contracts 0.1.0a7 cards carry exactly one selection context"
    sp, md, lo, hi = sels[0]
    return ModelCard(
        species=sp,
        mode=md,
        age_min=lo,
        age_max=hi,
        root_type=root_type,
        registry_id=registry_id or f"reg/{sp}-{root_type}",
        version=version,
        weights_checksum=weights_checksum,
        sleap_nn_version=sleap_nn_version,
    )


def raw_card_meta(
    *,
    selectors=None,
    species="rice",
    mode="cylinder",
    age_min=2,
    age_max=5,
    root_type="primary",
    drop=(),
) -> dict:
    """Build raw wandb-style card metadata; ``drop`` removes selection keys."""
    sels = _selector_tuples(selectors, species, mode, age_min, age_max)
    assert len(sels) == 1, "contracts 0.1.0a7 cards carry exactly one selection context"
    sp, md, lo, hi = sels[0]
    meta = {"species": sp, "mode": md, "age_min": lo, "age_max": hi, "root_type": root_type}
    for key in drop:
        del meta[key]
    return meta
```

- [ ] **Step 3: Route every helper through it, keeping each module's defaults**

`tests/conftest.py` (replace `_card`):

```python
from card_builders import make_card


def _card(
    root_type, registry_id, *, species="rice", version="v1", age_min=2, age_max=5
):
    """Build a ModelCard for the vendored-model LocalCardSources."""
    return make_card(
        root_type,
        registry_id,
        species=species,
        version=version,
        age_min=age_min,
        age_max=age_max,
    )
```

`tests/test_model_selection.py` (replace `_card`; drop the now-unused `ModelCard` import only if
nothing else uses it):

```python
from card_builders import make_card


def _card(
    root_type,
    *,
    species="rice",
    mode="cylinder",
    age_min=2,
    age_max=5,
    registry_id=None,
    ver="v1",
    checksum="sha",
    trained_with=None,
    selectors=None,
):
    """Build a ModelCard with sensible defaults for one root type."""
    return make_card(
        root_type,
        registry_id,
        selectors=selectors,
        species=species,
        mode=mode,
        age_min=age_min,
        age_max=age_max,
        version=ver,
        weights_checksum=checksum,
        sleap_nn_version=trained_with,
    )
```

`tests/test_output_contract.py`, `tests/test_warm_worker.py` (identical signatures to conftest's):

```python
from card_builders import make_card


def _card(
    root_type, registry_id, *, species="rice", version="v1", age_min=2, age_max=5
):
    return make_card(
        root_type,
        registry_id,
        species=species,
        version=version,
        age_min=age_min,
        age_max=age_max,
    )
```

`tests/test_param_resolution.py`:

```python
from card_builders import make_card


def _card(root_type, *, species="rice", mode="cylinder", age_min=2, age_max=5):
    """Build a ModelCard with sensible defaults for one root type."""
    return make_card(
        root_type,
        f"reg/{species}-{root_type}",
        species=species,
        mode=mode,
        age_min=age_min,
        age_max=age_max,
        weights_checksum="sha",
    )
```

`tests/test_parity.py`:

```python
def _card(
    root_type="primary",
    registry_id="reg/arabidopsis-primary",
    age_min=2,
    age_max=14,
    selectors=None,
):
    from card_builders import make_card

    return make_card(
        root_type,
        registry_id,
        selectors=selectors,
        species="arabidopsis",
        age_min=age_min,
        age_max=age_max,
        version="v0",
    )
```

`tests/test_model_registry.py` — `_card` at `:28`:

```python
from card_builders import make_card, raw_card_meta


def _card(root_type, registry_id, version="v1"):
    return make_card(root_type, registry_id, version=version)
```

and the fixtures at `:164-183`:

```python
def _good_meta(species="rice", root_type="primary"):
    """Metadata carrying every required selection field (validates to a card)."""
    return raw_card_meta(species=species, root_type=root_type)


def _malformed_artifact(registry_id="reg/bad"):
    # Missing the required ``species`` field -> pydantic ValidationError.
    return FakeArtifact(registry_id, metadata=raw_card_meta(drop=("species",)))
```

`tests/test_batch.py:826-835` (inside `test_changed_model_ref_causes_repredict`):

```python
    from card_builders import make_card
    from sleap_roots_predict.model_registry import LocalCardSource

    def _source(version):
        card = make_card("primary", "reg/rice-primary", version=version)
        return LocalCardSource([(card, native_model_dir)])
```

- [ ] **Step 4: Verify the refactor changed no assertion and left no flat construction**

Run: the gate. Expected: all pass (still on a7).
Run: `git diff -U0 tests/ | rg "^[-+]\s*assert"` — Expected: no output.
Run: Step 1's command — Expected: `1` (the builder).
Run: `rg -n "[\"'](age_min|age_max)[\"']\s*:" tests/` — Expected: only `tests/card_builders.py`.

- [ ] **Step 5: Commit**

```bash
git add tests/card_builders.py tests/conftest.py tests/test_model_selection.py \
  tests/test_output_contract.py tests/test_param_resolution.py tests/test_parity.py \
  tests/test_warm_worker.py tests/test_model_registry.py tests/test_batch.py \
  openspec/changes/update-model-card-selectors/tasks.md
git commit -m "refactor(tests): share ModelCard and registry-metadata builders (#34)"
```

---

### Task 2: Pin a9, move the builders, interim sole-selector reads  [T 2.1, 2.2, 2.3-interim, 2.6, 2.7]

**Files:**
- Create: `tests/test_contract_assumptions.py`
- Modify: `pyproject.toml:24`, `uv.lock`, `tests/card_builders.py`,
  `sleap_roots_predict/model_selection.py:80-90`, `sleap_roots_predict/parity.py:378,904-909,1068-1076`,
  `tests/test_model_registry.py:315`, `tests/test_parity.py` (report-entry tests)

**Interfaces:**
- Consumes: Task 1's builders.
- Produces: builders emitting `Selector`s; `build_report_entry` emitting `"selectors"`.

- [ ] **Step 1: Contract-assumption tests (red on a7)**

```python
"""Pins the contract behavior predict's deploy ordering relies on.

If contracts ever gains a tolerant read of the flat card shape, both consumer generations would
validate the same cards and ``choose_models``' ambiguity raise would fire on live traffic.
"""

import pydantic
import pytest
from sleap_roots_contracts import ModelCard

_IDENTITY = {"root_type": "primary", "registry_id": "reg/x", "version": "v0"}


def test_flat_card_shape_does_not_validate():
    flat = {"species": "rice", "mode": "cylinder", "age_min": 2, "age_max": 5, **_IDENTITY}
    with pytest.raises(pydantic.ValidationError):
        ModelCard.model_validate(flat)


def test_empty_selectors_do_not_validate():
    with pytest.raises(pydantic.ValidationError):
        ModelCard.model_validate({"selectors": [], **_IDENTITY})
```

Run: `uv run pytest tests/test_contract_assumptions.py -v`
Expected: FAIL (`test_flat_card_shape_does_not_validate`: DID NOT RAISE — a7 accepts flat).

- [ ] **Step 2: Bump the pin**

Edit `pyproject.toml:24` to `"sleap-roots-contracts==0.1.0a9",`, then:

```bash
uv lock --upgrade-package sleap-roots-contracts
uv sync --extra dev --extra cpu
git diff --stat uv.lock
uv run python -c "import sleap_roots_contracts as c; print(c.Selector)"
```

Expected: `uv.lock` changes only the `sleap-roots-contracts` block and predict's `requires-dist`
line; `Selector` prints.

- [ ] **Step 3: Switch the builders to a9**

In `tests/card_builders.py`, import `Selector` and replace the two bodies:

```python
from sleap_roots_contracts import ModelCard, Selector
```

```python
    sels = _selector_tuples(selectors, species, mode, age_min, age_max)
    return ModelCard(
        selectors=tuple(
            Selector(species=sp, mode=md, age_min=lo, age_max=hi) for sp, md, lo, hi in sels
        ),
        root_type=root_type,
        registry_id=registry_id or f"reg/{sels[0][0]}-{root_type}",
        version=version,
        weights_checksum=weights_checksum,
        sleap_nn_version=sleap_nn_version,
    )
```

```python
    sels = _selector_tuples(selectors, species, mode, age_min, age_max)
    selector_dicts = [
        {"species": sp, "mode": md, "age_min": lo, "age_max": hi} for sp, md, lo, hi in sels
    ]
    for key in drop:
        del selector_dicts[0][key]
    return {"selectors": selector_dicts, "root_type": root_type}
```

Update the module docstring's shape sentence. Run: `uv run pytest tests/test_contract_assumptions.py -v`
Expected: PASS. Run the gate's pytest line and record the failure count (expected: failures in
selection/parity/worker/batch from `AttributeError: 'ModelCard' object has no attribute 'species'`).

- [ ] **Step 4: Interim sole-selector reads**

`sleap_roots_predict/model_selection.py` — replace the three flat conditions in the `matches`
comprehension with the first selector (temporary; Task 3 replaces it):

```python
        matches = [
            card
            for card in cards
            if card.root_type == root_type
            and card.selectors[0].species == species
            and card.selectors[0].mode == mode
            and card.selectors[0].age_min <= age <= card.selectors[0].age_max
        ]
```

`sleap_roots_predict/parity.py:378`:

```python
    window = card.selectors[0]
    age_matches = [
        c
        for c in pool
        if (hint := _age_hint(c)) is not None and window.age_min <= hint <= window.age_max
    ]
```

`sleap_roots_predict/parity.py:903-909` (`build_label_card`):

```python
    selector = card.selectors[0]
    return LabelCard(
        species=selector.species,
        mode=selector.mode,
        root_type=card.root_type,
        age_min=selector.age_min,
        age_max=selector.age_max,
        skeleton_name=labels.skeleton.name or f"{selector.species}_{card.root_type}",
```

- [ ] **Step 5: Report-entry tests, then `build_report_entry`**

Add to `tests/test_parity.py` after `test_build_report_entry_handles_missing_reference` (uses
the module's existing `_resolved` (`:544`) and `_metrics` (`:654`) helpers):

```python
def _entry_for(card, tmp_path):
    resolved = _resolved(card, tmp_path / "gt.slp", tmp_path)
    return build_report_entry(resolved, 1, _metrics(), None)


def test_report_entry_lists_every_selector_in_card_order(tmp_path):
    card = _card(selectors=[("canola", "cylinder", 2, 13), ("pennycress", "cylinder", 2, 14)])
    entry = _entry_for(card, tmp_path)
    assert entry["selectors"] == [
        {"species": "canola", "mode": "cylinder", "age_min": 2, "age_max": 13},
        {"species": "pennycress", "mode": "cylinder", "age_min": 2, "age_max": 14},
    ]
    assert not {"species", "mode", "age_min", "age_max"} & set(entry)
    json.dumps(entry)


def test_report_entry_single_selector_is_a_one_element_list(tmp_path):
    entry = _entry_for(_card(), tmp_path)
    assert len(entry["selectors"]) == 1
```

Run them: Expected FAIL (`KeyError:
'selectors'` or the flat-key assertion). Then in `parity.py` replace the four flat lines of the
`entry` dict (`:1071-1075`, keeping `root_type`):

```python
        "selectors": [s.model_dump(mode="json") for s in resolved.card.selectors],
        "root_type": resolved.card.root_type,
```

and in the `build_report_entry` docstring's Fields list replace
``species``/``mode``/…/``age_min``/``age_max`` with: ``selectors``: the card's selection contexts,
one ``{species, mode, age_min, age_max}`` object per selector in card order. Update any existing
test asserting `entry["species"]` to `entry["selectors"][0]["species"]`.

- [ ] **Step 6: Remaining flat reads**

Run: `rg -n --type py "\.(species|mode|age_min|age_max)\b|\[[\"'](species|mode|age_min|age_max)[\"']\]" tests/ scripts/ sleap_roots_predict/`
Fix every card read; allowed residue: params reads in `model_selection.py:56-57`, `LabelCard`
reads (`test_parity.py:635`), `snapshot.mode` (`run_manifest.py:218`). `test_model_registry.py:315`
becomes:

```python
    assert all(c.selectors and c.root_type for c in cards)
    assert all(s.species and s.mode for c in cards for s in c.selectors)
```

- [ ] **Step 7: Gate green; commit**

Run: the gate. Expected: all pass.

```bash
git add pyproject.toml uv.lock tests/card_builders.py tests/test_contract_assumptions.py \
  sleap_roots_predict/model_selection.py sleap_roots_predict/parity.py \
  tests/test_model_registry.py tests/test_parity.py \
  openspec/changes/update-model-card-selectors/tasks.md
git commit   # message below
```

```
feat(selection)!: pin sleap-roots-contracts 0.1.0a9; read ModelCard.selectors (#34)

<record: red count after the bump; contract-assumption tests red on a7>

BREAKING CHANGE: parity report entries carry a `selectors` list in place of
top-level species/mode/age_min/age_max.
```

(Tasks 2 and 3 may be squashed into this one commit if preferred; each is green alone.)

---

### Task 3: Any-selector matching  [T 2.3, 2.4, 2.5]

**Files:**
- Modify: `sleap_roots_predict/model_selection.py:1-11,80-90`, `tests/test_model_selection.py`,
  `tests/test_param_resolution.py`

**Interfaces:**
- Produces: `_card_matches(card: ModelCard, species: str, mode: str, age: int) -> bool` (private).

- [ ] **Step 1: Selection tests**

Append to `tests/test_model_selection.py`:

```python
_CANOLA_PENNYCRESS = [("canola", "cylinder", 2, 13), ("pennycress", "cylinder", 2, 14)]


def test_card_matches_through_any_one_selector():
    card = _card("primary", selectors=_CANOLA_PENNYCRESS)
    assert "primary" in choose_models(_params(species="pennycress", age=14), [card])


def test_age_compared_against_the_matching_selectors_window_only():
    card = _card("primary", selectors=_CANOLA_PENNYCRESS)
    assert choose_models(_params(species="canola", age=14), [card]) == {}


@pytest.mark.parametrize("species,age,expected", [("pennycress", "14", True), ("canola", "14", False)])
def test_string_age_at_a_per_selector_boundary(species, age, expected):
    card = _card("primary", selectors=_CANOLA_PENNYCRESS)
    assert ("primary" in choose_models(_params(species=species, age=age), [card])) is expected


def test_disjoint_windows_of_one_species_are_not_merged():
    card = _card("primary", selectors=[("canola", "cylinder", 2, 5), ("canola", "cylinder", 10, 13)])
    assert choose_models(_params(species="canola", age=7), [card]) == {}


@pytest.mark.parametrize("age", [2, 5, 13])
def test_selectors_are_never_combined(age):
    card = _card(
        "primary",
        selectors=[("canola", "cylinder", 2, 13), ("arabidopsis", "multiplant cylinder", 2, 14)],
    )
    params = _params(species="canola", mode="multiplant cylinder", age=age)
    assert choose_models(params, [card]) == {}


def test_overlapping_selectors_on_one_card_are_one_match():
    card = _card("primary", selectors=[("rice", "cylinder", 2, 5), ("rice", "cylinder", 3, 8)])
    assert "primary" in choose_models(_params(age=4), [card])


def test_duplicate_identical_selectors_are_one_match():
    card = _card("primary", selectors=[("rice", "cylinder", 2, 5), ("rice", "cylinder", 2, 5)])
    assert "primary" in choose_models(_params(age=3), [card])


def test_two_cards_matching_through_different_selectors_are_ambiguous():
    a = _card("primary", ver="a", selectors=[("canola", "cylinder", 2, 13)])
    b = _card(
        "primary",
        ver="b",
        registry_id="reg/other",
        selectors=[("pennycress", "cylinder", 2, 14), ("canola", "cylinder", 5, 9)],
    )
    with pytest.raises(ValueError, match="Ambiguous"):
        choose_models(_params(species="canola", age=6), [a, b])


@pytest.mark.parametrize("age,expected", [(2, True), (13, True), (1, False), (14, False)])
def test_inclusive_boundaries_per_selector(age, expected):
    card = _card("primary", selectors=_CANOLA_PENNYCRESS)
    assert ("primary" in choose_models(_params(species="canola", age=age), [card])) is expected
```

Append to `tests/test_param_resolution.py`:

```python
def test_round_trip_selects_a_multi_selector_card():
    from card_builders import make_card

    card = make_card(
        "primary",
        "reg/shared-primary",
        selectors=[("canola", "cylinder", 2, 13), ("pennycress", "cylinder", 2, 14)],
        weights_checksum="sha",
    )
    row = _row(species_name="Pennycress", plant_age_days=14)
    assert set(choose_models(resolve_params(row), [card])) == {"primary"}
```

- [ ] **Step 2: Run them — expect the right reds**

Run: `uv run pytest tests/test_model_selection.py tests/test_param_resolution.py -v`
Expected FAIL (against the interim `selectors[0]` predicate): `..._through_any_one_selector`,
`test_string_age_..._[pennycress-14-True]`, `test_round_trip_selects_a_multi_selector_card`,
`test_two_cards_matching_..._ambiguous`. Record the list.

- [ ] **Step 3: Implement the predicate**

In `model_selection.py`, add above `choose_models`:

```python
def _card_matches(card: ModelCard, species: str, mode: str, age: int) -> bool:
    """Whether some single selector on ``card`` matches species, mode and age together.

    The age is compared against the *matching* selector's own window, never a window taken
    across the card's selectors, and selectors are never combined (no cross product).
    """
    return any(
        s.species == species and s.mode == mode and s.age_min <= age <= s.age_max
        for s in card.selectors
    )
```

and the comprehension becomes:

```python
        matches = [
            card
            for card in cards
            if card.root_type == root_type and _card_matches(card, species, mode, age)
        ]
```

Rewrite the module docstring's second sentence: "…otherwise a card matches when some single
one of its ``selectors`` matches ``species``/``mode``/inclusive age window together; exactly one
match selects, zero skips, more than one is an ambiguity error."

- [ ] **Step 4: Verify**

Run: `uv run pytest tests/test_model_selection.py tests/test_param_resolution.py -v` — Expected: PASS.
Run: `git diff sleap_roots_predict/model_selection.py` — Expected: the `if len(matches) > 1:`
block and the override/`if not matches` lines are unchanged.

- [ ] **Step 5: Mutation check (do not commit)**

Temporarily replace `_card_matches`' body with each, run the selection tests, confirm ≥1 failure,
then restore:

```python
# (a) card-level envelope
return any(s.species == species and s.mode == mode for s in card.selectors) and (
    min(s.age_min for s in card.selectors) <= age <= max(s.age_max for s in card.selectors))
# (b) cross product
return (any(s.species == species for s in card.selectors) and any(s.mode == mode for s in card.selectors)
        and any(s.age_min <= age <= s.age_max for s in card.selectors))
# (c) first selector only
s = card.selectors[0]; return s.species == species and s.mode == mode and s.age_min <= age <= s.age_max
```

Expected: (a) fails `test_age_compared_...` and `test_disjoint_windows_...`; (b) fails
`test_selectors_are_never_combined` and `test_age_compared_...`; (c) fails
`test_card_matches_through_any_one_selector`. Record in the commit body.

- [ ] **Step 6: Gate; commit**

```bash
git add sleap_roots_predict/model_selection.py tests/test_model_selection.py \
  tests/test_param_resolution.py openspec/changes/update-model-card-selectors/tasks.md
git commit -m "feat(selection): match a card through any single selector (#34)" \
  -m "<record: reds from Step 2; mutation results from Step 5>"
```

---

### Task 4: Registry guard and batch-level catalog load  [T 3.1–3.5]

**Files:**
- Modify: `sleap_roots_predict/model_registry.py:163-230`, `sleap_roots_predict/warm_worker.py:56-87`,
  `sleap_roots_predict/batch.py:314-330,355-370`, `sleap_roots_predict/__main__.py:1-12,86-90`,
  `tests/test_model_registry.py`, `tests/test_warm_worker.py`, `tests/test_batch.py`

**Interfaces:**
- Produces: `class NoReadableModelCardsError(ValueError)` in `model_registry.py`;
  `WarmModelWorker.load_catalog() -> None`.

- [ ] **Step 1: Characterization — flat cards skipped alongside readable ones**

Append to `tests/test_model_registry.py`:

```python
def _flat_meta():
    return {"species": "rice", "mode": "cylinder", "age_min": 2, "age_max": 5, "root_type": "primary"}


def test_collect_cards_skips_flat_cards_alongside_readable_ones(caplog):
    """Pins #34 fact 1 against a9: flat cards are skipped, selector-shaped ones returned."""
    source = WandbRegistrySource(alias="production")
    flat = FakeArtifact("reg/flat", metadata=_flat_meta())
    good = _good_artifact("reg/good")
    with caplog.at_level(logging.WARNING, logger="sleap_roots_predict.model_registry"):
        cards = source._collect_cards([flat, good])
    assert [c.registry_id for c in cards] == ["reg/good"]
    assert "reg/flat" in caplog.text and "selectors" in caplog.text
```

Run it. Expected: PASS on first run. If it fails, STOP: #34 fact 1 is false.

- [ ] **Step 2: Guard tests (invert the #32 test)**

Replace `test_collect_cards_all_malformed_returns_empty` with:

```python
def test_collect_cards_all_invalid_raises(caplog):
    """Every alias-carrying artifact unreadable is a deployment fault, not an empty catalog.

    Reverses #32's "empty, not an exception" for this one case (predict#34): an empty catalog
    fails every scan and exits 3, which the pipeline's exit gate passes.
    """
    source = WandbRegistrySource(entity="ent", registry="reg", alias="production")
    with pytest.raises(NoReadableModelCardsError, match=r"ent-org/wandb-registry-reg.*production.*\b2\b"):
        source._collect_cards([_malformed_artifact("reg/a"), FakeArtifact("reg/b", metadata=None)])
    assert issubclass(NoReadableModelCardsError, ValueError)
```

Add `NoReadableModelCardsError` to the `from sleap_roots_predict.model_registry import (...)`
block. Keep `test_collect_cards_alias_filtered_is_silent` unchanged (zero-alias → `[]`).
Run: Expected FAIL (`ImportError: cannot import name 'NoReadableModelCardsError'`).

- [ ] **Step 3: Implement the guard**

In `model_registry.py`, after the imports/logger:

```python
class NoReadableModelCardsError(ValueError):
    """Every artifact carrying the production alias failed card validation.

    A ``ValueError`` so the CLI's one-line staging-error path logs it (``__main__.py``).
    """
```

In `_collect_cards`, count alias-carrying artifacts that fail, and raise after the loop:

```python
        cards: List[ModelCard] = []
        failed = 0
        for artifact in artifacts:
            if self._alias and self._alias not in (
                getattr(artifact, "aliases", None) or []
            ):
                continue
            try:
                cards.append(self._card_from_artifact(artifact))
            except Exception as e:
                failed += 1
                label = getattr(artifact, "qualified_name", None) or getattr(
                    artifact, "name", "<unknown>"
                )
                logger.warning(
                    "Skipping non-conforming model artifact %r: %s", label, e
                )
        if failed and not cards:
            raise NoReadableModelCardsError(
                f"none of the {failed} model artifact(s) carrying alias {self._alias!r} in "
                f"{self._registry_project()} validated as a ModelCard (this consumer requires "
                f"sleap-roots-contracts {version('sleap-roots-contracts')} selector-shaped "
                "cards; has the registry been re-seeded?)"
            )
        return cards
```

Add `from importlib.metadata import version` to the imports. Update the `_collect_cards` docstring
("…skipped with a logged warning … **unless none** validates, which raises
``NoReadableModelCardsError``; zero alias-carrying artifacts still return ``[]``") and add to
`list_cards`' `Raises:` — ``NoReadableModelCardsError``. Run: Step 2 test PASS; whole file PASS.

- [ ] **Step 4: `load_catalog` tests**

Append to `tests/test_warm_worker.py`:

```python
class _CountingSource:
    def __init__(self, inner):
        self.inner, self.n = inner, 0

    def list_cards(self):
        self.n += 1
        return self.inner.list_cards()

    def materialize(self, ref):
        return self.inner.materialize(ref)


def test_load_catalog_lists_once_and_resolve_reuses_it(rice_source):
    source = _CountingSource(rice_source)
    worker = WarmModelWorker(source=source)
    worker.load_catalog()
    worker.load_catalog()
    worker.resolve(_params())
    worker.resolve(_params())
    assert source.n == 1


def test_load_catalog_after_resolve_does_not_list_again(rice_source):
    source = _CountingSource(rice_source)
    worker = WarmModelWorker(source=source)
    worker.resolve(_params())
    worker.load_catalog()
    assert source.n == 1


def test_load_catalog_without_key_names_it(clean_wandb_env):
    with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
        WarmModelWorker().load_catalog()
```

Run: Expected FAIL (`AttributeError: ... 'load_catalog'`).

- [ ] **Step 5: Implement `load_catalog`**

In `warm_worker.py`:

```python
    def load_catalog(self) -> None:
        """List the source's cards once and cache them (idempotent).

        ``resolve``/``get_predictors`` reuse the cache. Call this before a batch to surface
        catalog failures (missing credentials, registry errors, an unreadable registry) once,
        outside any per-scan error isolation.
        """
        if self._cards is None:
            self._cards = self._source.list_cards()
```

and `resolve` calls `self.load_catalog()` instead of its inline check. Update the constructor
comment ("…fails loud on the first ``load_catalog()``/``resolve()``/``get_predictors()``…") and
the module docstring's API list. Run: Step 4 tests PASS.

- [ ] **Step 6: Batch tests**

Append to `tests/test_batch.py`:

```python
def _counting(inner):
    calls = {"list": 0, "order": []}

    class _Counting:
        def list_cards(self):
            calls["list"] += 1
            calls["order"].append("list")
            return inner.list_cards()

        def materialize(self, ref):
            return inner.materialize(ref)

    return _Counting(), calls


def test_catalog_loaded_once_before_the_first_resolve(all_roots_source, tmp_path, monkeypatch):
    from sleap_roots_predict import warm_worker as ww

    inp = tmp_path / "in"
    _real_scan(inp, "scanA", _RICE)
    _real_scan(inp, "scanB", _RICE)
    source, calls = _counting(all_roots_source)
    real_resolve = ww.WarmModelWorker.resolve

    def spy_resolve(self, *a, **k):
        calls["order"].append("resolve")
        return real_resolve(self, *a, **k)

    monkeypatch.setattr(ww.WarmModelWorker, "resolve", spy_resolve)
    run_batch(inp, tmp_path / "out", source=source)
    assert calls["list"] == 1
    assert calls["order"][0] == "list"


def test_unreadable_registry_aborts_the_batch_with_exit_1(tmp_path, monkeypatch, caplog):
    import wandb

    from sleap_roots_predict import batch as batch_mod
    from sleap_roots_predict.__main__ import main
    from sleap_roots_predict.model_registry import NoReadableModelCardsError, WandbRegistrySource
    from test_model_registry import FakeApi, FakeArtifact, _flat_meta

    monkeypatch.setenv("WANDB_API_KEY", "dummy")
    monkeypatch.setattr(
        wandb, "Api", lambda: FakeApi({"col": [FakeArtifact("reg/flat", metadata=_flat_meta())]})
    )
    inp = tmp_path / "in"
    _real_scan(inp, "scanA", _RICE)
    out = tmp_path / "out"
    with pytest.raises(NoReadableModelCardsError):
        run_batch(inp, out, source=WandbRegistrySource(entity="ent", registry="reg"))
    assert not out.exists() or {p.name for p in out.iterdir()} <= {"run_manifest.json"}

    real_run_batch = batch_mod.run_batch

    def _with_source(*args, **kwargs):
        kwargs.setdefault("source", WandbRegistrySource(entity="ent", registry="reg"))
        return real_run_batch(*args, **kwargs)

    monkeypatch.setattr(batch_mod, "run_batch", _with_source)
    with caplog.at_level(logging.ERROR):
        with pytest.raises(NoReadableModelCardsError):
            main([str(inp), str(tmp_path / "out2")])
    assert "Batch aborted" in caplog.text


def test_missing_key_fails_the_batch_not_each_scan(tmp_path, clean_wandb_env):
    inp = tmp_path / "in"
    _real_scan(inp, "scanA", _RICE)
    with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
        run_batch(inp, tmp_path / "out")


def test_stop_before_first_scan_skips_the_catalog(tmp_path):
    inp = tmp_path / "in"
    _real_scan(inp, "scanA", _RICE)
    source, calls = _recording_source()
    run_batch(inp, tmp_path / "out", source=source, should_stop=lambda: True)
    assert calls["n"] == 0


def test_only_errored_scans_skip_the_catalog(tmp_path):
    inp = tmp_path / "in"
    inp.mkdir()
    (inp / "run_manifest.json").write_text(
        json.dumps({"schema_version": "1", "pipeline_run_id": "r", "scan_keys": ["ghost"]})
    )
    source, calls = _recording_source()
    result = run_batch(inp, tmp_path / "out", source=source)
    assert [s.status for s in result.scans] == ["failed"]
    assert calls["n"] == 0
```

(`main()` logs "Batch aborted" then re-raises staging errors, as `test_batch.py:584,1123` show;
the process exit code `1` is Python's for an uncaught exception.)
Run: Expected FAIL — `test_catalog_loaded_once_before_the_first_resolve` (order starts with
`resolve`), `test_unreadable_registry_...` (every scan fails, no raise), `test_missing_key_...`
(no raise). The stop/errored tests pass already (they pin unchanged behavior).

- [ ] **Step 7: Implement the load in `run_batch`**

In `batch.py`'s loop, after the `if scan.error is not None: … continue` block and before
`out_scan_dir = …`/the `try`:

```python
        if not catalog_loaded:
            # Once, outside per-scan isolation: an unreadable or unreachable catalog is a
            # batch-level error (exit 1), not one isolated failure per scan (exit 3, which the
            # pipeline's exit gate passes). Placed after this iteration's stop check so it adds
            # no should_stop() call, and skipped entirely when no scan is processable.
            worker.load_catalog()
            catalog_loaded = True
```

with `catalog_loaded = False` next to `worker = WarmModelWorker(source=source)`. Update the
`run_batch` `Raises:` docstring: add "``NoReadableModelCardsError`` (a ``ValueError``) if the
registry holds production artifacts none of which validates, and ``RuntimeError`` for missing
registry credentials — both before the first processable scan is predicted". Update
`__main__.py`'s module docstring list of staging errors ("…zero scans discovered, or no readable
production model card …") and the comment at `:86-90` to name the guard.

- [ ] **Step 8: Verify; gate; commit**

Run: `uv run pytest tests/test_batch.py tests/test_warm_worker.py tests/test_model_registry.py -v`
Expected: PASS, including the unchanged `test_run_batch_copy_failure_raises_before_any_prediction`
(`calls["n"] == 0`), `test_should_stop_stops_after_first_scan` and the SIGTERM compose test.
Run: the gate.

```bash
git add sleap_roots_predict/model_registry.py sleap_roots_predict/warm_worker.py \
  sleap_roots_predict/batch.py sleap_roots_predict/__main__.py tests/test_model_registry.py \
  tests/test_warm_worker.py tests/test_batch.py openspec/changes/update-model-card-selectors/tasks.md
git commit -m "feat(registry): fail loud when no production card is readable; load the catalog once per batch (#34)" \
  -m "Inverts test_collect_cards_all_malformed_returns_empty (#32's 'empty, not an exception') for the all-invalid case. Missing credentials and registry errors now exit 1 instead of 3."
```

- [ ] **Step 9: Draft the #34 comment [T 3.6]** — write it to
  `C:\Users\ELIZAB~1\AppData\Local\Temp\claude\c--repos-sleap-roots-predict\bf1ac106-8de2-448c-afb4-7156b504a5e2\scratchpad\comment-34-fact1.md`
  (fact 1's one exception, why, its limit). Show it to the user; post only on approval.

---

### Task 5: Parity selector rule  [T 4.1–4.7]

**Files:**
- Modify: `sleap_roots_predict/parity.py:67-69,335-395,397-470,471-559,863-930,1006-1008`,
  `scripts/run_parity_harness.py:64-70,99-104`, `tests/test_parity.py`

**Interfaces:**
- Produces: `_resolve_selector(card: ModelCard, selector: Optional[Selector]) -> Optional[Selector]`;
  `_pick_best_candidate(broken_path, candidates, card, selector=None)`;
  `relink_ground_truth_by_basename_search(bundle_dir, basename_index, card, out_path, *, selector=None)`;
  `resolve_ground_truth(card, bundle_dir, workdir, *, …, selector=None)`;
  `build_label_card(labels_path, card, *, images_embedded, selector=None, …)`.

- [ ] **Step 1: Helper tests**

```python
from sleap_roots_contracts import Selector

from sleap_roots_predict.parity import _resolve_selector

_TWO = [("canola", "cylinder", 2, 5), ("pennycress", "cylinder", 10, 13)]


def test_resolve_selector_explicit_on_card_and_value_equal():
    card = _card(selectors=_TWO)
    fresh = Selector(species="pennycress", mode="cylinder", age_min=10, age_max=13)
    assert _resolve_selector(card, fresh) == card.selectors[1]


def test_resolve_selector_not_on_card_raises_naming_card():
    card = _card(selectors=_TWO)
    foreign = Selector(species="rice", mode="cylinder", age_min=2, age_max=5)
    with pytest.raises(ValueError, match="reg/arabidopsis-primary"):
        _resolve_selector(card, foreign)


def test_resolve_selector_defaults():
    assert _resolve_selector(_card(), None) == _card().selectors[0]
    assert _resolve_selector(_card(selectors=_TWO), None) is None
```

Run: Expected FAIL (`ImportError`). Implement in `parity.py`:

```python
def _resolve_selector(card: ModelCard, selector: Optional[Selector]) -> Optional[Selector]:
    """Return the selector a ground-truth operation applies to, or ``None`` if unknown.

    The supplied selector (validated by value equality against the card), else the card's
    only selector, else ``None`` for a multi-selector card with none supplied.

    Raises:
        ValueError: If ``selector`` is not one of ``card.selectors``.
    """
    if selector is not None:
        if selector not in card.selectors:
            raise ValueError(
                f"selector {selector!r} is not one of card "
                f"{card.registry_id}:{card.version}'s selectors"
            )
        return selector
    return card.selectors[0] if len(card.selectors) == 1 else None
```

(import `Selector` from `sleap_roots_contracts`). Run: PASS.

- [ ] **Step 2: `_pick_best_candidate` tests**

```python
_DAY_CANDIDATES = [
    "Z:/share/expA/Day3/plant.h5",
    "Z:/share/expB/Day11/plant.h5",
]
_BROKEN = "D:/old/plants/plant.h5"  # parent 'plants' matches neither candidate


@pytest.mark.parametrize("which,expected", [(0, 0), (1, 1)])
def test_pick_best_candidate_supplied_selector_chooses_the_window(which, expected):
    card = _card(selectors=[("canola", "cylinder", 2, 5), ("pennycress", "cylinder", 10, 13)])
    winner = _pick_best_candidate(_BROKEN, _DAY_CANDIDATES, card, card.selectors[which])
    assert winner == _DAY_CANDIDATES[expected]


def test_pick_best_candidate_no_selector_skips_the_age_step():
    # Day15 lies outside every window, Day11 inside one. A card-level 2..13 envelope would keep
    # only Day11; path-segment scoring prefers Day15 (it shares 'expA' with the broken path).
    # Skipping the age step must therefore pick Day15.
    card = _card(selectors=[("canola", "cylinder", 2, 5), ("pennycress", "cylinder", 10, 13)])
    broken = "D:/old/expA/plants/plant.h5"
    candidates = ["Z:/share/expA/Day15/plant.h5", "Z:/share/other/Day11/plant.h5"]
    assert _pick_best_candidate(broken, candidates, card) == candidates[0]


def test_pick_best_candidate_no_selector_tie_is_none():
    card = _card(selectors=[("canola", "cylinder", 2, 5), ("pennycress", "cylinder", 10, 13)])
    assert _pick_best_candidate(_BROKEN, _DAY_CANDIDATES, card) is None
```

(The existing `test_pick_best_candidate_disambiguates_by_age_hint_in_range` at `:294` is the
sole-selector scenario's coverage; leave it.) Run: FAIL (signature). Implement: add
`selector: Optional[Selector] = None` to `_pick_best_candidate`; replace the Task 2 interim block:

```python
    pool = parent_matches if parent_matches else candidates
    window = _resolve_selector(card, selector)
    if window is not None:
        age_matches = [
            c
            for c in pool
            if (hint := _age_hint(c)) is not None
            and window.age_min <= hint <= window.age_max
        ]
        if len(age_matches) == 1:
            return age_matches[0]
        pool = age_matches if age_matches else pool
```

Update its docstring step (3) ("…inside the resolved selector's window; skipped for a
multi-selector card with no selector supplied"). Run: PASS; existing `_pick_best_candidate` tests PASS.

- [ ] **Step 3: Thread `selector` through resolution**

Tests:

```python
def test_resolve_ground_truth_rejects_a_foreign_selector_before_any_tier(tmp_path):
    card = _card(selectors=_TWO)
    looked = []
    foreign = Selector(species="rice", mode="cylinder", age_min=2, age_max=5)
    with pytest.raises(ValueError, match="reg/arabidopsis-primary"):
        resolve_ground_truth(
            card, tmp_path, tmp_path / "work",
            labels_registry_lookup=lambda c: looked.append(c) or tmp_path / "x.slp",
            selector=foreign,
        )
    assert looked == []
    assert not (tmp_path / "work").exists() or not any((tmp_path / "work").iterdir())
```

and, modelled on `test_resolve_ground_truth_uses_basename_search_as_last_resort` (`:403`):

```python
def test_resolve_ground_truth_threads_the_selector_to_basename_search(tmp_path, skeleton):
    card = _card(selectors=_TWO)
    video = sio.Video(filename="D:/old/plants/plant.h5", open_backend=False)
    labels = _make_labels(video, skeleton, [[[1, 1], [2, 2]]], sio.Instance)
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    sio.save_slp(labels, (bundle_dir / "labels_gt.val.slp").as_posix())
    search_dir = tmp_path / "search_root"
    for day in ("expA/Day3", "expB/Day11"):
        (search_dir / day).mkdir(parents=True)
        save_array_as_h5(np.zeros((1, 32, 32, 1), dtype="uint8"), search_dir / day / "plant.h5")
    index = build_basename_index(search_dir)

    result = resolve_ground_truth(
        card,
        bundle_dir=bundle_dir,
        workdir=tmp_path,
        labels_registry_lookup=lambda _card: None,
        basename_index=index,
        selector=card.selectors[1],
    )

    assert isinstance(result, ResolvedGroundTruth)
    relinked = sio.load_slp(result.ground_truth_path.as_posix())
    assert Path(relinked.videos[0].filename).parent.name == "Day11"
```

Implement: `resolve_ground_truth(..., selector: Optional[Selector] = None)` calls
`_resolve_selector(card, selector)` as its **first** statement (discarding the result; this is the
validation), and passes `selector=selector` to `relink_ground_truth_by_basename_search`, which
passes it to `_pick_best_candidate`. Update the three docstrings, and `parity.py:67-69`, `:423-424`
("card and selector used for age-range disambiguation"), `:480` ("the lookup's join criterion").

- [ ] **Step 4: `build_label_card` tests and implementation**

```python
def _gt_file(tmp_path, image_files, skeleton):
    video = sio.Video(filename=[str(f) for f in image_files])
    labels = _make_labels(video, skeleton, [[[1, 1], [2, 2]]], sio.Instance)
    path = tmp_path / "gt.slp"
    sio.save_slp(labels, path.as_posix())
    return path


def test_build_label_card_multi_selector_requires_a_selector(tmp_path, image_files, skeleton):
    with pytest.raises(ValueError, match="reg/arabidopsis-primary"):
        build_label_card(
            _gt_file(tmp_path, image_files, skeleton), _card(selectors=_TWO), images_embedded=True
        )


def test_build_label_card_uses_the_supplied_selector(tmp_path, image_files, skeleton):
    card = _card(selectors=_TWO)
    fresh = Selector(species="pennycress", mode="cylinder", age_min=10, age_max=13)
    lc = build_label_card(
        _gt_file(tmp_path, image_files, skeleton), card, images_embedded=True, selector=fresh
    )
    assert (lc.species, lc.age_min, lc.age_max) == ("pennycress", 10, 13)


def test_build_label_card_unnamed_skeleton_uses_the_selectors_species(tmp_path, image_files):
    card = _card(selectors=_TWO)
    path = _gt_file(tmp_path, image_files, sio.Skeleton(nodes=["A", "B"]))
    lc = build_label_card(path, card, images_embedded=True, selector=card.selectors[1])
    assert lc.skeleton_name == "pennycress_primary"
```

(If the module's `skeleton` fixture is already unnamed, the third test's explicit
`sio.Skeleton(nodes=[...])` is what guarantees it; check `lc.skeleton_name` is not the fixture's
name.) Implement:

```python
    resolved = _resolve_selector(card, selector)
    if resolved is None:
        raise ValueError(
            f"card {card.registry_id}:{card.version} carries {len(card.selectors)} selectors; "
            "pass selector= naming the labeling package's species/mode/age window"
        )
```

and use `resolved` in place of Task 2's `card.selectors[0]`.

- [ ] **Step 5: End-to-end harness with a two-selector card**

Modelled on `test_run_parity_harness_writes_one_entry_per_card` (`:887`):

```python
def test_run_parity_harness_multi_selector_card_round_trips(tmp_path, video, native_model_dir):
    card = _card(registry_id="reg/shared", selectors=_TWO)
    skeleton = sio.Skeleton(nodes=["A", "B"])
    gt = _make_labels(video, skeleton, [[[1, 1], [2, 2]], [[3, 3], [4, 4]]], sio.Instance)
    gt_path = tmp_path / "gt.slp"
    sio.save_slp(gt, gt_path.as_posix())
    out_path = tmp_path / "report.json"

    run_parity_harness(
        [card],
        LocalCardSource([(card, native_model_dir)]),
        tmp_path,
        out_path,
        labels_registry_lookup=lambda c: gt_path,
    )

    (entry,) = json.loads(out_path.read_text())
    assert "gap_stage" not in entry
    assert [s["species"] for s in entry["selectors"]] == ["canola", "pennycress"]
```

Expected: PASS once Steps 1–4 are in; a `gap_stage="evaluation"` entry means a flat read survived.

- [ ] **Step 6: Require `--out` in the harness script**

Test (`tests/test_parity.py`):

```python
def test_run_parity_harness_script_requires_out():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_parity_harness", Path(__file__).parents[1] / "scripts" / "run_parity_harness.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with pytest.raises(SystemExit) as exc:
        mod.main([])
    assert exc.value.code == 2
```

Implement: delete `_DEFAULT_OUT_PATH` and make `--out` `required=True` with help "Where to write
the report. Required: never overwrite the committed 2026-08-04 results JSON, a pre-selectors
snapshot." Run: PASS.

- [ ] **Step 7: Docstrings; gate; commit**

Update `parity.py:346-348` and `:1006-1008` if not already. Run the gate.

```bash
git add sleap_roots_predict/parity.py scripts/run_parity_harness.py tests/test_parity.py \
  openspec/changes/update-model-card-selectors/tasks.md
git commit -m "feat(parity): resolve an explicit selector for label cards and ground truth (#34)"
```

---

### Task 6: Docs  [T 5.1–5.3]

**Files:** `openspec/project.md:23-24,129`, `API.md:209,214-227`, `CLAUDE.md:93`,
`README.md:270-277`, `CHANGELOG.md`

- [ ] **Step 1: Edits**
  - `openspec/project.md:129`: `**sleap-roots-contracts** (`==0.1.0a9`)`; `:23-24`: "…the 13
    production `ModelCard`s registered at the 2026-08-04 measurement (8 physical weight sets)…",
    drop the `0.1.0a7` note.
  - `API.md:209`: drop the version from the parenthetical; `:214-227`: add "the model-card catalog
    is loaded once, before the first processable scan; a registry with no readable production card
    raises `NoReadableModelCardsError` (exit 1)".
  - `CLAUDE.md:93`: replace "(… `species`/`mode`/inclusive-age match …)" with "(matching rules:
    `model-management` spec)". Add nothing else.
  - `README.md:270-277`: "`--out` is required; write to a new dated path — the 2026-08-04 JSON is
    a pre-selectors snapshot and must not be overwritten."
  - `CHANGELOG.md` via `/update-changelog`, under Unreleased: Changed (BREAKING) contracts
    `0.1.0a9`, any-selector matching, report `selectors`; Added the guard and `load_catalog()`;
    Changed exit `3` → `1` for missing credentials / registry errors / unreadable catalog; Note
    that after merge `:latest`/`:main` read only the canary's card until the full re-seed.
- [ ] **Step 2: Claim grep**

Run: `git grep -nE "0\.1\.0a7|13 production|card's age|card's \[age_min|species/root-type|species ==|age_min <=|card-level|lazily once" -- . ':!openspec/changes/archive' ':!docs/superpowers/specs/2026-0[78]*' ':!docs/superpowers/plans/2026-0[78]*'`
Expected: only intentional hits (this change's own files, dated measurements). Fix the rest.
Run: `git grep -n "registry_id" -- sleap_roots_predict/` and confirm nothing beyond design §4's
list persists a `registry_id` key; note the result for the PR.
- [ ] **Step 3: Gate; commit**

```bash
git add openspec/project.md API.md CLAUDE.md README.md CHANGELOG.md \
  openspec/changes/update-model-card-selectors/tasks.md
git commit -m "docs: document selector matching, the guard and the 0.1.0a9 pin (#34)"
```

---

### Task 7: Draft PR  [T 6.0]

- [ ] **Step 1: Check every commit is green**

Run: `GIT_SEQUENCE_EDITOR=: git rebase -x "uv run pytest -m 'not gpu and not acceptance and not wandb and not parity' tests/ -q" main`
Expected: completes with no stop.
- [ ] **Step 2: Push and open the draft** (ask the user first — this is outward-facing)

```bash
git push -u origin migrate-model-card-selectors
gh pr create --draft --title "feat(selection)!: migrate to ModelCard.selectors (contracts 0.1.0a9)" --body-file <body>
```

Body from `/pr-description`, with its literal `Closes #…` line replaced by "Part of #34", a
ready-made squash message ending in the `BREAKING CHANGE:` footer, and the gate command stated as
replacing `/pre-merge`'s `-m "not gpu"`. Do not link #34 in the Development sidebar.

---

### Task 8: A1 — selection-equivalence oracle  [T 6.1]

**Files:**
- Create: `scripts/a1_selection_oracle.py`, `scripts/a1_compare.py`,
  `docs/superpowers/specs/<date>-model-card-selectors-a1-{old-cards,new-cards,baseline,grid,old-table,new-table}.json`

**Interfaces:**
- `a1_selection_oracle.py dump --live --out cards.json` → `[card.model_dump(mode="json"), …]`.
- `a1_selection_oracle.py table (--live | --cards-json cards.json) --grid grid.json --out table.json`
  → `{"species|mode|age": {"raise": msg} | {root_type: registry_id, …}}`.
- `a1_compare.py grid old-cards.json new-cards.json --out grid.json`.
- `a1_compare.py check old-table.json new-table.json --baseline baseline.json --new-cards new-cards.json`
  → exit `0` iff equivalent.

- [ ] **Step 1: Oracle script** (runs unmodified on `main` and the branch; reads no card field)

```python
"""A1 selection-equivalence oracle (predict#34): tabulate choose_models over a grid.

Deliberately reads no ModelCard field, so the same file runs on contracts 0.1.0a7 (flat) and
0.1.0a9 (selectors). Cards come from the live registry or a JSON list of card dicts.
"""

import argparse
import json
from pathlib import Path

from sleap_roots_contracts import ModelCard, ResolvedParams

from sleap_roots_predict.model_selection import choose_models


def _cards(args):
    if args.live:
        from sleap_roots_predict.model_registry import WandbRegistrySource

        return WandbRegistrySource().list_cards()
    return [ModelCard.model_validate(d) for d in json.loads(Path(args.cards_json).read_text())]


def main(argv=None):
    """Run the ``dump`` or ``table`` subcommand."""
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("dump", "table"):
        s = sub.add_parser(name)
        src = s.add_mutually_exclusive_group(required=True)
        src.add_argument("--live", action="store_true")
        src.add_argument("--cards-json")
        s.add_argument("--out", required=True)
        if name == "table":
            s.add_argument("--grid", required=True)
    args = p.parse_args(argv)
    cards = _cards(args)
    if args.cmd == "dump":
        data = [c.model_dump(mode="json") for c in cards]
    else:
        data = {}
        for cell in json.loads(Path(args.grid).read_text()):
            key = f"{cell['species']}|{cell['mode']}|{cell['age']}"
            try:
                refs = choose_models(ResolvedParams(values=cell), cards)
                data[key] = {rt: ref.registry_id for rt, ref in sorted(refs.items())}
            except ValueError as e:
                data[key] = {"raise": str(e)}
    Path(args.out).write_text(json.dumps(data, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Compare script** (plain JSON; runs on the branch)

```python
"""A1 compare step (predict#34): build the grid, then check old vs new tables."""

import argparse
import json
import sys
from pathlib import Path

_MODES = ("cylinder", "multiplant cylinder", "plate")


def _contexts(card):
    if "selectors" in card:
        return [(s["species"], s["mode"], s["age_min"], s["age_max"]) for s in card["selectors"]]
    return [(card["species"], card["mode"], card["age_min"], card["age_max"])]


def _grid(old, new):
    ctx = [c for card in old + new for c in _contexts(card)]
    species = sorted({c[0] for c in ctx}) + ["unmodelled-species"]
    lo, hi = min(c[2] for c in ctx) - 1, max(c[3] for c in ctx) + 1
    return [
        {"species": s, "mode": m, "age": a}
        for s in species
        for m in _MODES
        for a in range(max(lo, 0), hi + 1)
    ]


def _check(old, new, baseline, new_cards):
    to_model_old = {c["collection"]: c["source_model_id"] for c in baseline["collections"]}
    to_model_new = {c["registry_id"]: c["source_model_id"] for c in new_cards}
    errors, selected = [], {"old": 0, "new": 0}
    if any("raise" in v for v in old.values()):
        errors.append("precondition: the old side raises in some cell")
    for key in sorted(old):
        o, n = old[key], new[key]
        if "raise" in n:
            errors.append(f"{key}: new side raises: {n['raise']}")
            continue
        o_m = {rt: to_model_old[rid.rsplit("/", 1)[-1]] for rt, rid in o.items() if rt != "raise"}
        n_m = {rt: to_model_new[rid] for rt, rid in n.items()}
        selected["old"] += bool(o_m)
        selected["new"] += bool(n_m)
        if o_m != n_m:
            errors.append(f"{key}: old {o_m} != new {n_m}")
    if not selected["new"] or selected["old"] != selected["new"]:
        errors.append(f"selected-cell counts differ or are zero: {selected}")
    for card in new_cards:
        for sp, *_ in _contexts(card):
            if sp != sp.lower():
                errors.append(f"{card['registry_id']}: species {sp!r} is not lowercase")
    return errors, selected


def main(argv=None):
    """Run the ``grid`` or ``check`` subcommand; ``check`` exits 1 on any mismatch."""
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grid")
    g.add_argument("old_cards")
    g.add_argument("new_cards")
    g.add_argument("--out", required=True)
    c = sub.add_parser("check")
    c.add_argument("old_table")
    c.add_argument("new_table")
    c.add_argument("--baseline", required=True)
    c.add_argument("--new-cards", required=True)
    args = p.parse_args(argv)

    def load(path):
        return json.loads(Path(path).read_text())

    if args.cmd == "grid":
        grid = _grid(load(args.old_cards), load(args.new_cards))
        Path(args.out).write_text(json.dumps(grid, indent=2))
        return 0
    errors, selected = _check(
        load(args.old_table), load(args.new_table), load(args.baseline), load(args.new_cards)
    )
    print(json.dumps({"selected_cells": selected, "errors": errors}, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
```

(`_MODES` equals a9's `Mode = Literal["cylinder", "multiplant cylinder", "plate"]`; the baseline
JSON's `collections` entries carry `collection` and `source_model_id` — both confirmed
2026-09-23.)

- [ ] **Step 3: Training dump** (training worktree venv; pure library path, no zips, no wandb)

```bash
cd /c/repos/sleap-roots-training-talmolab/.worktrees/migrate-model-card-selectors
uv run python -c "
import json
from sleap_roots_training.registry import cards, chooser
out = []
for card in cards.expand_rows_to_cards(chooser.load_selection_matrix().rows):
    meta = cards.card_to_metadata(card)
    meta.update(registry_id=cards.collection_id(card), version='v0')
    out.append(meta)
print(json.dumps(out, indent=2))
" > <A1 dir>/new-cards.json
cp docs/migration/2026-09-22-pre-reseed-baseline.json <A1 dir>/baseline.json
git rev-parse HEAD; sha256sum docs/migration/2026-09-22-pre-reseed-baseline.json
```

Expected: 8 cards. Record the sha and hash.

- [ ] **Step 4: `main` side** — `git worktree add <scratchpad>/predict-main main`, `uv sync --extra dev --extra cpu`
  there, copy both scripts in unmodified (record `sha256sum` on both sides), then with
  `WANDB_API_KEY`/`SRP_WANDB_ENTITY` set:

```bash
uv run python scripts/a1_selection_oracle.py dump --live --out <A1 dir>/old-cards.json
```

- [ ] **Step 5: Grid, tables, check** (branch venv for grid/new/check; `main` worktree for old):

```bash
uv run python scripts/a1_compare.py grid <A1>/old-cards.json <A1>/new-cards.json --out <A1>/grid.json
# in the main worktree:
uv run python scripts/a1_selection_oracle.py table --live --grid <A1>/grid.json --out <A1>/old-table.json
# on the branch:
uv run python scripts/a1_selection_oracle.py table --cards-json <A1>/new-cards.json --grid <A1>/grid.json --out <A1>/new-table.json
uv run python scripts/a1_compare.py check <A1>/old-table.json <A1>/new-table.json --baseline <A1>/baseline.json --new-cards <A1>/new-cards.json
```

Expected: exit `0`, `errors: []`, equal non-zero selected-cell counts. Any error: STOP and report.

- [ ] **Step 6: Commit evidence + scripts; remove the worktree**

Copy the six JSON files to `docs/superpowers/specs/<date>-model-card-selectors-a1-*.json`.

```bash
git add scripts/a1_selection_oracle.py scripts/a1_compare.py docs/superpowers/specs/*-model-card-selectors-a1-*.json \
  openspec/changes/update-model-card-selectors/tasks.md
git commit -m "test(evidence): A1 selection-equivalence oracle, 13 flat vs 8 selector cards (#34)"
git worktree remove <scratchpad>/predict-main
```

---

### Task 9: A2 — real-inference equivalence  [T 6.2]

**Files:** Create `scripts/a2_run_local.py`

- [ ] **Step 1: Driver**

```python
"""A2 driver (predict#34): run_batch over real weights from a local card source.

Cards come from a JSON list of card dicts (A1's old-cards.json on main, new-cards.json on the
branch); ``--dirs`` maps each card's registry_id to an extracted model directory.
"""

import argparse
import json
from pathlib import Path

from sleap_roots_contracts import ModelCard

from sleap_roots_predict.batch import run_batch
from sleap_roots_predict.model_registry import LocalCardSource


def main(argv=None):
    """Run one A2 batch and print its per-scan statuses."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input_dir")
    p.add_argument("output_dir")
    p.add_argument("--cards-json", required=True)
    p.add_argument("--dirs", required=True, help="JSON {registry_id: model_dir}")
    args = p.parse_args(argv)
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"output dir must be fresh and empty: {out.as_posix()}")
    dirs = json.loads(Path(args.dirs).read_text())
    cards = [ModelCard.model_validate(d) for d in json.loads(Path(args.cards_json).read_text())]
    source = LocalCardSource([(c, Path(dirs[c.registry_id])) for c in cards if c.registry_id in dirs])
    result = run_batch(Path(args.input_dir), out, source=source)
    statuses = {s.scan_key: s.status for s in result.scans}
    print(json.dumps(statuses, indent=2))
    if set(statuses.values()) != {"ok"}:
        raise SystemExit(f"not all scans ok: {statuses}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Stage inputs**
  - Copy `scan_289`, `scan_577`, `scan_1009` (dirs + sidecars) from
    `Z:\users\eberrigan\pipeline_orchestration_tests\a4_poc\input` to `<scratchpad>/a2/in`.
  - Extract `20250204_models/<source_model_id>.zip` for each model the canola scans need
    (`canola/lateral/240611_083419.multi_instance.n=631`,
    `canola_pennycress_arabidopsis/primary/240611_102513.multi_instance.n=743`) into
    `<scratchpad>/a2/models/<source_model_id>/`.
  - Write `dirs-main.json` (old flat `registry_id` → dir, via the baseline) and
    `dirs-branch.json` (new `registry_id` = `collection_id` → dir).
  - `uv pip freeze` in both environments; diff, excluding the editable `sleap-roots-predict` line.
    Expected: only `sleap-roots-contracts` differs.
- [ ] **Step 3: Three runs, CPU, fresh output dirs**

```bash
# main worktree (driver copied in unmodified):
SRP_DEVICE=cpu uv run python scripts/a2_run_local.py <a2>/in <a2>/out-main-1 --cards-json <A1>/old-cards.json --dirs <a2>/dirs-main.json
SRP_DEVICE=cpu uv run python scripts/a2_run_local.py <a2>/in <a2>/out-main-2 --cards-json <A1>/old-cards.json --dirs <a2>/dirs-main.json
# branch:
SRP_DEVICE=cpu uv run python scripts/a2_run_local.py <a2>/in <a2>/out-branch --cards-json <A1>/new-cards.json --dirs <a2>/dirs-branch.json
```

Expected: each prints all three scans `ok`.
- [ ] **Step 4: Compare** (throwaway script in the scratchpad, branch venv): for each scan and root
  type, load the one `.slp` per root in each output via `sio.load_slp`; assert equal frame counts
  (> 0), equal instance counts per frame, `np.array_equal(points, equal_nan=True)` and equal
  scores between `out-main-1` and `out-main-2` (control) and between `out-main-1` and
  `out-branch` (bitwise if the control was bitwise, else within the control's max |Δ|). Diff the
  `predictions.json` manifests with the allowed keys removed (`registry_id`, `version`,
  `weights_checksum`, `.slp` path/filename, idempotency key, timestamps). Unequal control instance
  counts → report "inconclusive", not pass.
- [ ] **Step 5: Commit the driver; record results in the PR**

```bash
git add scripts/a2_run_local.py openspec/changes/update-model-card-selectors/tasks.md
git commit -m "test(evidence): A2 local-run driver for real-inference equivalence (#34)"
```

---

### Task 10: Pre-canary wandb check, live canary, final gates  [T 6.3–6.7]

- [ ] **Step 1: [T 6.3]** `uv run pytest -m wandb -rA` with `WANDB_API_KEY` set.
  Expected: 0 skipped; both wandb tests fail naming `NoReadableModelCardsError`, the registry and
  alias; the CLI test shows "Batch aborted", exit `1`. Record the output.
- [ ] **Step 2: Canary check script** — create `scripts/canary_check.py`: lists live cards with
  `WandbRegistrySource`, runs `choose_models` for `rice/cylinder/3`, prints the selected
  `registry_id`s and the count of "Skipping non-conforming" warnings captured via a logging
  handler; `--expect-registry-suffix` and `--expect-skips` make it exit `1` on mismatch. Commit it.
- [ ] **Step 3: [T 6.4] STOP and ask the user** to confirm `seed-registry --execute --only
  rice-younger-primary-230104_182346.multi_instance.n-720` (or the crown collection) in the
  training worktree. Only after an explicit yes, run it; then:
  - branch: `uv run python scripts/canary_check.py --expect-registry-suffix <new collection> --expect-skips 13`
  - `main` worktree: `... --expect-registry-suffix rice-cylinder-primary-age2-5 --expect-skips 1`
  - branch: `uv run pytest -m wandb -rA` → both pass.
  Not while training's 6.0(b)/(e) alias-drop rehearsal is live.
- [ ] **Step 4: [T 6.5]** the gate + `uv build` + `uv run pytest -m gpu -rs` in a `windows_cuda`
  venv reporting 0 skipped.
- [ ] **Step 5: [T 6.6]** `/review-pr`; address findings.
- [ ] **Step 6: [T 6.7]** draft the pipeline design correction to the scratchpad; show the user.
- [ ] **Step 7: [T 7.1]** draft the #34 post-merge checklist (C1–C3 as in `tasks.md` 7.1); show
  the user; post on approval; tick 7.1. Mark the PR ready and present READY TO MERGE — the user
  merges.
