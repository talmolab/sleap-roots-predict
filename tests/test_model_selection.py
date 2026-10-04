"""Tests for the pure model-selection matcher (``choose_models``).

Real, no-mock tests over tiny hand-built ``ModelCard`` lists. The matcher is pure:
no network, no per-call filesystem I/O (the runtime sleap-nn version is resolved
once at import).
"""

import logging
from importlib.metadata import version

import pytest
from sleap_roots_contracts import ModelRef, ResolvedParams, compute_param_hash

from card_builders import make_card, production_cards
from sleap_roots_predict.model_selection import choose_models, past_window_age


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


def _params(species="rice", mode="cylinder", age=3):
    return ResolvedParams(values={"species": species, "mode": mode, "age": age})


def test_exact_match_selects_one_ref_per_root_type():
    """Each present root type with exactly one matching card yields a ModelRef."""
    cards = [
        _card("primary", ver="p1", checksum="pc"),
        _card("crown", ver="c1", checksum="cc"),
    ]
    result = choose_models(_params(age=3), cards)
    assert set(result) == {"primary", "crown"}
    assert isinstance(result["primary"], ModelRef)
    # ModelRef carries the card's concrete pin + the runtime sleap-nn version.
    assert result["primary"].version == "p1"
    assert result["primary"].weights_checksum == "pc"
    assert result["primary"].root_type == "primary"
    assert result["primary"].sleap_nn_version == version("sleap-nn")


@pytest.mark.parametrize("age", [2, 5])
def test_age_window_boundaries_are_inclusive(age):
    """Age at age_min and at age_max both match (inclusive window)."""
    result = choose_models(_params(age=age), [_card("primary", age_min=2, age_max=5)])
    assert "primary" in result


def test_age_below_window_does_not_match():
    """An age below age_min leaves that root type unmatched (it is not clamped)."""
    result = choose_models(_params(age=1), [_card("primary", age_min=2, age_max=5)])
    assert result == {}


def test_no_match_returns_empty_mapping():
    """A species/mode mismatch matches nothing and is not an error."""
    cards = [_card("primary", species="rice")]
    assert choose_models(_params(species="soybean"), cards) == {}


def test_ambiguous_match_raises_naming_root_type():
    """Two cards matching the same root type is an ambiguity error."""
    cards = [_card("primary", ver="a"), _card("primary", ver="b")]
    with pytest.raises(ValueError, match="primary"):
        choose_models(_params(age=3), cards)


def test_override_bypasses_matching_even_without_cards():
    """An explicit override resolves a root type with no matching card."""
    override = ModelRef(
        registry_id="reg/override",
        version="ov",
        sleap_nn_version="x",
        root_type="primary",
    )
    result = choose_models(_params(), cards=[], overrides={"primary": override})
    assert result == {"primary": override}


@pytest.mark.parametrize("missing", ["species", "mode", "age"])
def test_missing_required_param_raises(missing):
    """Absent species/mode/age raises a clear error naming the param."""
    values = {"species": "rice", "mode": "cylinder", "age": 3}
    del values[missing]
    with pytest.raises(ValueError, match=missing):
        choose_models(ResolvedParams(values=values), [_card("primary")])


def test_age_accepts_int_coercible_string():
    """Bloom metadata age may arrive as a string; it is coerced to int."""
    result = choose_models(_params(age="3"), [_card("primary", age_min=2, age_max=5)])
    assert "primary" in result


@pytest.mark.parametrize("bad_age", [3.5, "3.5", True])
def test_age_rejects_non_whole_number(bad_age):
    """Fractional/bool ages raise rather than silently truncating to a wrong model."""
    with pytest.raises(ValueError, match="whole number|integer"):
        choose_models(_params(age=bad_age), [_card("primary", age_min=2, age_max=5)])


def test_runtime_sleap_nn_version_resolved_at_import():
    """The stamped version is the installed sleap-nn, resolved once at import."""
    from sleap_roots_predict import model_selection

    assert model_selection._RUNTIME_SLEAP_NN_VERSION == version("sleap-nn")


_CANOLA_PENNYCRESS = [("canola", "cylinder", 2, 13), ("pennycress", "cylinder", 2, 14)]


def test_card_matches_through_any_one_selector():
    card = _card("primary", selectors=_CANOLA_PENNYCRESS)
    assert "primary" in choose_models(_params(species="pennycress", age=14), [card])


def test_age_compared_against_the_matching_selectors_window_only():
    card = _card(
        "primary",
        selectors=[("canola", "cylinder", 5, 13), ("pennycress", "cylinder", 2, 14)],
    )
    assert choose_models(_params(species="canola", age=3), [card]) == {}


@pytest.mark.parametrize(
    "species,age,expected",
    # canola "14" is past canola's 2-13 window, so it is matched at 13.
    [("pennycress", "14", True), ("canola", "14", True), ("canola", "1", False)],
)
def test_string_age_at_a_per_selector_boundary(species, age, expected):
    card = _card("primary", selectors=_CANOLA_PENNYCRESS)
    assert (
        "primary" in choose_models(_params(species=species, age=age), [card])
    ) is expected


def test_string_age_is_compared_per_selector():
    """Canola "3" is below its own 5-13 window; a card-level 2-14 window would match it."""
    card = _card(
        "primary",
        selectors=[("canola", "cylinder", 5, 13), ("pennycress", "cylinder", 2, 14)],
    )
    assert choose_models(_params(species="canola", age="3"), [card]) == {}


def test_disjoint_windows_of_one_species_are_not_merged():
    card = _card(
        "primary",
        selectors=[("canola", "cylinder", 2, 5), ("canola", "cylinder", 10, 13)],
    )
    assert choose_models(_params(species="canola", age=7), [card]) == {}


@pytest.mark.parametrize("age", [2, 5, 13])
def test_selectors_are_never_combined(age):
    card = _card(
        "primary",
        selectors=[
            ("canola", "cylinder", 2, 13),
            ("arabidopsis", "multiplant cylinder", 2, 14),
        ],
    )
    params = _params(species="canola", mode="multiplant cylinder", age=age)
    assert choose_models(params, [card]) == {}


def test_overlapping_selectors_on_one_card_are_one_match():
    card = _card(
        "primary", selectors=[("rice", "cylinder", 2, 5), ("rice", "cylinder", 3, 8)]
    )
    assert "primary" in choose_models(_params(age=4), [card])


def test_duplicate_identical_selectors_are_one_match():
    card = _card(
        "primary", selectors=[("rice", "cylinder", 2, 5), ("rice", "cylinder", 2, 5)]
    )
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


@pytest.mark.parametrize(
    # 14 is past canola's 2-13 window, so it is matched at 13.
    "age,expected",
    [(2, True), (13, True), (1, False), (14, True)],
)
def test_inclusive_boundaries_per_selector(age, expected):
    card = _card("primary", selectors=_CANOLA_PENNYCRESS)
    assert (
        "primary" in choose_models(_params(species="canola", age=age), [card])
    ) is expected


def _stems(result):
    return {root: ref.registry_id.removeprefix("reg/") for root, ref in result.items()}


_CPA = "canola_pennycress_arabidopsis-primary"

_IN_WINDOW = [
    ("arabidopsis", 10, {"primary": _CPA, "lateral": "arabidopsis-lateral"}),
    ("arabidopsis", 14, {"primary": _CPA, "lateral": "arabidopsis-lateral"}),
    ("rice", 4, {"primary": "rice-younger-primary", "crown": "rice-younger-crown"}),
    ("rice", 8, {"crown": "rice-older-crown"}),
    ("canola", 13, {"primary": _CPA, "lateral": "canola-lateral"}),
    ("soybean", 8, {"primary": "soybean-primary", "lateral": "soybean-lateral"}),
]


@pytest.mark.parametrize("species,age,expected", _IN_WINDOW)
def test_in_window_production_selection_is_pinned(species, age, expected):
    """In-window refs on the production-shaped catalog; the clamp must not change them."""
    result = choose_models(_params(species=species, age=age), production_cards())
    assert _stems(result) == expected


# --- Past-window ages (bloom#971 phase 1) ----------------------------------------------

_PAST_WINDOW = [
    ("arabidopsis", 28, 14, {"primary": _CPA, "lateral": "arabidopsis-lateral"}),
    ("rice", 18, 10, {"crown": "rice-older-crown"}),
    ("soybean", 10, 8, {"primary": "soybean-primary", "lateral": "soybean-lateral"}),
    ("canola", 14, 13, {"primary": _CPA, "lateral": "canola-lateral"}),
    ("pennycress", 15, 14, {"primary": _CPA, "lateral": "canola-lateral"}),
]


@pytest.mark.parametrize("species,age,window_max,expected", _PAST_WINDOW)
def test_past_window_scan_matches_at_window_maximum(species, age, window_max, expected):
    """A scan older than every window selects what its species selects at age_max."""
    cards = production_cards()
    result = choose_models(_params(species=species, age=age), cards)
    assert _stems(result) == expected
    assert result == choose_models(_params(species=species, age=window_max), cards)


@pytest.mark.parametrize("age", [365, "28"])
def test_past_window_clamps_string_and_extreme_ages(age):
    """No upper limit, and a whole-number string clamps like an int."""
    result = choose_models(_params(species="arabidopsis", age=age), production_cards())
    assert _stems(result) == {"primary": _CPA, "lateral": "arabidopsis-lateral"}


def test_window_maximum_is_scoped_by_mode():
    """Another mode's higher window does not raise this mode's window maximum."""
    cyl = _card("primary", selectors=[("canola", "cylinder", 2, 13)])
    multi = _card("lateral", selectors=[("canola", "multiplant cylinder", 2, 20)])
    result = choose_models(_params(species="canola", age=15), [cyl, multi])
    assert set(result) == {"primary"}


def test_clamping_never_selects_through_a_lower_window():
    """Only cards reaching the window maximum match a clamped scan."""
    low = _card(
        "lateral", registry_id="reg/low", selectors=[("arabidopsis", "cylinder", 2, 10)]
    )
    high = _card(
        "lateral",
        registry_id="reg/high",
        selectors=[("arabidopsis", "cylinder", 2, 14)],
    )
    result = choose_models(_params(species="arabidopsis", age=28), [low, high])
    assert result["lateral"].registry_id == "reg/high"


def test_ambiguity_at_matching_age_names_both_ages():
    """Two cards matching at the clamped age is ambiguous; the error names both ages."""
    a = _card("lateral", ver="a", selectors=[("arabidopsis", "cylinder", 2, 14)])
    b = _card(
        "lateral",
        ver="b",
        registry_id="reg/b",
        selectors=[("arabidopsis", "cylinder", 5, 14)],
    )
    with pytest.raises(ValueError, match=r"Ambiguous.*28.*14"):
        choose_models(_params(species="arabidopsis", age=28), [a, b])


def _override(root_type):
    return ModelRef(
        registry_id=f"reg/ov-{root_type}",
        version="ov",
        sleap_nn_version="x",
        root_type=root_type,
    )


def test_override_wins_on_a_clamped_scan():
    """An override still wins; the other root types are clamped-matched."""
    ov = _override("primary")
    result = choose_models(
        _params(species="arabidopsis", age=28),
        production_cards(),
        overrides={"primary": ov},
    )
    assert result["primary"] == ov
    assert result["lateral"].registry_id == "reg/arabidopsis-lateral"


def test_overridden_root_types_count_toward_window_maximum():
    """The window maximum includes overridden root types' cards."""
    primary = _card("primary", selectors=[("rice", "cylinder", 2, 10)])
    lateral = _card("lateral", selectors=[("rice", "cylinder", 2, 5)])
    ov = _override("primary")
    result = choose_models(
        _params(age=18), [primary, lateral], overrides={"primary": ov}
    )
    assert result == {"primary": ov}


@pytest.mark.parametrize(
    "species,mode,age",
    [
        ("arabidopsis", "cylinder", 1),
        ("canola", "cylinder", 0),
        ("alfalfa", "cylinder", 30),
        ("canola", "multiplant cylinder", 20),
    ],
)
def test_unclamped_controls_match_nothing(species, mode, age):
    """Younger-than-window, no-card and other-mode scans are not clamped."""
    params = _params(species=species, mode=mode, age=age)
    assert choose_models(params, production_cards()) == {}


def test_empty_catalog_and_overrides_only():
    """No cards means no window maximum; overrides alone are returned."""
    assert choose_models(_params(age=100), []) == {}
    ov = _override("primary")
    assert choose_models(_params(age=100), [], overrides={"primary": ov}) == {
        "primary": ov
    }


def test_gap_between_disjoint_windows_is_not_clamped():
    """An age between two windows is below the window maximum, so it is not clamped."""
    card = _card(
        "primary",
        selectors=[("canola", "cylinder", 2, 5), ("canola", "cylinder", 10, 13)],
    )
    assert choose_models(_params(species="canola", age=7), [card]) == {}


@pytest.mark.parametrize("fn", [choose_models, past_window_age])
def test_bad_age_raises_with_no_cards(fn):
    """Validation happens before any card inspection."""
    with pytest.raises(ValueError, match="whole number"):
        fn(_params(age=3.5), [])


def test_clamped_call_keeps_params_and_logs_nothing(caplog):
    """choose_models never mutates params and never logs."""
    params = _params(species="arabidopsis", age=28)
    with caplog.at_level(logging.DEBUG, logger="sleap_roots_predict.model_selection"):
        choose_models(params, production_cards())
    assert params.values["age"] == 28
    # Recompute from the (mutable) values: param_hash itself is fixed at construction.
    assert (
        compute_param_hash(params.values)
        == _params(species="arabidopsis", age=28).param_hash
    )
    assert not [r for r in caplog.records if r.name.startswith("sleap_roots_predict")]


@pytest.mark.parametrize(
    "species,mode,age,expected",
    [
        ("arabidopsis", "cylinder", 28, 14),
        ("arabidopsis", "cylinder", "28", 14),
        ("arabidopsis", "cylinder", 10, None),
        ("arabidopsis", "cylinder", 14, None),
        ("rice", "cylinder", 4, None),
        ("canola", "cylinder", 13, None),
        ("arabidopsis", "cylinder", 1, None),
        ("alfalfa", "cylinder", 30, None),
        ("canola", "multiplant cylinder", 20, None),
    ],
)
def test_past_window_age_on_production_catalog(species, mode, age, expected):
    """The matching age when clamped, else None."""
    params = _params(species=species, mode=mode, age=age)
    assert past_window_age(params, production_cards()) == expected


def test_past_window_age_empty_catalog_is_none():
    assert past_window_age(_params(age=100), []) is None


def test_past_window_age_is_mode_scoped_and_counts_overridden_cards():
    cyl = _card("primary", selectors=[("canola", "cylinder", 2, 13)])
    multi = _card("lateral", selectors=[("canola", "multiplant cylinder", 2, 20)])
    assert past_window_age(_params(species="canola", age=15), [cyl, multi]) == 13
    primary = _card("primary", selectors=[("rice", "cylinder", 2, 10)])
    lateral = _card("lateral", selectors=[("rice", "cylinder", 2, 5)])
    overrides = {"primary": _override("primary")}
    assert past_window_age(_params(age=18), [primary, lateral], overrides) == 10


def test_past_window_age_none_when_every_species_mode_root_type_overridden():
    primary = _card("primary", selectors=[("rice", "cylinder", 2, 10)])
    lateral = _card("lateral", selectors=[("arabidopsis", "cylinder", 2, 14)])
    overrides = {"primary": _override("primary")}
    assert past_window_age(_params(age=18), [primary, lateral], overrides) is None


# --- Shared-case rows from talmolab/sleap-roots#272 not pinned above ----------------------

_RICE_YOUNGER = {"primary": "rice-younger-primary", "crown": "rice-younger-crown"}

_SHARED_PRODUCTION_ROWS = [
    ("rice", "cylinder", 1, {}, None),
    ("rice", "cylinder", 3, _RICE_YOUNGER, None),
    ("rice", "cylinder", 10, {"crown": "rice-older-crown"}, None),
    ("rice", "cylinder", 11, {"crown": "rice-older-crown"}, 10),
    ("rice", "cylinder", 99, {"crown": "rice-older-crown"}, 10),
    (
        "soybean",
        "cylinder",
        9,
        {"primary": "soybean-primary", "lateral": "soybean-lateral"},
        8,
    ),
    ("pennycress", "cylinder", 20, {"primary": _CPA, "lateral": "canola-lateral"}, 14),
    (
        "arabidopsis",
        "multiplant cylinder",
        28,
        {"primary": _CPA, "lateral": "arabidopsis-lateral"},
        14,
    ),
]


@pytest.mark.parametrize(
    "species,mode,age,expected,matched_as", _SHARED_PRODUCTION_ROWS
)
def test_shared_case_rows_on_production_catalog(
    species, mode, age, expected, matched_as
):
    """Rows of the cross-repo shared case list, on the production-shaped catalog."""
    params = _params(species=species, mode=mode, age=age)
    assert _stems(choose_models(params, production_cards())) == expected
    assert past_window_age(params, production_cards()) == matched_as


def test_shared_case_gap_rows():
    """Gap cards 2-5 and 8-10: 6 is in the gap (not clamped), 11 is matched at 10."""
    young = _card(
        "primary", registry_id="reg/young", selectors=[("x", "cylinder", 2, 5)]
    )
    old = _card("primary", registry_id="reg/old", selectors=[("x", "cylinder", 8, 10)])
    gap = _params(species="x", age=6)
    assert choose_models(gap, [young, old]) == {}
    assert past_window_age(gap, [young, old]) is None
    past = _params(species="x", age=11)
    assert choose_models(past, [young, old])["primary"].registry_id == "reg/old"
    assert past_window_age(past, [young, old]) == 10


def test_shared_case_per_mode_multiplant_row():
    """Canola multiplant 15 is inside its own 2-20 window, so it is not clamped."""
    cyl = _card("primary", selectors=[("canola", "cylinder", 2, 13)])
    multi = _card(
        "lateral",
        registry_id="reg/multi",
        selectors=[("canola", "multiplant cylinder", 2, 20)],
    )
    params = _params(species="canola", mode="multiplant cylinder", age=15)
    assert set(choose_models(params, [cyl, multi])) == {"lateral"}
    assert past_window_age(params, [cyl, multi]) is None


def test_window_maximum_uses_only_the_matching_selectors():
    """On a card shared by canola (2-13) and pennycress (2-14), each keeps its own maximum."""
    card = _card("primary", selectors=_CANOLA_PENNYCRESS)
    assert past_window_age(_params(species="canola", age=14), [card]) == 13
    assert past_window_age(_params(species="pennycress", age=15), [card]) == 14


def _reference_selection(params, cards, overrides):
    """Independent reference: match non-overridden root types at past_window_age or age."""
    age = int(params.values["age"])
    species, mode = params.values["species"], params.values["mode"]
    match_age = past_window_age(params, cards, overrides) or age
    selected = dict(overrides)
    for root_type in {c.root_type for c in cards} - set(overrides):
        matches = [
            c
            for c in cards
            if c.root_type == root_type
            and any(
                s.species == species
                and s.mode == mode
                and s.age_min <= match_age <= s.age_max
                for s in c.selectors
            )
        ]
        if matches:
            selected[root_type] = matches[0].registry_id
    return {
        r: (v if isinstance(v, str) else v.registry_id) for r, v in selected.items()
    }


_CONTEXTS = sorted(
    {(s.species, s.mode) for c in production_cards() for s in c.selectors}
)


@pytest.mark.parametrize("species,mode", _CONTEXTS)
@pytest.mark.parametrize("override_primary", [False, True])
def test_choose_models_agrees_with_past_window_age(species, mode, override_primary):
    """The clamp has one source: selection equals matching at past_window_age(...) or age."""
    cards = production_cards()
    overrides = {"primary": _override("primary")} if override_primary else {}
    for age in range(0, 41):
        params = _params(species=species, mode=mode, age=age)
        got = choose_models(params, cards, overrides=overrides)
        got = {r: ref.registry_id for r, ref in got.items()}
        assert got == _reference_selection(params, cards, overrides), (
            species,
            mode,
            age,
        )
