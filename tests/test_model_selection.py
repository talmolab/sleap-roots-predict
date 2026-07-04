"""Tests for the pure model-selection matcher (``choose_models``).

Real, no-mock tests over tiny hand-built ``ModelCard`` lists. The matcher is pure:
no network, no per-call filesystem I/O (the runtime sleap-nn version is resolved
once at import).
"""

from importlib.metadata import version

import pytest
from sleap_roots_contracts import ModelCard, ModelRef, ResolvedParams

from sleap_roots_predict.model_selection import choose_models


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
):
    """Build a ModelCard with sensible defaults for one root type."""
    return ModelCard(
        species=species,
        mode=mode,
        age_min=age_min,
        age_max=age_max,
        root_type=root_type,
        registry_id=registry_id or f"reg/{species}-{root_type}",
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


def test_age_outside_window_skips_root_type():
    """An age past age_max leaves that root type unmatched."""
    result = choose_models(_params(age=6), [_card("primary", age_min=2, age_max=5)])
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


def test_runtime_sleap_nn_version_resolved_at_import():
    """The stamped version is the installed sleap-nn, resolved once at import."""
    from sleap_roots_predict import model_selection

    assert model_selection._RUNTIME_SLEAP_NN_VERSION == version("sleap-nn")
