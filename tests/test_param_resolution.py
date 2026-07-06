"""Tests for the pure param-resolution oracle (``resolve_params``).

Real, no-mock, offline tests. ``resolve_params`` maps a single Bloom
``cyl_scans_extended`` row (the dict bloomcli writes to ``scans.csv``) to a
``ResolvedParams`` carrying ``species``/``mode``/``age``, so predict's
``choose_models`` can select production models from real Bloom metadata.
"""

import math

import pytest
from sleap_roots_contracts import ModelCard, ResolvedParams

from sleap_roots_predict import param_resolution
from sleap_roots_predict.model_selection import choose_models
from sleap_roots_predict.param_resolution import (
    PLANT_AGE_DAYS_FIELD,
    SPECIES_NAME_FIELD,
    _coerce_age,
    _mode_for_scan,
    _normalize_mode,
    _normalize_species,
    resolve_params,
)

# The verified seeded-species set (Bloom species_name -> ModelCard vocabulary).
# Recorded here as the executable home for the "these lowercase cleanly" fact
# (design decision: _ALIASES ships empty; this test owns the seeded record).
_SEEDED_SPECIES = {
    "Pennycress": "pennycress",
    "Arabidopsis": "arabidopsis",
    "Rice": "rice",
    "Soybean": "soybean",
    "Canola": "canola",
}


def _row(species_name="Pennycress", plant_age_days=14, **extra):
    """A minimal cyl_scans_extended row with optional extra/override columns."""
    row = {SPECIES_NAME_FIELD: species_name, PLANT_AGE_DAYS_FIELD: plant_age_days}
    row.update(extra)
    return row


def _card(root_type, *, species="rice", mode="cylinder", age_min=2, age_max=5):
    """Build a ModelCard with sensible defaults for one root type."""
    return ModelCard(
        species=species,
        mode=mode,
        age_min=age_min,
        age_max=age_max,
        root_type=root_type,
        registry_id=f"reg/{species}-{root_type}",
        version="v1",
        weights_checksum="sha",
    )


# --- field constants (bloomcli coupling) -----------------------------------


def test_field_constants_match_bloomcli_columns():
    """The load-bearing field constants match bloomcli's scans.csv columns."""
    assert SPECIES_NAME_FIELD == "species_name"
    assert PLANT_AGE_DAYS_FIELD == "plant_age_days"


# --- _normalize_species ----------------------------------------------------


@pytest.mark.parametrize("bloom_name,expected", list(_SEEDED_SPECIES.items()))
def test_seeded_species_normalize_to_card_vocabulary(bloom_name, expected):
    """Each seeded Bloom species_name maps to its lowercase card string."""
    assert _normalize_species(bloom_name) == expected


def test_species_whitespace_and_case_normalized():
    """Surrounding whitespace and case are stripped/lowered."""
    assert _normalize_species("  Rice  ") == "rice"


def test_unknown_species_passes_through_lowercased():
    """An unmodelled species passes through (registry is the authority)."""
    assert _normalize_species("Sorghum") == "sorghum"


@pytest.mark.parametrize("blank", ["", "   ", None, math.nan])
def test_blank_or_nonstring_species_returns_empty(blank):
    """Blank/None/NaN species normalize to '' (treated as absent upstream)."""
    assert _normalize_species(blank) == ""


def test_alias_map_substitutes_a_non_identity_alias(monkeypatch):
    """The (normally empty) _ALIASES seam maps a non-trivial name when populated."""
    monkeypatch.setitem(param_resolution._ALIASES, "thlaspi arvense", "pennycress")
    assert _normalize_species("Thlaspi arvense") == "pennycress"


# --- _normalize_mode -------------------------------------------------------


def test_mode_normalizes_case_and_whitespace():
    """Mode is stripped and lowercased, mirroring species normalization."""
    assert _normalize_mode("  Cylinder ") == "cylinder"


# --- _mode_for_scan --------------------------------------------------------


def test_mode_for_scan_returns_cylinder():
    """The cylinder stage-in path resolves mode 'cylinder'."""
    assert _mode_for_scan(_row()) == "cylinder"


def test_mode_matches_seeded_card_vocabulary():
    """The mode string equals the seeded ModelCard mode vocabulary."""
    card = _card("primary", mode="cylinder")
    assert _mode_for_scan(_row()) == card.mode


# --- _coerce_age -----------------------------------------------------------


def test_int_age_passes_through():
    """An integer age passes through as days."""
    assert _coerce_age(14) == 14


def test_string_age_is_coerced_to_int():
    """An int-coercible string age is coerced to the same int."""
    result = _coerce_age("14")
    assert result == 14
    assert isinstance(result, int)


def test_age_zero_is_valid():
    """Age 0 is a valid coerced value (not treated as missing)."""
    assert _coerce_age(0) == 0


def test_whole_float_age_is_coerced_to_int():
    """A whole float (pandas float64 from a NaN-containing column) coerces cleanly."""
    result = _coerce_age(14.0)
    assert result == 14
    assert isinstance(result, int)


@pytest.mark.parametrize("bad_age", [14.5, "14.5", "abc", True])
def test_non_whole_or_non_coercible_age_raises_naming_age(bad_age):
    """Fractional/non-coercible/bool ages raise a ValueError naming 'age'."""
    with pytest.raises(ValueError, match="age"):
        _coerce_age(bad_age)


# --- resolve_params: core mapping ------------------------------------------


def test_sample_row_resolves_to_species_mode_age():
    """The oracle: a sample Bloom row -> {pennycress, cylinder, 14} + hash."""
    params = resolve_params(_row(species_name="Pennycress", plant_age_days=14))
    assert isinstance(params, ResolvedParams)
    assert params.values == {"species": "pennycress", "mode": "cylinder", "age": 14}
    assert params.param_hash  # populated by the contract


def test_extra_columns_ignored_and_input_not_mutated():
    """Only the 3 fields are used; the passed-in metadata dict is unchanged."""
    row = _row(
        species_name="Rice",
        plant_age_days=3,
        species_genus="Oryza",
        scan_id="abc123",
        captured_at="2026-07-06T00:00:00Z",
    )
    before = dict(row)
    params = resolve_params(row)
    assert set(params.values) == {"species", "mode", "age"}
    assert row == before  # not mutated


def test_string_age_row_hashes_identically_to_int_age_row():
    """A CSV string age produces the same param_hash as the int age."""
    as_str = resolve_params(_row(species_name="Rice", plant_age_days="14"))
    as_int = resolve_params(_row(species_name="Rice", plant_age_days=14))
    assert as_str.values == as_int.values
    assert as_str.param_hash == as_int.param_hash


def test_row_age_zero_resolves():
    """A row with plant_age_days=0 resolves (0 is not treated as missing)."""
    params = resolve_params(_row(species_name="Rice", plant_age_days=0))
    assert params.values["age"] == 0


def test_row_non_whole_age_raises_naming_age():
    """A non-whole plant_age_days in the row raises naming 'age'."""
    with pytest.raises(ValueError, match="age"):
        resolve_params(_row(species_name="Rice", plant_age_days=14.5))


# --- resolve_params: overrides ---------------------------------------------


def test_override_wins_per_field():
    """Overrides replace derived fields; non-overridden fields stay derived."""
    params = resolve_params(
        _row(species_name="Rice", plant_age_days=3),
        overrides={"mode": "graviscan", "species": "canola"},
    )
    assert params.values == {"species": "canola", "mode": "graviscan", "age": 3}


def test_empty_overrides_equals_no_overrides():
    """An empty overrides dict changes nothing."""
    row = _row(species_name="Rice", plant_age_days=3)
    assert resolve_params(row, overrides={}).values == resolve_params(row).values


def test_unknown_override_key_raises_naming_key():
    """A typo'd override key raises a ValueError naming the offending key."""
    with pytest.raises(ValueError, match="specis"):
        resolve_params(_row(), overrides={"specis": "rice"})


def test_override_values_are_canonicalized_like_derived():
    """Override values are normalized/coerced so param_hash is stable."""
    overridden = resolve_params(
        _row(species_name="Pennycress", plant_age_days=99),
        overrides={"species": "Rice", "age": "14"},
    )
    derived = resolve_params(_row(species_name="Rice", plant_age_days=14))
    assert overridden.values == {"species": "rice", "mode": "cylinder", "age": 14}
    assert overridden.param_hash == derived.param_hash


def test_mode_override_is_canonicalized():
    """A mode override is normalized like a derived mode (representation-stable)."""
    overridden = resolve_params(_row(), overrides={"mode": "  Cylinder "})
    derived = resolve_params(_row())
    assert overridden.values["mode"] == "cylinder"
    assert overridden.param_hash == derived.param_hash


def test_override_age_true_raises_naming_age():
    """A bool age override is rejected on the override path too, naming age."""
    with pytest.raises(ValueError, match="age"):
        resolve_params(_row(), overrides={"age": True})


# --- resolve_params: strict validation -------------------------------------


def test_missing_both_fields_raises_naming_each():
    """A row missing species_name and plant_age_days names both params."""
    with pytest.raises(ValueError) as excinfo:
        resolve_params({})
    message = str(excinfo.value)
    assert "species" in message
    assert "age" in message


@pytest.mark.parametrize("blank", ["", "   ", None, math.nan])
def test_blank_species_raises_naming_species(blank):
    """A blank species_name (any form, no override) fails loud naming species."""
    with pytest.raises(ValueError, match="species"):
        resolve_params(_row(species_name=blank, plant_age_days=3))


@pytest.mark.parametrize("blank", ["", "   ", None, math.nan])
def test_blank_age_treated_as_missing_naming_age(blank):
    """A blank plant_age_days is treated as not provided (names age, not 'whole')."""
    with pytest.raises(ValueError, match="age"):
        resolve_params(_row(species_name="Rice", plant_age_days=blank))


def test_blank_age_can_be_supplied_by_override():
    """A blank plant_age_days is satisfied by an age override (defer, not raise)."""
    params = resolve_params(
        _row(species_name="Rice", plant_age_days=math.nan), overrides={"age": 5}
    )
    assert params.values["age"] == 5


def test_both_blank_fields_name_each_missing_param():
    """Blank species AND blank age name both (age blank must not mask species)."""
    with pytest.raises(ValueError) as excinfo:
        resolve_params(_row(species_name="", plant_age_days=math.nan))
    message = str(excinfo.value)
    assert "species" in message
    assert "age" in message


def test_blank_mode_override_raises_naming_mode():
    """A blank mode override is treated as absent and fails loud naming mode."""
    with pytest.raises(ValueError, match="mode"):
        resolve_params(
            _row(species_name="Rice", plant_age_days=3), overrides={"mode": ""}
        )


def test_missing_species_supplied_by_override_succeeds():
    """A missing species_name compensated by an override resolves."""
    row = {PLANT_AGE_DAYS_FIELD: 3}  # no species_name
    params = resolve_params(row, overrides={"species": "rice"})
    assert params.values["species"] == "rice"


# --- payoff round-trip: metadata -> params -> model ------------------------


def test_round_trip_selects_expected_models():
    """choose_models(resolve_params(row), cards) selects a ref per root type."""
    cards = [_card("primary"), _card("crown")]
    row = _row(species_name="Rice", plant_age_days=3)
    result = choose_models(resolve_params(row), cards)
    assert set(result) == {"primary", "crown"}
    assert result["primary"].version == "v1"


def test_round_trip_unknown_species_selects_nothing():
    """An unmodelled species resolves and zero-matches (skip, not error)."""
    cards = [_card("primary"), _card("crown")]
    row = _row(species_name="Sorghum", plant_age_days=3)
    assert choose_models(resolve_params(row), cards) == {}
