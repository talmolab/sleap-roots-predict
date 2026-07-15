"""choose_models round-trip vs. contracts' resolve_params (offline).

``resolve_params`` (Bloom scan metadata -> ``ResolvedParams``) is implemented and
tested in ``sleap-roots-contracts`` (the single source of truth; see its
``tests/test_params.py``). What remains to test here is the interop: that a
``ResolvedParams`` produced by that external implementation still drives
predict's ``choose_models`` correctly (the metadata -> params -> model
round-trip), since ``choose_models`` lives in predict.
"""

from sleap_roots_contracts import ModelCard, resolve_params

from sleap_roots_predict.model_selection import choose_models


def _row(species_name="Pennycress", plant_age_days=14, **extra):
    """A minimal cyl_scans_extended row with optional extra/override columns."""
    row = {"species_name": species_name, "plant_age_days": plant_age_days}
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
