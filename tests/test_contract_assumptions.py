"""Pins the contract behavior predict's deploy ordering relies on.

If contracts ever gains a tolerant read of the flat card shape, both consumer generations would
validate the same cards and ``choose_models``' ambiguity raise would fire on live traffic.
"""

import pydantic
import pytest
from sleap_roots_contracts import ModelCard

_IDENTITY = {"root_type": "primary", "registry_id": "reg/x", "version": "v0"}


def test_flat_card_shape_does_not_validate():
    flat = {
        "species": "rice",
        "mode": "cylinder",
        "age_min": 2,
        "age_max": 5,
        **_IDENTITY,
    }
    with pytest.raises(pydantic.ValidationError):
        ModelCard.model_validate(flat)


def test_empty_selectors_do_not_validate():
    with pytest.raises(pydantic.ValidationError):
        ModelCard.model_validate({"selectors": [], **_IDENTITY})
