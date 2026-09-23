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
    meta = {
        "species": sp,
        "mode": md,
        "age_min": lo,
        "age_max": hi,
        "root_type": root_type,
    }
    for key in drop:
        del meta[key]
    return meta
