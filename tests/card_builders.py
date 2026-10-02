"""Shared ``ModelCard`` builders for tests.

The one place that knows the card's selection shape, so a contract reshape touches only this
file. Each test module keeps a thin ``_card`` wrapper with its own defaults. Cards carry a
tuple of ``Selector``s (contracts 0.1.0a9); ``selectors=`` takes an iterable of
``(species, mode, age_min, age_max)`` tuples, one per selection context.
"""

from typing import Iterable, Optional, Sequence, Tuple

from sleap_roots_contracts import ModelCard, Selector

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
    return ModelCard(
        selectors=tuple(
            Selector(species=sp, mode=md, age_min=lo, age_max=hi)
            for sp, md, lo, hi in sels
        ),
        root_type=root_type,
        registry_id=registry_id or f"reg/{sels[0][0]}-{root_type}",
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
    selector_dicts = [
        {"species": sp, "mode": md, "age_min": lo, "age_max": hi}
        for sp, md, lo, hi in sels
    ]
    for key in drop:
        del selector_dicts[0][key]
    return {"selectors": selector_dicts, "root_type": root_type}


_CYL = "cylinder"
_MULTI = "multiplant cylinder"

# The live production catalog's shape (registry sleap-roots-models, alias production) as
# verified for update-past-window-model-selection: (root_type, registry-id stem, selectors).
_PRODUCTION = (
    (
        "primary",
        "canola_pennycress_arabidopsis-primary",
        (
            ("arabidopsis", _CYL, 2, 14),
            ("arabidopsis", _MULTI, 2, 14),
            ("canola", _CYL, 2, 13),
            ("pennycress", _CYL, 2, 14),
        ),
    ),
    ("primary", "soybean-primary", (("soybean", _CYL, 2, 8),)),
    ("primary", "rice-younger-primary", (("rice", _CYL, 2, 5),)),
    (
        "lateral",
        "canola-lateral",
        (("canola", _CYL, 2, 13), ("pennycress", _CYL, 2, 14)),
    ),
    (
        "lateral",
        "arabidopsis-lateral",
        (("arabidopsis", _CYL, 2, 14), ("arabidopsis", _MULTI, 2, 14)),
    ),
    ("lateral", "soybean-lateral", (("soybean", _CYL, 2, 8),)),
    ("crown", "rice-younger-crown", (("rice", _CYL, 2, 5),)),
    ("crown", "rice-older-crown", (("rice", _CYL, 6, 10),)),
)


def production_cards():
    """Cards shaped like the production catalog; registry ids are ``reg/<stem>``."""
    return [
        make_card(root, f"reg/{stem}", selectors=sels, weights_checksum=f"sha-{stem}")
        for root, stem, sels in _PRODUCTION
    ]
