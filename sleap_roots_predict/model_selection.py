"""Pure model-selection matcher.

Maps resolved scan params (species/mode/age) and a list of production
``ModelCard``s to a ``ModelRef`` per root type, mirroring the proven
models-downloader selection semantics: explicit override wins; otherwise filter
by ``species``/``mode``/inclusive age window; exactly one match selects, zero
skips, more than one is an ambiguity error.

The matcher is pure: no network and no per-call filesystem I/O. The runtime
sleap-nn version stamped into each ``ModelRef`` is resolved once at import.
"""

from importlib.metadata import version
from typing import Dict, List, Optional

from sleap_roots_contracts import ModelCard, ModelRef, ResolvedParams, RootType

# Resolved once at import so choose_models performs no per-call filesystem I/O.
_RUNTIME_SLEAP_NN_VERSION = version("sleap-nn")

_REQUIRED_PARAMS = ("species", "mode", "age")


def choose_models(
    params: ResolvedParams,
    cards: List[ModelCard],
    overrides: Optional[Dict[RootType, ModelRef]] = None,
) -> Dict[RootType, ModelRef]:
    """Select at most one model per root type for a scan.

    Args:
        params: Resolved scan params; ``values`` must contain ``species``,
            ``mode``, and ``age`` (``age`` may be any int-coercible value).
        cards: Candidate production model cards, each already carrying a concrete
            ``version``/``weights_checksum`` (alias resolution happens upstream in
            the card source, not here).
        overrides: Optional explicit ``ModelRef`` per root type. An override wins
            for its root type and bypasses card matching entirely.

    Returns:
        A mapping of root type to the selected ``ModelRef``. Root types with no
        matching card (and no override) are absent (skipped); the mapping is empty
        when nothing matches.

    Raises:
        ValueError: If a required param is missing, ``age`` is not int-coercible,
            or more than one card matches a single root type (ambiguous).
    """
    overrides = overrides or {}
    values = params.values

    for key in _REQUIRED_PARAMS:
        if key not in values:
            raise ValueError(f"Missing required scan param: {key!r}")

    species = values["species"]
    mode = values["mode"]
    try:
        age = int(values["age"])
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Scan param 'age' must be int-coercible, got {values['age']!r}"
        ) from e

    # Resolve every root type present among the cards plus any override keys.
    root_types = {card.root_type for card in cards} | set(overrides)

    selected: Dict[RootType, ModelRef] = {}
    for root_type in root_types:
        if root_type in overrides:
            selected[root_type] = overrides[root_type]
            continue

        matches = [
            card
            for card in cards
            if card.root_type == root_type
            and card.species == species
            and card.mode == mode
            and card.age_min <= age <= card.age_max
        ]
        if not matches:
            continue
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous model selection for root type {root_type!r}: "
                f"{len(matches)} cards match species={species!r}, mode={mode!r}, "
                f"age={age}"
            )
        selected[root_type] = matches[0].to_model_ref(_RUNTIME_SLEAP_NN_VERSION)

    return selected
