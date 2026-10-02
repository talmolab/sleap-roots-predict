"""Pure model-selection matcher.

Maps resolved scan params (species/mode/age) and a list of production
``ModelCard``s to a ``ModelRef`` per root type, mirroring the proven
models-downloader selection semantics: explicit override wins; otherwise a
card matches when some single one of its ``selectors`` matches
``species``/``mode``/inclusive age window together; exactly one match selects,
zero skips, more than one is an ambiguity error. A scan older than every window
for its species and mode is matched at that species' highest ``age_max`` (see
``past_window_age``); its params keep the real age.

The matcher is pure: no network, no per-call filesystem I/O and no logging. The
runtime sleap-nn version stamped into each ``ModelRef`` is resolved once at import.
"""

from importlib.metadata import version
from typing import Dict, List, Optional, Tuple

from sleap_roots_contracts import ModelCard, ModelRef, ResolvedParams, RootType

# Resolved once at import so choose_models performs no per-call filesystem I/O.
_RUNTIME_SLEAP_NN_VERSION = version("sleap-nn")

_REQUIRED_PARAMS = ("species", "mode", "age")


def _validated(params: ResolvedParams) -> Tuple[str, str, int]:
    """Return the scan's ``(species, mode, age)``, with ``age`` coerced to an int.

    Args:
        params: Resolved scan params; ``values`` must contain ``species``, ``mode`` and
            ``age``.

    Raises:
        ValueError: If a required param is missing or ``age`` is not a whole number.
    """
    values = params.values
    for key in _REQUIRED_PARAMS:
        if key not in values:
            raise ValueError(f"Missing required scan param: {key!r}")

    # Reject bools and non-whole numbers rather than silently truncating: 3.5 -> 3
    # would select the wrong age-window model and diverge from the reproducibility
    # hash (ResolvedParams.param_hash keeps 3.5). A whole-number string ("3") is fine.
    raw_age = values["age"]
    if isinstance(raw_age, bool):
        raise ValueError(f"Scan param 'age' must be an integer, got bool {raw_age!r}")
    try:
        age = int(raw_age)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Scan param 'age' must be a whole number, got {raw_age!r}"
        ) from e
    if isinstance(raw_age, float) and float(age) != raw_age:
        raise ValueError(f"Scan param 'age' must be a whole number, got {raw_age!r}")
    return values["species"], values["mode"], age


def _has_context(card: ModelCard, species: str, mode: str) -> bool:
    """Whether ``card`` has a selector for this species and mode, at any age."""
    return any(s.species == species and s.mode == mode for s in card.selectors)


def _window_max(cards: List[ModelCard], species: str, mode: str) -> Optional[int]:
    """The largest ``age_max`` among selectors with this species and mode, on any card."""
    return max(
        (
            s.age_max
            for card in cards
            for s in card.selectors
            if s.species == species and s.mode == mode
        ),
        default=None,
    )


def _card_matches(card: ModelCard, species: str, mode: str, age: int) -> bool:
    """Whether some single selector on ``card`` matches species, mode and age together.

    ``age`` is the *matching* age (the scan age, or the window maximum for a past-window
    scan). It is compared against the matching selector's own window, never a window taken
    across the card's selectors, and selectors are never combined (no cross product).
    """
    return any(
        s.species == species and s.mode == mode and s.age_min <= age <= s.age_max
        for s in card.selectors
    )


def past_window_age(
    params: ResolvedParams,
    cards: List[ModelCard],
    overrides: Optional[Dict[RootType, ModelRef]] = None,
) -> Optional[int]:
    """The age a past-window scan is matched at, or ``None`` when it is not clamped.

    A scan is clamped when its age is above every selector window for its species and
    mode (the window maximum is taken across all cards, including those of overridden
    root types) and some root type that is not overridden has a card with a selector for
    that species and mode. Callers use this to warn; ``choose_models`` matches the
    non-overridden root types at the same age.

    Args:
        params: Resolved scan params, validated exactly as ``choose_models`` does.
        cards: Candidate production model cards.
        overrides: Optional explicit ``ModelRef`` per root type.

    Returns:
        The window maximum when the scan is clamped, else ``None``.

    Raises:
        ValueError: If a required param is missing or ``age`` is not a whole number.
    """
    species, mode, age = _validated(params)
    window_max = _window_max(cards, species, mode)
    if window_max is None or age <= window_max:
        return None
    overrides = overrides or {}
    affected = any(
        card.root_type not in overrides and _has_context(card, species, mode)
        for card in cards
    )
    return window_max if affected else None


def choose_models(
    params: ResolvedParams,
    cards: List[ModelCard],
    overrides: Optional[Dict[RootType, ModelRef]] = None,
) -> Dict[RootType, ModelRef]:
    """Select at most one model per root type for a scan.

    Args:
        params: Resolved scan params; ``values`` must contain ``species``,
            ``mode``, and ``age`` (``age`` may be any int-coercible value). They are
            never modified: a past-window scan keeps its real age.
        cards: Candidate production model cards, each already carrying a concrete
            ``version``/``weights_checksum`` (alias resolution happens upstream in
            the card source, not here).
        overrides: Optional explicit ``ModelRef`` per root type. An override wins
            for its root type and bypasses card matching entirely.

    Returns:
        A mapping of root type to the selected ``ModelRef``. Cards are matched at the
        scan age, or at the species' window maximum when the scan is older than every
        window for its species and mode. Root types with no matching card (and no
        override) are absent (skipped); the mapping is empty when nothing matches.

    Raises:
        ValueError: If a required param is missing, ``age`` is not int-coercible,
            or more than one card matches a single root type (ambiguous).
    """
    overrides = overrides or {}
    species, mode, age = _validated(params)

    # One source for the clamp. When past_window_age returns None for a past-window age,
    # every root type with a selector for this species and mode is overridden, so the
    # matching age cannot change any selection.
    clamped = past_window_age(params, cards, overrides)
    match_age = age if clamped is None else clamped
    age_desc = (
        f"age={age}" if match_age == age else f"age={age} (matched as {match_age})"
    )

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
            and _card_matches(card, species, mode, match_age)
        ]
        if not matches:
            continue
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous model selection for root type {root_type!r}: "
                f"{len(matches)} cards match species={species!r}, mode={mode!r}, "
                f"{age_desc}"
            )
        selected[root_type] = matches[0].to_model_ref(_RUNTIME_SLEAP_NN_VERSION)

    return selected
