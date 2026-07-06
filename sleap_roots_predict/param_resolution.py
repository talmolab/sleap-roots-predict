"""Pure param-resolution oracle: Bloom scan metadata -> ``ResolvedParams``.

Maps a single Bloom ``cyl_scans_extended`` row (the dict bloomcli's download
writes to ``scans.csv``) to a ``ResolvedParams`` carrying ``species``, ``mode``,
and ``age``, so predict's ``choose_models`` can select production models from
real Bloom metadata (metadata -> params -> model). The mapper is pure: no
network access and no filesystem I/O, and it does not mutate its input.
"""

import math
from typing import Any, Dict, Optional

from sleap_roots_contracts import ResolvedParams

# Bloom ``cyl_scans_extended`` column names (bloomcli's ``scans.csv``). Module
# constants so the cross-repo coupling to bloomcli's schema is explicit/greppable.
SPECIES_NAME_FIELD = "species_name"
PLANT_AGE_DAYS_FIELD = "plant_age_days"

# The resolvable param space; overrides are restricted to these keys.
_PARAM_KEYS = ("species", "mode", "age")

# Extension seam: Bloom ``species_name`` (lowercased) -> ``ModelCard`` species
# vocabulary, for names that do NOT lowercase cleanly to the card string (e.g. a
# Latin binomial). The seeded common names lowercase cleanly, so this ships
# empty; add lowercase-keyed entries only for genuine non-identity aliases.
_ALIASES: Dict[str, str] = {}


def _normalize_species(name: Any) -> str:
    """Normalize a Bloom species_name to the ModelCard species vocabulary.

    Strips and lowercases the name, then applies the (lowercase-keyed) alias
    map with a lowercase passthrough fallback. Blank, ``None``, or non-string
    (e.g. ``NaN``) inputs normalize to ``""`` so callers can treat them as not
    provided.
    """
    if name is None or (isinstance(name, float) and math.isnan(name)):
        return ""
    key = str(name).strip().lower()
    return _ALIASES.get(key, key)


def _mode_for_scan(metadata: Dict[str, Any]) -> str:
    """Return the imaging modality for a scan (the single mode-decision point).

    The cylinder pipeline yields cylinder scans only, so this returns
    ``"cylinder"``. GraviScan/multiscanner modes slot in here once their
    scanners and models exist; the returned string MUST match the exact seeded
    ``ModelCard`` mode vocabulary.
    """
    return "cylinder"


def _coerce_age(raw_age: Any) -> int:
    """Coerce a plant_age_days value to a whole number of days (int).

    Accepts an int or an int-coercible whole-number string; rejects bools,
    non-whole floats, and non-coercible values with a ``ValueError`` naming
    ``age`` — mirroring ``choose_models`` so the resolved ``age`` (and therefore
    ``param_hash``) is never derived from a lossy conversion.
    """
    if isinstance(raw_age, bool):
        raise ValueError(f"Scan param 'age' must be a whole number, got {raw_age!r}")
    try:
        age = int(raw_age)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Scan param 'age' must be a whole number, got {raw_age!r}"
        ) from e
    if isinstance(raw_age, float) and float(age) != raw_age:
        raise ValueError(f"Scan param 'age' must be a whole number, got {raw_age!r}")
    return age


def resolve_params(
    metadata: Dict[str, Any],
    overrides: Optional[Dict[str, Any]] = None,
) -> ResolvedParams:
    """Resolve a Bloom scan-metadata row to ``ResolvedParams`` (species/mode/age).

    Args:
        metadata: A single Bloom ``cyl_scans_extended`` row — the shape
            bloomcli's download writes to ``scans.csv``. Load-bearing fields:
            ``species_name`` (-> ``species``), ``plant_age_days`` (-> ``age``,
            days); the scanner determines ``mode``. Other columns are ignored.
            A blank or absent load-bearing field is treated as not provided.
        overrides: Optional param-space dict whose keys are a subset of
            ``{"species", "mode", "age"}``. Each key wins its field over the
            derived value; override values are normalized/coerced by the same
            rules as derived values so ``param_hash`` is representation-
            independent.

    Returns:
        A ``ResolvedParams`` whose ``values`` contain ``species``, ``mode``, and
        ``age``; the contract computes ``param_hash``.

    Raises:
        ValueError: If an override key is not in ``{"species", "mode", "age"}``;
            if a present ``plant_age_days`` (or ``age`` override) is not a whole
            number; or if ``species``, ``mode``, or ``age`` is still missing
            after merging overrides.
    """
    overrides = overrides or {}
    unknown = set(overrides) - set(_PARAM_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown override key(s): {sorted(unknown)}; "
            f"allowed keys are {list(_PARAM_KEYS)}"
        )

    # Tolerant read: mode always; species/age only when present (blank -> handled
    # below). Absent fields are omitted, deferring to overrides then validation.
    values: Dict[str, Any] = {"mode": _mode_for_scan(metadata)}
    if metadata.get(SPECIES_NAME_FIELD) is not None:
        values["species"] = metadata[SPECIES_NAME_FIELD]
    if metadata.get(PLANT_AGE_DAYS_FIELD) is not None:
        values["age"] = metadata[PLANT_AGE_DAYS_FIELD]

    values = {**values, **overrides}  # override wins, per field

    # Canonicalize derived OR override values so param_hash is representation-
    # independent; a blank species is treated as absent (fails validation below).
    if "species" in values:
        species = _normalize_species(values["species"])
        if species:
            values["species"] = species
        else:
            del values["species"]
    if "age" in values:
        values["age"] = _coerce_age(values["age"])

    missing = [key for key in _PARAM_KEYS if key not in values]
    if missing:
        raise ValueError(f"Missing required scan param(s): {missing}")

    return ResolvedParams(values=values)
