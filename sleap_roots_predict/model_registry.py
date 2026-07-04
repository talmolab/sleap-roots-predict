"""Model-card sources.

A ``ModelCardSource`` has two jobs: ``list_cards`` (the catalog used by the
selection matcher) and ``materialize`` (turn a chosen ``ModelRef`` into a local
model directory that ``make_predictor`` can load).

``LocalCardSource`` is the offline/no-network implementation used for tests and
filesystem-only deployments. ``WandbRegistrySource`` (added later) confines all
network access to itself.
"""

from pathlib import Path
from typing import Dict, List, Protocol, Tuple, Union, runtime_checkable

from sleap_roots_contracts import ModelCard, ModelRef


@runtime_checkable
class ModelCardSource(Protocol):
    """A catalog of production model cards plus a way to fetch them locally."""

    def list_cards(self) -> List[ModelCard]:
        """Return the catalog of available model cards (metadata only)."""

    def materialize(self, ref: ModelRef) -> Path:
        """Return a local model directory for ``ref`` (loadable by make_predictor)."""


class LocalCardSource:
    """A filesystem-backed ``ModelCardSource`` over on-disk model directories.

    ``ModelCard`` carries no filesystem path, so this source is built from
    ``(card, path)`` pairs and keeps a ``(registry_id, version) -> Path`` map to
    resolve a ``ModelRef`` back to its directory. It performs no network access,
    which lets the full selection -> load path run offline with real models.
    """

    def __init__(self, entries: List[Tuple[ModelCard, Union[str, Path]]]) -> None:
        """Build a source from ``(ModelCard, path)`` pairs.

        Args:
            entries: Pairs of a card and the local directory holding its model.
        """
        self._cards: List[ModelCard] = [card for card, _ in entries]
        self._paths: Dict[Tuple[str, str], Path] = {
            (card.registry_id, card.version): Path(path) for card, path in entries
        }

    def list_cards(self) -> List[ModelCard]:
        """Return the cards this source was built with."""
        return list(self._cards)

    def materialize(self, ref: ModelRef) -> Path:
        """Return the on-disk directory mapped to ``ref``'s identity.

        Args:
            ref: The chosen model reference.

        Returns:
            The local model directory for ``(ref.registry_id, ref.version)``.

        Raises:
            KeyError: If no directory is mapped for the ref (fail-loud, so the
                warm worker surfaces it rather than silently skipping).
        """
        key = (ref.registry_id, ref.version)
        if key not in self._paths:
            raise KeyError(
                f"No local model directory for {ref.registry_id}:{ref.version}"
            )
        return self._paths[key]
