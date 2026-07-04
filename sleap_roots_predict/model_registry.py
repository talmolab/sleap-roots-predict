"""Model-card sources.

A ``ModelCardSource`` has two jobs: ``list_cards`` (the catalog used by the
selection matcher) and ``materialize`` (turn a chosen ``ModelRef`` into a local
model directory that ``make_predictor`` can load).

``LocalCardSource`` is the offline/no-network implementation used for tests and
filesystem-only deployments. ``WandbRegistrySource`` (added later) confines all
network access to itself.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable

from sleap_roots_contracts import ModelCard, ModelRef

# The lab wandb entity (org form is used for registry paths). Overridable via
# SRP_WANDB_ENTITY. Kept as a default so the offline path never needs it.
_DEFAULT_ENTITY = "eberrigan-salk-institute-for-biological-studies"
_DEFAULT_ALIAS = "production"


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


class WandbRegistrySource:
    """A ``ModelCardSource`` backed by the wandb model registry.

    All network access is confined to this class and uses a lazy ``import wandb``,
    so importing this module (and the offline ``LocalCardSource`` path) never
    touches wandb. Configuration is taken from the constructor or, when omitted,
    the environment: ``SRP_WANDB_ENTITY``, ``SRP_WANDB_REGISTRY``,
    ``SRP_MODEL_CACHE_DIR`` (falling back to ``WANDB_CACHE_DIR``); ``WANDB_API_KEY``
    authenticates. ``list_cards`` pins each artifact to its concrete version (not a
    moving alias) so the ``ModelRef`` built downstream is reproducible.

    Note: ``list_cards`` traverses the production registry and is validated by the
    ``@pytest.mark.wandb`` gated test against a live, populated registry; the exact
    registry identity is settled with ``sleap-roots-training``.
    """

    def __init__(
        self,
        *,
        entity: Optional[str] = None,
        registry: Optional[str] = None,
        alias: str = _DEFAULT_ALIAS,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """Configure the source (no network access happens here).

        Args:
            entity: wandb entity; defaults to ``SRP_WANDB_ENTITY`` then the lab entity.
            registry: registry name; defaults to ``SRP_WANDB_REGISTRY``.
            alias: only artifacts carrying this alias are listed (default ``production``).
            cache_dir: download cache; defaults to ``SRP_MODEL_CACHE_DIR`` then
                ``WANDB_CACHE_DIR`` then wandb's own default.
        """
        self._entity = entity or os.environ.get("SRP_WANDB_ENTITY", _DEFAULT_ENTITY)
        self._registry = registry or os.environ.get("SRP_WANDB_REGISTRY")
        self._alias = alias
        if cache_dir is not None:
            self._cache_dir: Optional[str] = str(cache_dir)
        else:
            self._cache_dir = os.environ.get("SRP_MODEL_CACHE_DIR") or os.environ.get(
                "WANDB_CACHE_DIR"
            )

    def _require_key(self) -> None:
        """Raise before any network call if credentials are absent."""
        if not os.environ.get("WANDB_API_KEY"):
            raise RuntimeError(
                "WANDB_API_KEY is not set; cannot access the wandb registry."
            )

    def _registry_project(self) -> str:
        """Return the registry's wandb project path (``<entity>-org/wandb-registry-<r>``)."""
        if not self._registry:
            raise RuntimeError(
                "No wandb registry configured; set SRP_WANDB_REGISTRY or pass registry=."
            )
        return f"{self._entity}-org/wandb-registry-{self._registry}"

    def list_cards(self) -> List[ModelCard]:
        """List the production registry's model artifacts as pinned ``ModelCard``s.

        Only artifacts carrying the configured alias are returned, each pinned to
        its concrete version + ``weights_checksum`` (digest).

        Raises:
            RuntimeError: If ``WANDB_API_KEY`` is unset (raised before any network
                call) or no registry is configured.
        """
        self._require_key()
        import wandb

        api = wandb.Api()
        project = self._registry_project()
        cards: List[ModelCard] = []
        for collection in api.artifact_collections(
            project_name=project, type_name="model"
        ):
            name = f"{project}/{collection.name}"
            for artifact in api.artifacts(type_name="model", name=name):
                if self._alias and self._alias not in (artifact.aliases or []):
                    continue
                cards.append(self._card_from_artifact(artifact))
        return cards

    @staticmethod
    def _card_from_artifact(artifact: object) -> ModelCard:
        """Build a pinned ``ModelCard`` from a wandb artifact (metadata + identity)."""
        meta = dict(getattr(artifact, "metadata", None) or {})
        qualified = getattr(artifact, "qualified_name", None) or artifact.name
        registry_id = qualified.split(":", 1)[0]  # versionless path
        meta.update(
            registry_id=registry_id,
            version=artifact.version,
            weights_checksum=artifact.digest,
        )
        return ModelCard.model_validate(meta)

    def materialize(self, ref: ModelRef) -> Path:
        """Download the pinned artifact for ``ref`` to the cache and return its dir.

        Args:
            ref: The chosen model reference (concrete ``registry_id``/``version``).

        Returns:
            The local directory the artifact was downloaded to (reused on repeat
            calls via the wandb cache).

        Raises:
            RuntimeError: If ``WANDB_API_KEY`` is unset (before any network call).
        """
        self._require_key()
        import wandb

        api = wandb.Api()
        name = (
            ref.registry_id
            if ":" in ref.registry_id
            else f"{ref.registry_id}:{ref.version}"
        )
        artifact = api.artifact(name, type="model")
        return Path(artifact.download(root=self._cache_dir))
