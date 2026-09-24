"""Duck-typed wandb fakes shared by the registry and batch tests.

Data holders, not mocks: each carries exactly the attributes the code under test reads.
"""


class FakeArtifact:
    """A duck-typed stand-in for a wandb artifact (data holder, not a mock).

    Carries exactly the attributes ``_collect_cards`` / ``_card_from_artifact``
    read: ``aliases``, ``metadata``, ``qualified_name``, ``version``, ``digest``.
    """

    def __init__(self, registry_id, *, metadata, version="v1", aliases=("production",)):
        """Build a fake artifact with the read attributes set from the args."""
        self.qualified_name = f"{registry_id}:{version}"
        self.version = version
        self.digest = f"sha256:{registry_id}"
        self.aliases = list(aliases)
        self.metadata = metadata


class FakeCollection:
    """A duck-typed stand-in for a wandb artifact collection."""

    def __init__(self, name):
        """Store the collection name the traversal reads."""
        self.name = name


class FakeApi:
    """A duck-typed stand-in for ``wandb.Api`` recording the calls it receives."""

    def __init__(self, collections):
        """Build from a ``{collection_name: [artifacts]}`` mapping."""
        self._collections = collections
        self.artifacts_calls = []
        self.project_name = None

    def artifact_collections(self, project_name, type_name):
        """Record the project + type and return the fake collections."""
        self.project_name = project_name
        self.type_name = type_name
        return [FakeCollection(name) for name in self._collections]

    def artifacts(self, type_name, name):
        """Record the query and return the collection's artifacts."""
        self.artifacts_calls.append((type_name, name))
        return list(self._collections[name.rsplit("/", 1)[-1]])


def _flat_meta():
    return {
        "species": "rice",
        "mode": "cylinder",
        "age_min": 2,
        "age_max": 5,
        "root_type": "primary",
    }
