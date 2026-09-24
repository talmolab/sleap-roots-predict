"""Tests for the model-card source layer.

The offline ``LocalCardSource`` tests run with no network and no mocks: cards
point at the vendored model directories and ``materialize`` returns a real dir
that ``make_predictor`` can load. Gated ``WandbRegistrySource`` tests are added in
a later task (``@pytest.mark.wandb``).
"""

import logging
import os
import re
from pathlib import Path

import pytest
from sleap_nn.inference import Predictor
from sleap_roots_contracts import ModelCard

from card_builders import make_card, raw_card_meta
from registry_fakes import FakeApi, FakeArtifact, _flat_meta
from sleap_roots_predict.model_registry import (
    LocalCardSource,
    ModelCardSource,
    NoReadableModelCardsError,
    WandbRegistrySource,
)
from sleap_roots_predict.predict import make_predictor

WANDB_API_KEY = os.environ.get("WANDB_API_KEY")


def _card(root_type, registry_id, version="v1"):
    return make_card(root_type, registry_id, version=version)


def test_local_card_source_is_a_model_card_source(native_model_dir: Path):
    """LocalCardSource satisfies the ModelCardSource protocol."""
    source = LocalCardSource([(_card("primary", "reg/native"), native_model_dir)])
    assert isinstance(source, ModelCardSource)


def test_local_card_source_lists_cards(native_model_dir: Path):
    """list_cards returns the cards it was built with, no network."""
    card = _card("primary", "reg/native")
    source = LocalCardSource([(card, native_model_dir)])
    assert source.list_cards() == [card]


def test_materialize_resolves_ref_to_mapped_dir(native_model_dir: Path):
    """Materialize resolves a ModelRef's identity to its on-disk directory."""
    card = _card("primary", "reg/native", version="v2")
    source = LocalCardSource([(card, native_model_dir)])
    ref = card.to_model_ref("runtime")
    assert source.materialize(ref) == native_model_dir


def test_materialized_dir_is_loadable_by_make_predictor(native_model_dir: Path):
    """The directory materialize returns loads as a real Predictor (no mocks)."""
    card = _card("primary", "reg/native")
    source = LocalCardSource([(card, native_model_dir)])
    model_dir = source.materialize(card.to_model_ref("runtime"))
    assert isinstance(make_predictor([model_dir]), Predictor)


def test_materialize_unknown_ref_raises(native_model_dir: Path):
    """A ModelRef with no mapped path fails loud (not a silent empty result)."""
    card = _card("primary", "reg/native")
    source = LocalCardSource([(card, native_model_dir)])
    unknown = _card("crown", "reg/missing").to_model_ref("runtime")
    with pytest.raises(KeyError, match="reg/missing"):
        source.materialize(unknown)


# --- WandbRegistrySource ------------------------------------------------------

# ``clean_wandb_env`` (hermetic env fixture) lives in tests/conftest.py.


def test_wandb_source_missing_key_raises_before_network(monkeypatch):
    """With no WANDB_API_KEY, list_cards raises a clear error (no network call)."""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    source = WandbRegistrySource(entity="an-entity", registry="a-registry")
    with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
        source.list_cards()


# --- default registry + env-var rename (group 1) ------------------------------


def test_registry_defaults_to_sleap_roots_models(clean_wandb_env):
    """With no registry arg and no env, the registry defaults to the live one."""
    assert WandbRegistrySource()._registry == "sleap-roots-models"


def test_model_registry_env_var_is_honored(clean_wandb_env):
    """SRP_WANDB_MODEL_REGISTRY sets the registry when no arg is passed."""
    clean_wandb_env.setenv("SRP_WANDB_MODEL_REGISTRY", "some-registry")
    assert WandbRegistrySource()._registry == "some-registry"


def test_legacy_registry_env_var_is_ignored(clean_wandb_env):
    """The legacy SRP_WANDB_REGISTRY is not read (hard rename): default applies."""
    clean_wandb_env.setenv("SRP_WANDB_REGISTRY", "legacy-registry")
    assert WandbRegistrySource()._registry == "sleap-roots-models"


def test_model_alias_env_var_is_honored(clean_wandb_env):
    """SRP_WANDB_MODEL_ALIAS sets the alias when no arg is passed."""
    clean_wandb_env.setenv("SRP_WANDB_MODEL_ALIAS", "staging")
    assert WandbRegistrySource()._alias == "staging"


def test_legacy_alias_env_var_is_ignored(clean_wandb_env):
    """The legacy SRP_WANDB_ALIAS is not read (hard rename): default applies."""
    clean_wandb_env.setenv("SRP_WANDB_ALIAS", "legacy-alias")
    assert WandbRegistrySource()._alias == "production"


def test_empty_registry_env_falls_back_to_default(clean_wandb_env):
    """A set-but-empty SRP_WANDB_MODEL_REGISTRY falls back to the default."""
    clean_wandb_env.setenv("SRP_WANDB_MODEL_REGISTRY", "")
    assert WandbRegistrySource()._registry == "sleap-roots-models"


def test_empty_alias_env_falls_back_to_default(clean_wandb_env):
    """A set-but-empty SRP_WANDB_MODEL_ALIAS falls back to the default alias.

    Regression guard: an empty alias must NOT disable the alias filter (which
    would list every artifact version) — it falls back to ``production``.
    """
    clean_wandb_env.setenv("SRP_WANDB_MODEL_ALIAS", "")
    assert WandbRegistrySource()._alias == "production"


def test_default_registry_still_fails_loud_without_key(clean_wandb_env):
    """WandbRegistrySource() (default registry) still raises on a missing key."""
    source = WandbRegistrySource()
    with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
        source.list_cards()


# --- per-artifact error isolation in list_cards (group 2) ---------------------


def _good_meta(species="rice", root_type="primary"):
    """Metadata carrying every required selection field (validates to a card)."""
    return raw_card_meta(species=species, root_type=root_type)


def _good_artifact(registry_id="reg/good", **kw):
    return FakeArtifact(registry_id, metadata=_good_meta(**kw))


def _malformed_artifact(registry_id="reg/bad"):
    # Missing the required ``species`` field -> pydantic ValidationError.
    return FakeArtifact(registry_id, metadata=raw_card_meta(drop=("species",)))


def test_collect_cards_skips_malformed_and_warns(caplog):
    """One malformed artifact is skipped with a warning; the good one survives."""
    source = WandbRegistrySource(alias="production")
    good, bad = _good_artifact(), _malformed_artifact()
    with caplog.at_level(logging.WARNING, logger="sleap_roots_predict.model_registry"):
        cards = source._collect_cards([good, bad])
    assert [c.registry_id for c in cards] == ["reg/good"]
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    # The warning names the offending artifact and includes the underlying error.
    assert "reg/bad" in caplog.text
    assert "species" in caplog.text


def test_collect_cards_skips_flat_cards_alongside_readable_ones(caplog):
    """Pins #34 fact 1 against a9: flat cards are skipped, selector-shaped ones returned."""
    source = WandbRegistrySource(alias="production")
    flat = FakeArtifact("reg/flat", metadata=_flat_meta())
    good = _good_artifact("reg/good")
    with caplog.at_level(logging.WARNING, logger="sleap_roots_predict.model_registry"):
        cards = source._collect_cards([flat, good])
    assert [c.registry_id for c in cards] == ["reg/good"]
    assert "reg/flat" in caplog.text and "selectors" in caplog.text


def test_collect_cards_all_invalid_raises(caplog):
    """Every alias-carrying artifact unreadable is a deployment fault, not an empty catalog.

    Reverses #32's "empty, not an exception" for this one case (predict#34): an empty catalog
    fails every scan and exits 3, which the pipeline's exit gate passes.
    """
    source = WandbRegistrySource(entity="ent", registry="reg", alias="production")
    with pytest.raises(NoReadableModelCardsError) as exc:
        source._collect_cards(
            [_malformed_artifact("reg/a"), FakeArtifact("reg/b", metadata=None)]
        )
    assert "ent-org/wandb-registry-reg" in str(exc.value)
    assert "production" in str(exc.value)
    assert re.search(r"\b2\b", str(exc.value))
    assert issubclass(NoReadableModelCardsError, ValueError)


def test_collect_cards_drops_only_the_bad_one_preserving_order(caplog):
    """A single bad artifact drops only itself; good ones keep their order."""
    source = WandbRegistrySource(alias="production")
    a = _good_artifact("reg/a")
    b = _malformed_artifact("reg/bad")
    c = _good_artifact("reg/c")
    with caplog.at_level(logging.WARNING, logger="sleap_roots_predict.model_registry"):
        cards = source._collect_cards([a, b, c])
    assert [card.registry_id for card in cards] == ["reg/a", "reg/c"]
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


def test_collect_cards_alias_filtered_is_silent(caplog):
    """An artifact lacking the configured alias is filtered out, no warning."""
    source = WandbRegistrySource(alias="production")
    wrong_alias = FakeArtifact(
        "reg/experimental", metadata=_good_meta(), aliases=["experimental"]
    )
    with caplog.at_level(logging.WARNING, logger="sleap_roots_predict.model_registry"):
        cards = source._collect_cards([wrong_alias])
    assert cards == []
    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


def test_collect_cards_pins_concrete_version_and_checksum():
    """The built card carries the artifact's concrete version + digest (pin)."""
    source = WandbRegistrySource(alias="production")
    art = _good_artifact("reg/good")
    (card,) = source._collect_cards([art])
    assert card.version == art.version
    assert card.weights_checksum == art.digest


# --- offline coverage of the registry traversal ------------------------------


def test_iter_registry_artifacts_yields_across_collections():
    """The traversal yields every model artifact across all collections, in order."""
    source = WandbRegistrySource(entity="ent", registry="reg")
    a, b, c = _good_artifact("reg/a"), _good_artifact("reg/b"), _good_artifact("reg/c")
    api = FakeApi({"colA": [a, b], "colB": [c]})
    result = list(source._iter_registry_artifacts(api))
    assert result == [a, b, c]
    # Correct registry project path + one per-collection artifact query.
    assert api.project_name == "ent-org/wandb-registry-reg"
    assert api.artifacts_calls == [
        ("model", "ent-org/wandb-registry-reg/colA"),
        ("model", "ent-org/wandb-registry-reg/colB"),
    ]


@pytest.mark.wandb
@pytest.mark.skipif(
    not WANDB_API_KEY,
    reason="requires WANDB_API_KEY + a populated production registry",
)
def test_wandb_source_lists_and_materializes(tmp_path, monkeypatch):
    """With creds and NO registry env, the default registry lists + materializes."""
    # Prove the default path: unset the registry/alias/entity env so the source
    # falls back to the live-registry defaults with only WANDB_API_KEY set.
    for var in (
        "SRP_WANDB_MODEL_REGISTRY",
        "SRP_WANDB_REGISTRY",
        "SRP_WANDB_MODEL_ALIAS",
        "SRP_WANDB_ALIAS",
        "SRP_WANDB_ENTITY",
    ):
        monkeypatch.delenv(var, raising=False)
    source = WandbRegistrySource(cache_dir=tmp_path)
    assert source._registry == "sleap-roots-models"  # default in effect, no env
    cards = source.list_cards()
    # Count-agnostic: the production card set grows over time (do not assert exactly N).
    assert cards and all(isinstance(c, ModelCard) for c in cards)
    # Cards are pinned to a concrete version (not the moving alias) with a checksum,
    # and carry the selection metadata the matcher needs.
    assert all(c.version and c.version != source._alias for c in cards)
    assert all(c.weights_checksum for c in cards)
    assert all(c.selectors and c.root_type for c in cards)
    assert all(s.species and s.mode for c in cards for s in c.selectors)
    ref = cards[0].to_model_ref("runtime")
    first = source.materialize(ref)
    assert Path(first).exists() and any(Path(first).iterdir())
    assert source.materialize(ref) == first  # cached, no re-download


class _UnreadableDigestArtifact(FakeArtifact):
    """An artifact whose metadata is valid but whose digest cannot be fetched."""

    @property
    def digest(self):
        """Raise as a transient registry error would."""
        raise ConnectionError("registry unreachable")

    @digest.setter
    def digest(self, value):
        """Ignore the base class's assignment."""


def test_collect_cards_non_validation_errors_name_the_real_cause():
    """Unreadable-for-other-reasons artifacts raise, naming the error, not a re-seed."""
    source = WandbRegistrySource(entity="ent", registry="reg", alias="production")
    artifacts = [_UnreadableDigestArtifact("reg/a", metadata=raw_card_meta())]
    with pytest.raises(NoReadableModelCardsError) as exc:
        source._collect_cards(artifacts)
    message = str(exc.value)
    assert "ConnectionError" in message and "registry unreachable" in message
    assert "re-seeded" not in message
    assert isinstance(exc.value.__cause__, ConnectionError)


def test_collect_cards_validation_failures_keep_the_reseed_hint():
    """All-flat artifacts still point the operator at the re-seed."""
    source = WandbRegistrySource(entity="ent", registry="reg", alias="production")
    with pytest.raises(NoReadableModelCardsError, match="re-seeded"):
        source._collect_cards([FakeArtifact("reg/flat", metadata=_flat_meta())])


def test_installed_version_falls_back_when_metadata_is_missing():
    """A missing distribution never masks the guard's own error."""
    from sleap_roots_predict.model_registry import _installed_version

    assert _installed_version("definitely-not-an-installed-dist-xyz") == "unknown"
    assert _installed_version("sleap-roots-contracts") != "unknown"


def test_skip_warning_prefix_matches_the_logged_warning(caplog):
    """The exported prefix is what the skip warning actually starts with."""
    from sleap_roots_predict.model_registry import SKIP_WARNING_PREFIX

    source = WandbRegistrySource(alias="production")
    with caplog.at_level(logging.WARNING, logger="sleap_roots_predict.model_registry"):
        source._collect_cards([_good_artifact(), _malformed_artifact()])
    (record,) = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert record.getMessage().startswith(SKIP_WARNING_PREFIX)
