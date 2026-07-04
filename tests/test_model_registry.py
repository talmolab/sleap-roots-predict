"""Tests for the model-card source layer.

The offline ``LocalCardSource`` tests run with no network and no mocks: cards
point at the vendored model directories and ``materialize`` returns a real dir
that ``make_predictor`` can load. Gated ``WandbRegistrySource`` tests are added in
a later task (``@pytest.mark.wandb``).
"""

import os
from pathlib import Path

import pytest
from sleap_nn.inference import Predictor
from sleap_roots_contracts import ModelCard

from sleap_roots_predict.model_registry import (
    LocalCardSource,
    ModelCardSource,
    WandbRegistrySource,
)
from sleap_roots_predict.predict import make_predictor

WANDB_API_KEY = os.environ.get("WANDB_API_KEY")


def _card(root_type, registry_id, version="v1"):
    return ModelCard(
        species="rice",
        mode="cylinder",
        age_min=2,
        age_max=5,
        root_type=root_type,
        registry_id=registry_id,
        version=version,
    )


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


def test_wandb_source_missing_key_raises_before_network(monkeypatch):
    """With no WANDB_API_KEY, list_cards raises a clear error (no network call)."""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    source = WandbRegistrySource(entity="an-entity", registry="a-registry")
    with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
        source.list_cards()


@pytest.mark.wandb
@pytest.mark.skipif(
    not WANDB_API_KEY,
    reason="requires WANDB_API_KEY + a populated production registry",
)
def test_wandb_source_lists_and_materializes(tmp_path):
    """With creds, list_cards yields pinned cards and materialize caches the dir."""
    source = WandbRegistrySource(cache_dir=tmp_path)
    cards = source.list_cards()
    assert cards and all(isinstance(c, ModelCard) for c in cards)
    # Cards are pinned to a concrete version (not a moving alias).
    assert all(c.version for c in cards)
    ref = cards[0].to_model_ref("runtime")
    first = source.materialize(ref)
    assert Path(first).exists()
    assert source.materialize(ref) == first  # cached, no re-download
