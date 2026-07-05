"""Warm model worker.

A long-lived object that keeps sleap-nn ``Predictor``s resident in memory across
scans. It composes the pure selection matcher (Layer 1 of this capability) and a
``ModelCardSource`` with the built inference core (``make_predictor`` /
``predict_on_video``):

    resolve(params)        -> ModelRefs, no weights loaded
    get_predictors(params) -> Predictors, fetched once + loaded once + reused
    predict(params, video) -> sio.Labels per root type

Predictors are cached by model identity ``(registry_id, version)``, so different
scan types that resolve to the same model version reuse one resident predictor.
If any resolved root type cannot be materialized or loaded, the worker fails loud
(raises, naming the root type + model id) rather than returning partial results.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import sleap_io as sio
from sleap_nn.inference import Predictor
from sleap_roots_contracts import ModelRef, ResolvedParams, RootType

from sleap_roots_predict.model_registry import ModelCardSource
from sleap_roots_predict.model_selection import choose_models
from sleap_roots_predict.predict import (
    _resolve_device,
    make_predictor,
    predict_on_video,
)


class WarmModelWorker:
    """Resolves, fetches, and holds sleap-nn ``Predictor``s resident across scans."""

    def __init__(
        self,
        source: ModelCardSource,
        *,
        device: str = "auto",
        peak_threshold: float = 0.2,
        batch_size: int = 4,
    ) -> None:
        """Create a warm worker over a model-card source.

        Args:
            source: The card source (catalog + ``materialize``). Only
                ``WandbRegistrySource`` touches the network; ``LocalCardSource`` is
                fully offline.
            device: Inference device (``"auto"``, ``"cpu"``, ``"cuda"``, ``"mps"``).
            peak_threshold: Confidence threshold for peak detection (output-defining).
            batch_size: Samples per inference batch (diagnostic, not output-defining).
        """
        self._source = source
        # Resolve "auto" to a concrete device once, at construction, so the value
        # recorded by inference_config() is exactly what make_predictor builds with.
        self._device = _resolve_device(device)
        self._peak_threshold = peak_threshold
        self._batch_size = batch_size
        self._cards = None
        self._predictors: Dict[Tuple[str, str], Predictor] = {}

    def resolve(
        self,
        params: ResolvedParams,
        overrides: Optional[Dict[RootType, ModelRef]] = None,
    ) -> Dict[RootType, ModelRef]:
        """Select a ``ModelRef`` per root type without loading any weights.

        Args:
            params: Resolved scan params (species/mode/age).
            overrides: Optional explicit ``ModelRef`` per root type.

        Returns:
            The selected refs; root types with no match (and no override) are absent.
        """
        if self._cards is None:
            self._cards = self._source.list_cards()
        return choose_models(params, self._cards, overrides)

    def get_predictors(
        self,
        params: ResolvedParams,
        overrides: Optional[Dict[RootType, ModelRef]] = None,
    ) -> Dict[RootType, Predictor]:
        """Return a resident ``Predictor`` per resolved root type.

        Each model is materialized and loaded at most once and cached by
        ``(registry_id, version)``; repeat calls reuse the resident predictor.

        Args:
            params: Resolved scan params.
            overrides: Optional explicit ``ModelRef`` per root type.

        Returns:
            A mapping of root type to a live ``Predictor``.

        Raises:
            RuntimeError: If any resolved root type cannot be materialized or
                loaded (fail-loud; no partial results are returned).
        """
        refs = self.resolve(params, overrides)
        predictors: Dict[RootType, Predictor] = {}
        for root_type, ref in refs.items():
            predictors[root_type] = self._predictor_for(root_type, ref)
        return predictors

    def _predictor_for(self, root_type: RootType, ref: ModelRef) -> Predictor:
        """Return (building + caching if needed) the predictor for one ref."""
        key = (ref.registry_id, ref.version)
        if key not in self._predictors:
            try:
                model_dir = self._source.materialize(ref)
                self._predictors[key] = make_predictor(
                    [model_dir],
                    peak_threshold=self._peak_threshold,
                    batch_size=self._batch_size,
                    device=self._device,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to prepare model for root type {root_type!r} "
                    f"({ref.registry_id}:{ref.version}): {e}"
                ) from e
        return self._predictors[key]

    def predict(
        self,
        params: ResolvedParams,
        video: "sio.Video",
        save_dir: Optional[Union[str, Path]] = None,
        overrides: Optional[Dict[RootType, ModelRef]] = None,
    ) -> Dict[RootType, Union[sio.Labels, Path]]:
        """Run inference for each resolved root type over one video.

        Args:
            params: Resolved scan params.
            video: A ``sleap_io.Video`` to predict on.
            save_dir: If given, writes one raw ``<root_type>.slp`` per root type
                here. No ``predictions.csv`` manifest and no
                ``{scan}.model{id}.root{type}.slp`` naming is applied — those are
                deferred to the output-contract slice.
            overrides: Optional explicit ``ModelRef`` per root type.

        Returns:
            A mapping of root type to ``sio.Labels`` (or the saved ``.slp`` ``Path``
            when ``save_dir`` is given).
        """
        predictors = self.get_predictors(params, overrides)
        results: Dict[RootType, Union[sio.Labels, Path]] = {}
        for root_type, predictor in predictors.items():
            save_path = None
            if save_dir is not None:
                save_path = Path(save_dir) / f"{root_type}.slp"
            results[root_type] = predict_on_video(predictor, video, save_path)
        return results

    def inference_config(self) -> Dict[str, Any]:
        """Return the full effective inference config used to build predictors.

        The resolved (concrete) device is reported, not ``"auto"``. This is the
        audit record; downstream layers fold the output-defining subset (see
        :meth:`output_params`) into the reproducibility hash and keep the
        hardware/throughput knobs as diagnostics.

        Returns:
            A mapping with ``device`` (resolved), ``peak_threshold``, and
            ``batch_size``.
        """
        return {
            "device": self._device,
            "peak_threshold": self._peak_threshold,
            "batch_size": self._batch_size,
        }

    def output_params(self) -> Dict[str, Any]:
        """Return the output-defining subset of the inference config.

        Only knobs that change the predictions belong here (``peak_threshold``);
        hardware/throughput knobs (``device``, ``batch_size``) are excluded so a
        rerun on different hardware still dedups on the reproducibility hash.
        """
        return {"peak_threshold": self._peak_threshold}
