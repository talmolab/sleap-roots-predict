"""Tests for the timelapse orchestrator after prediction was deferred.

Prediction within ``process_timelapse_experiment`` is intentionally disabled
(see the rebuild-inference-core change). These tests guard that the module
still imports under sleap-nn 0.3.0 and that supplying ``model_paths`` is a
logged no-op rather than an error.
"""

import logging

from sleap_roots_predict.plates_timelapse_experiment import (
    process_timelapse_experiment,
)


def test_orchestrator_has_no_prediction_symbols():
    """The removed predict helpers are no longer imported into the module."""
    import sleap_roots_predict.plates_timelapse_experiment as m

    assert not hasattr(m, "predict_on_h5")
    assert not hasattr(m, "batch_predict")
    assert not hasattr(m, "make_predictor")
    assert callable(process_timelapse_experiment)


def test_prediction_deferred_warns_and_skips(
    image_directory_with_tiffs, metadata_csv, caplog
):
    """Passing model_paths logs a deferral warning and runs no prediction."""
    base_dir = image_directory_with_tiffs.parent

    with caplog.at_level(logging.WARNING):
        results = process_timelapse_experiment(
            base_dir=base_dir,
            metadata_csv=metadata_csv,
            experiment_name="exp1",
            model_paths=["some/model/dir"],
            dry_run=True,
        )

    assert isinstance(results, dict)
    messages = " ".join(r.getMessage().lower() for r in caplog.records)
    assert "deferred" in messages or "temporarily disabled" in messages
