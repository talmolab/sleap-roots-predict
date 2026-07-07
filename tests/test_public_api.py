"""The package's public surface imports offline, with no wandb credentials."""

import importlib


def test_public_surface_importable_without_credentials(monkeypatch):
    """Importing the package needs no network and no WANDB_API_KEY."""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    pkg = importlib.import_module("sleap_roots_predict")
    for name in (
        "WarmModelWorker",
        "choose_models",
        "resolve_params",
        "LocalCardSource",
        "WandbRegistrySource",
        "ModelCardSource",
        "make_predictor",
        "predict_on_video",
        "process_timelapse_experiment",
        "PredictionArtifact",
        "PredictionManifest",
        "ScanRequest",
        "write_prediction_outputs",
        "predict_and_write_batch",
        "run_batch",
        "discover_scans",
        "BatchResult",
        "ScanResult",
    ):
        assert hasattr(pkg, name), name
        assert name in pkg.__all__, f"{name} missing from __all__"
