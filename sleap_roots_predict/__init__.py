"""sleap-roots-predict: SLEAP-NN prediction and timelapse processing for root systems.

This package provides:
- sleap-nn inference on ``sleap_io.Video`` objects (``make_predictor`` builds a
  reusable predictor; ``predict_on_video`` runs inference and optionally saves a
  ``.slp``)
- Model management: resolve Bloom scan metadata to params (``resolve_params``),
  choose models from those params (``choose_models``), fetch them from the wandb
  registry or a local dir (``ModelCardSource`` / ``WandbRegistrySource`` /
  ``LocalCardSource``), and keep sleap-nn predictors resident across scans
  (``WarmModelWorker``)
- Output contract: write the per-scan artifacts the downstream traits stage reads
  (named per-root ``.slp`` + a combined ``{scan}.predictions.json`` manifest) via
  ``write_prediction_outputs`` / ``predict_and_write_batch`` (see the
  ``prediction-output`` spec)
- Timelapse experiment processing with metadata extraction (video/H5/metadata;
  prediction within this flow is currently deferred)
"""

__version__ = "0.0.1a0"

# High-level API
from sleap_roots_predict.plates_timelapse_experiment import (  # noqa: F401
    process_timelapse_experiment,
)

from sleap_roots_predict.predict import (  # noqa: F401
    make_predictor,
    predict_on_video,
)

from sleap_roots_predict.model_selection import choose_models  # noqa: F401

from sleap_roots_contracts import resolve_params  # noqa: F401

from sleap_roots_predict.model_registry import (  # noqa: F401
    LocalCardSource,
    ModelCardSource,
    WandbRegistrySource,
)

from sleap_roots_predict.warm_worker import WarmModelWorker  # noqa: F401

from sleap_roots_predict.output_contract import (  # noqa: F401
    PredictionArtifact,
    PredictionManifest,
    ScanRequest,
    predict_and_write_batch,
    write_prediction_outputs,
)

from sleap_roots_predict.batch import (  # noqa: F401
    BatchResult,
    ScanResult,
    discover_scans,
    run_batch,
)

__all__ = [
    "process_timelapse_experiment",
    "make_predictor",
    "predict_on_video",
    "choose_models",
    "resolve_params",
    "ModelCardSource",
    "LocalCardSource",
    "WandbRegistrySource",
    "WarmModelWorker",
    "PredictionArtifact",
    "PredictionManifest",
    "ScanRequest",
    "write_prediction_outputs",
    "predict_and_write_batch",
    "run_batch",
    "discover_scans",
    "BatchResult",
    "ScanResult",
]
