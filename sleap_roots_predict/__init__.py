"""sleap-roots-predict: SLEAP-NN prediction and timelapse processing for root systems.

This package provides:
- sleap-nn inference on ``sleap_io.Video`` objects (``make_predictor`` builds a
  reusable predictor; ``predict_on_video`` runs inference and optionally saves a
  ``.slp``)
- Model management: choose models from Bloom scan metadata (``choose_models``),
  fetch them from the wandb registry or a local dir (``ModelCardSource`` /
  ``WandbRegistrySource`` / ``LocalCardSource``), and keep sleap-nn predictors
  resident across scans (``WarmModelWorker``)
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

from sleap_roots_predict.model_registry import (  # noqa: F401
    LocalCardSource,
    ModelCardSource,
    WandbRegistrySource,
)

from sleap_roots_predict.warm_worker import WarmModelWorker  # noqa: F401

__all__ = [
    "process_timelapse_experiment",
    "make_predictor",
    "predict_on_video",
    "choose_models",
    "ModelCardSource",
    "LocalCardSource",
    "WandbRegistrySource",
    "WarmModelWorker",
]
