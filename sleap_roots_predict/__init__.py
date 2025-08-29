"""sleap-roots-predict: SLEAP-NN prediction and timelapse processing for root systems.

This package provides:
- SLEAP neural network model integration for pose estimation
- Timelapse experiment processing with metadata extraction
- Batch prediction on videos and H5 files
- JSON export for experiment results
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
