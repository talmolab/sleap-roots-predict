"""Tracking functionality for SLEAP predictions.

This module provides functions to add tracking to SLEAP predictions,
associating instances across frames to maintain consistent identities.
"""

from typing import Optional, Dict, Any
import logging
import sleap_io as sio
from sleap_nn.tracking.tracker import Tracker, run_tracker
from sleap_nn.inference.predictors import (
    TopDownPredictor,
    BottomUpPredictor,
    SingleInstancePredictor,
)

logger = logging.getLogger(__name__)


def add_tracking_to_labels(
    labels: sio.Labels,
    window_size: int = 5,
    tracker_method: str = "hungarian",
    similarity_method: str = "centroid",
    max_tracks: Optional[int] = None,
    max_tracking: bool = True,
    **kwargs,
) -> sio.Labels:
    """Add tracking to predicted labels using sleap-nn tracking.

    Args:
        labels: Predicted labels without tracking
        window_size: Number of frames to consider for matching
        tracker_method: Matching method - "hungarian" or "greedy"
        similarity_method: How to compute similarity:
            - "centroid": Use centroid distance
            - "iou": Use intersection over union of bounding boxes
            - "instance": Use full instance similarity
        max_tracks: Maximum number of tracks to maintain
        max_tracking: If True, track at most max_tracks instances
        **kwargs: Additional tracking parameters

    Returns:
        Labels with track IDs assigned to instances

    Raises:
        ValueError: If invalid tracking parameters are provided
    """
    # Validate parameters
    if tracker_method not in ["hungarian", "greedy"]:
        raise ValueError(
            f"Invalid tracker_method: {tracker_method}. "
            "Must be 'hungarian' or 'greedy'"
        )

    # Map our similarity method names to sleap-nn parameters
    similarity_config = {
        "centroid": {
            "candidates_method": "fixed_window",
            "features": "centroids",
            "scoring_method": "euclidean_dist",
        },
        "iou": {
            "candidates_method": "fixed_window",
            "features": "bboxes",
            "scoring_method": "iou",
        },
        "instance": {
            "candidates_method": "fixed_window",
            "features": "keypoints",
            "scoring_method": "oks",  # Object Keypoint Similarity
        },
    }

    if similarity_method not in similarity_config:
        raise ValueError(
            f"Invalid similarity_method: {similarity_method}. "
            f"Must be one of: {list(similarity_config.keys())}"
        )

    # Get similarity configuration
    sim_config = similarity_config[similarity_method]

    # max_tracking is not used by run_tracker, just log it
    if max_tracking:
        logger.debug(f"max_tracking={max_tracking} (advisory parameter)")

    logger.info(
        f"Running tracking with method={tracker_method}, "
        f"similarity={similarity_method}, window={window_size}"
    )

    # Use sleap-nn's run_tracker function with unpacked parameters
    tracked_labels = run_tracker(
        labels,
        window_size=window_size,
        track_matching_method=tracker_method,
        max_tracks=max_tracks,
        **sim_config,
        **kwargs,  # Allow additional parameters
    )
    
    # Log tracking results
    if hasattr(tracked_labels, "tracks") and tracked_labels.tracks:
        num_tracks = len(tracked_labels.tracks)
        logger.info(f"Tracking complete: {num_tracks} tracks created")
    else:
        logger.info("Tracking complete")

    return tracked_labels


def validate_predictor_for_tracking(predictor, require_topdown: bool = True) -> bool:
    """Validate that predictor is suitable for multi-instance tracking.

    Args:
        predictor: The predictor instance to validate
        require_topdown: If True, require TopDownPredictor for tracking

    Returns:
        True if predictor is valid for tracking

    Raises:
        ValueError: If predictor is not suitable for tracking
    """
    predictor_type = type(predictor).__name__

    if isinstance(predictor, SingleInstancePredictor):
        raise ValueError(
            "SingleInstancePredictor cannot be used for multi-instance tracking. "
            "Please use a TopDownPredictor or BottomUpPredictor model."
        )

    if require_topdown and not isinstance(predictor, TopDownPredictor):
        logger.warning(
            f"TopDownPredictor is recommended for multi-plant tracking, "
            f"but got {predictor_type}. Tracking may not work optimally."
        )

    logger.info(f"Using {predictor_type} for tracking")
    return True
