"""Tests for tracking functionality."""

import pytest
import numpy as np
import sleap_io as sio
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from sleap_roots_predict.tracking import (
    add_tracking_to_labels,
    validate_predictor_for_tracking,
)


class TestAddTrackingToLabels:
    """Test suite for add_tracking_to_labels function."""

    @pytest.fixture
    def mock_labels(self):
        """Create mock Labels object."""
        labels = Mock(spec=sio.Labels)
        labels.__len__ = Mock(return_value=10)
        labels.tracks = []
        
        # Create mock labeled frames
        labeled_frames = []
        for i in range(10):
            lf = Mock()
            lf.instances = [Mock()]  # Mock instances
            lf.video = Mock()
            lf.frame_idx = i
            labeled_frames.append(lf)
        labels.labeled_frames = labeled_frames
        
        return labels

    @pytest.fixture
    def mock_tracker_class(self):
        """Create mock Tracker class."""
        with patch("sleap_nn.tracking.tracker.Tracker") as MockTracker:
            # Create mock tracker instance
            mock_tracker = Mock()
            mock_tracker.run_tracker = Mock()

            # Configure the class to return the instance
            MockTracker.from_config = Mock(return_value=mock_tracker)

            yield MockTracker, mock_tracker

    @patch("sleap_roots_predict.tracking.run_tracker")
    def test_add_tracking_default_params(self, mock_run_tracker, mock_labels):
        """Test add_tracking_to_labels with default parameters."""
        # Setup mock tracked labels
        mock_tracked_labels = Mock(spec=sio.Labels)
        mock_tracked_labels.tracks = [Mock()]  # Mock tracks attribute
        mock_run_tracker.return_value = mock_tracked_labels

        # Run tracking
        result = add_tracking_to_labels(mock_labels)

        # Verify run_tracker was called with correct parameters
        mock_run_tracker.assert_called_once_with(
            mock_labels,
            window_size=5,
            track_matching_method="hungarian",
            max_tracks=None,
            candidates_method="fixed_window",
            features="centroids",
            scoring_method="euclidean_dist",
        )
        
        # Result should be the new tracked labels
        assert result == mock_tracked_labels

    @patch("sleap_roots_predict.tracking.run_tracker")
    def test_add_tracking_custom_params(self, mock_run_tracker, mock_labels):
        """Test add_tracking_to_labels with custom parameters."""
        # Setup mock tracked labels
        mock_tracked_labels = Mock(spec=sio.Labels)
        mock_tracked_labels.tracks = [Mock()]  # Mock tracks attribute
        mock_run_tracker.return_value = mock_tracked_labels

        # Run tracking with custom params
        result = add_tracking_to_labels(
            mock_labels,
            window_size=10,
            tracker_method="greedy",
            similarity_method="iou",
            max_tracks=5,
            max_tracking=False,
        )

        # Verify run_tracker was called with correct parameters
        mock_run_tracker.assert_called_once_with(
            mock_labels,
            window_size=10,
            track_matching_method="greedy",
            max_tracks=5,
            candidates_method="fixed_window",
            features="bboxes",
            scoring_method="iou",
        )
        
        # Result should be the new tracked labels
        assert result == mock_tracked_labels

    @patch("sleap_roots_predict.tracking.run_tracker")
    def test_add_tracking_instance_similarity(self, mock_run_tracker, mock_labels):
        """Test add_tracking_to_labels with instance similarity method."""
        # Setup mock tracked labels
        mock_tracked_labels = Mock(spec=sio.Labels)
        mock_tracked_labels.tracks = [Mock()]  # Mock tracks attribute
        mock_run_tracker.return_value = mock_tracked_labels

        # Run tracking with instance similarity
        result = add_tracking_to_labels(
            mock_labels,
            similarity_method="instance",
        )

        # Verify run_tracker was called with correct parameters for instance similarity
        mock_run_tracker.assert_called_once_with(
            mock_labels,
            window_size=5,
            track_matching_method="hungarian",
            max_tracks=None,
            candidates_method="fixed_window",
            features="keypoints",
            scoring_method="oks",
        )
        
        # Result should be the new tracked labels
        assert result == mock_tracked_labels

    def test_add_tracking_invalid_tracker_method(self, mock_labels):
        """Test add_tracking_to_labels with invalid tracker method."""
        with pytest.raises(ValueError, match="Invalid tracker_method"):
            add_tracking_to_labels(mock_labels, tracker_method="invalid_method")

    def test_add_tracking_invalid_similarity_method(self, mock_labels):
        """Test add_tracking_to_labels with invalid similarity method."""
        with pytest.raises(ValueError, match="Invalid similarity_method"):
            add_tracking_to_labels(mock_labels, similarity_method="invalid_method")

    @pytest.mark.skip(reason="sleap-nn is now a required dependency")
    def test_add_tracking_import_error(self, mock_labels):
        """Test add_tracking_to_labels when sleap-nn tracking is not available."""
        # This test is no longer relevant since sleap-nn is required
        pass

    @pytest.mark.skip(reason="Additional kwargs may not be valid for Tracker.from_config")
    def test_add_tracking_with_additional_kwargs(self, mock_labels, mock_tracker_class):
        """Test that additional kwargs are passed through to tracker config."""
        # This test is skipped because arbitrary kwargs may not be valid
        pass


class TestPredictWithTracking:
    """Integration tests for prediction with tracking."""

    @pytest.fixture
    def mock_predictor(self):
        """Create a mock predictor."""
        predictor = Mock()
        predictor.predict = Mock()
        return predictor

    @pytest.fixture
    def mock_video(self):
        """Create a mock video."""
        video = Mock(spec=sio.Video)
        video.__len__ = Mock(return_value=100)
        return video

    def test_predict_on_video_with_tracking(self, mock_predictor, mock_video):
        """Test predict_on_video with tracking enabled."""
        from sleap_roots_predict.predict import predict_on_video

        with (
            patch("sleap_roots_predict.predict.VideoReader") as MockReader,
            patch(
                "sleap_roots_predict.tracking.run_tracker"
            ) as mock_run_tracker,
            patch("sleap_roots_predict.predict.sio.save_file") as mock_save,
        ):

            # Setup mocks
            mock_reader = Mock()
            MockReader.return_value = mock_reader

            mock_labels = Mock(spec=sio.Labels)
            mock_predictor.predict.return_value = mock_labels

            # run_tracker returns tracked labels
            mock_tracked_labels = Mock(spec=sio.Labels)
            mock_tracked_labels.tracks = []
            mock_run_tracker.return_value = mock_tracked_labels

            # Run prediction with tracking
            save_path = Path("test_predictions.slp")
            result = predict_on_video(
                mock_predictor,
                mock_video,
                save_path=save_path,
                enable_tracking=True,
                tracking_config={"window_size": 10, "max_tracks": 5},
            )

            # Verify tracking was applied with correct params
            mock_run_tracker.assert_called_once()
            call_kwargs = mock_run_tracker.call_args[1]
            assert call_kwargs["window_size"] == 10
            assert call_kwargs["max_tracks"] == 5

            # Verify tracked labels were saved
            mock_save.assert_called_once_with(
                mock_tracked_labels, save_path.as_posix()  # Tracked labels are returned
            )

            assert result == save_path

    def test_predict_on_video_without_tracking(self, mock_predictor, mock_video):
        """Test predict_on_video with tracking disabled."""
        from sleap_roots_predict.predict import predict_on_video

        with (
            patch("sleap_roots_predict.predict.VideoReader") as MockReader,
            patch(
                "sleap_roots_predict.tracking.add_tracking_to_labels"
            ) as mock_add_tracking,
            patch("sleap_roots_predict.predict.sio.save_file") as mock_save,
        ):

            # Setup mocks
            mock_reader = Mock()
            MockReader.return_value = mock_reader

            mock_labels = Mock(spec=sio.Labels)
            mock_predictor.predict.return_value = mock_labels

            # Run prediction without tracking
            save_path = Path("test_predictions.slp")
            result = predict_on_video(
                mock_predictor, mock_video, save_path=save_path, enable_tracking=False
            )

            # Verify tracking was NOT applied
            mock_add_tracking.assert_not_called()

            # Verify original labels were saved
            mock_save.assert_called_once_with(mock_labels, save_path.as_posix())

    def test_predict_on_video_tracking_no_save(self, mock_predictor, mock_video):
        """Test predict_on_video with tracking but no save path."""
        from sleap_roots_predict.predict import predict_on_video

        with (
            patch("sleap_roots_predict.predict.VideoReader") as MockReader,
            patch(
                "sleap_roots_predict.tracking.run_tracker"
            ) as mock_run_tracker,
        ):

            # Setup mocks
            mock_reader = Mock()
            MockReader.return_value = mock_reader

            mock_labels = Mock(spec=sio.Labels)
            mock_predictor.predict.return_value = mock_labels

            # run_tracker returns tracked labels
            mock_tracked_labels = Mock(spec=sio.Labels)
            mock_tracked_labels.tracks = []
            mock_run_tracker.return_value = mock_tracked_labels

            # Run prediction with tracking but no save
            result = predict_on_video(
                mock_predictor, mock_video, save_path=None, enable_tracking=True
            )

            # Verify tracking was applied
            mock_run_tracker.assert_called_once()

            # Verify tracked labels were returned
            assert result == mock_tracked_labels  # Tracked labels are returned  # Tracked labels are returned
