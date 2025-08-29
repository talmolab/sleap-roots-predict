"""Tests for prediction module."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from sleap_roots_predict.predict import (
    make_predictor,
    predict_on_video,
    predict_on_h5,
    batch_predict,
)


class TestMakePredictor:
    """Test the make_predictor function."""

    @pytest.mark.gpu
    def test_make_predictor_with_valid_model(self, tmp_path):
        """Test creating a predictor with a valid model path."""
        # Create a mock model directory
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        # Create minimal required files for a SLEAP model
        (model_dir / "confmap_config.json").write_text('{"test": "config"}')
        (model_dir / "initial_config.json").write_text('{"test": "config"}')

        with patch(
            "sleap_roots_predict.predict.Predictor.from_model_paths"
        ) as mock_predictor:
            mock_instance = Mock()
            mock_predictor.return_value = mock_instance

            predictor = make_predictor(
                model_path=[model_dir], peak_threshold=0.3, batch_size=8, device="cpu"
            )

            assert predictor == mock_instance
            mock_predictor.assert_called_once_with(
                [model_dir.as_posix()], peak_threshold=0.3, batch_size=8, device="cpu"
            )

    def test_make_predictor_auto_device_cpu(self, tmp_path):
        """Test auto device selection defaults to CPU when no GPU available."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=False):
                with patch(
                    "sleap_roots_predict.predict.Predictor.from_model_paths"
                ) as mock_predictor:
                    make_predictor([model_dir], device="auto")

                    # Check that CPU was selected
                    mock_predictor.assert_called_once()
                    call_args = mock_predictor.call_args[1]
                    assert call_args["device"] == "cpu"

    @pytest.mark.gpu
    def test_make_predictor_auto_device_cuda(self, tmp_path):
        """Test auto device selection chooses CUDA when available."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        with patch("torch.cuda.is_available", return_value=True):
            with patch(
                "sleap_roots_predict.predict.Predictor.from_model_paths"
            ) as mock_predictor:
                make_predictor([model_dir], device="auto")

                # Check that CUDA was selected
                mock_predictor.assert_called_once()
                call_args = mock_predictor.call_args[1]
                assert call_args["device"] == "cuda"

    @pytest.mark.gpu
    def test_make_predictor_auto_device_mps(self, tmp_path):
        """Test auto device selection chooses MPS when available (macOS)."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=True):
                with patch(
                    "sleap_roots_predict.predict.Predictor.from_model_paths"
                ) as mock_predictor:
                    make_predictor([model_dir], device="auto")

                    # Check that MPS was selected
                    mock_predictor.assert_called_once()
                    call_args = mock_predictor.call_args[1]
                    assert call_args["device"] == "mps"

    def test_make_predictor_missing_model(self, tmp_path):
        """Test error handling for missing model directory."""
        non_existent = tmp_path / "non_existent"

        with pytest.raises(FileNotFoundError, match="Model dir not found"):
            make_predictor([non_existent])

    def test_make_predictor_multiple_models(self, tmp_path):
        """Test creating predictor with multiple model paths."""
        model1 = tmp_path / "model1"
        model2 = tmp_path / "model2"
        model1.mkdir()
        model2.mkdir()

        with patch(
            "sleap_roots_predict.predict.Predictor.from_model_paths"
        ) as mock_predictor:
            # Don't specify device, let it auto-detect
            make_predictor([model1, model2])

            # Check that the predictor was called with correct paths and parameters
            # Device will be auto-detected based on the actual hardware
            mock_predictor.assert_called_once()
            call_args = mock_predictor.call_args

            # Check the paths
            assert call_args[0][0] == [model1.as_posix(), model2.as_posix()]

            # Check other parameters
            assert call_args[1]["peak_threshold"] == 0.2
            assert call_args[1]["batch_size"] == 4

            # Device should be one of the valid options based on hardware
            assert call_args[1]["device"] in ["cpu", "cuda", "mps"]


class TestPredictOnVideo:
    """Test the predict_on_video function."""

    def test_predict_on_video_without_saving(self):
        """Test prediction on video without saving to file."""
        mock_video = Mock()
        mock_predictor = Mock()
        mock_labels = Mock()
        mock_predictor.predict.return_value = mock_labels

        with patch("sleap_roots_predict.predict.VideoReader") as mock_reader:
            result = predict_on_video(mock_predictor, mock_video)

            mock_reader.assert_called_once_with(mock_video, queue_maxsize=8)
            mock_predictor.predict.assert_called_once()
            assert result == mock_labels

    def test_predict_on_video_with_saving(self, tmp_path):
        """Test prediction on video with saving to file."""
        mock_video = Mock()
        mock_predictor = Mock()
        mock_labels = Mock()
        mock_predictor.predict.return_value = mock_labels

        save_path = tmp_path / "predictions.slp"

        with patch("sleap_roots_predict.predict.VideoReader"):
            with patch("sleap_roots_predict.predict.sio.save_file") as mock_save:
                result = predict_on_video(mock_predictor, mock_video, save_path)

                mock_save.assert_called_once_with(mock_labels, save_path.as_posix())
                assert result == save_path
                assert save_path.parent.exists()


class TestPredictOnH5:
    """Test the predict_on_h5 function."""

    def test_predict_on_h5_file_exists(self, tmp_path):
        """Test prediction on existing H5 file."""
        import h5py

        # Create a test H5 file
        h5_path = tmp_path / "test.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("vol", data=np.zeros((5, 10, 10, 1)))

        mock_predictor = Mock()
        mock_labels = Mock()
        mock_predictor.predict.return_value = mock_labels

        with patch(
            "sleap_roots_predict.predict.VideoReader.from_filename"
        ) as mock_reader:
            mock_reader_instance = Mock()
            mock_reader.return_value = mock_reader_instance

            result = predict_on_h5(mock_predictor, h5_path)

            mock_reader.assert_called_once_with(
                filename=h5_path.as_posix(), dataset="vol", queue_maxsize=8
            )
            mock_predictor.predict.assert_called_once_with(mock_reader_instance)
            assert result == mock_labels

    def test_predict_on_h5_file_not_found(self, tmp_path):
        """Test error handling for missing H5 file."""
        h5_path = tmp_path / "non_existent.h5"
        mock_predictor = Mock()

        with pytest.raises(FileNotFoundError, match="H5 file not found"):
            predict_on_h5(mock_predictor, h5_path)

    def test_predict_on_h5_custom_dataset(self, tmp_path):
        """Test prediction with custom dataset name."""
        import h5py

        h5_path = tmp_path / "test.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("custom_data", data=np.zeros((5, 10, 10, 1)))

        mock_predictor = Mock()
        mock_labels = Mock()
        mock_predictor.predict.return_value = mock_labels

        with patch(
            "sleap_roots_predict.predict.VideoReader.from_filename"
        ) as mock_reader:
            predict_on_h5(mock_predictor, h5_path, dataset="custom_data")

            mock_reader.assert_called_once_with(
                filename=h5_path.as_posix(), dataset="custom_data", queue_maxsize=8
            )


class TestBatchPredict:
    """Test the batch_predict function."""

    def test_batch_predict_success(self, tmp_path):
        """Test batch prediction on multiple H5 files."""
        import h5py

        # Create test H5 files
        h5_files = []
        for i in range(3):
            h5_path = tmp_path / f"test_{i}.h5"
            with h5py.File(h5_path, "w") as f:
                f.create_dataset("vol", data=np.zeros((5, 10, 10, 1)))
            h5_files.append(h5_path)

        output_dir = tmp_path / "output"
        mock_predictor = Mock()

        with patch("sleap_roots_predict.predict.predict_on_h5") as mock_predict:
            results = batch_predict(
                mock_predictor, h5_files, output_dir, file_suffix="_predicted"
            )

            assert len(results) == 3
            assert all(h5_path.as_posix() in results for h5_path in h5_files)
            assert mock_predict.call_count == 3
            assert output_dir.exists()

    def test_batch_predict_with_errors(self, tmp_path):
        """Test batch prediction handles errors gracefully."""
        import h5py

        # Create one valid and one invalid path
        valid_h5 = tmp_path / "valid.h5"
        with h5py.File(valid_h5, "w") as f:
            f.create_dataset("vol", data=np.zeros((5, 10, 10, 1)))

        invalid_h5 = tmp_path / "invalid.h5"

        output_dir = tmp_path / "output"
        mock_predictor = Mock()

        with patch("sleap_roots_predict.predict.predict_on_h5") as mock_predict:
            # Make the second call raise an error
            mock_predict.side_effect = [None, FileNotFoundError("Test error")]

            results = batch_predict(mock_predictor, [valid_h5, invalid_h5], output_dir)

            assert valid_h5.as_posix() in results
            assert "Error:" in results[invalid_h5.as_posix()]

    def test_batch_predict_custom_dataset(self, tmp_path):
        """Test batch prediction with custom dataset name."""
        import h5py

        h5_path = tmp_path / "test.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("custom", data=np.zeros((5, 10, 10, 1)))

        output_dir = tmp_path / "output"
        mock_predictor = Mock()

        with patch("sleap_roots_predict.predict.predict_on_h5") as mock_predict:
            batch_predict(mock_predictor, [h5_path], output_dir, dataset="custom")

            # Check that custom dataset name was passed
            call_args = mock_predict.call_args
            assert call_args[1]["dataset"] == "custom"


# Integration tests that require actual SLEAP installation
@pytest.mark.gpu
class TestIntegration:
    """Integration tests requiring SLEAP and GPU."""

    def test_full_prediction_pipeline(self, tmp_path):
        """Test the full prediction pipeline with real SLEAP models."""
        pytest.importorskip("sleap_nn")
        pytest.importorskip("sleap_io")

        # This test would require actual model files
        # For now, we'll skip if models aren't available
        model_path = Path("path/to/test/model")
        if not model_path.exists():
            pytest.skip("Test models not available")

        # Create test video
        from sleap_roots_predict.video_utils import make_video_from_images

        # Create dummy images
        img_dir = tmp_path / "images"
        img_dir.mkdir()

        for i in range(5):
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            from PIL import Image

            Image.fromarray(img).save(img_dir / f"img_{i:03d}.tif")

        # Create video
        image_files = sorted(img_dir.glob("*.tif"))
        video = make_video_from_images(image_files)

        # Create predictor
        predictor = make_predictor([model_path], device="cuda")

        # Run prediction
        save_path = tmp_path / "predictions.slp"
        result = predict_on_video(predictor, video, save_path)

        assert save_path.exists()
        assert result == save_path
