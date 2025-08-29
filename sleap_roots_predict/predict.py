import numpy as np
import sleap_io as sio

from typing import Union, Optional, List, Dict
from pathlib import Path
from sleap_nn.inference.predictors import (
    Predictor,
)
from sleap_nn.data.providers import VideoReader

# https://github.com/talmolab/sleap-nn/blob/78b90e1b964ecc10639d9560c79872c2f9f1ec67/sleap_nn/inference/predictors.py#L504


def make_predictor(
    model_path: List[Union[str, Path]],
    peak_threshold: float = 0.2,
    batch_size: int = 4,
    device: str = "auto",
) -> Predictor:
    """Create a Predictor from a model dir(s).

    Args:
        model_path: List of Path(s) to the trained model directory.
        peak_threshold: Confidence threshold for peak detection.
        batch_size: Number of samples per batch for inference.
        device: Device for inference ("auto", "cpu", "cuda", or "mps").
               "auto" will select the best available device.

    Returns:
        An instance of Predictor.

    Raises:
        FileNotFoundError: If the model file is not found.
        TypeError: If the loaded predictor is not a Predictor.
    """
    model_paths = [Path(p) for p in model_path]
    for model_path in model_paths:
        if not model_path.exists():
            raise FileNotFoundError(f"Model dir not found: {model_path}")

    # Determine the best device if auto
    if device == "auto":
        import torch

        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Load the predictor with specified parameters
    predictor = Predictor.from_model_paths(
        [model_path.as_posix() for model_path in model_paths],
        peak_threshold=peak_threshold,
        batch_size=batch_size,
        device=device,
    )

    return predictor


def predict_on_video(
    predictor: Predictor,
    video: "sio.Video", 
    save_path: Optional[Union[str, Path]] = None,
) -> Union[Path, "sio.Labels"]:
    """Run prediction on a sleap_io.Video object using a Predictor.

    Args:
        predictor: The Predictor instance to use for inference.
        video: A sleap_io.Video object.
        save_path: Optional path to save predictions as .slp file.
                  If None, returns the Labels object without saving.

    Returns:
        Either the path to the saved .slp file or the Labels object with predictions.
    """
    # Create VideoReader from Video object
    video_reader = VideoReader(video, queue_maxsize=8)
    
    # Run inference
    labels = predictor.predict(video_reader)

    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        sio.save_file(labels, str(save_path))
        return save_path

    return labels


def predict_on_images(
    predictor: Predictor,
    images: Union[np.ndarray, List[Union[str, Path]]],
    greyscale: bool = False,
    save_path: Optional[Union[str, Path]] = None,
) -> Union[Path, "sio.Labels"]:
    """Run prediction on images (either array or file paths).

    Args:
        predictor: The Predictor instance to use for inference.
        images: Either a numpy array (frames, height, width, channels) or 
                list of image file paths.
        greyscale: Whether to convert images to greyscale (only used if images are paths).
        save_path: Optional path to save predictions as .slp file.

    Returns:
        Either the path to the saved .slp file or the Labels object with predictions.
    """
    # Convert to Video object
    if isinstance(images, np.ndarray):
        # Create Video from numpy array
        video = sio.Video.from_numpy(images)
    else:
        # Assume it's a list of paths
        from .video_utils import make_video_from_images
        video = make_video_from_images(images, greyscale=greyscale)
    
    # Run prediction on the video
    return predict_on_video(predictor, video, save_path=save_path)


def predict_on_h5(
    predictor: Predictor,
    h5: Union[str, Path],
    dataset: str = "vol",
    save_path: Optional[Union[str, Path]] = None,
) -> Union[Path, "sio.Labels"]:
    """Run prediction on an H5 file using a Predictor.
    
    This function is kept for backward compatibility.

    Args:
        predictor: The Predictor instance to use for inference.
        h5: H5 file containing the video data.
        dataset: Name of the dataset within the H5 file (default: "vol").
        save_path: Optional path to save predictions as .slp file.
                  If None, returns the Labels object without saving.

    Returns:
        Either the path to the saved .slp file or the Labels object with predictions.

    Raises:
        FileNotFoundError: If the H5 file is not found.
    """
    h5_path = Path(h5)
    if not h5_path.exists():
        raise FileNotFoundError(f"H5 file not found: {h5_path}")
    
    # Create VideoReader from H5 file
    video_reader = VideoReader.from_filename(
        filename=h5_path.as_posix(),
        dataset=dataset,
        queue_maxsize=8,
    )

    # Run inference
    labels = predictor.predict(video_reader)

    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        sio.save_file(labels, str(save_path))
        return save_path

    return labels


def batch_predict(
    predictor: Predictor,
    input_paths: List[Union[str, Path]],
    output_dir: Union[str, Path],
    dataset: str = "vol",
    file_suffix: str = "",
) -> Dict[str, Union[Path, str]]:
    """Run predictions on multiple H5 files.

    Args:
        predictor: The Predictor instance to use for inference.
        input_paths: List of paths to H5 files.
        output_dir: Directory to save prediction files.
        dataset: Name of the dataset within the H5 files (default: "vol").
        file_suffix: Optional suffix to add to output filenames.

    Returns:
        Dictionary mapping input paths to output paths or error messages.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for input_path in input_paths:
        input_path = Path(input_path)

        try:
            # Generate output filename
            base_name = input_path.stem
            if file_suffix:
                output_name = f"{base_name}{file_suffix}.slp"
            else:
                output_name = f"{base_name}.slp"
            output_path = output_dir / output_name

            # Run prediction
            predict_on_h5(predictor, input_path, dataset=dataset, save_path=output_path)
            results[str(input_path)] = str(output_path)

        except Exception as e:
            results[str(input_path)] = f"Error: {str(e)}"

    return results
