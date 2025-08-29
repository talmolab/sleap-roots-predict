"""Utilities for processing directories of images into H5 files with metadata."""

import logging
import os
import io
import re
import h5py
import imageio.v3 as iio
import numpy as np
import pandas as pd
import sleap_io as sio

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Literal


# Initialize logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def natural_sort(items: List[Union[str, Path]]) -> List[str]:
    """Sort a list of strings in a way that considers numerical values.

    For example, natural_sort(["img2.png", "img10.png", "img1.png"])
    will return ["img1.png", "img2.png", "img10.png"].

    Args:
        items: List of strings or Path objects to sort.

    Returns:
        List of sorted strings.
    """

    def convert(text: str) -> Union[int, str]:
        """Convert text to int if digit, otherwise lowercase string."""
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key: str) -> List[Union[int, str]]:
        """Split string into list of ints and strings for sorting."""
        if key is None:
            return [""]  # Sort None values first
        return [convert(c) for c in re.split(r"([0-9]+)", key)]

    # Convert Path objects to strings, handle None
    string_items = []
    for item in items:
        if item is None:
            string_items.append(None)
        elif isinstance(item, Path):
            string_items.append(item.as_posix())
        else:
            string_items.append(item)

    return sorted(string_items, key=alphanum_key)


def convert_to_greyscale(image: np.ndarray, method: str = "weights") -> np.ndarray:
    """Convert an RGB image to greyscale.

    Args:
        image: RGB image array with shape (..., 3).
        method: Conversion method - "weights" (ITU-R BT.601) or "average".

    Returns:
        Greyscale image array with shape (..., 1).
    """
    if image.ndim < 3 or image.shape[-1] != 3:
        raise ValueError(
            f"Expected RGB image with shape (..., 3), got shape {image.shape}"
        )

    if method == "weights":
        # Standard ITU-R BT.601 luma coefficients (same as PIL's 'L' mode)
        weights = np.array([0.2989, 0.5870, 0.1140])
        grey = np.dot(image[..., :3], weights)
    elif method == "average":
        # Simple averaging of RGB channels
        grey = np.mean(image[..., :3], axis=-1)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'weights' or 'average'.")

    grey = grey.astype(np.uint8)
    # Add channel dimension to maintain consistency
    return np.expand_dims(grey, axis=-1)


def load_images(
    image_files: List[Path], greyscale: bool = False
) -> Tuple[np.ndarray, List[str]]:
    """Load images from file paths into a numpy array.

    Args:
        image_files: List of paths to image files.
        greyscale: Whether to convert images to greyscale.

    Returns:
        Tuple of (stacked images array, list of filenames).
    """
    if not image_files:
        raise ValueError("No image files provided")

    images = []
    filenames = []

    for img_file in image_files:
        logger.debug(f"Reading {img_file}")

        if greyscale:
            # Read image and convert to greyscale
            img = iio.imread(img_file)
            if img.ndim == 3:
                # Convert RGB to greyscale
                img = convert_to_greyscale(img)
            elif img.ndim == 2:
                # Already greyscale, add channel dimension
                img = np.expand_dims(img, axis=-1)
        else:
            img = iio.imread(img_file)
            if img.ndim == 2:
                # If already greyscale, add channel dimension
                img = np.expand_dims(img, axis=-1)

        images.append(img)
        filenames.append(img_file.name)

    # Stack images along first axis (frames)
    volume = np.stack(images, axis=0)
    return volume, filenames


def make_video_from_images(
    image_files: List[Union[str, Path]],
    greyscale: bool = False,
) -> sio.Video:
    """Create a sleap_io.Video object from a list of image files.

    Args:
        image_files: List of paths to image files.
        greyscale: Whether to convert images to greyscale.

    Returns:
        A sleap_io.Video object containing the images.

    Raises:
        ValueError: If no image files provided.
        ImportError: If sleap_io is not installed.
    """
    logger.debug(f"Creating Video from {len(file_paths)} image files")

    # Create Video object from image files
    # sleap_io.Video can be created from a list of image filenames
    video = sio.Video.from_filename(
        filenames=image_files,
        grayscale=greyscale,  # Note: sleap_io uses 'grayscale' not 'greyscale'
    )

    logger.info(f"Created Video with {len(video)} frames")
    return video


def save_array_as_h5(
    array: np.ndarray,
    output_path: Union[str, Path],
    dataset_name: str = "vol",
    compression: str = "gzip",
    compression_opts: int = 1,
    chunks: Optional[Union[bool, tuple]] = True,
) -> Path:
    """Save a numpy array as an H5 file.

    This is a utility function for cases where H5 export is still desired,
    but the main workflow uses sleap_io.Video objects.

    Args:
        array: Array to save, typically with shape (frames, height, width, channels).
        output_path: Path where the H5 file will be saved.
        dataset_name: Name of the dataset inside the HDF5 file.
        compression: Compression algorithm for the dataset.
        compression_opts: Compression level (1-9 for gzip).
        chunks: True/tuple for chunking; None/False disables chunking.

    Returns:
        Path to the saved H5 file.

    Raises:
        ValueError: If array is empty or output_path is not provided.
    """
    if array.size == 0:
        raise ValueError("Cannot save empty array")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        f.create_dataset(
            dataset_name,
            data=array,
            compression=compression,
            compression_opts=compression_opts,
            chunks=chunks,
        )

    logger.info(f"Saved array with shape {array.shape} to {output_path}")
    return output_path


def find_image_directories(base_dir: Union[str, Path]) -> List[Path]:
    """Find all subdirectories containing TIFF images.

    Args:
        base_dir: Path to the base directory to search.

    Returns:
        List of Paths to directories containing TIFF images.
    """
    base_dir = Path(base_dir)
    logger.debug(f"Searching for image directories in: {base_dir}")

    if not base_dir.exists():
        logger.error(f"Base directory does not exist: {base_dir}")
        return []

    if not base_dir.is_dir():
        logger.error(f"Base path is not a directory: {base_dir}")
        return []

    image_dirs = []
    total_dirs_scanned = 0

    for dirpath, dirnames, filenames in os.walk(base_dir):
        total_dirs_scanned += 1
        tiff_files = [fn for fn in filenames if fn.endswith((".tif", ".tiff"))]

        if tiff_files:
            image_dirs.append(Path(dirpath))
            logger.debug(f"Found {len(tiff_files)} TIFF files in: {dirpath}")

    logger.info(
        f"Scanned {total_dirs_scanned} directories, found {len(image_dirs)} with TIFF images"
    )
    return image_dirs
