"""Utilities for processing directories of images into H5 files with metadata."""

import logging
import os
import re
import h5py
import imageio.v3 as iio
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


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


def extract_metadata_from_filename(
    filename: Union[str, Path],
) -> dict:
    """Extract metadata from filename including datetime, prefix, and suffix.

    Expects datetime in YYYYMMDD-HHMMSS format (always present in automated captures).
    Also extracts variable prefix and suffix information.

    Expected pattern:
    - [prefix_]YYYYMMDD-HHMMSS[_suffix].ext

    Where:
    - datetime: YYYYMMDD-HHMMSS format (required, automated timestamp)
    - suffix: final numeric identifier (e.g., "001")
    - prefix: descriptive prefix before datetime (may include set/day info)

    Args:
        filename: Path or name of the file to parse.

    Returns:
        Dictionary with extracted metadata fields.
    """
    import re
    from datetime import datetime

    # Convert to Path and get just the filename
    if isinstance(filename, str):
        filename = Path(filename)

    name = filename.stem  # Remove extension

    # Initialize result
    metadata = {
        "filename": filename.name,
        "datetime_str": None,
        "datetime": None,
        "suffix": None,
        "prefix": None,
        "set_info": None,
        "day_info": None,
        "full_prefix": None,
    }

    # Pattern for standard datetime format: YYYYMMDD-HHMMSS (automated capture format)
    # This is the expected format from the automated imaging system
    datetime_pattern = r"(\d{8})-(\d{6})"

    datetime_match = re.search(datetime_pattern, name)

    if datetime_match:
        # Extract and parse the datetime
        date_str = datetime_match.group(1)  # YYYYMMDD
        time_str = datetime_match.group(2)  # HHMMSS
        metadata["datetime_str"] = f"{date_str}-{time_str}"

        try:
            metadata["datetime"] = datetime.strptime(
                f"{date_str} {time_str}", "%Y%m%d %H%M%S"
            )
        except ValueError:
            pass  # Keep datetime as None if parsing fails

        # Split filename at the datetime to get prefix and suffix parts
        dt_start = datetime_match.start()
        dt_end = datetime_match.end()

        prefix_part = name[:dt_start].rstrip("_- ")
        suffix_part = name[dt_end:].lstrip("_- ")

        # Extract suffix (numeric part after datetime, typically plate number like "001")
        suffix_match = re.search(r"(\d+)", suffix_part)
        if suffix_match:
            metadata["suffix"] = suffix_match.group(1)

        # Process prefix part
        if prefix_part:
            metadata["full_prefix"] = prefix_part

            # Look for set/day information
            set_match = re.search(r"set(\d+)", prefix_part, re.IGNORECASE)
            if set_match:
                metadata["set_info"] = int(set_match.group(1))

            day_match = re.search(r"day(\d+)", prefix_part, re.IGNORECASE)
            if day_match:
                metadata["day_info"] = int(day_match.group(1))

            # Get the main prefix (before set/day info)
            prefix_clean = re.sub(
                r"[_-]?set\d+[_-]?", "", prefix_part, flags=re.IGNORECASE
            )
            prefix_clean = re.sub(
                r"[_-]?day\d+[_-]?", "", prefix_clean, flags=re.IGNORECASE
            )
            prefix_clean = prefix_clean.strip("_- ")
            if prefix_clean:
                metadata["prefix"] = prefix_clean
    else:
        # No datetime found - try to get at least suffix
        # Look for final numeric identifier
        parts = name.split("_")
        if parts and parts[-1].isdigit():
            metadata["suffix"] = parts[-1]
            if len(parts) > 1:
                metadata["prefix"] = "_".join(parts[:-1])

    return metadata


def make_h5_from_images(
    images: np.ndarray,
    output_path: Path,
    compression: str = "gzip",
    compression_opts: int = 1,
) -> Path:
    """Create an H5 file from a numpy array of images.

    Args:
        images: Array of images with shape (frames, height, width, channels).
        output_path: Path where the H5 file will be saved.
        compression: Compression algorithm to use.
        compression_opts: Compression level (1-9, where 1 is fastest, 9 is best compression).

    Returns:
        Path to the created H5 file.
    """
    if images.ndim != 4:
        raise ValueError(
            f"Expected 4D array (frames, height, width, channels), got shape {images.shape}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        f.create_dataset(
            "vol",
            data=images,
            compression=compression,
            compression_opts=compression_opts,
        )
        logger.info(f"Saved volume with shape {images.shape} to {output_path}")

    return output_path


def create_metadata_dataframe(
    filenames: List[str],
    experiment_name: str,
    treatment: str,
    num_plants: int,
) -> pd.DataFrame:
    """Create a metadata dataframe from filenames and experimental info.

    Args:
        filenames: List of image filenames.
        experiment_name: Name of the experiment.
        treatment: Chemical or physical alterations to the plate media.
        num_plants: Number of plants expected on a plate image.

    Returns:
        DataFrame with metadata for each frame.
    """
    df_rows = []

    for frame_idx, filename in enumerate(filenames):
        metadata = extract_metadata_from_filename(filename)

        row = {
            "experiment": experiment_name,
            "filename": filename,
            "treatment": treatment,
            "expected_num_plants": num_plants,
            "frame": frame_idx,
            # From extracted metadata
            "datetime_str": metadata.get("datetime_str"),
            "datetime": metadata.get("datetime"),
            "plate_number": metadata.get("suffix"),  # Use suffix as plate number
            "prefix": metadata.get("prefix"),
            "set_info": metadata.get("set_info"),
            "day_info": metadata.get("day_info"),
        }

        df_rows.append(row)

    # Create DataFrame with proper columns even if empty
    df = pd.DataFrame(df_rows)

    # Ensure all expected columns exist even if dataframe is empty
    expected_columns = [
        "experiment",
        "filename",
        "treatment",
        "expected_num_plants",
        "frame",
        "datetime_str",
        "datetime",
        "plate_number",
        "prefix",
        "set_info",
        "day_info",
    ]
    for col in expected_columns:
        if col not in df.columns:
            df[col] = None

    return df


def process_timelapse_image_directory(
    source_dir: Union[str, Path],
    experiment_name: str,
    treatment: str,
    num_plants: int,
    greyscale: bool = False,
    output_dir: Optional[Union[str, Path]] = None,
    image_pattern: str = "*.tif",
) -> Tuple[Optional[Path], Optional[Path]]:
    """Process a directory of timelapse images into an H5 file and metadata CSV.

    Args:
        source_dir: Path to the source directory containing images.
        experiment_name: Name of the experiment.
        treatment: Chemical or physical alterations to the plate media.
        num_plants: Number of plants expected on a plate image.
        greyscale: Whether to convert images to greyscale.
        output_dir: Directory to store output files. If None, uses source directory.
        image_pattern: Glob pattern for finding image files.

    Returns:
        Tuple of (H5 file path, metadata CSV path), or (None, None) if processing failed.
    """
    # Convert to Path objects
    source_dir = Path(source_dir)
    if output_dir is None:
        output_dir = source_dir
    else:
        output_dir = Path(output_dir)

    # Validate source directory
    if not source_dir.exists():
        logger.error(f"Source directory {source_dir} does not exist.")
        return None, None

    if not source_dir.is_dir():
        logger.error(f"Source path {source_dir} is not a directory.")
        return None, None

    # Find image files
    image_files = list(source_dir.glob(image_pattern))

    if not image_files:
        logger.warning(
            f"No image files matching pattern '{image_pattern}' found in {source_dir}."
        )
        return None, None

    logger.info(f"Found {len(image_files)} image files in {source_dir}")

    # Sort files naturally
    sorted_files = natural_sort(image_files)
    image_files = [Path(f) for f in sorted_files]

    # Load and process images
    try:
        volume, filenames = load_images(image_files, greyscale=greyscale)
    except Exception as e:
        logger.error(f"Failed to load images: {e}")
        return None, None

    # Create output paths
    suffix = "_greyscale" if greyscale else "_color"
    h5_name = f"plate_{source_dir.name}{suffix}.h5"
    h5_path = output_dir / h5_name

    csv_name = f"plate_{source_dir.name}_metadata.csv"
    csv_path = output_dir / csv_name

    # Save H5 file
    try:
        make_h5_from_images(volume, h5_path)
    except Exception as e:
        logger.error(f"Failed to create H5 file: {e}")
        return None, None

    # Create and save metadata
    try:
        metadata_df = create_metadata_dataframe(
            filenames, experiment_name, treatment, num_plants
        )
        metadata_df.to_csv(csv_path, index=False)
        logger.info(f"Saved metadata to {csv_path}")
    except Exception as e:
        logger.error(f"Failed to save metadata: {e}")
        return h5_path, None

    return h5_path, csv_path
