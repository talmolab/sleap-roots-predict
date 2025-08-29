"""Functions for processing plate-based timelapse experiments.

This module provides functionality for processing directories of timelapse images
from plate-based experiments, including metadata extraction, validation, and
conversion to H5 format with associated metadata CSV files.
"""

import json
import logging
import imageio.v3 as iio
import numpy as np
import pandas as pd
import sleap_io as sio

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .video_utils import (
    find_image_directories,
    load_images,
    make_video_from_images,
    save_array_as_h5,
    natural_sort,
)
from sleap_roots_predict.predict import (
    predict_on_h5,
    predict_on_video,
    make_predictor,
)

# Initialize logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def extract_timelapse_metadata_from_filename(
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


def create_timelapse_metadata_dataframe(
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
        metadata = extract_timelapse_metadata_from_filename(filename)

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


def check_timelapse_image_directory(
    image_dir: Union[str, Path],
    expected_suffix_pattern: Optional[str] = None,
    min_images: int = 1,
    max_images: Optional[int] = None,
    check_datetime: bool = True,
    check_suffix_consistency: bool = True,
) -> Dict[str, Any]:
    r"""Check an image directory for validity and consistency.

    Args:
        image_dir: Path to the directory containing TIFF images.
        expected_suffix_pattern: Optional regex pattern for suffix validation (e.g., r'^\\d{3}$' for '001', '002', etc.)
        min_images: Minimum number of images required (default 1).
        max_images: Maximum number of images allowed (None for unlimited).
        check_datetime: Whether to check for valid datetime in filenames.
        check_suffix_consistency: Whether all files should have the same suffix (for plates).

    Returns:
        Dictionary with check results including:
            - valid: Boolean indicating if all checks passed
            - image_count: Number of TIFF images found
            - suffixes: Set of unique suffixes found
            - errors: List of error messages
            - warnings: List of warning messages
            - metadata: List of metadata dicts for each image
    """
    image_dir = Path(image_dir)
    logger.debug(f"Starting validation checks for directory: {image_dir}")

    results = {
        "valid": True,
        "image_count": 0,
        "suffixes": set(),
        "errors": [],
        "warnings": [],
        "metadata": [],
        "directory": image_dir.as_posix(),
    }

    # Check directory exists
    if not image_dir.exists():
        error_msg = f"Directory does not exist: {image_dir}"
        logger.error(error_msg)
        results["valid"] = False
        results["errors"].append(error_msg)
        return results

    if not image_dir.is_dir():
        error_msg = f"Path is not a directory: {image_dir}"
        logger.error(error_msg)
        results["valid"] = False
        results["errors"].append(error_msg)
        return results

    # Find TIFF images
    logger.debug(f"Searching for TIFF files in: {image_dir}")
    tiff_files = list(image_dir.glob("*.tif")) + list(image_dir.glob("*.tiff"))
    results["image_count"] = len(tiff_files)
    logger.info(f"Found {results['image_count']} TIFF files in {image_dir.name}")

    # Check image count
    if results["image_count"] < min_images:
        error_msg = f"Too few images: found {results['image_count']}, minimum required {min_images}"
        logger.warning(error_msg)
        results["valid"] = False
        results["errors"].append(error_msg)

    if max_images and results["image_count"] > max_images:
        error_msg = f"Too many images: found {results['image_count']}, maximum allowed {max_images}"
        logger.warning(error_msg)
        results["valid"] = False
        results["errors"].append(error_msg)

    if results["image_count"] == 0:
        logger.warning(f"No TIFF images found in {image_dir}")
        return results

    # Process each file and extract metadata
    logger.debug(f"Extracting metadata from {len(tiff_files)} files")
    missing_datetime_count = 0

    for tiff_file in tiff_files:
        metadata = extract_timelapse_metadata_from_filename(tiff_file)
        results["metadata"].append(metadata)

        # Check datetime if required
        if check_datetime and metadata["datetime"] is None:
            missing_datetime_count += 1
            warning_msg = f"No valid datetime found in filename: {tiff_file.name}"
            logger.debug(warning_msg)
            results["warnings"].append(warning_msg)

        # Collect suffixes
        if metadata["suffix"]:
            results["suffixes"].add(metadata["suffix"])
            logger.debug(f"Found suffix '{metadata['suffix']}' in {tiff_file.name}")

    if missing_datetime_count > 0:
        logger.warning(
            f"Missing datetime in {missing_datetime_count}/{len(tiff_files)} files"
        )

    # Check suffix consistency if required (all files should have same suffix for a plate)
    if check_suffix_consistency:
        if len(results["suffixes"]) > 1:
            error_msg = (
                f"Inconsistent suffixes found: {results['suffixes']}. "
                "All images in a plate directory should have the same suffix."
            )
            logger.error(error_msg)
            results["valid"] = False
            results["errors"].append(error_msg)
        elif len(results["suffixes"]) == 0:
            warning_msg = "No suffixes found in any filenames"
            logger.warning(warning_msg)
            results["warnings"].append(warning_msg)
        elif len(results["suffixes"]) == 1:
            logger.info(
                f"All files have consistent suffix: {list(results['suffixes'])[0]}"
            )

    # Validate suffix pattern if provided
    if expected_suffix_pattern and results["suffixes"]:
        import re

        logger.debug(f"Validating suffixes against pattern: {expected_suffix_pattern}")
        pattern = re.compile(expected_suffix_pattern)

        for suffix in results["suffixes"]:
            if not pattern.match(suffix):
                error_msg = f"Suffix '{suffix}' does not match expected pattern '{expected_suffix_pattern}'"
                logger.error(error_msg)
                results["valid"] = False
                results["errors"].append(error_msg)
            else:
                logger.debug(f"Suffix '{suffix}' matches expected pattern")

    # Check for duplicate timestamps (shouldn't happen in automated capture)
    timestamps = [m["datetime_str"] for m in results["metadata"] if m["datetime_str"]]
    if timestamps:
        unique_timestamps = set(timestamps)
        if len(timestamps) != len(unique_timestamps):
            duplicate_count = len(timestamps) - len(unique_timestamps)
            warning_msg = f"Found {duplicate_count} duplicate timestamps in filenames"
            logger.warning(warning_msg)
            results["warnings"].append(warning_msg)

    # Sort check - ensure files are in chronological order when sorted naturally
    if timestamps and all(t is not None for t in timestamps):
        # Get timestamps in natural sort order
        natural_sorted_files = natural_sort([f.name for f in tiff_files])
        natural_order_timestamps = [
            extract_timelapse_metadata_from_filename(f)["datetime_str"]
            for f in natural_sorted_files
        ]
        # Check if they're in chronological order
        sorted_natural_timestamps = sorted([t for t in natural_order_timestamps if t])
        if natural_order_timestamps != sorted_natural_timestamps:
            warning_msg = "Files are not in chronological order when naturally sorted"
            logger.warning(warning_msg)
            results["warnings"].append(warning_msg)
        else:
            logger.debug("Files are in correct chronological order")

    # Log final validation result
    if results["valid"]:
        logger.info(f"[OK] Directory passed all validation checks: {image_dir.name}")
    else:
        logger.error(
            f"✗ Directory failed validation: {image_dir.name} ({len(results['errors'])} errors)"
        )

    if results["warnings"]:
        logger.info(
            f"[WARNING] Directory has {len(results['warnings'])} warnings: {image_dir.name}"
        )

    return results


def process_timelapse_image_directory(
    source_dir: Union[str, Path],
    experiment_name: str,
    treatment: str,
    num_plants: int,
    save_h5: bool = False,
    greyscale: bool = False,
    output_dir: Optional[Union[str, Path]] = None,
    image_pattern: str = "*.tif",
) -> Union[
    Tuple[Optional[Path], Optional[Path]], Tuple[Optional[sio.Video], Optional[Path]]
]:
    """Process a directory of timelapse images into an H5 file or Video and metadata CSV.

    Args:
        source_dir: Path to the source directory containing images.
        experiment_name: Name of the experiment.
        treatment: Chemical or physical alterations to the plate media.
        num_plants: Number of plants expected on a plate image.
        greyscale: Whether to convert images to greyscale.
        output_dir: Directory to store output files. If None, uses source directory.
        image_pattern: Glob pattern for finding image files.

    Returns:
        Tuple of (H5 file path, metadata CSV path), or (`sio.Video`, metadata CSV path)
            or (None, None) if processing failed.
    """
    # Convert to Path objects
    source_dir = Path(source_dir)
    if output_dir is None:
        output_dir = source_dir
    else:
        output_dir = Path(output_dir)
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

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
    sorted_image_paths = [Path(f) for f in sorted_files]
    sorted_image_names = [Path(f).name for f in sorted_files]

    csv_name = f"plate_{source_dir.name}_metadata.csv"
    csv_path = output_dir / csv_name

    # Create and save metadata
    try:
        metadata_df = create_timelapse_metadata_dataframe(
            sorted_image_names, experiment_name, treatment, num_plants
        )
        metadata_df.to_csv(csv_path, index=False)
        logger.info(f"Saved metadata to {csv_path}")
    except Exception as e:
        logger.error(f"Failed to save metadata: {e}")
        csv_path = None

    if save_h5:
        try:
            # Load and process images
            try:
                volume, filenames = load_images(sorted_image_paths, greyscale=greyscale)
            except Exception as e:
                logger.error(f"Failed to load images: {e}")
                h5_path = None
            # Create output paths
            suffix = "_greyscale" if greyscale else "_color"
            h5_name = f"plate_{source_dir.name}{suffix}.h5"
            h5_path = output_dir / h5_name
            save_array_as_h5(volume, h5_path)
        except Exception as e:
            logger.error(f"Failed to create H5 file: {e}")
            h5_path = None
        return h5_path, csv_path

    else:
        try:
            # Make `sio.Video`
            video = make_video_from_images(
                image_files=sorted_image_paths, greyscale=greyscale
            )
        except Exception as e:
            logger.error(f"Failed to create video: {e}")
            video = None
        return video, csv_path


def process_timelapse_experiment(
    base_dir: Union[str, Path],
    metadata_csv: Union[str, Path],
    experiment_name: str,
    save_h5: bool = False,
    output_dir: Optional[Union[str, Path]] = None,
    model_paths: List[Union[str, Path]] = [],
    # Predictor parameters
    peak_threshold: float = 0.2,
    batch_size: int = 4,
    device: str = "auto",
    # Optional processing parameters
    greyscale: bool = False,
    image_pattern: str = "*.tif",
    # Validation parameters
    expected_suffix_pattern: Optional[str] = r"^\d{3}$",  # Default to 3-digit format
    min_images: int = 1,
    max_images: Optional[int] = None,
    check_datetime: bool = True,
    check_suffix_consistency: bool = True,
    # Control parameters
    dry_run: bool = False,
    log_file: Optional[Union[str, Path]] = None,
    results_json: Optional[Union[str, Path]] = None,  # Add JSON output parameter
) -> Dict[str, List[Dict[str, Any]]]:
    r"""Process an entire experiment with multiple image directories using metadata from CSV.

    This function:
    1. Loads plate metadata from a CSV file
    2. Finds all subdirectories containing TIFF images
    3. Validates each directory using check_image_directory
    4. Matches directories to plate metadata using suffix/plate number
    5. Processes valid directories to create H5 timelapses with appropriate metadata

    Args:
        base_dir: Base directory containing subdirectories with TIFF images.
        metadata_csv: Path to CSV file containing per-plate metadata.
                     Must have columns: plate_number, treatment, num_plants
                     May also have: accesion, num_images, experiment_start, growth_media, etc.
        experiment_name: Name of the experiment for metadata.
        save_h5: Whether to save H5 files (if False, uses Video objects directly).
        output_dir: Output directory for H5 and CSV files (defaults to base_dir).
        model_paths: List of paths to SLEAP model directories for prediction.
        peak_threshold: Confidence threshold for peak detection in predictions.
        batch_size: Number of samples per batch for inference.
        device: Device for inference ("auto", "cpu", "cuda", or "mps").
        greyscale: Whether to convert images to greyscale.
        image_pattern: Glob pattern for finding image files (default "*.tif").
        expected_suffix_pattern: Regex pattern for suffix validation (default r'^\d{3}$').
        min_images: Minimum number of images required per directory (default 1).
        max_images: Maximum number of images allowed per directory (default None).
        check_datetime: Whether to check for valid datetime in filenames (default True).
        check_suffix_consistency: Whether all files should have the same suffix (default True).
        dry_run: If True, only perform checks without processing (default False).
        log_file: Optional path to save log output to a file (default None).
        results_json: Optional path to save results as JSON file (default None).

    Returns:
        Dictionary with:
            - processed: List of successfully processed directories
            - failed: List of directories that failed checks
            - skipped: List of directories skipped due to errors
    """
    base_dir = Path(base_dir)
    metadata_csv = Path(metadata_csv)
    output_dir = Path(output_dir) if output_dir else base_dir

    # Set up file logging if requested
    file_handler = None
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")

    # Log experiment configuration
    logger.info("=" * 60)
    logger.info(f"Starting experiment: {experiment_name}")
    logger.info(f"Base directory: {base_dir}")
    logger.info(f"Metadata CSV: {metadata_csv}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Processing mode: {'DRY RUN' if dry_run else 'FULL PROCESSING'}")
    logger.info(f"Greyscale conversion: {greyscale}")
    logger.info(f"Image pattern: {image_pattern}")
    logger.info(f"Validation settings:")
    logger.info(f"  - Expected suffix pattern: {expected_suffix_pattern}")
    logger.info(f"  - Min images: {min_images}")
    logger.info(f"  - Max images: {max_images if max_images else 'unlimited'}")
    logger.info(f"  - Check datetime: {check_datetime}")
    logger.info(f"  - Check suffix consistency: {check_suffix_consistency}")
    logger.info("=" * 60)

    results = {
        "processed": [],
        "failed": [],
        "skipped": [],
    }

    # Initialize predictor if model paths provided
    predictor = None
    if model_paths:
        logger.info(f"Initializing predictor with {len(model_paths)} model(s)")
        logger.info(f"  - Device: {device}")
        logger.info(f"  - Peak threshold: {peak_threshold}")
        logger.info(f"  - Batch size: {batch_size}")
        try:
            predictor = make_predictor(
                model_path=model_paths,
                peak_threshold=peak_threshold,
                batch_size=batch_size,
                device=device,
            )
            logger.info("[OK] Predictor initialized successfully")
        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize predictor: {e}")
            logger.error("Continuing without predictions")
            predictor = None

    # Load metadata from CSV
    logger.info(f"Loading metadata from CSV: {metadata_csv.name}")
    if not metadata_csv.exists():
        logger.error(f"X Metadata CSV file not found: {metadata_csv}")
        logger.error("Cannot proceed without metadata file")
        return results

    try:
        metadata_df = pd.read_csv(metadata_csv)
        logger.debug(f"Successfully loaded CSV with {len(metadata_df)} rows")

        # Clean column names (remove BOM and strip whitespace)
        metadata_df.columns = metadata_df.columns.str.replace("\ufeff", "").str.strip()
        # Remove unnamed columns
        metadata_df = metadata_df.loc[:, ~metadata_df.columns.str.contains("^Unnamed")]
        logger.debug(f"CSV columns after cleaning: {list(metadata_df.columns)}")

        # Check required columns
        required_cols = ["plate_number", "treatment", "num_plants"]
        missing_cols = [col for col in required_cols if col not in metadata_df.columns]
        if missing_cols:
            logger.error(f"X Missing required columns in CSV: {missing_cols}")
            logger.error(f"Available columns: {list(metadata_df.columns)}")
            logger.error("CSV must contain: plate_number, treatment, num_plants")
            return results

        logger.info(f"[OK] All required columns present in metadata CSV")

        # Convert plate_number to string for matching
        metadata_df["plate_number"] = metadata_df["plate_number"].astype(str)

        # Create a dictionary for quick lookup by plate number
        # Convert single-digit plate numbers to 3-digit format for matching with suffixes
        metadata_dict = {}
        invalid_plates = []
        logger.debug("Converting plate numbers to 3-digit format for matching")

        for idx, row in metadata_df.iterrows():
            # Convert plate number to 3-digit format (1 -> 001, 2 -> 002, etc.)
            plate_num = str(row["plate_number"]).strip()
            if plate_num.isdigit():
                plate_num_padded = plate_num.zfill(3)  # Pad with zeros to 3 digits
                metadata_dict[plate_num_padded] = row.to_dict()
                logger.debug(
                    f"  Plate {plate_num} -> {plate_num_padded}: {row['treatment']}"
                )
            else:
                invalid_plates.append(plate_num)
                logger.warning(
                    f"[WARNING] Invalid plate number in CSV row {idx + 1}: '{plate_num}'"
                )

        logger.info(
            f"[OK] Loaded metadata for {len(metadata_dict)} valid plates from {metadata_csv.name}"
        )
        if invalid_plates:
            logger.warning(
                f"[WARNING] Skipped {len(invalid_plates)} invalid plate numbers: {invalid_plates}"
            )

    except Exception as e:
        logger.error(f"[ERROR] Failed to load metadata CSV: {e}")
        logger.exception("Full traceback:")
        return results

    logger.info("-" * 60)
    logger.info(f"Starting directory scan in: {base_dir}")

    # Find all image directories
    image_dirs = find_image_directories(base_dir)
    logger.info(f"Found {len(image_dirs)} directories with TIFF images")

    if not image_dirs:
        logger.warning("No image directories found")
        return results

    # Process each directory
    logger.info("-" * 60)
    logger.info(f"Processing {len(image_dirs)} directories")

    for dir_idx, image_dir in enumerate(image_dirs, 1):
        logger.info("")  # Empty line for readability
        logger.info(
            f"[{dir_idx}/{len(image_dirs)}] Processing directory: {image_dir.name}"
        )
        logger.debug(f"  Full path: {image_dir}")

        # Run checks with individual parameters
        check_results = check_timelapse_image_directory(
            image_dir=image_dir,
            expected_suffix_pattern=expected_suffix_pattern,
            min_images=min_images,
            max_images=max_images,
            check_datetime=check_datetime,
            check_suffix_consistency=check_suffix_consistency,
        )

        if not check_results["valid"]:
            logger.error(f"  [ERROR] Directory failed validation")
            for error in check_results["errors"]:
                logger.error(f"    - {error}")
            results["failed"].append(
                {
                    "directory": str(image_dir),
                    "check_results": check_results,
                }
            )
            logger.debug(f"  Skipping {image_dir.name} due to validation errors")
            continue
        else:
            logger.info(f"  [OK] Directory passed validation")

        # Log warnings if any
        if check_results["warnings"]:
            logger.info(f"  [WARNING] Found {len(check_results['warnings'])} warnings:")
            for warning in check_results["warnings"]:
                logger.warning(f"    - {warning}")

        # Get the plate suffix from the directory
        if not check_results["suffixes"]:
            logger.error(
                f"  [ERROR] No suffix found in filenames, cannot match to metadata"
            )
            logger.debug(f"    Directory: {image_dir}")
            results["skipped"].append(
                {
                    "directory": str(image_dir),
                    "reason": "no_suffix_for_metadata_matching",
                    "check_results": check_results,
                }
            )
            continue

        # Should have exactly one suffix per directory
        plate_suffix = list(check_results["suffixes"])[0]
        logger.info(f"  [INFO] Plate suffix detected: {plate_suffix}")

        # Look up metadata for this plate
        if plate_suffix not in metadata_dict:
            logger.error(f"  [ERROR] No metadata found for plate {plate_suffix}")
            logger.debug(f"    Available plates in CSV: {sorted(metadata_dict.keys())}")
            results["skipped"].append(
                {
                    "directory": str(image_dir),
                    "reason": f"no_metadata_for_plate_{plate_suffix}",
                    "check_results": check_results,
                }
            )
            continue

        plate_metadata = metadata_dict[plate_suffix]
        logger.info(
            f"  [OK] Matched to metadata: treatment='{plate_metadata.get('treatment')}', "
            f"num_plants={plate_metadata.get('num_plants')}"
        )

        # Log additional metadata if present
        extra_metadata = [
            k
            for k in plate_metadata.keys()
            if k not in ["plate_number", "treatment", "num_plants"]
        ]
        if extra_metadata:
            logger.debug(f"    Additional metadata fields: {', '.join(extra_metadata)}")

        if dry_run:
            logger.info(
                f"  [PROCESSING] DRY RUN - would process with: {plate_metadata.get('treatment')}"
            )
            results["skipped"].append(
                {
                    "directory": str(image_dir),
                    "reason": "dry_run",
                    "check_results": check_results,
                    "plate_metadata": plate_metadata,
                }
            )
            continue

        # Process the directory with plate-specific metadata
        try:
            # Determine output paths
            rel_path = image_dir.relative_to(base_dir)
            output_subdir = output_dir / rel_path.parent
            output_subdir.mkdir(parents=True, exist_ok=True)

            # Use directory name as base for output files
            output_base = output_subdir / image_dir.name

            logger.info(f"  [PROCESSING] Starting processing...")
            logger.debug(f"    Output directory: {output_subdir}")
            logger.debug(f"    Output base name: {output_base.name}")

            # Extract required parameters from metadata
            treatment = plate_metadata.get("treatment", "unknown")
            num_plants = plate_metadata.get("num_plants", 1)

            # Call process_timelapse_image_directory with plate-specific parameters
            video, csv_path = process_timelapse_image_directory(
                source_dir=image_dir,
                experiment_name=experiment_name,
                treatment=treatment,
                save_h5=save_h5,
                num_plants=int(num_plants) if pd.notna(num_plants) else 1,
                greyscale=greyscale,
                output_dir=output_subdir,
                image_pattern=image_pattern,
            )

                # If processing succeeded, append additional metadata to the CSV
                if csv_path and csv_path.exists():
                    try:
                        # Read the generated CSV
                        generated_df = pd.read_csv(csv_path)
                        logger.debug(
                            f"    Enhancing CSV with additional metadata columns"
                        )

                        # Add additional metadata columns from the metadata CSV
                        added_cols = []
                        for col, val in plate_metadata.items():
                            if col not in [
                                "treatment",
                                "num_plants",
                                "plate_number",
                            ]:  # Don't duplicate
                                generated_df[col] = val
                                added_cols.append(col)

                        # Save the enhanced CSV
                        generated_df.to_csv(csv_path, index=False)
                        if added_cols:
                            logger.debug(
                                f"    Added {len(added_cols)} additional columns: {', '.join(added_cols)}"
                            )

                    except Exception as e:
                        logger.warning(
                            f"    [WARNING] Could not enhance metadata CSV: {e}"
                        )

                results["processed"].append(
                    {
                        "directory": str(image_dir),
                        "h5_path": str(h5_path),
                        "csv_path": str(csv_path) if csv_path else None,
                        "check_results": check_results,
                        "plate_metadata": plate_metadata,
                    }
                )
                logger.info(f"Successfully processed {image_dir}")
            else:
                results["skipped"].append(
                    {
                        "directory": str(image_dir),
                        "reason": "processing_failed",
                        "check_results": check_results,
                        "plate_metadata": plate_metadata,
                    }
                )
                logger.error(f"Failed to process {image_dir}")

        except Exception as e:
            logger.error(f"Error processing {image_dir}: {e}")
            results["skipped"].append(
                {
                    "directory": str(image_dir),
                    "reason": str(e),
                    "check_results": check_results,
                    "plate_metadata": (
                        plate_metadata if "plate_metadata" in locals() else None
                    ),
                }
            )

    # Summary
    logger.info(f"Processing complete:")
    logger.info(f"  - Processed: {len(results['processed'])} directories")
    logger.info(f"  - Failed validation: {len(results['failed'])} directories")
    logger.info(f"  - Skipped: {len(results['skipped'])} directories")

    # Clean up file handler if it was added
    if file_handler:
        file_handler.flush()  # Ensure all logs are written
        logger.removeHandler(file_handler)
        file_handler.close()

    return results
