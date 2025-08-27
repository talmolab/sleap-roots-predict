"""Shared test fixtures for sleap-roots-predict tests."""

import shutil
import tempfile
from pathlib import Path
from typing import Generator, List

import h5py
import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for testing."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # Cleanup
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture
def sample_rgb_image() -> np.ndarray:
    """Create a sample RGB image array."""
    # Create a 100x100 RGB image with different patterns in each channel
    height, width = 100, 100
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Red channel: gradient from top to bottom
    image[:, :, 0] = np.linspace(0, 255, height).reshape(-1, 1)

    # Green channel: gradient from left to right
    image[:, :, 1] = np.linspace(0, 255, width).reshape(1, -1)

    # Blue channel: circular pattern
    y, x = np.ogrid[:height, :width]
    center_y, center_x = height // 2, width // 2
    mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= 30**2
    image[:, :, 2][mask] = 200

    return image


@pytest.fixture
def sample_greyscale_image() -> np.ndarray:
    """Create a sample greyscale image array."""
    # Create a 100x100 greyscale image with a gradient
    height, width = 100, 100
    image = np.linspace(0, 255, height * width).reshape(height, width).astype(np.uint8)
    return image


@pytest.fixture
def sample_image_sequence(sample_rgb_image: np.ndarray) -> List[np.ndarray]:
    """Create a sequence of images with slight variations."""
    images = []
    for i in range(5):
        # Create variations by rotating brightness
        modified = sample_rgb_image.copy()
        modified = np.clip(modified + i * 10, 0, 255).astype(np.uint8)
        images.append(modified)
    return images


@pytest.fixture
def image_directory_with_tiffs(
    temp_dir: Path, sample_image_sequence: List[np.ndarray]
) -> Path:
    """Create a directory with TIFF images following naming convention."""
    image_dir = temp_dir / "images"
    image_dir.mkdir()

    # Create images with proper naming convention (YYYYMMDD-HHMMSS)
    timestamps = [
        "20240101-120000",
        "20240101-130000",
        "20240101-140000",
        "20240101-150000",
        "20240101-160000",
    ]
    plate_numbers = ["001", "001", "001", "001", "001"]

    for idx, (img, timestamp, plate_num) in enumerate(
        zip(sample_image_sequence, timestamps, plate_numbers)
    ):
        filename = f"exp1_set1_day{idx+1}_{timestamp}_{plate_num}.tif"
        filepath = image_dir / filename
        Image.fromarray(img).save(filepath)

    return image_dir


@pytest.fixture
def empty_directory(temp_dir: Path) -> Path:
    """Create an empty directory for testing."""
    empty_dir = temp_dir / "empty"
    empty_dir.mkdir()
    return empty_dir


@pytest.fixture
def non_existent_path(temp_dir: Path) -> Path:
    """Return a path that doesn't exist."""
    return temp_dir / "non_existent"


@pytest.fixture
def sample_4d_array() -> np.ndarray:
    """Create a sample 4D array for H5 testing."""
    # Shape: (frames, height, width, channels)
    frames, height, width, channels = 5, 50, 50, 3
    array = np.random.randint(0, 256, (frames, height, width, channels), dtype=np.uint8)
    return array


@pytest.fixture
def sample_h5_file(temp_dir: Path, sample_4d_array: np.ndarray) -> Path:
    """Create a sample H5 file."""
    h5_path = temp_dir / "test_data.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("vol", data=sample_4d_array, compression="gzip")
    return h5_path


@pytest.fixture
def metadata_csv(temp_dir: Path) -> Path:
    """Create a sample metadata CSV file for testing."""
    import pandas as pd

    csv_path = temp_dir / "metadata.csv"

    # Create metadata matching test plate numbers
    metadata = pd.DataFrame(
        {
            "plate_number": [1, 2, 3],
            "treatment": ["control", "treatment_A", "treatment_B"],
            "num_plants": [1, 3, 6],
            "accesion": ["KitaakeX", "hk1-3", "KitaakeX"],
            "num_images": [100, 100, 100],
            "growth_media": ["1/2 MS", "1/2 MS", "1/2 MS"],
            "experiment_start": ["2025-01-01", "2025-01-01", "2025-01-01"],
        }
    )

    metadata.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def metadata_csv_missing_columns(temp_dir: Path) -> Path:
    """Create a metadata CSV with missing required columns."""
    import pandas as pd

    csv_path = temp_dir / "bad_metadata.csv"

    # Missing 'num_plants' column
    metadata = pd.DataFrame(
        {"plate_number": [1, 2], "treatment": ["control", "treatment_A"]}
    )

    metadata.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def malformed_image_directory(temp_dir: Path) -> Path:
    """Create a directory with malformed image filenames."""
    bad_dir = temp_dir / "malformed"
    bad_dir.mkdir()

    # Create images with improper naming
    img = np.ones((10, 10, 3), dtype=np.uint8) * 127

    bad_names = [
        "image1.tif",  # Missing underscores
        "only_one_underscore.tif",  # Not enough parts
        "no_extension",  # No file extension
    ]

    for name in bad_names:
        filepath = bad_dir / name
        if name.endswith(".tif"):
            Image.fromarray(img).save(filepath)
        else:
            filepath.touch()  # Create empty file

    return bad_dir


@pytest.fixture
def real_world_filenames() -> List[str]:
    """Collection of real-world filename patterns for testing."""
    return [
        # Your actual example
        r"\\multilab-na.ad.salk.edu\hpi_dev\users\eberrigan\circumnutation\20250819_Suyash_Patil_CMTN_Kitx_vs_Hk1-3_07-30-25\001\_set1_day1_20250730-212631_001.tif",
        # Variations with different formats
        "_set1_day1_20250730-212631_001.tif",
        "_set2_day10_20250801-143022_025.tif",
        "exp1_set1_day1_20250730-212631_001.tif",
        "CMTN_Kitx_vs_Hk1-3_set1_day1_20250730-212631_001.tif",
        # Non-standard formats (should not parse datetime)
        "prefix_20250730_212631_001.tif",  # Underscore separator (non-standard)
        "prefix_20250730212631_001.tif",  # No separator (non-standard)
        "test_20250730-2126_001.tif",  # Shorter time (non-standard)
        # Different suffixes
        "image_20250730-212631_1.tif",
        "image_20250730-212631_001.tif",
        "image_20250730-212631_9999.tif",
        # No suffix
        "image_20250730-212631.tif",
        # Complex prefixes
        "2024-01-15_exp_name_with_many_parts_set1_day2_20240115-143022_001.tif",
        "PI_Name_Project123_Condition_A_set3_day5_20250101-090000_042.tif",
        # Edge cases
        "20250730-212631_001.tif",  # No prefix
        "_20250730-212631_001.tif",  # Empty prefix
        "test__20250730-212631__001.tif",  # Multiple underscores
        # Missing datetime (should handle gracefully)
        "no_datetime_here_001.tif",
        "completely_random_name.tif",
    ]


@pytest.fixture
def complex_filename_directory(temp_dir: Path) -> Path:
    """Create a directory with complex real-world filename patterns."""
    complex_dir = temp_dir / "complex_filenames"
    complex_dir.mkdir()

    img = np.ones((10, 10, 3), dtype=np.uint8) * 100

    filenames = [
        "_set1_day1_20250730-212631_001.tif",
        "_set1_day1_20250730-212632_002.tif",
        "_set1_day2_20250731-083000_001.tif",
        "_set2_day1_20250730-212631_001.tif",
        "exp123_set1_day1_20250730-143000_001.tif",
        "CMTN_Kitx_set1_day1_20250730-090000_001.tif",
    ]

    for filename in filenames:
        filepath = complex_dir / filename
        Image.fromarray(img).save(filepath)

    return complex_dir


@pytest.fixture
def mixed_format_directory(temp_dir: Path, sample_rgb_image: np.ndarray) -> Path:
    """Create a directory with mixed image formats."""
    mixed_dir = temp_dir / "mixed_formats"
    mixed_dir.mkdir()

    # Save same image in different formats
    formats = [
        ("image_20240101-120000_001.tif", "TIFF"),
        ("image_20240101-120000_002.png", "PNG"),
        ("image_20240101-120000_003.jpg", "JPEG"),
    ]

    for filename, fmt in formats:
        filepath = mixed_dir / filename
        Image.fromarray(sample_rgb_image).save(filepath, format=fmt)

    # Also add a non-image file
    (mixed_dir / "readme.txt").write_text("This is not an image")

    return mixed_dir


@pytest.fixture
def large_image() -> np.ndarray:
    """Create a large image for performance testing."""
    # 1920x1080 HD resolution RGB image
    return np.random.randint(0, 256, (1920, 1080, 3), dtype=np.uint8)


@pytest.fixture
def single_channel_image() -> np.ndarray:
    """Create a single-channel image without channel dimension."""
    return np.random.randint(0, 256, (100, 100), dtype=np.uint8)


@pytest.fixture
def rgba_image() -> np.ndarray:
    """Create an RGBA image with alpha channel."""
    image = np.zeros((100, 100, 4), dtype=np.uint8)
    image[:, :, :3] = 128  # Set RGB to gray
    image[:, :, 3] = 255  # Set alpha to opaque
    return image
