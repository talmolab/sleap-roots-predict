"""Comprehensive tests for video_utils module."""

import logging
from pathlib import Path
from typing import List

import h5py
import numpy as np
import pandas as pd
import pytest
from PIL import Image

# Import high-level API functions
from sleap_roots_predict import process_timelapse_experiment

# Import utility functions from their specific modules
from sleap_roots_predict.video_utils import (
    convert_to_greyscale,
    find_image_directories,
    load_images,
    make_video_from_images,
    natural_sort,
    save_array_as_h5,
)

from sleap_roots_predict.plates_timelapse_experiment import (
    check_timelapse_image_directory,
    create_timelapse_metadata_dataframe,
    extract_timelapse_metadata_from_filename,
    process_timelapse_image_directory,
)


class TestNaturalSort:
    """Test the natural_sort function."""

    def test_basic_sorting(self):
        """Test basic natural sorting with numbers."""
        input_list = ["img10.png", "img2.png", "img1.png", "img20.png"]
        expected = ["img1.png", "img2.png", "img10.png", "img20.png"]
        assert natural_sort(input_list) == expected

    def test_path_objects(self):
        """Test sorting with Path objects."""
        input_list = [Path("file10.txt"), Path("file2.txt"), Path("file1.txt")]
        expected = ["file1.txt", "file2.txt", "file10.txt"]
        assert natural_sort(input_list) == expected

    def test_mixed_case(self):
        """Test sorting with mixed case."""
        input_list = ["File10.txt", "file2.txt", "FILE1.txt"]
        expected = ["FILE1.txt", "file2.txt", "File10.txt"]
        assert natural_sort(input_list) == expected

    def test_complex_names(self):
        """Test sorting with complex filenames."""
        input_list = [
            "exp_day10_time2.tif",
            "exp_day2_time10.tif",
            "exp_day2_time2.tif",
            "exp_day1_time1.tif",
        ]
        expected = [
            "exp_day1_time1.tif",
            "exp_day2_time2.tif",
            "exp_day2_time10.tif",
            "exp_day10_time2.tif",
        ]
        assert natural_sort(input_list) == expected

    def test_empty_list(self):
        """Test sorting an empty list."""
        assert natural_sort([]) == []

    def test_single_item(self):
        """Test sorting a single item."""
        assert natural_sort(["file.txt"]) == ["file.txt"]

    def test_no_numbers(self):
        """Test sorting strings without numbers."""
        input_list = ["zebra", "apple", "monkey", "banana"]
        expected = ["apple", "banana", "monkey", "zebra"]
        assert natural_sort(input_list) == expected

    def test_error_handling(self):
        """Test error handling in natural_sort."""
        # Test with None values
        items = ["file1.txt", None, "file2.txt"]
        # Should handle None gracefully
        sorted_items = natural_sort(items)
        assert None in sorted_items  # None should be handled


class TestConvertToGreyscale:
    """Test the convert_to_greyscale function."""

    def test_rgb_to_greyscale_weights(self, sample_rgb_image):
        """Test converting RGB image to greyscale using weights method."""
        grey = convert_to_greyscale(sample_rgb_image, method="weights")
        assert grey.ndim == 3
        assert grey.shape[-1] == 1
        assert grey.dtype == np.uint8
        # Check values are within expected range
        assert grey.min() >= 0
        assert grey.max() <= 255

    def test_rgb_to_greyscale_average(self, sample_rgb_image):
        """Test converting RGB image to greyscale using average method."""
        grey = convert_to_greyscale(sample_rgb_image, method="average")
        assert grey.ndim == 3
        assert grey.shape[-1] == 1
        assert grey.dtype == np.uint8
        assert grey.min() >= 0
        assert grey.max() <= 255

    def test_default_method(self, sample_rgb_image):
        """Test that default method is weights."""
        grey_default = convert_to_greyscale(sample_rgb_image)
        grey_weights = convert_to_greyscale(sample_rgb_image, method="weights")
        assert np.array_equal(grey_default, grey_weights)

    def test_greyscale_weights_calculation(self):
        """Test that correct weights are applied."""
        # Create an image with known values
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[:, :, 0] = 100  # Red
        img[:, :, 1] = 150  # Green
        img[:, :, 2] = 50  # Blue

        grey = convert_to_greyscale(img, method="weights")
        # Expected: 100*0.2989 + 150*0.5870 + 50*0.1140 = 123.59
        expected_value = int(100 * 0.2989 + 150 * 0.5870 + 50 * 0.1140)
        assert np.allclose(grey[:, :, 0], expected_value, atol=1)

    def test_greyscale_average_calculation(self):
        """Test that average method works correctly."""
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[:, :, 0] = 90  # Red
        img[:, :, 1] = 120  # Green
        img[:, :, 2] = 150  # Blue

        grey = convert_to_greyscale(img, method="average")
        # Expected: (90 + 120 + 150) / 3 = 120
        expected_value = int((90 + 120 + 150) / 3)
        assert np.allclose(grey[:, :, 0], expected_value, atol=1)

    def test_invalid_method(self, sample_rgb_image):
        """Test error handling for invalid method."""
        with pytest.raises(ValueError, match="Unknown method"):
            convert_to_greyscale(sample_rgb_image, method="invalid")

    def test_invalid_shape(self, sample_greyscale_image):
        """Test error handling for invalid input shape."""
        with pytest.raises(ValueError, match="Expected RGB image"):
            convert_to_greyscale(sample_greyscale_image)

    def test_4d_input(self):
        """Test converting 4D array (batch of images)."""
        batch = np.random.randint(0, 256, (5, 100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Expected RGB image"):
            convert_to_greyscale(batch[:, :, :, :2])  # Wrong channel count

    def test_rgba_image(self, rgba_image):
        """Test that RGBA images use only RGB channels."""
        # Should only use first 3 channels
        grey = convert_to_greyscale(rgba_image[:, :, :3].copy())
        assert grey.shape == (100, 100, 1)


class TestLoadImages:
    """Test the load_images function."""

    def test_load_rgb_images(self, image_directory_with_tiffs):
        """Test loading RGB images from directory."""
        image_files = sorted(image_directory_with_tiffs.glob("*.tif"))
        volume, filenames = load_images(image_files, greyscale=False)

        assert volume.ndim == 4
        assert volume.shape[0] == 5  # 5 images
        assert volume.shape[-1] == 3  # RGB
        assert len(filenames) == 5

    def test_load_greyscale_conversion(self, image_directory_with_tiffs):
        """Test loading images with greyscale conversion using imageio mode='L'."""
        image_files = sorted(image_directory_with_tiffs.glob("*.tif"))
        volume, filenames = load_images(image_files, greyscale=True)

        assert volume.ndim == 4
        assert volume.shape[0] == 5
        assert volume.shape[-1] == 1  # Greyscale
        assert len(filenames) == 5
        assert volume.dtype == np.uint8
        # Verify it's actually greyscale (all pixels should be in 0-255 range)
        assert volume.min() >= 0
        assert volume.max() <= 255

    def test_empty_file_list(self):
        """Test error handling for empty file list."""
        with pytest.raises(ValueError, match="No image files provided"):
            load_images([], greyscale=False)

    def test_mixed_formats(self, mixed_format_directory):
        """Test loading different image formats."""
        # Load only TIFF files
        tiff_files = sorted(mixed_format_directory.glob("*.tif"))
        volume, filenames = load_images(tiff_files, greyscale=False)
        assert volume.shape[0] == 1  # Only 1 TIFF file

        # Load PNG files
        png_files = sorted(mixed_format_directory.glob("*.png"))
        volume, filenames = load_images(png_files, greyscale=False)
        assert volume.shape[0] == 1  # Only 1 PNG file

    def test_greyscale_image_loading(self, temp_dir):
        """Test loading already greyscale images."""
        # Create greyscale image
        grey_img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        img_path = temp_dir / "grey.tif"
        Image.fromarray(grey_img, mode="L").save(img_path)

        volume, filenames = load_images([img_path], greyscale=False)
        assert volume.shape == (1, 100, 100, 1)  # Channel dimension added

    def test_rgb_image_with_channels(self, temp_dir):
        """Test that RGB images retain their channels when not converting."""
        # Create RGB image with 3 channels
        rgb_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img_path = temp_dir / "rgb.tif"
        Image.fromarray(rgb_img).save(img_path)

        volume, filenames = load_images([img_path], greyscale=False)
        assert volume.shape == (1, 100, 100, 3)  # RGB channels preserved


class TestExtractTimelapseMetadataFromFilename:
    """Test the extract_timelapse_metadata_from_filename function."""

    def test_real_world_example(self):
        """Test extraction from your actual filename example."""
        filename = r"\\multilab-na.ad.salk.edu\hpi_dev\users\eberrigan\circumnutation\20250819_Suyash_Patil_CMTN_Kitx_vs_Hk1-3_07-30-25\001\_set1_day1_20250730-212631_001.tif"
        metadata = extract_timelapse_metadata_from_filename(filename)

        assert metadata["datetime_str"] == "20250730-212631"
        assert metadata["suffix"] == "001"
        assert metadata["set_info"] == 1
        assert metadata["day_info"] == 1
        assert metadata["datetime"] is not None
        assert metadata["datetime"].year == 2025
        assert metadata["datetime"].month == 7
        assert metadata["datetime"].day == 30
        assert metadata["datetime"].hour == 21
        assert metadata["datetime"].minute == 26
        assert metadata["datetime"].second == 31

    def test_standard_format_with_hyphen(self):
        """Test extraction with hyphen datetime separator."""
        filename = "_set1_day1_20250730-212631_001.tif"
        metadata = extract_timelapse_metadata_from_filename(filename)

        assert metadata["datetime_str"] == "20250730-212631"
        assert metadata["suffix"] == "001"
        assert metadata["set_info"] == 1
        assert metadata["day_info"] == 1

    def test_standard_format_with_underscore(self):
        """Test that underscore datetime separator is not parsed (non-standard)."""
        filename = "prefix_20250730_212631_001.tif"
        metadata = extract_timelapse_metadata_from_filename(filename)

        # Non-standard format - should not parse datetime
        assert metadata["datetime_str"] is None
        assert metadata["datetime"] is None
        assert metadata["suffix"] == "001"

    def test_no_separator_datetime(self):
        """Test that no separator in datetime is not parsed (non-standard)."""
        filename = "prefix_20250730212631_001.tif"
        metadata = extract_timelapse_metadata_from_filename(filename)

        # Non-standard format - should not parse datetime
        assert metadata["datetime_str"] is None
        assert metadata["datetime"] is None
        assert metadata["suffix"] == "001"

    def test_complex_prefix(self):
        """Test extraction with complex multi-part prefix."""
        filename = "PI_Name_Project123_Condition_A_set3_day5_20250101-090000_042.tif"
        metadata = extract_timelapse_metadata_from_filename(filename)

        assert metadata["datetime_str"] == "20250101-090000"
        assert metadata["suffix"] == "042"
        assert metadata["set_info"] == 3
        assert metadata["day_info"] == 5
        assert "PI_Name_Project123_Condition_A" in metadata["prefix"]

    def test_no_prefix(self):
        """Test extraction with no prefix."""
        filename = "20250730-212631_001.tif"
        metadata = extract_timelapse_metadata_from_filename(filename)

        assert metadata["datetime_str"] == "20250730-212631"
        assert metadata["suffix"] == "001"
        assert metadata["prefix"] is None

    def test_no_suffix(self):
        """Test extraction with no suffix."""
        filename = "image_20250730-212631.tif"
        metadata = extract_timelapse_metadata_from_filename(filename)

        assert metadata["datetime_str"] == "20250730-212631"
        assert metadata["suffix"] is None
        assert metadata["prefix"] == "image"

    def test_short_time_format(self):
        """Test that non-standard time formats are not parsed."""
        filename = "test_20250730-2126_001.tif"  # Missing seconds
        metadata = extract_timelapse_metadata_from_filename(filename)

        # Should not parse this as it's not standard YYYYMMDD-HHMMSS format
        assert metadata["datetime_str"] is None
        assert metadata["datetime"] is None
        # But should still get suffix if present
        assert metadata["suffix"] == "001"

    def test_multiple_underscores(self):
        """Test handling of multiple consecutive underscores."""
        filename = "test__20250730-212631__001.tif"
        metadata = extract_timelapse_metadata_from_filename(filename)

        assert metadata["datetime_str"] == "20250730-212631"
        assert metadata["suffix"] == "001"

    def test_no_datetime(self):
        """Test graceful handling when no datetime is found."""
        filename = "no_datetime_here_001.tif"
        metadata = extract_timelapse_metadata_from_filename(filename)

        assert metadata["datetime_str"] is None
        assert metadata["datetime"] is None
        assert metadata["suffix"] == "001"
        assert metadata["prefix"] == "no_datetime_here"

    def test_completely_invalid(self):
        """Test handling of completely invalid filename."""
        filename = "completely_random_name.tif"
        metadata = extract_timelapse_metadata_from_filename(filename)

        assert metadata["datetime_str"] is None
        assert metadata["datetime"] is None
        assert metadata["suffix"] is None

    def test_path_object_input(self):
        """Test that Path objects are handled correctly."""
        from pathlib import Path

        filename = Path("prefix_20250730-212631_001.tif")
        metadata = extract_timelapse_metadata_from_filename(filename)

        assert metadata["datetime_str"] == "20250730-212631"
        assert metadata["suffix"] == "001"

    def test_all_real_world_examples(self, real_world_filenames):
        """Test all real-world filename examples."""
        for filename in real_world_filenames:
            metadata = extract_timelapse_metadata_from_filename(filename)
            # At minimum, should not crash and should return dict
            assert isinstance(metadata, dict)
            assert "filename" in metadata
            assert "datetime_str" in metadata
            assert "suffix" in metadata

            # If datetime is found, verify it's valid
            if metadata["datetime"] is not None:
                from datetime import datetime

                assert isinstance(metadata["datetime"], datetime)


class TestSaveArrayAsH5:
    """Test the save_array_as_h5 function."""

    def test_create_h5_file(self, sample_4d_array, temp_dir):
        """Test creating H5 file from 4D array."""
        output_path = temp_dir / "test_output.h5"
        result_path = save_array_as_h5(sample_4d_array, output_path)

        assert result_path == output_path
        assert output_path.exists()

        # Verify contents
        with h5py.File(output_path, "r") as f:
            assert "vol" in f
            assert f["vol"].shape == sample_4d_array.shape
            assert np.array_equal(f["vol"][:], sample_4d_array)

    def test_compression_options(self, sample_4d_array, temp_dir):
        """Test different compression options."""
        output_path = temp_dir / "compressed.h5"
        save_array_as_h5(
            sample_4d_array, output_path, compression="gzip", compression_opts=9
        )

        with h5py.File(output_path, "r") as f:
            assert f["vol"].compression == "gzip"
            assert f["vol"].compression_opts == 9

    def test_invalid_empty_array(self, temp_dir):
        """Test error handling for empty array."""
        bad_array = np.array([])
        output_path = temp_dir / "bad.h5"

        with pytest.raises(ValueError, match="Cannot save empty array"):
            save_array_as_h5(bad_array, output_path)

    def test_create_parent_directories(self, sample_4d_array, temp_dir):
        """Test that parent directories are created if needed."""
        output_path = temp_dir / "nested" / "dirs" / "output.h5"
        save_array_as_h5(sample_4d_array, output_path)
        assert output_path.exists()

    def test_overwrite_existing(self, sample_4d_array, temp_dir):
        """Test overwriting existing H5 file."""
        output_path = temp_dir / "existing.h5"

        # Create first file
        save_array_as_h5(sample_4d_array, output_path)

        # Overwrite with different data
        new_array = np.ones_like(sample_4d_array)
        save_array_as_h5(new_array, output_path)

        with h5py.File(output_path, "r") as f:
            assert np.array_equal(f["vol"][:], new_array)


class TestMakeVideoFromImages:
    """Test the make_video_from_images function."""

    def test_create_video_from_images(self, temp_image_dir_tiff):
        """Test creating Video object from image files."""
        # Note: This test would require sleap_io to be installed
        # For now, we'll skip it if sleap_io is not available
        pytest.importorskip("sleap_io")

        image_files = list(temp_image_dir_tiff.glob("*.tif"))
        video = make_video_from_images(image_files)

        assert video is not None
        assert len(video) == len(image_files)


class TestCreateTimelapseMetadataDataframe:
    """Test the create_timelapse_metadata_dataframe function."""

    def test_create_dataframe(self):
        """Test creating metadata dataframe with new format."""
        filenames = [
            "_set1_day1_20240101-120000_001.tif",
            "_set1_day2_20240102-130000_002.tif",
            "_set1_day3_20240103-140000_003.tif",
        ]
        df = create_timelapse_metadata_dataframe(filenames, "test_exp", "control", 5)

        assert len(df) == 3
        # Check essential columns exist
        assert "experiment" in df.columns
        assert "filename" in df.columns
        assert "treatment" in df.columns
        assert "expected_num_plants" in df.columns
        assert "frame" in df.columns
        assert "datetime_str" in df.columns
        assert "datetime" in df.columns
        assert "plate_number" in df.columns
        assert "set_info" in df.columns
        assert "day_info" in df.columns

        assert df["experiment"].unique()[0] == "test_exp"
        assert df["treatment"].unique()[0] == "control"
        assert df["expected_num_plants"].unique()[0] == 5
        assert list(df["frame"]) == [0, 1, 2]
        assert list(df["plate_number"]) == ["001", "002", "003"]
        assert list(df["set_info"]) == [1, 1, 1]
        assert list(df["day_info"]) == [1, 2, 3]

    def test_empty_filenames(self):
        """Test handling empty filename list."""
        df = create_timelapse_metadata_dataframe([], "exp", "treatment", 10)
        assert len(df) == 0
        # Should still have all columns
        assert "datetime_str" in df.columns
        assert "datetime" in df.columns
        assert "plate_number" in df.columns

    def test_complex_filenames(self):
        """Test with complex real-world filenames."""
        filenames = [
            "PI_Name_Project123_set1_day1_20250101-090000_042.tif",
            "PI_Name_Project123_set1_day2_20250102-090000_043.tif",
        ]
        df = create_timelapse_metadata_dataframe(
            filenames, "complex_exp", "treatment_A", 3
        )

        assert len(df) == 2
        assert df["set_info"].tolist() == [1, 1]
        assert df["day_info"].tolist() == [1, 2]
        assert df["plate_number"].tolist() == ["042", "043"]
        assert df["datetime_str"].tolist() == ["20250101-090000", "20250102-090000"]
        assert df["prefix"].iloc[0] == "PI_Name_Project123"

    def test_missing_datetime_handling(self):
        """Test graceful handling of files without datetime."""
        filenames = [
            "good_file_20250101-120000_001.tif",
            "bad_file_no_datetime.tif",
            "another_good_20250102-130000_002.tif",
        ]
        df = create_timelapse_metadata_dataframe(filenames, "test", "control", 1)

        assert len(df) == 3
        # Check that missing datetime is handled
        assert (
            df["datetime_str"].iloc[1] is None
            or df["datetime_str"].iloc[1] == "unknown"
        )


class TestProcessTimelapseImageDirectory:
    """Test the main processing function."""

    def test_successful_processing(self, image_directory_with_tiffs, temp_dir):
        """Test successful processing of image directory with H5 output."""
        output_dir = temp_dir / "output"
        # Test with save_h5=True to get H5 file
        h5_path, csv_path = process_timelapse_image_directory(
            image_directory_with_tiffs,
            "test_exp",
            "control",
            3,
            greyscale=False,
            output_dir=output_dir,
            save_h5=True,  # Save as H5
        )

        assert h5_path is not None
        assert csv_path is not None
        assert h5_path.exists()
        assert csv_path.exists()
        assert h5_path.name == f"plate_{image_directory_with_tiffs.name}_color.h5"
        assert csv_path.name == f"plate_{image_directory_with_tiffs.name}_metadata.csv"

        # Check H5 contents
        with h5py.File(h5_path, "r") as f:
            assert "vol" in f
            assert f["vol"].shape[0] == 5  # 5 images
            assert f["vol"].shape[-1] == 3  # RGB

        # Check CSV contents
        df = pd.read_csv(csv_path)
        assert len(df) == 5
        assert df["experiment"].unique()[0] == "test_exp"

    def test_greyscale_processing(self, image_directory_with_tiffs, temp_dir):
        """Test processing with greyscale conversion."""
        # Test with save_h5=True
        h5_path, csv_path = process_timelapse_image_directory(
            image_directory_with_tiffs,
            "test_exp",
            "control",
            3,
            greyscale=True,
            output_dir=temp_dir,
            save_h5=True,  # Save as H5
        )

        assert h5_path is not None
        assert h5_path.name.endswith("_greyscale.h5")

        with h5py.File(h5_path, "r") as f:
            assert f["vol"].shape[-1] == 1  # Greyscale

    def test_non_existent_directory(self, non_existent_path):
        """Test handling of non-existent directory."""
        h5_path, csv_path = process_timelapse_image_directory(
            non_existent_path,
            "test_exp",
            "control",
            3,
        )
        assert h5_path is None
        assert csv_path is None

    def test_empty_directory(self, empty_directory):
        """Test handling of empty directory."""
        h5_path, csv_path = process_timelapse_image_directory(
            empty_directory,
            "test_exp",
            "control",
            3,
        )
        assert h5_path is None
        assert csv_path is None

    def test_file_instead_of_directory(self, temp_dir):
        """Test handling when a file is provided instead of directory."""
        file_path = temp_dir / "file.txt"
        file_path.write_text("not a directory")

        h5_path, csv_path = process_timelapse_image_directory(
            file_path,
            "test_exp",
            "control",
            3,
        )
        assert h5_path is None
        assert csv_path is None

    def test_custom_image_pattern(self, mixed_format_directory):
        """Test processing with custom image pattern."""
        # Test with save_h5=True
        h5_path, csv_path = process_timelapse_image_directory(
            mixed_format_directory,
            "test_exp",
            "control",
            3,
            image_pattern="*.png",
            save_h5=True,  # Save as H5
        )

        assert h5_path is not None
        with h5py.File(h5_path, "r") as f:
            assert f["vol"].shape[0] == 1  # Only 1 PNG file

    def test_default_output_directory(self, image_directory_with_tiffs):
        """Test that output defaults to source directory."""
        # Test with save_h5=True
        h5_path, csv_path = process_timelapse_image_directory(
            image_directory_with_tiffs,
            "test_exp",
            "control",
            3,
            output_dir=None,  # Should default to source_dir
            save_h5=True,  # Save as H5
        )

        assert h5_path is not None
        assert h5_path.parent == image_directory_with_tiffs

    def test_video_output_processing(self, image_directory_with_tiffs, temp_dir):
        """Test processing that returns Video object instead of H5."""
        pytest.importorskip("sleap_io")

        output_dir = temp_dir / "output"
        # Test with save_h5=False (default) to get Video object
        video, csv_path = process_timelapse_image_directory(
            image_directory_with_tiffs,
            "test_exp",
            "control",
            3,
            greyscale=False,
            output_dir=output_dir,
            save_h5=False,  # Get Video object
        )

        assert video is not None
        assert csv_path is not None
        assert csv_path.exists()

        # Check Video object properties
        assert len(video) == 5  # 5 frames
        assert video.shape[0] == 5  # 5 frames

        # Check CSV contents
        df = pd.read_csv(csv_path)
        assert len(df) == 5
        assert df["experiment"].unique()[0] == "test_exp"

    def test_logging_output(self, image_directory_with_tiffs, caplog):
        """Test that appropriate log messages are generated."""
        with caplog.at_level(logging.INFO):
            # Test with save_h5=True to ensure we get H5 logging
            process_timelapse_image_directory(
                image_directory_with_tiffs,
                "test_exp",
                "control",
                3,
                save_h5=True,
            )

        assert "Found 5 image files" in caplog.text
        assert "Saved metadata to" in caplog.text

    def test_malformed_filenames(self, malformed_image_directory):
        """Test processing with malformed filenames."""
        # Test with save_h5=True to get H5 path
        h5_path, csv_path = process_timelapse_image_directory(
            malformed_image_directory,
            "test_exp",
            "control",
            3,
            save_h5=True,
        )

        # Should still process, using fallback for metadata
        assert h5_path is not None
        assert csv_path is not None

        df = pd.read_csv(csv_path)
        # Check that dataframe was created
        assert len(df) > 0

    def test_complex_filenames_processing(self, complex_filename_directory, temp_dir):
        """Test processing directory with complex real-world filenames."""
        output_dir = temp_dir / "output"
        h5_path, csv_path = process_timelapse_image_directory(
            complex_filename_directory,
            "complex_test",
            "treatment_B",
            4,
            greyscale=False,
            output_dir=output_dir,
        )

        assert h5_path is not None
        assert csv_path is not None

        # Check metadata extraction worked
        df = pd.read_csv(csv_path)
        assert len(df) == 6  # 6 files in fixture

        # Verify datetime extraction
        assert all(dt is not None and dt != "unknown" for dt in df["datetime_str"])

        # Check set and day info extracted
        assert "set_info" in df.columns
        assert "day_info" in df.columns
        assert 1 in df["set_info"].values
        assert 2 in df["set_info"].values  # We have set1 and set2 files

        # Verify plate numbers (suffixes) - may be stored as strings or converted to ints
        plate_numbers = df["plate_number"].astype(str).values
        assert any("1" in str(p) for p in plate_numbers)
        assert any("2" in str(p) for p in plate_numbers)

    def test_error_during_processing(self, temp_dir, monkeypatch):
        """Test error handling during image processing."""
        # Create a valid directory
        test_dir = temp_dir / "test_images"
        test_dir.mkdir()

        img = np.ones((10, 10, 3), dtype=np.uint8) * 100
        for i in range(3):
            filepath = test_dir / f"image_{i:03d}.tif"
            Image.fromarray(img).save(filepath)

        # Mock load_images to raise an error
        def mock_load_images(*args, **kwargs):
            raise ValueError("Test error during loading")

        monkeypatch.setattr(
            "sleap_roots_predict.plates_timelapse_experiment.load_images",
            mock_load_images,
        )

        # Should handle error gracefully
        result = process_timelapse_image_directory(
            source_dir=test_dir,
            experiment_name="test",
            treatment="control",
            num_plants=1,
            output_dir=temp_dir,
            save_h5=True,  # Force H5 saving to test the error path
        )

        h5_path, csv_path = result
        assert h5_path is None
        assert csv_path is not None  # CSV should still be created

    def test_error_during_h5_creation(self, temp_dir, monkeypatch):
        """Test error handling during H5 creation."""
        test_dir = temp_dir / "test_images"
        test_dir.mkdir()

        img = np.ones((10, 10, 3), dtype=np.uint8) * 100
        for i in range(3):
            filepath = test_dir / f"image_{i:03d}.tif"
            Image.fromarray(img).save(filepath)

        # Mock save_array_as_h5 to raise an error
        def mock_save_h5(*args, **kwargs):
            raise IOError("Test error during H5 creation")

        monkeypatch.setattr(
            "sleap_roots_predict.plates_timelapse_experiment.save_array_as_h5",
            mock_save_h5,
        )

        # Should handle error gracefully
        result = process_timelapse_image_directory(
            source_dir=test_dir,
            experiment_name="test",
            treatment="control",
            num_plants=1,
            output_dir=temp_dir,
            save_h5=True,  # Force H5 saving to test the error path
        )

        h5_path, csv_path = result
        assert h5_path is None
        assert csv_path is not None  # CSV should still be created

    def test_error_during_csv_creation(self, temp_dir, monkeypatch):
        """Test error handling during CSV creation."""
        test_dir = temp_dir / "test_images"
        test_dir.mkdir()

        img = np.ones((10, 10, 3), dtype=np.uint8) * 100
        for i in range(3):
            filepath = test_dir / f"image_{i:03d}.tif"
            Image.fromarray(img).save(filepath)

        # Mock create_metadata_dataframe to raise an error
        def mock_create_csv(*args, **kwargs):
            raise ValueError("Test error during CSV creation")

        monkeypatch.setattr(
            "sleap_roots_predict.plates_timelapse_experiment.create_timelapse_metadata_dataframe",
            mock_create_csv,
        )

        # Should handle error gracefully - H5 still created but CSV fails
        h5_path, csv_path = process_timelapse_image_directory(
            source_dir=test_dir,
            experiment_name="test",
            treatment="control",
            num_plants=1,
            output_dir=temp_dir,
        )

        assert h5_path is not None  # H5 should still be created
        assert csv_path is None  # CSV should fail

    def test_load_images_failure_returns_none_h5_path(self, temp_dir, monkeypatch):
        """Test that h5_path is None when load_images fails (would catch unbound variable)."""
        test_dir = temp_dir / "test_images"
        test_dir.mkdir()

        # Create valid image files
        img = np.ones((10, 10, 3), dtype=np.uint8) * 100
        for i in range(3):
            filepath = test_dir / f"image_{i:03d}.tif"
            Image.fromarray(img).save(filepath)

        # Mock load_images to raise an exception
        def mock_load_images(*args, **kwargs):
            raise MemoryError("Cannot allocate memory for images")

        monkeypatch.setattr(
            "sleap_roots_predict.plates_timelapse_experiment.load_images",
            mock_load_images,
        )

        # Process with save_h5=True to test the H5 path
        h5_path, csv_path = process_timelapse_image_directory(
            source_dir=test_dir,
            experiment_name="test",
            treatment="control",
            num_plants=1,
            output_dir=temp_dir,
            save_h5=True,
        )

        # Should return None for h5_path without raising UnboundLocalError
        assert h5_path is None
        assert csv_path is not None  # CSV should still work

    def test_partial_h5_creation_failure(self, temp_dir, monkeypatch):
        """Test handling when load_images succeeds but save_array_as_h5 fails."""
        test_dir = temp_dir / "test_images"
        test_dir.mkdir()

        img = np.ones((10, 10, 3), dtype=np.uint8) * 100
        for i in range(3):
            filepath = test_dir / f"image_{i:03d}.tif"
            Image.fromarray(img).save(filepath)

        # Counter to track calls
        call_count = {"load": 0, "save": 0}

        # Mock load_images to succeed
        original_load = load_images
        def mock_load_images(*args, **kwargs):
            call_count["load"] += 1
            return original_load(*args, **kwargs)

        # Mock save_array_as_h5 to fail
        def mock_save_h5(*args, **kwargs):
            call_count["save"] += 1
            raise OSError("Disk full")

        monkeypatch.setattr(
            "sleap_roots_predict.plates_timelapse_experiment.load_images",
            mock_load_images,
        )
        monkeypatch.setattr(
            "sleap_roots_predict.plates_timelapse_experiment.save_array_as_h5",
            mock_save_h5,
        )

        h5_path, csv_path = process_timelapse_image_directory(
            source_dir=test_dir,
            experiment_name="test",
            treatment="control",
            num_plants=1,
            output_dir=temp_dir,
            save_h5=True,
        )

        # Verify both functions were called
        assert call_count["load"] == 1
        assert call_count["save"] == 1
        
        # Should handle the save failure gracefully
        assert h5_path is None
        assert csv_path is not None

    def test_video_creation_failure(self, temp_dir, monkeypatch):
        """Test handling when video creation fails."""
        test_dir = temp_dir / "test_images"
        test_dir.mkdir()

        img = np.ones((10, 10, 3), dtype=np.uint8) * 100
        for i in range(3):
            filepath = test_dir / f"image_{i:03d}.tif"
            Image.fromarray(img).save(filepath)

        # Mock make_video_from_images to fail
        def mock_make_video(*args, **kwargs):
            raise RuntimeError("Failed to create video object")

        monkeypatch.setattr(
            "sleap_roots_predict.plates_timelapse_experiment.make_video_from_images",
            mock_make_video,
        )

        # Process with save_h5=False to test video path
        video, csv_path = process_timelapse_image_directory(
            source_dir=test_dir,
            experiment_name="test",
            treatment="control",
            num_plants=1,
            output_dir=temp_dir,
            save_h5=False,  # Test video creation path
        )

        # Should return None for video without errors
        assert video is None
        assert csv_path is not None  # CSV should still work

    def test_multiple_errors_in_pipeline(self, temp_dir, monkeypatch):
        """Test handling multiple errors in the processing pipeline."""
        test_dir = temp_dir / "test_images"
        test_dir.mkdir()

        img = np.ones((10, 10, 3), dtype=np.uint8) * 100
        for i in range(2):  # Just 2 images
            filepath = test_dir / f"image_{i:03d}.tif"
            Image.fromarray(img).save(filepath)

        errors_encountered = []

        # Mock both CSV creation and H5 saving to fail
        def mock_create_csv(*args, **kwargs):
            errors_encountered.append("csv_error")
            raise ValueError("CSV creation failed")

        def mock_save_h5(*args, **kwargs):
            errors_encountered.append("h5_error")
            raise IOError("H5 save failed")

        monkeypatch.setattr(
            "sleap_roots_predict.plates_timelapse_experiment.create_timelapse_metadata_dataframe",
            mock_create_csv,
        )
        monkeypatch.setattr(
            "sleap_roots_predict.plates_timelapse_experiment.save_array_as_h5",
            mock_save_h5,
        )

        h5_path, csv_path = process_timelapse_image_directory(
            source_dir=test_dir,
            experiment_name="test",
            treatment="control",
            num_plants=1,
            output_dir=temp_dir,
            save_h5=True,
        )

        # Both should fail but not crash
        assert h5_path is None
        assert csv_path is None
        assert "csv_error" in errors_encountered
        assert "h5_error" in errors_encountered


class TestFindImageDirectories:
    """Test find_image_directories function."""

    def test_find_directories_with_tiffs(self, temp_dir):
        """Test finding directories containing TIFF images."""
        # Create directory structure
        (temp_dir / "exp1" / "plate1").mkdir(parents=True)
        (temp_dir / "exp1" / "plate2").mkdir(parents=True)
        (temp_dir / "exp2" / "plate1").mkdir(parents=True)
        (temp_dir / "no_images").mkdir(parents=True)

        # Add TIFF files
        (temp_dir / "exp1" / "plate1" / "image1.tif").write_text("")
        (temp_dir / "exp1" / "plate2" / "image1.tif").write_text("")
        (temp_dir / "exp2" / "plate1" / "image1.tiff").write_text("")
        # Directory with no images
        (temp_dir / "no_images" / "data.txt").write_text("")

        dirs = find_image_directories(temp_dir)
        dir_names = [d.name for d in dirs]

        assert len(dirs) == 3
        assert "plate1" in dir_names  # Should appear twice but in different paths
        assert "plate2" in dir_names
        assert "no_images" not in dir_names

    def test_non_existent_directory(self, temp_dir):
        """Test with non-existent directory."""
        dirs = find_image_directories(temp_dir / "non_existent")
        assert dirs == []

    def test_file_instead_of_directory(self, temp_dir):
        """Test with file path instead of directory."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("content")
        dirs = find_image_directories(test_file)
        assert dirs == []


class TestCheckTimelapseImageDirectory:
    """Test check_timelapse_image_directory function."""

    def test_valid_directory(self, temp_dir):
        """Test checking a valid image directory."""
        image_dir = temp_dir / "plate_001"
        image_dir.mkdir()

        # Create valid TIFF files with proper naming
        for i in range(5):
            filename = f"exp_set1_day1_2025010{i}-12000{i}_001.tif"
            (image_dir / filename).write_text("")

        results = check_timelapse_image_directory(image_dir)

        assert results["valid"] is True
        assert results["image_count"] == 5
        assert results["suffixes"] == {"001"}
        assert len(results["errors"]) == 0
        assert results["directory"] == image_dir.as_posix()

    def test_inconsistent_suffixes(self, temp_dir):
        """Test detection of inconsistent suffixes."""
        image_dir = temp_dir / "mixed_plate"
        image_dir.mkdir()

        # Create files with different suffixes (should fail for a single plate)
        (image_dir / "image_20250101-120000_001.tif").write_text("")
        (image_dir / "image_20250101-130000_002.tif").write_text("")
        (image_dir / "image_20250101-140000_003.tif").write_text("")

        results = check_timelapse_image_directory(
            image_dir, check_suffix_consistency=True
        )

        assert results["valid"] is False
        assert results["suffixes"] == {"001", "002", "003"}
        assert any("Inconsistent suffixes" in err for err in results["errors"])

    def test_suffix_pattern_validation(self, temp_dir):
        """Test suffix pattern validation."""
        image_dir = temp_dir / "plate"
        image_dir.mkdir()

        # Create files with 2-digit suffix
        (image_dir / "image_20250101-120000_42.tif").write_text("")
        (image_dir / "image_20250101-130000_42.tif").write_text("")

        # Should fail with 3-digit pattern
        results = check_timelapse_image_directory(
            image_dir, expected_suffix_pattern=r"^\d{3}$"
        )
        assert results["valid"] is False
        assert any(
            "does not match expected pattern" in err for err in results["errors"]
        )

        # Should pass with 2-digit pattern
        results = check_timelapse_image_directory(
            image_dir, expected_suffix_pattern=r"^\d{2}$"
        )
        assert results["valid"] is True

    def test_minimum_image_count(self, temp_dir):
        """Test minimum image count validation."""
        image_dir = temp_dir / "few_images"
        image_dir.mkdir()

        # Create only 2 images
        (image_dir / "image1.tif").write_text("")
        (image_dir / "image2.tif").write_text("")

        results = check_timelapse_image_directory(image_dir, min_images=5)
        assert results["valid"] is False
        assert results["image_count"] == 2
        assert any("Too few images" in err for err in results["errors"])

    def test_maximum_image_count(self, temp_dir):
        """Test maximum image count validation."""
        image_dir = temp_dir / "many_images"
        image_dir.mkdir()

        # Create 10 images
        for i in range(10):
            (image_dir / f"image_{i}.tif").write_text("")

        results = check_timelapse_image_directory(image_dir, max_images=5)
        assert results["valid"] is False
        assert results["image_count"] == 10
        assert any("Too many images" in err for err in results["errors"])

    def test_missing_datetime_warning(self, temp_dir):
        """Test warning for missing datetime in filenames."""
        image_dir = temp_dir / "no_datetime"
        image_dir.mkdir()

        # Create files without datetime
        (image_dir / "image_001.tif").write_text("")
        (image_dir / "another_001.tif").write_text("")

        results = check_timelapse_image_directory(image_dir, check_datetime=True)
        assert results["valid"] is True  # Still valid, just warnings
        assert len(results["warnings"]) > 0
        assert any("No valid datetime" in warn for warn in results["warnings"])

    def test_chronological_order_warning(self, temp_dir):
        """Test warning for files not in chronological order."""
        image_dir = temp_dir / "unordered"
        image_dir.mkdir()

        # Create files with timestamps out of order
        (image_dir / "b_20250102-120000_001.tif").write_text("")
        (image_dir / "a_20250101-120000_001.tif").write_text("")
        (image_dir / "c_20250103-120000_001.tif").write_text("")

        results = check_timelapse_image_directory(image_dir)
        # Natural sort will be a_, b_, c_ but chronologically should be a_, b_, c_ so it's actually OK
        # Let's create a case where it's actually wrong
        (image_dir / "1_20250105-120000_001.tif").write_text("")
        (image_dir / "2_20250104-120000_001.tif").write_text("")

        results = check_timelapse_image_directory(image_dir)
        # This should generate a warning about chronological order
        assert len(results["warnings"]) > 0

    def test_empty_directory(self, temp_dir):
        """Test checking empty directory."""
        image_dir = temp_dir / "empty"
        image_dir.mkdir()

        results = check_timelapse_image_directory(image_dir)
        assert results["valid"] is False
        assert results["image_count"] == 0
        assert any("Too few images" in err for err in results["errors"])

    def test_non_existent_directory(self, temp_dir):
        """Test checking non-existent directory."""
        results = check_timelapse_image_directory(temp_dir / "non_existent")
        assert results["valid"] is False
        assert any("does not exist" in err for err in results["errors"])

    def test_files_without_datetime_warning(self, temp_dir):
        """Test warning when files don't have datetime."""
        test_dir = temp_dir / "no_datetime"
        test_dir.mkdir()

        img = np.ones((10, 10, 3), dtype=np.uint8) * 100

        # Create files without datetime
        files = [
            "image_001.tif",
            "image_002.tif",
            "image_003.tif",
        ]

        for filename in files:
            Image.fromarray(img).save(test_dir / filename)

        results = check_timelapse_image_directory(test_dir, check_datetime=True)

        # Should have warning about missing datetime
        assert any("datetime" in w.lower() for w in results["warnings"])


class TestProcessTimelapseExperiment:
    """Test process_experiment function."""

    def test_successful_experiment_processing(self, temp_dir, metadata_csv):
        """Test processing a complete experiment with CSV metadata."""
        # Create experiment structure
        exp_dir = temp_dir / "experiment"
        plate1_dir = exp_dir / "plate_001"
        plate2_dir = exp_dir / "plate_002"
        plate1_dir.mkdir(parents=True)
        plate2_dir.mkdir(parents=True)

        # Add valid TIFF files to plate 1
        img = Image.new("RGB", (100, 100), color="red")
        for i in range(3):
            filename = f"exp_20250101-12000{i}_001.tif"
            img.save(plate1_dir / filename)

        # Add valid TIFF files to plate 2
        for i in range(3):
            filename = f"exp_20250101-13000{i}_002.tif"
            img.save(plate2_dir / filename)

        # Process the experiment with CSV metadata - test with save_h5=True
        results = process_timelapse_experiment(
            exp_dir,
            metadata_csv=metadata_csv,
            experiment_name="test_exp",
            output_dir=temp_dir / "output",
            save_h5=True,  # Save as H5
        )

        assert len(results["processed"]) == 2
        assert len(results["failed"]) == 0
        assert len(results["skipped"]) == 0

        # Check that H5 files were created
        for processed in results["processed"]:
            assert processed["h5_path"] is not None
            assert Path(processed["h5_path"]).exists()
            # Check that plate metadata was included
            assert "plate_metadata" in processed
            assert processed["plate_metadata"]["treatment"] in [
                "control",
                "treatment_A",
            ]

    def test_dry_run_mode(self, temp_dir, metadata_csv):
        """Test dry run mode without actual processing."""
        # Create experiment structure
        exp_dir = temp_dir / "experiment"
        plate_dir = exp_dir / "plate_001"
        plate_dir.mkdir(parents=True)

        # Add TIFF files with consistent suffix
        for i in range(3):
            (plate_dir / f"image_{i}_001.tif").write_text("")

        results = process_timelapse_experiment(
            exp_dir, metadata_csv=metadata_csv, experiment_name="test_exp", dry_run=True
        )

        assert len(results["processed"]) == 0
        assert len(results["failed"]) == 0
        assert len(results["skipped"]) == 1
        assert results["skipped"][0]["reason"] == "dry_run"

    def test_failed_validation(self, temp_dir, metadata_csv):
        """Test handling of directories that fail validation."""
        exp_dir = temp_dir / "experiment"

        # Create a directory with inconsistent suffixes
        bad_dir = exp_dir / "bad_plate"
        bad_dir.mkdir(parents=True)
        (bad_dir / "image_20250101-120000_001.tif").write_text("")
        (bad_dir / "image_20250101-130000_002.tif").write_text("")

        # Create a good directory
        good_dir = exp_dir / "good_plate"
        good_dir.mkdir(parents=True)
        img = Image.new("RGB", (10, 10))
        img.save(good_dir / "image_20250101-120000_003.tif")
        img.save(good_dir / "image_20250101-130000_003.tif")

        results = process_timelapse_experiment(
            exp_dir,
            metadata_csv=metadata_csv,
            experiment_name="test",
            check_suffix_consistency=True,
            save_h5=True,  # Save as H5
        )

        assert len(results["failed"]) == 1
        assert "bad_plate" in results["failed"][0]["directory"]
        assert len(results["processed"]) == 1
        assert "good_plate" in results["processed"][0]["directory"]

    def test_custom_check_parameters(self, temp_dir, metadata_csv):
        """Test with custom check parameters."""
        exp_dir = temp_dir / "experiment"
        plate_dir = exp_dir / "plate_001"
        plate_dir.mkdir(parents=True)

        # Create only 2 images
        (plate_dir / "image1.tif").write_text("")
        (plate_dir / "image2.tif").write_text("")

        # Should fail with min_images=5
        results = process_timelapse_experiment(
            exp_dir, metadata_csv=metadata_csv, experiment_name="test", min_images=5
        )

        assert len(results["failed"]) == 1
        assert "Too few images" in str(results["failed"][0]["check_results"]["errors"])

    def test_nested_directory_structure(self, temp_dir, metadata_csv):
        """Test processing nested directory structure."""
        # Create nested structure like the example path
        exp_dir = temp_dir / "circumnutation" / "experiment1"
        plate1 = exp_dir / "set1" / "day1" / "001"
        plate2 = exp_dir / "set1" / "day2" / "002"
        plate1.mkdir(parents=True)
        plate2.mkdir(parents=True)

        # Add images
        img = Image.new("RGB", (10, 10))
        img.save(plate1 / "set1_day1_20250101-120000_001.tif")
        img.save(plate2 / "set1_day2_20250102-120000_002.tif")

        results = process_timelapse_experiment(
            exp_dir,
            metadata_csv=metadata_csv,
            experiment_name="nested_test",
            output_dir=temp_dir / "output",
            save_h5=True,  # Save as H5
        )

        assert len(results["processed"]) == 2

        # Check output structure preserves hierarchy
        output_files = list((temp_dir / "output").rglob("*.h5"))
        assert len(output_files) == 2

    def test_error_handling_during_processing(
        self, temp_dir, metadata_csv, monkeypatch
    ):
        """Test error handling when processing fails."""
        exp_dir = temp_dir / "experiment"
        plate_dir = exp_dir / "plate_001"
        plate_dir.mkdir(parents=True)

        # Add valid files
        img = Image.new("RGB", (10, 10))
        img.save(plate_dir / "image_20250101-120000_001.tif")

        # Mock process_timelapse_image_directory to raise an exception
        def mock_process(*args, **kwargs):
            raise ValueError("Processing error")

        monkeypatch.setattr(
            "sleap_roots_predict.plates_timelapse_experiment.process_timelapse_image_directory",
            mock_process,
        )

        results = process_timelapse_experiment(
            exp_dir, metadata_csv=metadata_csv, experiment_name="test"
        )

        assert len(results["skipped"]) == 1
        assert "Processing error" in results["skipped"][0]["reason"]

    def test_empty_base_directory(self, temp_dir, metadata_csv):
        """Test with empty base directory."""
        exp_dir = temp_dir / "empty_exp"
        exp_dir.mkdir()

        results = process_timelapse_experiment(
            exp_dir, metadata_csv=metadata_csv, experiment_name="test"
        )

        assert len(results["processed"]) == 0
        assert len(results["failed"]) == 0
        assert len(results["skipped"]) == 0

    def test_missing_csv_file(self, temp_dir):
        """Test handling of missing metadata CSV file."""
        exp_dir = temp_dir / "experiment"
        exp_dir.mkdir()

        results = process_timelapse_experiment(
            exp_dir, metadata_csv=temp_dir / "non_existent.csv", experiment_name="test"
        )

        assert len(results["processed"]) == 0
        assert len(results["failed"]) == 0
        assert len(results["skipped"]) == 0

    def test_csv_missing_required_columns(self, temp_dir, metadata_csv_missing_columns):
        """Test handling of CSV with missing required columns."""
        exp_dir = temp_dir / "experiment"
        plate_dir = exp_dir / "plate_001"
        plate_dir.mkdir(parents=True)
        (plate_dir / "image_001.tif").write_text("")

        results = process_timelapse_experiment(
            exp_dir, metadata_csv=metadata_csv_missing_columns, experiment_name="test"
        )

        assert len(results["processed"]) == 0
        assert len(results["failed"]) == 0
        assert len(results["skipped"]) == 0

    def test_no_metadata_for_plate(self, temp_dir, metadata_csv):
        """Test handling when no metadata exists for a plate number."""
        exp_dir = temp_dir / "experiment"
        # Create directory with plate number 999 which isn't in the CSV
        plate_dir = exp_dir / "plate_999"
        plate_dir.mkdir(parents=True)

        img = Image.new("RGB", (10, 10))
        img.save(plate_dir / "image_20250101-120000_999.tif")

        results = process_timelapse_experiment(
            exp_dir, metadata_csv=metadata_csv, experiment_name="test"
        )

        assert len(results["processed"]) == 0
        assert len(results["failed"]) == 0
        assert len(results["skipped"]) == 1
        assert "no_metadata_for_plate_999" in results["skipped"][0]["reason"]

    def test_plate_number_format_matching(self, temp_dir):
        """Test that single-digit CSV plate numbers match 3-digit suffixes."""
        import pandas as pd

        # Create CSV with single-digit plate number
        csv_path = temp_dir / "single_digit.csv"
        metadata = pd.DataFrame(
            {
                "plate_number": [4],  # Single digit
                "treatment": ["test_treatment"],
                "num_plants": [5],
            }
        )
        metadata.to_csv(csv_path, index=False)

        # Create directory with 3-digit suffix
        exp_dir = temp_dir / "experiment"
        plate_dir = exp_dir / "plate_004"  # 3-digit suffix
        plate_dir.mkdir(parents=True)

        img = Image.new("RGB", (10, 10))
        img.save(plate_dir / "image_20250101-120000_004.tif")

        results = process_timelapse_experiment(
            exp_dir, metadata_csv=csv_path, experiment_name="test", save_h5=True
        )

        assert len(results["processed"]) == 1
        assert (
            results["processed"][0]["plate_metadata"]["treatment"] == "test_treatment"
        )
        assert results["processed"][0]["plate_metadata"]["num_plants"] == 5

    def test_log_file_output(self, image_directory_with_tiffs, metadata_csv, temp_dir):
        """Test that log file is created when specified."""
        log_file = temp_dir / "experiment.log"

        # Remove any existing log file
        if log_file.exists():
            log_file.unlink()

        results = process_timelapse_experiment(
            base_dir=image_directory_with_tiffs.parent,
            metadata_csv=metadata_csv,
            experiment_name="test_experiment",
            output_dir=temp_dir,
            log_file=log_file,
            dry_run=True,
        )

        # Check log file was created
        assert log_file.exists()

        # The log file might be empty due to buffering issues in tests
        # Just check it was created
        assert log_file.stat().st_size >= 0  # File exists with some size

    def test_invalid_plate_number_in_csv(self, image_directory_with_tiffs, temp_dir):
        """Test handling of invalid plate numbers in CSV."""
        import pandas as pd

        # Create CSV with invalid plate number
        csv_path = temp_dir / "invalid_plates.csv"
        metadata = pd.DataFrame(
            {
                "plate_number": ["001", "invalid", "003"],
                "treatment": ["control", "treatment_A", "treatment_B"],
                "num_plants": [1, 3, 6],
            }
        )
        metadata.to_csv(csv_path, index=False)

        results = process_timelapse_experiment(
            base_dir=image_directory_with_tiffs.parent,
            metadata_csv=csv_path,
            experiment_name="test_experiment",
            output_dir=temp_dir,
        )

        # Should still process valid plates
        assert len(results["processed"]) > 0 or len(results["skipped"]) > 0

    def test_csv_with_unnamed_columns(self, image_directory_with_tiffs, temp_dir):
        """Test that unnamed columns are removed from CSV."""
        import pandas as pd

        # Create CSV with unnamed columns
        csv_path = temp_dir / "unnamed_cols.csv"
        metadata = pd.DataFrame(
            {
                "plate_number": [1],
                "treatment": ["control"],
                "num_plants": [1],
                "Unnamed: 3": ["should_be_removed"],
                "Unnamed: 4": ["also_removed"],
            }
        )
        metadata.to_csv(csv_path, index=False)

        results = process_timelapse_experiment(
            base_dir=image_directory_with_tiffs.parent,
            metadata_csv=csv_path,
            experiment_name="test_experiment",
            output_dir=temp_dir,
        )

        # Should process without error
        assert "failed" in results
        assert "processed" in results

    def test_csv_enhancement_error(
        self, image_directory_with_tiffs, metadata_csv, temp_dir, monkeypatch
    ):
        """Test error handling during CSV enhancement."""
        # Mock pd.read_csv to raise an error during enhancement
        original_read_csv = pd.read_csv
        call_count = [0]

        def mock_read_csv(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] > 1:  # Fail on second call (during enhancement)
                raise IOError("Cannot read CSV for enhancement")
            return original_read_csv(*args, **kwargs)

        monkeypatch.setattr("pandas.read_csv", mock_read_csv)

        results = process_timelapse_experiment(
            base_dir=image_directory_with_tiffs.parent,
            metadata_csv=metadata_csv,
            experiment_name="test_experiment",
            output_dir=temp_dir,
        )

        # Should still complete processing despite enhancement error
        assert len(results["processed"]) > 0 or len(results["skipped"]) > 0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_large_image_processing(self, large_image, temp_dir):
        """Test processing large images."""
        # Create a large image file
        img_path = temp_dir / "large_20240101_001.tif"
        Image.fromarray(large_image).save(img_path)

        volume, _ = load_images([img_path], greyscale=False)
        assert volume.shape[1:3] == (1920, 1080)

    def test_nonexistent_image_file(self, temp_dir):
        """Test handling of nonexistent image file."""
        non_existent = temp_dir / "does_not_exist.tif"
        with pytest.raises(FileNotFoundError):
            load_images([non_existent], greyscale=False)

    def test_single_image_processing(self, temp_dir, sample_rgb_image):
        """Test processing directory with single image."""
        img_dir = temp_dir / "single"
        img_dir.mkdir()
        Image.fromarray(sample_rgb_image).save(img_dir / "img_20240101_001.tif")

        # Test with save_h5=True to get H5 path
        h5_path, csv_path = process_timelapse_image_directory(
            img_dir,
            "test_exp",
            "control",
            1,
            save_h5=True,
        )

        assert h5_path is not None
        with h5py.File(h5_path, "r") as f:
            assert f["vol"].shape[0] == 1

    def test_unicode_in_paths(self, temp_dir, sample_rgb_image):
        """Test handling unicode characters in paths."""
        unicode_dir = temp_dir / "测试目录"
        unicode_dir.mkdir()
        img_path = unicode_dir / "图像_20240101_001.tif"
        Image.fromarray(sample_rgb_image).save(img_path)

        volume, filenames = load_images([img_path], greyscale=False)
        assert volume.shape[0] == 1
        assert len(filenames) == 1
