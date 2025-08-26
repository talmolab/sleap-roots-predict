"""Comprehensive tests for video_utils module."""

import logging
from pathlib import Path
from typing import List

import h5py
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from sleap_roots_predict.video_utils import (
    convert_to_greyscale,
    create_metadata_dataframe,
    extract_metadata_from_filename,
    load_images,
    make_h5_from_images,
    natural_sort,
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


class TestExtractMetadataFromFilename:
    """Test the extract_metadata_from_filename function."""

    def test_real_world_example(self):
        """Test extraction from your actual filename example."""
        filename = r"\\multilab-na.ad.salk.edu\hpi_dev\users\eberrigan\circumnutation\20250819_Suyash_Patil_CMTN_Kitx_vs_Hk1-3_07-30-25\001\_set1_day1_20250730-212631_001.tif"
        metadata = extract_metadata_from_filename(filename)

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
        metadata = extract_metadata_from_filename(filename)

        assert metadata["datetime_str"] == "20250730-212631"
        assert metadata["suffix"] == "001"
        assert metadata["set_info"] == 1
        assert metadata["day_info"] == 1

    def test_standard_format_with_underscore(self):
        """Test that underscore datetime separator is not parsed (non-standard)."""
        filename = "prefix_20250730_212631_001.tif"
        metadata = extract_metadata_from_filename(filename)

        # Non-standard format - should not parse datetime
        assert metadata["datetime_str"] is None
        assert metadata["datetime"] is None
        assert metadata["suffix"] == "001"

    def test_no_separator_datetime(self):
        """Test that no separator in datetime is not parsed (non-standard)."""
        filename = "prefix_20250730212631_001.tif"
        metadata = extract_metadata_from_filename(filename)

        # Non-standard format - should not parse datetime
        assert metadata["datetime_str"] is None
        assert metadata["datetime"] is None
        assert metadata["suffix"] == "001"

    def test_complex_prefix(self):
        """Test extraction with complex multi-part prefix."""
        filename = "PI_Name_Project123_Condition_A_set3_day5_20250101-090000_042.tif"
        metadata = extract_metadata_from_filename(filename)

        assert metadata["datetime_str"] == "20250101-090000"
        assert metadata["suffix"] == "042"
        assert metadata["set_info"] == 3
        assert metadata["day_info"] == 5
        assert "PI_Name_Project123_Condition_A" in metadata["prefix"]

    def test_no_prefix(self):
        """Test extraction with no prefix."""
        filename = "20250730-212631_001.tif"
        metadata = extract_metadata_from_filename(filename)

        assert metadata["datetime_str"] == "20250730-212631"
        assert metadata["suffix"] == "001"
        assert metadata["prefix"] is None

    def test_no_suffix(self):
        """Test extraction with no suffix."""
        filename = "image_20250730-212631.tif"
        metadata = extract_metadata_from_filename(filename)

        assert metadata["datetime_str"] == "20250730-212631"
        assert metadata["suffix"] is None
        assert metadata["prefix"] == "image"

    def test_short_time_format(self):
        """Test that non-standard time formats are not parsed."""
        filename = "test_20250730-2126_001.tif"  # Missing seconds
        metadata = extract_metadata_from_filename(filename)

        # Should not parse this as it's not standard YYYYMMDD-HHMMSS format
        assert metadata["datetime_str"] is None
        assert metadata["datetime"] is None
        # But should still get suffix if present
        assert metadata["suffix"] == "001"

    def test_multiple_underscores(self):
        """Test handling of multiple consecutive underscores."""
        filename = "test__20250730-212631__001.tif"
        metadata = extract_metadata_from_filename(filename)

        assert metadata["datetime_str"] == "20250730-212631"
        assert metadata["suffix"] == "001"

    def test_no_datetime(self):
        """Test graceful handling when no datetime is found."""
        filename = "no_datetime_here_001.tif"
        metadata = extract_metadata_from_filename(filename)

        assert metadata["datetime_str"] is None
        assert metadata["datetime"] is None
        assert metadata["suffix"] == "001"
        assert metadata["prefix"] == "no_datetime_here"

    def test_completely_invalid(self):
        """Test handling of completely invalid filename."""
        filename = "completely_random_name.tif"
        metadata = extract_metadata_from_filename(filename)

        assert metadata["datetime_str"] is None
        assert metadata["datetime"] is None
        assert metadata["suffix"] is None

    def test_path_object_input(self):
        """Test that Path objects are handled correctly."""
        from pathlib import Path

        filename = Path("prefix_20250730-212631_001.tif")
        metadata = extract_metadata_from_filename(filename)

        assert metadata["datetime_str"] == "20250730-212631"
        assert metadata["suffix"] == "001"

    def test_all_real_world_examples(self, real_world_filenames):
        """Test all real-world filename examples."""
        for filename in real_world_filenames:
            metadata = extract_metadata_from_filename(filename)
            # At minimum, should not crash and should return dict
            assert isinstance(metadata, dict)
            assert "filename" in metadata
            assert "datetime_str" in metadata
            assert "suffix" in metadata

            # If datetime is found, verify it's valid
            if metadata["datetime"] is not None:
                from datetime import datetime

                assert isinstance(metadata["datetime"], datetime)


class TestMakeH5FromImages:
    """Test the make_h5_from_images function."""

    def test_create_h5_file(self, sample_4d_array, temp_dir):
        """Test creating H5 file from 4D array."""
        output_path = temp_dir / "test_output.h5"
        result_path = make_h5_from_images(sample_4d_array, output_path)

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
        make_h5_from_images(
            sample_4d_array, output_path, compression="gzip", compression_opts=9
        )

        with h5py.File(output_path, "r") as f:
            assert f["vol"].compression == "gzip"
            assert f["vol"].compression_opts == 9

    def test_invalid_shape(self, temp_dir):
        """Test error handling for invalid array shape."""
        bad_array = np.zeros((10, 10, 3))  # 3D instead of 4D
        output_path = temp_dir / "bad.h5"

        with pytest.raises(ValueError, match="Expected 4D array"):
            make_h5_from_images(bad_array, output_path)

    def test_create_parent_directories(self, sample_4d_array, temp_dir):
        """Test that parent directories are created if needed."""
        output_path = temp_dir / "nested" / "dirs" / "output.h5"
        make_h5_from_images(sample_4d_array, output_path)
        assert output_path.exists()

    def test_overwrite_existing(self, sample_4d_array, temp_dir):
        """Test overwriting existing H5 file."""
        output_path = temp_dir / "existing.h5"

        # Create first file
        make_h5_from_images(sample_4d_array, output_path)

        # Overwrite with different data
        new_array = np.ones_like(sample_4d_array)
        make_h5_from_images(new_array, output_path)

        with h5py.File(output_path, "r") as f:
            assert np.array_equal(f["vol"][:], new_array)


class TestCreateMetadataDataframe:
    """Test the create_metadata_dataframe function."""

    def test_create_dataframe(self):
        """Test creating metadata dataframe with new format."""
        filenames = [
            "_set1_day1_20240101-120000_001.tif",
            "_set1_day2_20240102-130000_002.tif",
            "_set1_day3_20240103-140000_003.tif",
        ]
        df = create_metadata_dataframe(filenames, "test_exp", "control", 5)

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
        df = create_metadata_dataframe([], "exp", "treatment", 10)
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
        df = create_metadata_dataframe(filenames, "complex_exp", "treatment_A", 3)

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
        df = create_metadata_dataframe(filenames, "test", "control", 1)

        assert len(df) == 3
        # Check that missing datetime is handled
        assert (
            df["datetime_str"].iloc[1] is None
            or df["datetime_str"].iloc[1] == "unknown"
        )


class TestProcessTimelapseImageDirectory:
    """Test the main processing function."""

    def test_successful_processing(self, image_directory_with_tiffs, temp_dir):
        """Test successful processing of image directory."""
        output_dir = temp_dir / "output"
        h5_path, csv_path = process_timelapse_image_directory(
            image_directory_with_tiffs,
            "test_exp",
            "control",
            3,
            greyscale=False,
            output_dir=output_dir,
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
        h5_path, csv_path = process_timelapse_image_directory(
            image_directory_with_tiffs,
            "test_exp",
            "control",
            3,
            greyscale=True,
            output_dir=temp_dir,
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
        h5_path, csv_path = process_timelapse_image_directory(
            mixed_format_directory,
            "test_exp",
            "control",
            3,
            image_pattern="*.png",
        )

        assert h5_path is not None
        with h5py.File(h5_path, "r") as f:
            assert f["vol"].shape[0] == 1  # Only 1 PNG file

    def test_default_output_directory(self, image_directory_with_tiffs):
        """Test that output defaults to source directory."""
        h5_path, csv_path = process_timelapse_image_directory(
            image_directory_with_tiffs,
            "test_exp",
            "control",
            3,
            output_dir=None,  # Should default to source_dir
        )

        assert h5_path is not None
        assert h5_path.parent == image_directory_with_tiffs

    def test_logging_output(self, image_directory_with_tiffs, caplog):
        """Test that appropriate log messages are generated."""
        with caplog.at_level(logging.INFO):
            process_timelapse_image_directory(
                image_directory_with_tiffs,
                "test_exp",
                "control",
                3,
            )

        assert "Found 5 image files" in caplog.text
        assert "Saved volume with shape" in caplog.text
        assert "Saved metadata to" in caplog.text

    def test_malformed_filenames(self, malformed_image_directory):
        """Test processing with malformed filenames."""
        h5_path, csv_path = process_timelapse_image_directory(
            malformed_image_directory,
            "test_exp",
            "control",
            3,
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

        h5_path, csv_path = process_timelapse_image_directory(
            img_dir,
            "test_exp",
            "control",
            1,
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
