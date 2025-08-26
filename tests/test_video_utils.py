"""Tests for video_utils module."""

from pathlib import Path

import pytest

from sleap_roots_predict.video_utils import natural_sort


def test_natural_sort():
    """Test natural sorting function."""
    # Test with strings
    input_list = ["img10.png", "img2.png", "img1.png", "img20.png"]
    expected = ["img1.png", "img2.png", "img10.png", "img20.png"]
    assert natural_sort(input_list) == expected

    # Test with Path objects
    path_list = [Path(p) for p in input_list]
    assert natural_sort(path_list) == expected

    # Test with mixed numbers and text
    mixed_list = ["file_2_a.txt", "file_10_b.txt", "file_1_c.txt"]
    expected_mixed = ["file_1_c.txt", "file_2_a.txt", "file_10_b.txt"]
    assert natural_sort(mixed_list) == expected_mixed
