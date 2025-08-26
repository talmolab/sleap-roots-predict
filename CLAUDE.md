# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is `sleap-roots-predict`, a lightweight CLI and library that uses sleap-nn for prediction and produces artifacts in the format expected by sleap-roots pipelines. It's designed to interoperate with the sleap-roots model registry.

## Development Commands

### Linting and Formatting
- **Format code**: `black .` (configured for 88-character line length)
- **Lint code**: `ruff check .` (configured for Google-style docstrings)
- **Check spelling**: `codespell` (configured in pyproject.toml)

### Testing
- **Run tests**: `pytest`
- **Run tests with coverage**: `pytest --cov`

### Build and Package
- **Build package**: `python -m build`
- **Upload to TestPyPI**: `twine upload --repository testpypi dist/*`

## Architecture

### Package Structure
The main package is `sleap_roots_predict/` which contains:
- `video_utils.py`: Modular utilities for processing directories of images into H5 files with metadata
- `__init__.py`: Package initialization with version and exports

### Key Dependencies
- **Core**: sleap-nn, sleap-io for pose estimation
- **Data Processing**: numpy, pandas, h5py, imageio
- **Testing**: pytest, pytest-cov, PIL

### Configuration
- Uses `pyproject.toml` for all project configuration
- Package name is `sleap-roots-predict` with directory `sleap_roots_predict`
- Development tools include pytest, black, ruff, codespell, and ipython
- Platform-specific extras: `cpu`, `windows_cuda`, `linux_cuda`, `macos`

### Image Processing Pipeline
The `video_utils.py` module provides modular functions:

#### Core Functions
- `natural_sort()`: Natural sorting for filenames with numbers
- `convert_to_greyscale()`: Proper RGB to greyscale conversion with channel preservation
- `load_images()`: Batch loading of images with optional greyscale conversion
- `extract_metadata_from_filename()`: Parse timestamp and plate number from filenames
- `make_h5_from_images()`: Create compressed H5 files from 4D image arrays
- `create_metadata_dataframe()`: Generate metadata CSV from filenames
- `process_timelapse_image_directory()`: Main pipeline orchestrator

#### Processing Flow
1. Validates and reads images from directory using glob patterns
2. Naturally sorts filenames to maintain temporal order
3. Optionally converts to greyscale using standard RGB weights
4. Stacks images into 4D volume (frames, height, width, channels)
5. Saves as compressed H5 file with configurable compression
6. Generates accompanying CSV with experimental metadata

### Testing
- Comprehensive test suite with fixtures in `conftest.py`
- 100% code coverage target
- Tests for edge cases, error handling, and various image formats
- Fixtures for RGB, greyscale, RGBA, and large images
- Tests for Unicode paths and malformed filenames