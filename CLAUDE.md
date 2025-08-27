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
- `video_utils.py`: Core utilities for image processing (natural_sort, convert_to_greyscale, load_images, make_h5_from_images, find_image_directories)
- `plates_timelapse_experiment.py`: Experiment processing functions (extract_metadata_from_filename, create_metadata_dataframe, check_image_directory, process_timelapse_image_directory, process_experiment)
- `__init__.py`: Package initialization with version and exports from both modules

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
The modules provide modular functions for processing timelapse experiments:

#### Core Functions (video_utils.py)
- `natural_sort()`: Natural sorting for filenames with numbers
- `convert_to_greyscale()`: Proper RGB to greyscale conversion with channel preservation
- `load_images()`: Batch loading of images with optional greyscale conversion
- `make_h5_from_images()`: Create compressed H5 files from 4D image arrays
- `find_image_directories()`: Recursively find directories containing TIFF images

#### Timelapse Experiment Processing Functions (plates_timelapse_experiment.py)
- `extract_timelapse_metadata_from_filename()`: Parse timestamp and plate number from timelapse filenames
- `create_timelapse_metadata_dataframe()`: Generate metadata CSV from timelapse filenames
- `check_timelapse_image_directory()`: Validate timelapse image directories (suffix consistency, datetime presence, file counts)
- `process_timelapse_image_directory()`: Main pipeline orchestrator for single timelapse directory
- `process_timelapse_experiment()`: Process entire timelapse experiments with CSV metadata, including validation and logging

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