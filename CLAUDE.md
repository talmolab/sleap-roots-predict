# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is `sleap-roots-predict`, a lightweight CLI and library that uses sleap-nn for prediction and produces artifacts in the format expected by sleap-roots pipelines. It's designed to interoperate with the sleap-roots model registry.

## Development Commands

### Installation
Install with platform-specific extras for proper hardware acceleration:
- **CPU-only**: `uv sync --extra dev --extra cpu`
- **Windows with CUDA**: `uv sync --extra dev --extra windows_cuda`
- **Linux with CUDA**: `uv sync --extra dev --extra linux_cuda`
- **macOS**: `uv sync --extra dev --extra macos`

### Linting and Formatting
- **Format code**: `black .` (configured for 88-character line length)
- **Lint code**: `ruff check .` (configured for Google-style docstrings)
- **Check spelling**: `codespell` (configured in pyproject.toml)

### Testing
- **Run tests**: `pytest`
- **Run tests with coverage**: `pytest --cov`

### Build and Package
- **Build package**: `uv build`
- **Upload to TestPyPI**: `uv publish --index testpypi --trusted-publishing always`
- **Upload to PyPI**: `uv publish`

## Architecture

### Package Structure
The main package is `sleap_roots_predict/` which contains:
- `predict.py`: SLEAP-NN prediction interface (make_predictor, predict_on_video, predict_on_h5, batch_predict)
- `video_utils.py`: Core utilities for image processing (natural_sort, convert_to_greyscale, load_images, make_video_from_images, save_array_as_h5, find_image_directories)
- `plates_timelapse_experiment.py`: Experiment processing functions (extract_timelapse_metadata_from_filename, create_timelapse_metadata_dataframe, check_timelapse_image_directory, process_timelapse_image_directory, process_timelapse_experiment)
- `__init__.py`: Package initialization with version and exports from all modules

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

#### Prediction Functions (predict.py)
- `make_predictor()`: Create SLEAP predictor with automatic device selection
- `predict_on_video()`: Run inference on sleap_io.Video objects
- `predict_on_h5()`: Run inference on H5 files (backward compatibility)
- `batch_predict()`: Process multiple H5 files in batch

#### Core Functions (video_utils.py)
- `natural_sort()`: Natural sorting for filenames with numbers
- `convert_to_greyscale()`: Proper RGB to greyscale conversion with channel preservation
- `load_images()`: Batch loading of images with optional greyscale conversion
- `make_video_from_images()`: Create sleap_io.Video objects from image sequences
- `save_array_as_h5()`: Save numpy arrays as compressed H5 files
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
4. Creates either:
   - sleap_io.Video object for direct prediction (default)
   - Compressed H5 file for storage (optional)
5. Generates accompanying CSV with experimental metadata
6. Optionally runs SLEAP predictions if models provided
7. Exports results as JSON for downstream analysis

### Testing
- Comprehensive test suite with fixtures in `conftest.py`
- 105 tests passing with high coverage
- Separate test modules for predict and video_utils
- GPU tests marked with `@pytest.mark.gpu` for conditional execution
- Tests for edge cases, error handling, and various image formats
- Fixtures for RGB, greyscale, RGBA, and large images
- Tests for Unicode paths and malformed filenames

### CI/CD
GitHub Actions workflow runs on every pull request:
- **Linting**: black formatting, ruff linting, codespell
- **Testing**: Full test suite on Ubuntu, Windows, macOS, and self-hosted GPU runners
- Platform-specific installations with appropriate hardware acceleration
- All tests run with pytest and coverage reporting