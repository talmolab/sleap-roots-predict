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
The main package is `sleap_roots_predict/` which currently contains:
- `video_utils.py`: Utilities for processing directories of images into H5 files with metadata

### Key Dependencies
- **Core**: sleap-nn, sleap-io for pose estimation
- **Data Processing**: numpy, pandas, h5py, imageio

### Configuration
- Uses `pyproject.toml` for all project configuration
- Package name is `sleap-roots-predict` with directory `sleap_roots_predict`
- Development tools include pytest, black, ruff, and ipython

### Image Processing Pipeline
The `video_utils.py` module processes plate images over time:
1. Reads TIFF images from a directory
2. Extracts metadata from filenames (timestamp, plate number)
3. Stacks images into a 4D volume (frames, height, width, channels)
4. Saves as compressed H5 file with accompanying CSV metadata