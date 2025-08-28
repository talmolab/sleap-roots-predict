# sleap-roots-predict

A lightweight CLI and library that uses sleap-nn for prediction and produces artifacts in the format expected by sleap-roots pipelines. Intended to interoperate with the sleap-roots model registry.

## Features

- Process timelapse experiments from plate-based imaging systems
- Extract metadata from standardized filenames (datetime, plate numbers, etc.)
- Convert image directories to compressed H5 format with metadata CSV files
- Validate experimental data with comprehensive checks
- Support for multi-plate experiments with CSV-based metadata

## Installation

We recommend doing this in an isolated environment.

Install using [uv](https://docs.astral.sh/uv/getting-started/installation/) for faster, more reliable package management.

Create an isolated environment using uv:
```bash
# Create and activate a new virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

Then install the package:

```bash
# CPU-only installation (all platforms)
uv pip install sleap-roots-predict[cpu]

# Windows with CUDA support
uv pip install sleap-roots-predict[windows_cuda]

# Linux with CUDA support
uv pip install sleap-roots-predict[linux_cuda]

# macOS (Apple Silicon or Intel)
uv pip install sleap-roots-predict[macos]
```

Otherwise just use `pip`. 

### Development Setup

```bash
# Install with development dependencies (replace with your platform's extra)
uv sync --extra dev --extra cpu         # For CPU-only
uv sync --extra dev --extra windows_cuda  # For Windows with CUDA
uv sync --extra dev --extra linux_cuda    # For Linux with CUDA
uv sync --extra dev --extra macos         # For macOS

# Run tests
uv run pytest

# Format code
uv run black sleap_roots_predict tests

# Lint code
uv run ruff check sleap_roots_predict/

# Check spelling
uv run codespell
```

## Usage

### Processing Timelapse Experiments

```python
from sleap_roots_predict import (
    process_timelapse_experiment,
    check_timelapse_image_directory,
    find_image_directories
)

# Process an entire experiment with metadata
results = process_timelapse_experiment(
    base_dir="path/to/experiment",
    metadata_csv="path/to/metadata.csv",
    experiment_name="my_experiment",
    output_dir="path/to/output"
)

# Check a single directory for validity
check_results = check_timelapse_image_directory(
    image_dir="path/to/plate_001",
    check_datetime=True,
    check_suffix_consistency=True
)

# Find all directories containing TIFF images
image_dirs = find_image_directories("path/to/experiment")
```

## CI/CD

The project uses GitHub Actions for continuous integration and deployment:

### Continuous Integration
On every pull request:
- **Linting**: black formatting, ruff linting, codespell
- **Testing**: Full test suite on multiple platforms
  - Ubuntu (latest)
  - Windows (latest)
  - macOS (Apple Silicon)
  - Self-hosted GPU runners (Linux with CUDA)

All platforms run with appropriate hardware acceleration:
- CPU-only on GitHub-hosted runners
- CUDA on self-hosted GPU runners
- Metal Performance Shaders on macOS

### Build and Publish
On release or manual trigger:
- **PyPI Publishing**: Automated wheel building and publishing using uv
- **Trusted Publishing**: Uses PyPI trusted publishing (no API tokens needed)
- **TestPyPI Support**: Manual workflow dispatch option for test publishing

To publish a new release:
0. For testing, manually trigger the workflow with TestPyPI option enabled
1. Update the semantic version in `sleap_roots_predict/__init__.py`.
2. Create a new GitHub release with the same semantic version tag (e.g., `v0.1.0`)
3. The workflow automatically builds and publishes to PyPI

## Project Structure

```
sleap_roots_predict/
├── video_utils.py              # Core image processing utilities
├── plates_timelapse_experiment.py  # Timelapse experiment processing
└── __init__.py                 # Package exports

tests/
├── test_video_utils.py         # Comprehensive test suite
└── conftest.py                 # Shared test fixtures

examples/
└── example_process_experiment.py  # Usage examples
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Ensure all tests pass (`uv run pytest`)
5. Format your code (`uv run black .`)
6. Check linting (`uv run ruff check .`)
7. Commit your changes (`git commit -m 'Add amazing feature'`)
8. Push to your branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

## License

See [LICENSE](LICENSE) file for details.
