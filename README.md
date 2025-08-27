# sleap-roots-predict

A lightweight CLI and library that uses sleap-nn for prediction and produces artifacts in the format expected by sleap-roots pipelines. Intended to interoperate with the sleap-roots model registry.

## Features

- Process timelapse experiments from plate-based imaging systems
- Extract metadata from standardized filenames (datetime, plate numbers, etc.)
- Convert image directories to compressed H5 format with metadata CSV files
- Validate experimental data with comprehensive checks
- Support for multi-plate experiments with CSV-based metadata

## Installation

### Using uv (recommended)

```bash
# Install with CPU support
uv sync --extra cpu

# Install with CUDA support (Windows)
uv sync --extra windows_cuda

# Install with CUDA support (Linux)
uv sync --extra linux_cuda

# Install with Apple Silicon support (macOS)
uv sync --extra macos
```

### Development Setup

```bash
# Install with development dependencies
uv sync --extra dev --extra windows_cuda  # or your platform's extra

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

The project uses GitHub Actions for continuous integration with the following checks:

### Linting
- **black**: Code formatting
- **ruff**: Python linting
- **codespell**: Spell checking

### Testing
Tests run on multiple platforms:
- Ubuntu (latest)
- Windows (latest)
- macOS (Apple Silicon)
- Self-hosted GPU runners (Linux with CUDA)

All platforms run the full test suite with appropriate hardware acceleration:
- CPU-only on GitHub-hosted runners
- CUDA on self-hosted GPU runners
- Metal Performance Shaders on macOS

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
