# sleap-roots-predict

A lightweight CLI and library that uses sleap-nn for prediction and produces artifacts in the format expected by sleap-roots pipelines. Intended to interoperate with the sleap-roots model registry.

## Features

- **SLEAP-NN Integration**: Direct inference using SLEAP neural network models
- **Flexible Video Processing**: Create `sleap_io.Video` objects from image sequences for prediction
- **Timelapse Experiment Support**: Process plate-based imaging systems with automated metadata extraction
- **Metadata Extraction**: Parse datetime, plate numbers, and experimental conditions from standardized filenames
- **Dual Output Formats**: Generate either Video objects for direct prediction or compressed H5 files for storage
- **Comprehensive Validation**: Check image directories for consistency, datetime formats, and suffix patterns
- **Batch Processing**: Handle multi-plate experiments with CSV-based metadata
- **GPU Acceleration**: Automatic device selection (CUDA, MPS, or CPU) for optimal performance
- **JSON Export**: Save experiment results and metadata in JSON format for downstream analysis

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

### Quick Start - High-Level API

```python
from sleap_roots_predict import (
    process_timelapse_experiment,
    make_predictor,
    predict_on_video
)

# Process an entire timelapse experiment with predictions
results = process_timelapse_experiment(
    base_dir="path/to/experiment",
    metadata_csv="path/to/metadata.csv",
    experiment_name="my_experiment",
    output_dir="path/to/output",
    model_paths=["path/to/sleap/model"],  # Optional: run predictions
    device="cuda",  # Use GPU for predictions
    results_json="results.json"  # Save results as JSON
)

# Or use the prediction API directly
predictor = make_predictor(
    model_path=["path/to/model"],
    peak_threshold=0.2,
    batch_size=4,
    device="auto"  # Automatically selects GPU if available
)

# Create a Video object and run predictions
from sleap_roots_predict.video_utils import make_video_from_images
image_files = sorted(Path("path/to/images").glob("*.tif"))
video = make_video_from_images(image_files, greyscale=False)

predictions = predict_on_video(
    predictor,
    video,
    save_path="predictions.slp"  # Optional: save predictions
)
```

### Advanced Usage - Utility Functions

For more control, you can import utility functions directly from their modules:

```python
# Import utility functions from their modules
from sleap_roots_predict.plates_timelapse_experiment import (
    process_timelapse_image_directory,
    check_timelapse_image_directory,
    find_image_directories,
    extract_timelapse_metadata_from_filename,
    create_timelapse_metadata_dataframe
)

from sleap_roots_predict.predict import (
    predict_on_h5,
    batch_predict
)

from sleap_roots_predict.video_utils import (
    make_video_from_images,
    load_images,
    convert_to_greyscale,
    save_array_as_h5,
    natural_sort
)

# Process a single directory
video, csv_path = process_timelapse_image_directory(
    source_dir="path/to/plate_001",
    experiment_name="exp1",
    treatment="control",
    num_plants=3,
    save_h5=False,  # Returns Video object
    output_dir="output/"
)

# Validate directories before processing
check_results = check_timelapse_image_directory(
    image_dir="path/to/plate_001",
    expected_suffix_pattern=r'^\d{3}$',  # e.g., "001", "002"
    min_images=5,
    max_images=1000,
    check_datetime=True,
    check_suffix_consistency=True
)

# Batch process H5 files
results = batch_predict(
    predictor,
    input_paths=["file1.h5", "file2.h5"],
    output_dir="predictions/",
    dataset="vol"
)

# Load and process images
image_paths = ["img1.tif", "img2.tif", "img3.tif"]
volume, filenames = load_images(image_paths, greyscale=True)
print(f"Loaded volume shape: {volume.shape}")  # (frames, height, width, channels)

# Convert RGB to greyscale with proper weights
grey_image = convert_to_greyscale(
    rgb_image,
    method="weights"  # Uses standard RGB weights (0.299, 0.587, 0.114)
)

# Save processed data as H5
save_array_as_h5(
    volume,
    output_path="processed_data.h5",
    compression="gzip",
    compression_opts=4
)

# Natural sorting for filenames with numbers
files = ["img_2.tif", "img_10.tif", "img_1.tif"]
sorted_files = natural_sort(files)
# Result: ["img_1.tif", "img_2.tif", "img_10.tif"]
```

## CI/CD

The project uses GitHub Actions for continuous integration and deployment:

### Continuous Integration
On every pull request:
- **Linting**: black formatting, ruff linting, codespell
- **Testing**: Full test suite on multiple platforms
  - Ubuntu (latest) - CPU only
  - Windows (latest) - CPU only
  - macOS (Apple Silicon) - with Metal Performance Shaders (MPS) GPU support
  - Self-hosted GPU runners (Linux with CUDA)

GPU tests are automatically run on:
- macOS runners using Metal Performance Shaders
- Self-hosted Linux runners with CUDA support

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
├── predict.py                      # SLEAP-NN prediction interface
├── video_utils.py                  # Core image processing utilities
├── plates_timelapse_experiment.py  # Timelapse experiment processing
└── __init__.py                     # Package exports and version

tests/
├── test_predict.py             # Prediction module tests
├── test_video_utils.py         # Video utilities tests
└── conftest.py                 # Shared test fixtures

.github/workflows/
├── ci.yml                      # Continuous integration
└── publish.yml                 # PyPI publishing workflow
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
