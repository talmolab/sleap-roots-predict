# API Reference

## Prerequisites

This package requires `sleap-nn` and `sleap-io` to be installed. These libraries provide the core neural network inference and data I/O capabilities. All functions in this package depend on these libraries being available.

## High-Level API

These are the primary functions exposed directly by the package:

```python
from sleap_roots_predict import (
    process_timelapse_experiment,
    make_predictor, 
    predict_on_video
)
```

### `process_timelapse_experiment`
```python
process_timelapse_experiment(
    base_dir: Union[str, Path],
    metadata_csv: Union[str, Path],
    experiment_name: str,
    save_h5: bool = False,
    output_dir: Optional[Union[str, Path]] = None,
    model_paths: List[Union[str, Path]] = [],
    peak_threshold: float = 0.2,
    batch_size: int = 4,
    device: str = "auto",
    enable_tracking: bool = False,
    tracking_config: Optional[Dict[str, Any]] = None,
    greyscale: bool = False,
    dry_run: bool = False,
    image_pattern: str = "*.tif",
    expected_suffix_pattern: Optional[str] = r'^\d{3}$',
    min_images: int = 1,
    max_images: Optional[int] = None,
    check_datetime: bool = True,
    check_suffix_consistency: bool = True,
    log_file: Optional[Union[str, Path]] = None,
    results_json: Optional[Union[str, Path]] = None
) -> Dict[str, List[Dict]]
```
**Main entry point** for processing entire timelapse experiments with multiple plates.

**Parameters:**
- `base_dir`: Base directory containing plate subdirectories
- `metadata_csv`: CSV file with plate metadata (requires: plate_number, treatment, num_plants)
- `experiment_name`: Name for the experiment
- `save_h5`: Save as H5 files (True) or create Video objects (False)
- `output_dir`: Directory for output files
- `model_paths`: Optional SLEAP model paths for predictions
- `device`: Device for predictions ("auto", "cpu", "cuda", "mps")
- `enable_tracking`: Apply tracking to maintain instance IDs across frames
- `tracking_config`: Tracking parameters (window_size, tracker_method, similarity_method, max_tracks)
- `results_json`: Optional path to save results as JSON

**Returns:** Dictionary with processed, failed, and skipped directories

### `make_predictor`
```python
make_predictor(
    model_path: List[Union[str, Path]],
    peak_threshold: float = 0.2,
    batch_size: int = 4,
    device: str = "auto"
) -> Predictor
```
Create a SLEAP predictor with automatic device selection.

**Parameters:**
- `model_path`: List of paths to trained SLEAP model directories
- `peak_threshold`: Confidence threshold for peak detection (0.0-1.0)
- `batch_size`: Number of samples per batch for inference
- `device`: Device for inference ("auto", "cpu", "cuda", or "mps")

**Returns:** Configured `Predictor` instance

### `predict_on_video`
```python
predict_on_video(
    predictor: Predictor,
    video: sio.Video,
    save_path: Optional[Union[str, Path]] = None,
    enable_tracking: bool = False,
    tracking_config: Optional[Dict[str, Any]] = None
) -> Union[Path, sio.Labels]
```
Run prediction on a sleap_io.Video object with optional tracking.

**Parameters:**
- `predictor`: Configured Predictor instance
- `video`: sleap_io.Video object
- `save_path`: Optional path to save predictions as .slp file
- `enable_tracking`: If True, apply tracking to maintain instance IDs across frames
- `tracking_config`: Dictionary of tracking parameters:
  - `window_size`: Number of frames for matching (default: 5)
  - `tracker_method`: "hungarian" or "greedy" (default: "hungarian")
  - `similarity_method`: "centroid", "iou", or "instance" (default: "centroid")
  - `max_tracks`: Maximum number of tracks (default: None)

**Returns:** Path to saved .slp file or Labels object with predictions

---

## Utility Modules

For advanced usage, import these functions directly from their modules:

### Prediction Module (`sleap_roots_predict.predict`)

```python
from sleap_roots_predict.predict import (
    predict_on_h5,
    batch_predict
)
```

#### `predict_on_h5`
```python
predict_on_h5(
    predictor: Predictor,
    h5: Union[str, Path],
    dataset: str = "vol",
    save_path: Optional[Union[str, Path]] = None
) -> Union[Path, sio.Labels]
```
Run prediction on an H5 file (backward compatibility).

#### `batch_predict`
```python
batch_predict(
    predictor: Predictor,
    input_paths: List[Union[str, Path]],
    output_dir: Union[str, Path],
    dataset: str = "vol",
    file_suffix: str = ""
) -> Dict[str, Union[Path, str]]
```
Run predictions on multiple H5 files.

### Video Utilities Module (`sleap_roots_predict.video_utils`)

```python
from sleap_roots_predict.video_utils import (
    make_video_from_images,
    natural_sort,
    convert_to_greyscale,
    load_images,
    save_array_as_h5,
    find_image_directories
)
```

#### `make_video_from_images`
```python
make_video_from_images(
    image_files: List[Union[str, Path]],
    greyscale: bool = False
) -> sio.Video
```
Create a sleap_io.Video object from image files.

#### `natural_sort`
```python
natural_sort(items: List[Union[str, Path]]) -> List[str]
```
Sort items naturally, handling numbers within strings correctly.

#### `convert_to_greyscale`
```python
convert_to_greyscale(
    image: np.ndarray,
    method: str = "weights"
) -> np.ndarray
```
Convert RGB image to greyscale using standard weights or averaging.

#### `load_images`
```python
load_images(
    image_paths: List[Union[str, Path]],
    greyscale: bool = False
) -> Tuple[np.ndarray, List[str]]
```
Load multiple images into a 4D array.

#### `save_array_as_h5`
```python
save_array_as_h5(
    array: np.ndarray,
    output_path: Union[str, Path],
    dataset_name: str = "vol",
    compression: str = "gzip",
    compression_opts: int = 4
) -> Path
```
Save a numpy array as an H5 file.

#### `find_image_directories`
```python
find_image_directories(
    base_dir: Union[str, Path]
) -> List[Path]
```
Find all directories containing TIFF images.

### Timelapse Processing Module (`sleap_roots_predict.plates_timelapse_experiment`)

```python
from sleap_roots_predict.plates_timelapse_experiment import (
    process_timelapse_image_directory,
    check_timelapse_image_directory,
    extract_timelapse_metadata_from_filename,
    create_timelapse_metadata_dataframe
)
```

#### `process_timelapse_image_directory`
```python
process_timelapse_image_directory(
    source_dir: Union[str, Path],
    experiment_name: str,
    treatment: str,
    num_plants: int,
    save_h5: bool = False,
    greyscale: bool = False,
    output_dir: Optional[Union[str, Path]] = None,
    image_pattern: str = "*.tif"
) -> Union[Tuple[Optional[Path], Optional[Path]], 
          Tuple[Optional[sio.Video], Optional[Path]]]
```
Process a single directory of timelapse images.

**Returns:** Tuple of (H5 path or Video object, metadata CSV path)

#### `check_timelapse_image_directory`
```python
check_timelapse_image_directory(
    image_dir: Union[str, Path],
    expected_suffix_pattern: Optional[str] = None,
    min_images: int = 1,
    max_images: Optional[int] = None,
    check_datetime: bool = True,
    check_suffix_consistency: bool = True
) -> Dict[str, Any]
```
Validate an image directory for consistency and correctness.

**Returns:** Dictionary with validation results including:
- `valid`: Boolean indicating if all checks passed
- `image_count`: Number of images found
- `suffixes`: Set of unique suffixes found
- `errors`: List of error messages
- `warnings`: List of warning messages

#### `extract_timelapse_metadata_from_filename`
```python
extract_timelapse_metadata_from_filename(
    filename: Union[str, Path]
) -> Dict[str, Optional[str]]
```
Extract metadata from standardized timelapse filenames.

**Expected formats:**
- `prefix_YYYYMMDD-HHMMSS_suffix.ext`
- `prefix_YYYYMMDD_HHMMSS_suffix.ext`

#### `create_timelapse_metadata_dataframe`
```python
create_timelapse_metadata_dataframe(
    filenames: List[str],
    experiment: str,
    treatment: str,
    expected_num_plants: int
) -> pd.DataFrame
```
Create a metadata DataFrame from timelapse filenames.

---

## Data Types

### Processing Results Dictionary
```python
{
    "directory": str,                # Directory path
    "h5_path": Optional[str],        # Path to H5 file (if save_h5=True)
    "video_frames": Optional[int],   # Number of frames (if save_h5=False)
    "csv_path": Optional[str],       # Path to metadata CSV
    "predictions_path": Optional[str], # Path to predictions (if model provided)
    "check_results": Dict,           # Validation results
    "plate_metadata": Dict           # Metadata from CSV
}
```

### Validation Results Dictionary
```python
{
    "valid": bool,                    # Overall validation status
    "image_count": int,              # Number of images found
    "suffixes": Set[str],            # Unique suffixes found
    "errors": List[str],             # List of error messages
    "warnings": List[str],           # List of warning messages
    "metadata": List[Dict],          # Extracted metadata per file
    "directory": str                 # Directory path (as_posix)
}
```