# API Reference

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
- `model_paths`, `peak_threshold`, `batch_size`, `device`: Accepted but currently **ignored** — prediction within this flow is deferred. Use `predict_on_video` directly to run inference.
- `results_json`: Optional path to save results as JSON

**Returns:** Dictionary with processed, failed, and skipped directories

### `make_predictor`
```python
make_predictor(
    model_paths: List[Union[str, Path]],
    peak_threshold: float = 0.2,
    batch_size: int = 4,
    device: str = "auto"
) -> Predictor
```
Create a reusable sleap-nn predictor with automatic device selection. The
predictor loads the model(s) once and can be reused across many videos.

**Parameters:**
- `model_paths`: List of paths to trained model directories (one per model, e.g. per root type)
- `peak_threshold`: Confidence threshold for peak detection (0.0-1.0)
- `batch_size`: Number of samples per batch for inference
- `device`: Device for inference ("auto", "cpu", "cuda", "cuda:N", or "mps"). "auto" honors the `SRP_DEVICE` env var if set.

**Returns:** Configured `Predictor` instance

**Raises:** `ValueError` if `model_paths` is empty; `FileNotFoundError` if a model directory does not exist.

### `predict_on_video`
```python
predict_on_video(
    predictor: Predictor,
    video: sio.Video,
    save_path: Optional[Union[str, Path]] = None
) -> Union[Path, sio.Labels]
```
Run prediction on a sleap_io.Video object.

**Parameters:**
- `predictor`: Configured Predictor instance
- `video`: sleap_io.Video object
- `save_path`: Optional path to save predictions as .slp file

**Returns:** Path to saved .slp file or Labels object with predictions

### Output Contract

Write the per-scan artifacts the downstream sleap-roots traits stage reads — named
per-root `.slp` (`{scan_key}.model{model_id}.root{root_type}.slp`) plus a combined
`{scan_key}.predictions.json` manifest. See the `prediction-output` OpenSpec spec and the
`sleap_roots_predict.output_contract` docstrings for the full artifact grammar and manifest
schema (kept single-sourced there to avoid drift).

```python
from sleap_roots_predict import (
    write_prediction_outputs,  # write one scan's artifacts, return a PredictionManifest
    predict_and_write_batch,   # drive a warm worker over N scans (one subdir per scan)
    ScanRequest,               # a batch scan input (scan_key, video, params, ...)
    PredictionManifest,        # per-scan manifest (manifest + predict-side provenance)
    PredictionArtifact,        # one per-root record (model_id, ModelRef, slp_path, checksum, size)
)
```

#### `write_prediction_outputs`
```python
write_prediction_outputs(
    labels_by_root: Dict[str, sio.Labels],
    refs_by_root: Dict[str, ModelRef],
    out_dir: Union[str, Path],
    *,
    scan_key: str,
    plant_qr_code: Optional[str] = None,
    inference_config: Dict[str, Any],
    output_params: Dict[str, Any],
    predict_code_sha: Optional[str] = None,
    predict_container_digest: Optional[str] = None,
) -> PredictionManifest
```
Write the named per-root `.slp` files and a combined `{scan_key}.predictions.json` into
`out_dir` (created if missing); returns the `PredictionManifest`. `plant_qr_code` defaults
to `scan_key`. Build identity falls back to `SRP_PREDICT_CODE_SHA` /
`SRP_PREDICT_CONTAINER_DIGEST` then `""`. Re-runs overwrite in place.

**Raises:** `ValueError` if `scan_key` is unsafe as a path segment, or `labels_by_root`
and `refs_by_root` cover different root types.

#### `predict_and_write_batch`
```python
predict_and_write_batch(
    worker: WarmModelWorker,
    requests: Iterable[ScanRequest],
    out_dir: Union[str, Path],
    *,
    predict_code_sha: Optional[str] = None,
    predict_container_digest: Optional[str] = None,
) -> List[PredictionManifest]
```
Drive one warm worker over N scans, writing `out_dir/{scan_key}/` per scan and reusing
resident predictors. Returns one manifest per scan, in request order.

---

## Utility Modules

For advanced usage, import these functions directly from their modules:

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
    "predictions_path": Optional[str], # Always None (prediction in this flow is deferred)
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