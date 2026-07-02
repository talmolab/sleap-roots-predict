## ADDED Requirements

### Requirement: Predictor Construction

The system SHALL provide `make_predictor(model_paths, peak_threshold=0.2, batch_size=4, device="auto")` that builds a reusable sleap-nn 0.3.0 `Predictor` from one or more model directories. When `device="auto"`, it SHALL select a compute device in the order CUDA → MPS → CPU. It SHALL raise `FileNotFoundError` if any model directory does not exist. The returned `Predictor` SHALL be reusable across multiple videos (loaded once).

#### Scenario: Build predictor from a valid model directory

- **WHEN** `make_predictor` is called with an existing model directory
- **THEN** it returns a live sleap-nn `Predictor` instance without error

#### Scenario: Automatic device selection

- **WHEN** `make_predictor` is called with `device="auto"`
- **THEN** the resolved device is `"cuda"` if a CUDA device is available, else `"mps"` if available, else `"cpu"`

#### Scenario: Missing model directory

- **WHEN** `make_predictor` is called with a path that does not exist
- **THEN** it raises `FileNotFoundError`

### Requirement: Video Prediction

The system SHALL provide `predict_on_video(predictor, video, save_path=None)` that runs inference on an in-memory `sio.Video` using the sleap-nn 0.3.0 API (`predictor.predict(video, make_labels=True)`) and returns predicted `sio.Labels`. When `save_path` is given, it SHALL write a `.slp` file and return its `Path`; otherwise it SHALL return the `sio.Labels`.

#### Scenario: Predict returns labels

- **WHEN** `predict_on_video` is called with a predictor and a video and no `save_path`
- **THEN** it returns an `sio.Labels` object containing predicted instances

#### Scenario: Predict and save to disk

- **WHEN** `predict_on_video` is called with a `save_path`
- **THEN** it writes a `.slp` file at that path that can be reloaded with `sio.load_file` and contains labeled frames, and returns the `Path`

#### Scenario: Persistent predictor reused across videos

- **WHEN** a single predictor from `make_predictor` is passed to `predict_on_video` for two different videos
- **THEN** both calls return valid `sio.Labels` without rebuilding the predictor

### Requirement: sleap-nn 0.3.0 Backend

The service SHALL depend on `sleap-nn==0.3.0` and `sleap-io>=0.8.0,<0.9.0`, and the prediction code SHALL invoke the current sleap-nn 0.3.0 inference API rather than the removed 0.0.x API (`VideoReader` + `make_pipeline`). The legacy 0.0.x call path SHALL NOT be used.

#### Scenario: Prediction uses the 0.3.0 API

- **WHEN** the prediction module runs inference
- **THEN** it constructs the predictor via `Predictor.from_model_paths` and produces labels via `predictor.predict(video, make_labels=True)`, with no reference to `VideoReader` or `make_pipeline`

### Requirement: Real Non-Mocked Test Coverage

Inference tests SHALL exercise real sleap-nn inference against vendored minimal models, with no mocking, monkeypatching, or fake replacement of the sleap-nn or sleap-io boundary. The vendored fixtures SHALL include a native-format model (`best.ckpt` + `training_config.yaml`) and a legacy SLEAP UNet model (`training_config.json` + `best_model.h5`) matching the production model format.

#### Scenario: Native-format model runs real inference

- **WHEN** the test suite runs on CPU with the vendored native minimal model
- **THEN** `make_predictor` + `predict_on_video` produce real `sio.Labels` with no mocks involved

#### Scenario: Legacy SLEAP model loads and runs

- **WHEN** the test suite runs on CPU with the vendored legacy SLEAP UNet model
- **THEN** the model loads under sleap-nn 0.3.0 and `predict_on_video` produces real `sio.Labels`

### Requirement: GPU Inference Execution

The system SHALL run inference on a CUDA device when one is available, verifiable via GPU-marked tests runnable locally with `uv run pytest -m gpu`. GPU-marked tests SHALL skip cleanly when no accelerator is present and are deselected on default and non-GPU CI runs.

#### Scenario: Inference on CUDA when available

- **WHEN** a CUDA device is available and `make_predictor(model_dirs, device="cuda")` is used
- **THEN** the predictor's model reports a CUDA device and `predict_on_video` returns real `sio.Labels`

#### Scenario: GPU tests skip without an accelerator

- **WHEN** the GPU-marked tests run on a host with no CUDA/MPS device
- **THEN** they skip with a message rather than failing

### Requirement: Hardware-Appropriate Backend Installation

Platform install extras SHALL pull a hardware-appropriate torch build. The CUDA extras (`windows_cuda`, `linux_cuda`) SHALL resolve CUDA torch wheels via PyTorch indexes configured in this repository, and the `cpu`/`macos` extras SHALL resolve the CPU/MPS torch build.

#### Scenario: CUDA extra resolves CUDA torch

- **WHEN** the environment is synced with `--extra windows_cuda` (or `--extra linux_cuda`)
- **THEN** the resolved `torch` is a CUDA-enabled build sourced from a configured PyTorch CUDA index

#### Scenario: CPU extra resolves CPU torch

- **WHEN** the environment is synced with `--extra cpu`
- **THEN** the resolved `torch` is the CPU/MPS build

### Requirement: Local Acceptance Validation on Real Data

The system SHALL provide an acceptance test, marked `@pytest.mark.acceptance` and deselected in CI, that is gated on the environment variables `SRP_CYLINDER_DIR` (a directory of real cylinder image frames) and `SRP_MODEL_DIRS` (one or more real root-model directories). When set, it SHALL run the full `make_video_from_images` → `make_predictor` → `predict_on_video` path with a configurable image pattern and write a `.slp`. When unset, it SHALL skip with guidance.

#### Scenario: Acceptance test skips when unconfigured

- **WHEN** `SRP_CYLINDER_DIR` or `SRP_MODEL_DIRS` is not set
- **THEN** the acceptance test skips with a message explaining how to enable it

#### Scenario: Acceptance test runs on real data

- **WHEN** `SRP_CYLINDER_DIR` and `SRP_MODEL_DIRS` point to real images and loadable models
- **THEN** the test builds a video from the image directory, runs prediction, asserts real `sio.Labels`, and writes a `.slp`

#### Scenario: Model load failure is surfaced clearly

- **WHEN** the supplied real models cannot be loaded by sleap-nn 0.3.0
- **THEN** the test fails with a clear error identifying the model directory and load failure, rather than a mocked or silent pass
