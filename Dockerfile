# syntax=docker/dockerfile:1
#
# Service image for sleap-roots-predict: the warm-batch predict CLI over the GPU inference
# stack (sleap-nn[torch-cuda128]). Runs as `docker run <image> <input_scan_dir> <output_dir>`.
# The cu128 torch wheels bundle the CUDA runtime, so this slim base + the nvidia container
# runtime is enough (no CUDA base image needed); it also runs CPU-only (device auto-detect).
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# System libs the inference stack needs at import time on a slim base:
#   build-essential          compiles wheels
#   tk                       sleap-nn imports `turtle` (→ tkinter) → libtk8.6.so
#   libgl1, libglib2.0-0     OpenCV (cv2), pulled in by sleap-nn → libGL.so.1
#   libegl1                  skia (sleap-nn skia_augmentation) → libEGL.so.1
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        tk \
        libgl1 \
        libglib2.0-0 \
        libegl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Copy dependency metadata + lockfile first so the install layer caches across source edits.
COPY pyproject.toml uv.lock README.md ./
COPY sleap_roots_predict ./sleap_roots_predict

# Install the project with the GPU (linux_cuda) inference extra from the frozen lockfile.
# The committed lock already resolves linux_cuda; --no-dev keeps test/build tooling out.
RUN uv sync --frozen --no-dev --extra linux_cuda --python 3.12

# Put the project venv on PATH.
ENV PATH="/app/.venv/bin:$PATH"

# Headless matplotlib: the inference stack pulls in matplotlib, whose default Tk backend
# needs system Tk libs absent from the slim base. Force the Agg backend and a writable
# cache dir so importing the package works in a headless container.
ENV MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib

# Bake the build git sha AFTER the heavy layers so a per-commit sha does not bust the
# dependency cache. write_prediction_outputs reads SRP_PREDICT_CODE_SHA into each manifest's
# predict_code_sha (feeding the downstream idempotency key).
ARG SRP_PREDICT_CODE_SHA=""
ENV SRP_PREDICT_CODE_SHA=${SRP_PREDICT_CODE_SHA}

# Real exec-form entrypoint: the warm-batch predict CLI. Positional args are the input scan
# dir and the output dir; the batch exit code propagates to the caller (Argo).
ENTRYPOINT ["python", "-m", "sleap_roots_predict"]
