# syntax=docker/dockerfile:1
#
# Service image for sleap-roots-predict. Ships the importable library + its CPU inference
# stack (sleap-nn[torch-cpu]). The warm-GPU worker entrypoint is defined later in roadmap
# tier A3-predict; for now the image installs the package from the frozen uv lockfile.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# System libs the inference stack needs at import time on a slim base:
#   build-essential          compiles wheels
#   tk                       sleap-nn imports `turtle` (→ tkinter) → libtk8.6.so
#   libgl1, libglib2.0-0     OpenCV (cv2), pulled in by sleap-nn → libGL.so.1
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        tk \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Copy dependency metadata + lockfile first so the install layer caches across source edits.
COPY pyproject.toml uv.lock README.md ./
COPY sleap_roots_predict ./sleap_roots_predict

# Install the project with the CPU inference extra from the frozen lockfile.
RUN uv sync --frozen --extra cpu

# Put the project venv on PATH.
ENV PATH="/app/.venv/bin:$PATH"

# Headless matplotlib: the inference stack pulls in matplotlib, whose default Tk backend
# needs system Tk libs absent from the slim base. Force the Agg backend and a writable
# cache dir so importing the package works in a headless container.
ENV MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib

# No CLI entrypoint is wired yet (`[project.scripts]` is empty). Default to a Python REPL
# with the package importable; downstream callers override the command.
ENTRYPOINT ["python"]
CMD ["-c", "import sleap_roots_predict; print('sleap-roots-predict', sleap_roots_predict.__version__)"]
