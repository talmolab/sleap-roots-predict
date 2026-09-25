"""Write RunManifest JSON fixtures for tests (importable like card_builders)."""

import json
from pathlib import Path


def write_run_manifest(
    directory: Path, filename: str, *, pipeline_run_id: str, scan_keys
) -> Path:
    """Write a manifest as raw bytes (never write_text: CRLF on Windows)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    body = {"pipeline_run_id": pipeline_run_id, "scan_keys": list(scan_keys)}
    path.write_bytes(json.dumps(body).encode("utf-8"))
    return path
