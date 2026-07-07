"""Guards for the predict-container packaging + workflow wiring (no mocks)."""

import subprocess
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_console_script_declared():
    data = tomllib.loads((REPO / "pyproject.toml").read_text())
    scripts = data["project"]["scripts"]
    assert scripts["sleap-roots-predict"] == "sleap_roots_predict.__main__:main"


def test_module_help_lists_positional_args():
    proc = subprocess.run(
        [sys.executable, "-m", "sleap_roots_predict", "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "input_dir" in proc.stdout
    assert "output_dir" in proc.stdout


def test_docker_workflow_bakes_code_sha():
    wf = (REPO / ".github/workflows/docker-build.yml").read_text()
    assert "SRP_PREDICT_CODE_SHA=${{ github.sha }}" in wf
    assert "type=sha,format=long" in wf
