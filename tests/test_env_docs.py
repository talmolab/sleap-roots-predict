"""Consistency tests for the environment-variable documentation.

These guard against drift between the three places the env surface is described:
the canonical set the code reads, the ``.env.example`` template, and the README
"Configuration" section. They also pin the hard rename — the legacy
``SRP_WANDB_REGISTRY`` / ``SRP_WANDB_ALIAS`` names must appear nowhere.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_EXAMPLE = REPO_ROOT / ".env.example"
README = REPO_ROOT / "README.md"

# The canonical set of env vars this project documents for operators.
EXPECTED_VARS = {
    "WANDB_API_KEY",
    "SRP_WANDB_ENTITY",
    "SRP_WANDB_MODEL_REGISTRY",
    "SRP_WANDB_MODEL_ALIAS",
    "SRP_MODEL_CACHE_DIR",
    "SRP_DEVICE",
}
LEGACY_VARS = {"SRP_WANDB_REGISTRY", "SRP_WANDB_ALIAS"}


def _env_example_vars():
    """Return the var names declared in ``.env.example`` (commented or not)."""
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    return set(re.findall(r"^#?\s*([A-Z][A-Z0-9_]*)=", text, flags=re.MULTILINE))


def _readme_config_section():
    """Return the text of the README 'Configuration' section."""
    text = README.read_text(encoding="utf-8")
    match = re.search(
        r"^#+\s*Configuration\b(.*?)(?=^#+\s|\Z)", text, flags=re.DOTALL | re.MULTILINE
    )
    assert match, "README has no 'Configuration' section"
    return match.group(1)


def test_env_example_lists_exactly_the_expected_vars():
    """.env.example declares exactly the canonical env-var set."""
    assert _env_example_vars() == EXPECTED_VARS


def test_env_example_has_no_legacy_names():
    """.env.example never names the renamed-away legacy vars."""
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    for legacy in LEGACY_VARS:
        assert legacy not in text


def test_readme_configuration_covers_all_vars_and_no_legacy():
    """The README Configuration section names every var and no legacy name."""
    section = _readme_config_section()
    backticked = set(re.findall(r"`([A-Z][A-Z0-9_]*)`", section))
    missing = EXPECTED_VARS - backticked
    assert not missing, f"README Configuration missing: {sorted(missing)}"
    for legacy in LEGACY_VARS:
        assert legacy not in section
