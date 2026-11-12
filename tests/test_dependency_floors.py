"""Guards the declared dependency floors predict's live compatibility relies on.

wandb (#13): the registry round-trip ``WandbRegistrySource`` depends on (alias -> concrete
version, ``qualified_name``/``digest``/metadata) is proven only by the gated
``@pytest.mark.wandb`` test, run against the locked wandb. A floor keeps a re-lock or
downstream resolve from silently dropping below that proven version (bloom#1187 ran on an
unpinned 0.21.3).
"""

import tomllib
from importlib.metadata import version
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

REPO = Path(__file__).resolve().parents[1]

WANDB_FLOOR = Version("0.31")


def _declared(name: str) -> Requirement:
    """Return the single ``[project].dependencies`` requirement for ``name``."""
    deps = tomllib.loads((REPO / "pyproject.toml").read_text())["project"][
        "dependencies"
    ]
    matches = [Requirement(d) for d in deps if Requirement(d).name == name]
    assert len(matches) == 1, f"expected one {name!r} requirement, got {matches}"
    return matches[0]


def test_wandb_declares_a_floor_at_or_above_the_proven_version():
    req = _declared("wandb")
    lower = [
        Version(s.version) for s in req.specifier if s.operator in (">=", "~=", "==")
    ]
    assert lower, f"wandb is declared without a lower bound: {req}"
    assert max(lower) >= WANDB_FLOOR, f"wandb floor {req} is below {WANDB_FLOOR}"


def test_installed_wandb_satisfies_the_declared_requirement():
    req = _declared("wandb")
    installed = version("wandb")
    assert req.specifier.contains(
        installed, prereleases=True
    ), f"installed wandb {installed} does not satisfy {req}"
