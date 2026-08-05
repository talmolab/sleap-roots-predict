"""Regenerate the A3-predict parity results JSON against the live registry.

**Lab-only, not portable.** This script assumes a Windows machine with the
lab's ``Z:`` network share mapped, real ``wandb`` registry credentials, and
the lab's own historical prefix-map layout (see ``PREFIX_MAP_SOURCES``
below). It is not shipped in the PyPI wheel or the container image (outside
``sleap_roots_predict/``), and is not covered by pytest (``scripts/`` is
outside ``testpaths``) — it is a thin, credential-requiring wrapper around
:func:`sleap_roots_predict.parity.run_parity_harness`, whose own behavior
(per-card isolation, the no-clobber guard) is what's actually tested.

Usage::

    uv run python scripts/run_parity_harness.py

(``uv run`` matters: a bare ``python scripts/run_parity_harness.py`` puts
``scripts/`` on ``sys.path[0]`` instead of the repo root, and
``sleap_roots_predict`` only resolves via the editable venv.)

Requires ``WANDB_API_KEY`` and ``SRP_PARITY_DATA_DIR`` (the basename-search
root for ground truth whose video paths were reorganized, not just moved —
see the Ground Truth Resolution Per Model spec requirement's tier 3) set in
the environment. Neither is a new env var: both are the harness's existing,
already-documented configuration (see this repo's ``README.md`` Parity
Harness section) — nothing here should be added to ``.env.example``, which
is scoped to production/operator runtime config, not this test-gating var.

Why the lab's ``prefix_map`` source keys are hardcoded here rather than
configurable: they're immutable facts baked into the training-time
``.slp`` files' embedded video paths (can't change without re-training,
already documented in ``design.md``'s Decision 2). Why the share-root
*target* is a CLI flag rather than also hardcoded: it's a mapped Windows
drive letter, exactly as machine-specific as ``SRP_PARITY_DATA_DIR`` — see
``openspec/changes/define-parity-tolerance/design.md`` Decision 2 and
Decision 8, and the design doc's own fuller rationale, for why neither
value needed a new config-file mechanism.
"""

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Sequence

#: The two source prefixes found in production `ModelCard` bundles' embedded
#: video paths (confirmed by materializing real artifacts) — both remapped
#: against the same `--share-root` target.
PREFIX_MAP_SOURCES = ("D:/SLEAP", "C:/Users/pbiobgh/Desktop/SLEAP")

#: This lab's mapped network-share root, as of the original empirical run
#: that produced the checked-in results JSON. The single authoritative,
#: executable copy of this value going forward — every mention of it in the
#: dated design docs is a historical record, not something to keep in sync.
_DEFAULT_SHARE_ROOT = "Z:/users/eberrigan/SLEAP"

#: How many frames to sample per model, matching the original empirical run
#: (task 5.2). `run_parity_harness`/`evaluate_model_card` themselves default
#: `sample_n` to `None` (no magic number in the reusable library function) —
#: this is deliberately only this script's own default.
_SAMPLE_N = 100

_DEFAULT_OUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "superpowers"
    / "specs"
    / "2026-08-04-define-parity-tolerance-results.json"
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse args and regenerate the parity results JSON.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        ``0`` on success.
    """
    parser = argparse.ArgumentParser(
        prog="run_parity_harness",
        description=(
            "Regenerate the A3-predict parity results JSON against the live "
            "wandb registry and this lab's network share. Lab-only."
        ),
    )
    parser.add_argument(
        "--share-root",
        default=_DEFAULT_SHARE_ROOT,
        help=(
            "Network-share root the lab's prefix-map source paths relink "
            f"against (default: {_DEFAULT_SHARE_ROOT!r})."
        ),
    )
    parser.add_argument(
        "--out",
        default=str(_DEFAULT_OUT_PATH),
        help=(
            "Where to write the regenerated report (default: the existing "
            "checked-in results JSON, overwritten in place)."
        ),
    )
    parser.add_argument(
        "--workdir",
        default=None,
        help=(
            "Scratch directory for intermediate files (default: a "
            "non-auto-deleted temp directory, so a gapped model's "
            "intermediates stay inspectable after a long run)."
        ),
    )
    args = parser.parse_args(argv)

    # Lazy imports: `--help`/a parse error above exits before pulling in
    # wandb/sleap_nn.
    from sleap_roots_predict.model_registry import WandbRegistrySource
    from sleap_roots_predict.parity import build_basename_index, run_parity_harness

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logger = logging.getLogger(__name__)

    search_root = os.environ.get("SRP_PARITY_DATA_DIR")
    if not search_root:
        logger.error(
            "SRP_PARITY_DATA_DIR is not set; it must point at the "
            "basename-search root for ground truth whose video paths were "
            "reorganized (see README.md's Parity Harness section)."
        )
        return 1
    if not Path(search_root).is_dir():
        logger.error("SRP_PARITY_DATA_DIR %r is not a directory.", search_root)
        return 1

    workdir = (
        Path(args.workdir)
        if args.workdir
        else Path(tempfile.gettempdir()) / ("sleap-roots-predict-parity-harness")
    )
    workdir.mkdir(parents=True, exist_ok=True)

    prefix_map = {source: args.share_root for source in PREFIX_MAP_SOURCES}

    source = WandbRegistrySource()
    cards = source.list_cards()
    logger.info("Found %d production ModelCard(s) in the live registry.", len(cards))

    logger.info("Indexing %r for basename search...", search_root)
    basename_index = build_basename_index(search_root)

    out_path = run_parity_harness(
        cards,
        source,
        workdir,
        args.out,
        prefix_map=prefix_map,
        basename_index=basename_index,
        sample_n=_SAMPLE_N,
    )
    logger.info("Wrote parity report to %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
