"""CLI entrypoint: ``python -m sleap_roots_predict <input_dir> <output_dir>``.

Warm-batch predict over a directory of staged scans. Exit code is ``0`` when no
scan failed and ``1`` otherwise, so an Argo step sees a real batch result.
"""

import argparse
import logging
import sys
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse args, run the batch, and return a process exit code.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        ``0`` if no scan failed, ``1`` otherwise.
    """
    parser = argparse.ArgumentParser(
        prog="sleap_roots_predict",
        description="Warm-batch predict over a directory of staged scans.",
    )
    parser.add_argument(
        "input_dir",
        help="Directory of staged scans (each scan: a directory of image frames "
        "with a co-located {scan_key}.scan_metadata.json sidecar).",
    )
    parser.add_argument(
        "output_dir",
        help="Directory to write per-scan prediction outputs into.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # Lazy import: keeps ``--help`` import-light and lets this module load before
    # ``batch.py`` exists during incremental development.
    from sleap_roots_predict.batch import run_batch

    result = run_batch(args.input_dir, args.output_dir)
    n_ok = sum(1 for s in result.scans if s.status == "ok")
    n_skip = sum(1 for s in result.scans if s.status == "skipped")
    n_fail = sum(1 for s in result.scans if s.status == "failed")
    logging.getLogger(__name__).info(
        "Batch complete: %d ok, %d skipped, %d failed", n_ok, n_skip, n_fail
    )
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
