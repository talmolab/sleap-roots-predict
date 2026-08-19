"""CLI entrypoint: ``python -m sleap_roots_predict <input_dir> <output_dir>``.

Warm-batch predict over a directory of staged scans. Exit codes: ``0`` success (no
scan failed); ``3`` partial (the batch ran to completion but one or more scans
isolated-failed); Python's default ``1`` for every other failure (a pre-flight
staging error — missing input directory, duplicate ``scan_key``, malformed
``run_manifest.json``, or zero scans discovered — or a genuine crash). ``2`` is
reserved by ``argparse`` for a CLI usage error and is never returned by this
driver's own logic.
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
        ``0`` on success, ``3`` on a partial batch (isolated scan failures). A
        pre-flight staging error or other crash propagates uncaught, surfacing
        Python's default exit ``1`` (see the module docstring).
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

    # Lazy import: `--help` exits inside parse_args above, so it never pulls in torch.
    from sleap_roots_predict.batch import run_batch

    try:
        result = run_batch(args.input_dir, args.output_dir)
    except (FileNotFoundError, ValueError) as exc:
        # A pre-flight staging error (missing input mount, duplicate scan_key,
        # malformed run_manifest.json, or zero scans discovered — the latter two
        # also raise ValueError, so they land here too). Log a clean message, then
        # re-raise so the process still exits via Python's default unhandled-
        # exception code (1), identical to any other crash.
        logging.getLogger(__name__).error("Batch aborted: %s", exc)
        raise

    n_ok = sum(1 for s in result.scans if s.status == "ok")
    n_skip = sum(1 for s in result.scans if s.status == "skipped")
    n_fail = sum(1 for s in result.scans if s.status == "failed")
    logging.getLogger(__name__).info(
        "Batch complete: %d ok, %d skipped, %d failed", n_ok, n_skip, n_fail
    )
    return 0 if result.ok else 3


if __name__ == "__main__":
    sys.exit(main())
