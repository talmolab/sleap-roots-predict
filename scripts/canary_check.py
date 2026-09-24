"""Live-registry canary check (predict#34): what does this build resolve right now?

Lists the live production cards, selects models for one scan context, and counts the
"Skipping non-conforming model artifact" warnings the listing logs. Reads no ModelCard
field, so the same file runs on contracts 0.1.0a7 (``main``) and 0.1.0a9 (this branch).
Exits 1 when an ``--expect-*`` check fails. Needs ``WANDB_API_KEY``; read-only.
"""

import argparse
import json
import logging
import sys

from sleap_roots_contracts import ResolvedParams

from sleap_roots_predict.model_registry import WandbRegistrySource
from sleap_roots_predict.model_selection import choose_models

_SKIP_PREFIX = "Skipping non-conforming model artifact"


class _SkipCounter(logging.Handler):
    """Count the registry source's per-artifact skip warnings."""

    def __init__(self) -> None:
        """Start the count at zero."""
        super().__init__(level=logging.WARNING)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        """Count one record if it is a skip warning."""
        if record.getMessage().startswith(_SKIP_PREFIX):
            self.count += 1


def main(argv=None) -> int:
    """List live cards, select for one context, and check expectations."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--species", default="rice")
    p.add_argument("--mode", default="cylinder")
    p.add_argument("--age", type=int, default=3)
    p.add_argument("--expect-registry-suffix", action="append", default=[])
    p.add_argument("--expect-skips", type=int)
    args = p.parse_args(argv)

    counter = _SkipCounter()
    logging.getLogger("sleap_roots_predict.model_registry").addHandler(counter)
    cards = WandbRegistrySource().list_cards()
    params = ResolvedParams(
        values={"species": args.species, "mode": args.mode, "age": args.age}
    )
    refs = choose_models(params, cards)
    selected = {rt: ref.registry_id for rt, ref in sorted(refs.items())}
    print(
        json.dumps(
            {"cards_listed": len(cards), "skips": counter.count, "selected": selected},
            indent=2,
        )
    )

    failures = []
    for suffix in args.expect_registry_suffix:
        if not any(rid.endswith(suffix) for rid in selected.values()):
            failures.append(f"no selected registry_id ends with {suffix!r}")
    if args.expect_skips is not None and counter.count != args.expect_skips:
        failures.append(f"expected {args.expect_skips} skips, saw {counter.count}")
    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
