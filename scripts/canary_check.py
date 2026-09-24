r"""Live-registry canary check (predict#34): what does this build resolve right now?

Lists the live production cards once, selects models for one or more scan contexts, and
counts the "Skipping non-conforming model artifact" warnings the listing logs. Reads no
ModelCard field, so the same file runs on contracts 0.1.0a7 (``main``) and 0.1.0a9 (this
branch). Exits 1 when any ``--expect*`` check fails, printing one ``FAIL:`` line per
failure — including an ambiguous selection, which is reported rather than raised.
Needs ``WANDB_API_KEY``; read-only.

Example (branch, after the rice-younger-primary canary)::

    uv run python scripts/canary_check.py --context rice,cylinder,3 \
        --expect primary=rice-younger-primary-230104_182346.multi_instance.n-720 \
        --expect-roots primary --expect-skips 13
"""

import argparse
import json
import logging
import sys

from sleap_roots_contracts import ResolvedParams

from sleap_roots_predict import model_registry
from sleap_roots_predict.model_registry import WandbRegistrySource
from sleap_roots_predict.model_selection import choose_models

# The constant exists from this change on; the literal keeps the file runnable on a
# ``main`` that predates it (the canary compares the two).
_SKIP_PREFIX = getattr(
    model_registry, "SKIP_WARNING_PREFIX", "Skipping non-conforming model artifact"
)


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


def _parse_context(text: str) -> dict:
    """Parse ``species,mode,age`` (mode may contain spaces) into scan params."""
    species, mode, age = (part.strip() for part in text.split(","))
    return {"species": species, "mode": mode, "age": int(age)}


def _parse_expect(text: str) -> tuple:
    """Parse ``ROOT=SUFFIX`` into ``(root_type, registry_id_suffix)``."""
    root_type, _, suffix = text.partition("=")
    if not root_type or not suffix:
        raise argparse.ArgumentTypeError(f"expected ROOT=SUFFIX, got {text!r}")
    return root_type, suffix


def select_all(cards, contexts) -> dict:
    """Select models for each context; an ambiguity is recorded, not raised.

    Args:
        cards: The listed model cards.
        contexts: Scan params dicts (``species``/``mode``/``age``).

    Returns:
        ``{"species|mode|age": {root_type: registry_id} | {"error": message}}``.
    """
    results = {}
    for ctx in contexts:
        key = f"{ctx['species']}|{ctx['mode']}|{ctx['age']}"
        try:
            refs = choose_models(ResolvedParams(values=ctx), cards)
        except ValueError as e:
            results[key] = {"error": str(e)}
            continue
        results[key] = {rt: ref.registry_id for rt, ref in sorted(refs.items())}
    return results


def check(results, expects, expect_roots, skips, expect_skips) -> list:
    """Return one failure message per unmet expectation (empty when all hold).

    Args:
        results: Output of :func:`select_all`.
        expects: ``(root_type, suffix)`` pairs every context must resolve exactly.
        expect_roots: The exact set of root types every context must resolve, or None.
        skips: The observed skip-warning count.
        expect_skips: The expected skip-warning count, or None.

    Returns:
        Failure messages.
    """
    failures = []
    for key, selected in results.items():
        if "error" in selected:
            failures.append(f"{key}: selection raised: {selected['error']}")
            continue
        for root_type, suffix in expects:
            rid = selected.get(root_type)
            if rid is None or not rid.endswith(suffix):
                failures.append(
                    f"{key}: expected {root_type} to end with {suffix!r}, got {rid!r}"
                )
        if expect_roots is not None and set(selected) != set(expect_roots):
            failures.append(
                f"{key}: expected root types {sorted(expect_roots)}, "
                f"got {sorted(selected)}"
            )
    if expect_skips is not None and skips != expect_skips:
        failures.append(f"expected {expect_skips} skips, saw {skips}")
    return failures


def main(argv=None) -> int:
    """List live cards, select for each context, and check expectations."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--context",
        action="append",
        type=_parse_context,
        help="species,mode,age (repeatable; default rice,cylinder,3)",
    )
    p.add_argument("--expect", action="append", type=_parse_expect, default=[])
    p.add_argument(
        "--expect-roots", type=lambda s: [r.strip() for r in s.split(",") if r.strip()]
    )
    p.add_argument("--expect-skips", type=int)
    args = p.parse_args(argv)
    contexts = args.context or [_parse_context("rice,cylinder,3")]

    counter = _SkipCounter()
    logging.getLogger("sleap_roots_predict.model_registry").addHandler(counter)
    cards = WandbRegistrySource().list_cards()
    results = select_all(cards, contexts)
    print(
        json.dumps(
            {"cards_listed": len(cards), "skips": counter.count, "results": results},
            indent=2,
        )
    )
    failures = check(
        results, args.expect, args.expect_roots, counter.count, args.expect_skips
    )
    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
