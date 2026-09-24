"""A1 compare step (predict#34): build the grid, then check old vs new tables."""

import argparse
import json
import sys
from pathlib import Path

from typing import get_args

from sleap_roots_contracts import Mode

# Every mode the contract allows, so a new or renamed mode is covered on a rerun.
_MODES = get_args(Mode)


def _contexts(card):
    if "selectors" in card:
        return [
            (s["species"], s["mode"], s["age_min"], s["age_max"])
            for s in card["selectors"]
        ]
    return [(card["species"], card["mode"], card["age_min"], card["age_max"])]


def _grid(old, new):
    ctx = [c for card in old + new for c in _contexts(card)]
    species = sorted({c[0] for c in ctx}) + ["unmodelled-species"]
    lo, hi = min(c[2] for c in ctx) - 1, max(c[3] for c in ctx) + 1
    return [
        {"species": s, "mode": m, "age": a}
        for s in species
        for m in _MODES
        for a in range(max(lo, 0), hi + 1)
    ]


def _check(old, new, baseline, new_cards):
    to_model_old = {
        c["collection"]: c["source_model_id"] for c in baseline["collections"]
    }
    to_model_new = {c["registry_id"]: c["source_model_id"] for c in new_cards}
    errors, selected = [], {"old": 0, "new": 0}
    if any("raise" in v for v in old.values()):
        errors.append("precondition: the old side raises in some cell")
    for key in sorted(old):
        o, n = old[key], new[key]
        if "raise" in n:
            errors.append(f"{key}: new side raises: {n['raise']}")
            continue
        o_m = {
            rt: to_model_old[rid.rsplit("/", 1)[-1]]
            for rt, rid in o.items()
            if rt != "raise"
        }
        n_m = {rt: to_model_new[rid] for rt, rid in n.items()}
        selected["old"] += bool(o_m)
        selected["new"] += bool(n_m)
        if o_m != n_m:
            errors.append(f"{key}: old {o_m} != new {n_m}")
    if not selected["new"] or selected["old"] != selected["new"]:
        errors.append(f"selected-cell counts differ or are zero: {selected}")
    for card in new_cards:
        for sp, *_ in _contexts(card):
            if sp != sp.lower():
                errors.append(f"{card['registry_id']}: species {sp!r} is not lowercase")
    return errors, selected


def main(argv=None):
    """Run the ``grid`` or ``check`` subcommand; ``check`` exits 1 on any mismatch."""
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grid")
    g.add_argument("old_cards")
    g.add_argument("new_cards")
    g.add_argument("--out", required=True)
    c = sub.add_parser("check")
    c.add_argument("old_table")
    c.add_argument("new_table")
    c.add_argument("--baseline", required=True)
    c.add_argument("--new-cards", required=True)
    args = p.parse_args(argv)

    def load(path):
        return json.loads(Path(path).read_text())

    if args.cmd == "grid":
        grid = _grid(load(args.old_cards), load(args.new_cards))
        Path(args.out).write_text(json.dumps(grid, indent=2))
        return 0
    errors, selected = _check(
        load(args.old_table),
        load(args.new_table),
        load(args.baseline),
        load(args.new_cards),
    )
    print(json.dumps({"selected_cells": selected, "errors": errors}, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
