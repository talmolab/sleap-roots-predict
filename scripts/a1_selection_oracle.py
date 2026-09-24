"""A1 selection-equivalence oracle (predict#34): tabulate choose_models over a grid.

Deliberately reads no ModelCard field, so the same file runs on contracts 0.1.0a7 (flat) and
0.1.0a9 (selectors). Cards come from the live registry or a JSON list of card dicts.

Regenerating both sides needs two environments: the old side (flat cards) only validates
under contracts 0.1.0a7 -- run it from a ``main`` worktree synced to its own lock -- and the new
side only under 0.1.0a9 (this branch). A ValidationError on flat cards here is expected, not a
regression.
"""

import argparse
import json
from pathlib import Path

from sleap_roots_contracts import ModelCard, ResolvedParams

from sleap_roots_predict.model_selection import choose_models


def _cards(args):
    if args.live:
        from sleap_roots_predict.model_registry import WandbRegistrySource

        return WandbRegistrySource().list_cards()
    return [
        ModelCard.model_validate(d)
        for d in json.loads(Path(args.cards_json).read_text())
    ]


def main(argv=None):
    """Run the ``dump`` or ``table`` subcommand."""
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("dump", "table"):
        s = sub.add_parser(name)
        src = s.add_mutually_exclusive_group(required=True)
        src.add_argument("--live", action="store_true")
        src.add_argument("--cards-json")
        s.add_argument("--out", required=True)
        if name == "table":
            s.add_argument("--grid", required=True)
    args = p.parse_args(argv)
    cards = _cards(args)
    if args.cmd == "dump":
        data = [c.model_dump(mode="json") for c in cards]
    else:
        data = {}
        for cell in json.loads(Path(args.grid).read_text()):
            key = f"{cell['species']}|{cell['mode']}|{cell['age']}"
            try:
                refs = choose_models(ResolvedParams(values=cell), cards)
                data[key] = {rt: ref.registry_id for rt, ref in sorted(refs.items())}
            except ValueError as e:
                data[key] = {"raise": str(e)}
    Path(args.out).write_text(json.dumps(data, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
