"""A2 driver (predict#34): run_batch over real weights from a local card source.

Cards come from a JSON list of card dicts (A1's old-cards.json on main, new-cards.json on the
branch); ``--dirs`` maps each card's registry_id to an extracted model directory.
"""

import argparse
import json
from pathlib import Path

from sleap_roots_contracts import ModelCard

from sleap_roots_predict.batch import run_batch
from sleap_roots_predict.model_registry import LocalCardSource


def main(argv=None):
    """Run one A2 batch and print its per-scan statuses."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input_dir")
    p.add_argument("output_dir")
    p.add_argument("--cards-json", required=True)
    p.add_argument("--dirs", required=True, help="JSON {registry_id: model_dir}")
    args = p.parse_args(argv)
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"output dir must be fresh and empty: {out.as_posix()}")
    dirs = json.loads(Path(args.dirs).read_text())
    cards = [
        ModelCard.model_validate(d)
        for d in json.loads(Path(args.cards_json).read_text())
    ]
    source = LocalCardSource(
        [(c, Path(dirs[c.registry_id])) for c in cards if c.registry_id in dirs]
    )
    result = run_batch(Path(args.input_dir), out, source=source)
    statuses = {s.scan_key: s.status for s in result.scans}
    print(json.dumps(statuses, indent=2))
    if set(statuses.values()) != {"ok"}:
        raise SystemExit(f"not all scans ok: {statuses}")


if __name__ == "__main__":
    main()
