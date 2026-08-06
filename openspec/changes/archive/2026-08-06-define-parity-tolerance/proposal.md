## Why

`sleap-roots-predict`'s sleap-nn rewrite is functionally complete and running real cluster
traffic, but its "A3-predict ✅" status has been blocked since 2026-07-06 on an undefined
parity gate (sleap-roots-pipeline#15): nobody has decided what accuracy tolerance vs. the
legacy classic-SLEAP pipeline is acceptable, or which models/scans to measure it on. Because
the production wandb registry models are still legacy classic-SLEAP-trained weights loaded into
sleap-nn via a sanitization workaround (not retrained), this is an inference-engine-correctness
question, not a model-quality question — a tight, empirically-set tolerance is the right bar.
No parity harness or ground-truth-comparison capability exists in this repo today.

## What Changes

- Add a new `prediction-parity` capability: a harness that, for each production `ModelCard`,
  resolves real human-labeled ground truth (a matching collection in the separate
  `wandb-registry-sleap-roots-labels` registry when available, else the `labels_gt.val.slp`
  bundled in the model's own wandb artifact with network-path relinking for its stale
  drive-letter references), runs sleap-nn inference on that ground truth, and compares it
  against classic-SLEAP's own eval on the same ground truth via
  `sleap_nn.evaluation.run_evaluation` (`match_method="oks"` at the library's permissive default
  `match_threshold=0.0` for instance matching — OKS-derived *score* fields are never read, per
  `sleap-roots-training`#17's finding that they're miscalibrated for the root-keypoint domain;
  `match_method="centroid"` was tried and rejected, confirmed unusable for full-skeleton
  comparison).
- Add `sleap_roots_predict/parity.py`: ground-truth resolution + a thin wrapper around
  `run_evaluation`, reusable by predict#8's future `peak_threshold` sweep.
- Add a checked-in, `LabelCard`-shaped ground-truth manifest (one record per production
  `ModelCard`, unrecoverable provenance fields explicitly `None`; unresolved models listed as
  explicit gaps, not silently dropped).
- Bump `sleap-roots-contracts` pin `0.1.0a5` → `0.1.0a6` (adds `LabelCard`; relock `uv.lock`).
  Confirmed safe against the live registry — no `model-management` spec/code change needed
  (`WandbRegistrySource.list_cards()` already isolates a malformed artifact with a warning
  rather than aborting the listing; predates and already satisfies predict#32, which closes as
  a comment, not a code change).
- Add a `parity` pytest marker (mirrors `gpu`/`acceptance`/`wandb`: deselected by default, gated
  on env vars for the network-share root and `WANDB_API_KEY`, skips cleanly without them).
- Run the harness, observe the real sleap-nn-vs-classic-SLEAP deltas across the resolvable
  production models, and set the tolerance from that empirical baseline (not an a priori guess).

## Impact

- Affected specs: `prediction-parity` (ADDED)
- Affected code: `pyproject.toml`, `uv.lock`, `sleap_roots_predict/parity.py` (new),
  `tests/test_parity.py` (new), a new ground-truth manifest file, `CHANGELOG.md`,
  `openspec/project.md` (contracts version literal + a new module bullet).
- No behavior change for existing callers of `sleap_roots_predict`'s public API.
- Closes sleap-roots-pipeline#15 (after recording the decided tolerance + measured baseline in
  a closing comment) and predict#32 (comment only). Cross-links predict#8 (shares the
  reference set + `parity.py`) and `sleap-roots-training`#10/#11/#17/#22 (this repo is a
  downstream `LabelCard` consumer; their backfill/publish-labels work is not performed here).
- `docker-build.yml`'s PR trigger watches `pyproject.toml`/`uv.lock`/`sleap_roots_predict/**`,
  so this PR runs its build-only validation job automatically — expected.
