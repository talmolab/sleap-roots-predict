## Context

Full design rationale lives in
`docs/superpowers/specs/2026-07-05-flip-default-live-registry-design.md` (the brainstorming
output). This file captures only the decisions that shape the spec deltas.

The `model-management` capability already ships `WandbRegistrySource` (network confined,
lazy `import wandb`, alias→concrete-version pinning), `LocalCardSource` (offline), and
`WarmModelWorker` (fetch-once/load-once/reuse, fail-loud). This change flips the *default*
path onto the live registry and hardens `list_cards()` against a single bad artifact.

## Goals / Non-Goals

- **Goals:** live registry as the out-of-the-box source with only `WANDB_API_KEY` set;
  env-var names model-scoped to match the producer; one malformed artifact skips-with-warning.
- **Non-Goals:** #7 provenance-config reshape; #14 checksum dedupe; #13 wandb floor; #10
  `sleap_nn_version` mismatch; the parity gate. `python-dotenv` auto-loading (the service
  uses real k8s env/secrets).

## Decisions

- **Default source, fail-loud (no offline fallback).** `WarmModelWorker(source=None)` →
  `WandbRegistrySource()`; missing `WANDB_API_KEY` raises when cards are listed. A silent
  `LocalCardSource` fallback would hide a misconfigured production deploy, contradicting the
  worker's fetch-once/fail-loud posture. `LocalCardSource` stays an explicit opt-in.
- **Hard rename (no deprecated alias).** Package is alpha (`0.0.1a0`); the old names had no
  default and were referenced only by one gated test + docs, so a clean break beats
  transitional dual-read code. Alternative (read both for one release) rejected as
  needless carrying cost.
- **Default registry via constant + env.** `_DEFAULT_REGISTRY = "sleap-roots-models"`;
  `registry or os.environ.get("SRP_WANDB_MODEL_REGISTRY", _DEFAULT_REGISTRY)`.
- **Skip-and-warn via a pure collector seam.** Split `list_cards()` into
  `_iter_registry_artifacts(api)` (network) and `_collect_cards(artifacts)` (pure). The
  collector applies the alias filter, then wraps `_card_from_artifact` in
  `try/except Exception` → `logger.warning(...)` + continue. The pure seam lets the
  skip-and-warn behavior be unit-tested offline with duck-typed fake artifact objects
  (no wandb-boundary mocks). Warning uses a module `logging.getLogger(__name__)` (repo
  convention), asserted with `caplog`.
- **Isolation is scoped to card construction only — not the traversal.** The `try/except`
  wraps *only* `_card_from_artifact(artifact)`, and `_require_key()` stays at the top of
  `list_cards()` before any network. So genuine failures (missing credentials, registry /
  network / pagination errors) propagate fail-loud; only a single non-conforming artifact's
  card build is skipped. This avoids the trap of a broad `except` silently dropping a
  production model and caching a degraded catalog (the worker caches the card list once) —
  which would undercut the fail-loud posture. The warning includes the underlying error, so
  a skip is diagnosable. `except Exception` (not `BaseException`) keeps
  `KeyboardInterrupt`/`SystemExit` propagating.
- **DRY env docs: README is the single canonical table.** The env-var table lives once in
  the operator-facing `README.md`; `openspec/project.md` carries a 1–2 line pointer (not a
  second table). `.env.example` is the executable companion (a different artifact — a file
  you `cp`). A consistency test asserts the `.env.example` var set equals the README table's
  var set and that neither names the legacy vars — so the two prose surfaces cannot drift.
  Docstrings name only the vars relevant to that constructor, with their defaults.
- **Count-agnostic gated assertions.** The production card set grows (arabidopsis `plate`
  deferred to training #3), so the gated test asserts non-empty + conforming, never an
  exact count.
- **Doc home is OpenSpec + README, not CLAUDE.md** (CLAUDE.md is being retired). The
  `.env.example` is documentation-by-example, not auto-loaded.

## Risks / Trade-offs

- **Breaking env rename** → mitigated by alpha status + the new default (most operators
  need set nothing beyond `WANDB_API_KEY`); the README/`.env.example` make the surface
  discoverable; training #6 tracks the producer-side doc ripple.
- **Broad `except Exception` in the collector** → justified: the whole point is resilience
  to *any* single malformed artifact. `Exception` (not `BaseException`) still lets
  `KeyboardInterrupt`/`SystemExit` propagate. Each skip is logged, so silent data loss is
  avoided.

## Migration

Operators/CI setting `SRP_WANDB_REGISTRY` / `SRP_WANDB_ALIAS` must switch to
`SRP_WANDB_MODEL_REGISTRY` / `SRP_WANDB_MODEL_ALIAS` — or, since `sleap-roots-models` is
now the default, simply unset them. No code migration for callers passing an explicit
`source` (the new `source=None` default is additive/backward-compatible).

## Open Questions

None — the three brainstorming decisions (default+fail-loud, hard rename, defer #7) are
settled.
