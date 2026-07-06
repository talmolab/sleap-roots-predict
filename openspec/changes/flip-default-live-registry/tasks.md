> **Test hygiene (applies to every offline env test below):** use the `monkeypatch`
> fixture only, and `delenv(..., raising=False)` the **entire** family up front —
> `WANDB_API_KEY`, `SRP_WANDB_MODEL_REGISTRY`, `SRP_WANDB_REGISTRY`,
> `SRP_WANDB_MODEL_ALIAS`, `SRP_WANDB_ALIAS`, `SRP_WANDB_ENTITY` — before setting only the
> var under test. Without this, "no env" assertions false-fail on a dev box, and a
> missing-key test can make a **real network call** on a machine that has `WANDB_API_KEY`
> set. Default-suite command (matches CI): `uv run pytest -m "not gpu and not acceptance and not wandb"`.

## 1. Env-var rename + default registry (predict #11)

- [ ] 1.1 **Test first** (offline, `tests/test_model_registry.py`), each test delenv'ing the
      full env family (see banner):
      - `WandbRegistrySource()` → `._registry == "sleap-roots-models"`.
      - `SRP_WANDB_MODEL_REGISTRY="x"` set → `._registry == "x"` (honored).
      - legacy `SRP_WANDB_REGISTRY="legacy"` set with `SRP_WANDB_MODEL_REGISTRY` unset →
        `._registry == "sleap-roots-models"` (legacy **ignored**).
      - `SRP_WANDB_MODEL_ALIAS="staging"` set → `._alias == "staging"` (honored).
      - legacy `SRP_WANDB_ALIAS="legacy"` set with `SRP_WANDB_MODEL_ALIAS` unset →
        `._alias == "production"` (legacy **ignored**).
      - `WandbRegistrySource()` (default registry) with `WANDB_API_KEY` delenv'd →
        `list_cards()` raises `RuntimeError` matching `WANDB_API_KEY` before any network call
        (default-registry construction still fails loud on creds).
- [ ] 1.2 Implement in `model_registry.py`: add `_DEFAULT_REGISTRY = "sleap-roots-models"`;
      `self._registry = registry or os.environ.get("SRP_WANDB_MODEL_REGISTRY", _DEFAULT_REGISTRY)`;
      `self._alias = alias or os.environ.get("SRP_WANDB_MODEL_ALIAS", _DEFAULT_ALIAS)`; drop the
      legacy `SRP_WANDB_REGISTRY`/`SRP_WANDB_ALIAS` reads. Update the `_registry_project` error
      message to name `SRP_WANDB_MODEL_REGISTRY` (keep the guard — `registry=""` still reaches it).
      Update docstrings to state the **new default** (`SRP_WANDB_MODEL_REGISTRY` → then
      `sleap-roots-models`; `SRP_WANDB_MODEL_ALIAS` → then `production`) and fix the class
      docstring's pre-existing omission of the alias env var (lines ~86, 108, 109).
- [ ] 1.3 Run the offline tests → green; `/lint`.

## 2. Per-artifact error isolation in `list_cards()` (predict #12)

- [ ] 2.1 **Test first** (offline; construct `WandbRegistrySource(alias="production")`, no key
      needed; use duck-typed fake artifacts each carrying `aliases=["production"]`, `metadata`,
      `name`/`qualified_name`, `version`, `digest`):
      - `_collect_cards([good, malformed])` → `[good_card]`; exactly one `logging.WARNING`
        (via `caplog`) naming the bad artifact **and** including the error text; no raise.
        (`malformed` = metadata missing a required selection field → pydantic `ValidationError`.)
      - `_collect_cards([malformed])` → `[]`, one WARNING, no raise (all-bad is empty, not fatal).
      - `_collect_cards([good_a, malformed, good_b])` → `[good_a, good_b]` in order, exactly one
        WARNING (one bad artifact drops only itself).
      - `_collect_cards([wrong_alias])` (fake whose `.aliases` lacks the configured alias) → `[]`
        and **no** WARNING (alias-filtered ≠ malformed).
      - The returned good card's `version`/`weights_checksum` equal the fake's `.version`/`.digest`
        (concrete pin, offline coverage of the alias-pinning scenario).
- [ ] 2.2 Implement: add module `logger = logging.getLogger(__name__)`; split `list_cards()` into
      `_iter_registry_artifacts(api)` (the network `artifact_collections`/`artifacts` traversal)
      and `_collect_cards(artifacts)` (pure). In `_collect_cards`: apply the alias filter first,
      then wrap **only** `_card_from_artifact(artifact)` in `try/except Exception` →
      `logger.warning("Skipping non-conforming model artifact %r: %s", label, err)` + `continue`.
      Keep `_require_key()` at the **top** of `list_cards()`, before `_iter_registry_artifacts` is
      iterated (do not move it inside the generator) — the existing
      `test_wandb_source_missing_key_raises_before_network` guards this ordering.
- [ ] 2.3 Run the offline tests → green; confirm `test_wandb_source_missing_key_raises_before_network`
      still passes; `/lint`.

## 3. Warm worker default source (predict #11)

- [ ] 3.1 **Test first** (offline, `tests/test_warm_worker.py`; delenv the env family):
      - `WarmModelWorker()._source` is a `WandbRegistrySource`.
      - With `WANDB_API_KEY` delenv'd, `WarmModelWorker()` **construction does not raise**
        (network-free), and the first `resolve(params)` raises `RuntimeError` matching
        `WANDB_API_KEY` (fail-loud on first use, no offline fallback).
- [ ] 3.2 Implement in `warm_worker.py`: `source: Optional[ModelCardSource] = None`, and in the
      **body** `if source is None: source = WandbRegistrySource()` (never a mutable/eager default in
      the signature — that would freeze env at import and share one instance); import
      `WandbRegistrySource`.
- [ ] 3.3 Run offline tests → green; confirm the existing `LocalCardSource`-based warm-worker tests
      still pass (positional `WarmModelWorker(source)` backward compat); `/lint`.

## 4. Gated wandb test update (predict #11)

- [ ] 4.1 Update `test_wandb_source_lists_and_materializes`: `monkeypatch.delenv` the legacy +
      new `SRP_WANDB_MODEL_REGISTRY` (proving the default path needs no registry env); leave
      `SRP_WANDB_ENTITY` unset (default lab entity hosts the registry) or set it explicitly. Keep
      assertions **count-agnostic** (non-empty + all conforming; do NOT assert exactly 13).
- [ ] 4.2 If `WANDB_API_KEY` + live registry are available, run `uv run pytest -m wandb -q`
      → green; otherwise confirm it skips cleanly at collection time.

## 5. Env-var docs + `.env.example` + CHANGELOG (predict #11)

- [ ] 5.1 **Test first**: a consistency test that (a) `.env.example` mentions
      `SRP_WANDB_MODEL_REGISTRY` and `SRP_WANDB_MODEL_ALIAS` and **not** the legacy
      `SRP_WANDB_REGISTRY`/`SRP_WANDB_ALIAS`, and (b) the README "Configuration" table's var set
      equals the `.env.example` var set (guards both prose surfaces against drift). Resolve files
      via `Path(__file__).parents[1]`, not cwd.
- [ ] 5.2 Add `.env.example`: `WANDB_API_KEY=` uncommented; every optional var commented out with
      its default shown (`SRP_WANDB_ENTITY`, `SRP_WANDB_MODEL_REGISTRY`, `SRP_WANDB_MODEL_ALIAS`,
      `SRP_MODEL_CACHE_DIR`, `SRP_DEVICE`). Keep typo-clean (codespell scans it).
- [ ] 5.3 Docs: add a **"Configuration"** section to `README.md` with the required-vs-optional +
      defaults table (the single canonical table; default strings must match `_DEFAULT_ENTITY` /
      `_DEFAULT_REGISTRY` / `_DEFAULT_ALIAS` in code). In `openspec/project.md`, add a **1–2 line
      pointer** to that README section (not a second table — DRY). Correct the **one** stale
      old-name reference in `CLAUDE.md` (line 71: `SRP_WANDB_REGISTRY` → `SRP_WANDB_MODEL_REGISTRY`;
      note `SRP_WANDB_ALIAS` is **not** present there) — no new content added to CLAUDE.md.
- [ ] 5.4 Run the consistency test → green; `codespell` + `/lint`.
- [ ] 5.5 Add a `CHANGELOG.md` `[Unreleased]` entry under `### Changed (BREAKING)`: the env-var
      rename (old names no longer read), the `sleap-roots-models` default, `WarmModelWorker(source=None)`
      reading the live registry, and fail-loud-on-missing-key (no offline fallback).

## 6. Full gate + report-back

- [ ] 6.1 `/pre-merge` (format check + lint + full test suite + build); confirm the pytest step keeps
      `not wandb` deselected; run the `gpu` subset if on an accelerator.
- [ ] 6.2 Set every task above to `- [x]`; prepare the report-back note (close predict #11 + #12;
      advance training #6 and pipeline roadmap row A3-predict).
