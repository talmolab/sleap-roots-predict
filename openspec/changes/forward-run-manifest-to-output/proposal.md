## Why

`sleap-roots-predict` issue #39: `discover_scans` reads `run_manifest.json` from `input_dir` to
scope its own work, but nothing in this package ever writes that manifest into `output_dir`.
Since predict's `output_dir` **is** the downstream trait-extraction stage's `input_dir`, that
stage finds no manifest at the path it looks and activates its own "no manifest → unscoped
recursive discovery" fallback on every single run — reprocessing unrelated scans that happen to
share the output tree.

The blast radius is three stages, not one: trait-extraction's own manifest forward-copy sits
*inside* its `if scope is not None:` block, so no manifest reaches `traits/` either, and
`bloomctl`'s write-back stage is likewise unscoped — writing stale envelopes into Bloom's
production tables. Evidence and history are in `design.md` (§Context).

## What Changes

- New module `sleap_roots_predict/run_manifest.py` exposing
  `copy_run_manifest_forward(input_dir, output_dir)`, mirroring the reference implementation
  `trait_extractor/run_manifest.py::copy_run_manifest_forward` in `sleap-roots`:
  - **Raw byte copy**, not a re-serialization through the frozen `RunManifest` model, and no
    validation of contents (discovery has already validated, and as a public function the copy is
    content-agnostic by design).
  - **Absent → no-op**, preserving the unscoped fallback for local/dev/test runs. Presence is
    checked *before* path identity, so a nonexistent source never raises. If the output
    directory nonetheless already holds a manifest from an earlier run, it is **left in place
    and a warning is logged** — predict cannot distinguish a leftover from a concurrent
    invocation's file, but left silent it would scope the downstream stage to an earlier run's
    `scan_keys`. Argued in `design.md` §Decision 5.
  - **Same-file → no-op**, detected such that a symlinked or bind-mounted `output_dir` is caught
    (not path-string equality).
  - **Atomic**: temp file + replace, with the temp file created inside `output_dir` (a
    cross-device replace would fail on the production mount) under a name **unique to the writing
    process** — a fixed temp name is unsafe here because the destination is shared across
    concurrent invocations. Temp file cleaned up on failure.
  - Logs both directories before raising, since not every `OSError` carries a filename.
- `run_batch` calls it after `discover_scans` and the empty-batch guard, **before** the scan loop
  and before constructing the `WarmModelWorker`; and a copy failure **raises** rather than being
  best-effort, deliberately diverging from the reference implementation. Both choices are argued
  in `design.md` §Decision 2.
- `__main__.py`'s handler widens from `except (FileNotFoundError, ValueError)` to
  `except (OSError, ValueError)`. A copy failure raises `PermissionError`/`OSError`, which is a
  *sibling* of `FileNotFoundError` under `OSError`, not a subclass — so without this it would
  bypass the clean one-line staging-error log the exit-code requirement mandates and surface a
  bare traceback. The exit code itself (`1`) is already correct and unchanged.
- Export `copy_run_manifest_forward` from the package root (`__init__.py` + `__all__`).

## Non-Goals

- **Concurrency-safe merge** of the manifest's contents (`bloomctl` does a union-merge under a
  lockfile; this hop overwrites). Argued in `design.md` §Decision 1; a follow-up issue tracks
  promoting that merge+lock into `sleap-roots-contracts` for all three hops at once. Note the
  unique-temp-name rule above is a separate, narrower concurrency fix that *is* in scope, because
  without it concurrent writers can publish a truncated manifest.
- Changing trait-extractor's own naive `copy_run_manifest_forward` (`sleap-roots` repo).
- Per-run manifest addressing (e.g. `run_manifest.{pipeline_run_id}.json`), which would be a
  cross-repo contract change to predict, traits, and the Argo templates.

## Impact

- Affected specs: `predict-container`
  - **ADDED**: Run-manifest forward-copy to the output directory
  - **MODIFIED**: Per-scan failure isolation and batch exit code (adds the forward-copy failure to
    the exit-`1` enumeration and widens the CLI's clean-log catch to `OSError`)
  - **MODIFIED**: Graceful SIGTERM handling for Argo preemption (scopes the `143` override to the
    batch's own outcome, so a pre-flight staging error still propagates as that error)
- Affected code: `sleap_roots_predict/run_manifest.py` (new), `sleap_roots_predict/batch.py`,
  `sleap_roots_predict/__main__.py`, `sleap_roots_predict/__init__.py`,
  `tests/test_run_manifest.py` (new), `tests/test_batch.py`, `tests/test_public_api.py`
- Affected docs: `API.md`, `README.md` (prose, module tree, and tests tree), `CHANGELOG.md`,
  `openspec/project.md`
- No dependency change: `RUN_MANIFEST_FILENAME` already comes from the pinned
  `sleap-roots-contracts==0.1.0a7`. The new module imports only that constant, not `RunManifest`.

## Deployment notes

- **The forwarded manifest is sticky on rollback.** Rolling the predict image back does not
  remove `predictions/run_manifest.json`; a stale one left in place would scope every subsequent
  traits run to a frozen `scan_keys` set — silent *under*-processing, the mirror image of #39 and
  harder to detect. Any rollback MUST also delete that file.
- **Deployment ordering matters, and the fix is inert against the currently pinned image.**
  `sleap-roots-pipeline` main pins trait-extraction to an image that predates `sleap-roots#263`
  — the change that added manifest reading. Against that pin, this forward-copy is a complete
  no-op and a green run proves nothing. Sequence: land and apply the trait-extraction pin bump
  (PR `#57`, still open) **first**, then bump predict.
- **Downstream failure classification shifts in one case.** A batch that forwards a manifest but
  writes zero predictions (every scan failed, or an immediate stop) previously left traits with
  no manifest and no outputs, which traits treats as a staging error (exit `1`). With a manifest
  present, traits instead produces one failed entry per `scan_key` and exits `3`. Note this is a
  change in *reported classification*, not in Argo's behavior: the deployed template is
  `retryPolicy: Always` with no exit-code discrimination, so `1` and `3` are retried identically
  today (`sleap-roots-pipeline#56` tracks wiring that up).
- **The first scoped run will look like a regression.** Once the manifest is honored end to end,
  leftover scans in the shared tree stop being reprocessed — so traits output and DB rows stop
  changing for them. That is the fix working. Trait-extraction emits the corroborating signal:
  a warning naming pre-existing result files "outside this run's scope (from a prior run's wider
  manifest)". Capture that log line as expected-behavior evidence.
- **Scope is the accumulated staged set, not one run.** `images_input/` is a fixed shared
  hostPath and bloomctl *unions* `scan_keys` into its manifest across every invocation, so the
  forwarded scope covers everything ever staged there. This fix removes pre-manifest leftovers
  (which is what #39's `scan_1009` is); it does not make each run's scope minimal.
