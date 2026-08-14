# Consuming RunManifest: scoped discovery + real idempotency-key skip-if-done

## Context

Part of `sleap-roots-pipeline` issue #37's cross-repo idempotency chain (see that repo's
`docs/superpowers/specs/2026-08-03-manifest-scoped-processing-redesign.md`, the canonical design
rationale). Two incidents motivate this:

1. A leftover scan directory from a prior run got reprocessed by predict's directory-wide
   `discover_scans` scan, even though it wasn't in the current run's scope.
2. A stale prediction output blocked a corrected sidecar from propagating on a later run, because
   `run_batch`'s skip-if-done is a plain `Path.exists()` check, not an identity comparison.

`sleap-roots-contracts` 0.1.0a7 now ships `RunManifest` (written by `bloomctl` into the same
shared, already-mounted staging directory as the per-scan `{scan_key}.scan_metadata.json`
sidecars — a discoverable file, not a CLI arg, confirmed against this repo's and traits'
`argparse` signatures, both fixed two-positional-arg entrypoints). This change makes
`sleap-roots-predict` consume it.

## Goals

- `discover_scans` scopes to exactly `RunManifest.scan_keys` when a manifest is present.
- `run_batch`'s skip-if-done compares a real identity key, not just file existence.
- Preserve every existing behavior when no manifest is present (standalone/local/test usage).

## Non-goals (explicitly out of scope)

- `sleap-roots`/traits consuming the manifest (separate repo/handoff).
- write-back's identical unscoped-glob bug (bloom #678, fixed in bloomcli).
- Any change to `sleap-roots-contracts`. Not needed: every input the idempotency key requires is
  already recoverable from artifacts predict already writes (see below) — no schema addition
  earns its keep here.

## Confirmed decisions (from clarifying questions)

| Question | Decision |
|---|---|
| `run_manifest.json` absent from `input_dir` | Fall back to today's unscoped `rglob` — scoping only activates when a manifest is present. Keeps every existing test/local/standalone usage unchanged. |
| Sidecar on disk, `scan_key` NOT in manifest | Silently excluded from discovery (no `ScanInput`, no `ScanResult`); a single debug-level log line reports the count skipped. |
| Manifest `scan_key` with NO matching sidecar on disk | Isolated failure: a synthetic `ScanInput` with `.error` set (reuses the existing per-scan error → `ScanResult(status="failed")` path; batch continues). |
| Idempotency-key derivation | Reuse `sleap_roots_contracts.identity.compute_idempotency_key` directly (present in the installed package, just not re-exported in `__all__`), passing `traits_code_sha=""` as a fixed placeholder — predict only ever compares against its own previously-stored key, never against a traits-computed one, so the placeholder just needs to be consistent across runs. Avoids a second hashing implementation (this program's own contracts docs warn a duplicate "would silently break first-writer-wins idempotency"). |
| Malformed `run_manifest.json` (bad JSON / fails `RunManifest` validation) | Batch-level `ValueError`/`ValidationError`, not a per-scan failure — mirrors the existing duplicate-`scan_key` raise, since an invalid manifest means scope itself can't be trusted. |

## Data flow

```
input_dir/
├── run_manifest.json                 # RunManifest{schema_version, pipeline_run_id, scan_keys}
├── scan_1009/
│   ├── scan_1009.scan_metadata.json  # {scan_key, image_ids, images_checksum, params}
│   └── frame_*.png
└── scan_1010/  (leftover from a prior run, NOT in scan_keys -> excluded)
    └── ...
```

`discover_scans`:
1. Look for `input_dir / RUN_MANIFEST_FILENAME` (`"run_manifest.json"`, the constant imported
   from contracts, not hardcoded).
2. If present: `RunManifest.model_validate_json(...)` (raises on malformed content — batch-level
   abort). Build the sidecar set as today (`rglob`, dup-`scan_key` check), but only keep sidecars
   whose `scan_key ∈ manifest.scan_keys`; for any `scan_key` in the manifest with no discovered
   sidecar, append a synthetic error `ScanInput` (`error="no sidecar found for manifest scan_key
   {key!r}"`).
3. If absent: unchanged — full unscoped `rglob`.

`ScanInput` gains one field: `images_checksum: str = ""`, read from the sidecar's existing
`images_checksum` key (already present in every sidecar fixture today, just previously ignored).

`run_batch`'s per-scan loop, restructured:
1. `refs = worker.resolve(scan.params)` — moved up from `_predict_one`; cheap (matches against
   the already-listed `ModelCard`s, no network/download), safe to call before the skip decision.
2. `current_key = _identity_key(scan_key, images_checksum=scan.images_checksum, refs, param_hash=scan.params.param_hash, predict_code_sha=resolved_code_sha, output_params=worker.output_params())` —
   a thin wrapper around `compute_idempotency_key` (see below).
3. `previous_key = _stored_identity_key(out_scan_dir, scan.scan_key)` — `None` if nothing usable
   is on disk yet (see below).
4. `previous_key == current_key` → `skipped`. Otherwise (differs, or `None`) → predict, write
   outputs as today.

## No new storage — recompute both sides from artifacts predict already writes

`PredictionManifest` (from `sleap-roots-contracts`) has no `idempotency_key`/`param_hash` field,
and extending it would be a separate repo's contracts PR. It doesn't need to be: everything
`compute_idempotency_key` needs for the *previous* run is already sitting in `out_scan_dir` from
that run, because `write_prediction_outputs` writes the manifest and `run_batch` copies the
sidecar verbatim:

| Input | Recovered from |
|---|---|
| `scan_key` | the scan itself (identical either way) |
| `images_checksum` | `out_scan_dir/{scan_key}.scan_metadata.json`'s `images_checksum` (the previously-copied sidecar, byte-identical to what was actually used) |
| `param_hash` | recomputed via `compute_param_hash` over that same old sidecar's `params` — pure function, so this reproduces the exact value used at write time |
| `models` | `out_scan_dir/{scan_key}.predictions.json`'s `artifacts[].model` (the `ModelRef`s actually used) |
| `predict_code_sha` / `predict_output_params` | the same manifest's `predict_code_sha` / `predict_output_params` fields |

So `_stored_identity_key(out_scan_dir, scan_key)` just reads those two existing files (returning
`None` if either is missing/unreadable — e.g. a first run, or a crash left a partial write — which
naturally falls through to "re-run," matching the existing "prefer re-run over serving stale/
incomplete data" philosophy already baked into the sidecar-before-manifest ordering) and calls the
same `compute_idempotency_key` used for the current side. No new file, no contracts change, and
the exact same trick is available to traits later (it reads the identical two files plus its own
`traits_code_sha`).

## Error handling summary

- Malformed manifest → raises (batch-level, before any scan is touched).
- Sidecar not in scope → excluded silently (debug log only).
- Manifest scan_key with no sidecar → `ScanResult(status="failed")`, batch continues.
- Zero models resolved → unchanged, still `failed` (check happens right after the moved-up
  `resolve()` call).
- No previous manifest/sidecar on disk yet, or either is unreadable → `previous_key = None` →
  treated as changed → predict runs (first run, or a prior crash mid-write; never silently
  skips on ambiguous state).
- Everything else (missing input_dir, duplicate scan_key, unreadable sidecar, missing params) →
  unchanged.

## Testing plan (TDD)

New/changed tests in `tests/test_batch.py` (or a new `tests/test_run_manifest.py` if it reads
cleaner split out):

1. `discover_scans` scopes to `run_manifest.json`'s `scan_keys` when present; an extra sidecar
   outside scope is excluded (not returned, not an error).
2. `discover_scans` falls back to unscoped `rglob` when no manifest is present (regression guard
   for every existing test/fixture).
3. A manifest `scan_key` with no matching sidecar becomes a `failed` `ScanResult`.
4. A malformed `run_manifest.json` raises before any scan is processed.
5. `run_batch` re-run with an unchanged scan (same sidecar, same models, same
   `predict_code_sha`) skips via key match (replaces/extends
   `test_rerun_skips_completed_scan`).
6. `run_batch` re-run with a **changed** sidecar (different `params` → different `param_hash`, or
   different `images_checksum`) **re-predicts** rather than skipping — the actual incident-2 fix.
7. `run_batch` re-run with a different `SRP_PREDICT_CODE_SHA` re-predicts.
8. `pyproject.toml` bumped to `sleap-roots-contracts==0.1.0a7`; full existing suite still green
   (regression gate for the version bump itself).
9. A first run (nothing in `out_scan_dir` yet) always predicts — `_stored_identity_key` returns
   `None`.

## Open note (non-blocking)

`compute_idempotency_key` is not in `sleap_roots_contracts.__all__` — it's imported from its
submodule (`sleap_roots_contracts.identity`) rather than the package root. Worth flagging to the
contracts maintainers as a candidate for a future `__all__` addition, but not blocking (submodule
import is stable, already-shipped code).
