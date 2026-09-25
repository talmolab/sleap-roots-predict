# Design — adopt-per-run-run-manifest-reader

Rationale lives in `docs/superpowers/specs/2026-09-25-per-run-run-manifest-reader-design.md`
(D-numbers below); the cross-repo design of record is
`sleap-roots-pipeline/docs/superpowers/specs/2026-09-21-per-run-run-manifest-identity-design.md`
("srp §x"). This file is the decision index a reviewer of this repo's diff needs.

## Decisions

1. **Resolve once per batch via `load_run_manifest` (D1).** Private
   `run_manifest._resolve_run_manifest(input_dir, pipeline_run_id)` wraps
   `load_run_manifest(..., allow_legacy=True)` plus two log-only diagnostics (4). `run_batch`
   checks the input directory exists, then resolves once with `pipeline_run_id_from_env()`;
   discovery scopes to `.manifest.scan_keys`; the forward publishes `.read`. `_ManifestSnapshot` /
   `_read_manifest_snapshot` are deleted. Private keywords are renamed: `discover_scans(...,
   manifest_bytes=)` → `manifest=` (a `LoadedRunManifest | None`), `copy_run_manifest_forward(...,
   snapshot=)` → `read=` (a `RunManifestRead | None`). No public parameters change.
2. **Fail loud with a run id (D2).** `RunManifestMissingError`, `RunManifestIdentityError` and an
   unusable-id `ValueError` propagate before any model-source call or output-directory write.
   `__main__` adds `RunManifestError` to its staging-error tuple (D3) → `Batch aborted: …`, exit 1.
3. **Forward under the name read (D4).** Destination `output_dir / read.filename`; the same-file
   check compares `input_dir / read.filename` with the destination. The `mkstemp` temp prefix
   changes from `.run_manifest.json.` to `.{read.filename}.`. Standalone calls read with the
   non-parsing `read_run_manifest` inside the existing logged `try`; a failure before the read
   completes is logged without naming a filename (none is known yet).
4. **Diagnostics never change scope (D5).** Kept: stale-output warning, keyed to
   `RUN_MANIFEST_FILENAME` (reachable only with no run id). Added: (a) read not per-run while
   `run_manifest.*.json` files sit unread at the top level of `input_dir`; (b) legacy manifest
   read under a known id names a different run. Under Argo the old bloomctl rewrites the legacy
   file's `pipeline_run_id` to the current run on every merge, so (b) fires only when a different
   workflow merged into the directory afterwards — a genuine contamination signal, not noise.
5. **A directory or other unreadable entry at a manifest path raises (D6)** — `IsADirectoryError`
   on POSIX, `PermissionError` on Windows; tests assert `OSError`.
6. **predict#43: name-only unique temp path, not `mkstemp` (D7).** `_unique_tmp_path(dst)` →
   `dst.with_name(f".{dst.name}.{uuid4().hex[:16]}.tmp")`. `mkstemp` would leave the sidecar copy
   (`shutil.copyfile`) and `predictions.json` (`write_text`) at `0600` — both open the reserved
   file with `O_TRUNC`, keeping its mode — and at the `.slp` site sio unlinks an existing path
   before creating it, discarding any `O_EXCL` reservation. A name-only helper keeps each site's
   create → `os.replace` → unlink-on-failure shape, so modes are unchanged by construction.

## Risks / trade-offs

- **Fallback makes fail-loud inert during rollout.** A stale legacy file satisfies
  `allow_legacy=True`; mitigated by srp §4 steps 5–6 (delete stale files; srp#82), and made
  visible meanwhile by diagnostic (b).
- **SIGKILL orphans accumulate (#43).** A killed write leaves a hidden `.{name}.<hex>.tmp` that no
  later run overwrites (the fixed `.tmp` name used to be). No consumer glob matches it; reclaiming
  it automatically would risk deleting a concurrent writer's live temp. Accepted and documented.
- **Forward-copy temp name over `NAME_MAX`** for a run id over ~227 characters → `OSError`, exit 1.
  Argo ids are ~26 characters.
- **Rollback floor.** After bloomctl flips the writer, an image older than this change reads only
  the stale accumulated legacy manifest and forwards it, re-scoping traits to the union. Roll the
  writer back first (README rollback note).
- **Deploy coupling.** Must not share a deploy with C2 (srp#89).

## Migration

None in this repo. Rollout per srp §4: merge → release → predictor pin bump in
`sleap-roots-pipeline` (after C2 is confirmed; the template's "inert today" comment on
`ARGO_WORKFLOW_NAME` goes stale then) → traits pin bump (its current `sha-689cffb` predates
sleap-roots #269) → bloomctl writer flip → template update → delete stale `a4_poc` manifests →
srp#82.
