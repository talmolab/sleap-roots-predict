# Adopt the contracts 0.1.0a9 per-run run-manifest reader — design

**Date:** 2026-09-25
**Tracker:** talmolab/sleap-roots-pipeline#71 (root cause behind srp#37); this is predict's
reader half, rollout step 2 of the design of record.
**Also fixes:** talmolab/sleap-roots-predict#43 (fixed temp-file names in per-scan writes).
**Design of record:** `sleap-roots-pipeline/docs/superpowers/specs/2026-09-21-per-run-run-manifest-identity-design.md`
(srp PR #80, merged 2026-09-23; PR #87 still open and touches only §2.7/§4 0a, neither of which
changes predict's reader). Sections cited as "srp §x.y".
**Reference implementation:** talmolab/sleap-roots PR #269 (merged 2026-09-24, archived #270) —
`trait_extractor/extractor.py` `_resolve_run_manifest`, `trait_extractor/__main__.py`.

## Verified state (2026-09-25)

- Predict pins `sleap-roots-contracts==0.1.0a9` (PR #45, merged 2026-09-24); the installed
  package exports `load_run_manifest`, `read_run_manifest`, `pipeline_run_id_from_env`,
  `RunManifestRead`, `LoadedRunManifest`, `RunManifestError`, `RunManifestMissingError`,
  `RunManifestIdentityError`. No dependency change is needed.
- `grep ARGO_WORKFLOW_NAME sleap_roots_predict/` returns nothing: predict has never read its run id.
- The cluster `sleap-roots-predictor-template.yaml` on srp `origin/main` already sets
  `ARGO_WORKFLOW_NAME: "{{workflow.name}}"` (commented "inert today — predict does not read it
  yet"). The `local-WSL2-*` templates deliberately do not (srp §2.3).
- Predict reads `RUN_MANIFEST_FILENAME` in `run_manifest.py` (`_read_manifest_snapshot`,
  `copy_run_manifest_forward`) and `batch.py` (`discover_scans`, `run_batch`).

## Goal and success criteria

Predict resolves this run's manifest by the contract's policy, scopes discovery to it, and
forwards it under the name it read. Success:

1. The measured contamination (a 1-scan request scoped by a 12-key accumulated manifest) is
   reproduced in a test, and a per-run manifest stops it.
2. A legacy `run_manifest.json` is still honored while `allow_legacy=True`.
3. A stage that knows its run id and finds no manifest fails the batch (exit 1) rather than
   falling back to unscoped discovery (srp §2.2).
4. With no run id (local, `local-WSL2-*`, the test suite) behavior is byte-for-byte unchanged;
   `tests/assets/scans/` still stages no manifest.
5. No production behavior change on deploy: the old bloomctl still writes only the legacy name,
   which predict still reads (srp §4 step 2).

## Decisions

### D1. Resolution goes through `load_run_manifest`, once per batch

A private `run_manifest._resolve_run_manifest(input_dir, pipeline_run_id)` calls
`load_run_manifest(input_dir, pipeline_run_id, allow_legacy=True)` — the single literal
`allow_legacy=True` a future srp#82 flip changes for scoping. `run_batch` resolves the id with
`pipeline_run_id_from_env()` (the contract's single definition of "which run am I", srp §3.2) and
calls the resolver exactly once; discovery scopes against `loaded.manifest.scan_keys` and the
forward publishes `loaded.read.data`. This preserves the existing single-snapshot property: the
bytes forwarded are the bytes scoped against.

`_ManifestSnapshot` and `_read_manifest_snapshot` are deleted; `RunManifestRead` carries the same
`(data, mode)` plus `filename` and `is_per_run`, and its `mode` is taken by `fstat` on the open
descriptor — the same "no second stat" guarantee the snapshot existed to give.

No public `pipeline_run_id` parameter is added to `run_batch`/`discover_scans`: the environment
is the contract, and tests use `monkeypatch`.

### D2. Fail loud where the run id is known

`RunManifestMissingError` (id known, no candidate) and `RunManifestIdentityError` (per-run file
names a different run) propagate out of `run_batch` before any model-source interaction or
prediction, and before the output directory is touched. An unusable `ARGO_WORKFLOW_NAME` raises
`ValueError` from contracts. All three are batch-level staging errors → exit 1.

### D3. CLI: catch `RunManifestError` alongside `(OSError, ValueError)`

The two contract errors deliberately do not subclass `ValueError` (so a generic parse handler
cannot swallow them). `__main__` adds `RunManifestError` to its except tuple so they get the same
one-line `Batch aborted: …` log before re-raising (exit 1), matching traits.

### D4. Forward under the name read

`copy_run_manifest_forward` publishes to `output_dir / read.filename` (srp §2.5). Everything else
about the copy is unchanged: raw bytes, `mkstemp` in the destination directory, `chmod` to
`read.mode` **before** `os.replace`, cleanup of the temp file and any directory the call created,
same-file no-op, log naming both directories on failure.

The public signature `copy_run_manifest_forward(input_dir, output_dir)` is kept. The private
keyword `snapshot=` becomes `read=` (a `RunManifestRead`, or `None` for "already found absent").
A standalone call reads with `read_run_manifest(input_dir, pipeline_run_id_from_env(),
allow_legacy=True)` — the contract's non-parsing primitive, exported precisely "for a forwarding
stage that never parses" — so "the standalone copy validates nothing" remains true. It therefore
also raises `RunManifestMissingError` standalone when the id is known and nothing is found.

The temp prefix stays `.{read.filename}.`. For a run id over ~220 characters that exceeds
`NAME_MAX` and the publish raises `OSError` (exit 1); Argo ids are ~26 characters, so this is
documented, not engineered around (contracts' `run_manifest_filename` docstring flags it).

### D5. Diagnostics (log-only; scope never changes)

- **Stale output manifest** (existing warning): kept, still keyed to `RUN_MANIFEST_FILENAME`. It
  is reachable only with no run id (a known id with no manifest has already raised), and then the
  legacy name is the only one a downstream reader would consult.
- **Per-run manifests present but unread** (new, from traits): the read is not per-run (unscoped,
  or scoped by the legacy file) while `run_manifest.*.json` files sit at the top of `input_dir`.
  Typical cause: a copied cluster tree re-run locally.
- **Legacy manifest names another run** (new, from traits): id known, legacy file read, and its
  `pipeline_run_id` differs. Honored under `allow_legacy=True`, but it is another run's scope.

### D6. A directory at the manifest path now raises

Today `Path.is_file()` treats a non-regular entry as absent. Contracts opens rather than probes,
so a directory there raises (`IsADirectoryError`/`PermissionError` → exit 1). Accepted: a broken
tree is not an absent manifest, and silently treating it as absent is exactly the fall-through
the contract exists to prevent. Same as traits.

### D7. predict#43 — unique temp names for the three per-scan writes

The sidecar copy (`batch.py`), `.slp` write and `predictions.json` write (`output_contract.py`)
derive their temp path as `dst.with_name(dst.name + ".tmp")`, shared by any two concurrent
writers of the same destination. A private helper returns
`dst.with_name(f".{dst.name}.{uuid4().hex}.tmp")`: unique per writer, in the destination
directory (no cross-device replace on NFS), dot-prefixed.

**Why not `mkstemp` here, unlike the manifest copy:** `mkstemp` creates the file at `0600`. The
`.slp` is written by h5py, which truncates an existing file and keeps its mode, so every artifact
would become unreadable to the downstream container (a different uid on the shared NFS mount)
unless re-chmodded to a mode derived from the process umask. A name-only helper lets each site
keep creating its file exactly as today, so modes are unchanged by construction. `uuid4` rather
than a pid: every container is PID 1. Each site keeps its existing write → `os.replace` →
unlink-on-failure shape.

The sidecar-atomicity tests that patch the global `os.replace`/`shutil.copyfile` are made
path-conditional (fail only for the sidecar path), so the "stage no run manifest" constraint is
enforced by the test rather than by a code comment (predict#43's test note).

## Out of scope (separate)

- **predict#44** (narrow the forwarded manifest to `ok ∪ skipped`). Needs its own design: write-back
  reports manifest keys with no envelope as `missing_scan_keys` (salk-bloom `ingest.py:155`), so
  narrowing removes predict-failed scans from Bloom's view entirely; an all-failed batch yields an
  empty `scan_keys` that `RunManifest` rejects, and forwarding nothing would let traits fall back
  to the stale legacy manifest under `allow_legacy=True`; and it would make this deploy
  behavior-visible, which this change is designed not to be.
- **predict#40**: dissolved in predict by per-run naming (srp §2.1; comment 2026-09-23); its
  residues are in `sleap-roots` and bloomctl. Referenced, left open.
- **predict#41**: post-merge operations. The forward-copy it verifies already ships in C2's image
  (srp#89, `sha-9ac819f`); its checks can be observed on that or this deploy.
- **srp#82**: flipping `allow_legacy=False` (srp §4 step 6).

## Testing (TDD, tests first)

An autouse fixture deletes `ARGO_WORKFLOW_NAME`, so the default suite is a "no run id" run
regardless of the developer's shell.

- **Contamination reproduction:** 12 staged sidecars; legacy `run_manifest.json` with all 12 keys
  and `pipeline_run_id="sleap-roots-pipeline-hpdpf"`; `run_manifest.wf-a.json` with one key and
  `pipeline_run_id="wf-a"`. Under `ARGO_WORKFLOW_NAME=wf-a`: discovery yields exactly one scan;
  `output/run_manifest.wf-a.json` is byte-identical to the source with the source's mode; no
  `output/run_manifest.json` is written. With only the legacy file: all 12 are in scope (honored)
  and the "names another run" warning fires.
- Id known, no manifest → `RunManifestMissingError` from `run_batch`; no model-source call, no
  output directory created; CLI exits 1 and logs `Batch aborted`.
- Per-run file naming another run → `RunManifestIdentityError`; CLI exits 1 with `Batch aborted`.
- Invalid id (e.g. `../x`) → `ValueError`; CLI exits 1.
- No id, per-run files present → legacy/unscoped behavior plus the "unread" warning.
- Directory at `run_manifest.json` → raises.
- Standalone `copy_run_manifest_forward` with an id: forwards the per-run file under its own name
  without validating it.
- #43: two sequential writes to the same destination use different temp paths (recorded via a
  patched `os.replace`); failure still leaves no temp file; written files' modes equal a plain
  write's in the same directory.
- Every existing no-run-id test passes unchanged.

## Rollout (srp §4; order is load-bearing)

Merge anytime (readers before writer is the only merge constraint). Release, rebuild the image,
and bump the predictor pin in `sleap-roots-pipeline` — **deploying only after C2 (srp#89) is
confirmed**, never in the same deploy. Then bloomctl flips the writer (step 3). After merge: dated
roadmap entry in `sleap-roots-pipeline/docs/bloom-integration/roadmap.md` and a comment on srp#71.
