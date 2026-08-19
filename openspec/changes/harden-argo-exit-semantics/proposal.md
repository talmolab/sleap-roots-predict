## Why

The predict batch driver (`sleap_roots_predict/batch.py` + `__main__.py`) is driven by A4's Argo
`predictor` template, but three of its current behaviors are wrong for that runtime: exit code
`1` means both "a scan isolated-failed" and "the process crashed," so Argo's `retryStrategy`
can't tell a poison scan (should end `partial`, not retry) from a real pod crash (should retry);
an empty input directory silently exits `0` with nothing written, so a misconfigured mount
produces a green node; and the exec-form entrypoint has no `SIGTERM` handler, so Argo preemption
waits out the full grace period before `SIGKILL`. These are tracked in
[predict #26](https://github.com/talmolab/sleap-roots-predict/issues/26), reconciled uniformly
with the traits driver's identical issue ([sleap-roots#259](https://github.com/talmolab/sleap-roots/issues/259))
per the A4 design doc's §8.

**Cross-repo reconciliation (2026-08-19):** the sibling `sleap-roots#259` proposal landed on
`0`=success / `3`=partial / `1`=crash / `2` reserved for `argparse`, after its own
`review-openspec` pass caught that reusing `2` for a driver-owned "aborted" code collides with
`argparse`'s pre-existing usage-error exit code. This proposal adopts the identical scheme (see
below) rather than keeping this repo's independently-drafted `0`/`2`/`3`/default-`1` split, so both
producers report numerically identical codes for numerically identical situations — the load-bearing
agreement A4-wiring time needs.

## What Changes

- **MODIFIED** `predict-container`: `run_batch`/`__main__.py` exit-code scheme —
  `0`=success, `3`=**new**, partial (isolated scan failure(s), no crash); every other failure
  (staging error, empty-input, or any other uncaught exception) surfaces as Python's default `1`.
  **Revised during cross-repo reconciliation** (see design.md): the first draft of this proposal
  kept the CLI's existing `except (FileNotFoundError, ValueError): return 2` special-case and
  routed the new empty-input guard through it too. That collides with `argparse`'s own
  pre-existing `sys.exit(2)` on a CLI usage error (verified: `python -m sleap_roots_predict` with
  no args already exits `2` today) — the exact bug the sibling `sleap-roots#259` proposal's own
  review caught and fixed by leaving `2` alone. This proposal now does the same: drops the
  special-cased `2`, folds staging/empty-input errors into the default `1` crash bucket, and
  leaves `2` untouched for `argparse`.
- **MODIFIED** `predict-container`: an empty (zero scans discovered) input directory now raises
  (routed through the existing missing-directory/duplicate-`scan_key` abort path, itself no longer
  special-cased to `2` — see above) instead of silently no-op'ing and exiting `0`. **BREAKING**
  for any caller relying on the old no-op-on-empty behavior.
- **ADDED** `predict-container`: a `SIGTERM` handler — the CLI stops at the next scan boundary,
  finishes the in-flight scan, and exits `143` (`128+SIGTERM`), rather than ignoring the signal
  until `SIGKILL`.
- **MODIFIED** `predict-container`: the copied-through scan-metadata sidecar is now written
  atomically (temp file + rename), matching the manifest/`.slp` durability guarantee below.
- **MODIFIED** `prediction-output`: `write_prediction_outputs` writes each `.slp` and the
  `{scan_key}.predictions.json` manifest via temp-file-in-same-directory + `os.replace`, so no
  external reader can observe a torn/partial file. This is defense-in-depth, not a fix for a live
  correctness bug — predict #35's idempotency-key resume already treats any corrupt/truncated
  previous artifact as "changed, re-predict," never as "done."

## Impact

- Affected specs: `predict-container`, `prediction-output`
- Affected code: `sleap_roots_predict/batch.py`, `sleap_roots_predict/__main__.py`,
  `sleap_roots_predict/output_contract.py`
- Out of scope (tracked elsewhere, not in this change): the `sleap-roots-pipeline` Argo
  `WorkflowTemplate`/`retryStrategy` change to interpret exit code `3`; applying this same
  convention to the traits driver (`sleap-roots#259`, separate repo); `SIGINT` handling; per-scan
  retry-count semantics.
- Full design rationale: `docs/superpowers/specs/2026-08-18-argo-ready-exit-semantics-design.md`.
