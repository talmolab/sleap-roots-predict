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

**Cross-repo reconciliation (2026-08-19):** this proposal adopts the same `0`=success /
`3`=partial / `1`=crash / `2`=reserved-for-`argparse` scheme as the sibling `sleap-roots#259`
proposal, after that proposal's own `review-openspec` pass caught that reusing `2` for a
driver-owned "aborted" code collides with `argparse`'s pre-existing usage-error exit code —
see `design.md`'s "Alternative considered and reversed" for the full history.

## What Changes

- **MODIFIED, BREAKING** `predict-container`: `run_batch`/`__main__.py` exit-code scheme —
  `0`=success, `3`=**new**, partial (isolated scan failure(s), no crash); every other failure
  (staging error, empty-input, or any other uncaught exception) surfaces as Python's default `1`;
  `2` is left untouched, reserved for `argparse`'s own usage-error exit. **Breaking** for any
  caller that currently checks exit `1` for "some scan failed" (now `3`) or exit `2` for a staging
  error (no longer a distinct code — see `design.md`'s Migration Plan).
- **MODIFIED, BREAKING** `predict-container`: an empty (zero scans discovered) input directory now
  raises (routed through the same staging-error path as the missing-directory/duplicate-`scan_key`
  cases above) instead of silently no-op'ing and exiting `0`. Breaking for any caller relying on
  the old no-op-on-empty behavior.
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
- Affected docs: `README.md` ("Running the predict container" section — stale exit-code
  language, undocumented empty-input behavior), `API.md` (`run_batch`'s exit-code parenthetical),
  `CHANGELOG.md` (`[Unreleased]` "Predict container CLI" entry)
- Out of scope (tracked elsewhere, not in this change): the `sleap-roots-pipeline` Argo
  `WorkflowTemplate`/`retryStrategy` change to interpret exit code `3`; applying this same
  convention to the traits driver (`sleap-roots#259`, separate repo); `SIGINT` handling; per-scan
  retry-count semantics.
- Full design rationale: `docs/superpowers/specs/2026-08-18-argo-ready-exit-semantics-design.md`.
