## Context

Full design rationale, alternatives considered, and cross-repo coordination notes live in
`docs/superpowers/specs/2026-08-18-argo-ready-exit-semantics-design.md` (approved via
`superpowers:brainstorming`). This file summarizes the decisions for OpenSpec purposes; see that
doc for the "why not the alternative" reasoning.

## Goals / Non-Goals

- Goals: make the CLI's exit code distinguish "isolated scan failure(s), batch otherwise
  completed" from "the process crashed"; make an empty input directory a loud error instead of a
  silent no-op; add a `SIGTERM` handler for prompt-enough preemption; close the residual
  torn-file-visible-to-an-external-reader window via atomic writes.
- Non-Goals: changing the Argo `WorkflowTemplate`/`retryStrategy` itself (pipeline-repo concern);
  applying this to the traits driver (a separate repo/session, `sleap-roots#259`); `SIGINT`
  handling; per-scan retry-count/`MAX_SCAN_ATTEMPTS` semantics.

## Decisions

- **Exit codes:** `0`=success, `2`=aborted (staging error, incl. empty-input via a new raise in
  `run_batch`), `3`=partial (new — replaces the old overloaded `1` for "some scan failed"),
  default Python `1`=uncaught crash. `143` (`128+SIGTERM`) overrides all of the above when the
  process was asked to stop early.
  - Alternative considered: exit `0` on any completed run regardless of per-scan failures,
    pushing failure detection entirely onto the written manifests. Rejected — throws away a
    cheap wire-level signal Argo can act on without reading output files.
- **Empty input → raise:** reuses the existing `except (FileNotFoundError, ValueError)` → exit
  `2` path in `__main__.py`; no new exception type or CLI branch.
- **Atomic writes are defense-in-depth, not a correctness fix:** verified (not assumed) that
  predict #35's `_previous_identity_key` already treats a corrupt/truncated previous artifact as
  "changed," so a SIGKILL-truncated file was never silently treated as done under the *current*
  resume logic. Ships anyway — cheap, closes an external-reader race, brings parity with the
  traits driver's existing atomic writes.
- **SIGTERM: stop at scan boundary, not mid-inference:** a `threading.Event` set by the signal
  handler, checked at the top of `run_batch`'s per-scan loop. Bounded worst-case latency = one
  scan's duration (accepted; there's no safe interrupt point inside sleap-nn/GPU inference).

## Risks / Trade-offs

- Exit code `3` has no effect on Argo behavior until the pipeline-repo template is updated to
  treat it as `partial` — until then it's simply "a non-zero code Argo retries," identical to
  today's behavior under `1`. Not a regression, just not yet load-bearing on the Argo side.
- `os.replace` atomicity requires the temp file and final path to share a filesystem (true for
  the shared mount A4 assumes; would not hold if `out_dir` ever spanned mounts).

## Migration Plan

No data migration. Behavioral/CLI-contract change only:
- Any caller depending on exit `1` meaning "some scan failed" must switch to checking for `3`
  (or treat any non-zero as failure, which still works, just loses the new distinction).
- Any caller depending on empty-input silently exiting `0` will now see it raise/exit `2` — a
  deliberate **BREAKING** change per the issue's own ask.

## Open Questions

None outstanding — all four forks (exit codes, empty-input, atomic writes, SIGTERM) were resolved
during brainstorming; see the linked design doc for the full decision trail.
