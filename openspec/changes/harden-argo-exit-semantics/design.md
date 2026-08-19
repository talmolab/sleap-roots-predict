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

- **Exit codes:** `0`=success, `3`=partial (new — replaces the old overloaded `1` for "some scan
  failed"), default Python `1`=every other failure (staging error, empty-input, or a genuine
  uncaught crash — no longer split out). `143` (`128+SIGTERM`) overrides all of the above when the
  process was asked to stop early. `2` is deliberately left alone.
  - Alternative considered: exit `0` on any completed run regardless of per-scan failures,
    pushing failure detection entirely onto the written manifests. Rejected — throws away a
    cheap wire-level signal Argo can act on without reading output files.
  - **Alternative considered and reversed after cross-repo reconciliation (2026-08-19): keep a
    distinct `2`=aborted code for staging errors, separate from a generic `1`=crash.** This was
    the original decision reached during this proposal's own brainstorming, before checking the
    sibling `sleap-roots#259` proposal. Rejected once checked, because:
    - `sleap_roots_predict/__main__.py` parses args via `argparse`, and
      `argparse.ArgumentParser.error()` already calls `sys.exit(2)` on a bad invocation (verified
      empirically: `python -m sleap_roots_predict` with missing args exits `2` today) — this
      predates this proposal (shipped with #24's CLI) and was never previously flagged. Reusing
      `2` for "aborted" (already true today for `FileNotFoundError`/`ValueError`, and this
      proposal was about to extend it to empty-input too) makes a genuine CLI-usage
      misconfiguration indistinguishable from a staging error at the exit-code level.
    - The `sleap-roots#259` proposal's own `review-openspec` pass caught the identical bug in its
      first draft (which had picked `2` for "partial") and fixed it by reserving `2` for
      `argparse` and using `3` for the one code that's actually load-bearing for Argo. Diverging
      from that here — keeping a *different* meaning for `2` in this repo — would mean the same
      exit code means three different things across the two producers (argparse usage error in
      both, PLUS "aborted" only in predict), which is exactly the confusion A4's design doc §8
      asks to avoid by resolving this "the same way for both" producers.
    - The diagnostic granularity lost (distinguishing "staging error" from "generic crash") isn't
      operationally load-bearing: A4's `retryStrategy`/`continueOn` only needs to special-case `3`
      (partial); every other nonzero code is retried the same way regardless of whether it's `1`
      or a separate `2`. Losing that distinction costs log readability, not Argo behavior.
- **Empty input → raise:** previously reused the existing `except (FileNotFoundError, ValueError)`
  → exit `2` path in `__main__.py`. Per the reversed decision above, that special case is removed
  entirely — `run_batch` still raises on empty input (unchanged goal), but `__main__.py` no longer
  catches `FileNotFoundError`/`ValueError` into a custom code; they (and any other staging
  exception) simply propagate, surfacing as Python's default `1`, identical to any other crash.
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
- Any caller depending on the current (pre-this-proposal) `except (FileNotFoundError,
  ValueError): return 2` special-case must instead check for the default `1` — this drops a
  distinction (`2` vs `1`) that existed since #24, not something this proposal introduces to
  preserve. No known consumer depends on it (A4 hasn't wired predict's exit code into any Argo
  `retryStrategy`/`continueOn` logic yet).
- Any caller depending on empty-input silently exiting `0` will now see it raise/exit `1` — a
  deliberate **BREAKING** change per the issue's own ask.

## Open Questions

- Cross-repo numeric alignment is now resolved (`0`/`1`/`3`/`143`, `2` reserved for `argparse`,
  matching `sleap-roots#259` exactly — see the reversed decision above). Nothing else outstanding
  from the original four forks (exit codes, empty-input, atomic writes, SIGTERM); see the linked
  design doc for the full decision trail on those.
