# Argo-ready exit semantics + empty-input guard + write hardening

## Context & goal

The predict batch driver (`sleap_roots_predict/batch.py` `run_batch` + `__main__.py`, from
predict #24/#27) is packaged as the `ghcr.io/talmolab/sleap-roots-predict` image and will be
driven by A4's Argo `predictor` template (`sleap-roots-pipeline`). [predict #26](https://github.com/talmolab/sleap-roots-predict/issues/26)
tracks three driver-side hardening items needed for that runtime: exit-code semantics vs Argo
`retryStrategy`, an empty-input guard, and SIGTERM/write hardening for graceful preemption.

None of these are defects in the current runner — they're the predict-side half of a hardening
pass shared symmetrically with the traits driver ([sleap-roots #259](https://github.com/talmolab/sleap-roots/issues/259)).
The A4 design doc (`sleap-roots-pipeline` `docs/superpowers/specs/2026-07-06-a4-request-driven-pipeline-design.md`
§8, "Resumability & error handling") names this exact reconciliation as an open, not-yet-made
decision for both producers ("**Producer Argo-readiness — reconcile *both* producers
uniformly**... Resolve the exit-code / empty-input / SIGTERM policy the *same way for both* at
wiring time"). This design is that decision, made from the predict side, since neither producer
had it decided yet as of this writing (verified: sleap-roots#259 has no branch/PR). The bloomctl
`batch-ingest-result` precedent ("zero exit on empty input") is **not** followed here — it's a
different semantic context (an empty write-back call can legitimately mean "nothing new to write
back"; an empty predict input dir means the stage-in mount was misconfigured, which is exactly
the silent-green failure mode #26 exists to close).

**Coordination note for the sleap-roots#259 session:** items 1 (exit codes) and 2 (empty-input)
below are meant to match exactly across both repos. Item 3 is *not* expected to be symmetric —
the traits driver's per-scan writes are already atomic (per its issue text), so it only needs the
SIGTERM handler half.

**Amendment (2026-08-19), after checking in with the sleap-roots#259 session:** D1's original
scheme below (`0`/`2`-aborted/`3`-partial/default-`1`-crash) is superseded. That session's own
`review-openspec` pass caught that reusing an exit code for a driver-owned "aborted" bucket
collides with `argparse`'s pre-existing usage-error `sys.exit(2)` — a bug this repo's CLI shares
(`sleap_roots_predict/__main__.py` also parses via `argparse`, and its pre-existing `except
(FileNotFoundError, ValueError): return 2` already collides with it, unnoticed until now). D1 and
D2 below are kept as a record of the original reasoning but are corrected in the OpenSpec
`design.md` for this change: the scheme is now `0`=success / `3`=partial / default `1`=every other
failure (staging error, empty-input, or crash, no longer split) / `143`=SIGTERM, with `2` left
untouched for `argparse` — identical to `sleap-roots#259`. Read `design.md`'s "Decisions" section
for the full corrected reasoning; treat every `2` mentioned below as historical, not current.

## What was already decided (we conform, not re-litigate)

- **Per-scan isolation.** A scan-level error is caught, recorded `failed`, and the batch
  continues (`run_batch`'s per-scan `try/except`). This slice does not change isolation — only
  how the *aggregate* outcome is reported to the OS/Argo.
- **Idempotency-key resume** (predict #35, merged 2026-08-18). `_previous_identity_key` already
  treats any unreadable/corrupt previous manifest or sidecar as "changed, never as a failure" —
  existence-only resume (the thing #26's original atomic-writes concern was about) no longer
  exists. This closes the *silent-skip* half of item 3's original correctness concern (see
  Decision 3 below) — confirmed by re-reading `batch.py`'s `_previous_identity_key` and its
  docstring, not assumed.
- **`out_scan_dir` layout**: `{scan_key}.predictions.json` + per-root `.slp` + copied-through
  sidecar, one `out_dir/{scan_key}/` per scan (`output_contract.py`, `batch.py::_predict_one`).
  Unchanged by this slice.

## Decisions (the forks this slice resolves)

### D1 — Exit codes: a dedicated `partial` code, distinct from a real crash

Today: `0` = no scan failed, `1` = any scan failed *or* an uncaught exception (both collide on
`1`) . A4 §8's own principle is to **distinguish scan-level error (mark failed, continue) from
pod-level death (Argo retry + resume-skip)** — the current scheme can't express that distinction
on the wire.

New scheme (**corrected 2026-08-19** — see the Amendment note above; the originally-drafted `2`
= aborted code is dropped):

| Code | Meaning | Argo should... |
|---|---|---|
| `0` | Success — ≥1 scan discovered, none failed (`ok`/`skipped` only) | proceed normally |
| `3` | **Partial** — ≥1 scan discovered and processed, ≥1 isolated scan failure, process did not crash | *not* retry the whole batch (per-scan isolation already ran); mark the step `partial` (Argo-template-side handling — out of scope here, see below) |
| *(none — Python default)* `1` | Every other failure: a pre-flight/staging error (missing input dir, duplicate `scan_key`, zero scans discovered — D2) or a genuine uncaught crash (e.g. model-registry auth failure before the per-scan loop starts) — no longer split into a separate code | Argo `retryStrategy` retries |
| `2` | *(not used by this driver)* — reserved: `argparse` already calls `sys.exit(2)` on a CLI usage error, before `run_batch` ever runs | n/a — a CLI invocation problem, unrelated to a batch outcome |

This is a two-line change: `__main__.py`'s `return 0 if result.ok else 1` becomes
`return 0 if result.ok else 3`, its `except (FileNotFoundError, ValueError): return 2` special
case is **removed** (those exceptions now propagate uncaught, surfacing as the default `1`, like
any other crash), and the docstring/module comment is updated to describe the three driver-owned
codes plus the `argparse`-owned `2`.

**Out of scope, flagged not implemented:** the Argo template still needs to interpret `3` as
"partial, continue" rather than "step failed" (a `sleap-roots-pipeline` template change, e.g.
`retryStrategy` conditioned on exit code, or `continueOn`). Noted per #26's own scope boundary.

### D2 — Empty input becomes a staging error, not a silent no-op

`run_batch` currently: `discover_scans` returns `[]` → logs a warning → returns
`BatchResult(scans=[])` (`ok=True`) → CLI exits `0`. A misconfigured or empty `-v …:/in` mount
today produces a green Argo node that emitted nothing.

Fix: `run_batch` raises `ValueError(f"no scans discovered under {input_dir}")` in place of the
current warn-and-return-empty branch, *before* constructing the `WarmModelWorker` (no needless
model load). **Corrected 2026-08-19:** `__main__.py` no longer catches this into a special exit
code — it propagates like any other staging exception, surfacing as the default `1` (see D1).

This applies uniformly, including when a `run_manifest.json` is present and scopes discovery to
zero `scan_keys` — a manifest declaring nothing to do is itself worth surfacing loudly rather
than silently no-op'ing, consistent with the rest of this design's stance on empty input.

**Not empty input, unaffected:** a manifest that lists scan_keys with no matching sidecar
produces isolated per-scan `error` entries (`ScanInput.error` set) — `discover_scans` returns a
*non-empty* list in that case, so it flows through the existing per-scan-failure path and ends in
D1's `partial` (`3`), not the crash (`1`) path. Only a *literally empty* discovered-scan list
triggers D2.

### D3 — Atomic writes: still worth it, as defense-in-depth (not a closed correctness hole)

Re-investigated per the handoff's explicit instruction to re-verify this item's scope: the
issue's original framing ("atomic writes + checksum-verified skip must ship together, or a
SIGKILL-truncated manifest is silently treated as done") is **no longer accurate** — predict #35
already replaced existence-only resume with the idempotency-key recompute, and a corrupt/partial
manifest or sidecar is unconditionally treated as "changed" (never skipped) by
`_previous_identity_key`. Tracing every write ordering in `_predict_one`/`write_prediction_outputs`
confirms no remaining path where a truncated write is later mistaken for "done" by predict's own
resume logic.

The correctness hole is closed. What's *not* addressed by resume logic is a different, narrower
concern: an **external reader** (not predict's own resume — e.g. a future direct read of
`{scan}.predictions.json` or a `.slp` that races predict's write, outside the normal
"Argo step completes, then the next step starts" ordering) could observe a torn/partial file.
Given the fix is cheap (temp-file-in-same-dir + `os.replace`) and brings predict to parity with
the traits driver's existing atomic writes, this ships as defense-in-depth, explicitly **not**
because it closes a live correctness bug.

Scope: `write_prediction_outputs`'s two write sites (`sio.save_file` for each `.slp`,
`.write_text` for the manifest) both move to write-temp-then-`os.replace` in the same directory
(guarantees the rename is atomic on the same filesystem — no cross-filesystem temp dirs). The
manifest write stays strictly last (already documented as the resume commit-marker; unchanged).
The sidecar copy in `batch.py::_predict_one` (`shutil.copyfile`, which happens *before* the
manifest for the same reason) gets the same temp+rename treatment for consistency — it's the
third file in the same "external reader could see a torn file" category, and the change is a few
lines.

The pre-write stale-`.slp` cleanup pass (removes orphaned artifacts from a changed model slug)
stays unchanged — it doesn't gate resume (which only reads the manifest) and moving it doesn't
change the correctness story either way.

### D4 — SIGTERM handler: finish current scan, stop at the next boundary

The image's exec-form `ENTRYPOINT ["python","-m","sleap_roots_predict"]` makes `python` PID 1
with no signal handler, so Argo preemption waits out the full `terminationGracePeriodSeconds`
before SIGKILL.

Design: `main()` creates a `threading.Event`, installs a `signal.signal(signal.SIGTERM, ...)`
handler that sets it (signal-handler-safe — just a flag write, no blocking calls), and passes
`should_stop=event.is_set` into `run_batch` as a new optional keyword parameter (default a no-op
returning `False`, so every existing caller/test is unaffected). `run_batch`'s per-scan loop
checks `should_stop()` at the top of each iteration (a scan boundary, never mid-inference — there
is no safe interrupt point inside sleap-nn/GPU inference) and breaks early, logging how many of
the discovered scans were attempted.

After `run_batch` returns, `main()` checks the event directly (no new `BatchResult` field needed
— this is CLI-level signal reporting, not a batch-outcome concern): if set, log "terminated by
SIGTERM after a partial batch" and return `143` (`128 + SIGTERM`, standard Unix convention for
"killed by signal N"), overriding whatever D1 exit code the completed-so-far scans would
otherwise produce. This makes the container's own reported exit code honestly reflect "I was
asked to stop," distinct from `0`/`3`/default `1`.

**Known limitation, accepted:** worst-case latency is bounded by one scan's predict duration, not
truly immediate. A pathologically long single scan can still exceed the grace period and get
SIGKILLed anyway — accepted, since interrupting mid-inference isn't safely possible and D3's
atomic writes mean a SIGKILL mid-scan still can't leave a torn artifact that looks done.

## Architecture

No new modules. Changes are localized:

- `sleap_roots_predict/batch.py`: `run_batch` gains `should_stop: Callable[[], bool] = <no-op>`
  (checked at loop top, D4); the empty-scans branch raises instead of returning (D2); `_predict_one`'s
  sidecar copy becomes atomic (D3).
- `sleap_roots_predict/__main__.py`: installs the `SIGTERM` handler + `threading.Event` around the
  `run_batch` call (D4); exit-code logic changes from `0/1` to `0/3/(default 1)/143` (D1, D4),
  removing the special-cased `except (FileNotFoundError, ValueError): return 2` (D1); docstring
  updated to enumerate the three driver-owned codes plus the `argparse`-owned `2`.
- `sleap_roots_predict/output_contract.py`: `write_prediction_outputs`'s `.slp` and manifest
  writes move to temp-then-`os.replace` (D3).

## Data flow

Unchanged at the scan level (discover → resolve → predict → write). What changes is only the
*aggregate* signal surfaced after the loop (exit code) and the *durability* of each individual
write (atomic vs. direct) — no change to `PredictionManifest`/`PredictionArtifact` shape, no
change to what's discovered or how scans are matched to models.

## Testing (real TDD, no mocks — mirrors existing suite's convention)

- D1: `test___main__.py` (or equivalent) — a batch with one failed scan and otherwise-ok scans
  exits `3`; an all-ok batch exits `0`; existing `FileNotFoundError`/duplicate-key cases now exit
  the default `1` (no longer a distinct `2` — a regression test pinning the removal, not just the
  addition). A test asserting a CLI usage error (missing required argument) exits `2` via
  `argparse`, documenting that this is unrelated to the driver's own `0`/`1`/`3` codes.
- D2: `run_batch` over an empty (but existing) directory raises `ValueError`; CLI exits the
  default `1` with a clear logged message. A `run_manifest.json` scoping to zero `scan_keys` also
  raises. A manifest scoping to keys with no matching sidecar still produces isolated `failed`
  entries (exit `3`, not `1`) — a regression test pinning this distinction.
- D3: a write interrupted between temp-write and rename (simulate by asserting the temp file
  never exists at the final path's name until the whole write succeeds) — assert no
  partially-written file is ever visible under the final filename; existing round-trip tests
  (`write_prediction_outputs` writes then is re-readable) must keep passing unchanged.
- D4: `run_batch` with `should_stop` returning `True` after the first scan stops before the
  second, and the first scan's outputs are complete and valid; CLI-level test sends `SIGTERM` to
  a subprocess mid-batch (or, more practically given sandboxing constraints, a unit test on
  `main()`'s signal-handling logic in isolation, with `run_batch` mocked... **actually avoid a
  mock here** — real TDD is this repo's convention; prefer driving the handler + event logic
  directly (call the registered handler function, assert the event becomes set, assert
  `run_batch` observes `should_stop() is True` and stops) over a full subprocess-signal test if a
  real subprocess test proves too flaky in CI.

## OpenSpec scope

Modifies the existing `predict-container` capability (established by predict #24/#27) — this is
a behavior change to an already-specced capability (exit codes, empty-input handling, write
durability, signal handling), not a new capability. One `openspec/changes/<id>/specs/predict-container/spec.md`
delta with `MODIFIED Requirements` covering D1–D4.

## Risks & assumptions

- Assumes Argo's own template/`retryStrategy` will eventually be updated to treat exit `3`
  specially (D1's "out of scope, flagged" item) — until that pipeline-repo change lands, exit `3`
  behaves like any other non-zero exit under today's `retryStrategy` (i.e., Argo still retries the
  whole batch on a `partial` result). This design does not regress that — it's already true today
  under exit `1` — it just makes the *distinction* representable once the template catches up.
- `os.replace` atomicity assumes the temp file and final path are on the same filesystem (same
  parent directory) — true for the shared mount described in A4 §8.
- The SIGTERM handler assumes CPython delivers signals promptly between bytecode instructions,
  which holds for the pure-Python loop-boundary check but does not shorten a single blocking
  native call (e.g. a long `sio.save_file` or model inference call) already in flight.

## Out of scope

- The Argo `WorkflowTemplate`/`retryStrategy` change to interpret exit code `3` as "partial,
  continue" (`sleap-roots-pipeline` repo).
- Matching this decision into the traits driver (`sleap-roots#259`) — tracked there, not
  implemented here; this doc is written to be directly referenceable from that session.
- SIGINT/Ctrl+C handling (Argo only sends SIGTERM; interactive use is out of this issue's scope).
- Any change to per-scan retry-count/`MAX_SCAN_ATTEMPTS` semantics (A4 §8 mentions retry-then-isolate
  as a *pipeline-level* concept — the pod is retried by Argo, not the individual scan by predict
  itself).
