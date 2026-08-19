## 1. Exit codes (D1): partial (3) distinct from crash (default 1); drop the special-cased 2

**Revised 2026-08-19 after cross-repo reconciliation with `sleap-roots#259`:** the original plan
here kept a distinct `2` = "aborted" code (reusing the existing `except (FileNotFoundError,
ValueError): return 2` in `__main__.py`) alongside the new `3` = partial. That collides with
`argparse`'s own pre-existing `sys.exit(2)` on a CLI usage error — verified: `python -m
sleap_roots_predict` with a missing argument already exits `2` today, via a code path this
special-case never accounted for. The sibling `sleap-roots#259` proposal hit the identical bug
in its first draft and fixed it by dropping the special code and reserving `2` for `argparse`
everywhere; this task list now does the same, so both producers use numerically identical codes.

**Revised again after `/review-openspec` round 1:** the reviewer found that
`tests/test_batch.py` already has two tests hardcoding the *old* scheme —
`test_cli_main_exit_codes` (asserts `main(...) == 1` for a failed batch) and
`test_cli_missing_input_dir_returns_nonzero` (asserts `main(...) == 2`). Neither was named in
the original 1.1/1.3 tasks, so implementing this section as originally written would have left
the suite self-contradictory (the second case worse than a mere assertion mismatch — once 1.5
removes the `except` clause, that test's `main()` call raises `FileNotFoundError` uncaught,
an error, not a failure). Tasks 1.1 and 1.3 below now name both tests explicitly. Also decided:
the log-quality regression from dropping the `except` clause (a missing-mount error would
otherwise dump a raw traceback instead of a clean one-line log) is worth preserving — 1.5 now
keeps a narrow `except (FileNotFoundError, ValueError)` that only logs before re-raising, so the
final exit code is still Python's unhandled-exception default `1` but the log line survives.

- [ ] 1.1 In `tests/test_batch.py` (or `test___main__.py`), write a failing test asserting
      `main()` returns `3` (not `1`) for a batch with one failed scan among otherwise-ok scans,
      **and update `test_cli_main_exit_codes`'s existing `assert main(...) == 1` (failed-batch
      case) to `== 3`** — do not leave the old assertion alongside the new test; they cover the
      same scenario and must agree.
- [ ] 1.2 In the same file, write a failing test asserting `main()` still returns `0` for an
      all-ok/all-skipped batch (pins the unchanged success path).
- [ ] 1.3 In the same file, write a failing test asserting the existing
      `FileNotFoundError`/duplicate-`scan_key` abort paths now return the default `1` (not the old
      `2`) — this is a **behavior-change** regression test, not a no-op pin: it must fail against
      the current code (which still returns `2`) until task 1.5 removes the special case.
      **Rewrite `test_cli_missing_input_dir_returns_nonzero`** (currently `assert main(...) == 2`)
      to expect `main(...) == 1` (per the logged-then-propagated behavior decided above, the
      exception is caught, logged, and re-raised inside `main()`, so this is still an assertion on
      `main()`'s return/raise, not a bare `pytest.raises` around a raw exception — confirm the
      exact shape once 1.5's implementation is written, and adjust the test to match rather than
      leaving the stale `== 2` assertion in place). Also add a CLI-level test asserting the new
      empty-input `ValueError` (from section 2) surfaces through `main()` as exit `1` too, for the
      same logged-then-reraised reason.
- [ ] 1.4 In the same file, write a failing test asserting a CLI usage error (invoke `main()`/the
      module with a missing required argument) exits `2` via `argparse`, and that this is
      independent of the driver's own `0`/`1`/`3` codes (documents the boundary so a future change
      can't quietly blur it again).
- [ ] 1.5 Implement: in `sleap_roots_predict/__main__.py`, change `return 0 if result.ok else 1` to
      `return 0 if result.ok else 3`. Replace the existing `except (FileNotFoundError, ValueError):
      return 2` clause with `except (FileNotFoundError, ValueError) as exc:
      logging.getLogger(__name__).error("Batch aborted: %s", exc); raise` — this keeps the
      existing clean one-line log message for these two known staging-error types (a real
      operational-log-quality regression if dropped entirely, per `/review-openspec` round 1) while
      letting the exception continue on to Python's default unhandled-exception exit `1`, same as
      any other uncaught crash. Update the module docstring and `main()`'s docstring to enumerate
      the three driver-owned codes (`0`/`3`/default `1`) plus a note that `2` is reserved by
      `argparse`, not by this driver. Run 1.1–1.4 green.

## 2. Empty-input guard (D2): raise instead of silent no-op

- [ ] 2.1 In `tests/test_batch.py`, write a failing test asserting `run_batch` raises `ValueError`
      over a present-but-empty input directory (no `*.scan_metadata.json`), and that it raises
      *before* any `WarmModelWorker`/model-source interaction (assert via a source stub that
      records whether it was ever called).
- [ ] 2.2 In `tests/test_batch.py`, write a failing test asserting a `run_manifest.json` present
      but scoping to zero `scan_keys` also raises via the same path (pydantic validation of
      `RunManifest` may already reject an empty `scan_keys` list — confirm which layer raises
      and assert that one; do not add a redundant check if `RunManifest` itself already enforces
      non-empty `scan_keys`).
- [ ] 2.3 In `tests/test_batch.py`, write a regression test asserting a `run_manifest.json` that
      lists `scan_keys` with **no** matching sidecar still returns a non-empty `BatchResult` with
      `failed` entries via `run_batch` directly (i.e. does NOT raise the empty-input `ValueError`)
      — pins the D2/D1 boundary described in the spec delta at the `run_batch` level, distinct
      from the CLI-level `main()` exit-code tests in section 1.
- [ ] 2.4 Implement: in `sleap_roots_predict/batch.py::run_batch`, replace the
      `if not scans: logger.warning(...); return result` branch with
      `raise ValueError(f"no scans discovered under {input_dir.as_posix()}")`, placed before
      `WarmModelWorker(source=source)` is constructed. Update `run_batch`'s docstring (`Raises:`
      section) accordingly. Run 2.1–2.3 green; also re-run the existing "Empty input directory is
      a no-op" test from `test_batch.py` if present and update/replace it to assert the new
      raising behavior (do not leave a stale test asserting the old no-op contract).

## 3. Atomic writes (D3): `.slp`, manifest, and sidecar copy

**Note found in `/review-openspec` round 1:** `sio.save_file` infers the output format purely
from the destination filename's extension (confirmed by reading `sleap_io`'s source) and raises
`ValueError: Unknown format` if it doesn't recognize one. This repo's own existing atomic-write
precedent (`sleap_roots_predict/parity.py`'s `write_report`, `foo.json` → `foo.json.tmp`) would
break `.slp` writes if copy-pasted verbatim, since `foo.slp.tmp` no longer ends in `.slp`. Task
3.4 below calls this out explicitly so it's designed in from the start, not discovered as a test
failure partway through.

- [ ] 3.1 In `tests/test_output_contract.py`, write a failing test asserting that during
      `write_prediction_outputs`, no file ever exists at a `.slp`'s final path except either
      fully absent or fully written (e.g. by monkeypatching `os.replace` to raise once before it's
      actually called, then asserting the final path does not exist and any temp file is cleaned
      up or at least never mistaken for the final artifact).
- [ ] 3.2 In `tests/test_output_contract.py`, write the equivalent failing test for
      `{scan_key}.predictions.json` (temp file created, final path untouched until rename).
- [ ] 3.3 In `tests/test_output_contract.py`, write a failing test asserting every `.slp`'s atomic
      write completes (final `os.replace` observed) before the manifest's atomic write begins —
      pins the "manifest is still written last" ordering invariant the spec requires, distinct
      from the interrupted-write tests in 3.1/3.2 (e.g. via a call-order spy wrapping `os.replace`).
- [ ] 3.4 In `tests/test_batch.py`, write a failing test for the sidecar copy in
      `batch.py::_predict_one` with the same interrupted-before-rename assertion as 3.1.
- [ ] 3.5 Implement atomic writes in `sleap_roots_predict/output_contract.py`'s
      `write_prediction_outputs`: write each `.slp` to a temp path in the same directory via
      `sio.save_file(labels, tmp_path.as_posix(), format="slp")` — **pass `format="slp"` explicitly
      rather than relying on the temp filename's extension** (see the note above; do not use a
      bare `.tmp`-suffix-appended name unless it still ends in `.slp`), then `os.replace` into the
      final path; same pattern for the manifest's `.write_text`, keeping the manifest write
      strictly last. Run 3.1–3.3 green; confirm all existing round-trip tests in
      `test_output_contract.py` still pass unchanged.
- [ ] 3.6 Implement the same atomic pattern for the sidecar copy in
      `sleap_roots_predict/batch.py::_predict_one` (replace the direct `shutil.copyfile` with
      copy-to-temp + `os.replace`; a plain file copy has no format-inference concern, unlike 3.5).
      Run 3.4 green. Land 3.5 and 3.6 as separate commits — independent write sites, no shared
      code path (per `/review-openspec` round 1's commit-strategy feedback).

## 4. SIGTERM handler (D4): stop at scan boundary, exit 143

**Note found in `/review-openspec` round 1:** on Windows, `signal.signal(signal.SIGTERM, ...)`
registers without error, but real cross-process delivery (`os.kill(pid, signal.SIGTERM)`) is
implemented via `TerminateProcess`, which kills the process immediately **without invoking the
registered handler** — unlike Linux/macOS, where real delivery does invoke it. The plan below
already avoids this trap (4.3 tests the handler by calling it directly, never via `os.kill`), so
there is no CI risk today — but if a future change "improves" that test to be more "realistic"
via `os.kill`, it would silently hang/kill the `windows-latest` CI job instead of failing
cleanly. `design.md` and the spec now document this platform split explicitly so the constraint
in 4.3 doesn't get quietly removed later.

- [ ] 4.1 In `tests/test_batch.py`, write a failing test asserting `run_batch(..., should_stop=fn)`
      stops after the first scan (of two) when `fn` returns `True` starting from the second
      iteration, and that the first scan's outputs are complete and valid (reloadable manifest +
      `.slp`).
- [ ] 4.2 In `tests/test_batch.py`, write a failing test asserting `run_batch` with the default
      `should_stop` (no argument passed) is unaffected — processes all scans exactly as before
      (regression pin for existing callers/tests).
- [ ] 4.3 Add a test (`test_batch.py` or a new `tests/test_main.py` if signal-handling tests don't
      fit `test_batch.py`'s existing scope) that: obtains the registered `SIGTERM` handler and its
      backing `threading.Event` independently of running a full batch (see 4.5's implementation
      note — `main()` must expose a seam for this, e.g. a small `_install_sigterm_handler() ->
      threading.Event` helper it calls, so a test can call that helper directly rather than
      needing to invoke all of `main()` to obtain the handler); invoke the handler directly (never
      `os.kill` — see the Windows note above), assert the event becomes set, assert `main()`
      returns `143` when the event is already set going into its post-`run_batch` check — without
      mocking `run_batch` itself (real TDD). **Save `signal.getsignal(signal.SIGTERM)` before the
      test and restore it in a `finally`/fixture teardown**, so the handler registered by this test
      doesn't leak into later tests or interact with CI's job-level timeout kill.
- [ ] 4.4 Implement: add `should_stop: Callable[[], bool] = lambda: False` (keyword-only) to
      `run_batch`, checked at the top of the per-scan `for` loop (`if should_stop(): logger.warning(...);
      break`). Run 4.1–4.2 green.
- [ ] 4.5 Implement: add a small `_install_sigterm_handler() -> threading.Event` helper in
      `sleap_roots_predict/__main__.py` that creates the `Event`, registers a
      `signal.signal(signal.SIGTERM, ...)` handler that sets it, and returns the `Event` — giving
      4.3 a seam to call directly. `main()` calls this helper, passes `should_stop=event.is_set`
      into `run_batch`, and after `run_batch` returns, checks the event first — if set, log and
      `return 143` before falling through to the normal `0`/`3` logic (`2` is never part of this
      fallthrough — `argparse` exits `2` on its own, before `run_batch` runs). Run 4.3 green.

## 5. Docs and closeout

- [ ] 5.1 Update `sleap_roots_predict/__main__.py`'s module docstring (already touched in 1.5) and
      `sleap_roots_predict/batch.py`'s `run_batch` docstring (`Raises:`/behavior description) to
      match the final implemented behavior — re-read both after 1–4 land to catch any drift from
      the design doc's phrasing.
- [ ] 5.2 Update `README.md`'s "Running the predict container" section: replace "it exits
      non-zero if any scan failed" with a one-line pointer to the exit-code contract (`0`=success,
      `3`=partial, default `1`=staging-error-or-crash, `143`=`SIGTERM`, `2` reserved for
      `argparse`) rather than re-describing the table in prose — point at the `predict-container`
      OpenSpec spec as the source of truth (matches this repo's existing single-sourcing
      convention for `resolve_params`/`choose_models`/output-contract, per `openspec/project.md`).
      Add a sentence documenting that an empty (zero-scan) input directory now raises/exits
      non-zero instead of the old silent no-op — this behavior is currently undocumented anywhere
      in `README.md`.
- [ ] 5.3 Fix `API.md`'s `run_batch` section: the parenthetical "`BatchResult.ok` is `False` iff
      any scan failed (the CLI exit code)" is misleading post-change — `ok=False` now maps
      specifically to exit `3`, not "the CLI exit code" in general (a crash never reaches
      `BatchResult` construction at all). Reword to something like "(maps to CLI exit `3`, distinct
      from a staging-error/crash `1`)".
- [ ] 5.4 Add a `CHANGELOG.md` entry under `[Unreleased]`: amend the existing "Predict container
      CLI" bullet's "Per-scan failures are isolated; the process exits non-zero iff any scan
      failed" sentence to describe the final `0`/`3`/default-`1`/`143` scheme and the
      empty-input-now-raises change, and drop/update its closing "Argo-readiness hardening (#26)
      are follow-ups" note now that #26 has landed alongside it.
- [ ] 5.5 Run `/lint` and `/test` (full suite, non-gpu/wandb/acceptance markers) — all green.
- [ ] 5.6 Run `openspec validate harden-argo-exit-semantics --strict` — resolve every issue.
