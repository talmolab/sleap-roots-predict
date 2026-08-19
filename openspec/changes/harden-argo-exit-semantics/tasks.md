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
the suite self-contradictory. Tasks 1.1 and 1.3 below now name both tests explicitly. Also
decided: the log-quality regression from dropping the `except` clause entirely (a missing-mount
error would otherwise dump a raw traceback instead of a clean one-line log) is worth preserving
— 1.5 keeps a narrow `except (FileNotFoundError, ValueError)` that only logs before re-raising.

**Revised again after `/review-openspec` round 2:** two more corrections.
1. Task 1.3's first draft prescribed `assert main(...) == 1` for the rewritten
   `test_cli_missing_input_dir_returns_nonzero` — **wrong**, since a bare `raise` inside the new
   `except` clause re-raises the exception rather than returning a value, so calling `main([...])`
   directly (as every test in this file does) raises `FileNotFoundError` out of the call; it
   never reaches a `return` statement to compare against `1`. Corrected below to
   `pytest.raises(FileNotFoundError)` around the `main()` call.
2. The `except (FileNotFoundError, ValueError)` clause actually covers **all four** staging-error
   cases, not "two known staging-error types" as earlier drafts said: `json.JSONDecodeError` and
   `pydantic.ValidationError` both subclass `ValueError` (confirmed empirically), so a malformed
   `run_manifest.json` and the new zero-scans-discovered `ValueError` (section 2) both get the
   clean log line too, alongside the missing-directory and duplicate-`scan_key` cases. Wording
   below corrected accordingly.

- [ ] 1.1 In `tests/test_batch.py` (or `test___main__.py`), write a failing test asserting
      `main()` returns `3` (not `1`) for a batch with one failed scan among otherwise-ok scans,
      **and update `test_cli_main_exit_codes`'s existing `assert main(...) == 1` (failed-batch
      case) to `== 3`** — do not leave the old assertion alongside the new test; they cover the
      same scenario and must agree.
- [ ] 1.2 In the same file, write a failing test asserting `main()` still returns `0` for an
      all-ok/all-skipped batch (pins the unchanged success path).
- [ ] 1.3 In the same file, write a failing test asserting a missing input directory now
      propagates as an uncaught `FileNotFoundError` from `main()` (not a returned `2`) — this is a
      **behavior-change** regression test, not a no-op pin: it must fail against the current code
      (which still returns `2`) until task 1.5 removes the special case.
      **Rewrite `test_cli_missing_input_dir_returns_nonzero`** (currently
      `assert main(...) == 2`) to `with pytest.raises(FileNotFoundError): main([...])` — NOT
      `assert main(...) == 1` (that assertion is unreachable: `main()` re-raises on this path, it
      never returns). Optionally assert the clean log line was emitted too (e.g. via `caplog`).
      Depends on task 1.5 landing first — write this test failing against today's code, confirm it
      passes only after 1.5's `except ...: log; raise` change is in place.
      **This section's CLI-level empty-input test has a cross-section dependency, noted here
      explicitly (per `/review-openspec` round 2's commit-sequencing finding):** add a second test
      asserting the new empty-input `ValueError` (implemented in task 2.4, section 2) also
      propagates uncaught from `main()` the same way (`pytest.raises(ValueError)`). This sub-test
      has nothing to exercise until section 2's `run_batch` change lands — sequence section 2's
      commit before finalizing this one, or write both together in one commit if landing them
      separately isn't worth the overhead for a change this size.
- [ ] 1.4 In the same file, write a failing test asserting a CLI usage error (invoke `main()`/the
      module with a missing required argument) exits `2` via `argparse`, and that this is
      independent of the driver's own `0`/`1`/`3` codes (documents the boundary so a future change
      can't quietly blur it again).
- [ ] 1.5 Implement: in `sleap_roots_predict/__main__.py`, change `return 0 if result.ok else 1` to
      `return 0 if result.ok else 3`. Replace the existing `except (FileNotFoundError, ValueError):
      return 2` clause with `except (FileNotFoundError, ValueError) as exc:
      logging.getLogger(__name__).error("Batch aborted: %s", exc); raise` — this keeps the
      existing clean one-line log message for every `FileNotFoundError`/`ValueError`-raising
      staging condition (missing directory, duplicate `scan_key`, malformed manifest, and
      zero-scans-discovered — see the note above) while letting the exception continue on to
      Python's default unhandled-exception exit `1`, same as any other uncaught crash type. Update
      the module docstring and `main()`'s docstring to enumerate the three driver-owned codes
      (`0`/`3`/default `1`) plus a note that `2` is reserved by `argparse`, not by this driver.
      Run 1.1–1.4 green.

## 2. Empty-input guard (D2): raise instead of silent no-op

- [ ] 2.1 In `tests/test_batch.py`, write a failing test asserting `run_batch` raises `ValueError`
      over a present-but-empty input directory (no `*.scan_metadata.json`), and that it raises
      *before* any `WarmModelWorker`/model-source interaction (assert via a source stub that
      records whether it was ever called).
- [ ] 2.2 In `tests/test_batch.py`, write a failing test asserting a `run_manifest.json` present
      but scoping to zero `scan_keys` also raises (via `RunManifest`'s own validation inside
      `discover_scans` — a distinct raise site from 2.1's zero-scans-discovered check, both
      landing on the same CLI exit code; confirm which layer raises and assert that one; do not
      add a redundant check if `RunManifest` itself already enforces non-empty `scan_keys`).
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
      **Land this commit before finalizing task 1.3's cross-section empty-input sub-test** (see
      the note there).

## 3. Atomic writes (D3): `.slp`, manifest, and sidecar copy

**Note found in `/review-openspec` round 1:** `sio.save_file` infers the output format purely
from the destination filename's extension (confirmed by reading `sleap_io`'s source) and raises
`ValueError: Unknown format` if it doesn't recognize one. This repo's own existing atomic-write
precedent (`sleap_roots_predict/parity.py`'s `write_report`, `foo.json` → `foo.json.tmp`) would
break `.slp` writes if copy-pasted verbatim, since `foo.slp.tmp` no longer ends in `.slp`. Task
3.5 below calls this out explicitly so it's designed in from the start, not discovered as a test
failure partway through. This constraint is also now recorded directly in
`specs/prediction-output/spec.md` (not just here), since `tasks.md`/`design.md` become historical
after archiving and the persisted spec is what a future maintainer changing the temp-naming
scheme would actually consult.

- [ ] 3.0 In `tests/test_output_contract.py`, write a failing test asserting `write_prediction_outputs`
      succeeds even when its internal `.slp` temp filename does not itself end in `.slp` (e.g. by
      monkeypatching the temp-path construction, or simply asserting success once 3.5 is
      implemented with a `.tmp`-suffixed temp name) — pins that `format="slp"` is passed explicitly
      rather than relied-upon-via-extension (see the note above and the corresponding new scenario
      in `specs/prediction-output/spec.md`).
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
      strictly last. Run 3.0–3.3 green; confirm all existing round-trip tests in
      `test_output_contract.py` still pass unchanged.
- [ ] 3.6 Implement the same atomic pattern for the sidecar copy in
      `sleap_roots_predict/batch.py::_predict_one` (replace the direct `shutil.copyfile` with
      copy-to-temp + `os.replace`; a plain file copy has no format-inference concern, unlike 3.5).
      Run 3.4 green; also confirm the existing `test_sidecar_copy_failure_leaves_no_manifest`
      (which monkeypatches `batch_mod.shutil.copyfile`) still passes — the copy-to-temp step still
      calls `shutil.copyfile` internally, so it should, but verify rather than assume. Land 3.5 and
      3.6 as separate commits — independent write sites, no shared code path (per
      `/review-openspec` round 1's commit-strategy feedback).

## 4. SIGTERM handler (D4): stop at scan boundary, exit 143

**Note found in `/review-openspec` round 1:** on Windows, real cross-process `SIGTERM` delivery
does not invoke a registered Python handler (see `specs/predict-container/spec.md`'s "Graceful
SIGTERM handling" requirement for the full explanation — that's the canonical statement of this
constraint; this note is just a pointer, not a restatement). The plan below already avoids the
trap (4.3 tests the handler by calling it directly, never via `os.kill`) — don't "improve" that
test to use `os.kill` later; on Windows CI that would silently hang or kill the job.

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
      `os.kill` — see the Windows note above), assert the event becomes set, and assert `main()`
      returns `143` when the event is already set going into its post-`run_batch` check — cover
      this against **both** a would-otherwise-be-`0` batch and a would-otherwise-be-`3` batch, to
      pin that `143` overrides either outcome, not just the success case — without mocking
      `run_batch` itself (real TDD). **Save `signal.getsignal(signal.SIGTERM)` before the test and
      restore it in a `finally`/fixture teardown**, so the handler registered by this test doesn't
      leak into later tests or interact with CI's job-level timeout kill.
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
      **Land 4.4 and 4.5 as separate commits** — `run_batch`'s `should_stop` hook
      (`sleap_roots_predict/batch.py`) and `__main__.py`'s signal-handler plumbing are independent
      write sites in different files with no shared code path, the same shape as 3.5/3.6 (per
      `/review-openspec` round 2's commit-strategy feedback).

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
- [ ] 5.4 Amend the existing `CHANGELOG.md` `[Unreleased]` "Predict container CLI" entry — it needs
      updates for **all three** behavior changes in this proposal, not just the exit-code one
      (found incomplete in `/review-openspec` round 2, which only scoped this to D1/D2):
      (a) replace "Per-scan failures are isolated; the process exits non-zero iff any scan failed"
      with the final `0`/`3`/default-`1`/`143` scheme and the empty-input-now-raises change (D1/D2);
      (b) add a mention that `.slp`/manifest/sidecar writes are now atomic (temp+rename) (D3);
      (c) add a mention of the new `SIGTERM` handler and its `143` exit (D4); (d) drop/update the
      closing "Argo-readiness hardening (#26) are follow-ups" note now that #26 has landed
      alongside this entry rather than following it.
- [ ] 5.5 Run `/lint` and `/test` (full suite, non-gpu/wandb/acceptance markers) — all green.
- [ ] 5.6 Run `openspec validate harden-argo-exit-semantics --strict` — resolve every issue.
