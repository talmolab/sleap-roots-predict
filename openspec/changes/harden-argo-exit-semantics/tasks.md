## 1. Exit codes (D1): partial (3) distinct from crash (default 1)

- [ ] 1.1 In `tests/test_batch.py`, write a failing test asserting `main()` returns `3` (not `1`)
      for a batch with one failed scan among otherwise-ok scans.
- [ ] 1.2 In `tests/test_batch.py`, write a failing test asserting `main()` still returns `0` for
      an all-ok/all-skipped batch (pins the unchanged success path).
- [ ] 1.3 In `tests/test_batch.py`, write a failing test asserting the existing
      `FileNotFoundError`/duplicate-`scan_key` abort paths still return `2` (regression pin — no
      behavior change here, but exercised alongside the new codes so the four-way split is
      tested together).
- [ ] 1.4 Implement: change `sleap_roots_predict/__main__.py`'s `return 0 if result.ok else 1` to
      `return 0 if result.ok else 3`; update the module docstring and `main()`'s docstring to
      enumerate all four codes (`0`/`2`/`3`/default `1`). Run 1.1–1.3 green.

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
      `failed` entries (exit `3` via `main()`), not the empty-input `ValueError` path — pins the
      D2/D1 boundary described in the spec delta.
- [ ] 2.4 Implement: in `sleap_roots_predict/batch.py::run_batch`, replace the
      `if not scans: logger.warning(...); return result` branch with
      `raise ValueError(f"no scans discovered under {input_dir.as_posix()}")`, placed before
      `WarmModelWorker(source=source)` is constructed. Update `run_batch`'s docstring (`Raises:`
      section) accordingly. Run 2.1–2.3 green; also re-run the existing "Empty input directory is
      a no-op" test from `test_batch.py` if present and update/replace it to assert the new
      raising behavior (do not leave a stale test asserting the old no-op contract).

## 3. Atomic writes (D3): `.slp`, manifest, and sidecar copy

- [ ] 3.1 In `tests/test_output_contract.py`, write a failing test asserting that during
      `write_prediction_outputs`, no file ever exists at a `.slp`'s final path except either
      fully absent or fully written (e.g. by monkeypatching the write step to raise after the
      temp file is created but before the rename, then asserting the final path does not exist).
- [ ] 3.2 In `tests/test_output_contract.py`, write the equivalent failing test for
      `{scan_key}.predictions.json` (temp file created, final path untouched until rename).
- [ ] 3.3 In `tests/test_batch.py`, write a failing test for the sidecar copy in
      `batch.py::_predict_one` with the same interrupted-before-rename assertion.
- [ ] 3.4 Implement atomic writes in `sleap_roots_predict/output_contract.py`'s
      `write_prediction_outputs`: write each `.slp` via `sio.save_file` to a temp path in the
      same directory, then `os.replace` into the final path; same pattern for the manifest's
      `.write_text`, keeping the manifest write strictly last. Run 3.1–3.2 green; confirm all
      existing round-trip tests in `test_output_contract.py` still pass unchanged.
- [ ] 3.5 Implement the same atomic pattern for the sidecar copy in
      `sleap_roots_predict/batch.py::_predict_one` (replace the direct `shutil.copyfile` with
      copy-to-temp + `os.replace`). Run 3.3 green.

## 4. SIGTERM handler (D4): stop at scan boundary, exit 143

- [ ] 4.1 In `tests/test_batch.py`, write a failing test asserting `run_batch(..., should_stop=fn)`
      stops after the first scan (of two) when `fn` returns `True` starting from the second
      iteration, and that the first scan's outputs are complete and valid (reloadable manifest +
      `.slp`).
- [ ] 4.2 In `tests/test_batch.py`, write a failing test asserting `run_batch` with the default
      `should_stop` (no argument passed) is unaffected — processes all scans exactly as before
      (regression pin for existing callers/tests).
- [ ] 4.3 Add a test (`test_batch.py` or a new `tests/test_main.py` if signal-handling tests don't
      fit `test_batch.py`'s existing scope) driving `__main__.py`'s handler registration and event
      logic directly: invoke the registered `SIGTERM` handler function, assert the shared event
      becomes set, assert `main()` returns `143` in that case — without mocking `run_batch` itself
      (real TDD; only the signal delivery is simulated by calling the handler directly rather than
      `os.kill`, since a real cross-process signal test is unnecessary to exercise this logic).
- [ ] 4.4 Implement: add `should_stop: Callable[[], bool] = lambda: False` (keyword-only) to
      `run_batch`, checked at the top of the per-scan `for` loop (`if should_stop(): logger.warning(...);
      break`). Run 4.1–4.2 green.
- [ ] 4.5 Implement: in `sleap_roots_predict/__main__.py`'s `main()`, create a `threading.Event`,
      register a `signal.signal(signal.SIGTERM, ...)` handler that sets it, pass
      `should_stop=event.is_set` into `run_batch`, and after `run_batch` returns, check the event
      first — if set, log and `return 143` before falling through to the normal 0/2/3 logic. Run
      4.3 green.

## 5. Docs and closeout

- [ ] 5.1 Update `sleap_roots_predict/__main__.py`'s module docstring (already touched in 1.4) and
      `sleap_roots_predict/batch.py`'s `run_batch` docstring (`Raises:`/behavior description) to
      match the final implemented behavior — re-read both after 1–4 land to catch any drift from
      the design doc's phrasing.
- [ ] 5.2 Run `/lint` and `/test` (full suite, non-gpu/wandb/acceptance markers) — all green.
- [ ] 5.3 Run `openspec validate harden-argo-exit-semantics --strict` — resolve every issue.
