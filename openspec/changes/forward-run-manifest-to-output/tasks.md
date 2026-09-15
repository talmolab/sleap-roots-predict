> **Commit discipline.** Each numbered section below is ONE commit. The "verify FAIL" steps are
> working-tree checkpoints, never commit points — committing between a red test and its
> implementation leaves CI red on all three OS legs.
>
> **Standing invariant:** never stage a `run_manifest.json` into `tests/assets/scans/`.
> `test_sidecar_copy_failure_leaves_no_manifest` and
> `test_sidecar_copy_leaves_no_partial_file_if_replace_fails` patch the **global**
> `shutil`/`os.replace`; once `run_batch` calls the forward-copy first, a manifest in that shared
> fixture would make them intercept the wrong call and fail misleadingly.

## 0. Proposal

- [x] 0.1 Commit the OpenSpec proposal (`proposal.md`, `design.md`, `tasks.md`, the
      `predict-container` delta) before any code.

## 1. Module: `copy_run_manifest_forward` (TDD)

- [x] 1.1 Write `tests/test_run_manifest.py` with failing unit tests, before any implementation.
      Rules for all of them: assert with `read_bytes()` against the source's own bytes (never a
      byte literal — `write_text` emits `\r\n` on Windows); write fixtures with `write_bytes`;
      never use `chmod` to simulate unwritability; never use a symlink. Assert residue by the
      directory's **exact contents** (`{p.name for p in out.iterdir()} == {"run_manifest.json"}`),
      never `glob("*.tmp")` — `mkstemp`'s default name has no `.tmp` suffix, so a glob assertion
      is vacuous.
  - **byte-copy, not re-serialization** — source is `b"{not valid json"`; assert the destination
    is byte-identical. A re-serializing implementation raises; a byte copy succeeds.
  - **byte-fidelity of valid content** — source has non-canonical key order, an undeclared extra
    field, extra whitespace, and no trailing newline; assert byte-identical.
  - **temp file is created inside `output_dir`** — spy on
    `sleap_roots_predict.run_manifest.tempfile.mkstemp`. The spy **must delegate** to the real
    `mkstemp` (a fake `(fd, name)` breaks `os.close`/`copyfile` downstream), and must read `dir`
    from `kwargs.get("dir", args[2] if len(args) > 2 else None)` so a positional call still
    works. Assert `Path(seen[0]).resolve() == output_dir.resolve()` — resolved, not string
    equality, or macOS `/private/var` breaks it. Nothing else catches a system-temp
    implementation: `tmp_path` and `/tmp` are the same device on all three CI runners, so the
    `EXDEV` bug is production-only.
  - **temp name is unique per writer** — patch `os.replace` to append its source to a list and
    raise; call the function **twice, each wrapped in its own `pytest.raises(OSError)`** (the
    implementation re-raises, so an unwrapped first call aborts the test before the second).
    Assert `len(seen) == 2` (guards an implementation that never calls `os.replace`), then
    `seen[0] != seen[1]`, then that the output directory is empty. Comparing recorded *strings*
    is fine even though cleanup unlinks the files. Nothing else pins Decision 1's reason 5, and
    the likeliest wrong implementation (copying `batch.py:392`'s `with_name(name + ".tmp")`)
    passes every other test here.
  - **forwarded file's mode matches the source** — POSIX only
    (`@pytest.mark.skipif(os.name == "nt", ...)`). `os.chmod(src, 0o640)` first: 0640 is neither
    `mkstemp`'s 0600 nor umask-022's 0644, so the assertion fails against both a missing
    `copymode` **and** a hardcoded `0o644`. Asserting against a `write_bytes` default instead
    would pass silently under `umask 077` — the exact hardened posture this protects.
  - **a malformed manifest is never forwarded** — parametrized over **both** arms of the
    scenario: `b"{not valid json"` (invalid JSON) and
    `b'{"pipeline_run_id":"r","scan_keys":[]}'` (valid JSON, fails `RunManifest` validation).
    Stage in the input dir, then `run_batch`: assert it raises and `output_dir` was never
    created. Paired with the unparsable-source unit test above, this is what pins Decision 2's
    strongest argument (discovery validates first); without it, moving the call above
    `discover_scans` passes every other test while regressing the decision the placement rests on.
  - **a file occupies the `output_dir` path** → `mkdir` raises `FileExistsError` (verified on
    both Windows and Linux): assert the error is an `OSError`, **not** a `NameError`, that an
    ERROR naming both directories is logged, and that nothing is left behind. This is the only
    test that catches the `UnboundLocalError` cleanup bug, and it needs no `chmod` or symlink.
  - **absent → no-op**, nothing written to `output_dir`, and a DEBUG line records the no-op
  - **absent source but a stale manifest already in `output_dir`** → the stale file is unchanged
    byte-for-byte and a WARNING naming it (`.as_posix()`) is logged
    (`caplog.at_level(logging.WARNING, logger="sleap_roots_predict.run_manifest")`)
  - **absent AND `input_dir == output_dir` → returns cleanly** (pins presence-checked-before-
    identity; an identity check first would raise on a nonexistent source). Also confirm no
    spurious stale-warning fires here: source and destination are the same path, so
    `destination.is_file()` is False and only DEBUG should be emitted.
  - **same-file → no-op**, manifest unchanged, no residue. Use a textually-different but
    equivalent path (`input_dir / "sub" / ".."`) — and **`mkdir` `sub` first**: Windows collapses
    `..` lexically and passes without it, POSIX raises `ENOENT`, so omitting the mkdir yields a
    Windows-green / Ubuntu-and-macOS-red test.
  - **`output_dir` created when missing**, including the nested case (`out/a/b`)
  - **stale output manifest replaced** — seed a *different-length* stale file; assert the result
    is byte-identical to the new source with no fragment of the old content surviving
  - **no residue on success** (exact-contents form)
  - **`os.replace` raises after a successful copy** → temp removed, no partial destination, error
    propagates. This is the test that actually exercises cleanup; patching `copyfile` alone is
    vacuous (the temp never existed), which is why there is no separate `copyfile`-raises test.
    Mirror `tests/test_batch.py::test_sidecar_copy_leaves_no_partial_file_if_replace_fails`.
  - **a directory at the *destination* path** → `os.replace` raises; assert `pytest.raises(OSError)`
    rather than a platform-specific subclass.
  - **`str` paths accepted on BOTH the success and failure paths** — the failure arm is the one
    that matters: it is what catches `input_dir.as_posix()` on a `str` argument, which would
    raise `AttributeError` and escape the CLI's `OSError` handler.
  - **success logs at INFO naming both directories** — requires
    `caplog.at_level(logging.INFO, logger="sleap_roots_predict.run_manifest")`; a bare `caplog`
    captures nothing below WARNING (this repo sets no `log_level`, and `basicConfig` in
    `__main__` is a no-op under pytest). Assert on `input_dir.as_posix()`, matching what the
    implementation logs — `str(Path)` yields backslashes on Windows and would flake that leg.
  - **failure logs at ERROR naming both directories before propagating** (same `.as_posix()` rule)
  - Verify these FAIL. Then add a no-op stub and re-run: the no-op cases pass trivially under a
    stub and are guard tests, not driver tests — do not count them as TDD coverage.
- [x] 1.2 Create `sleap_roots_predict/run_manifest.py` with `copy_run_manifest_forward(input_dir,
      output_dir) -> None`. Import **only** `RUN_MANIFEST_FILENAME` from `sleap_roots_contracts`
      (not `RunManifest` — unused, and this module performs no validation). Order is load-bearing:
  1. `source.is_file()` → else: if `destination.is_file()`, `logger.warning` naming the stale
     destination (it is left in place — deleting a file in a shared directory a concurrent
     invocation may have just written is not predict's call); otherwise `logger.debug`. Return
     either way. The warning is the only signal for a stale-pin/rolled-back upstream producer,
     which is the realistic way this arises.
  2. Identity check, **guarded**: `os.path.samefile` stats *both* operands and raises
     `FileNotFoundError` when the destination does not exist — the ordinary case — so it can
     never be called bare. Wrap in `except (FileNotFoundError, NotADirectoryError): pass`,
     those two specifically, so a `PermissionError` from `stat` still propagates.
  3. **`tmp: str | None = None` BEFORE the `try:`**, and the `try:` opens here, wrapping steps
     3-5. This pre-binding is load-bearing, not style: if `mkdir` or `mkstemp` raises, a cleanup
     block that unlinks an unbound `tmp` raises `UnboundLocalError` — which is a `NameError`,
     **not** an `OSError`, so the widened CLI handler would not catch it and the run would take
     the raw-traceback path. Reproduced on both Windows and Linux. Hoisting `mkdir`/`mkstemp`
     out of the `try` instead is also wrong: then those failures emit no error log at all.
  4. `destination_dir.mkdir(parents=True, exist_ok=True)` — required before `mkstemp`, which
     does not create its `dir=`.
  5. `fd, tmp = tempfile.mkstemp(dir=destination_dir, prefix=f".{RUN_MANIFEST_FILENAME}.",
     suffix=".tmp")` then **`os.close(fd)` immediately** — `mkstemp` hands back an open fd and
     on Windows an open handle makes `os.replace` fail with `WinError 32`, while `copyfile`
     still succeeds, so the bug surfaces only at the replace and only on the Windows leg.
     Explicit `suffix` keeps residue assertions meaningful; the dot prefix hides a `SIGKILL`
     orphan from any future `run_manifest*` glob. Closing early is safe: the name stays reserved
     by the file's existence, and 200 consecutive calls were verified collision-free.
  6. `shutil.copyfile(source, tmp)` → `shutil.copymode(source, tmp)` → `os.replace(tmp,
     destination)`. `copymode` **before** the replace, never after — after would publish the
     destination at the private temp mode first, creating exactly the window it exists to close.
     Verified: bloomctl writes the source at 0644 via an ordinary `write_bytes` (pinned by a
     test in that repo), so `copymode` yields 0644, not 0600. Do **not** add a hardcoded mode
     floor; it would contradict the spec's "match the source" wording.
  7. On failure, in this order — **log first, then clean up**:
     ```python
     except Exception:
         logger.error(
             "Failed to forward run_manifest.json from %s to %s",
             source_dir.as_posix(), destination_dir.as_posix(),
         )
         if tmp is not None:
             try:
                 Path(tmp).unlink(missing_ok=True)
             except OSError:
                 logger.warning("Could not remove temporary file %s", Path(tmp).as_posix())
         raise
     ```
     Cleaning up *before* logging is a real bug, not a style preference: if the `unlink` itself
     raises, the error log never fires and the exception that propagates is the **secondary**
     cleanup error, not the underlying filesystem one — violating the spec's "the error raised
     is the underlying filesystem error, not a secondary error from the cleanup path". It is
     reachable unmocked on Windows (a read-only source → `copymode` marks the temp read-only →
     `os.replace` fails → `unlink` fails too). The bare `raise` still re-raises the original
     after the inner handler runs. **Corrected in 6.4** — this was recorded here and in
     `design.md` as uncatchable "since 1.1 bans `chmod`"; it needs no `chmod`, only two
     injected failures, and is now pinned by
     `test_a_failing_cleanup_does_not_swallow_the_real_error`.
  - Naming: the signature is `(input_dir, output_dir)`; derive **`source_dir = Path(input_dir)`**,
    `source = source_dir / RUN_MANIFEST_FILENAME`, `destination_dir = Path(output_dir)`,
    `destination = destination_dir / RUN_MANIFEST_FILENAME` up front. `source_dir` is
    load-bearing, not tidiness: logging `input_dir.as_posix()` raises `AttributeError` on the
    `str` arguments `__main__.py` passes — and `AttributeError` is not an `OSError`, so it would
    escape the CLI's handler into the raw-traceback path. That is the same failure *shape* as
    the `UnboundLocalError` bug, in the same block.
  - Every log line names directories via `.as_posix()` — the ERROR, the success INFO, **and** the
    stale-destination WARNING. `str(Path)` yields backslashes on Windows and flakes that leg.
  - Docstrings: Google style (ruff `D`, `convention = "google"`). `D415` requires the summary to
    end in a period; `D417` requires every parameter in `Args:`. Include `Raises:` — ruff won't
    enforce it, and "this raises rather than degrading" is the contract. The module docstring
    must (a) disambiguate `RunManifest` from `PredictionManifest`, and (b) record the two
    deliberate divergences from `trait_extractor/run_manifest.py` — raises instead of
    best-effort, and adds temp cleanup — one sentence each, pointing at `design.md` rather than
    restating the reasoning. Keep lines <=88 by hand (ruff selects only `D`; no `E501`).
- [x] 1.3 `pytest tests/test_run_manifest.py` green; `black`/`ruff` clean. Commit.

## 2. Wire into `run_batch` + CLI (TDD)

- [x] 2.1 Add failing integration tests to `tests/test_batch.py`, before touching `batch.py`.
      **First extract `_RecordingSource` to a module-level `_recording_source()` factory** — it
      is currently function-local inside `test_empty_input_raises_before_worker_interaction`
      (:300), and five copy-pasted classes is not acceptable. Use the real `all_roots_source`
      **only** where a successful prediction is required (b1, b2), with `_real_scan` (:234) —
      **not** `_write_scan`, whose 16×16 frames yield `instances=0` and empty `.slp`s, and which
      sleap-nn logs as *slower* because it upscales and pads. Model loading, not frame count, is
      the cost (measured: ~2.5s steady-state per real-source batch, <0.005s per stub batch), so
      stubbing the source is the only real lever, and it makes b3-b7 stronger by proving the
      forward hop is independent of prediction. With `list_cards() -> []`, `_predict_one` raises
      at its `if not refs:` guard — before `out_scan_dir.mkdir()` — so a stub batch writes
      nothing at all. Stage manifests into `tmp_path` only.
  - b1 manifest forwarded byte-identically, and **not** written into any per-scan subdirectory
    (assert the top-level entry set). Stage the *pathological* bytes the scenario names —
    non-canonical key order, an undeclared extra field, no trailing newline — via `write_bytes`;
    verified safe, since `RunManifest` is `frozen=True` with pydantic's default `extra="ignore"`,
    so discovery accepts it while any re-serialization would drop the extra field — stub source
  - b2 no manifest staged → none written **and** each discovered scan's outputs are still written
    as usual — **real source** (the only integration test that needs one)
  - b3 `should_stop` returns `True` immediately → manifest still forwarded (regression test
    pinning the placement decision; zero-scans-predicted is the strongest form) — stub source
  - b4 every scan fails (manifest names a `scan_key` with no sidecar) → manifest still forwarded
    and `output_dir` created containing only it. Reachable: `discover_scans` returns a failed
    `ScanInput` for such a key, so `scans` is non-empty and the guard passes — stub source
  - b5 copy failure → raises out of `run_batch`, **no scan predicted** (assert via the recording
    source: `list_cards` count 0, `materialize` raises if touched) and `assert not out.exists()`.
    Do **not** assert "no residue" — the whole function is patched out, so there is no temp to
    leave and the assertion would be vacuous. Inject by patching
    `batch.copy_run_manifest_forward`, not global `shutil` — stub source
  - b6 `run_batch(inp, inp, ...)` with a manifest → manifest unchanged, batch does not raise, and
    the input directory's **exact contents** are unchanged. (With a stub source nothing is
    written at all, so the full-contents assertion holds and recovers the "no additional file"
    clause the same-path scenario needs) — stub source
  - b7 pre-existing sibling scan output in `output_dir` → assert the **positive**:
    `{p.name for p in out.iterdir()} == {"scanOTHER", "run_manifest.json"}`. Asserting only that
    the sibling is untouched passes against a no-op implementation. Also seed
    `out/scanOTHER/scanOTHER.predictions.json` with known bytes and assert those bytes **and**
    its `st_mtime_ns` are unchanged — a name-set check alone would pass against an
    implementation that rewrote the sibling's contents — stub source
  - b8 stale output manifest + no input manifest, through `run_batch` → the stale file is
    unchanged and the WARNING fires. The spec attributes this to "the runner", so it needs
    integration coverage, not only the 1.1 unit test — stub source
  - Verify these FAIL first.
- [x] 2.2 Call `copy_run_manifest_forward(input_dir, output_dir)` in `run_batch` **immediately
      after the `if not scans: raise` guard and before `resolve_identity` /
      `WarmModelWorker(...)`** (insertion at `batch.py:313`-`314`; only the inert
      `result = BatchResult()` sits between). No `try`/`except`.
      The ordering relative to the empty-batch guard is behaviorally unobservable (a non-empty
      manifest always yields >=1 scan entry; an empty `scan_keys` fails validation in discovery),
      so do not write a test for it.
- [x] 2.3 Widen `__main__.py`'s handler from `except (FileNotFoundError, ValueError)` to
      `except (OSError, ValueError)`. `FileNotFoundError` subclasses `OSError`, so existing
      coverage is preserved — verified: no existing test asserts the negative.
  - Add `test_cli_forward_copy_failure_propagates_as_default_exit_1`, mirroring
    `test_cli_missing_input_dir_propagates_as_default_exit_1` (:563): `caplog.at_level("ERROR")`
    + `pytest.raises(PermissionError)` + `assert any("Batch aborted" in r.message ...)`.
    **Do not name it "exits 1"** — `main()` re-raises; the `1` is the interpreter's default for
    an unhandled exception and is not observable in-process. The both-directories assertion
    belongs in 1.1's unit test, not here: the CLI's line interpolates only `str(exc)`.
  - Add a SIGTERM-ordering test: stop already requested + copy fails → the exception propagates
    (not a `143` return). Mechanism matters here — the four existing SIGTERM tests (:359-:447)
    all fire the handler *after* `run_batch` returns, which is the wrong side. Wrap
    `batch_mod.run_batch` with a delegating spy that first invokes
    `signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)` and then calls through, with
    `batch.copy_run_manifest_forward` patched to raise; assert `pytest.raises(PermissionError)`
    around `main([...])`, restoring the prior handler in a `finally` as :411-424 does. Name it
    `test_forward_copy_failure_during_requested_stop_propagates`, use `scan_input_dir` (it must
    hold at least one discoverable scan or `run_batch` raises `ValueError` at the empty-batch
    guard and the test passes for the wrong reason), and pass **no** `source` — the raise at
    `batch.py:313` precedes `WarmModelWorker(source=source)` at `:316`, so no registry call is
    possible. Same applies to the copy-failure CLI test above.
- [x] 2.4 Update docstrings: `run_batch` (forward-copy + new `Raises:` condition), `batch.py`'s
      module docstring, and **`__main__.py`'s module docstring and `main`'s `Returns:`**, both of
      which enumerate the staging-error causes and go stale, plus the inline comment describing
      the old catch.
- [x] 2.5 `pytest tests/test_batch.py` green; `black`/`ruff` clean. Commit.

## 3. Public API

- [x] 3.1 Add `copy_run_manifest_forward` to `tests/test_public_api.py`'s name tuple; confirm
      FAIL. The existing generic `hasattr` + `__all__` check is the whole test — no new function.
- [x] 3.2 Export it from `sleap_roots_predict/__init__.py` (import + `__all__`) and extend the
      package docstring's feature list.
- [x] 3.3 `pytest tests/test_public_api.py` green. Commit 3.1-3.3 **together** (3.1 alone is red).

## 4. Docs

- [x] 4.1 `API.md`: document `copy_run_manifest_forward`; extend the `run_batch` entry to cover
      forwarding. **Also fix pre-existing drift in that same block**: the documented `run_batch`
      signature lists `peak_threshold: float = 0.2` and `batch_size: int = 4`, neither of which
      exists in the real signature.
- [x] 4.2 `README.md`: three edits — (a) the container prose enumerating what is written per
      scan, now incomplete; (b) the **Project Structure module tree** (`run_manifest.py`);
      (c) the **tests tree**, which lists every test file (`test_run_manifest.py`).
- [x] 4.3 `CHANGELOG.md`: amend the existing `**Predict container CLI**` bullet under
      `## [Unreleased]` → `### Added` rather than adding a new bullet (the repo's pattern —
      `consume-run-manifest` was folded into that same bullet). `Added`, not `Fixed`: nothing has
      been released, so no published claim is being corrected. While there, fix the pre-existing
      garble in that bullet — the orphaned "and per scan," clause belongs with the skip-if-done
      sentence, not with manifest scoping.
- [x] 4.4 `openspec/project.md`: add `run_manifest.py` to the authoritative module layout.
- [x] 4.5 Commit docs.

## 5. Verification gate

- [x] 5.1 `openspec validate forward-run-manifest-to-output --strict` passes.
- [x] 5.2 Run the full local gate via `/pre-merge`. Use **ci.yml's exact marker expression**
      (`-m "not gpu and not acceptance and not wandb"`), which deliberately differs from
      `pyproject.toml`'s `addopts` (that one also excludes `parity`); a plain `-m "not gpu"`
      pulls in flaky wandb tests. ruff lints `sleap_roots_predict/` and `scripts/` only, not
      `tests/`; black covers all three.
- [x] 5.3 File the follow-up issue: promote `bloomctl`'s union-merge + lockfile into
      `sleap-roots-contracts` and switch predict *and* `sleap-roots` traits onto it (traits'
      copy has neither temp cleanup nor a unique temp name, so it carries the same truncation
      hazard fixed here). Include: `pipeline_run_id` is last-writer-wins by design; the
      `O_CREAT|O_EXCL` lock is already running on NFS in production but was never analyzed for
      it. **Frame it as a trip-wire, not a someday**: when concurrent chunked dispatch is enabled
      in the frontend, the union-merge becomes a blocker. Verified no equivalent issue exists in
      `sleap-roots` today. Filed: talmolab/sleap-roots-predict#40
- [x] 5.4 File the deploy/verify tracking issue described under "Post-merge handoff" below and
      recorded: talmolab/sleap-roots-predict#41
- [x] 5.5 Make the Decision 3 trip-wire real, since a note in a doc that gets archived is not a
      mechanism and the trigger fires in another repo. Comment on `sleap-roots-pipeline#56`
      (or add a task-list item to its body) recording that when exit-code discrimination lands
      and trait-extraction can run after a partial predict, predict's forwarded-manifest
      semantics must change from "requested scope" to "delivered scope" (`ok ∪ skipped`).
      Recorded: talmolab/sleap-roots-pipeline#56 (issuecomment-5626571836)
- [x] 5.6 Open the PR referencing issue #39 and this change-id. State in the body: the
      naive-copy-vs-merge decision, the forward-unchanged semantics decision and its #56
      trip-wire, the sticky-manifest rollback hazard, and the deployment ordering constraint.
      Then present READY TO MERGE and stop — do not merge.

## 6. Post-review fixes (adversarial `/review-pr`, 5-lens)

Five lenses reviewed PR #42 with CI already green and no prior review comments. No lens found a
shipped correctness defect in `copy_run_manifest_forward`; the required set below is one
normative spec clause the code did not satisfy, plus four tests that documented guarantees they
did not enforce — each *proven* vacuous by mutation rather than suspected. Everything else the
review raised is parked (see "Deliberately not done" below), because `design.md` records
revision as the dominant source of new defects in this change and the required set is kept
minimal on purpose.

**Each fix below was driven test-first.** For the guard tests (6.2–6.4) the red step is a
mutation, not a failing assertion against the shipped code: the property already holds, so the
only way to prove the test is load-bearing is to break the implementation and watch it fail.
Mutations were applied to a copy, run, and reverted — never committed.

- [x] 6.1 **Spec violation: a `PermissionError` from the identity check escaped the mandated
      error log.** `spec.md`'s forward-copy requirement says the copy SHALL log both directories
      before raising, and the scenario "A failure before the output directory can be prepared is
      still reported cleanly" names "or its parent denies permission" as an example. The
      identity check sat *outside* the `try` that logs, so that example raised with zero log
      records — reachable on the shared NFS mount, where the next stage runs as a different uid
      and can leave an output subtree this process cannot search. Fix: move the identity guard
      inside the logging `try` (inner `except (FileNotFoundError, NotADirectoryError)` kept).
      Red: `test_permission_error_from_the_identity_check_is_reported_cleanly` failed on
      `assert []`. Injected at the identity seam, not via `chmod` — 1.1 bans it.
- [x] 6.2 **The identity guarantee was untested, and the primitive had a fail-silent direction.**
      Two mutations each passed the whole suite: replacing `os.path.samefile` with
      `str(source) == str(destination)` — the one thing the spec forbids — and deleting the
      identity block outright. Both tests were blind because with no check the function copies
      to a temp in the same directory and replaces the file over itself: bytes and directory
      contents unchanged, which is all either test asserted. The `sub/..` spelling is defeated
      by `Path.resolve()`, so it never discriminated the normalization implementation the spec
      singles out either. Fix, two halves:
      (a) `test_hard_linked_destination_in_another_directory_is_a_noop` — a hard link gives two
      path strings sharing no prefix that name one file, which is the same property a
      bind-mounted `output_dir` has, testable without the symlink privilege 1.1 rules out; the
      inode assertion is the discriminator. Both identity tests were also strengthened with an
      mtime assertion (a replace publishes a *new* file), as was
      `test_run_batch_same_input_and_output_leaves_the_manifest_intact` at the integration level.
      (b) `_is_same_file` now open-codes the `(st_dev, st_ino)` comparison and rejects a
      degenerate stat: on Windows `os.stat` falls back to reporting both as `0` when a file
      cannot be opened, and two such stats compare *equal* — making an unrelated pair look
      identical and skipping the forward silently, the one outcome this module exists to
      prevent. A zero inode now means "not provably the same file"; re-copying a file onto
      itself is harmless, silently skipping a real forward is not.
      Red: `test_a_zero_inode_pair_is_not_treated_as_the_same_file` failed with the stale
      destination manifest surviving; the two identity tests fail against the string-equality
      mutant (they passed against it before).
- [x] 6.3 **`test_forward_copy_failure_during_requested_stop_propagates` passed with its own
      mechanism disabled.** 2.3 called the mechanism load-bearing, but replacing the spy's
      `signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)` with `pass` still passed:
      `PermissionError` propagates whether or not a stop was requested, so the test was a
      duplicate of `test_cli_forward_copy_failure_propagates_as_default_exit_1` and the scenario
      it claims ("A pre-flight staging error is not converted to the stop code") had no
      coverage. Fix: assert the precondition inside the spy (`assert kwargs["should_stop"]()`)
      and assert the stop warning is absent rather than implying it. Red: fails against the
      `pass` mutant. Note the fragility this pins — it bites only because `__main__` imports
      `run_batch` *inside* `main()`.
- [x] 6.4 **Two invariants the plan argues at greatest length survived mutation.**
      (a) Replace-atomicity over an *existing* manifest: `test_failed_replace_cleans_up_and_
      publishes_nothing` seeds an empty output dir, so only the "nothing" arm of the spec's
      "either nothing, or the complete prior manifest" was exercised — inserting
      `destination.unlink(missing_ok=True)` before `os.replace` passed everything, yet that lets
      a reader observe *no* manifest (the unscoped fallback this change prevents) and leaves the
      shared dir with none if the replace then fails. Added
      `test_failed_replace_leaves_a_prior_manifest_complete`.
      (b) Log-before-cleanup: added `test_a_failing_cleanup_does_not_swallow_the_real_error`,
      which also gives the only coverage of the "Could not remove temporary file" branch.
      **The 1.2 note claiming this was uncatchable "since 1.1 bans `chmod`" was wrong and is
      corrected in place** — it needs no `chmod`, only two injected failures. Worth recording
      *why* it looked uncatchable: a bare block swap is not the bug, because the inner
      `except OSError` absorbs the secondary error either way. The real mutant is cleanup before
      logging *with the inner guard removed*, and the new test fails against exactly that.
- [x] 6.5 Full local gate re-run: `black --check`, `ruff check`, `codespell`, the CPU suite
      under ci.yml's exact marker expression, and `openspec validate
      forward-run-manifest-to-output --strict`.

**Deliberately not done** (raised by the review, parked on purpose — none is a correctness
defect, and each would widen the diff at the end of a change whose own risk register names
revision as its dominant defect source):

- The **atomic-write idiom is duplicated at three sibling sites** (`output_contract.py:193`,
  `:235`, `batch.py:413`) that still carry the fixed-temp-name truncation hazard fixed here, and
  `design.md`'s "safe because the destination is namespaced by `scan_key`" argument does not
  hold: the input manifest grows by union, so two invocations sharing an `output_dir` overlap on
  `scan_key`. Pre-existing, but this PR is where that idiom's safety is newly asserted on the
  record. Filed: talmolab/sleap-roots-predict#43 — not a fix in this PR. The issue carries the
  concrete interleaving, why skip-if-done does not close it (nothing to skip for a key with no
  prior artifacts), and the four non-obvious details any shared helper must preserve
  (`os.close` before writing, temp in the destination directory, `copymode` before the replace,
  dot-prefix) — plus the `tests/assets/scans/` landmine at `batch.py:409`, which a comment
  cannot enforce.
- The **double-read hole**: a manifest disappearing between discovery's read and the copy's read
  is a no-op, not a failure, so predict can scope its own work and forward nothing (#39
  recurring, exit `0`, nothing above DEBUG). This is Open Question 1; the fix is to forward the
  bytes discovery already validated, which also makes `spec.md`'s "structurally impossible to
  forward a corrupt manifest" true unconditionally rather than only given an atomic upstream
  writer. Deferred as the Open Question already records.
- **No `pipeline_run_id` in any log line**, leaving the disclosed sticky-rollback hazard
  untriageable from pod logs; and the rollback instruction itself lives in `proposal.md`, which
  is archived on merge, rather than in `README.md` and the pipeline template.
- `exc_info=True` on the error log; orphaned-temp reclamation; narrowing `except Exception` to
  `except OSError`; `read_bytes()` for the `batch.py` manifest read; `b1`'s unfalsifiable
  per-scan-subdirectory assertion; the missing forward-plus-successful-prediction test; the
  hardcoded `_MANIFEST` literal in `test_batch.py`; and the `design.md` wording corrections
  (the CLI `OSError` over-capture that cannot actually occur, the content-regression harm label,
  the NFS `O_EXCL` protocol-version overstatement, the missing Decision 4).

## 7. Second-round review fixes (cross-repo 8-angle pass)

A second, independent review ran from another repo after §6 landed. It found one real gap in
§6's own fix, one new failure-path state, and corroborated the deferred double read — from the
opposite direction to the first review, which is what changed the call on it. Two of its
findings were checked and **not** acted on; both are recorded below with the evidence, because
"a reviewer raised it and we did nothing" is exactly what a later reader needs explained.

Same red-step discipline as §6: mutations for the guard tests, real failing assertions for the
rest.

- [x] 7.1 **§6.1 fixed the identity check but not the presence check two lines above it.**
      `source.is_file()` / `destination.is_file()` still ran outside the block that satisfies
      "log both directories before raising". Verified against this repo's Python:
      `pathlib._IGNORED_ERRNOS` is `(ENOENT, ENOTDIR, EBADF, WSAENOTSOCK)` — `EACCES` is absent,
      so `Path.is_file()` **re-raises** `PermissionError` rather than reading as "absent". An
      unreadable source manifest (an NFS ACL misconfiguration, or a race narrowing the mode)
      therefore skipped the mandated diagnostic entirely. Fix: move the presence check inside
      the same `try`. Red: `test_permission_error_from_the_presence_check_is_reported_cleanly`
      failed on `assert []`. Injected at `os.stat` for one specific path, so the test survives
      any later restructuring of how presence is determined — as 7.3 immediately proved.
- [x] 7.2 **A failed forward left behind an output directory it had created.** `mkdir` runs
      before the temp/replace sequence, and the failure path only unlinked the temp — so a
      failure left a newly-created, empty `output_dir` standing even though the batch raised.
      A state that could not occur before this hop existed, and orchestration reading "no
      output_dir" as "predict never ran" would misread it. Fix: record which directories this
      call creates (innermost first) and remove them on failure, stopping at the first that is
      not ours. Two tests, and note which is which: the "removes" one is a true red (it failed
      against the shipped code); the "keeps a pre-existing directory" one is a guard, and it
      deliberately uses a pre-existing **empty** directory — a non-empty one is protected by
      `rmdir`'s own semantics and would survive even an implementation that removed the
      destination unconditionally, proving nothing. Both fail against that mutant.
      Two pre-existing tests asserted `_names(out) == set()`; the guarantee is now stronger
      (`not out.exists()`), so they were updated rather than left asserting the weaker thing.
- [x] 7.3 **The double read became a single read** — Open Question 1, resolved rather than
      carried into the archive. `run_batch` takes one snapshot and passes it to both
      `discover_scans` and the forward-copy, so the bytes validated are the bytes published.
      Both directions of the race are now covered by tests that were genuinely red:
      the source **growing** between the two reads (forwarded a wider scope than predict
      predicted → spurious `result.failed` downstream, exit `3`, burned retries) and the source
      **disappearing** (forward silently became a no-op → #39 recurring with a green step and
      nothing above DEBUG). The second direction is the one that changed the call: §Decision 6
      had argued the deferral on the grow direction being loud, and the disappear direction is
      not loud.
      Shape: `discover_scans` and `copy_run_manifest_forward` each gain a keyword-only argument
      defaulting to a `_UNREAD` sentinel meaning "read it yourself", so **no existing caller or
      signature changes** — the objection that deferred this originally. The snapshot carries
      the source's mode alongside its bytes, which keeps "permissions match the source" true
      even when the source is gone by publish time, without a second stat that could observe a
      different file. `shutil.copyfile`/`copymode` give way to writing through the fd `mkstemp`
      already opened plus an explicit `chmod`; the `os.close`-before-replace constraint is
      preserved (the `with os.fdopen(...)` closes before the replace).
      Two incidental fixes fall out: `discover_scans` now reads the manifest as **bytes**, so
      the Windows leg no longer decodes it under the platform locale while the sibling stages
      read the same bytes as UTF-8; and the `tests/assets/scans/` landmine narrows, since the
      forward no longer calls `shutil.copyfile` at all (the comment at `batch.py` is corrected
      rather than deleted — the global `os.replace` patch still collides).
- [x] 7.4 Spec delta updated: the "structurally impossible to forward a corrupt manifest"
      clause no longer depends on the source being unchanged between two reads (it was true
      only given an atomic upstream writer, which §Decision 6 already contradicted); a
      single-read requirement and three scenarios added (created-directory cleanup, the
      presence/identity arm of the clean-reporting scenario, and a source changed after
      discovery). `design.md`'s Decision 6 and Open Question 1 both record the reversal and why.
- [x] 7.5 Full local gate: `black`, `ruff`, `codespell`, the CPU suite under ci.yml's exact
      marker expression (365 passed, 2 skipped), and `openspec validate --strict`.
- [x] 7.6 **`pytest -m gpu` genuinely executed — 3 passed**, including
      `test_predict_on_video_runs_on_cuda` (real CUDA inference) on an RTX A5000.
      **Recording a trap, because this box was ticked twice while the subset was
      vacuous:** a worktree synced with the default/CPU extra installs `torch==…+cpu`, so
      `torch.cuda.is_available()` is `False` and all three GPU tests **skip** — reporting
      `3 skipped` and a green run that proves nothing about GPU behaviour, on a machine that
      has a GPU. `openspec/project.md` calls this subset a required local `/pre-merge` step,
      so the gate has to be run as it documents: `uv sync --extra dev --extra windows_cuda`
      first, then `uv run pytest -m gpu`. Confirm `torch.version.cuda` is not `None` before
      believing the result. The full CPU suite was then re-run under the CUDA build too
      (`2.11.0+cu128`) and is unchanged at 365 passed — worth doing, since CI only ever
      exercises the CPU wheels. Neither result is evidence *for this change* (it touches no
      inference or device code), but "the required gate actually ran" and "the required gate
      silently skipped" are different claims and were being conflated.

**Raised by the second review and deliberately not acted on** — each checked against the code
rather than argued from the diff:

- **"Widening the CLI catch to `OSError` is a log-triage regression, because a wandb failure
  during `WarmModelWorker`'s eager model resolution would now log as a staging error."** The
  premise is false: there is no eager resolution. `WarmModelWorker.__init__` does no network I/O
  (its own docstring says so — wandb access is deferred to first use), and every registry call
  (`worker.resolve`, `worker.predict`) sits inside the per-scan `except Exception`, so it is
  isolated as a per-scan failure and cannot reach `main()`'s handler. The widened catch is
  *exactly* the staging set. What is actually wrong is the code comment claiming otherwise,
  already noted in §6's parked list. (Five of eight passes converging on this is agreement, not
  evidence — they shared the comment's premise.)
- **"A copy failure racing an in-flight SIGTERM should exit `143`, not `1`."** Exit `1` is
  normative in this change's own spec, with the rationale stated there: the exception is raised
  before the driver's stop-requested check, this is how every pre-flight staging error already
  behaves, and `1` is retryable in Argo just as `143` is. The finding's premise — that a retry
  policy keyed on `143` vs `1` would misclassify — has no consumer: the deployed templates are
  `retryPolicy: Always` with no exit-code discrimination, which is what
  `sleap-roots-pipeline#56` is open to add. §6.3's test pins the specified behaviour.

Also raised and already tracked, not re-litigated: the atomic-copy duplication (issue #43, filed
from §6); `pipeline_run_id` absent from the logs and the rollback instruction living in a
document that is archived on merge (§6 parked list); redundant stats per call; and the question
of whether `_is_same_file` earns its keep — it does, because the spec requires the no-op to
"leave that file intact", and a self-copy through a temp file republishes the file with a new
inode and mtime, which §6.2's hard-link and mtime assertions now pin.

---

## Post-merge handoff (not an archive gate)

Deliberately **checkbox-free**: these steps cannot complete until days after merge, whereas
`/cleanup-merged` runs minutes after merge and requires zero unchecked tasks. This repo has
already been burned by tracking post-merge cluster work as checkboxes —
`2026-07-05-add-predict-output-contract` was archived with an unchecked post-merge section and a
later commit had to reach into the archive to flip it. Tracked in the issue recorded at 5.4:

1. **Sequence first:** land PR `sleap-roots-pipeline#57` (the trait-extraction pin bump; `#54`
   and `#55` are still open) and apply it in-cluster. `sleap-roots-pipeline` main currently pins
   trait-extraction to an image predating `sleap-roots#263`, against which this fix is a no-op —
   a green run before that would prove nothing.
2. Then bump `sleap-roots-predictor-template.yaml`'s pin, verifying the tag against that repo's
   `docker-build.yml` run history rather than assuming one, and apply in-cluster
   (`argo template update`, runai-busch-lab — note prod and staging share that namespace).
3. Re-run the batch-oracle scenario against `A4-PIPELINE-E2E-TEST` (`experiment_id 12880747`,
   staging) with an unrelated scan (e.g. `scan_1009`) in the shared directory. Confirm
   `predictions/run_manifest.json` exists and matches `images_input/run_manifest.json`, and that
   the unrelated scan's trait output timestamp stays frozen. **The frozen timestamp is the
   pass/fail signal — not "the code changed".** Three preconditions must hold for that signal to
   mean anything: (a) the scan is genuinely absent from the *unioned* source manifest (bloomctl
   unions `scan_keys` across every invocation into a fixed shared `images_input/`); (b) traits'
   skip-if-done leaves an unchanged scan's `result.json` mtime untouched rather than rewriting an
   identical envelope — verified for predict, only assumed for traits; and (c) the run does not
   also bump the traits image, since `traits_code_sha` changing invalidates skip-if-done and
   would rewrite envelopes regardless of scoping.
4. Corroborating signal: trait-extraction logs a warning naming pre-existing result files
   "outside this run's scope (from a prior run's wider manifest)". Capture it — the expected
   outcome ("things stopped being reprocessed") otherwise looks like a regression to an operator.
5. If rolling back: delete `predictions/run_manifest.json` as well as reverting the image. A
   stale forwarded manifest scopes every later traits run to a frozen `scan_keys` set — silent
   under-processing, the mirror image of #39. (Predict itself is unaffected; it reads only from
   `images_input/`.)
6. Update `docs/bloom-integration/roadmap.md` in `sleap-roots-pipeline` with what was actually
   observed. Note there is currently **no 2026-09-10 entry** — the re-test is recorded only in
   `predict#39` and a `pipeline#37` comment — so this adds one rather than amending.
