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
     `os.replace` fails → `unlink` fails too), and no test in this plan can catch it, since 1.1
     bans `chmod`. The bare `raise` still re-raises the original after the inner handler runs.
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

- [ ] 4.1 `API.md`: document `copy_run_manifest_forward`; extend the `run_batch` entry to cover
      forwarding. **Also fix pre-existing drift in that same block**: the documented `run_batch`
      signature lists `peak_threshold: float = 0.2` and `batch_size: int = 4`, neither of which
      exists in the real signature.
- [ ] 4.2 `README.md`: three edits — (a) the container prose enumerating what is written per
      scan, now incomplete; (b) the **Project Structure module tree** (`run_manifest.py`);
      (c) the **tests tree**, which lists every test file (`test_run_manifest.py`).
- [ ] 4.3 `CHANGELOG.md`: amend the existing `**Predict container CLI**` bullet under
      `## [Unreleased]` → `### Added` rather than adding a new bullet (the repo's pattern —
      `consume-run-manifest` was folded into that same bullet). `Added`, not `Fixed`: nothing has
      been released, so no published claim is being corrected. While there, fix the pre-existing
      garble in that bullet — the orphaned "and per scan," clause belongs with the skip-if-done
      sentence, not with manifest scoping.
- [ ] 4.4 `openspec/project.md`: add `run_manifest.py` to the authoritative module layout.
- [ ] 4.5 Commit docs.

## 5. Verification gate

- [ ] 5.1 `openspec validate forward-run-manifest-to-output --strict` passes.
- [ ] 5.2 Run the full local gate via `/pre-merge`. Use **ci.yml's exact marker expression**
      (`-m "not gpu and not acceptance and not wandb"`), which deliberately differs from
      `pyproject.toml`'s `addopts` (that one also excludes `parity`); a plain `-m "not gpu"`
      pulls in flaky wandb tests. ruff lints `sleap_roots_predict/` and `scripts/` only, not
      `tests/`; black covers all three.
- [ ] 5.3 File the follow-up issue: promote `bloomctl`'s union-merge + lockfile into
      `sleap-roots-contracts` and switch predict *and* `sleap-roots` traits onto it (traits'
      copy has neither temp cleanup nor a unique temp name, so it carries the same truncation
      hazard fixed here). Include: `pipeline_run_id` is last-writer-wins by design; the
      `O_CREAT|O_EXCL` lock is already running on NFS in production but was never analyzed for
      it. **Frame it as a trip-wire, not a someday**: when concurrent chunked dispatch is enabled
      in the frontend, the union-merge becomes a blocker. Verified no equivalent issue exists in
      `sleap-roots` today. Record the issue number here: ______
- [ ] 5.4 File the deploy/verify tracking issue described under "Post-merge handoff" below and
      record its number here: ______
- [ ] 5.5 Make the Decision 3 trip-wire real, since a note in a doc that gets archived is not a
      mechanism and the trigger fires in another repo. Comment on `sleap-roots-pipeline#56`
      (or add a task-list item to its body) recording that when exit-code discrimination lands
      and trait-extraction can run after a partial predict, predict's forwarded-manifest
      semantics must change from "requested scope" to "delivered scope" (`ok ∪ skipped`).
      Record the link here: ______
- [ ] 5.6 Open the PR referencing issue #39 and this change-id. State in the body: the
      naive-copy-vs-merge decision, the forward-unchanged semantics decision and its #56
      trip-wire, the sticky-manifest rollback hazard, and the deployment ordering constraint.
      Then present READY TO MERGE and stop — do not merge.

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
