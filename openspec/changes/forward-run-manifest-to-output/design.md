# Design: forward `run_manifest.json` to predict's output directory

## Context

Three producers write into one shared storage tree, each stage's output being the next stage's
input:

```
images_input/   <- bloomctl  batch-download-for-predict   (writes run_manifest.json)
predictions/    <- sleap-roots-predict  run_batch          (READS the manifest; never wrote it)
traits/         <- sleap-roots  extract_batch              (reads manifest from predictions/)
(DB)            <- bloomctl  batch-ingest-result           (reads manifest from traits/)
```

Verified wiring (`sleap-roots-pipeline.yaml:75-78` plus the two step templates): one hostPath
volume `predictions-output-dir` is mounted at `/workspace/output` for the predictor and at
`/workspace/input` for the trait-extractor. Predict's output directory *is* trait-extraction's
input directory.

**This gap disables scoping in three stages, not one.** Trait-extraction's own
`copy_run_manifest_forward` call sits *inside* its `if scope is not None:` block
(`trait_extractor/extractor.py:271,299-300`), so when no manifest reaches `predictions/`, none
is written to `traits/` either — and `bloomctl cyl batch-ingest-result`, which reads the
manifest from `traits/`, also falls back to unscoped discovery and writes stale envelopes into
Bloom's production tables. Fixing this one hop is what unblocks the whole chain.

The 2026-08-03 manifest-scoped-processing redesign
(`sleap-roots-pipeline/docs/superpowers/specs/2026-08-03-manifest-scoped-processing-redesign.md`)
deliberately kept every producer **filesystem-scoped over a shared directory**, coordinating via
a passed-along `run_manifest.json`, rather than isolating per-run output paths — because
per-run paths would break the pipeline's dedup / skip-if-done model, which depends on a stable
shared path.

That design only works if every hop forwards the manifest. Predict does not, which is issue #39.

### Evidence

First identified 2026-08-19 in a comment on `sleap-roots-pipeline#37` by reading
`sleap_roots_predict/batch.py`; re-confirmed live 2026-09-10 by a batch-oracle re-test. The
decisive observation, with the full record in `sleap-roots-predict#39`: an unrelated scan's
*predictions* were untouched while its *traits* output was rewritten in the same run, and
`predictions/run_manifest.json` did not exist on the shared mount. That isolates the fault to
this hop and explains why the sibling stale-pin fixes did not resolve it — trait-extraction's
own scoping (`sleap-roots#263`) cannot take effect while the manifest never reaches it.

**Sibling stale-pin status, verified rather than assumed** (it drives deployment ordering):
`#51` is closed; `#54` and `#55` are **open** and their fix (PR `#57`) is **unmerged**, with its
in-cluster apply step unchecked. The re-test's evidence is only consistent with the newer images
having been applied live ahead of the merge — this program's established "apply live, merge
later" pattern — but that is an inference; live cluster state was not verified from here.

## Decision 1: naive copy, not merge-with-lock

The forwarding hop has two plausible implementations, and the choice was made deliberately
rather than defaulted.

**What `bloomctl` actually does** (`salk-bloom/bloomcli/src/bloomctl/cyl/download_for_predict.py:445-508`,
`bloomcli/src/bloomctl/cyl/_locks.py`): it **unions** `scan_keys` into any existing manifest
(only `ok`/`skipped` scans contribute), takes `pipeline_run_id` as last-writer-wins, and guards
the read-merge-write with a hand-rolled `O_CREAT|O_EXCL` advisory lockfile (~190 lines, 900s
staleness reclaim). It needs the merge because the pipeline chunks one logical request across
multiple bloomctl invocations, each its own Argo workflow, all targeting the same fixed
`out_dir`, up to K concurrently — an overwriting write would drop earlier invocations' scans.

**The honest case against a naive copy here.** Predict inherits the same shared-directory
structure, so the concern transfers. A plain copy is non-monotonic: if invocation A reads the
source manifest at T1 and B reads the (strictly larger, since the source only grows by union)
manifest at T2, a late write by A can regress `predictions/run_manifest.json` and drop B's keys.

**Chosen: the naive atomic copy anyway**, for five reasons:

1. Concurrent chunked dispatch is possible (`bloom#677` phase 2 is merged) but **not exercised
   in production** — no frontend triggers it today.
2. Because the source manifest only ever grows, the overwhelmingly common interleaving is
   *over*-scoping (a later, larger snapshot wins), which costs redundant work that the existing
   idempotency-key skip-if-done absorbs. The lossy case needs a specific read-read-write-write
   ordering.
3. `_locks.py` lives inside `bloomctl`'s CLI with no shared home. Porting it means ~190 lines
   duplicated into predict, and its `O_CREAT|O_EXCL` primitive — the classic NFS-unsafe
   primitive on older NFS without the link-based workaround — has **never been reasoned about
   for NFS** anywhere in that repo. (Note: it is already *running* on NFS in production; the
   hostPath at `sleap-roots-pipeline.yaml:50-51` is explicitly "hostPath onto the /hpi/hpi_dev
   NFS". So the gap is that nobody analyzed it, not that it is untried.) Shipping a
   copy of an unanalyzed lock would trade a documented, rare race for an undocumented one.
4. It would not make the chain safe end-to-end regardless: trait-extractor's own
   `copy_run_manifest_forward` stays a naive overwrite, so the traits hop remains the same hole.
   Fixing one of two hops buys a false sense of safety.
5. The one genuinely *corrupting* concurrency failure available to this hop is not content
   regression but a **shared temp path** — and that is fixed here directly, without a lock, by
   giving each writer a unique temp name (below). Content regression degrades to redundant work;
   a truncated manifest does not.

**Follow-up filed**, not built here: promote `bloomctl`'s union-merge + lock into
`sleap-roots-contracts` and switch predict *and* traits onto it together, so all three hops are
concurrency-safe at once. That follow-up should also carry the two gaps this investigation
surfaced — `pipeline_run_id` being last-writer-wins by design, and the unvalidated NFS
behavior of `O_CREAT|O_EXCL`.

Consistency with trait-extractor's existing pattern is a real secondary benefit: today both
forwarding hops behave identically, which keeps the eventual consolidation a mechanical swap.

## The atomic-write mechanism

The spec fixes same-directory placement and `os.replace`, matching how the sibling sidecar-copy
requirement is written, and leaves the temp file's *name* open. This section is the normative
home for the rest. Each rule below was empirically reproduced on both a Windows and a Linux leg
during review; none of them is inferable from reading the documentation.

### The temp-file name must be unique per writer

Predict's existing atomic writes derive the temp path from the destination
(`dst.with_name(dst.name + ".tmp")`). For the per-scan sidecar that is safe, because the
destination is namespaced by `scan_key`. **The run manifest's destination is shared by
construction** across every invocation writing into the same output directory, so a fixed
`run_manifest.json.tmp` admits: A creates the temp and begins writing → B replaces A's
half-written temp into the final path. That publishes a truncated manifest — worse than any
content-regression outcome, since a truncated file fails `RunManifest` validation downstream and
is precisely what atomicity is supposed to prevent.

Fix: allocate the temp inside the output directory with a process-unique name, then replace. The
placement is load-bearing in the other direction too — a system-temp file would make the replace
cross-device (`EXDEV`) on the production mount, where the output tree is NFS and `/tmp` is the
container overlay.

`tempfile.mkstemp(dir=output_dir, ...)` provides this, but **three of its properties are traps**,
each empirically reproduced during review rather than reasoned about:

- **It returns an open file descriptor.** On Windows an open handle makes `os.replace` fail with
  `PermissionError (WinError 32)` — while `shutil.copyfile` to the same path *succeeds*, so the
  bug surfaces only at the replace and only on the Windows CI leg. `os.close(fd)` must come
  before the copy.
- **It creates the file at mode 0600 regardless of umask**, and `shutil.copyfile` does not alter
  an existing file's mode. Every other file predict writes lands at umask default. The
  trait-extraction container runs `runAsUser: 0` while the predictor runs as a non-root default
  user, on a shared NFS mount where `root_squash` is the default — so a 0600 manifest is the one
  file predict writes that the next stage **cannot read**, silently reinstating #39 with every
  test green. `shutil.copymode(source, tmp)` after the copy fixes it; a POSIX-only test pins it.
- **Its default name is `tmpXXXXXXXX`, with no `.tmp` suffix.** Residue assertions written as
  `glob("*.tmp")` would match nothing and pass against an implementation with no cleanup at all.
  Pass an explicit `suffix=".tmp"`, and assert on the directory's exact contents rather than a
  glob.

A dot-prefix (`prefix=f".{RUN_MANIFEST_FILENAME}."`) is also worth having: unique names mean a
`SIGKILL` between create and replace now orphans an unrecognizable file in the shared output
directory forever, where the old fixed name would have been overwritten by the next run. Hiding
it keeps it out of any future `run_manifest*` glob. This is a real, if minor, new cost of
choosing unique names, and it belongs on the record next to the benefit.

**Path identity cannot be tested with a bare `os.path.samefile`**: it stats *both* operands and
raises `FileNotFoundError` whenever the destination does not exist — the ordinary case on every
fresh run. It must be guarded (`except (FileNotFoundError, NotADirectoryError)`), catching those
two specifically rather than blanket `OSError`, so a `PermissionError` from `stat` still
propagates as the real staging error it is. Both exception types are needed, not one: a file
occupying a mid-path component of the destination raises `NotADirectoryError` on Linux but
`FileNotFoundError` on Windows.

**The cleanup block must pre-bind its temp path.** With `mkdir` and `mkstemp` inside the `try`,
a cleanup that unlinks `tmp` unconditionally raises `UnboundLocalError` when either fails —
and `UnboundLocalError` is a `NameError`, *not* an `OSError`, so the CLI's widened handler does
not catch it and the run takes the raw-traceback path the spec forbids. The masked failure is
the likeliest production one: a permission error on the output mount. Binding `tmp = None`
before the `try` fixes it. The alternative (hoisting `mkdir`/`mkstemp` out of the `try`) trades
the bug for a different violation — those failures would then emit no error log at all.

## Decision 2: copy before the loop, and raise on failure

**Placement — before the scan loop**, immediately after `discover_scans` and the empty-batch
guard, and before the `WarmModelWorker` is constructed:

- **It makes forwarding a corrupt manifest structurally impossible.** `discover_scans` parses and
  validates the manifest (`RunManifest.model_validate_json`) and raises on a malformed one, so
  placing the copy after it guarantees only a valid manifest is ever forwarded into the shared
  directory. This is the strongest argument for the placement, and it is why "copy it even
  earlier, so it survives discovery errors" would be a regression rather than a hardening.
- A failure surfaces before any GPU work and before any model-source interaction, so it forfeits
  nothing and the run is cleanly retryable. Placing it before the worker construction also
  preserves the existing "staging errors surface before any model-source interaction" invariant.
- **It minimizes the window between the two manifest reads** (see §Decision 6). `discover_scans`
  reads the manifest to scope the batch; the copy reads it again. Post-loop placement would
  widen that window from the duration of discovery to the entire inference run — hours — turning
  a rare interleaving into a routine one. This is the strongest of the three placement arguments
  after corrupt-manifest impossibility.

  Two arguments that do *not* survive scrutiny, recorded so they are not re-invented: a post-loop
  copy would **not** be skipped on the `should_stop` path (`batch.py:324`'s `break` falls through
  to `return result` at `:358`), and the `SIGKILL`-after-grace-period case buys nothing either,
  because the deployed template is `retryPolicy: Always` — a killed predictor step is retried in
  full and trait-extraction never runs on the killed attempt.
- Unlike the per-scan sidecar (which must land before `write_prediction_outputs` writes the
  per-scan manifest, because that manifest is predict's own resume commit-marker), the run
  manifest is not a commit marker for anything predict does. No ordering constraint pulls it
  later.

Two consequences worth naming. First, `run_batch` does not create `output_dir` today — only
`_predict_one` does, per scan — so this copy becomes the first thing to create it. A batch where
every scan fails now leaves an `output_dir` containing only `run_manifest.json`, where
previously it might not have existed at all. Second, that state changes traits' retry semantics
for that case: with no manifest and no outputs traits raises a staging error (exit `1`,
retryable), whereas with a manifest present it produces one failed entry per `scan_key` and
exits `3`. That is the better classification — scoped and attributable rather than a silent
unscoped sweep. Two caveats, both easy to overstate: the deployed template is
`retryPolicy: Always` with no exit-code discrimination, so `1` and `3` are retried identically
today; and per §Decision 3 this case is currently **unreachable** anyway, since a predict run
that fails every scan or stops immediately fails its own step and the next stage never runs. It
is a latent change, not an observable one.

**Failure handling — raise, diverging from the reference.** Trait-extractor's `extract_batch`
catches `OSError` from its forward-copy and treats it as best-effort infrastructure. Predict
raises instead:

- A silently-skipped copy is not a degraded success; it is issue #39 recurring, undetected, with
  contamination downstream.
- Placed before the loop, raising costs zero prediction work — the usual argument for
  best-effort ("don't throw away a batch of good results over an infrastructure hiccup") does
  not apply.
- If `output_dir` cannot be written, the batch could not have written predictions there either.

This has a CLI consequence that must ship with it. `__main__.py` catches
`(FileNotFoundError, ValueError)` to emit the clean one-line staging-error log the exit-code
requirement mandates. A copy failure raises `PermissionError`/`OSError` — a *sibling* of
`FileNotFoundError` under `OSError`, not a subclass — so it would otherwise fall through to the
raw-traceback path. Widening the catch to `(OSError, ValueError)` preserves all existing
coverage (`FileNotFoundError ⊂ OSError`) and brings the new staging error under the same
contract. The exit code (`1`) was already correct; only the logging was wrong.

There is one acknowledged hole, not a regression: if the copy fails while a stop has already been
requested, the exception propagates before the driver's `stop_event` check, so the process exits
`1` rather than `143`. Every pre-flight staging error already behaves this way, and `1` is
retryable in Argo just as `143` is, so restructuring `main` to close it is not worth it.

A second, smaller divergence from the reference: the reference has **no** temp-file cleanup (a
bare `copyfile` then `replace`). Predict adds cleanup, matching its own `_predict_one` sidecar
copy. Both divergences belong in the module docstring so a future reader diffing the two modules
does not "harmonize" them back.

## Decision 3: forward unchanged (requested scope), not filtered to what predict delivered

`bloomctl`, the only stage that authors a manifest from scratch, deliberately filters:
`this_run_scan_keys = {s.scan_key for s in result.scans if s.status in ("ok", "skipped")}` —
its `scan_keys` mean *what this stage delivered and the next should therefore process*. Predict
forwarding the raw input list adopts the opposite meaning — *what this run was asked to
process* — and so declares `scan_key`s predict knows it produced nothing for. That is a
semantic inversion introduced mid-chain, and it was an unexamined default until review.

What the downstream stage actually does with such a key, traced rather than assumed
(`trait_extractor/extractor.py:271-280`): it appends one entry to `result.failed`, and the
driver returns `0 if result.ok else 3` (`trait_extractor/__main__.py:104`). So a raw-forwarded
manifest converts each of predict's failed scans into a *trait-extraction* failure, misattributed
to the wrong stage, and — since the deployed template is `retryPolicy: Always` — would burn its
retries and fail the step, blocking write-back for the scans that did succeed.

**That divergence is unreachable today via predict's own failures.** If predict fails any scan it
exits `3`, the predictor step fails after its own retries, and trait-extraction never runs. If
predict exits `0`, every key it *scoped to* has predictions. (It does not follow that every key
it *forwards* has predictions — the manifest is read twice and can grow in between; see
§Decision 6. That is a separate, bounded defect, not a reason to prefer filtering.)

**Chosen: forward unchanged**, because it is equivalent today and strictly simpler — it keeps
byte-fidelity (no re-serialization), and it keeps the copy pre-loop, which filtering would make
impossible since the filter needs the batch's results.

**Trip-wire — this decision has an expiry.** `sleap-roots-pipeline#56` proposes giving the
templates exit-code discrimination (`continueOn`/`retryStrategy` reacting to `3` = partial). The
moment trait-extraction can run after a partial predict, the two semantics diverge and filtered
(`ok ∪ skipped`) becomes correct: raw would then produce spurious, unfixable failures at both
the traits and write-back stages. Revisit this decision when #56 lands, not on a schedule.

## Decision 5: warn on a stale output manifest, do not delete it

The forward-copy no-ops when the input stages no manifest. That preserves the unscoped fallback
for local and standalone runs — but if the output directory already holds a manifest from an
earlier run, it survives and scopes the downstream stage to that run's `scan_keys`. Silent
under-processing: the mirror image of #39, and harder to spot, since over-processing at least
leaves fresh timestamps while under-processing leaves nothing at all.

**How reachable is it?** Argued from the *output* side, which is tighter than arguing from the
input side. A stale manifest in `output_dir` requires a prior successful forward, which requires
a prior *input* manifest — and bloomctl's manifest is union-only: never shrinks, never deleted.
So on fixed mounts the state requires the input manifest to have gone away while the output
mount persisted: a **rolled-back or stale images-downloader image predating manifest writing**
(that stage is the one place an old image silently stages *no* manifest rather than failing, and
stale pins are this program's most repeated failure — `#51`/`#54`/`#55`), the input manifest
being deleted by hand, or a standalone predict run pointed at the shared tree.

An earlier draft argued this from the input side — "bloomctl skips the write only on a fresh
`images_input`, where there are no sidecars either, so `run_batch` raises first". That inference
is wrong and is recorded here so it is not repeated: bloomctl's skip condition
(`download_for_predict.py:492`) tests the *manifest*, while `discover_scans` globs the whole tree
for *sidecars*. Sidecars predating the manifest era exist without a manifest — `scan_1009` is
exactly that — so discovery returns entries, `run_batch` does not raise, and the copy is reached.

**Chosen: log a warning, leave the file.** Deleting is unjustified — a concurrent invocation
sharing the output directory may have just written it, and predict cannot distinguish that from
a leftover. But silence is the wrong default precisely because the trigger is a stale pin, which
produces no other signal: the downstream stage would simply process fewer scans and report
success. The warning costs three lines and makes the one case where this bites greppable in pod
logs.

## Decision 6: the manifest is read twice, and the second read can differ

`discover_scans` reads and validates `input_dir/run_manifest.json` (`batch.py:117-119`);
`copy_run_manifest_forward` then independently re-reads the same path. There is no shared
buffer, so **predict scopes its work to snapshot v1 and forwards snapshot v2.**

This falsifies, as originally written, §Decision 3's claim that "if predict exits `0`, every
manifest key has predictions, so the raw and filtered sets are identical". That holds only if
the manifest does not change between the two reads. It can: `images_input/` is a fixed shared
hostPath and bloomctl unions into it under a lock from *any* invocation
(`download_for_predict.py:475-508`), so it does not require `bloom#677` chunked dispatch — two
overlapping ordinary submissions suffice. Predict then exits `0`, forwards a strictly larger
manifest, and trait-extraction appends one `result.failed` per extra key and exits `3`, burning
retries under `retryPolicy: Always` and blocking write-back for scans that genuinely succeeded.

**Not fixed in this change; the window is bounded and the fix has a cost.** Pre-loop placement
already holds the window to the duration of discovery rather than the duration of inference
(§Decision 2). Closing it properly means forwarding the bytes discovery already validated —
which is the better design, and would upgrade "structurally impossible to forward a corrupt
manifest" from an ordering property to an identity property — but it changes the shape of
`discover_scans`, an exported function, and does so at the end of four review rounds in which
revision has been the dominant source of new defects. Tracked as an Open Question and in the
5.3 follow-up rather than taken on here.

Note the failure mode is loud (a failed traits step), not silent corruption, which is why it is
tolerable to defer; it is a misattribution and a retry cost, not lost or wrong data.

## Testing strategy

The test plan lives in `tasks.md` §1.1/§2.1 and is not repeated here. Only the design-level
points belong in this document — the ones where a plausible-looking test proves nothing:

- **The byte-copy test needs an *unparsable* source.** Well-formed-but-non-canonical JSON does
  not discriminate reliably; `b"{not valid json"` does, because a re-serializing implementation
  raises where a byte copy succeeds.
- **The temp-uniqueness and temp-placement tests are the only things pinning Decision 1's reason
  5** and the same-directory rule. Every other test in the plan passes against both a shared
  fixed temp name and a system-temp file — and the latter's only symptom is `EXDEV` on the
  production NFS mount, which no CI runner reproduces.
- **Temp cleanup must be injected at `os.replace`, not `shutil.copyfile`.** Failing the copy
  means the temp was never created, so the assertion passes against an implementation with no
  cleanup at all.
- **`run_batch` must be tested against a stub model source wherever prediction is irrelevant** —
  it is the only real lever on runtime (model loading dominates; frame size does not), and it
  makes those tests stronger by proving the forward hop is independent of prediction.

## Risks / Trade-offs

- **Content regression under concurrency** (accepted, Decision 1): a late writer can overwrite a
  larger manifest with a smaller snapshot. Degrades to redundant work, absorbed by skip-if-done.
  The *corrupting* variant — publishing a truncated manifest — is fixed here via unique temp
  names.
- **Stale output manifest with no input manifest** (mitigated by a warning, see §Decision 5): a
  manifest left in the output directory from an earlier run would scope later runs to a frozen
  `scan_keys` set. Under-processing is harder to detect than #39's over-processing, which at
  least leaves fresh timestamps.
- **Orphaned temp files**: unique names mean a `SIGKILL` between create and replace leaves a
  file no later run reclaims, where the old fixed-name idiom self-healed. Dot-prefixed to keep
  it out of any future glob. Minor, but it is a real cost of the unique-name choice.
- **CLI `OSError` over-capture**: widening the handler labels some genuine crashes (registry
  network errors) as staging errors. Exit code unchanged; cosmetic, and accepted in exchange for
  the staging errors being logged cleanly.

## Migration Plan

Deployment ordering and rollback are operational, and are specified in `proposal.md`
(§Deployment notes) and `tasks.md` (§Post-merge handoff). The two constraints that must not be
lost: the trait-extraction pin bump lands and is applied **first** (this fix is inert against
the currently pinned image), and any rollback **also deletes** `predictions/run_manifest.json`.

## Open Questions

1. **Whether the double read should become a single read.** `discover_scans` reads and validates
   the manifest; the forward-copy independently re-reads it from disk. See §Decision 6.
2. **Whether "permissions match the source" should become "not more restrictive, and readable by
   the downstream user".** Equality is implementable and verified safe today only because
   bloomctl writes `0644`. If any future producer wrote `0600`, this spec would oblige predict
   to faithfully forward an unreadable file and silently reinstate #39. The stricter wording is
   more correct but needs a mode floor, which the current wording forbids.

