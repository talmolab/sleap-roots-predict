## ADDED Requirements

### Requirement: Run-manifest forward-copy to the output directory

The runner SHALL forward a present `run_manifest.json` from the input directory to the output
directory. Predict's output directory is the downstream trait-extraction stage's input
directory, and that stage scopes its own discovery by reading `run_manifest.json` from *its*
input directory — so absent this hop the manifest never reaches it and that stage silently falls
back to unscoped discovery. The *read* side of this file — scoping predict's own discovery to
its `scan_keys` — is specified by "Scan discovery and params from the scan-metadata sidecar";
this requirement covers only the forward hop.

Specifically: after discovering scans and before predicting any of them, `run_batch` SHALL copy
a `run_manifest.json` (`RUN_MANIFEST_FILENAME`, `sleap-roots-contracts`) present directly under
the input directory to the top level of the output directory, creating the output directory if
it does not exist. When no manifest is present the forward-copy SHALL create nothing, leaving
output-directory creation to the existing per-scan write path.

The copy SHALL be a **raw byte copy**, not a re-serialization through the `RunManifest` model,
so the forwarded file is byte-identical to what the upstream producer wrote. (This clause is
scoped to the forward-as-copy design: if this hop is later replaced by a union-merge — which
must re-serialize by construction — that change repeals this sentence rather than violating it.) The copy SHALL
perform **no validation** of the manifest's contents: within `run_batch` an invalid manifest has
already been rejected by discovery, and as a standalone public function the copy is
content-agnostic by design.

Within `run_batch` the manifest SHALL be read **exactly once** per batch, and the bytes
discovery validates SHALL be the bytes the forward-copy publishes. This makes forwarding a
corrupt manifest structurally impossible rather than merely unlikely: with two independent
reads the guarantee held only while the source was unchanged between them, and the upstream
producer writes into a directory shared across invocations. A source that is modified or
removed after discovery SHALL NOT change what is forwarded — in particular the forward SHALL
NOT silently become a no-op because the source has since disappeared, which would return the
downstream stage to the unscoped discovery this requirement exists to prevent.

The copy SHALL be performed **atomically** (written to a temporary file **in the same
directory**, then moved into place via `os.replace`), so no reader can ever observe a
partially-written `run_manifest.json` at the final path, and a copy that fails partway SHALL
leave no residue in the output directory. Same-directory placement is normative: a temporary
file elsewhere makes the move cross-device on the production mount, a failure no test
environment reproduces. Atomicity SHALL additionally hold **between concurrent invocations**
writing into the same shared output directory: no invocation's incomplete intermediate state
SHALL ever become visible at the destination path, including as a result of another
invocation's actions — which a temporary file whose name is shared between invocations cannot
satisfy. The temporary file's exact name is left to the implementation.

The forwarded file's permissions SHALL match the source manifest's, rather than being
restricted to the writing process — the downstream stage runs as a **different user** on shared
storage and must be able to read it.

If **no** `run_manifest.json` is present under the input directory, the forward-copy SHALL be a
no-op and no manifest SHALL be written to the output directory — preserving the unscoped
fallback for local, standalone, and test runs that stage no manifest. In that case, if the
output directory nonetheless **already holds** a `run_manifest.json`, the runner SHALL log a
warning naming it, and SHALL leave it in place. It is not predict's to delete — a concurrent
invocation may have just written it — but left silent it would scope the downstream stage to
some earlier run's `scan_keys`, under-processing this run's scans with no other signal. The
condition means the upstream producer staged no manifest where one was expected (most
plausibly a rolled-back or stale images-downloader image), so the warning is the only place
that misconfiguration becomes visible. If the source and
destination refer to the same file, the forward-copy SHALL be a no-op and SHALL leave that file
intact; detection SHALL NOT rely on path-string equality, and SHALL NOT fail when the
destination does not yet exist (the ordinary case). An existing `run_manifest.json` in the
output directory SHALL be replaced.

The copy SHALL occur **before** the per-scan prediction loop and before any model-source
interaction, so that it is performed even when the batch subsequently exits early via the
stop-requested (SIGTERM) path, and so that a failure is surfaced before any prediction work is
done. A failure to copy SHALL raise a batch-level error rather than being swallowed as
best-effort, since a silently missing forwarded manifest causes the downstream stage to fall
back to unscoped discovery — the contamination this requirement exists to prevent. Before
raising, the **forward-copy itself** SHALL log the source and destination directories, since not
every `OSError` carries a filename attribute and the CLI's own staging-error line reports only
the exception. Per the "Per-scan failure isolation and batch exit code" requirement, such a
failure additionally surfaces as exit `1` with the CLI's one-line staging-error log.

The forwarded manifest SHALL be copied **unchanged**: its `scan_keys` describe the scans the run
was *asked* to process, not the subset predict actually produced. A `scan_key` predict failed is
therefore still declared to the downstream stage, which reports it as an isolated failure of its
own.

`copy_run_manifest_forward` SHALL be exported from the package root, so the forward hop is
callable independently of `run_batch`.

#### Scenario: A present manifest is forwarded byte-identically

- **WHEN** `run_manifest.json` is present directly under the input directory, with on-disk bytes
  a `RunManifest` round-trip would not reproduce (non-canonical key order, extra whitespace, an
  undeclared extra field, and no trailing newline), and a batch runs
- **THEN** the file at the top level of the output directory is byte-identical to the input copy

#### Scenario: No manifest present writes no manifest

- **WHEN** the input directory contains sidecars but no `run_manifest.json`
- **THEN** no `run_manifest.json` is written to the output directory, the batch does not raise,
  and each discovered scan's outputs are written as usual

#### Scenario: A stale output manifest with no input manifest is kept, but warned about

- **WHEN** the input directory stages no `run_manifest.json` but the output directory already
  holds one from an earlier run
- **THEN** that file is left exactly as it is, and a warning naming it is logged — so the
  downstream stage's scoping to an earlier run's `scan_keys` is visible rather than silent

#### Scenario: The manifest is forwarded even when the batch stops early

- **WHEN** a manifest is present and a stop is requested before the first scan is predicted, so
  the batch exits its loop having predicted nothing
- **THEN** `run_manifest.json` is nonetheless present in the output directory, so the downstream
  stage remains scoped for the partial (here, empty) batch

#### Scenario: The output directory is created when the copy is the first writer

- **WHEN** a manifest is present, the output directory does not yet exist, and every discovered
  scan fails
- **THEN** the output directory is created and contains the forwarded `run_manifest.json`, even
  though no per-scan output directory was ever written

#### Scenario: A successful forward leaves the output directory otherwise unchanged

- **WHEN** a manifest is forwarded into an output directory that already holds prediction
  outputs for scans not in this run's manifest
- **THEN** the directory's exact contents are what they were plus `run_manifest.json` — those
  scan directories are untouched and no additional file of any name remains

#### Scenario: Concurrent invocations never publish each other's partial state

- **WHEN** two invocations forward manifests into the same output directory
- **THEN** each invocation's intermediate file is at a path private to that invocation, so no
  invocation's move can publish another's incomplete bytes, and neither leaves residue when its
  own move fails *(verified via per-writer temp-path uniqueness rather than a live race — a
  scheduled interleaving can only be made to fail by hand-injecting the corruption it claims to
  detect)*

#### Scenario: The intermediate file is created beside its destination

- **WHEN** a manifest is forwarded into an output directory on a different filesystem from the
  system temporary directory
- **THEN** the forward still succeeds — the intermediate file is created in the output directory
  itself, so publishing it is never a cross-device move

#### Scenario: A manifest that fails validation is never forwarded

- **WHEN** the input directory holds a `run_manifest.json` that is not valid JSON, or that fails
  `RunManifest` validation
- **THEN** the batch raises during discovery and nothing is forwarded — no `run_manifest.json`
  and no output directory are created, so a corrupt manifest can never reach the shared tree

#### Scenario: The standalone copy validates nothing

- **WHEN** `copy_run_manifest_forward` is called directly on a source file whose bytes are not
  valid JSON
- **THEN** the bytes are forwarded unchanged and no validation error is raised — content
  validation belongs to discovery, not to the copy

#### Scenario: A stale manifest in the output directory is replaced

- **WHEN** the output directory already contains a `run_manifest.json` from a prior run, of a
  different byte length, and a batch runs with a different manifest staged in its input directory
- **THEN** the output directory's `run_manifest.json` is byte-identical to the current input
  manifest, with no fragment of the prior content surviving

#### Scenario: A failure after the bytes are written publishes nothing and leaves nothing

- **WHEN** the copy fails after the source bytes have been written but before the new manifest
  becomes visible at the destination
- **THEN** the destination holds no partial manifest (either nothing, or the complete prior
  manifest), the output directory gains no leftover file of any name, and the error propagates

#### Scenario: A failed forward leaves behind no output directory it created

- **WHEN** the forward-copy creates the output directory (or any of its parents) and a later
  step then fails
- **THEN** the directories this call created are removed again, innermost first, so the failure
  leaves no directory that did not exist beforehand
- **AND WHEN** the output directory already existed — including when it is empty
- **THEN** it is left in place, since it is not this call's to remove

#### Scenario: A failure before the output directory can be prepared is still reported cleanly

- **WHEN** the output directory cannot be created or written to at all (e.g. a file occupies its
  path, or its parent denies permission)
- **THEN** the error raised is the underlying filesystem error — not a secondary error from the
  cleanup path — and it is logged with both directories named before it propagates
- **AND WHEN** the failure comes from the presence or identity check rather than the write —
  for example the source manifest exists but cannot be stat'd — the same log SHALL still name
  both directories, since those checks are the first steps that can fail

#### Scenario: A source changed after discovery does not change what is forwarded

- **WHEN** the input manifest is rewritten with a wider `scan_keys` set, or removed entirely,
  after discovery has read and validated it but before the forward-copy runs
- **THEN** the manifest published to the output directory is byte-identical to the one
  discovery scoped this batch against, and the forward does not become a no-op

#### Scenario: A copy failure fails the batch before any prediction

- **WHEN** a manifest is present but copying it into the output directory fails (e.g. a
  permission or disk error)
- **THEN** the batch raises before constructing a `WarmModelWorker`, no scan is predicted, the
  forward-copy logs an error naming the source and destination directories, and the exception
  propagates through the CLI's staging-error handler

#### Scenario: Forwarded permissions match the source, not the writing process

- **WHEN** a manifest whose permissions are neither the writing process's default nor the
  ambient-umask default is forwarded on a POSIX filesystem
- **THEN** the forwarded file carries the source manifest's permissions exactly — so neither a
  private temporary file's mode nor a hardcoded mode can satisfy this

#### Scenario: Input and output the same directory is a no-op

- **WHEN** `run_batch` is invoked with an input directory and an output directory that refer to
  the same location by *different* path spellings, and a manifest is present
- **THEN** the batch does not raise, the manifest at that path is unchanged byte-for-byte, and
  the directory gains no additional file — identity is decided by what the paths refer to, not
  by comparing them as strings

#### Scenario: copy_run_manifest_forward is exported from the public API

- **WHEN** the package is imported
- **THEN** `copy_run_manifest_forward` is accessible from the package root and listed in
  `__all__`

## MODIFIED Requirements

### Requirement: Per-scan failure isolation and batch exit code

A scan whose processing fails SHALL be isolated: the runner records it with status `failed`,
continues the batch, and still produces outputs for the other scans. `run_batch` SHALL return a
`BatchResult` whose per-scan status is one of `ok` / `skipped` / `failed` and which reports
`ok` (batch-level) iff no scan failed. A scan that resolves to **zero** models across all root
types SHALL be treated as `failed` (rather than emitting an empty-artifacts manifest that the
downstream trait-extractor would reject).

The process SHALL exit with one of three driver-owned codes so an Argo step can distinguish an
isolated per-scan failure from a genuine crash:
- `0` — success: at least one scan was discovered and none failed.
- `3` — partial: at least one scan was discovered and the batch ran to completion, but one or
  more scans isolated-failed. The batch's own per-scan isolation already ran (the other scans'
  outputs are written); this exit code exists so Argo does not conflate "retry the whole batch"
  with "this is done, some scans need attention."
- *(Python's default, produced by an uncaught exception, not an explicit `return`)* `1` — every
  other failure: a pre-flight/staging error before any scan ran (a missing input directory, two
  sidecars sharing a `scan_key`, a `run_manifest.json` that fails to parse or validate — including
  an empty `scan_keys` list, rejected by the "Scan discovery" requirement's own manifest
  validation before discovery ever runs — **a failure to forward `run_manifest.json` to the
  output directory, per the "Run-manifest forward-copy" requirement** — or **zero scans
  discovered**: `discover_scans` returns an empty list because no sidecar exists anywhere under a
  present input directory), or a genuine pod-level crash (e.g. model-registry authentication
  failing before any scan is attempted). All are "the batch could not meaningfully run"
  conditions and are not split into separate codes; Argo's `retryStrategy` should retry any of
  them. The CLI SHALL log a clear one-line message before propagating any `OSError` or
  `ValueError` (which, since `FileNotFoundError` subclasses `OSError`, and `json.JSONDecodeError`
  and `pydantic.ValidationError` both subclass `ValueError`, covers all five staging-error cases
  above — missing directory, duplicate `scan_key`, malformed manifest, forward-copy failure, and
  zero-scans-discovered); any other exception type is not specially logged and surfaces Python's
  default traceback. Note `OSError` is deliberately a **superset** of the staging set: some
  pod-level crashes subclass it (a model-registry network failure, for instance, since
  `requests`' exception base subclasses `OSError`), so those are now labelled with the same
  one-line message. This over-capture is accepted — the exit code is `1` either way and the
  traceback still surfaces — in exchange for the staging errors that matter being logged cleanly.

Exit code `2` is deliberately NOT part of this convention: `argparse` already exits `2` on a CLI
usage error (missing/extra positional arguments), before `run_batch` ever runs. This matches the
identical convention adopted by the sibling `sleap-roots` trait-extractor driver
(`sleap-roots#259`) — both producers report numerically identical codes for numerically identical
situations, per A4's design doc §8 ask to resolve this "the same way for both."

A `run_manifest.json`-scoped batch where every listed `scan_key` has no matching sidecar is
**not** the zero-scans-discovered case: `discover_scans` still returns one (failed) entry per
listed key, so that batch ends `partial` (`3`), not the crash/staging-error code (`1`).

A forward-copy failure occurring while a stop has already been requested SHALL propagate rather
than being converted into the stop-requested exit code, giving exit `1` — the exception is
raised before the driver's stop-requested check is reached. This is how every pre-flight staging
error already behaves, and `1` is retryable in Argo just as `143` is.

#### Scenario: One failing scan does not abort the batch

- **WHEN** one scan in a multi-scan batch fails (e.g. its frames are unreadable or absent) and
  the others are valid
- **THEN** the valid scans' outputs are written, that scan's status is `failed`, and the
  process exits `3`

#### Scenario: A scan resolving to zero models is failed

- **WHEN** a scan's params match no model for any root type
- **THEN** the scan's status is `failed` (no empty-artifacts manifest is written for it), the
  batch continues, and the process exits `3`

#### Scenario: All scans succeed

- **WHEN** every discovered scan predicts successfully
- **THEN** the process exits `0`

#### Scenario: Empty input directory is a staging error

- **WHEN** a present-but-empty input directory contains no `*.scan_metadata.json`
- **THEN** `run_batch` raises before constructing a `WarmModelWorker`, writes nothing, and the CLI
  logs a clear message and exits `1`

Note: this is distinct from a `run_manifest.json` scoping discovery to zero `scan_keys`, which is
rejected earlier by the "Scan discovery" requirement's own manifest validation (an empty
`scan_keys` list fails `RunManifest` validation before `discover_scans` returns at all) — see the
exit-code bullet above, which already separates the two raise sites. Both land on exit `1` either
way; the mechanism differs, the outcome doesn't.

#### Scenario: Missing input directory is an error

- **WHEN** the input directory path does not exist
- **THEN** the runner raises, the CLI logs a clear message before the exception propagates, and
  the process exits `1`, rather than reporting success with no outputs

#### Scenario: A failed run-manifest forward-copy is a staging error

- **WHEN** a `run_manifest.json` is present but cannot be copied into the output directory
- **THEN** the CLI logs its clear one-line staging-error message before the exception
  propagates, giving the process exit `1` — not the raw-traceback path, and not `3`, since no
  scan was predicted

#### Scenario: A manifest scoped to missing sidecars ends partial, not a crash

- **WHEN** `run_manifest.json` lists one or more `scan_keys` with no matching sidecar anywhere
  under the input directory, and no other scans are discovered
- **THEN** `discover_scans` returns one failed entry per listed key (not an empty list), and the
  process exits `3`, not `1`

#### Scenario: A CLI usage error exits via argparse, before the driver runs

- **WHEN** `python -m sleap_roots_predict` is invoked with a missing required argument
- **THEN** the process exits `2` via `argparse`'s own pre-existing usage-error handling, before
  `run_batch` ever runs

### Requirement: Graceful SIGTERM handling for Argo preemption

The CLI SHALL install a `SIGTERM` handler before running the batch. `run_batch` SHALL accept an
optional keyword-only `should_stop: Callable[[], bool]` parameter (default a no-op returning
`False`, so existing callers are unaffected) and SHALL check it at the top of each iteration of
its per-scan loop, stopping before starting the next scan when it returns `True` — never
interrupting a scan already in progress (there is no safe interrupt point inside sleap-nn/GPU
inference). When the CLI's handler has fired, `main()` SHALL exit `143` (`128 + SIGTERM`)
regardless of what exit code the completed-so-far scans would otherwise produce, so the
container's reported exit code honestly reflects "asked to stop," distinct from a normal
success/partial/aborted outcome. That override governs the batch's own outcome: a pre-flight
staging error raising before any scan has run — including a failed run-manifest forward-copy —
propagates as that error rather than being converted to `143`, which is how every pre-flight
staging error already behaves.

Note: on Windows, `signal.signal(signal.SIGTERM, ...)` registers without error, but real
cross-process delivery (`os.kill`) invokes `TerminateProcess` rather than the registered handler
— unlike Linux/macOS. Tests SHALL exercise the handler by calling it directly, never via
`os.kill`, so this requirement is verifiable identically across this project's CI matrix.

#### Scenario: Stops at the next scan boundary, not mid-scan

- **WHEN** `should_stop` becomes `True` while the first of two scans is being predicted
- **THEN** the first scan's outputs are written completely and validly, and the second scan is
  not attempted

#### Scenario: SIGTERM exit code overrides the batch outcome

- **WHEN** the CLI's `SIGTERM` handler fires during a batch that would otherwise exit `0` or `3`
- **THEN** the process exits `143` instead

#### Scenario: A pre-flight staging error is not converted to the stop code

- **WHEN** the CLI's `SIGTERM` handler has fired and a pre-flight staging error (such as a failed
  run-manifest forward-copy) then raises before any scan has run
- **THEN** that error propagates rather than being reported as `143`

#### Scenario: No signal received leaves existing behavior unchanged

- **WHEN** the batch runs to completion without `SIGTERM` ever being received
- **THEN** the exit code is determined exactly as before (`0`/`3`/default `1`), unaffected by
  the new handler's presence
