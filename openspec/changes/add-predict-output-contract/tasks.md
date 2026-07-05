# Tasks — add-predict-output-contract

TDD throughout: write the failing test first (red), implement the minimum to pass
(green), then refactor. Run `/lint` + `/test` after each task. No mocks — drive the real
warm worker over the vendored native + legacy models. Lab convention: use `pathlib.Path`
for all path handling and emit path strings via `Path.as_posix()`.

## 1. Test-only dependency + lockfile

- [ ] 1.1 Add `sleap-roots` to the `dev` extra in `pyproject.toml` (decision: keep in
  `dev` and run the acceptance test in CI). **Verify first:** run
  `uv sync --extra dev --extra cpu` and confirm it resolves with `sleap-io==0.8.x` (no
  downgrade of the sleap-nn pin, torch unchanged). If it conflicts, stop and fall back to
  `importorskip` + a documented compatible-env note (do NOT loosen the inference pin).
  - *Test:* `uv run python -c "import sleap_roots, sleap_io; print(sleap_io.__version__)"`
    imports both and prints an `0.8.x` version.
- [ ] 1.2 Run `uv lock` and commit the updated `uv.lock`. **Review the diff:** confirm the
  only changes are the additive `dev`-gated entries (`sleap-roots`, `scikit-image`,
  `pywavelets`, `scikit-learn`) and that no `cpu`-closure pin moved — in particular
  `sleap-io` stays `0.8.0` and `torch`/`torchvision` are unchanged. Push commit 1
  (deps + lock) **alone first** so the CI matrix confirms 3-OS resolution before code is
  built on top.
  - *Verify:* CI green on ubuntu + windows + macos for the deps-only commit.

## 2. Schema models (`PredictionArtifact`, `PredictionManifest`)

- [ ] 2.1 Add `sleap_roots_predict/output_contract.py` with the two pydantic models,
  reusing `ModelRef` from `sleap-roots-contracts`.
  - *Test (red first):* `tests/test_output_contract.py::test_manifest_round_trips`
    constructs a `PredictionManifest` (with a real `ModelRef`), dumps to JSON, reloads via
    the model, and asserts equality — verifies field set, `schema_version` default, and
    `ModelRef` round-trip.
  - *Test:* `test_plant_qr_code_defaults_to_scan_key` verifies the default.

## 3. `model_id` slug + `scan_key` validation helpers

- [ ] 3.1 Implement `slugify_model_id(ref)` (non-`[A-Za-z0-9-]` → `-`, dot/slash-free) and
  `scan_key` filename-safety validation (reject empty or any of `. / \ : * ? " < > |` or a
  control character).
  - *Test (red first):* `test_model_id_slug_is_filename_safe` asserts a `registry_id`
    with `/` and `.` produces a slug with neither; `test_rejects_unsafe_scan_key`
    (parametrized over `.`, `/`, `\`, `:`, `*`, a control char, and empty) asserts
    `ValueError`; `test_accepts_normal_scan_key` asserts an alnum/`_`/`-` key is accepted.

## 4. Pure writer `write_prediction_outputs`

- [ ] 4.1 Implement the writer: validate `set(labels_by_root) == set(refs_by_root)` and
  `scan_key` safety; create `out_dir` (`Path`); per root write
  `{scan_key}.model{slug}.root{root_type}.slp`, compute sha256 + size; build + write
  `{scan_key}.predictions.json`; thread `predict_code_sha`/`predict_container_digest`
  through (see task 5); store `slp_path` as a `Path.as_posix()` basename; return the
  manifest. Re-runs overwrite in place.
  - *Test (red first):* `test_writer_writes_named_slp_and_manifest` drives the warm worker
    (`rice_source` + `video` fixtures) to real labels + refs, calls the writer, and
    asserts the named `.slp` reload via `sio.load_file` with frames and the manifest maps
    root → path/model_id/`ModelRef`.
  - *Test:* `test_slp_path_is_relocatable_basename` asserts each `slp_path` is not absolute,
    contains no separator, equals the bare `{scan_key}.model{slug}.root{rt}.slp`, and
    `manifest_dir / slp_path` exists.
  - *Test:* `test_writer_covers_all_root_types` uses a source with primary=native,
    lateral=legacy, crown=native (reuse) so the `root{crown}` filename + manifest branch is
    exercised (all three `RootType` literals).
  - *Test:* `test_manifest_json_on_disk_round_trips` reloads the written JSON file and
    validates it equals the returned manifest.
  - *Test:* `test_checksums_and_sizes_match_files` recomputes sha256/size and compares.
  - *Test:* `test_mismatched_labels_and_refs_raise` asserts `ValueError`.
  - *Test:* `test_zero_roots_writes_empty_artifacts` asserts `artifacts == []` and the JSON
    is still written.
  - *Test:* `test_rerun_overwrites_in_place` writes once, writes again into the same
    `out_dir`/`scan_key`, and asserts the reloaded manifest reflects the second run.
  - *Test:* `test_writer_does_not_import_sleap_roots` imports
    `sleap_roots_predict.output_contract` and asserts `"sleap_roots" not in sys.modules`
    (runtime-purity guard).
  - *Test:* `test_plant_qr_code_recorded_verbatim` asserts an explicit `plant_qr_code` is
    stored as-is (non-default path).

## 5. Fail-soft build identity

- [ ] 5.1 Implement the explicit-arg → env → `""` precedence for `predict_code_sha` /
  `predict_container_digest` (helper used by the writer in task 4).
  - *Test (red first):* `test_build_identity_explicit_arg_wins`,
    `test_build_identity_env_fallback` (uses `monkeypatch.setenv` on
    `SRP_PREDICT_CODE_SHA` / `SRP_PREDICT_CONTAINER_DIGEST`), and
    `test_build_identity_absent_is_empty_string` (asserts `""`, no raise).

## 6. `ScanRequest` + batch `predict_and_write_batch`

- [ ] 6.1 Add the `ScanRequest` dataclass and `predict_and_write_batch` (one warm worker,
  one subdir per scan `out_dir/{scan_key}/`, `resolve`+`predict`+`write` per scan using the
  worker's `inference_config()`/`output_params()`, return `list[PredictionManifest]`).
  - *Test (red first):* `test_batch_writes_per_scan_subdirs` runs two scans and asserts
    `out_dir/{scan_key}/` exists per scan with its `.slp` + `.predictions.json`.
  - *Test:* `test_batch_reuses_resident_predictors` asserts the second scan reuses the
    first scan's `Predictor` instance by object identity (warmth).
  - *Test:* `test_batch_respects_overrides` passes a `ScanRequest.overrides` mapping a root
    type to an explicit `ModelRef` and asserts that scan's manifest records it.

## 7. Downstream acceptance — `sleap_roots.Series.load` (runs in CI)

- [ ] 7.1 Add the real acceptance test loading the writer's output via `Series.load`.
  Spike it locally first to confirm `Series.load` accepts the vendored skeleton before
  finalizing assertions.
  - *Test (red first):* `test_output_loads_via_sleap_roots_series` uses
    `pytest.importorskip("sleap_roots")` (belt-and-suspenders; in CI `sleap-roots` is
    installed so it executes), writes a scan's artifacts, calls
    `Series.load(series_name=scan_key, primary_path=<as_posix>, lateral_path=<as_posix>,
    crown_path=None)`, and asserts it loads without error and `primary_labels` /
    `lateral_labels` are populated. (No crown model is vendored → `crown_path=None`.)

## 8. Public API + documentation

- [ ] 8.1 Export `PredictionArtifact`, `PredictionManifest`, `ScanRequest`,
  `write_prediction_outputs`, `predict_and_write_batch` from
  `sleap_roots_predict/__init__.py`.
  - *Test (red first):* extend `tests/test_public_api.py` to assert the new names import
    from the package root.
- [ ] 8.2 Documentation — explicit per-file checklist (single-source the artifact format in
  the spec / `output_contract.py` docstrings; do NOT re-copy the filename grammar or JSON
  field list into every doc):
  - `CLAUDE.md`: **out of scope for this change** — deferred to
    [talmolab/sleap-roots-predict#15](https://github.com/talmolab/sleap-roots-predict/issues/15)
    (CLAUDE.md drifts from the OpenSpec specs; the stale `predictions.csv`/"deferred" line
    + duplicated API list are handled holistically there, not patched piecemeal here).
  - `README.md`: add `output_contract.py` to the Project Structure tree (one-line format
    description only, referencing the spec/docstrings).
  - `API.md`: document the 5 new public exports (or state the section is intentionally
    partial and point to `__init__`).
  - `CHANGELOG.md`: add an `### Added` entry under `[Unreleased]` (module + 5 exports + 2
    env vars + test-only `sleap-roots` dev dep).
  - `openspec/project.md`: surgically edit the Roadmap note (lines ~15-18) — strike the
    "`predictions.csv` output contract + `.slp` naming" clause (done; it's `.json`), keep
    serving/CLI, full `Provenance`/`ResultEnvelope`, and the parity harness as pending.

## 9. Gate

- [ ] 9.1 `/lint` clean (black, ruff `D`, codespell), `/test` green (default markers), and
  `uv build` succeeds (sdist + wheel) — or confirm `/pre-merge` covers the build step.
- [ ] 9.2 Confirm the `Series.load` acceptance test passes in a `dev` env with
  `sleap-roots` installed (it runs in CI, and `/pre-merge` re-runs it locally).
- [ ] 9.3 Run GPU tests locally (`uv run pytest -m gpu`) per the `/pre-merge` gate; they
  skip cleanly with no accelerator.
- [ ] 9.4 Mark all `tasks.md` items complete; ready `/pre-merge`.

## Post-merge (required, tracked separately)

- [ ] P.1 Update `sleap-roots-pipeline/docs/bloom-integration/roadmap.md` (A4 DAG): predict
  output contract done → unblocks A3-traits input. Note the predict-local schema and its
  future promotion to `sleap-roots-contracts`.
