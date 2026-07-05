# Tasks — add-predict-output-contract

TDD throughout: write the failing test first (red), implement the minimum to pass
(green), then refactor. Run `/lint` + `/test` after each task. No mocks — drive the real
warm worker over the vendored native + legacy models.

## 1. Test-only dependency + environment sanity

- [ ] 1.1 Add `sleap-roots` to the `dev` extra in `pyproject.toml`. **Verify first:** run
  `uv sync --extra dev --extra cpu` and confirm it resolves with `sleap-io==0.8.x`
  (no downgrade of the sleap-nn pin). If it conflicts, stop and fall back to
  `importorskip` + a documented compatible-env note (do NOT loosen the inference pin).
  - *Test:* `uv run python -c "import sleap_roots, sleap_io; print(sleap_io.__version__)"`
    imports both and prints an `0.8.x` version.

## 2. Schema models (`PredictionArtifact`, `PredictionManifest`)

- [ ] 2.1 Add `sleap_roots_predict/output_contract.py` with the two pydantic models,
  reusing `ModelRef` from `sleap-roots-contracts`.
  - *Test (red first):* `tests/test_output_contract.py::test_manifest_round_trips`
    constructs a `PredictionManifest` (with a real `ModelRef`), dumps to JSON, reloads,
    and asserts equality — verifies field set, `schema_version` default, and `ModelRef`
    round-trip.
  - *Test:* `test_plant_qr_code_defaults_to_scan_key` verifies the default.

## 3. `model_id` slug + `scan_key` validation helpers

- [ ] 3.1 Implement `slugify_model_id(ref)` (non-`[A-Za-z0-9-]` → `-`, dot/slash-free) and
  `scan_key` filename-safety validation.
  - *Test (red first):* `test_model_id_slug_is_filename_safe` asserts a `registry_id`
    with `/` and `.` produces a slug with neither; `test_rejects_unsafe_scan_key`
    (parametrized over `.`, `/`, `\`, empty) asserts `ValueError`.

## 4. Pure writer `write_prediction_outputs`

- [ ] 4.1 Implement the writer: validate `set(labels_by_root) == set(refs_by_root)`;
  create `out_dir`; per root write `{scan_key}.model{slug}.root{root_type}.slp`, compute
  sha256 + size; build + write `{scan_key}.predictions.json`; return the manifest.
  - *Test (red first):* `test_writer_writes_named_slp_and_manifest` drives the warm worker
    (`rice_source` + `video` fixtures) to real labels + refs, calls the writer, and
    asserts the named `.slp` reload via `sio.load_file` with frames and the manifest maps
    root → path/model_id/`ModelRef`.
  - *Test:* `test_checksums_and_sizes_match_files` recomputes sha256/size and compares.
  - *Test:* `test_mismatched_labels_and_refs_raise` asserts `ValueError`.
  - *Test:* `test_zero_roots_writes_empty_artifacts` asserts `artifacts == []` and the JSON
    is still written.

## 5. Fail-soft build identity

- [ ] 5.1 Implement the explicit-arg → env → `""` precedence for `predict_code_sha` /
  `predict_container_digest`.
  - *Test (red first):* `test_build_identity_explicit_arg_wins`,
    `test_build_identity_env_fallback` (uses `monkeypatch.setenv`), and
    `test_build_identity_absent_is_empty_string` (asserts `""`, no raise).

## 6. `ScanRequest` + batch `predict_and_write_batch`

- [ ] 6.1 Add the `ScanRequest` dataclass and `predict_and_write_batch` (one warm worker,
  one subdir per scan, `resolve`+`predict`+`write` per scan, return `list[...]`).
  - *Test (red first):* `test_batch_writes_per_scan_subdirs` runs two scans and asserts
    `out_dir/{scan_key}/` exists per scan with its `.slp` + `.predictions.json`.
  - *Test:* `test_batch_reuses_resident_predictors` asserts the second scan reuses the
    first scan's `Predictor` instance by object identity (warmth).

## 7. Downstream acceptance — `sleap_roots.Series.load`

- [ ] 7.1 Add the real acceptance test loading the writer's output via `Series.load`.
  - *Test (red first):* `test_output_loads_via_sleap_roots_series` uses
    `pytest.importorskip("sleap_roots")`, writes a scan's artifacts, calls
    `Series.load(series_name=scan_key, primary_path=…, lateral_path=…, crown_path=…)`, and
    asserts it loads without error and the per-root labels are populated.

## 8. Public API + docs

- [ ] 8.1 Export `PredictionArtifact`, `PredictionManifest`, `ScanRequest`,
  `write_prediction_outputs`, `predict_and_write_batch` from
  `sleap_roots_predict/__init__.py`.
  - *Test (red first):* extend `tests/test_public_api.py` to assert the new names import
    from the package root.
- [ ] 8.2 Update `CLAUDE.md` (module list + processing flow) and `README` if applicable to
  describe the output contract.

## 9. Gate

- [ ] 9.1 `/lint` clean (black, ruff `D`, codespell) and `/test` green (default markers).
- [ ] 9.2 Run the acceptance/`Series.load` test locally in a `dev` env with `sleap-roots`
  installed to confirm the real downstream load passes.
- [ ] 9.3 Mark all `tasks.md` items complete; ready `/pre-merge`.

## Post-merge (required, tracked separately)

- [ ] P.1 Update `sleap-roots-pipeline/docs/bloom-integration/roadmap.md` (A4 DAG): predict
  output contract done → unblocks A3-traits input. Note the predict-local schema and its
  future promotion to `sleap-roots-contracts`.
