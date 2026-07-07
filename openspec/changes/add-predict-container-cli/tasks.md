# Tasks

TDD / "check-first": each task states the verification (the assertion that must pass) before
the artifact that satisfies it. Real inference, no mocks — the offline CI gate drives
`run_batch` with an injected `LocalCardSource` over vendored models; the container run is a
manual pre-merge gate (mirrors talmolab/sleap-roots#256).

## 1. Committed fixture input scan directory

- [ ] 1.1 **Check first:** a test helper/fixture points at a committed input scan dir
  `tests/assets/scans/{scan_key}/` and fails until it exists.
- [ ] 1.2 Create the fixture: copy the existing `tests/assets/images/centered_pair/*.png`
  frames into `tests/assets/scans/{scan_key}/` and hand-author
  `{scan_key}.scan_metadata.json` (`scan_key`, `image_ids`, `images_checksum`,
  `params={"species":"rice","mode":"cylinder","age":3}`) matching the vendored
  `all_roots_source`/`rice_source` models used by the #16 tests.

## 2. Packaging + CLI skeleton

- [ ] 2.1 **Check first:** a `tomllib` guard test asserts `[project.scripts]` declares
  `sleap-roots-predict = "sleap_roots_predict.__main__:main"` (RED).
- [ ] 2.2 Add the `[project.scripts]` entry; confirm `uv sync --frozen` still passes (console
  scripts don't change deps); re-lock only if uv requires it.
- [ ] 2.3 **Check first:** a CI-safe test asserts `python -m sleap_roots_predict --help`
  parses two positional args (`input_dir`, `output_dir`) and exits 0 without touching the
  network (RED).
- [ ] 2.4 Create `sleap_roots_predict/__main__.py`: `argparse` (`input_dir`, `output_dir`),
  `main(argv=None) -> int` calling `run_batch` and returning `0 if result.ok else 1`;
  `if __name__ == "__main__": sys.exit(main())`.

## 3. Scan discovery + params (`batch.py`)

- [ ] 3.1 **Check first:** `discover_scans(fixture_dir)` returns the scan with `scan_key`, its
  frame paths, and a `ResolvedParams(species=rice, mode=cylinder, age=3)` (RED).
- [ ] 3.2 Implement `discover_scans`: `rglob "*.scan_metadata.json"`; `scan_key` = stem;
  frames = sibling images (`png/tif/tiff/jpg/jpeg`, natural-sorted); `ResolvedParams` from
  the sidecar's `params`; minimal `json.load` (no `trait_extractor` import).
- [ ] 3.3 **Check first + implement:** stem ≠ sidecar `scan_key` → that scan errors; duplicate
  `scan_key` across the tree → raises. (Add a `test_batch_does_not_import_trait_extractor`
  guard, mirroring the #16 no-`sleap-roots`-import test.)

## 4. `run_batch` happy path: output + sidecar pass-through

- [ ] 4.1 **Check first (real inference):** `run_batch(fixture_in, tmp_out,
  source=all_roots_source)` writes `tmp_out/{scan_key}/{scan_key}.predictions.json` (validates
  as a `PredictionManifest`) + one `.slp` per resolved root type, and a
  `{scan_key}.scan_metadata.json` byte-identical to the input sidecar; with
  `SRP_PREDICT_CODE_SHA` set, the manifest records it (RED).
- [ ] 4.2 Implement `run_batch(input_dir, output_dir, *, source=None, peak_threshold=0.2,
  batch_size=4, predict_code_sha=None, predict_container_digest=None)`: one
  `WarmModelWorker(source=source)`; per scan build the video (`greyscale=True`), `resolve` +
  `predict`, `write_prediction_outputs(... out_dir=output_dir/{scan_key}, ...)`, then copy the
  sidecar into `output_dir/{scan_key}/`. Return a `BatchResult` (per-scan statuses).
- [ ] 4.3 **Check first:** a two-scan batch resolving to the same model reuses the resident
  `Predictor` (object identity) — load-once (RED → green).
- [ ] 4.4 Export `run_batch` from `sleap_roots_predict/__init__.py`; update `test_public_api`.

## 5. Resume (skip-if-exists)

- [ ] 5.1 **Check first:** a second `run_batch` over the same `output_dir` skips a scan whose
  `{scan_key}.predictions.json` exists (no re-predict — e.g. asserted via unchanged mtime or a
  status of `skipped`) while still predicting a newly-added scan (RED).
- [ ] 5.2 Implement the skip check before per-scan inference.

## 6. Failure isolation + exit codes

- [ ] 6.1 **Check first:** a batch with one broken scan (unreadable frames) → `result.ok is
  False`, the good scan's outputs still written; an empty input dir → `result.ok is True`,
  nothing written (RED).
- [ ] 6.2 Implement per-scan `try/except` isolation + `BatchResult` + empty-input warn.
- [ ] 6.3 **Check first:** `main([in, out])` returns `0` when all scans pass and `1` when any
  fails (RED → green). Add a `@pytest.mark.wandb` subprocess test
  (`python -m sleap_roots_predict` over the real registry) for end-to-end wiring.

## 7. Dockerfile (real entrypoint + GPU)

- [ ] 7.1 **Check first (manual gate, documented):** `docker build --build-arg
  SRP_PREDICT_CODE_SHA=deadbeef .`; `docker inspect` shows exec-form
  `ENTRYPOINT ["python","-m","sleap_roots_predict"]`; `docker run <image> /in /out` over the
  fixture emits `{scan_key}.predictions.json` with `predict_code_sha == "deadbeef"` and exits 0.
- [ ] 7.2 Replace the REPL stub: exec-form ENTRYPOINT; `uv sync --frozen --no-dev --extra
  linux_cuda --python 3.12`; `ARG SRP_PREDICT_CODE_SHA=""` → `ENV` after the heavy layers;
  keep `MPLBACKEND=Agg` + apt libs.

## 8. CI workflow (bake the sha)

- [ ] 8.1 **Check first:** a guard test (or documented check) asserts `docker-build.yml`
  passes `build-args: SRP_PREDICT_CODE_SHA=${{ github.sha }}` and still emits a `sha-<sha>`
  tag.
- [ ] 8.2 Add the `build-args` block to the build step (triggers/tags otherwise unchanged).

## 9. Docs + validation

- [ ] 9.1 Update `openspec/project.md` (container extra `cpu`→`linux_cuda`; roadmap note
  "serving protocol/CLI" now landed) and `README.md` Configuration if needed.
- [ ] 9.2 Run `black`, `ruff check`, `codespell`; `pytest` (CPU suite green); the `gpu` subset
  + the docker-run gate via `/pre-merge`.
- [ ] 9.3 `openspec validate add-predict-container-cli --strict` passes.
