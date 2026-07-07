# Tasks

TDD / "check-first": each task states the verification (the assertion that must pass) before
the artifact that satisfies it. The RED test and its GREEN implementation land in the **same
commit** (red→green is a working-tree loop, never two commits). Real inference, no mocks — the
offline CI gate drives `run_batch` with an injected `LocalCardSource` over vendored models; the
container run is a manual pre-merge gate (mirrors talmolab/sleap-roots#256).

## 1. Test fixtures (shared)

- [ ] 1.1 Promote the `LocalCardSource` model fixtures to `tests/conftest.py`: `rice_source`
  and `all_roots_source` (and their `_card`/`_params` helpers) currently live module-locally in
  `tests/test_output_contract.py` / `tests/test_warm_worker.py` — move/share them so a new
  `tests/test_batch.py` can inject them. Keep the existing tests green.
- [ ] 1.2 Create the committed fixture input scan dir `tests/assets/scans/<scan_key>/` (pick a
  concrete literal `scan_key`, e.g. `scanCPTEST0`): copy the existing
  `tests/assets/images/centered_pair/*.png` frames in, and hand-author a co-located
  `<scan_key>.scan_metadata.json` whose internal `scan_key` equals the dir/stem, with
  `image_ids`, `images_checksum`, and `params={"species":"rice","mode":"cylinder","age":3}`
  (matching the vendored `all_roots_source` cards). Ensure it is codespell-clean.

## 2. Packaging + CLI skeleton

- [ ] 2.1 **Check first:** a `tomllib` guard test asserts `[project.scripts]` declares
  `sleap-roots-predict = "sleap_roots_predict.__main__:main"` (RED).
- [ ] 2.2 Add the `[project.scripts]` entry; confirm `uv sync --frozen` still passes (console
  scripts are not resolution inputs — no lock change expected); re-lock only if uv requires it.
- [ ] 2.3 **Check first:** a CI-safe test asserts `python -m sleap_roots_predict --help` parses
  two positional args (`input_dir`, `output_dir`) and exits 0 without touching the network (RED).
- [ ] 2.4 Create `sleap_roots_predict/__main__.py`: `argparse` (`input_dir`, `output_dir`),
  `main(argv=None) -> int` that imports `run_batch` **lazily inside `main`** (function-local, so
  the CLI is importable before `batch.py` exists) and returns `0 if result.ok else 1`;
  `if __name__ == "__main__": sys.exit(main())`.

## 3. Scan discovery + params (`batch.py`)

- [ ] 3.1 **Check first:** `discover_scans(fixture_dir)` returns the scan with `scan_key`, its
  frame paths (only image files; `.json`/non-image files excluded), and a
  `ResolvedParams(species=rice, mode=cylinder, age=3)` (RED).
- [ ] 3.2 Implement `discover_scans`: `rglob "*.scan_metadata.json"`; `scan_key` = stem; frames
  = co-located image files matched by a **case-folded** extension check
  (`png/tif/tiff/jpg/jpeg`), natural-sorted; `ResolvedParams` from the sidecar's `params`;
  minimal `json.load` (no `trait_extractor` import).
- [ ] 3.3 **Check first + implement (separate RED tests):** (a) non-image file present is
  ignored as a frame; (b) `test_batch_does_not_import_trait_extractor` guard (mirrors #16's
  no-`sleap-roots`-import test).
- [ ] 3.4 **Check first + implement (separate RED tests):** stem ≠ sidecar `scan_key` → that
  scan is `failed` (isolated); a sidecar missing `params` or a required field → that scan is
  `failed`; duplicate `scan_key` across the tree → `discover_scans` raises.

## 4. `run_batch` happy path: output + sidecar pass-through

- [ ] 4.1 **Check first (real inference):** `run_batch(fixture_in, tmp_out,
  source=all_roots_source)` writes `tmp_out/{scan_key}/{scan_key}.predictions.json` (validates
  as a `PredictionManifest`) + one `.slp` per resolved root type, and a
  `{scan_key}.scan_metadata.json` **byte-identical** to the input sidecar; with
  `SRP_PREDICT_CODE_SHA` set, the manifest records it (RED).
- [ ] 4.2 Implement `run_batch(input_dir, output_dir, *, source=None, peak_threshold=0.2,
  batch_size=4, predict_code_sha=None, predict_container_digest=None)`: one
  `WarmModelWorker(source=source)`; per scan build the video (`greyscale=True`), `resolve` +
  `predict`, `write_prediction_outputs(... out_dir=output_dir/{scan_key}, ...)`, then copy the
  sidecar via a **binary** copy (`shutil.copyfile`, not a text read/rewrite — avoids CRLF
  translation on Windows). Return a `BatchResult` (per-scan `ok`/`skipped`/`failed`).
- [ ] 4.3 **Check first:** a two-scan batch (separate dirs) writes **both** scans'
  `predictions.json` + `.slp`, and resolving to the same model constructs a single worker /
  reuses the resident `Predictor` (object identity) — asserts "predicts every scan" + load-once
  (RED → green).
- [ ] 4.4 **(atomic commit)** Export `run_batch` from `sleap_roots_predict/__init__.py` and
  update `tests/test_public_api.py` in the **same** commit as 4.2 (a `run_batch` import in
  `__init__` before `batch.py` exists fails collection → whole suite red).

## 5. Single-channel prediction input

- [ ] 5.1 **Check first:** assert the video `run_batch`/`discover_scans` builds is 1-channel
  (e.g. `video.shape[-1] == 1`) and inference completes without a channel-mismatch error (RED).
  Covers the "Single-channel prediction input" requirement explicitly.

## 6. Resume (skip-if-exists)

- [ ] 6.1 **Check first:** a second `run_batch` over the same `output_dir` skips a scan whose
  `{scan_key}.predictions.json` exists (status `skipped`, no re-predict — e.g. asserted via
  unchanged mtime) while still predicting a newly-added scan (RED).
- [ ] 6.2 Implement the skip check before per-scan inference.

## 7. Failure isolation + exit codes

- [ ] 7.1 **Check first:** a batch with one broken scan (unreadable/absent frames) →
  `result.ok is False`, the good scan's outputs still written; a scan resolving to zero models
  → `failed`; an empty input dir → `result.ok is True`, nothing written (RED).
- [ ] 7.2 Implement per-scan `try/except` isolation + `BatchResult` + zero-model→failed +
  empty-input warn.
- [ ] 7.3 **Check first:** `main([in, out])` returns `0` when all scans pass and `1` when any
  fails (RED → green). Add a `@pytest.mark.wandb` subprocess test
  (`python -m sleap_roots_predict` over the real registry) for end-to-end wiring.

## 8. Dockerfile (real entrypoint + GPU)

- [ ] 8.1 **Check first (manual gate, documented + reproducible):**
  `docker build -t srp:test --build-arg SRP_PREDICT_CODE_SHA=deadbeef .`;
  `docker inspect srp:test` shows exec-form `ENTRYPOINT ["python","-m","sleap_roots_predict"]`
  and **no** leftover `Cmd`; then, since the fixture is dockerignored,
  `docker run --rm -v "$PWD/tests/assets/scans:/in" -v "$PWD/out:/out" srp:test /in /out` emits
  `{scan_key}.predictions.json` with `predict_code_sha == "deadbeef"` and exits 0. Note: the GPU
  path needs a Linux+NVIDIA host; the CPU fixture run works on any host (device auto→cpu).
- [ ] 8.2 Replace the REPL stub: exec-form ENTRYPOINT and **remove the stale `CMD`**;
  `uv sync --frozen --no-dev --extra linux_cuda --python 3.12`; `ARG SRP_PREDICT_CODE_SHA=""` →
  `ENV` after the heavy layers; keep `MPLBACKEND=Agg` + the apt libs.

## 9. CI workflow (bake the sha + CUDA build reliability)

- [ ] 9.1 **Check first:** a guard test (or documented check) asserts `docker-build.yml` passes
  `build-args: SRP_PREDICT_CODE_SHA=${{ github.sha }}` and tags `type=sha,format=long`.
- [ ] 9.2 Add the `build-args` block; set `type=sha,format=long` (so the `sha-<full>` tag equals
  the baked `predict_code_sha`); add a free-disk-space step and switch `cache-to` to `mode=min`
  (or a registry cache) so the multi-GB CUDA build fits GitHub's 10 GB cache + `ubuntu-latest`
  disk on the build-only PR job. (Triggers otherwise unchanged.)

## 10. Docs + validation

- [ ] 10.1 Update docs (respecting the repo's SSOT direction — `openspec/project.md` for module
  layout, `__all__`+`API.md` for the API, README for env/usage; CLAUDE.md is being retired):
  - `API.md` — add `run_batch` to the Output Contract / public-API section.
  - `README.md` — add `batch.py` + `__main__.py` to the Project Structure tree and a
    "Run the container" Usage snippet (`docker run <image> <in> <out>`); note provenance SHAs
    (`SRP_PREDICT_CODE_SHA`) are image-baked at build time.
  - `openspec/project.md` — container extra `cpu`→`linux_cuda`; drop "the serving protocol/CLI"
    from the remaining-work note; bring the Architecture Patterns module list current
    (`output_contract.py`, `batch.py`, `__main__.py`).
  - `CLAUDE.md` — prune (don't expand) its duplicated package-list/export-list/env-list,
    replacing with pointers to project.md / API.md / README.
  - `CHANGELOG.md` — add an `[Unreleased] → Added` entry.
- [ ] 10.2 Run `black`, `ruff check`, `codespell` (scans the whole tree — sidecar JSON + docs
  must be typo-clean); `pytest` (CPU suite green); the `gpu` subset + the docker-run gate via
  `/pre-merge`.
- [ ] 10.3 `openspec validate add-predict-container-cli --strict` passes.
