# Design: predict output contract

- **Date:** 2026-07-05
- **Branch:** `add-predict-output-contract`
- **Status:** Approved (brainstorming) — proceeding to OpenSpec proposal + TDD
- **Slice origin:** deferred task 9.3 of the archived `add-warm-model-worker` change
- **Roadmap tier:** A3-predict → the predict→traits handoff on the A4 DAG
  (`images-downloader → predict(warm) → traits → write-back`)

## 1. Problem & goal

PR #9 shipped `WarmModelWorker`. Its `predict(params, video, save_dir=…)` currently
writes only raw `save_dir/<root_type>.slp` — no manifest, no scan-aware naming, no
provenance. This slice makes predict emit the **downstream output contract**: the
on-disk artifacts the traits stage (sleap-roots / A3-traits) reads to assemble a
`ResultEnvelope`.

This slice is the **output FORMAT only** — the files predict writes. It is independent
of *how* predict is invoked (per-scan container vs long-lived service — an open
A4/serving decision that writes the same files either way). The serving
protocol/entrypoint is a separate slice, out of scope here.

### The consumer grounds the format

`sleap_roots.Series.load(series_name, primary_path, lateral_path, crown_path, csv_path)`
(`c:/repos/sleap-roots/sleap_roots/series.py`) takes **explicit per-root `.slp` paths**
(loaded via `sio.load_slp`). `csv_path` is a plant-metadata CSV keyed on
`plant_qr_code` and is **optional** — Series loads fine without it, and it is upstream
Bloom scan metadata, not predict's inference output. Because `Series.load` accepts
explicit paths, the **manifest is the robust interface**; filenames are for
organization/discovery.

`load_series_from_h5s` documents the naming convention predict must match:
`{series_name}.model{model_id}.root{primary|lateral|crown}.slp` — no underscores,
`root`+type concatenated, and `series_name = Path(filename).name.split(".")[0]`
(⟹ `scan_key` and `model_id` must be **dot-free**).

## 2. Legacy reference (ported + redone)

The previous implementation is the GitLab `salk-tm/sleap-roots-predict` service (cloned
to `c:/repos/legacy-salk-tm-sleap-roots-predict`). Its `src/main.py` + `src/predict.py`:

- Invocation: `python src/main.py <images_input> <models_input> <output>` — read staged
  frames + the models-downloader output (`model_paths.csv` + model zips) and wrote
  predictions to `<output>`.
- Per-root `.slp` named `scan_{scan_id}.model_{model_id}.root_{model_type}.slp`.
- One **run-level** `predictions.csv` = the input `scans.csv` merged with a per-scan
  record `{scan_id, <model_type>: <slp filename>}`.
- **No provenance sidecar.**

### Deliberate deviations in this version

- **Drop `models_input`** — models are fetched from the wandb registry in-process
  (consolidated in PR #9). No `model_paths.csv`.
- **Modernize output** to `sleap-roots-contracts` types (`ModelRef` and the fields of
  `Provenance`/`BlobRef`) and `Series`-loadable `.slp`.
- **Per-scan artifacts** (not one run-level CSV): the warm worker's `predict()` is
  per-scan (one video), and traits is 1 envelope : 1 scan, so the per-scan combined
  JSON is the atomic unit. A run-level aggregate is deferred (YAGNI; trivially additive).
- **Add a provenance sidecar** so traits can assemble `Provenance` + `BlobRef` without
  recomputing predict's identity.

## 3. Settled decisions (with rationale)

1. **Schema home → predict-local now, upstream to contracts later.** The load-bearing
   contract (`ModelRef`, and the `Provenance`/`BlobRef` *fields*) already lives in
   `sleap-roots-contracts`. The only un-promoted shapes are (a) the top-level per-scan
   envelope and (b) a "partial `BlobRef`" (predict has no `s3_location`/`box_link` yet —
   those are filled downstream at A4 step G, and `BlobRef`'s validator rejects a
   location-less object, so predict cannot emit a valid `BlobRef` today). The reader
   (A3-traits) does not exist yet, so freezing a shared type now risks freezing it
   wrong; migrating later is cheap because it moves **one wrapper class**, not the
   boundary types. Define the wrapper in `sleap_roots_predict`, reuse `ModelRef` from
   contracts, document the JSON in the spec, and promote when A3-traits consumes it.
2. **Artifact layout → single combined per-scan JSON** (`{scan_key}.predictions.json`)
   holding *both* the manifest (per-root `.slp` paths + `model_id` + `plant_qr_code`)
   and the predict-side provenance (resolved `ModelRef`s, inference config, code
   sha/digest, per-`.slp` checksum + file_size). Atomic; traits reads one file per scan.
3. **Writer integration → thin standalone writer on top of `predict()`.** No change to
   `WarmModelWorker`. The writer is independently unit-testable.
4. **Build identity → env var + explicit-arg override, fail-soft.** The writer's only
   job is to record two strings (`predict_code_sha`, `predict_container_digest`). It
   accepts them as optional explicit args, defaulting to reading
   `SRP_PREDICT_CODE_SHA` / `SRP_PREDICT_CONTAINER_DIGEST` from the environment, and
   fails soft to `""` when absent (local runs still produce a sidecar). *How* the image
   stamps these (Dockerfile `ARG`/`ENV` + CI `--build-arg`) is a deployment detail owned
   by the serving/entrypoint slice; the writer stays agnostic and env-free-testable.
5. **Batch warmth → first-class now.** Warmth across a batch already lives in
   `WarmModelWorker` (predictors cached by `(registry_id, version)`; construct once,
   reuse across scans). A `predict_and_write_batch` convenience drives one warm worker
   over N scans and writes one subdirectory per scan. The output format is unchanged for
   batches.
6. **Plant-metadata CSV → out of scope.** Optional in `Series`, and it is upstream Bloom
   scan metadata, not inference output. `plant_qr_code` is recorded in the manifest
   (defaulting to `scan_key`) so traits can key metadata later, but predict does not
   produce the CSV.

## 4. Architecture

New module `sleap_roots_predict/output_contract.py`: two pydantic models, one dataclass,
one pure writer, one batch convenience. Exported from `__init__.py`. `WarmModelWorker` is
unchanged.

### 4.1 Schema (pydantic, predict-local)

```python
class PredictionArtifact(BaseModel):          # one per predicted root type
    root_type: Literal["primary", "lateral", "crown"]
    model_id: str          # filename-safe slug (discovery label)
    model: ModelRef        # full ref from contracts (source of truth for identity)
    slp_path: str          # BASENAME, relative to the manifest's directory
    checksum: str          # sha256 hex of the .slp
    file_size: int         # bytes

class PredictionManifest(BaseModel):          # one per scan
    schema_version: str = "1"                 # versions this predict-local shape
    scan_key: str
    plant_qr_code: str                        # defaults to scan_key
    artifacts: list[PredictionArtifact]       # may be [] (predict produced nothing)
    predict_inference_config: dict            # worker.inference_config() (FULL, audit)
    predict_output_params: dict               # worker.output_params() (output-defining)
    predict_code_sha: str                     # "" when unset
    predict_container_digest: str             # "" when unset
```

`Provenance.predict_models` is **derived** by traits as `[a.model for a in artifacts]`
— not duplicated in the file, so there is a single source of truth within the manifest.
`InputRef`/`images_checksum` and the full `Provenance`/`ResultEnvelope`/`idempotency_key`
are traits' responsibility (traits adds `traits_code_sha` etc.).

### 4.2 Public API

```python
def write_prediction_outputs(
    labels_by_root: dict[str, sio.Labels],    # from worker.predict(params, video)
    refs_by_root: dict[str, ModelRef],         # from worker.resolve(params)
    out_dir: str | Path,
    *,
    scan_key: str,
    plant_qr_code: str | None = None,          # None -> scan_key
    inference_config: dict,                    # worker.inference_config()
    output_params: dict,                       # worker.output_params()
    predict_code_sha: str | None = None,       # None -> env SRP_PREDICT_CODE_SHA -> ""
    predict_container_digest: str | None = None,  # None -> env SRP_PREDICT_CONTAINER_DIGEST -> ""
) -> PredictionManifest:
    """Write named .slp per root + {scan_key}.predictions.json into out_dir."""

@dataclass(frozen=True)
class ScanRequest:
    scan_key: str
    video: "sio.Video"
    params: ResolvedParams
    plant_qr_code: str | None = None
    overrides: dict[str, ModelRef] | None = None

def predict_and_write_batch(
    worker: WarmModelWorker,
    requests: Iterable[ScanRequest],
    out_dir: str | Path,
    *,
    predict_code_sha: str | None = None,
    predict_container_digest: str | None = None,
) -> list[PredictionManifest]:
    """Drive one warm worker over N scans, one subdir per scan; reuses residents."""
```

### 4.3 Filenames

- Per-root `.slp`: `{scan_key}.model{model_id}.root{root_type}.slp` (matches
  `load_series_from_h5s`).
- `model_id = slugify(f"{registry_id}_{version}")` — every char outside `[A-Za-z0-9-]`
  becomes `-` (drops the `reg/rice-primary` slash and any dots). Discovery label only;
  the full `ModelRef` is in the manifest.
- `scan_key` is **identity, not mangled**: if it contains `.` / `/` / `\` (or is empty)
  the writer **raises** rather than silently rewriting it.
- Combined JSON: `{scan_key}.predictions.json`.

### 4.4 Example on-disk output (batch of one scan; primary + lateral)

```
out_dir/
└── scan0731/
    ├── scan0731.modelreg-rice-primary-v1.rootprimary.slp
    ├── scan0731.modelreg-rice-lateral-v1.rootlateral.slp
    └── scan0731.predictions.json
```
```json
{
  "schema_version": "1",
  "scan_key": "scan0731",
  "plant_qr_code": "scan0731",
  "predict_code_sha": "",
  "predict_container_digest": "",
  "predict_inference_config": {"device": "cpu", "peak_threshold": 0.2, "batch_size": 4},
  "predict_output_params": {"peak_threshold": 0.2},
  "artifacts": [
    {"root_type": "primary", "model_id": "reg-rice-primary-v1",
     "model": {"registry_id": "reg/rice-primary", "version": "v1",
               "sleap_nn_version": "0.3.0", "root_type": "primary",
               "weights_checksum": null},
     "slp_path": "scan0731.modelreg-rice-primary-v1.rootprimary.slp",
     "checksum": "9f2c…", "file_size": 20481}
  ]
}
```
(The single-scan pure writer writes these three files directly into whatever `out_dir`
it is given; the batch convenience nests them under `out_dir/{scan_key}/`.)

## 5. Data flow (pure writer)

1. Validate `scan_key` is filename-safe (non-empty; no `.`/`/`/`\`); validate
   `set(labels_by_root) == set(refs_by_root)`.
2. `out_dir` created if missing (writer owns its output dir).
3. Per root type: write `{scan_key}.model{slug}.root{root_type}.slp`
   (`labels.save` / `sio.save_file`); compute sha256 hex + `file_size`; build a
   `PredictionArtifact` (with basename `slp_path`).
4. Build `PredictionManifest`; write `{scan_key}.predictions.json`
   (`manifest.model_dump_json`).
5. Return the manifest.

## 6. Error handling & edge cases

- Key mismatch between `labels_by_root` and `refs_by_root` → `ValueError`.
- Non-filename-safe / empty `scan_key` → `ValueError` (identity must not be mangled).
- Zero resolved roots → writes a manifest with `artifacts: []` (valid "predict produced
  nothing" record).
- `checksum` = sha256 hex; `slp_path` is a basename so the whole scan dir is
  relocatable/uploadable (BlobRef locations are filled downstream).

## 7. Scope boundaries

**IN:** the output format — named per-root `.slp` + single combined per-scan JSON
(manifest + provenance) — plus a pure writer and a batch convenience over the warm
worker.

**OUT:** MinIO/Box upload that fills `BlobRef` locations (A4 step G — predict writes
local paths + checksums); traits' consumption + full `ResultEnvelope`/`Provenance`/
`idempotency_key` (A3-traits, needs `traits_code_sha`); `InputRef`/`images_checksum`
(upstream from images-downloader); the plant-metadata CSV (upstream); the serving
protocol/entrypoint; the Dockerfile build-stamp wiring (deployment; the writer only
*reads* the two strings); a run-level aggregate manifest across scans (deferred, additive).

## 8. Testing — real TDD, no mocks

Drive the vendored native + legacy models through the warm worker (the existing
`rice_source` / `video` fixtures) to get **real** `Labels` + `ModelRef`s, then:

- named `.slp` reload via `sio.load_file` and contain labeled frames;
- `{scan_key}.predictions.json` round-trips `PredictionManifest`; per-root paths are
  correct; `ModelRef` round-trips; `checksum`/`file_size` match recomputation;
- **acceptance:** `sleap_roots.Series.load(series_name=scan_key, primary_path=…,
  lateral_path=…, crown_path=…)` loads the output cleanly and exposes labels;
- error cases: bad `scan_key` raises; key mismatch raises;
- build identity: explicit arg wins → env fallback → `""` when absent;
- **batch warmth:** one worker + `predict_and_write_batch` over 2 scans writes per-scan
  subdirs and reuses the resident `Predictor`s (asserted by object identity).

`sleap-roots` is added as a **test-only** dependency (`dev` extra); the acceptance test
uses `pytest.importorskip("sleap_roots")` so it skips gracefully if unavailable. Predict
must **not** runtime-depend on sleap-roots (no cycle: sleap-roots depends on the
contract, not on predict).

## 9. Risks

- **sleap-io co-resolution (checked; not a conflict):** `sleap-roots` requires
  `sleap-io>=0.0.11` (open upper) and has **no** `sleap-nn` dependency; sleap-nn 0.3.0
  pins `sleap-io>=0.8.0,<0.9.0`; the env resolves to 0.8.0, satisfying both. Adding
  `sleap-roots` to the `dev` extra also pulls `scikit-image`/`shapely`/`scipy`/
  `pywavelets`/`seaborn`/`matplotlib` — flexible, no contention expected. First TDD step
  still confirms the dev env resolves; if a real conflict appears, fall back to
  `importorskip` + a documented compatible-env note (do **not** loosen the inference
  pin). Optional low-priority advisory issue on `talmolab/sleap-roots` to tighten its
  `sleap-io` range — not blocking.

## 10. Post-merge (required)

Update the pipeline roadmap
(`sleap-roots-pipeline/docs/bloom-integration/roadmap.md`, A4 DAG ~line 109): predict
output contract done → unblocks the A3-traits input. If/when the manifest+sidecar schema
is promoted to `sleap-roots-contracts`, note the new contract + version there.
