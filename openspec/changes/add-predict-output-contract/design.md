## Context

Full brainstorming design: `docs/superpowers/specs/2026-07-05-predict-output-contract-design.md`.
This captures the technical decisions that shape the spec deltas. The consumer is
sleap-roots `Series.load` (explicit per-root `.slp` paths; optional metadata CSV). The
legacy reference is the GitLab `salk-tm/sleap-roots-predict` service (per-root `.slp` +
one run-level `predictions.csv`, no provenance).

## Goals / Non-Goals

- **Goals:** a stable per-scan output format traits can read — named per-root `.slp`
  (Series-compatible) + one combined JSON (manifest + predict-side provenance) — plus a
  pure writer and a batch convenience that reuses the warm worker.
- **Non-Goals:** upload/`BlobRef` locations, full `Provenance`/`ResultEnvelope`, run
  identity (`idempotency_key`), `InputRef`/`images_checksum`, plant-metadata CSV, serving
  entrypoint, Dockerfile build stamping. `WarmModelWorker` itself is unchanged.

## Decisions

- **Schema home = predict-local now, upstream later.** The load-bearing contract
  (`ModelRef`, the `Provenance`/`BlobRef` fields) already lives in `sleap-roots-contracts`.
  The only un-promoted shapes are the per-scan envelope and a "partial `BlobRef`" (predict
  has no location yet; `BlobRef`'s validator rejects a location-less object). A3-traits (the
  reader) does not exist yet, so freezing a shared type now risks freezing it wrong;
  migrating later moves one wrapper class. Reuse `ModelRef` from contracts inside the
  predict-local manifest.
  - *Alternative:* add the schema to `sleap-roots-contracts` now — rejected: coordinated
    multi-repo release + pin bump to freeze a shape with no live consumer.
- **One combined per-scan JSON** (`{scan_key}.predictions.json`), not two files — atomic;
  traits reads one file per scan. `predict_models` is derived by traits as
  `[a.model for a in artifacts]` (single source of truth in-file, no duplication).
- **Thin standalone writer** on top of `predict()`; `WarmModelWorker` untouched.
- **Fail-soft build identity** (explicit arg → env → `""`): the writer only records two
  strings; how they are stamped into the image is a deployment concern.
- **Batch = first-class now.** Warmth already lives in `WarmModelWorker` (predictors cached
  by `(registry_id, version)`). `predict_and_write_batch` drives one worker over N scans,
  one subdir per scan. No run-level aggregate manifest (YAGNI; additive later).
- **Filenames:** `{scan_key}.model{model_id}.root{primary|lateral|crown}.slp` matches
  `load_series_from_h5s`. `series_name = filename.split(".")[0]`, so `scan_key` and
  `model_id` must be dot-free: `model_id = slugify(f"{registry_id}_{version}")` (non
  `[A-Za-z0-9-]` → `-`); `scan_key` is identity and is rejected (not mangled) if unsafe.

## Risks / Trade-offs

- **`sleap-roots` co-resolution (checked, not a conflict):** `sleap-roots` needs
  `sleap-io>=0.0.11` (open upper) and no `sleap-nn`; sleap-nn 0.3.0 pins
  `sleap-io>=0.8.0,<0.9.0`; env resolves to 0.8.0 → satisfies both. Mitigation: first TDD
  task confirms the `dev` env resolves; the acceptance test uses
  `pytest.importorskip("sleap_roots")` and never loosens the inference pin.
- **Predict-local schema drift vs traits:** mitigated by `schema_version` on the manifest
  and by co-designing the shape with A3-traits before promotion to contracts.

## Migration Plan

Additive: new module + new capability + a test-only dep. No behavior change to existing
capabilities. Promotion of the manifest schema to `sleap-roots-contracts` is a future,
separately-versioned step once A3-traits consumes it.

## Open Questions

- None blocking. Promotion timing to `sleap-roots-contracts` is deferred to the A3-traits
  work (design decision 1).
