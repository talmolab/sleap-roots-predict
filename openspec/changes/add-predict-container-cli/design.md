## Context

Full design: `docs/superpowers/specs/2026-07-06-predict-container-cli-design.md` (approved).
This slice is the deferred "serving protocol/entrypoint" that every prior predict slice
pushed to A4. The container contract is fixed by A4's Argo `predictor` template
(`docker run <image> <in> <out>`, `WANDB_API_KEY`, GPU, `sha-<sha>` pin) and by the
trait-extractor's input format (nested `out/{scan_key}/` = manifest + sidecar + `.slp`).

## Goals / Non-Goals

- **Goals:** a runnable batch container over the existing library; load-once; per-scan
  outputs matching #16; sidecar pass-through; existence-based resume; `predict_code_sha`
  baked build-arg→ENV→manifest; real exec-form GPU ENTRYPOINT.
- **Non-Goals:** changing the #16 manifest/`.slp` format or the worker/param APIs; S3/boto
  (filesystem-only); checksum-verified resume + atomic writes (#26); model-derived channel
  handling (#25); the sidecar's schema/producer (owned by A3-traits + the A4 downloader).

## Decisions

- **D1 — sidecar copy-through.** Predict reads the sidecar for params, then copies it verbatim
  into `out/{scan_key}/`, making its output a self-contained trait-extractor input tree. This
  closes a real gap in the A4 plan (sidecar authored into predict's *input* mount but read by
  traits from predict's *output* mount, no bridge). Copy ≠ author: predict never computes
  `image_ids`/`images_checksum`.
- **D2 — evolve the existing image.** Predict's single GHCR image *is* the service (unlike
  traits, which added a second image beside a library), and `type=sha` already emits the
  `sha-<sha>` tag A4 pins. So modify the root `Dockerfile` + `docker-build.yml` in place;
  switch `cpu`→`linux_cuda`; keep the existing `release:`/`type=semver` (harmless to A4).
  Verified: A4 is the first GHCR consumer (live templates still pull the legacy GitLab image).
  Alternative (a second image mirroring traits) rejected: redundant identity + A4 rewire.
- **D3 — hardcode `grayscale=True`.** sleap-nn 0.3.0 only converts channels on
  `ensure_rgb`/`ensure_grayscale` (both `false` in every config) and never reads `in_channels`
  — a mismatch is a shape crash. Today's cylinder models are all 1-channel, and `grayscale=None`
  autodetect is fragile on JPEG, so hardcode `True` and defer model-derived selection to #25
  (plates can be color). Alternative (derive from model config) is the #25 hardening.
- **Reuse over new code.** `run_batch` orchestrates discovery + resume + isolation + sidecar
  copy, delegating inference to `WarmModelWorker` and writing to `write_prediction_outputs`
  (#16). `predict_and_write_batch` is left untouched.

## Risks / Trade-offs

- **`linux_cuda` image size / CI build time** grows vs. the `cpu` image → accepted (GPU is the
  point); mitigated by placing the `ARG`/`ENV` sha-bake after the heavy layers.
- **Existence-only resume + non-atomic writes** → a pod killed mid-manifest-write leaves a
  truncated file a later run skips as done. Acceptable for the manual PoC; the two halves
  (checksum-verified skip + atomic temp→rename) must land together in #26.
- **`grayscale=True` breaks a future color model** → tracked (#25); the spec records the
  assumption.
- **Argo-readiness policy** (exit-on-empty, whole-batch-retry, SIGTERM/PID-1) mirrors the
  traits driver and is reconciled uniformly across both producers at A4 wiring (#26 ↔ traits
  #259).

## Migration Plan

Additive: new modules + a real ENTRYPOINT replacing an unused REPL stub. No consumer of the
GHCR `cpu` image exists yet, so flipping to `linux_cuda` is safe. Post-merge: hand the
published `sha-<sha>` tag to A4's Argo plan (Task 8) and update the roadmap A3-predict/A4 rows.

## Open Questions

None blocking. The multi-scan input layout is undecided in the A4 docs; `rglob
"*.scan_metadata.json"` works for both flat-single and nested-per-scan, so predict does not
force A4's hand.
