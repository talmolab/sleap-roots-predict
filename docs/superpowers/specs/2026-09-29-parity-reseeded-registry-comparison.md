# C3: parity against the re-seeded registry (2026-09-29)

Tracking issue: talmolab/sleap-roots-predict#34, step C3. This is a read-only verification
run. It changes no code. C3 does not gate sleap-roots-training 6.3, and in fact 6.3 was
executed the same morning (see [Registry state at run time](#registry-state-at-run-time)).

## Result

**Pass.** All 8 physical models were evaluated. None is missing, gapped or out of tolerance,
and every model has the same weights checksum as on 2026-08-04.

- For 7 of the models, every sleap-nn metric is within **6.4e-3** of 2026-08-04. They are
  close but not bit-identical, which is consistent with GPU-vs-CPU numerics everywhere.
- The eighth, `rice/older/crown`, has a sleap-nn `distance_p95` that moved by +2.4% on GPU. A
  CPU re-run of that card reproduces 2026-08-04 **exactly**. So the shift comes from the
  device and/or the torch build, not from the registry.

## What was run

- **Main run:**
  `uv run python scripts/run_parity_harness.py --out docs/superpowers/specs/2026-09-29-parity-reseeded-registry-results.json`.
  - It ran from 15:01:39 to 15:26:09 UTC on 2026-09-29, at predict `main` `9a6f20c`.
  - Settings: `SRP_PARITY_DATA_DIR=Z:/users/eberrigan/SLEAP`, the default `--share-root`, and
    `sample_n=100`.
  - Environment: `windows_cuda` extra, torch 2.11.0+cu128 on an RTX A5000 (`device=cuda`),
    sleap-nn 0.3.0, sleap-io 0.8.0, sleap-roots-contracts 0.1.0a9.
- **CPU cross-check:**
  [`2026-09-29-parity-reseeded-registry-cpu-rice-older-crown.json`](2026-09-29-parity-reseeded-registry-cpu-rice-older-crown.json).
  - It ran from 15:27:35 to 15:33:16 UTC with the `cpu` extra (torch 2.12.1+cpu, `device=cpu`).
  - `run_parity_harness.py` has no card filter, so this ran through the same library entry
    point with the same arguments the script passes, and only the one card:

    ```python
    source = WandbRegistrySource()
    cards = [c for c in source.list_cards() if "rice-older-crown" in c.registry_id]
    run_parity_harness(
        cards, source, workdir, out_path,
        prefix_map={p: "Z:/users/eberrigan/SLEAP" for p in PREFIX_MAP_SOURCES},
        basename_index=build_basename_index("Z:/users/eberrigan/SLEAP"),
        sample_n=100,
    )
    ```

## Registry state at run time

Training 6.3 unlinked the 13 flat collections from `production` between 15:02:40 and
15:03:19 UTC. The source is training's `docs/migration/2026-09-29-retire-flat-collections-record.json`.

- The **main run** listed the registry in the minute before that. `list_cards()` returned the
  8 selector-shaped cards and skipped all 13 flat cards with a `ValidationError` warning.
  That skip is expected under contracts 0.1.0a9, because it requires `selectors`.
- The **CPU cross-check** ran after the retirement. Its listing logged 0 skips.
- The retirement doesn't affect the metrics. It only removed the flat collections, and both
  runs evaluated the same 8 selector cards, all at `v0`, with the digests below.

## Mapping to the baseline

The two results files are joined on `source_model_id`, since the registry ids differ
between them. The maps come from sleap-roots-training, branch
`migrate-model-card-selectors` at `dc216c7`. Both files are unchanged at the branch tip,
`41436bb`.

- Old ids, the 13 flat collections, map to `source_model_id` through
  `docs/migration/2026-09-22-pre-reseed-baseline.json`, the 6.0(a) baseline.
- New ids, the 8 selector collections, map to `source_model_id` through
  `docs/migration/2026-09-24-post-reseed-state.json`. Its `weights_checksum_mismatches`
  is `[]`.

You can also check the join from this repo alone by using `weights_checksum`, which gives
the same 8 groups. The 13 old entries collapse to 8 physical models, and old entries that
share weights had identical metrics. Each new card's `selectors` are exactly the old flat
cards' (species, mode, age window) contexts for that `source_model_id`: none were lost or
added.

## Per physical model

The gate is `parity.within_tolerance`:

- **Distance:** the relative delta, |Δ`distance_p95`| / reference, must be ≤ 0.25.
- **Recall:** sleap-nn minus reference must be ≥ −0.10.

The recall column is signed. That differs from the results JSON: its `*_delta` fields are
absolute values, so the signs in this table were recomputed.

The classic-SLEAP reference was **bit-identical** to 2026-08-04 for all 8 models. Ground-truth
source and frame counts were also unchanged for every model. In the frames column, *e* is
frames evaluated and *r*/*t* is frames resolved over the total in the ground truth.

| `source_model_id` | old collections → new | frames *e* (*r*/*t*) | sleap-nn p95, 08-04 → 09-29 | sleap-nn recall, 08-04 → 09-29 | gate rel. Δp95, 08-04 → 09-29 | gate Δrecall, 08-04 → 09-29 | gate |
|---|---|---|---|---|---|---|---|
| `arabidopsis/lateral/240130_140452.multi_instance.n=337` | arabidopsis-, arabidopsis-multiplant-cylinder-lateral-age2-14 → `arabidopsis-lateral-…n-337` | 34 (34/34) | 66.17 → 66.17 | 0.8893 → 0.8893 | 0.075 → 0.075 | −0.043 → −0.043 | pass |
| `canola/lateral/240611_083419.multi_instance.n=631` | canola-cylinder-lateral-age2-13, pennycress-cylinder-lateral-age2-14 → `canola-lateral-…n-631` | 63 (63/63) | 63.83 → 63.83 | 0.9752 → 0.9752 | 0.061 → 0.061 | +0.001 → +0.001 | pass |
| `canola_pennycress_arabidopsis/primary/240611_102513.multi_instance.n=743` | arabidopsis-, arabidopsis-multiplant-, pennycress-cylinder-primary-age2-14, canola-cylinder-primary-age2-13 → `canola_pennycress_arabidopsis-primary-…n-743` | 66 (66/74) | 66.55 → 66.55 | 0.8889 → 0.8889 | 0.013 → 0.013 | −0.085 → −0.085 | pass |
| `rice/older/crown/221208_113552.multi_instance.n=574` | rice-cylinder-crown-age6-10 → `rice-older-crown-…n-574` | 57 (57/57) | 194.12 → **198.84** (+2.4%) | 0.7439 → 0.7427 | 0.170 → **0.150** | −0.053 → −0.054 | pass |
| `rice/younger/crown/220821_163331.multi_instance.n=867` | rice-cylinder-crown-age2-5 → `rice-younger-crown-…n-867` | 87 (87/87) | 36.52 → 36.52 | 0.8913 → 0.8913 | 0.059 → 0.059 | −0.010 → −0.010 | pass |
| `rice/younger/primary/230104_182346.multi_instance.n=720` | rice-cylinder-primary-age2-5 → `rice-younger-primary-…n-720` | 72 (72/72) | 19.57 → 19.57 | 0.9861 → 0.9861 | 0.039 → 0.039 | +0.000 → +0.000 | pass |
| `soybean/lateral/lateral_root_221006_172103.multi_instance.n=482` | soybean-cylinder-lateral-age2-8 → `soybean-lateral-…n-482` | 48 (48/48) | 127.69 → 127.69 | 0.9621 → 0.9621 | 0.117 → 0.117 | +0.001 → +0.001 | pass |
| `soybean/primary/221003_111420.multi_instance.n=1389` | soybean-cylinder-primary-age2-8 → `soybean-primary-…n-1389` | 100 (139/139) | 68.25 → 68.25 | 0.9816 → 0.9816 | 0.006 → 0.006 | +0.008 → +0.008 | pass |

- **Largest change among the other 7:** 6.37e-3 on any sleap-nn metric, which is
  arabidopsis/lateral `distance_avg`. That is below the table's display precision.
  For `rice/older/crown` the largest change is 7.2 px, on `distance_p90`.
- **Multi-selector cards:** the three cards with several selectors all resolved through
  `relinked_bundle`. The runner supplies no selector, which skips the basename-search age
  step on those cards. That skipped step therefore played no part in this run. The
  basename-search models are all single-selector.
- **What the checksums cover:** weights only. For the 7 models with no CPU re-run, the claim
  that inference config is also unchanged rests on their metric agreement.

## The one mover: `rice/older/crown`

| sleap-nn metric | 2026-08-04 | 09-29 CPU | 09-29 GPU | classic-SLEAP ref |
|---|---|---|---|---|
| `distance_p95` | 194.1156 | 194.1156 | 198.8364 | 233.90 |
| `distance_p90` | 132.3414 | 132.3414 | 139.5104 | 166.83 |
| `distance_p99` | 377.9419 | 377.9419 | 371.9860 | 399.10 |
| `visibility_recall` | 0.743908 | 0.743908 | 0.742663 | 0.796868 |
| `pck_mean` | 0.13045 | 0.13045 | 0.129114 | 0.13233 |

- **The re-seed changed nothing.** The CPU run on the re-seeded card reproduces every
  2026-08-04 metric exactly, so the re-seeded card gives the same result as the old flat
  card.
- **The 2026-08-04 baseline almost certainly ran on CPU.** The bit-exact match strongly
  indicates it, but its device and torch version were never recorded.
- **The GPU shift has two confounded causes.** The GPU run differs in both device and torch
  build (2.11.0+cu128 vs. 2.12.1+cpu), and this run does not separate the two. The
  attribution is therefore "device and/or torch build".
- **Why this model moved, probably.** Its high-error tail is large: p99 ≈ 372–378 px. On
  frames like that, small numeric differences in bottom-up peak finding and PAF grouping can
  flip instance assignments. That explanation is plausible but untested, since no second
  GPU run was made to rule out run-to-run GPU nondeterminism.
- **Possibly related to C2.** C2 saw cross-node GPU-vs-GPU differences on 3 canola scans (see
  the "C2 done" comment on predict#34), and its cause there was also unproven.
- **Direction of the shift.** It stays well inside tolerance at 0.150 against the 0.25 gate.
  The distance metrics p95 and p90 moved toward the classic-SLEAP reference. Recall and
  `pck_mean` moved slightly away: gate Δrecall went from −0.053 to −0.054.

## Caveat for future parity runs

- **GPU runs aren't bit-comparable to the CPU baseline.** Only this model visibly differed.
  A future run that needs to separate registry changes from numeric noise should use the
  `cpu` extra. Otherwise, re-run any model that moved on CPU, as this note did. README's
  Parity Harness section now says the same.
- **The report doesn't record its environment.** It carries no device, torch or sleap-nn
  version, and no commit. Recording those in the report would remove the inference in this
  note, and that would be a harness follow-up.
- **This run superseded the standing parity test.** It covered all 8 cards, so the
  `@pytest.mark.parity` test was not run separately.
