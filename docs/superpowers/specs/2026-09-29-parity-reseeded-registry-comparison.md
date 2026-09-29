# C3: parity against the re-seeded registry (2026-09-29)

Tracking issue: talmolab/sleap-roots-predict#34, step C3. This is a read-only verification
run. It changes no code. C3 does not gate sleap-roots-training 6.3.

## Result

**Pass.** All 8 physical models were evaluated. None is missing, gapped or out of tolerance,
and every model has the same weights checksum as on 2026-08-04. Seven models match the
2026-08-04 baseline to within 6e-3 on every sleap-nn metric. The eighth is
`rice/older/crown`. Its sleap-nn `distance_p95` moved by +2.4% on GPU. A CPU re-run of that
card reproduces 2026-08-04 **exactly**, so the shift comes from the device, not from the
registry.

## What was run

- `uv run python scripts/run_parity_harness.py --out docs/superpowers/specs/2026-09-29-parity-reseeded-registry-results.json`
  ran from 08:07 to 08:26 local on 2026-09-29, at predict `main` `9a6f20c`. The run used
  `SRP_PARITY_DATA_DIR=Z:/users/eberrigan/SLEAP` with the default `--share-root` and
  `sample_n=100`.
- Environment: `windows_cuda` extra, torch 2.11.0+cu128 on an RTX A5000 (`device=cuda`),
  sleap-nn 0.3.0, sleap-io 0.8.0, sleap-roots-contracts 0.1.0a9.
- `list_cards()` returned the **8** selector-shaped `production` cards. It skipped all 13
  flat cards with a `ValidationError` warning. That is expected, because contracts 0.1.0a9
  cards require `selectors`.
- Cross-check: [`2026-09-29-parity-reseeded-registry-cpu-rice-older-crown.json`](2026-09-29-parity-reseeded-registry-cpu-rice-older-crown.json)
  holds the same harness settings run on the `rice-older-crown` card only, on the `cpu`
  extra (torch 2.12.1+cpu, `device=cpu`).

## Mapping to the baseline

The two results files are joined on `source_model_id`, since the registry ids differ
between them. The maps come from sleap-roots-training, branch
`migrate-model-card-selectors` at `dc216c7`:

- Old ids, the 13 flat collections, map to `source_model_id` through
  `docs/migration/2026-09-22-pre-reseed-baseline.json`, the 6.0(a) baseline.
- New ids, the 8 selector collections, map to `source_model_id` through
  `docs/migration/2026-09-24-post-reseed-state.json`. Its `weights_checksum_mismatches`
  is `[]`.

The 13 old entries collapse to 8 physical models. Old entries that share weights had
identical metrics.

## Per physical model

The tolerance comes from `parity.py`. The relative distance delta is
|Δ`distance_p95`| / reference and must be ≤ 0.25. The recall delta is sleap-nn minus
reference and must be ≥ −0.10. The classic-SLEAP reference was **bit-identical** to
2026-08-04 for all 8 models. Frame resolution and ground-truth source were also unchanged
for every model.

| `source_model_id` | old collection(s) → new | frames | sleap-nn p95, 08-04 → 09-29 | sleap-nn recall, 08-04 → 09-29 | gate rel. Δp95, 08-04 → 09-29 | gate Δrecall, 08-04 → 09-29 | |
|---|---|---|---|---|---|---|---|
| `arabidopsis/lateral/240130_140452.multi_instance.n=337` | arabidopsis(-multiplant)-cylinder-lateral-age2-14 → `arabidopsis-lateral-…n-337` | 34 | 66.17 → 66.17 | 0.8893 → 0.8893 | 0.075 → 0.075 | −0.043 → −0.043 | ✅ |
| `canola/lateral/240611_083419.multi_instance.n=631` | canola-…-age2-13, pennycress-…-age2-14 → `canola-lateral-…n-631` | 63 | 63.83 → 63.83 | 0.9752 → 0.9752 | 0.061 → 0.061 | +0.001 → +0.001 | ✅ |
| `canola_pennycress_arabidopsis/primary/240611_102513.multi_instance.n=743` | 4 primary collections → `canola_pennycress_arabidopsis-primary-…n-743` | 66/74 | 66.55 → 66.55 | 0.8889 → 0.8889 | 0.013 → 0.013 | −0.085 → −0.085 | ✅ |
| `rice/older/crown/221208_113552.multi_instance.n=574` | rice-cylinder-crown-age6-10 → `rice-older-crown-…n-574` | 57 | 194.12 → **198.84** (+2.4%) | 0.7439 → 0.7427 | 0.170 → **0.150** | −0.053 → −0.054 | ✅ |
| `rice/younger/crown/220821_163331.multi_instance.n=867` | rice-cylinder-crown-age2-5 → `rice-younger-crown-…n-867` | 87 | 36.52 → 36.52 | 0.8913 → 0.8913 | 0.059 → 0.059 | −0.010 → −0.010 | ✅ |
| `rice/younger/primary/230104_182346.multi_instance.n=720` | rice-cylinder-primary-age2-5 → `rice-younger-primary-…n-720` | 72 | 19.57 → 19.57 | 0.9861 → 0.9861 | 0.039 → 0.039 | +0.000 → +0.000 | ✅ |
| `soybean/lateral/lateral_root_221006_172103.multi_instance.n=482` | soybean-cylinder-lateral-age2-8 → `soybean-lateral-…n-482` | 48 | 127.69 → 127.69 | 0.9621 → 0.9621 | 0.117 → 0.117 | +0.001 → +0.001 | ✅ |
| `soybean/primary/221003_111420.multi_instance.n=1389` | soybean-cylinder-primary-age2-8 → `soybean-primary-…n-1389` | 100 of 139 | 68.25 → 68.25 | 0.9816 → 0.9816 | 0.006 → 0.006 | +0.008 → +0.008 | ✅ |

The largest absolute change across all sleap-nn metrics in the seven unchanged rows is
5.7e-3, which is below the table's display precision. For `rice/older/crown` the change is
7.2 px, on `distance_p90`.

## The one mover: `rice/older/crown`

| sleap-nn metric | 2026-08-04 | 09-29 CPU | 09-29 GPU |
|---|---|---|---|
| `distance_p95` | 194.1156 | 194.1156 | 198.8364 |
| `distance_p90` | 132.3414 | 132.3414 | 139.5104 |
| `distance_p99` | 377.9419 | 377.9419 | 371.9860 |
| `visibility_recall` | 0.743908 | 0.743908 | 0.742663 |
| `pck_mean` | 0.13045 | 0.13045 | 0.129114 |

The CPU run reproduces every 2026-08-04 metric exactly, so the re-seeded card gives the
same result as the old flat card. That also shows the 2026-08-04 baseline ran on CPU.

On GPU this model's high-error frames reach p99 ≈ 378 px. On those frames, small numeric
differences in bottom-up peak finding and PAF grouping can flip instance assignments. This
is the same kind of cross-device noise that C2 saw on 3 canola scans (see the
"C2 done" comment on predict#34). The shift is still well inside tolerance: 0.150 against
the 0.25 gate. It moved closer to the classic-SLEAP reference, not further from it.

## Caveat for future parity runs

A parity run on GPU is not bit-comparable to the CPU baseline. Only this model showed a
visible difference, but a future run that needs to separate registry changes from device
noise should run on CPU. Otherwise, add a CPU re-run of any model that moved, as this note
did.
