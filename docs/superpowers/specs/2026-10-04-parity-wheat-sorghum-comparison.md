# Parity for the wheat and sorghum candidate cards (2026-10-04)

Tracking issue: talmolab/sleap-roots-predict#51, step 2 of talmolab/sleap-roots-pipeline#118.
This is a read-only verification run. It changes no code and makes no registry change. It gates
linking these three cards to `production` (#118 step 6).

## Result

**Pass.** All 3 cards are within the `prediction-parity` tolerance decided 2026-08-04 in
predict#33: relative Δ`distance_p95` ≤ 0.25 and Δ`visibility_recall` ≥ −0.10.

- Every card resolved its ground truth by bundle relinking alone, with every frame resolved.
  There were no gaps.
- The classic-SLEAP reference and sleap-nn were scored on the identical frames.
- **Wheat crown passes marginally.** Its relative Δp95 is +0.206, with sleap-nn worse. A frame
  bootstrap puts that between −0.03 and +0.60 (95%), and above 0.25 with probability 0.40.
- **Wheat frames hint at a gap.** On the 11 wheat frames alone, sleap-nn's Δp95 is +0.825. One
  mis-paired root accounts for much of that, but without that frame it's still +0.321. So a
  wheat-specific difference isn't ruled out, and 11 frames can't settle it (95% range −0.04 to
  +1.97). See [Wheat's Δp95](#wheats-δp95). The trait check, talmolab/sleap-roots-pipeline#120
  part 1 (≥ 20 real wheat scans), should be read with this in mind before #118 step 6.
- Sorghum primary is a clear pass. Sorghum lateral passes but is noisy.
- The model-selection checks from #51 all pass, with the 8 production cards and the 3 candidates
  loaded together.

## What was run

- **Cards:** the 3 cards carrying the W&B alias `candidate`, read with
  `SRP_WANDB_MODEL_ALIAS=candidate`. `production` was not read for parity and was not changed.
- **Parity run:** 21:08:23 to 21:33:47 UTC on 2026-10-04, at predict `main` `c25ef52`.
  - Environment: the `cpu` extra, torch 2.12.1+cpu (`device=cpu` in the logs), sleap-nn 0.3.0,
    sleap-io 0.8.0, sleap-roots-contracts 0.1.0a9. This is the same environment as the
    2026-09-29 CPU cross-check. GPU runs aren't bit-comparable to the CPU baseline (see README).
  - Settings: `SRP_PARITY_DATA_DIR=Z:/users/eberrigan/SLEAP`, the default `--share-root`, and
    `sample_n=100`.
- **One registry listing.** Listing walks every artifact and takes several minutes. So the
  registry was walked once, from 21:00:14 to 21:03:50 UTC, and every artifact's aliases, metadata
  and digest were snapshotted. Each alias's cards were then built from that snapshot with
  `WandbRegistrySource(alias=…)._collect_cards`, which applies the real alias filter and
  `ModelCard` validation.
- **Entry point.** `run_parity_harness.py` lists the registry itself and has no card filter. So,
  as in the 2026-09-29 CPU cross-check, the run called the same library entry point with exactly
  the script's arguments:

  ```python
  source = WandbRegistrySource()          # SRP_WANDB_MODEL_ALIAS=candidate
  cards = candidate_cards_from_snapshot   # the 3 cards below
  run_parity_harness(
      cards, source, workdir, out_path,
      prefix_map={p: _DEFAULT_SHARE_ROOT for p in PREFIX_MAP_SOURCES},
      basename_index=build_basename_index(os.environ["SRP_PARITY_DATA_DIR"]),
      sample_n=_SAMPLE_N,                 # 100
  )
  ```

  The report is [`2026-10-04-parity-wheat-sorghum-results.json`](2026-10-04-parity-wheat-sorghum-results.json).

## Registry state at run time

The snapshot holds 104 model artifacts: `production` on 8 and `candidate` on 3. No artifact was
skipped as non-conforming.

- The 8 `production` cards and their weights checksums are unchanged from 2026-09-29.
- The 3 `candidate` cards were published by sleap-roots-training#73 (squash `06f8bb4`, seed run
  `4ihxu1mg`).

| root type | collection | version | `weights_checksum` | selector |
|---|---|---|---|---|
| crown | `20250401_wheat_models-250328_095645.multi_instance.n-1658` | v0 | `c600f25e7ed81482020ede734ba1ac4c` | wheat, cylinder, 5–14 |
| primary | `20250204_sorghum_experimental-sorghum_soybean_primary_6nodes-250203_181521.multi_instance.n-1689` | v0 | `56817fe98fb639c0fdffbe37495f5c34` | sorghum, cylinder, 3–14 |
| lateral | `20250204_sorghum_experimental-sorghum_soybean_lateral_4nodes-250203_214033.multi_instance.n-590` | v0 | `891a4ef59da6c116e557fab1aa8d54a7` | sorghum, cylinder, 3–14 |

## Ground truth and like-for-like checks

Ground truth for each card is its own bundle's `labels_gt.val.slp`, the held-out validation
split. The labels registry wasn't used: the harness never passes `labels_registry_lookup`, and
the wheat labels set may be the model's own training data.

All three bundles were trained under `D:/SLEAP/SLEAP_{wheat,sorghum,Rice,Soy}/…`.
`PREFIX_MAP_SOURCES` already maps `D:/SLEAP` to the share root, so **no harness change was
needed.** The run's intermediates were audited per card:

- **Resolved by relink only.** `ground_truth_source` is `relinked_bundle`, with
  `n_frames_resolved == n_frames_total`. No frame fell through to basename search or to a gap.
- **Right files.** Every evaluated video path is under `Z:/users/eberrigan/SLEAP/` and exists.
  Every frame is 1080×2048×1, which matches the training configs' `target_height`/`target_width`.
- **Same frames.** The reference's `settings` is `recomputed` from the bundle's own
  `labels_pr.val.slp`, not read from the full-split `metrics.val.npz`. The sampled ground truth,
  the classic-SLEAP predictions and the sleap-nn predictions have identical
  `(video, frame_idx)` sets, with no duplicates.
- **Same skeleton.** The node names and order are identical on all three sides.

## Per card

*e* is frames evaluated, and *r*/*t* is frames resolved over the total in the ground truth. The
deltas are signed, as sleap-nn minus reference. A positive Δp95 means sleap-nn is worse. The gate
uses the absolute relative Δp95.

| card | frames *e* (*r*/*t*) | p95, sleap-nn vs ref | rel. Δp95 | recall, sleap-nn vs ref | Δrecall | gate |
|---|---|---|---|---|---|---|
| wheat crown n-1658 | 100 (166/166) | 236.54 vs 196.11 | +0.206 | 0.7958 vs 0.7990 | −0.003 | pass |
| sorghum primary n-1689 | 100 (169/169) | 91.33 vs 92.84 | −0.016 | 0.8783 vs 0.8950 | −0.017 | pass |
| sorghum lateral n-590 | 59 (59/59) | 104.63 vs 112.48 | −0.070 | 0.9440 vs 0.9449 | −0.001 | pass |

## The validation sets are mostly the other species

These are joint models. Wheat crown was trained on wheat seminal plus rice labels, and both
sorghum models on sorghum plus soybean labels. Their held-out sets are mixed the same way.

| card | full val set | evaluated (first *n*) | target-species ages in the val set |
|---|---|---|---|
| wheat crown | 149 rice, 17 wheat | 89 rice, **11 wheat** | days 5, 11, 14 |
| sorghum primary | 133 soybean, 36 sorghum | 78 soybean, **22 sorghum** | days 5, 6, 10, 12 |
| sorghum lateral | 49 soybean, 10 sorghum | 49 soybean, **10 sorghum** | days 5, 10 |

The gate above is computed over the whole evaluated set, as for every earlier model. Everything
below is informational and doesn't change the gate. The data is in
[`2026-10-04-parity-wheat-sorghum-diagnostics.json`](2026-10-04-parity-wheat-sorghum-diagnostics.json)
(`pairing_comparison`, `bootstrap`, `drop_one_frame`, plus a `method` block). All of it uses the
same frames and predictions as the harness report, with no new inference.

## Wheat's Δp95

On its 11 wheat frames, wheat crown's Δp95 is +0.825 (140.4 vs 76.9 px).

**The pipeline was checked for a fault, and none was found.** The files, frames, skeleton and
image size are all as above. The training config's preprocessing and model settings are
identical to the sorghum primary model's.

### Instance pairing explains part of it

`sleap_nn.evaluation` pairs predictions to ground truth greedily, in descending prediction score,
with an OKS threshold of 0 (`OKS_MATCH_THRESHOLD` in `parity.py`).

- A pair with OKS exactly 0 is rejected. Its ground-truth root becomes a false negative and adds
  no distances.
- Any pair above 0 is accepted, however poor.
- On frames with several roots close together, a high-scoring prediction can take a neighbouring
  root. The true root is then left with a far-away leftover at a low but nonzero OKS.
- Classic SLEAP and sleap-nn score their instances differently, so they mis-pair on different
  frames.
- p95 over a few hundred points is sensitive to a handful of such pairs.

How this shows up in the wheat card:

- **Rice frames are crowded.** They carry up to 12 labelled roots. The 89 rice frames have 54
  near-zero-OKS pairs for classic and 68 for sleap-nn (OKS < 0.01).
- **Wheat frames are not.** They carry 3–6 roots and have 1 near-zero-OKS pair on each side.
  - sleap-nn's one near-zero pair, on `LSSSVVVVOJ` f56, is 208–317 px and dominates the slice's
    tail.
  - Without that frame, the wheat-slice Δp95 is still **+0.321**. The remainder comes from
    accepted pairs with moderate OKS, for example on `142QU1HI9A` f29 and `BZ52UFGM79` f41.
  - So pairing doesn't explain the whole wheat difference.

### How noisy the numbers are

These are paired frame-bootstrap ranges on the greedy-OKS signed rel. Δp95: 2,000 resamples,
with the same frames drawn for both engines.

| card | frames | point estimate | 95% range | P(\|rel\| > 0.25) |
|---|---|---|---|---|
| wheat crown | all 100 | +0.206 | −0.03 to +0.60 | 0.40 |
| | wheat 11 | +0.825 | −0.04 to +1.97 | 0.83 |
| sorghum primary | all 100 | −0.016 | −0.13 to +0.16 | 0.00 |
| | sorghum 22 | −0.184 | −0.47 to +0.14 | 0.14 |
| sorghum lateral | all 59 | −0.070 | −0.43 to +0.45 | 0.29 |
| | sorghum 10 | −0.174 | −0.60 to +0.80 | 0.45 |

Frames are the first *n* in bundle order, so they cluster by video and plate. These ranges treat
frames as independent, so if anything they are too narrow.

### Re-scoring with a different pairing

The same predictions were also paired per frame by a minimum-cost assignment (Hungarian
algorithm) on mean point distance.

| card | frames | greedy Δp95 | optimal Δp95 | optimal p50, ref / sleap-nn | near-zero-OKS pairs, ref / sleap-nn |
|---|---|---|---|---|---|
| wheat crown | all 100 | +0.206 | −0.031 | 11.6 / 11.6 | 55 / 69 |
| | rice 89 | +0.169 | −0.106 | 11.2 / 11.3 | 54 / 68 |
| | wheat 11 | +0.825 | +0.084 | 13.9 / 13.8 | 1 / 1 |
| sorghum primary | all 100 | −0.016 | −0.042 | 15.3 / 14.8 | 2 / 1 |
| | soybean 78 | −0.035 | −0.042 | 14.4 / 14.3 | 0 / 0 |
| | sorghum 22 | −0.184 | +0.042 | 19.9 / 20.3 | 2 / 1 |
| sorghum lateral | all 59 | −0.070 | +0.058 | 2.7 / 2.9 | 118 / 119 |
| | soybean 49 | +0.077 | +0.116 | 2.6 / 2.7 | 104 / 106 |
| | sorghum 10 | −0.174 | +0.161 | 4.7 / 4.6 | 14 / 13 |

- **The pairing effect runs both ways.** It made the wheat card look worse and the sorghum slices
  look better. On sorghum-only frames, −0.18 and −0.17 under greedy pairing become +0.04 and
  +0.16 under optimal pairing.
- **This pairing has a known bias.** Its cost averages only over nodes visible on *both* sides,
  so partial (fragment) predictions are cheap to pair. It yields fewer scored points than greedy.
  On sorghum primary, classic goes from 537 to 520 points and sleap-nn from 527 to 516. Both engines emit fragments at similar rates, but the
  bias favours whichever emits more. Treat these numbers as supporting evidence, not as a
  corrected metric.
- **It's not a substitute gate.** The gate stays greedy-OKS, unchanged from predict#33.

### Conclusion for wheat

The whole-card result passes as decided, but marginally.

- **On wheat frames, sleap-nn may be somewhat worse** than classic SLEAP. Part of the measured
  gap is pairing, but not all of it.
- **The bundle can't settle it.** Its 17 wheat labels are too few to confirm or rule this out.
- **The trait check is the better evidence.** It compares traits from ≥ 20 real wheat scans,
  talmolab/sleap-roots-pipeline#120 part 1, and should be read with this in mind before #118
  step 6.

## Selection checks

`choose_models` was run against the 8 `production` cards plus the 3 `candidate` cards from the
same snapshot.

| scan (cylinder) | selected | `past_window_age` |
|---|---|---|
| wheat 10 | crown → wheat n-1658 | None |
| wheat 20 | crown → wheat n-1658 | 14 (clamped) |
| sorghum 8 | primary → n-1689, lateral → n-590 | None |
| sorghum 17 | primary → n-1689, lateral → n-590 | 14 (clamped) |
| sorghum 2 | nothing | None |

- **Soybean is unaffected.** For every age from 0 to 20, soybean's selection and
  `past_window_age` are identical with and without the candidates. Ages 0–1 match nothing,
  ages 2–8 select `soybean-primary-…n-1389` and `soybean-lateral-…n-482`, and ages 9–20 clamp
  to 8. The sorghum models are joint sorghum+soybean, but their selectors name only `sorghum`.
- **Related test fixes are in predict#53.** `tests/test_model_selection.py` used `sorghum cylinder
  30` as its no-card example, which would go stale once the test catalog mirrors these cards.
  `tests/test_param_resolution.py` called sorghum "an unmodelled species". predict#53 switches
  both to `alfalfa`.

## Coverage caveats

- **Only part of each age window is tested.** The labels cover only part of each card's window.
  - Wheat (5–14) has wheat labels at days 5, 11 and 14.
  - Sorghum (3–14) has sorghum labels at days 5, 6, 10 and 12 (primary) and days 5 and 10
    (lateral).
  - Wheat days 6–10 and 12–13, and sorghum days 3–4, 7–9, 11 and 13–14, have no labelled frames.
    Those ages are untested here.
  - The sorghum list is the union of both cards. The lateral card also has no sorghum labels at
    days 6 and 12.

  Past production ran these models at those ages the same way.
- **Most evaluated frames aren't the target species.** The gate rests mostly on rice and soybean
  frames (see above). The target-species slices are 11, 22 and 10 frames.
- **`sample_n=100` takes the first 100 frames.** For wheat this left out 6 of the 17 wheat
  frames, and for sorghum primary 14 of the 36 sorghum frames.
- **This is not the trait check.** That is talmolab/sleap-roots-pipeline#120 part 1, which
  compares traits against past runs.

## Possible follow-ups (not in this change)

- **Pairing in the harness.** The harness's greedy OKS pairing with threshold 0 is fragile on
  crowded crown and lateral frames. Changing the matching or tolerance would be its own decision
  and would need a re-baseline, the same as predict#33.
- **Uncertainty in the gate.** predict#33's gate is a point estimate. A frame-bootstrap range,
  like the one above, would show when a pass or fail is marginal.
- **Environment in the report.** The report still doesn't record device, torch, sleap-nn or the
  commit, as already noted on 2026-09-29.
