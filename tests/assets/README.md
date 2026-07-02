# Test assets

Real fixtures for no-mock inference tests. All vendored from
[sleap-nn](https://github.com/talmolab/sleap-nn) **v0.3.0** test assets
(commit `f92d8127`). These are minimal **bottom-up** models, chosen to mirror
the production sleap-roots model architecture (legacy SLEAP `.multi_instance`
UNet bottom-up).

## Models

- `models/minimal_bottomup_native/` — native sleap-nn format
  (`best.ckpt` + `training_config.yaml`). From
  `tests/assets/model_ckpts/minimal_instance_bottomup/`.
- `models/minimal_bottomup_legacy/` — **legacy SLEAP** format
  (`training_config.json` + `best_model.h5`), the exact format the production
  root models use. From `tests/assets/legacy_models/minimal_instance.UNet.bottomup/`.
  Verifies that sleap-nn 0.3.0 can load legacy SLEAP UNet weights.

## Video / images

- `videos/centered_pair_small.mp4` — small real video (384x384, grayscale).
  From `tests/assets/datasets/centered_pair_small.mp4`.
- `images/centered_pair/frame_000.png … frame_007.png` — first 8 frames of the
  above video, extracted to exercise the real `make_video_from_images` →
  `predict_on_video` path quickly (all 8 frames contain instances).

These models are trained on the fly-pair dataset, not roots; they exist only to
prove the sleap-nn 0.3.0 inference API path end-to-end on CPU. Real root-model
validation is the acceptance test (`tests/test_acceptance.py`) and the deferred
parity slice.
