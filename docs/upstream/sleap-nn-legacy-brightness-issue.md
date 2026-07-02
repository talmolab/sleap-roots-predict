# Upstream issue: legacy config load rejects inert `brightness_min_val < 0`

**Target repo:** talmolab/sleap-nn (observed on v0.3.0)
**Filed:** https://github.com/talmolab/sleap-nn/issues/684

## Summary

Loading a legacy SLEAP model (`training_config.json` + `best_model.h5`, UNet)
fails when the legacy augmentation config has `brightness_min_val < 0`, even when
brightness augmentation is disabled (`brightness: false`) and the value is inert.

```
ValueError: 'brightness_min' must be >= 0: -10.0
```

## Root cause

`sleap_nn/config/data_config.py`, `data_mapper` (~lines 423-431) maps the legacy
`brightness_min_val` into the new `IntensityConfig.brightness_min` with only an
**upper** clamp:

```python
intensity_args["brightness_min"] = min(
    legacy_config_optimization["augmentation_config"]["brightness_min_val"], 2.0
)
```

`IntensityConfig.brightness_min` is validated `validators.ge(0)` (data_config.py
~line 100). New brightness is a **multiplicative** factor centered on 1.0, so
`>= 0` is required — but classic SLEAP's additive/imgaug brightness could hold
negative or placeholder values. `brightness_max` gets a symmetric `min(x, 2.0)`
upper clamp (to satisfy `le(2)`), but `brightness_min` has no lower clamp. attrs
validates the field regardless of `brightness_p`, so a disabled, inert value
still fails.

The repo's legacy fixtures all use `brightness_min_val: 0.0`, so the negative
case is untested.

## Impact

Real production models (e.g. sleap-roots root models) that were trained with
classic SLEAP and carry `brightness_min_val: -10.0` cannot be loaded for
inference under 0.3.0. The documented conversion path
(`TrainingJobConfig.load_sleap_config(...)` + `OmegaConf.save`) hits the same
mapper and fails identically.

## Suggested fix

Clamp the lower bound symmetric to the existing upper clamp in `data_mapper`:

```python
intensity_args["brightness_min"] = min(max(brightness_min_val, 0.0), 2.0)
```

Consider the same for other legacy min fields mapped into `>= 0`-validated
targets (contrast/scale/noise), and/or skip populating intensity fields when the
corresponding legacy enable flag is false. Add a regression test with a negative
`brightness_min_val`.

## Downstream workaround (in sleap-roots-predict)

`make_predictor` sanitizes the legacy `training_config.json` on a temporary copy
before load (clamps `brightness_min_val >= 0`); the original model dir is not
modified. This is behavior-preserving because augmentation does not run at
inference.
