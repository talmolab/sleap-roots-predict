## ADDED Requirements

### Requirement: Past-window scan warning in batch prediction-and-write

`predict_and_write_batch` SHALL log exactly one warning for each request whose selection was
clamped to its species' window maximum, as reported by `model_selection.past_window_age(
req.params, catalog, req.overrides)`, after resolving the request's models and before predicting
it. The warning SHALL name the scan key, species, mode, the scan age and the matching age, and
SHALL be logged by the `sleap_roots_predict.output_contract` logger. It SHALL NOT change the
returned manifests or the written outputs.

#### Scenario: A clamped request warns once

- **WHEN** `predict_and_write_batch` writes a rice, cylinder, day-9 request against a rice 2–5
  catalog
- **THEN** exactly one warning from `sleap_roots_predict.output_contract` names the scan key, rice,
  cylinder, 9 and 5

#### Scenario: In-window or fully overridden requests do not warn

- **WHEN** a request is in-window (rice day 3), or overrides every root type present among the
  catalog's cards
- **THEN** no past-window warning is logged
