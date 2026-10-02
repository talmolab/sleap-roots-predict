## ADDED Requirements

### Requirement: Past-window scan warning in batch prediction-and-write

`predict_and_write_batch` SHALL log exactly one warning for each request whose selection was
clamped to its species' window maximum, as reported by `model_selection.past_window_age(
req.params, catalog, req.overrides)`, after that request's prediction and outputs are written.
The warning SHALL be logged by the `sleap_roots_predict.output_contract` logger with the same
message as `run_batch`'s past-window warning (`past-window age: scan_key=<key>
species=<species!r> mode=<mode!r> age=<scan age> matched as age=<matching age>`). A request whose
model resolution, prediction or output writing raises SHALL NOT log it. The warning SHALL NOT
change the returned manifests or the written outputs.

#### Scenario: A clamped request warns once

- **WHEN** `predict_and_write_batch` writes a rice, cylinder, day-9 request against a rice 2–5
  catalog
- **THEN** exactly one warning from `sleap_roots_predict.output_contract` reads `past-window age:
  scan_key=<key> species='rice' mode='cylinder' age=9 matched as age=5`

#### Scenario: A clamped request that raises does not warn

- **WHEN** a clamped request's model resolution raises (two cards match at the matching age) or
  its prediction raises
- **THEN** the error propagates and no past-window warning is logged

#### Scenario: In-window or fully overridden requests do not warn

- **WHEN** a request is in-window (rice day 3), or overrides every root type present among the
  catalog's cards
- **THEN** no past-window warning is logged
