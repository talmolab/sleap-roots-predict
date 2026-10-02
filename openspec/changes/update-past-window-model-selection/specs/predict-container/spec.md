## ADDED Requirements

### Requirement: Past-window scan warning

`run_batch` SHALL log exactly one warning for each scan it predicts whose selection was clamped
to its species' window maximum, as reported by `model_selection.past_window_age(scan.params,
catalog)` (`run_batch` passes no overrides). The warning SHALL name the scan key, species, mode,
the scan age and the matching age, and SHALL be logged by the `sleap_roots_predict.batch` logger
after the resume check and before prediction. A scan skipped on resume (idempotency key
unchanged), a scan whose model resolution raises, and an in-window scan SHALL NOT log it. The warning SHALL NOT
change the scan's status, outputs or exit code: a clamped scan that predicts successfully is `ok`.

#### Scenario: A clamped scan predicts and warns once

- **WHEN** `run_batch` processes a rice, cylinder, day-9 scan against a rice 2–5 catalog
- **THEN** the scan ends `ok`, its copied sidecar keeps `age: 9`, and exactly one warning from
  `sleap_roots_predict.batch` names the scan key, rice, cylinder, 9 and 5

#### Scenario: A resumed clamped scan is skipped without a warning

- **WHEN** the same day-9 batch is run again with its outputs already present
- **THEN** the scan is `skipped` and no past-window warning is logged

#### Scenario: An in-window scan does not warn

- **WHEN** `run_batch` predicts a rice, cylinder, day-3 scan against a rice 2–5 catalog
- **THEN** no past-window warning is logged
