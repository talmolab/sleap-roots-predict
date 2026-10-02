## ADDED Requirements

### Requirement: Past-window scan warning

`run_batch` SHALL log exactly one warning for each scan it predicts successfully whose selection
was clamped to its species' window maximum, as reported by `model_selection.past_window_age(
scan.params, catalog)` (`run_batch` passes no overrides). The warning SHALL be logged by the
`sleap_roots_predict.batch` logger after the scan's prediction and outputs succeed, with the
message `past-window age: scan_key=<key> species=<species!r> mode=<mode!r> age=<scan age> matched
as age=<matching age>` (the prefix shared with the trait extractor's warning, so one search finds
both). A scan skipped on resume (idempotency key unchanged), a scan whose model resolution raises,
a scan whose prediction or output writing fails, and an in-window scan SHALL NOT log it, so a
failed clamped scan re-run later is not warned about until it succeeds. The warning SHALL NOT
change the scan's status, outputs or exit code: a clamped scan that predicts successfully is `ok`.

#### Scenario: A clamped scan predicts and warns once

- **WHEN** `run_batch` processes a rice, cylinder, day-9 scan against a rice 2–5 catalog
- **THEN** the scan ends `ok`, its copied sidecar keeps `age: 9`, and exactly one warning from
  `sleap_roots_predict.batch` reads `past-window age: scan_key=<key> species='rice'
  mode='cylinder' age=9 matched as age=5`

#### Scenario: A resumed clamped scan is skipped without a warning

- **WHEN** the same day-9 batch is run again with its outputs already present
- **THEN** the scan is `skipped` and no past-window warning is logged

#### Scenario: A clamped scan that fails does not warn

- **WHEN** a clamped scan fails after resolution (e.g. it has no image frames), or its model
  resolution raises (e.g. two cards match at the matching age)
- **THEN** the scan is `failed` and no past-window warning is logged for it

#### Scenario: An in-window scan does not warn

- **WHEN** `run_batch` predicts a rice, cylinder, day-3 scan against a rice 2–5 catalog
- **THEN** no past-window warning is logged
