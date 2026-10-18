# The Training Ground — scorecard

Experiment: **origin-price-pilot-evidence**. Status: **UNVERIFIED**.

Hypothesis: Consolidate preserved matched origin-price decisions, forecast diagnostics and settlement evidence; reporting only, no new model evaluation or promotion.

Candidate: Preserved candidate; modern capture-price policy; control: Preserved control; modern capture-price policy.

Metric: `actual_net_points`; gameweeks: [141, 142, 143, 144].

| Measured outcome | Value |
| --- | ---: |
| Candidate points | 203.000 |
| Control points | 197.000 |
| Difference | +6.000 |
| Paired 95% interval | [-8.000, +11.000] |
| Positive bootstrap fraction | 0.499 |
| One-sided sign-test p-value | 0.68750 |

The bootstrap fraction is not a probability of future superiority. The sign test assumes independent signs; block choice affects the interval.

Comparability: **NOT ASSESSED**. Contract validation checks bytes and declared assumptions only.

Forecast diagnostics: **MEASURED** (forecast_player_round_v1).
Population: Preserved original pilot forecast rows including any rows excluded later by origin registration; diagnostic coverage is not the admitted decision pool..
Protocol: origin; declared grain: fixture. Candidate/control coverage matches exactly.
Forecast-to-decision input linkage: **NOT ASSESSED**. A hash-bound declaration alone does not prove how decisions were produced.

**Origin forecasts: lead 0 (target GW minus origin GW)**

| Diagnostic | Candidate | Control |
| --- | ---: | ---: |
| Player-round forecasts | 2843.0000 | 2843.0000 |
| Fixture rows | 2928.0000 | 2928.0000 |
| MAE | 0.8938 | 0.7408 |
| RMSE | 1.7968 | 1.9069 |
| Bias (prediction minus actual) | 0.0558 | -0.4204 |
| Aggregate calibration ratio | 1.0564 | 0.5752 |
| Within-group Spearman | 0.6873 | 0.6567 |
| Top-ranked player capture ratio | 0.3365 | 0.3317 |
| Spearman groups used | 16.0000 | 16.0000 |
| Spearman groups skipped | 0.0000 | 0.0000 |
| Groups with tied top forecasts | 0.0000 | 0.0000 |

| Position | Candidate rows | Control rows | Candidate MAE | Control MAE |
| --- | ---: | ---: | ---: | ---: |
| DEF | 918 | 918 | 0.9406 | 0.7697 |
| FWD | 373 | 373 | 1.0038 | 0.8523 |
| GK | 324 | 324 | 0.6115 | 0.5244 |
| MID | 1228 | 1228 | 0.8999 | 0.7424 |

**Origin forecasts: lead 1 (target GW minus origin GW)**

| Diagnostic | Candidate | Control |
| --- | ---: | ---: |
| Player-round forecasts | 2009.0000 | 2009.0000 |
| Fixture rows | 2093.0000 | 2093.0000 |
| MAE | 0.9225 | 0.7488 |
| RMSE | 1.8317 | 1.9042 |
| Bias (prediction minus actual) | 0.0799 | -0.3977 |
| Aggregate calibration ratio | 1.0824 | 0.5896 |
| Within-group Spearman | 0.6595 | 0.6242 |
| Top-ranked player capture ratio | 0.3117 | 0.2792 |
| Spearman groups used | 12.0000 | 12.0000 |
| Spearman groups skipped | 0.0000 | 0.0000 |
| Groups with tied top forecasts | 0.0000 | 0.0000 |

| Position | Candidate rows | Control rows | Candidate MAE | Control MAE |
| --- | ---: | ---: | ---: | ---: |
| DEF | 649 | 649 | 0.9383 | 0.7423 |
| FWD | 265 | 265 | 1.1102 | 0.9437 |
| GK | 230 | 230 | 0.5873 | 0.4914 |
| MID | 865 | 865 | 0.9422 | 0.7625 |

**Origin forecasts: lead 2 (target GW minus origin GW)**

| Diagnostic | Candidate | Control |
| --- | ---: | ---: |
| Player-round forecasts | 1174.0000 | 1174.0000 |
| Fixture rows | 1174.0000 | 1174.0000 |
| MAE | 0.9404 | 0.7633 |
| RMSE | 1.9045 | 1.9433 |
| Bias (prediction minus actual) | 0.0808 | -0.3852 |
| Aggregate calibration ratio | 1.0866 | 0.5874 |
| Within-group Spearman | 0.6398 | 0.5915 |
| Top-ranked player capture ratio | 0.2100 | 0.3200 |
| Spearman groups used | 8.0000 | 8.0000 |
| Spearman groups skipped | 0.0000 | 0.0000 |
| Groups with tied top forecasts | 0.0000 | 0.0000 |

| Position | Candidate rows | Control rows | Candidate MAE | Control MAE |
| --- | ---: | ---: | ---: | ---: |
| DEF | 380 | 380 | 0.9825 | 0.7907 |
| FWD | 156 | 156 | 1.1298 | 0.9413 |
| GK | 135 | 135 | 0.5640 | 0.4796 |
| MID | 503 | 503 | 0.9509 | 0.7635 |

MASE: **NOT MEASURED**; training-only scales and refit cutoffs were not supplied. Original model coverage: **NOT ASSESSED**.
Top-ranked ties use the lowest stable player code. Capture is a ranking diagnostic, not realized captaincy points. Zero-denominator ratios are undefined; negative outcomes can produce ratios outside [0, 1]. Origin leads are reported separately, without pooling repeated target outcomes.

## Saved decision and settlement evidence

Both strict replay bundles were hash-verified at report generation. This does not validate their planning inputs, native certificates or full site rules.

| Measured settlement | Candidate | Control |
| --- | ---: | ---: |
| Gross points | 203 | 197 |
| Transfer hit points | 0 | 0 |
| Net points | 203 | 197 |
| Recorded decision rounds | 4 | 4 |
| Recorded incoming players | 18 | 18 |
| Recorded outgoing players | 3 | 3 |

Incoming players include initial squad construction when captured as transfers. Decision rounds are not an attempted-solve denominator.

| Evidence gate | Assessment |
| --- | --- |
| fixture causality | NOT ASSESSED |
| forecast blank planning | NOT ASSESSED |
| full outcome blind execution | NOT ASSESSED |
| independent legality | NOT ASSESSED |
| native certificate | NOT ASSESSED |
| price information validity | NOT ASSESSED |

| Recorded claim | Candidate | Control |
| --- | --- | --- |
| Solver status | Optimal (4 rounds) | Optimal (4 rounds) |
| Relative solver gap | 0.0 (4 rounds) | 0.0 (4 rounds) |
| Solver seconds per recorded window | 1.88345375, 3.79748275, 1.248889792, 0.2224935 | 2.066008166, 3.982602542, 1.610680333, 0.289626583 |
| Planning price policy | origin_capture_half_profit_static_horizon_v1 (4 rounds) | origin_capture_half_profit_static_horizon_v1 (4 rounds) |
| Player pool policy | registered_forecast_rows (4 rounds) | registered_forecast_rows (4 rounds) |
| Solver outcome columns | ABSENT (4 rounds) | ABSENT (4 rounds) |

**Candidate recorded evidence**

Chip usage: none recorded.
Receipt SHA256: `a06690b30281cf50edf0fc4e9891fddfe3b5f4c277fbf3cab278569faf5fc338`.
Snapshot SHA256: `27983afbb44dc40df876141ff997c826896edb85f3545c96256641e7fcb2a1cc`.

**Control recorded evidence**

Chip usage: none recorded.
Receipt SHA256: `1cd7c3c790885fbcd211d8b59981f3d67b52163eecda4676eb6bd20c5482f4eb`.
Snapshot SHA256: `60d593798d4d340a6d7dff9644edd37dbdb3ce7b4d86e75ad9ccd05eb0e8f745`.

Recorded Optimal status and zero gap do not substitute for exported native certificates. Settlement blank evidence does not validate zero-filled planning forecasts. Original fit runtime, attempted solves and failure rate remain NOT MEASURED. Source replay bundles are required to audit the captured receipts again.

Probabilistic diagnostics: **NOT MEASURED**. Native optimality, independent legality and causal validity: **NOT ASSESSED**.

Reporting invocation: 0.007 seconds. Original fit runtime and failure rate: **NOT MEASURED**.

**Promotion: NOT ASSESSED.** Saved-outcome reanalysis only; full registered promotion evidence has not been assessed.

[Machine-readable scorecard](scorecard.json) · [Comparison receipt](comparison.json)
