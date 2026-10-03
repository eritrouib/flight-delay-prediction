# Model Card: Flight Departure Delay Classifier

*Generated automatically by `run_pipeline.py` on 03 October 2026. Do not edit by hand; re-run the pipeline.*

## Model details
- **Task:** binary classification. Will a flight depart 15+ minutes late?
- **Algorithm:** LightGBM gradient-boosted trees (145 trees, early-stopped on validation AUC).
- **Benchmark:** historical delay rate by origin airport and scheduled hour, the simplest credible alternative.
- **Version:** 0.1.0 (portfolio prototype, not a production system).

## Intended use
- **Intended:** illustrating how an airport could estimate delay risk for scheduled departures a day ahead, to support resourcing decisions (stand planning, staffing, passenger communications) and to prioritise attention, alongside human judgement.
- **Out of scope:** automated decisions affecting individual passengers or staff; compensation or liability decisions; any airport other than the three New York airports in the training data without retraining and revalidation.

## Data
- **Source:** `nycflights13` open dataset: all 328,521 operated departures from JFK, LGA and EWR in 2013, joined to hourly weather at the origin airport. No personal data is used.
- **Excluded:** 8,255 cancelled flights (cancellation is a different outcome needing a separate model).
- **Split (by time, never shuffled):** train Jan-Sep (245,723), validation Oct (28,653), test Nov-Dec (54,145).
- **Features (13):** only information available before departure: carrier, origin, destination, day of week, scheduled hour, distance, scheduled departures from the origin in that hour, and origin weather.

### Data quality checks
| Check | Result | Detail |
|---|---|---|
| `no_duplicate_flights` | PASS | 0 duplicate flight records |
| `target_is_binary` | PASS | values: [0, 1] |
| `sched_hour_in_range` | PASS | 0 rows outside 0-23 |
| `weather_join_coverage` | PASS | 99.5% of flights matched to an hourly weather record (min 95%) |
| `feature_missingness` | PASS | highest missing: pressure: 11.1%, wind_speed: 0.5%, humid: 0.5% (max 15%) |
| `no_leakage_features` | PASS | post-departure features used: none |

## Performance (held-out test set, Nov-Dec)
Test-period delay rate: 21.8%. Decision threshold 0.23, chosen on the validation month (max F1).

| Metric | Baseline | LightGBM |
|---|---|---|
| ROC AUC | 0.648 | 0.693 |
| PR AUC (avg precision) | 0.314 | 0.386 |
| Brier score (lower is better) | 0.164 | 0.157 |
| Precision @ threshold | 0.298 | 0.346 |
| Recall @ threshold | 0.651 | 0.598 |
| F1 @ threshold | 0.409 | 0.438 |

ROC AUC measures ranking ability across all thresholds; PR AUC is more informative when delays are the minority class; the Brier score measures calibration. See `outputs/figures/calibration.png`.

## Performance by segment
Large gaps between segments indicate where the model is less reliable, and where decisions based on it could treat some airports or airlines less fairly.

**By origin airport**

| origin | n_flights | actual_delay_rate | mean_predicted | roc_auc | calibration_gap |
|---|---|---|---|---|---|
| EWR | 19,071 | 0.253 | 0.256 | 0.666 | 0.003 |
| JFK | 17,637 | 0.196 | 0.208 | 0.682 | 0.012 |
| LGA | 17,437 | 0.203 | 0.196 | 0.727 | -0.006 |

**By carrier** (carriers with 500+ test flights)

| carrier | n_flights | actual_delay_rate | mean_predicted | roc_auc | calibration_gap |
|---|---|---|---|---|---|
| UA | 9,671 | 0.244 | 0.208 | 0.634 | -0.036 |
| B6 | 8,956 | 0.225 | 0.235 | 0.693 | 0.010 |
| EV | 8,298 | 0.293 | 0.338 | 0.666 | 0.045 |
| DL | 7,923 | 0.151 | 0.159 | 0.677 | 0.008 |
| AA | 5,157 | 0.158 | 0.171 | 0.676 | 0.013 |
| MQ | 4,009 | 0.203 | 0.213 | 0.697 | 0.011 |
| US | 3,212 | 0.121 | 0.129 | 0.714 | 0.007 |
| 9E | 3,119 | 0.245 | 0.252 | 0.695 | 0.007 |
| WN | 2,102 | 0.324 | 0.246 | 0.750 | -0.078 |
| VX | 911 | 0.139 | 0.186 | 0.716 | 0.047 |

Largest calibration gap: carrier `WN` (predicted 24.6% vs actual 32.4%). Where the model under-predicts, that carrier's delay risk would be systematically understated in any resourcing decision.

## Drift (training vs test period)
Population Stability Index per numeric feature (<0.1 stable, 0.1-0.25 moderate, >0.25 significant).
Features with PSI >= 0.1 (moderate or significant shift): `temp` (4.27), `pressure` (0.39). Under the monitoring rule below, this shift would already trigger a retraining review. The model learned temperature patterns from Jan-Sep and is now applying them to much colder months. PSI understates change for `precip`, which is almost always zero, so quantile bins collapse.

| feature | psi | status |
|---|---|---|
| temp | 4.268 | significant shift |
| pressure | 0.387 | significant shift |
| origin_hourly_departures | 0.057 | stable |
| wind_speed | 0.021 | stable |
| humid | 0.021 | stable |
| visib | 0.009 | stable |
| distance | 0.004 | stable |
| day_of_week | 0.004 | stable |
| sched_dep_hour | 0.002 | stable |
| precip | 0.000 | stable |

## Explainability
Top drivers of predicted delay risk by mean |SHAP|: `sched_dep_hour`, `carrier`, `humid`, `pressure`, `temp`. See `outputs/figures/shap_summary.png`.

## Limitations and risks
- **Weather leakage risk:** the model uses *observed* weather at the scheduled hour. In real use only a *forecast* would be available, so live performance would be lower. A production version should train on archived forecasts.
- **Missing drivers:** no inbound aircraft delay (reactionary delay is the largest cause of delay in practice), air traffic control restrictions, or ground-handling data.
- **Seasonality:** trained on Jan-Sep and tested on Nov-Dec. Holiday-period behaviour is unlike the training months, which the drift section quantifies.
- **Single year, single region:** 2013 New York data; patterns will not transfer to Heathrow without retraining on local data.
- **Threshold:** set by F1 as a neutral placeholder; an operational threshold should reflect the agreed cost of missed delays vs false alarms.

## Governance and monitoring (if deployed)
- Owner and approver named; model registered in an AI/model inventory with its risk tier.
- Monthly monitoring of AUC, calibration and feature PSI, with retraining triggered if any feature PSI > 0.25 or AUC falls below the baseline.
- Segment performance reviewed at each retrain; outputs used as decision support only, with a human accountable for operational decisions.
- Data protection: no personal data at present; a DPIA would be required before adding passenger- or staff-level data.
