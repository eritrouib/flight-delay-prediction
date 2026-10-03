"""Generate MODEL_CARD.md from the pipeline's actual results, so the documentation
can never drift out of date with the model it describes."""
from datetime import date
from pathlib import Path

import pandas as pd

from . import config, quality


def _table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, float) else f"{v:,}" if isinstance(v, int) else str(v)
                 for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_model_card(path: Path, *, metrics, qa, seg_origin, seg_carrier, drift, importance, sizes) -> None:
    b, m = metrics["baseline"], metrics["lightgbm"]
    metric_rows = pd.DataFrame([
        {"Metric": "ROC AUC", "Baseline": b["roc_auc"], "LightGBM": m["roc_auc"]},
        {"Metric": "PR AUC (avg precision)", "Baseline": b["pr_auc"], "LightGBM": m["pr_auc"]},
        {"Metric": "Brier score (lower is better)", "Baseline": b["brier"], "LightGBM": m["brier"]},
        {"Metric": "Precision @ threshold", "Baseline": b["precision"], "LightGBM": m["precision"]},
        {"Metric": "Recall @ threshold", "Baseline": b["recall"], "LightGBM": m["recall"]},
        {"Metric": "F1 @ threshold", "Baseline": b["f1"], "LightGBM": m["f1"]},
    ])
    top_features = ", ".join(f"`{f}`" for f in importance["feature"].head(5))
    shifted = drift[drift["psi"] >= 0.1]
    drift_note = (
        "Features with PSI >= 0.1 (moderate or significant shift): "
        + ", ".join(f"`{r.feature}` ({r.psi:.2f})" for r in shifted.itertuples())
        if not shifted.empty else "No feature shows PSI >= 0.1."
    )
    if (drift["psi"] > 0.25).any():
        drift_note += (". Under the monitoring rule below, this shift would already trigger a "
                       "retraining review.")
    if "temp" in set(shifted["feature"]):
        drift_note += (" The model learned temperature patterns from Jan-Sep and is now "
                       "applying them to much colder months.")
    drift_note += (" PSI understates change for `precip`, which is almost always zero, "
                   "so quantile bins collapse.")
    worst = seg_carrier.loc[seg_carrier["calibration_gap"].abs().idxmax()]
    seg_note = (
        f"Largest calibration gap: carrier `{worst['carrier']}` (predicted "
        f"{worst['mean_predicted']:.1%} vs actual {worst['actual_delay_rate']:.1%}). "
        "Where the model under-predicts, that carrier's delay risk would be systematically "
        "understated in any resourcing decision."
    )

    card = f"""# Model Card: Flight Departure Delay Classifier

*Generated automatically by `run_pipeline.py` on {date.today():%d %B %Y}. Do not edit by hand; re-run the pipeline.*

## Model details
- **Task:** binary classification. Will a flight depart {config.DELAY_THRESHOLD_MIN}+ minutes late?
- **Algorithm:** LightGBM gradient-boosted trees ({metrics['best_iteration']} trees, early-stopped on validation AUC).
- **Benchmark:** historical delay rate by origin airport and scheduled hour, the simplest credible alternative.
- **Version:** 0.1.0 (portfolio prototype, not a production system).

## Intended use
- **Intended:** illustrating how an airport could estimate delay risk for scheduled departures a day ahead, to support resourcing decisions (stand planning, staffing, passenger communications) and to prioritise attention, alongside human judgement.
- **Out of scope:** automated decisions affecting individual passengers or staff; compensation or liability decisions; any airport other than the three New York airports in the training data without retraining and revalidation.

## Data
- **Source:** `nycflights13` open dataset: all {sizes['train'] + sizes['valid'] + sizes['test']:,} operated departures from JFK, LGA and EWR in 2013, joined to hourly weather at the origin airport. No personal data is used.
- **Excluded:** {sizes['cancelled_excluded']:,} cancelled flights (cancellation is a different outcome needing a separate model).
- **Split (by time, never shuffled):** train Jan-Sep ({sizes['train']:,}), validation Oct ({sizes['valid']:,}), test Nov-Dec ({sizes['test']:,}).
- **Features ({len(config.FEATURES)}):** only information available before departure: carrier, origin, destination, day of week, scheduled hour, distance, scheduled departures from the origin in that hour, and origin weather.

### Data quality checks
{quality.to_markdown(qa)}

## Performance (held-out test set, Nov-Dec)
Test-period delay rate: {metrics['test_delay_rate']:.1%}. Decision threshold {metrics['threshold']:.2f}, chosen on the validation month (max F1).

{_table(metric_rows)}

ROC AUC measures ranking ability across all thresholds; PR AUC is more informative when delays are the minority class; the Brier score measures calibration. See `outputs/figures/calibration.png`.

## Performance by segment
Large gaps between segments indicate where the model is less reliable, and where decisions based on it could treat some airports or airlines less fairly.

**By origin airport**

{_table(seg_origin)}

**By carrier** (carriers with 500+ test flights)

{_table(seg_carrier)}

{seg_note}

## Drift (training vs test period)
Population Stability Index per numeric feature (<0.1 stable, 0.1-0.25 moderate, >0.25 significant).
{drift_note}

{_table(drift)}

## Explainability
Top drivers of predicted delay risk by mean |SHAP|: {top_features}. See `outputs/figures/shap_summary.png`.

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
"""
    path.write_text(card)
