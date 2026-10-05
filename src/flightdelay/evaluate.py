"""Evaluation: headline metrics, calibration, performance by segment and drift."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score, brier_score_loss, f1_score, precision_score,
    recall_score, roc_auc_score,
)


def headline_metrics(y_true, y_prob, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def choose_threshold(y_true, y_prob) -> float:
    """Pick the threshold that maximises F1 on the validation set.

    In practice the threshold should come from the operational cost of a missed
    delay versus a false alarm; F1 is a neutral placeholder until that is agreed.
    """
    grid = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y_true, (y_prob >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def segment_performance(df: pd.DataFrame, y_prob, segment: str, min_n: int = 500) -> pd.DataFrame:
    """Compare performance across groups. Large gaps flag segments where the model
    is less reliable and decisions based on it may be less fair."""
    tmp = df[[segment, "delayed"]].assign(prob=y_prob)
    rows = []
    for name, g in tmp.groupby(segment):
        if len(g) < min_n or g["delayed"].nunique() < 2:
            continue
        rows.append({
            segment: name,
            "n_flights": len(g),
            "actual_delay_rate": g["delayed"].mean(),
            "mean_predicted": g["prob"].mean(),
            "roc_auc": roc_auc_score(g["delayed"], g["prob"]),
        })
    out = pd.DataFrame(rows).sort_values("n_flights", ascending=False)
    out["calibration_gap"] = out["mean_predicted"] - out["actual_delay_rate"]
    return out.reset_index(drop=True)


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population Stability Index between training and new data.

    Rule of thumb: <0.1 stable, 0.1-0.25 moderate shift, >0.25 significant shift.
    Note: for zero-inflated features (e.g. precipitation) quantile bins collapse and
    PSI understates change; compare the share of non-zero values instead.
    """
    expected, actual = expected.dropna(), actual.dropna()
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = [{"feature": f, "psi": psi(train[f], test[f])} for f in features]
    out = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    out["status"] = pd.cut(out["psi"], [-np.inf, 0.1, 0.25, np.inf],
                           labels=["stable", "moderate shift", "significant shift"])
    return out


def plot_calibration(y_true, probs: dict, path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfectly calibrated")
    for name, p in probs.items():
        frac, mean = calibration_curve(y_true, p, n_bins=10, strategy="quantile")
        ax.plot(mean, frac, marker="o", label=name)
    ax.set_xlabel("Predicted probability of delay")
    ax.set_ylabel("Observed delay rate")
    ax.set_title("Calibration on held-out test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def route_summary(df: pd.DataFrame, y_prob, min_n_auc: int = 300) -> pd.DataFrame:
    """Per-route (origin -> destination) test performance, used by the web map.

    AUC is only reported where a route has enough flights and both outcomes,
    because AUC on a handful of flights is noise rather than evidence.
    """
    tmp = df[["origin", "dest", "delayed"]].assign(prob=y_prob)
    rows = []
    for (o, d), g in tmp.groupby(["origin", "dest"]):
        enough = len(g) >= min_n_auc and g["delayed"].nunique() == 2
        rows.append({
            "origin": o, "dest": d, "n_flights": len(g),
            "actual_delay_rate": g["delayed"].mean(),
            "mean_predicted": g["prob"].mean(),
            "roc_auc": roc_auc_score(g["delayed"], g["prob"]) if enough else None,
        })
    out = pd.DataFrame(rows)
    out["calibration_gap"] = out["mean_predicted"] - out["actual_delay_rate"]
    return out
