"""End-to-end pipeline: data -> QA -> train -> evaluate -> explain -> model card.

Usage:  python run_pipeline.py
"""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from flightdelay import config, data, evaluate, explain, models, quality, report  # noqa: E402

# Silence noisy library deprecation notices so the QA log stays readable.
warnings.filterwarnings("ignore", category=UserWarning, module="shap")
warnings.filterwarnings("ignore", message=".*eval_set.*")

OUT = Path("outputs")
FIG = OUT / "figures"


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)

    # 1. Data
    print("Loading and building dataset...")
    flights, weather = data.load_raw()
    df = data.build_dataset(flights, weather)
    n_cancelled = int(flights["dep_delay"].isna().sum())

    # 2. Quality assurance - stop on any critical failure
    qa = quality.run_all(df)
    for r in qa:
        print(f"  [{'PASS' if r.passed else 'FAIL'}] {r.name}: {r.detail}")
    if any(r.critical and not r.passed for r in qa):
        sys.exit("Critical data quality check failed - stopping.")

    # 3. Time-based split and features
    train, valid, test = data.time_split(df)
    X_train, cats = models.prepare_features(train)
    X_valid, _ = models.prepare_features(valid, cats)
    X_test, _ = models.prepare_features(test, cats)
    y_train, y_valid, y_test = (d[config.TARGET] for d in (train, valid, test))

    # 4. Models
    print("Training baseline and LightGBM...")
    baseline = models.HistoricalRateBaseline().fit(train)
    lgbm = models.train_lgbm(X_train, y_train, X_valid, y_valid)

    p_valid = lgbm.predict_proba(X_valid)[:, 1]
    threshold = evaluate.choose_threshold(y_valid, p_valid)

    p_test_base = baseline.predict_proba(test)
    p_test = lgbm.predict_proba(X_test)[:, 1]
    base_valid = baseline.predict_proba(valid)
    base_threshold = evaluate.choose_threshold(y_valid, base_valid)

    metrics = {
        "baseline": evaluate.headline_metrics(y_test, p_test_base, base_threshold),
        "lightgbm": evaluate.headline_metrics(y_test, p_test, threshold),
        "threshold": threshold,
        "best_iteration": int(lgbm.best_iteration_),
        "test_delay_rate": float(y_test.mean()),
    }

    # 5. Responsible-ML checks
    seg_origin = evaluate.segment_performance(test, p_test, "origin")
    seg_carrier = evaluate.segment_performance(test, p_test, "carrier")
    drift = evaluate.drift_report(train, test, config.NUMERIC_FEATURES)
    evaluate.plot_calibration(
        y_test, {"Baseline": p_test_base, "LightGBM": p_test}, FIG / "calibration.png"
    )

    # 6. Explainability
    print("Computing SHAP values...")
    importance = explain.shap_importance(lgbm, X_test, FIG / "shap_summary.png")

    # 7. Outputs
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    seg_origin.to_csv(OUT / "segment_origin.csv", index=False)
    seg_carrier.to_csv(OUT / "segment_carrier.csv", index=False)
    drift.to_csv(OUT / "drift_psi.csv", index=False)
    importance.to_csv(OUT / "shap_importance.csv", index=False)

    report.write_model_card(
        Path("MODEL_CARD.md"),
        metrics=metrics, qa=qa, seg_origin=seg_origin, seg_carrier=seg_carrier,
        drift=drift, importance=importance,
        sizes={"train": len(train), "valid": len(valid), "test": len(test),
               "cancelled_excluded": n_cancelled},
    )

    b, m = metrics["baseline"], metrics["lightgbm"]
    print(f"\nTest ROC AUC  baseline {b['roc_auc']:.3f} | LightGBM {m['roc_auc']:.3f}")
    print(f"Test PR AUC   baseline {b['pr_auc']:.3f} | LightGBM {m['pr_auc']:.3f}")
    print("Model card written to MODEL_CARD.md")


if __name__ == "__main__":
    main()
