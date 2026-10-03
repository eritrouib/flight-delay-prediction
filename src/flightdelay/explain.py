"""Global explainability with SHAP."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from . import config


def shap_importance(model, X: pd.DataFrame, path, sample: int = 5000) -> pd.DataFrame:
    """Compute mean |SHAP| per feature on a sample and save a summary plot."""
    Xs = X.sample(min(sample, len(X)), random_state=config.RANDOM_STATE)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(Xs)
    if isinstance(values, list):  # older SHAP returns one array per class
        values = values[1]

    shap.summary_plot(values, Xs, show=False, max_display=15)
    plt.title("What drives predicted delay risk (SHAP)")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    return (
        pd.DataFrame({"feature": X.columns, "mean_abs_shap": np.abs(values).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
