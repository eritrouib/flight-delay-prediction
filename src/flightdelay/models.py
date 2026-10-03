"""Baseline and gradient-boosted models."""
import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config


class HistoricalRateBaseline:
    """Predicts the training-period delay rate for each (origin, scheduled hour).

    This is the "what would a sensible analyst do in a spreadsheet" benchmark.
    A model is only worth deploying if it clearly beats this.
    """

    def __init__(self, keys: tuple[str, ...] = ("origin", "sched_dep_hour")):
        self.keys = list(keys)

    def fit(self, df: pd.DataFrame) -> "HistoricalRateBaseline":
        self.overall_ = df[config.TARGET].mean()
        self.rates_ = df.groupby(self.keys)[config.TARGET].mean().rename("rate")
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        rates = df[self.keys].join(self.rates_, on=self.keys)["rate"]
        return rates.fillna(self.overall_).to_numpy()


def prepare_features(df: pd.DataFrame, categories: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Select model features and fix categorical levels to those seen in training,
    so unseen categories at prediction time are handled consistently (as missing)."""
    X = df[config.FEATURES].copy()
    if categories is None:
        categories = {c: sorted(X[c].dropna().unique()) for c in config.CATEGORICAL_FEATURES}
    for c in config.CATEGORICAL_FEATURES:
        known = X[c].where(X[c].isin(categories[c]))  # unseen levels -> missing
        X[c] = pd.Categorical(known, categories=categories[c])
    return X, categories


def train_lgbm(X_train, y_train, X_valid, y_valid) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=config.RANDOM_STATE,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model
