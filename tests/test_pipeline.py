"""Unit tests on small synthetic data: fast, and independent of the full dataset."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from flightdelay import config, data, evaluate, models, quality  # noqa: E402


def make_flights(n=40):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "year": 2013, "month": rng.integers(1, 13, n), "day": rng.integers(1, 28, n),
        "sched_dep_time": rng.choice([600, 930, 1745, 2100], n),
        "dep_delay": np.r_[rng.integers(-10, 60, n - 2), [np.nan, np.nan]],
        "carrier": rng.choice(["AA", "UA"], n), "flight": np.arange(n),
        "origin": rng.choice(["JFK", "LGA"], n), "dest": "BOS", "distance": 200,
        "time_hour": "2013-01-01T06:00:00Z",
        # post-departure columns that must never become features
        "dep_time": 0, "arr_time": 0, "arr_delay": 0, "air_time": 0,
    })


def make_weather():
    return pd.DataFrame({
        "origin": ["JFK", "LGA"], "time_hour": "2013-01-01T06:00:00Z",
        "temp": [40.0, 41.0], "humid": [60.0, 61.0], "wind_speed": [10.0, 9.0],
        "precip": [0.0, 0.0], "visib": [10.0, 10.0], "pressure": [1012.0, 1013.0],
    })


def test_cancelled_flights_excluded_and_target_correct():
    df = data.build_dataset(make_flights(), make_weather())
    assert len(df) == 38
    assert (df[config.TARGET] == (df["dep_delay"] >= config.DELAY_THRESHOLD_MIN)).all()


def test_scheduled_hour_derived():
    df = data.build_dataset(make_flights(), make_weather())
    assert set(df["sched_dep_hour"]) <= {6, 9, 17, 21}


def test_congestion_counts_include_cancelled_flights():
    flights = make_flights()
    df = data.build_dataset(flights, make_weather())
    expected = flights.groupby("origin").size()
    for origin, n in expected.items():
        assert (df.loc[df["origin"] == origin, "origin_hourly_departures"] == n).all()


def test_time_split_has_no_overlap():
    df = data.build_dataset(make_flights(200), make_weather())
    train, valid, test = data.time_split(df)
    assert train["month"].max() < valid["month"].min() <= valid["month"].max() < test["month"].min()


def test_no_leakage_features_configured():
    assert quality.check_no_leakage_features().passed
    assert not quality.check_no_leakage_features(config.FEATURES + ["arr_delay"]).passed


def test_unseen_category_becomes_missing():
    train = pd.DataFrame({c: [1.0] for c in config.NUMERIC_FEATURES} | {"carrier": ["AA"], "origin": ["JFK"], "dest": ["BOS"]})
    new = train.assign(carrier="ZZ")
    _, cats = models.prepare_features(train)
    X_new, _ = models.prepare_features(new, cats)
    assert X_new["carrier"].isna().all()


def test_psi_zero_for_identical_and_large_for_shifted():
    rng = np.random.default_rng(1)
    a = pd.Series(rng.normal(0, 1, 5000))
    assert evaluate.psi(a, a) < 1e-9
    assert evaluate.psi(a, a + 3) > 0.25


def test_baseline_falls_back_to_overall_rate():
    train = pd.DataFrame({"origin": ["JFK", "JFK"], "sched_dep_hour": [6, 6], "delayed": [0, 1]})
    b = models.HistoricalRateBaseline().fit(train)
    p = b.predict_proba(pd.DataFrame({"origin": ["EWR"], "sched_dep_hour": [6]}))
    assert p[0] == 0.5
