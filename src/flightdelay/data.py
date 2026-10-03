"""Load and assemble the modelling dataset from the nycflights13 open data."""
import pandas as pd
import nycflights13

from . import config


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return raw flights and hourly weather tables (New York airports, 2013)."""
    return nycflights13.flights.copy(), nycflights13.weather.copy()


def build_dataset(flights: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Create one row per operated flight with the target and pre-departure features.

    Cancelled flights (no departure delay recorded) are excluded: cancellation is a
    different outcome and would need its own model. This is documented in the model card.
    """
    df = flights.dropna(subset=["dep_delay"]).copy()

    df[config.TARGET] = (df["dep_delay"] >= config.DELAY_THRESHOLD_MIN).astype(int)
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    df["day_of_week"] = df["date"].dt.dayofweek
    df["sched_dep_hour"] = df["sched_dep_time"] // 100

    # Scheduled congestion: departures planned from this airport in this hour.
    # Known from the timetable in advance, so it is a legitimate feature.
    # Counted on the full timetable (including later-cancelled flights).
    sched_counts = (
        flights.groupby(["origin", "time_hour"]).size().rename("origin_hourly_departures")
    )
    df = df.join(sched_counts, on=["origin", "time_hour"])

    wx_cols = ["origin", "time_hour", "temp", "humid", "wind_speed", "precip", "visib", "pressure"]
    df = df.merge(weather[wx_cols], on=["origin", "time_hour"], how="left", validate="m:1")

    return df.reset_index(drop=True)


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by calendar month so the model is always evaluated on the future."""
    train = df[df["month"].isin(config.TRAIN_MONTHS)]
    valid = df[df["month"].isin(config.VALID_MONTHS)]
    test = df[df["month"].isin(config.TEST_MONTHS)]
    return train, valid, test
