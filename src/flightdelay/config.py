"""Central configuration: one place to change thresholds, splits and features."""

# A flight is "delayed" if it departs 15+ minutes late (the industry/CAA on-time standard).
DELAY_THRESHOLD_MIN = 15

# Time-based split (months, inclusive). Flights are never shuffled across time:
# a random split leaks information about the same day's disruption into training.
TRAIN_MONTHS = range(1, 10)   # Jan-Sep
VALID_MONTHS = range(10, 11)  # Oct (early stopping and threshold choice)
TEST_MONTHS = range(11, 13)   # Nov-Dec (held out, reported once)

# Only features knowable BEFORE departure. Actual departure/arrival times,
# air time and arrival delay are excluded to prevent target leakage.
# "month" is deliberately excluded: with a time-based split the test months are
# never seen in training, so the model would be extrapolating on that feature.
CATEGORICAL_FEATURES = ["carrier", "origin", "dest"]
NUMERIC_FEATURES = [
    "day_of_week", "sched_dep_hour", "distance",
    "origin_hourly_departures",
    "temp", "humid", "wind_speed", "precip", "visib", "pressure",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "delayed"

RANDOM_STATE = 42
