"""Data quality checks.

Each check returns a result object rather than failing silently, so the pipeline
can write a QA report and stop on critical failures.
"""
from dataclasses import dataclass

import pandas as pd

from . import config

LEAKY_COLUMNS = {"dep_time", "dep_delay", "arr_time", "arr_delay", "air_time"}


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    critical: bool = True


def check_no_duplicate_flights(df: pd.DataFrame) -> CheckResult:
    keys = ["date", "carrier", "flight", "origin", "sched_dep_time"]
    n = int(df.duplicated(keys).sum())
    return CheckResult("no_duplicate_flights", n == 0, f"{n} duplicate flight records")


def check_target_binary(df: pd.DataFrame) -> CheckResult:
    values = sorted(int(v) for v in df[config.TARGET].unique())
    return CheckResult("target_is_binary", set(values) <= {0, 1}, f"values: {values}")


def check_scheduled_hour_range(df: pd.DataFrame) -> CheckResult:
    bad = int((~df["sched_dep_hour"].between(0, 23)).sum())
    return CheckResult("sched_hour_in_range", bad == 0, f"{bad} rows outside 0-23")


def check_weather_coverage(df: pd.DataFrame, min_coverage: float = 0.95) -> CheckResult:
    cov = float(df["temp"].notna().mean())
    return CheckResult(
        "weather_join_coverage", cov >= min_coverage,
        f"{cov:.1%} of flights matched to an hourly weather record (min {min_coverage:.0%})",
    )


def check_missingness(df: pd.DataFrame, max_missing: float = 0.15) -> CheckResult:
    miss = df[config.FEATURES].isna().mean().sort_values(ascending=False)
    detail = ", ".join(f"{c}: {v:.1%}" for c, v in miss.head(3).items())
    return CheckResult(
        "feature_missingness", bool((miss <= max_missing).all()),
        f"highest missing: {detail} (max {max_missing:.0%})", critical=False,
    )


def check_no_leakage_features(features: list[str] = config.FEATURES) -> CheckResult:
    found = sorted(LEAKY_COLUMNS & set(features))
    return CheckResult("no_leakage_features", not found, f"post-departure features used: {found or 'none'}")


def run_all(df: pd.DataFrame) -> list[CheckResult]:
    return [
        check_no_duplicate_flights(df),
        check_target_binary(df),
        check_scheduled_hour_range(df),
        check_weather_coverage(df),
        check_missingness(df),
        check_no_leakage_features(),
    ]


def to_markdown(results: list[CheckResult]) -> str:
    lines = ["| Check | Result | Detail |", "|---|---|---|"]
    for r in results:
        status = "PASS" if r.passed else ("FAIL" if r.critical else "WARN")
        lines.append(f"| `{r.name}` | {status} | {r.detail} |")
    return "\n".join(lines)
