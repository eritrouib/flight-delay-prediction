"""Build the data file for the interactive route map in docs/.

Run after run_pipeline.py:   python build_webmap.py
Then open docs/index.html in a browser (no server needed), or publish the docs/
folder with GitHub Pages.

The data is written as a .js file (JSON assigned to a variable) rather than .json,
because browsers block loading .json from a page opened straight from disk.
"""
import json
from datetime import date
from pathlib import Path

import nycflights13
import pandas as pd

OUT = Path("outputs")
DOCS = Path("docs")

# Below this many test flights a route is drawn but not coloured: rates on a handful
# of flights are noise, and colouring them would invite false conclusions.
MIN_FLIGHTS = 50

# nycflights13's airport table has no entries for Puerto Rico and the US Virgin
# Islands. Coordinates added by hand; recorded here so the provenance is explicit.
SUPPLEMENTARY_AIRPORTS = {
    "BQN": ("Rafael Hernandez Airport, Aguadilla", 18.4949, -67.1294),
    "PSE": ("Mercedita Airport, Ponce", 18.0083, -66.5630),
    "SJU": ("Luis Munoz Marin International, San Juan", 18.4394, -66.0018),
    "STT": ("Cyril E. King Airport, St Thomas", 18.3373, -64.9734),
}


def main() -> None:
    routes = pd.read_csv(OUT / "route_summary.csv")
    metrics = json.loads((OUT / "metrics.json").read_text())

    airports = nycflights13.airports.set_index("faa")[["name", "lat", "lon"]]
    extra = pd.DataFrame.from_dict(
        SUPPLEMENTARY_AIRPORTS, orient="index", columns=["name", "lat", "lon"]
    )
    airports = pd.concat([airports, extra[~extra.index.isin(airports.index)]])

    needed = set(routes["origin"]) | set(routes["dest"])
    missing = needed - set(airports.index)
    if missing:
        raise SystemExit(f"No coordinates for: {sorted(missing)}. Add them to SUPPLEMENTARY_AIRPORTS.")

    def r(x, nd=4):
        return None if pd.isna(x) else round(float(x), nd)

    payload = {
        "generated": f"{date.today():%d %B %Y}",
        "min_flights": MIN_FLIGHTS,
        "test_period": "November-December 2013",
        "headline": {
            "auc": r(metrics["lightgbm"]["roc_auc"], 3),
            "baseline_auc": r(metrics["baseline"]["roc_auc"], 3),
            "delay_rate": r(metrics["test_delay_rate"], 3),
            "n_flights": int(routes["n_flights"].sum()),
        },
        "airports": {
            code: {"name": row["name"], "lat": r(row["lat"]), "lon": r(row["lon"])}
            for code, row in airports.loc[sorted(needed)].iterrows()
        },
        "routes": [
            {
                "o": row.origin, "d": row.dest, "n": int(row.n_flights),
                "actual": r(row.actual_delay_rate), "pred": r(row.mean_predicted),
                "auc": r(row.roc_auc, 3),
            }
            for row in routes.itertuples()
        ],
    }

    DOCS.mkdir(exist_ok=True)
    (DOCS / "data.js").write_text("window.MAP_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n")
    print(f"Wrote docs/data.js: {len(payload['routes'])} routes, {len(payload['airports'])} airports")


if __name__ == "__main__":
    main()
