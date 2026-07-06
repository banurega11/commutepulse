"""
data_gen.py
Generates a synthetic "city commute sensor" dataset simulating what would
normally be ingested from a BigQuery table fed by GPS/traffic-sensor streams.

Columns:
  timestamp      - datetime of the reading
  route_id       - short code, e.g. R1
  route_name     - human name, e.g. "Bus 12 - Downtown Loop"
  hour           - hour of day (0-23)
  day_of_week    - Mon..Sun
  weather        - clear / rain / fog
  incident       - 1 if an incident (accident/closure) was logged, else 0
  passenger_load - 0..1 estimated occupancy
  delay_minutes  - target variable: how late the vehicle ran

Run: python data_gen.py --rows 200000 --out commute_data.csv
"""
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

ROUTES = [
    ("R1", "Bus 12 - Downtown Loop"),
    ("R2", "Bus 27 - Riverside Express"),
    ("R3", "Metro Line A - North/South"),
    ("R4", "Metro Line B - Airport Link"),
    ("R5", "Bus 45 - University Shuttle"),
    ("R6", "Bus 8 - Industrial Park"),
]

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
WEATHER = ["clear", "clear", "clear", "rain", "fog"]  # weighted toward clear


def generate(rows: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    start = datetime(2026, 1, 1)
    minutes_span = 180 * 24 * 60  # ~6 months of minutes to sample from

    route_idx = rng.integers(0, len(ROUTES), size=rows)
    offset_minutes = rng.integers(0, minutes_span, size=rows)
    timestamps = [start + timedelta(minutes=int(m)) for m in offset_minutes]

    hours = np.array([t.hour for t in timestamps])
    dows = np.array([t.weekday() for t in timestamps])
    weather = rng.choice(WEATHER, size=rows)
    incident = rng.choice([0, 1], size=rows, p=[0.93, 0.07])
    passenger_load = np.clip(rng.normal(0.5, 0.2, size=rows), 0, 1)

    # base delay depends on rush hour, weather, incidents, and route congestion
    is_rush = np.isin(hours, [7, 8, 9, 17, 18, 19]).astype(float)
    is_weekend = (dows >= 5).astype(float)

    route_bias = np.array([[0, 1, 2, 1.5, 0.5, 3][r] for r in route_idx])  # some routes chronically slower
    weather_bias = np.where(weather == "rain", 4.0, np.where(weather == "fog", 2.5, 0.0))
    incident_bias = incident * rng.uniform(8, 25, size=rows)
    rush_bias = is_rush * (1 - is_weekend) * rng.uniform(3, 9, size=rows)
    noise = rng.normal(0, 2.0, size=rows)

    delay = (route_bias + weather_bias + incident_bias + rush_bias + noise)
    delay = np.clip(delay, -3, None)  # allow "early" a little, floor it

    df = pd.DataFrame({
        "timestamp": timestamps,
        "route_id": [ROUTES[r][0] for r in route_idx],
        "route_name": [ROUTES[r][1] for r in route_idx],
        "hour": hours,
        "day_of_week": [DAY_NAMES[d] for d in dows],
        "weather": weather,
        "incident": incident,
        "passenger_load": np.round(passenger_load, 2),
        "delay_minutes": np.round(delay, 1),
    })
    return df.sort_values("timestamp").reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=200_000)
    ap.add_argument("--out", type=str, default="commute_data.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = generate(args.rows, args.seed)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df):,} rows to {args.out}")
