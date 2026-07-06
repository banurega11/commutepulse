"""
analysis.py
Core decision-support logic for CommutePulse.

Acceleration note:
This module tries to use NVIDIA's cudf.pandas (zero-code-change GPU
acceleration for pandas) when a GPU runtime is available, e.g. on a
Vertex AI Workbench / Colab GPU instance. On a CPU-only machine (like this
dev sandbox) it silently falls back to plain pandas so the app still runs -
but the *same code* runs faster with no changes on a GPU box. See
benchmark.py for a before/after timing you can drop straight into your
slides as the "acceleration evidence".
"""
import time
import numpy as np

GPU_ACCELERATED = False
try:
    import cudf.pandas  # noqa: F401
    cudf.pandas.install()
    GPU_ACCELERATED = True
except Exception:
    pass

import pandas as pd  # noqa: E402  (import after cudf.pandas.install() so it patches pandas)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def route_hour_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Average / p90 delay per route per hour - powers the dashboard heatmap."""
    g = (
        df.groupby(["route_id", "route_name", "hour"])["delay_minutes"]
        .agg(avg_delay="mean", p90_delay=lambda s: s.quantile(0.9), n="count")
        .reset_index()
    )
    return g


def detect_anomalies(df: pd.DataFrame, z_thresh: float = 3.0) -> pd.DataFrame:
    """Flag route/hour/day combos whose delay is a statistical outlier (z-score)."""
    stats = df.groupby(["route_id", "hour"])["delay_minutes"].agg(["mean", "std"]).reset_index()
    stats.columns = ["route_id", "hour", "mu", "sigma"]
    merged = df.merge(stats, on=["route_id", "hour"], how="left")
    merged["sigma"] = merged["sigma"].replace(0, np.nan)
    merged["z"] = (merged["delay_minutes"] - merged["mu"]) / merged["sigma"]
    anomalies = merged[merged["z"].abs() >= z_thresh].copy()
    return anomalies.sort_values("z", ascending=False)


def risk_score(df: pd.DataFrame, route_id: str, hour: int, weather: str = "clear") -> dict:
    """
    Decision-support output: given a planned route + hour (+ weather),
    return a 0-100 risk score and a plain-language recommendation, plus
    the best alternative route for that hour if the risk is high.
    """
    subset = df[(df["route_id"] == route_id) & (df["hour"] == hour)]
    if len(subset) == 0:
        return {"error": "no data for that route/hour"}

    weather_subset = subset[subset["weather"] == weather]
    sample = weather_subset if len(weather_subset) >= 20 else subset

    avg_delay = float(sample["delay_minutes"].mean())
    p90_delay = float(sample["delay_minutes"].quantile(0.9))
    incident_rate = float(sample["incident"].mean())

    # simple, explainable risk formula (weights are documented, not a black box)
    risk = (
        min(avg_delay / 20, 1.0) * 45
        + min(p90_delay / 30, 1.0) * 35
        + incident_rate * 20
    )
    risk = round(min(risk, 100), 1)

    if risk < 30:
        level, advice = "Low", "On time is likely. Go ahead with this route."
    elif risk < 60:
        level, advice = "Moderate", "Some delay is likely. Build in a buffer of 5-10 minutes."
    else:
        level, advice = "High", "Significant delay likely. Consider an alternate route or leaving earlier."

    # find a lower-risk alternative for the same hour
    all_routes = df["route_id"].unique()
    alt_best = None
    if risk >= 60:
        candidates = []
        for r in all_routes:
            if r == route_id:
                continue
            alt = df[(df["route_id"] == r) & (df["hour"] == hour)]
            if len(alt) >= 20:
                candidates.append((r, alt["delay_minutes"].mean()))
        if candidates:
            candidates.sort(key=lambda x: x[1])
            alt_best = candidates[0][0]

    return {
        "route_id": route_id,
        "hour": hour,
        "weather": weather,
        "risk_score": risk,
        "risk_level": level,
        "avg_delay_minutes": round(avg_delay, 1),
        "p90_delay_minutes": round(p90_delay, 1),
        "incident_rate": round(incident_rate, 3),
        "recommendation": advice,
        "alternate_route_id": alt_best,
        "gpu_accelerated": GPU_ACCELERATED,
    }


def time_operation(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


if __name__ == "__main__":
    df = load_data("commute_data.csv")
    print(f"GPU accelerated (cudf.pandas active): {GPU_ACCELERATED}")
    stats, t = time_operation(route_hour_stats, df)
    print(f"route_hour_stats: {t*1000:.1f} ms, {len(stats)} rows")
    anomalies, t2 = time_operation(detect_anomalies, df)
    print(f"detect_anomalies: {t2*1000:.1f} ms, {len(anomalies)} anomalies found")
    result = risk_score(df, "R6", 8, "rain")
    print("sample risk_score:", result)
