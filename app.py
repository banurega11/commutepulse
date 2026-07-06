"""
app.py - CommutePulse backend

Serves:
  GET  /                 -> dashboard (static/index.html)
  GET  /api/routes       -> list of routes
  GET  /api/heatmap      -> avg delay per route/hour (for the heatmap)
  GET  /api/anomalies    -> top anomalous route/hour/day incidents
  GET  /api/risk         -> risk_score() decision-support output
       params: route_id, hour, weather (optional, default 'clear')
  GET  /api/meta         -> dataset size + whether GPU acceleration is active

In production this would sit behind Cloud Run, with commute_data.csv
replaced by a BigQuery query (see README "Swapping in BigQuery").
"""
import os
from flask import Flask, jsonify, request, send_from_directory

import analysis as an

app = Flask(__name__, static_folder="static", static_url_path="")

DATA_PATH = os.environ.get("DATA_PATH", "commute_data.csv")
DF = an.load_data(DATA_PATH)
HEATMAP = an.route_hour_stats(DF)
ANOMALIES = an.detect_anomalies(DF).head(50)


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/meta")
def meta():
    return jsonify({
        "rows": int(len(DF)),
        "routes": int(DF["route_id"].nunique()),
        "gpu_accelerated": an.GPU_ACCELERATED,
    })


@app.route("/api/routes")
def routes():
    r = DF[["route_id", "route_name"]].drop_duplicates().sort_values("route_id")
    return jsonify(r.to_dict(orient="records"))


@app.route("/api/heatmap")
def heatmap():
    return jsonify(HEATMAP.round(2).to_dict(orient="records"))


@app.route("/api/anomalies")
def anomalies():
    cols = ["timestamp", "route_id", "route_name", "hour", "day_of_week",
            "weather", "delay_minutes", "z"]
    out = ANOMALIES[cols].copy()
    out["timestamp"] = out["timestamp"].astype(str)
    out["z"] = out["z"].round(2)
    return jsonify(out.to_dict(orient="records"))


@app.route("/api/risk")
def risk():
    route_id = request.args.get("route_id", "R1")
    hour = int(request.args.get("hour", 8))
    weather = request.args.get("weather", "clear")
    result = an.risk_score(DF, route_id, hour, weather)
    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
