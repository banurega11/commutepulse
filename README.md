# CommutePulse — AI/Data Decision-Support for Daily Commuters

**Track:** Practical data analytics / decision-support application

**Problem it solves:** Daily commuters and city transit planners can't easily
tell *before they leave* whether a bus/metro route is likely to run late, or
which route is statistically the safer bet right now. CommutePulse ingests
transit sensor data (timestamps, route, weather, incidents, occupancy),
detects delay patterns and anomalies, and returns a plain-language
recommendation with a 0–100 risk score.

## Who is the user?
- A commuter deciding which bus/metro line to take this morning.
- A transit operations team spotting which route/hour combos are chronically
  unreliable (the anomaly table) so they can investigate root causes.

## The pipeline
1. **Ingest** — `data_gen.py` simulates 200k+ rows of sensor readings
   (swap for a real feed / BigQuery table in production — see below).
2. **Clean & analyze** — `analysis.py` aggregates delay by route/hour,
   computes a p90 delay and incident rate, and flags statistical anomalies
   (z-score ≥ 3) per route/hour/day.
3. **Decision output** — `risk_score()` returns a risk level, a plain-English
   recommendation, and — if risk is high — a lower-risk alternate route.
4. **Serve** — `app.py` (Flask) exposes this as an API and dashboard.
5. **Acceleration** — the same analysis code runs unmodified through
   `cudf.pandas` (NVIDIA's zero-code-change GPU acceleration for pandas).
   On CPU it silently falls back to plain pandas. `benchmark.py` produces
   the before/after timing to quote in your slides.

## Tech used (2+ required items)
- **Data layer:** CSV standing in for a **BigQuery** table (swap instructions below) / **Cloud Storage** for the raw file.
- **Acceleration layer:** **cudf.pandas** — see `analysis.py` / `benchmark.py`.
- **Serving:** Flask app, containerized, deployed on **Cloud Run**.

## Run it locally
```bash
pip install -r requirements.txt
python data_gen.py --rows 200000      # creates commute_data.csv
python app.py                          # serves on http://localhost:8080
```

## Deploy to Cloud Run (step by step, beginner-friendly)
1. Install the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) and run `gcloud init` to log in and pick/create a project.
2. Enable the needed APIs once:
   ```bash
   gcloud services enable run.googleapis.com artifactregistry.googleapis.com
   ```
3. From inside this project folder, deploy directly from source (Cloud Build + Cloud Run in one command):
   ```bash
   gcloud run deploy commutepulse \
     --source . \
     --region asia-southeast1 \
     --allow-unauthenticated
   ```
4. When it finishes, gcloud prints a **Service URL** — that's your public
   deployment link for the submission form.
5. Every time you change code, re-run the same `gcloud run deploy` command.

## Swapping in real BigQuery (optional, for extra credit)
Replace `load_data()` in `analysis.py`:
```python
from google.cloud import bigquery
def load_data(_path=None):
    client = bigquery.Client()
    return client.query("SELECT * FROM your_dataset.commute_data").to_dataframe()
```
Upload `commute_data.csv` to a BigQuery table first (BigQuery console →
Create table → Upload → CSV). Grant your Cloud Run service's identity
`BigQuery Data Viewer` + `BigQuery Job User` roles.

## Producing the acceleration evidence for your slides
Run `benchmark.py` on a free Google Colab GPU runtime (see comments inside
the file for the 3-step setup). It prints CPU-vs-GPU timings for the exact
same functions the app uses — paste that table straight into your PPT.

## What's in this repo
```
app.py            - Flask backend + API
analysis.py       - aggregation, anomaly detection, risk scoring (GPU-ready)
data_gen.py       - synthetic dataset generator
benchmark.py       - CPU vs GPU timing script (run on a GPU notebook)
static/index.html - dashboard frontend
Dockerfile        - container spec for Cloud Run
requirements.txt  - Python deps
```
