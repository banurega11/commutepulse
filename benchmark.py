"""
benchmark.py
Run this on a GPU-enabled notebook (Google Colab free GPU runtime, or a
Vertex AI Workbench instance with an NVIDIA GPU) to produce the
"acceleration evidence" numbers for your slides/PPT.

On Colab:
  1. Runtime -> Change runtime type -> GPU (T4 is fine)
  2. !pip install cudf-cu12 --extra-index-url=https://pypi.nvidia.com
  3. Upload commute_data.csv (or run data_gen.py --rows 2000000 for a bigger,
     more dramatic comparison)
  4. Run: python benchmark.py

It runs the exact same route_hour_stats() and detect_anomalies() functions
from analysis.py twice: once forcing plain pandas, once with cudf.pandas
active, and prints a speedup table.
"""
import subprocess
import sys
import time

DATA = "commute_data.csv"


def run_pandas_only():
    code = f"""
import pandas as pd, time
import analysis as an
an.GPU_ACCELERATED = False
df = pd.read_csv("{DATA}", parse_dates=["timestamp"])
t0=time.perf_counter(); an.route_hour_stats(df); t1=time.perf_counter()
t2=time.perf_counter(); an.detect_anomalies(df); t3=time.perf_counter()
print(f"PANDAS route_hour_stats={{(t1-t0)*1000:.1f}}ms detect_anomalies={{(t3-t2)*1000:.1f}}ms")
"""
    subprocess.run([sys.executable, "-c", code])


def run_cudf_pandas():
    code = f"""
import cudf.pandas
cudf.pandas.install()
import pandas as pd, time
import analysis as an
df = pd.read_csv("{DATA}", parse_dates=["timestamp"])
t0=time.perf_counter(); an.route_hour_stats(df); t1=time.perf_counter()
t2=time.perf_counter(); an.detect_anomalies(df); t3=time.perf_counter()
print(f"CUDF.PANDAS route_hour_stats={{(t1-t0)*1000:.1f}}ms detect_anomalies={{(t3-t2)*1000:.1f}}ms")
"""
    subprocess.run([sys.executable, "-c", code])


if __name__ == "__main__":
    print("=== CPU (plain pandas) ===")
    run_pandas_only()
    print("\n=== GPU (cudf.pandas) — requires NVIDIA GPU runtime ===")
    try:
        run_cudf_pandas()
    except Exception as e:
        print(f"cudf.pandas not available here ({e}). Run this file on a GPU notebook.")
