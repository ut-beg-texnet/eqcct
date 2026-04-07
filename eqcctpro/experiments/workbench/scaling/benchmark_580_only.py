#!/usr/bin/env python3
"""Run PhaseNet CPU classify() for 580 stations only."""
import os
import sys
import time
import glob
import warnings

warnings.filterwarnings("ignore", message="Selected high corner frequency")
warnings.filterwarnings("ignore", message="download precheck failed")
os.environ["SEISBENCH_LOG_LEVEL"] = "ERROR"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eqcctpro.tools import build_station_list_from_dir
from eqcctpro.seisbench_models import SeisBenchModels, mseed2stream_3c

BASE = "/home/skevofilaxc/workspace/clean_eqcct/eqcct/eqcctpro"
TIMECHUNK = "20241215T120000Z_20241215T120100Z"
N_RUNS = 2  # Quick run for 580

input_dir = os.path.join(BASE, "data", "580_stations_1_min_dt", TIMECHUNK)
stations = build_station_list_from_dir(input_dir)[:580]
streams = []
for sta in stations:
    files = glob.glob(os.path.join(input_dir, sta, "*mseed"))
    if files:
        try:
            stream, _, _ = mseed2stream_3c({}, files, sta)
            streams.append(stream)
        except Exception:
            pass

print(f"Loaded {len(streams)} streams", flush=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sb = SeisBenchModels("PhaseNet", "original")
model = sb.load_model()
times = []
for run in range(N_RUNS):
    start = time.perf_counter()
    for stream in streams:
        model.classify(stream)
    elapsed = time.perf_counter() - start
    times.append(elapsed)
    print(f"  Run {run+1}: {elapsed:.2f} s", flush=True)
print(f"Min: {min(times):.2f} s", flush=True)
print(f"Linear extrap from 228 (33.70s): {33.70 * 580/228:.2f} s", flush=True)
