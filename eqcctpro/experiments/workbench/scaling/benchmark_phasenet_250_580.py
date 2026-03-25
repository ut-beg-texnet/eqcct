#!/usr/bin/env python3
"""
Quick PhaseNet CPU benchmark for 250 and 580 stations to verify paper scaling claims.

Paper: PhaseNet CPU ~30.8 s (250 stations, TexNet), ~71 s (580 stations, NCSN)
Methodology: pre-copied streams, 5 runs, min time
"""
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
DATASETS = [
    ("250_stations_1_min_dt", 250),
    ("580_stations_1_min_dt", 580),
]
N_RUNS = 5


def preload_streams(input_dir, n_stations):
    stations = build_station_list_from_dir(input_dir)[:n_stations]
    streams = []
    for sta in stations:
        files = glob.glob(os.path.join(input_dir, sta, "*mseed"))
        if not files:
            continue
        try:
            stream, _, _ = mseed2stream_3c({}, files, sta)
            streams.append(stream)
        except Exception as e:
            print(f"  Skip {sta}: {e}", flush=True)
    return streams


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    sb = SeisBenchModels("PhaseNet", "original")
    model = sb.load_model()

    for dataset_name, n_stations in DATASETS:
        input_dir = os.path.join(BASE, "data", dataset_name, TIMECHUNK)
        if not os.path.isdir(input_dir):
            print(f"SKIP {dataset_name}: not found", flush=True)
            continue

        print(f"\n{dataset_name} ({n_stations} stations) - PhaseNet CPU", flush=True)
        streams = preload_streams(input_dir, n_stations)
        print(f"Loaded {len(streams)} streams", flush=True)

        times = []
        for run in range(N_RUNS):
            start = time.perf_counter()
            for stream in streams:
                model.classify(stream)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            print(f"  Run {run+1}: {elapsed:.2f} s", flush=True)

        t_min = min(times)
        print(f"  Min: {t_min:.2f} s (paper: ~30.8 s for 250, ~71 s for 580)", flush=True)


if __name__ == "__main__":
    main()
