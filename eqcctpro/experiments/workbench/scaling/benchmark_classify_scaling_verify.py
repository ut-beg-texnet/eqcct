#!/usr/bin/env python3
"""
Verify classify() scaling: run 228, 250, 580 stations on same machine.
Compares observed runtimes to linear extrapolation from 228.
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
N_RUNS = 5

# (dataset_dir, n_stations)
DATASETS = [
    ("230_stations_1_min_dt", 228),
    ("250_stations_1_min_dt", 250),
    ("580_stations_1_min_dt", 580),
]


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


def run_classify_benchmark(streams, use_gpu=False):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" if use_gpu else "-1"
    sb = SeisBenchModels("PhaseNet", "original")
    model = sb.load_model()
    times = []
    for run in range(N_RUNS):
        start = time.perf_counter()
        for stream in streams:
            model.classify(stream)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    return min(times)


def main():
    print("PhaseNet CPU - classify() scaling verification")
    print("=" * 60)
    results = []
    baseline_228 = None

    for dataset_name, n_stations in DATASETS:
        input_dir = os.path.join(BASE, "data", dataset_name, TIMECHUNK)
        if not os.path.isdir(input_dir):
            print(f"SKIP {dataset_name}: not found\n", flush=True)
            continue

        print(f"\n{dataset_name} ({n_stations} stations)", flush=True)
        streams = preload_streams(input_dir, n_stations)
        print(f"  Loaded {len(streams)} streams", flush=True)

        t_min = run_classify_benchmark(streams, use_gpu=False)
        results.append((n_stations, len(streams), t_min))

        if n_stations == 228:
            baseline_228 = t_min
            print(f"  Min: {t_min:.2f} s (baseline)", flush=True)
        else:
            linear_pred = baseline_228 * (n_stations / 228) if baseline_228 else None
            ratio = t_min / linear_pred if linear_pred else None
            print(f"  Min: {t_min:.2f} s", flush=True)
            if linear_pred:
                print(f"  Linear extrap from 228: {linear_pred:.2f} s (ratio: {ratio:.2f}x)", flush=True)

    print("\n" + "=" * 60)
    print("Summary:")
    for n, loaded, t in results:
        per_sta = t / loaded if loaded else 0
        print(f"  {n} stations ({loaded} loaded): {t:.2f} s ({per_sta:.3f} s/station)")
    print("=" * 60)


if __name__ == "__main__":
    main()
