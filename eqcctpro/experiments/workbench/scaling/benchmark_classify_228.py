#!/usr/bin/env python3
"""
Reproduce Table 4 Classify-Per-Station baseline and verify 250/580 station scaling.

Methodology (from paper):
- Sequential classify() calls on individual station streams
- Pre-copied streams (load into memory first, exclude disk I/O from timing)
- Five repeated runs; minimum time reported
- Both CPU and GPU for each SeisBench model

Paper claims: ~30.8 s for 250 stations (TexNet), ~71 s for 580 stations (NCSN) - PhaseNet CPU
"""
import os
import sys
import time
import glob
import warnings

# Suppress verbose warnings during benchmark
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

MODELS = [
    ("PhaseNet", "original"),
    ("PhaseNetLight", "stead"),
    ("EQTransformer", "original"),
    ("EQTransformer", "original_nonconservative"),
]


def preload_streams(input_dir, n_stations):
    """Pre-copy all streams into memory (matches paper 'pre-copied streams')."""
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
            print(f"  Skip {sta}: {e}")
    return streams


def run_classify_benchmark(streams, parent, child, use_gpu):
    """Run N_RUNS iterations of classify on all streams, return minimum time."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" if use_gpu else "-1"
    sb = SeisBenchModels(parent, child)
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
    import csv
    out_dir = os.path.join(BASE, "results", "csv")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "classify_250_580_scaling.csv")
    rows = []

    for dataset_name, n_stations in DATASETS:
        input_dir = os.path.join(BASE, "data", dataset_name, TIMECHUNK)
        if not os.path.isdir(input_dir):
            print(f"SKIP {dataset_name}: not found at {input_dir}\n", flush=True)
            continue

        print(f"\n{'='*60}", flush=True)
        print(f"  {dataset_name} ({n_stations} stations)", flush=True)
        print("=" * 60, flush=True)
        print(f"Pre-loading {n_stations} streams into memory (pre-copied streams)...", flush=True)
        streams = preload_streams(input_dir, n_stations)
        print(f"Loaded {len(streams)} streams\n", flush=True)

        if len(streams) < n_stations:
            print(f"WARNING: Only {len(streams)} stations available, expected {n_stations}\n", flush=True)

        results = []
        for parent, child in MODELS:
            name = f"{parent}/{child}" if "original" in child else parent
            for use_gpu in [False, True]:
                device = "GPU" if use_gpu else "CPU"
                t = run_classify_benchmark(streams, parent, child, use_gpu)
                results.append((name, device, t))
                rows.append({"dataset": dataset_name, "n_stations": n_stations, "model": name, "device": device, "time_s": f"{t:.2f}"})
                print(f"  {name} {device}: {t:.2f} s (min of {N_RUNS} runs)", flush=True)

        print(f"\n  Range: {min(r[2] for r in results):.2f} – {max(r[2] for r in results):.2f} s", flush=True)

    if rows:
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["dataset", "n_stations", "model", "device", "time_s"])
            w.writeheader()
            w.writerows(rows)

    print("\n" + "=" * 60, flush=True)
    print("Paper claims: PhaseNet CPU ~30.8 s (250), ~71 s (580)", flush=True)
    print(f"Results saved to {out_path}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
