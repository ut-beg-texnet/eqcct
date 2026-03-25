#!/usr/bin/env python3
"""Run all SeisBench models CPU sequential classify for 250 and 580 stations to verify paper claims."""
import os
import sys
import time
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eqcctpro.tools import build_station_list_from_dir
from eqcctpro.seisbench_models import SeisBenchModels, mseed2stream_3c

BASE = "/home/skevofilaxc/workspace/clean_eqcct/eqcct/eqcctpro"

# All SeisBench models from the paper
SEISBENCH_MODELS = [
    ("PhaseNet", "original"),
    ("PhaseNetLight", "stead"),
    ("EQTransformer", "original"),
    ("EQTransformer", "original_nonconservative"),  # EQT-NC
]


def run_model_cpu_sequential(input_mseed_dir, n_stations, parent, child):
    """Run a SeisBench model CPU classify() sequentially for n_stations."""
    model_name = f"{parent}/{child}" if "original" in child else parent
    stations = build_station_list_from_dir(input_mseed_dir)[:n_stations]
    sb = SeisBenchModels(parent, child)
    model = sb.load_model()
    start = time.time()
    for i, sta in enumerate(stations):
        files = glob.glob(os.path.join(input_mseed_dir, sta, "*mseed"))
        if not files:
            continue
        stream, _, _ = mseed2stream_3c({}, files, sta)
        model.classify(stream)
        if (i + 1) % 50 == 0:
            print(f"    {model_name}: {i+1}/{n_stations}...")
    elapsed = time.time() - start
    return elapsed


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU only
    for name, n in [("250_stations_1_min_dt", 250), ("580_stations_1_min_dt", 580)]:
        path = os.path.join(BASE, "data", name, "20241215T120000Z_20241215T120100Z")
        if not os.path.isdir(path):
            print(f"  {name}: dataset not found at {path}")
            continue
        print(f"\n=== {name} ({n} stations) ===")
        for parent, child in SEISBENCH_MODELS:
            model_label = f"{parent}/{child}" if "original" in child else parent
            elapsed = run_model_cpu_sequential(path, n, parent, child)
            print(f"  {model_label}: {elapsed:.2f} s")
