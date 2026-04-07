#!/usr/bin/env python3
"""
Empirical serial classify() spot-check for Figures 7 & 8 serial baselines.

Station counts: 10, 20, 30, then 40, 50, …, 220, 228 (CPU only; matches CPU
Table~1 / figure serial curves).

Methodology: pre-load up to 228 streams from data/230_stations_1_min_dt,
one warm model load per architecture, then for each N take streams[:N] and
record min wall time over N_RUNS full sequential classify passes.

Outputs: docs/tables/serial_classify_spotcheck_cpu.json

Usage (from eqcctpro repo root):
  python experiments/workbench/scaling/benchmark_serial_classify_spotcheck.py
"""
from __future__ import annotations

import gc
import glob
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Selected high corner frequency")
warnings.filterwarnings("ignore", message="download precheck failed")
os.environ["SEISBENCH_LOG_LEVEL"] = "ERROR"

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO))

from eqcctpro.tools import build_station_list_from_dir  # noqa: E402
from eqcctpro.seisbench_models import SeisBenchModels, mseed2stream_3c  # noqa: E402

try:
    import torch
except ImportError:
    torch = None

TIMECHUNK = "20241215T120000Z_20241215T120100Z"
DATASET = "230_stations_1_min_dt"
N_RUNS = 5
MAX_STATIONS = 228

# (parent, child, json_label) — same four SeisBench models as Fig. 7/8 serial table
MODEL_SPECS = [
    ("PhaseNet", "original", "PhaseNet"),
    ("PhaseNetLight", "stead", "PhaseNetLight"),
    ("EQTransformer", "original", "EQTransformer"),
    ("EQTransformer", "original_nonconservative", "EQT-NC"),
]

STATION_COUNTS = [10, 20, 30] + list(range(40, 221, 10)) + [228]


def _classify_kw(parent: str) -> dict:
    if parent == "PhaseNet":
        return dict(
            P_threshold=0.3,
            S_threshold=0.3,
            Detection_threshold=0.3,
            strict=False,
            flexible_horizontal_components=True,
        )
    return {}


def cuda_sync():
    if torch and torch.cuda.is_available():
        torch.cuda.synchronize()


def preload_streams(input_dir: Path, n_stations: int) -> list:
    stations = build_station_list_from_dir(str(input_dir))[:n_stations]
    streams = []
    for sta in stations:
        files = glob.glob(str(input_dir / sta / "*mseed"))
        if not files:
            continue
        try:
            stream, _, _ = mseed2stream_3c({}, files, sta)
            streams.append(stream)
        except Exception as e:
            print(f"  skip {sta}: {e}", flush=True)
    return streams


def bench_model_over_station_counts(
    all_streams: list, station_counts: list[int], parent: str, child: str
) -> dict[str, float]:
    """One model load per architecture; for each N, min over N_RUNS full classify passes (CPU)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    kw = _classify_kw(parent)
    sb = SeisBenchModels(parent, child)
    model = sb.load_model()
    cuda_sync()
    series: dict[str, float] = {}
    for n in station_counts:
        if n > len(all_streams):
            continue
        subset = all_streams[:n]
        print(f"  N={n} classify…", flush=True)
        times = []
        for _ in range(N_RUNS):
            cuda_sync()
            t0 = time.perf_counter()
            for stream in subset:
                model.classify(stream, **kw)
            cuda_sync()
            times.append(time.perf_counter() - t0)
        tmin = min(times)
        series[str(n)] = round(tmin, 3)
        print(f"    min {tmin:.3f} s", flush=True)
    del model, sb
    gc.collect()
    cuda_sync()
    return series


def main():
    input_dir = REPO / "data" / DATASET / TIMECHUNK
    out_path = REPO / "docs" / "tables" / "serial_classify_spotcheck_cpu.json"
    if not input_dir.is_dir():
        print(f"ERROR: missing data dir {input_dir}", flush=True)
        sys.exit(1)

    print(f"Loading up to {MAX_STATIONS} streams from {input_dir}…", flush=True)
    all_streams = preload_streams(input_dir, MAX_STATIONS)
    print(f"  got {len(all_streams)} streams", flush=True)
    if len(all_streams) < MAX_STATIONS:
        print(f"WARNING: fewer than {MAX_STATIONS} streams; high-N rows may be partial", flush=True)

    payload = {
        "device": "CPU",
        "timechunk": TIMECHUNK,
        "dataset": DATASET,
        "n_runs": N_RUNS,
        "station_counts_requested": STATION_COUNTS,
        "models": {},
    }

    for parent, child, label in MODEL_SPECS:
        print(f"\n=== {label} ({parent}/{child}) ===", flush=True)
        counts = [n for n in STATION_COUNTS if n <= len(all_streams)]
        for n in STATION_COUNTS:
            if n > len(all_streams):
                print(f"  N={n}: skip (only {len(all_streams)} streams)", flush=True)
        series = bench_model_over_station_counts(all_streams, counts, parent, child)
        payload["models"][label] = series
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"  (checkpoint) wrote {out_path}", flush=True)

    print(f"\nDone. Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
